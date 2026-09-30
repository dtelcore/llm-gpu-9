"""
generate_config.py

Interactive (or flag-driven) writer for setup/*.json training recipes.

Tweaks C / H / L / T / B / accum, residual scale, layer streaming, and dataset
without going through ``--menu`` (which overrides the JSON). Prints a 2 GB
train estimate and an ``auto_train.py`` command. New C/L/H always needs a new
checkpoint — do not resume chat8b / run8+16 into a different width or depth.

    python generate_config.py
    python generate_config.py --from setup/chat_c256_l6_config.json \\
        --embedding-dim 384 --num-layers 8 --no-prompt \\
        --output setup/chat_c384_l8_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import DATA_DIR, SETUP_DIR
from setup.model_config import estimate_vram_footprint
from training.memory_controller import plan_train, usable_bytes

_BASE_RECIPES = (
    ("story", ROOT / "legacy" / "setup" / "story_c256_l6_config.json"),
    ("chat", ROOT / "legacy" / "setup" / "chat_c256_l6_config.json"),
    ("fast", ROOT / "legacy" / "setup" / "story_sub1m_config.json"),
)

_EST_VOCAB = 4112


def load_recipe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Recipe not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "model" not in data or "hyperparameters" not in data or "dataset" not in data:
        raise ValueError(f"Not a training recipe JSON: {path}")
    return data


def list_datasets(data_dir: Path = DATA_DIR) -> List[str]:
    names = ["data_dir"]
    if data_dir.is_dir():
        for path in sorted(data_dir.glob("*.txt")):
            names.append(path.stem)
    return names


def default_heads(embedding_dim: int) -> int:
    for heads in (8, 4, 6, 2):
        if embedding_dim % heads == 0:
            return heads
    return 1


def validate_architecture(
    *,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    max_len: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    errors: List[str] = []
    if embedding_dim < 16 or embedding_dim > 1024:
        errors.append(f"C={embedding_dim} out of 16–1024")
    if num_heads < 1:
        errors.append("num_heads must be >= 1")
    if embedding_dim % num_heads != 0:
        errors.append(f"C={embedding_dim} must be divisible by H={num_heads}")
    if not (1 <= num_layers <= 48):
        errors.append(f"L={num_layers} out of 1–48")
    if not (32 <= max_len <= 1024):
        errors.append(f"T={max_len} out of 32–1024")
    if not (1 <= batch_size <= 64):
        errors.append(f"B={batch_size} out of 1–64")
    if not (1 <= grad_accum <= 64):
        errors.append(f"grad_accum={grad_accum} out of 1–64")
    if errors:
        raise ValueError("\n".join(errors))


def suggested_output_name(kind: str, embedding_dim: int, num_layers: int) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", kind.lower()).strip("_") or "run"
    return SETUP_DIR / f"{slug}_c{embedding_dim}_l{num_layers}_config.json"


def apply_overrides(
    recipe: Mapping[str, Any],
    *,
    embedding_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    residual_scale: Optional[bool] = None,
    layer_strategy: Optional[str] = None,
    dataset: Optional[str] = None,
    combine: Optional[bool] = None,
    bpe_merges: Optional[int] = None,
    learning_rate: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    dropout: Optional[float] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = deepcopy(dict(recipe))
    model = dict(cfg.get("model") or {})
    hyper = dict(cfg.get("hyperparameters") or {})
    data = dict(cfg.get("dataset") or {})
    meta = dict(cfg.get("metadata") or {})

    C = int(embedding_dim if embedding_dim is not None else model["embedding_dim"])
    H = int(num_heads if num_heads is not None else model.get("num_heads") or default_heads(C))
    L = int(num_layers if num_layers is not None else model["num_layers"])
    T = int(max_len if max_len is not None else model["max_len"])
    B = int(batch_size if batch_size is not None else hyper["batch_size"])
    accum = int(
        grad_accum if grad_accum is not None
        else hyper.get("gradient_accumulation_steps", 1)
    )
    validate_architecture(
        embedding_dim=C,
        num_heads=H,
        num_layers=L,
        max_len=T,
        batch_size=B,
        grad_accum=accum,
    )

    if residual_scale is None:
        residual_scale = bool(model.get("residual_scale", L >= 6))
    if L >= 6 and not residual_scale:
        raise ValueError("L>=6 requires residual_scale (GPT-2 1/sqrt(2L))")

    strategy = str(layer_strategy or model.get("layer_strategy") or "resident").strip().lower()
    if strategy not in ("resident", "stream"):
        raise ValueError("layer_strategy must be resident or stream")

    if dataset is not None:
        data["name"] = str(dataset)
    if combine is not None:
        data["combine"] = bool(combine)
    elif data.get("name") == "chat_train":
        data["combine"] = False
    if bpe_merges is not None:
        data["bpe_merges"] = int(bpe_merges)
        data["tokenizer"] = "bpe"

    kind = "chat" if str(data.get("name", "")).startswith("chat") else "story"
    display = name or f"{kind.title()} C={C} L={L} T={T}"

    model.update({
        "name": display,
        "vocab_size": None,
        "max_len": T,
        "embedding_dim": C,
        "num_heads": H,
        "num_layers": L,
        "dropout_prob": float(dropout if dropout is not None else model.get("dropout_prob", 0.0)),
        "tie_embeddings": True,
        "norm_type": model.get("norm_type", "rmsnorm"),
        "pos_encoding": model.get("pos_encoding", "rope"),
        "init_scale": float(model.get("init_scale", 0.02)),
        "residual_scale": bool(residual_scale),
        "layer_strategy": strategy,
        "description": (
            f"{display}. H={H}, residual_scale={'on' if residual_scale else 'off'}, "
            f"layers={strategy}. 2 GB process cap is hardcoded."
        ),
    })
    hyper.update({
        "name": display,
        "batch_size": B,
        "gradient_accumulation_steps": accum,
        "learning_rate": float(
            learning_rate if learning_rate is not None
            else hyper.get("learning_rate", 0.00025)
        ),
        "warmup_steps": int(
            warmup_steps if warmup_steps is not None
            else hyper.get("warmup_steps", 400)
        ),
        "weight_decay": float(hyper.get("weight_decay", 0.01)),
        "num_epochs": int(hyper.get("num_epochs", 1)),
        "gradient_clip": float(hyper.get("gradient_clip", 1.0)),
        "window_stride": int(hyper.get("window_stride", 64)),
        "min_lr_ratio": float(hyper.get("min_lr_ratio", 0.1)),
        "optimizer": hyper.get("optimizer", "adamw"),
        "beta1": float(hyper.get("beta1", 0.9)),
        "beta2": float(hyper.get("beta2", 0.999)),
        "epsilon": float(hyper.get("epsilon", 1e-8)),
    })
    data.setdefault("tokenizer", "bpe")
    data.setdefault("bpe_merges", 4000)
    data.setdefault("vocab_size", None)
    data.setdefault("combine", False)

    slug = f"{kind}-c{C}-l{L}-t{T}"
    meta.update({
        "created": slug,
        "description": (
            f"Generated recipe. New C/L/H needs a new checkpoint "
            f"(do not --resume chat8b or run8+16). Dataset={data.get('name')} "
            f"combine={bool(data.get('combine'))}."
        ),
    })
    cfg["model"] = model
    cfg["hyperparameters"] = hyper
    cfg["dataset"] = data
    cfg["metadata"] = meta
    cfg.setdefault("weight_initialization", {})
    return cfg


def memory_summary(cfg: Mapping[str, Any]) -> Tuple[int, Any]:
    model = cfg["model"]
    hyper = cfg["hyperparameters"]
    vocab = int(cfg.get("dataset", {}).get("vocab_size") or _EST_VOCAB)
    footprint = estimate_vram_footprint({
        **model,
        "vocab_size": vocab,
    })
    n_params = int(footprint["total_params"])
    plan = plan_train(
        n_params=n_params,
        batch_size=int(hyper["batch_size"]),
        max_len=int(model["max_len"]),
        embedding_dim=int(model["embedding_dim"]),
        num_heads=int(model["num_heads"]),
        num_layers=int(model["num_layers"]),
        vocab_size=vocab,
        grad_accum=int(hyper.get("gradient_accumulation_steps", 1)),
        layer_strategy=str(model.get("layer_strategy", "resident")),
        autoscale=True,
    )
    return n_params, plan


def format_report(cfg: Mapping[str, Any], path: Path) -> str:
    model = cfg["model"]
    hyper = cfg["hyperparameters"]
    data = cfg["dataset"]
    n_params, plan = memory_summary(cfg)
    stem = path.stem
    if stem.endswith("_config"):
        stem = stem[: -len("_config")]
    ckpt = f"output/checkpoints/{stem}"
    lines = [
        f"Wrote {path}",
        (
            f"  C={model['embedding_dim']} H={model['num_heads']} "
            f"L={model['num_layers']} T={model['max_len']} "
            f"B={hyper['batch_size']} accum={hyper['gradient_accumulation_steps']} "
            f"residual_scale={'on' if model.get('residual_scale') else 'off'} "
            f"layers={model.get('layer_strategy', 'resident')}"
        ),
        (
            f"  dataset={data.get('name')} combine={bool(data.get('combine'))} "
            f"bpe_merges={data.get('bpe_merges')} "
            f"~{n_params:,} params (vocab~{_EST_VOCAB})"
        ),
        f"  [memory] {plan.summary_line()}",
        f"  usable={usable_bytes() / (1024 ** 2):.0f}MB  (2 GB cap is hardcoded)",
        "",
        "Train (new checkpoint, no --menu, no --resume of a different C/L):",
        (
            f"  python auto_train.py --config {path} \\\n"
            f"    --checkpoint {ckpt} \\\n"
            f"    --steps 8000 --run-budget 16000 \\\n"
            f"    --no-prompt --log-every 50"
        ),
    ]
    if not plan.fits:
        lines.append("  WARNING: estimate does not fit 2 GB even after autoscale.")
    elif plan.actions:
        lines.append("  NOTE: controller will change B/T/stream at train start; C/L/H stay.")
    if int(model["num_layers"]) >= 12 and model.get("layer_strategy") != "stream":
        lines.append("  NOTE: L>=12 usually wants --layer-stream (or layer_strategy=stream).")
    if str(data.get("name")) == "data_dir" and data.get("combine"):
        lines.append("  NOTE: combine=true concatenates every data/*.txt (not chat-only).")
    return "\n".join(lines)


def write_recipe(cfg: Mapping[str, Any], path: Path, *, force: bool = False) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (pass --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")
    return path


def _prompt(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def _prompt_int(label: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = int(raw)
        except ValueError:
            print(f"  need an integer {lo}–{hi}")
            continue
        if lo <= value <= hi:
            return value
        print(f"  need {lo}–{hi}")


def _prompt_float(label: str, default: float) -> float:
    while True:
        raw = _prompt(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print("  need a number")


def _prompt_bool(label: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{label} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "on")


def _pick_base(from_path: Optional[Path]) -> Tuple[str, Dict[str, Any], Path]:
    if from_path is not None:
        path = from_path if from_path.is_absolute() else ROOT / from_path
        return path.stem, load_recipe(path), path
    print("Base recipe:")
    for i, (kind, path) in enumerate(_BASE_RECIPES, 1):
        mark = "" if path.is_file() else " (missing)"
        print(f"  {i}. {kind:5}  {path.relative_to(ROOT)}{mark}")
    print("  4. path to an existing JSON")
    choice = _prompt("Select", "2")
    if choice == "4":
        raw = _prompt("JSON path", str(ROOT / "legacy" / "setup" / "chat_c256_l6_config.json"))
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        return path.stem, load_recipe(path), path
    idx = {"1": 0, "2": 1, "3": 2}.get(choice, 1)
    kind, path = _BASE_RECIPES[idx]
    return kind, load_recipe(path), path


def run_interactive(args: argparse.Namespace) -> Dict[str, Any]:
    print("\nApple MLX recipe generator (2 GB cap is hardcoded; C/L/H never autoscale)\n")
    from_path = Path(args.from_config) if args.from_config else None
    kind, recipe, _src = _pick_base(from_path)
    model = recipe["model"]
    hyper = recipe["hyperparameters"]
    data = recipe["dataset"]

    C = _prompt_int("C  embedding dim", int(args.embedding_dim or model["embedding_dim"]), 16, 1024)
    default_h = int(args.num_heads or model.get("num_heads") or default_heads(C))
    if C % default_h != 0:
        default_h = default_heads(C)
    H = _prompt_int("H  attention heads", default_h, 1, max(1, C))
    L = _prompt_int("L  layers", int(args.num_layers or model["num_layers"]), 1, 48)
    T = _prompt_int("T  context", int(args.max_len or model["max_len"]), 32, 1024)
    B = _prompt_int("B  micro-batch", int(args.batch_size or hyper["batch_size"]), 1, 64)
    accum = _prompt_int(
        "grad-accum",
        int(args.grad_accum or hyper.get("gradient_accumulation_steps", 1)),
        1, 64,
    )
    residual = _prompt_bool("residual_scale (on if L>=6)", L >= 6 if args.residual_scale is None else args.residual_scale)
    stream_default = L >= 12 or str(model.get("layer_strategy")) == "stream" or args.layer_strategy == "stream"
    stream = _prompt_bool("layer_strategy=stream", stream_default)

    datasets = list_datasets()
    print("datasets:", ", ".join(datasets[:12]) + (" …" if len(datasets) > 12 else ""))
    dataset = _prompt("dataset name", str(args.dataset or data.get("name") or "chat_train"))
    combine_default = False if dataset == "chat_train" else bool(data.get("combine", dataset == "data_dir"))
    combine = _prompt_bool("combine all data/*.txt", combine_default)
    merges = _prompt_int("BPE merges", int(args.bpe_merges or data.get("bpe_merges") or 4000), 50, 32000)
    lr = _prompt_float("learning rate", float(args.learning_rate or hyper.get("learning_rate") or 0.00025))
    warmup = _prompt_int("warmup steps", int(args.warmup_steps or hyper.get("warmup_steps") or 400), 0, 20000)

    kind_out = "chat" if dataset.startswith("chat") else kind.split("_")[0]
    suggested = suggested_output_name(kind_out, C, L)
    out_raw = _prompt("write JSON to", str(args.output or suggested.relative_to(ROOT)))

    cfg = apply_overrides(
        recipe,
        embedding_dim=C,
        num_heads=H,
        num_layers=L,
        max_len=T,
        batch_size=B,
        grad_accum=accum,
        residual_scale=residual,
        layer_strategy="stream" if stream else "resident",
        dataset=dataset,
        combine=combine,
        bpe_merges=merges,
        learning_rate=lr,
        warmup_steps=warmup,
        name=args.name,
    )
    args.output = out_raw
    return cfg


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a setup/*.json recipe (C/H/L/T/B) with a 2 GB train estimate",
    )
    parser.add_argument("--from", dest="from_config", default=None, help="Base recipe JSON")
    parser.add_argument("--output", type=str, default=None, help="Destination JSON path")
    parser.add_argument("--name", type=str, default=None, help="Display name in the JSON")
    parser.add_argument("--embedding-dim", type=int, default=None, help="C")
    parser.add_argument("--num-heads", type=int, default=None, help="H")
    parser.add_argument("--num-layers", type=int, default=None, help="L")
    parser.add_argument("--max-len", type=int, default=None, help="T")
    parser.add_argument("--batch-size", type=int, default=None, help="B")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument(
        "--residual-scale", choices=("on", "off"), default=None,
        help="GPT-2 1/sqrt(2L). Required on when L>=6",
    )
    parser.add_argument("--layer-strategy", choices=("resident", "stream"), default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--combine", action="store_true", default=None)
    parser.add_argument("--no-combine", dest="combine", action="store_false")
    parser.add_argument("--bpe-merges", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--no-prompt", action="store_true", help="No questions; flags + --from only")
    args = parser.parse_args(argv)
    if args.residual_scale is not None:
        args.residual_scale = args.residual_scale == "on"
    flags = list(argv if argv is not None else sys.argv[1:])
    if "--combine" not in flags and "--no-combine" not in flags:
        args.combine = None
    return args


def _cfg_from_flags(args: argparse.Namespace, recipe: Mapping[str, Any]) -> Dict[str, Any]:
    return apply_overrides(
        recipe,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.max_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        residual_scale=args.residual_scale,
        layer_strategy=args.layer_strategy,
        dataset=args.dataset,
        combine=args.combine,
        bpe_merges=args.bpe_merges,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
        name=args.name,
    )


def _resolve_base(args: argparse.Namespace) -> Path:
    base_path = Path(args.from_config) if args.from_config else _BASE_RECIPES[1][1]
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    return base_path


def _resolve_dest(args: argparse.Namespace, cfg: Mapping[str, Any], base_path: Path) -> Path:
    if args.output:
        dest = Path(args.output)
    else:
        model = cfg["model"]
        kind = "chat" if str(cfg["dataset"].get("name", "")).startswith("chat") else "story"
        dest = suggested_output_name(kind, int(model["embedding_dim"]), int(model["num_layers"]))
    if not dest.is_absolute():
        dest = ROOT / dest
    if dest.resolve() == base_path.resolve() and args.no_prompt and not args.force:
        raise ValueError(f"refusing to overwrite base recipe {_rel(dest)}; pass --output or --force")
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.no_prompt:
            base_path = _resolve_base(args)
            cfg = _cfg_from_flags(args, load_recipe(base_path))
            dest = _resolve_dest(args, cfg, base_path)
        else:
            cfg = run_interactive(args)
            dest = Path(args.output)
            if not dest.is_absolute():
                dest = ROOT / dest
        if dest.exists() and not args.force:
            if args.no_prompt:
                raise FileExistsError(f"{_rel(dest)} exists (pass --force to overwrite)")
            if not _prompt_bool(f"overwrite {_rel(dest)}", False):
                print("aborted")
                return 1
        write_recipe(cfg, dest, force=True)
        print(format_report(cfg, _rel(dest)))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
