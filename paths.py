"""
paths.py

Central path layout for the GT 730 training workspace. All runtime artifacts
(logs, checkpoints, configs, tokenizer exports, caches) live under output/
at the project root. Input corpora stay in data/ (not gitignored artifacts).

Layout (mirrors llm gpu 5 output/ convention, adapted for gpu 8's
directory-based checkpoint bundles):

    output/
      logs/          training logs (.log), plots
      checkpoints/   per-run checkpoint dirs (weights.npz, config.json, ...)
      configs/       training_config.json and wizard exports
      tokenizer/     vocab.json sidecars / tokenizer exports
      cache/         optional tokenizer / encoding caches
"""

from pathlib import Path
from typing import List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTPUT_LOGS = OUTPUT_ROOT / "logs"
OUTPUT_CHECKPOINTS = OUTPUT_ROOT / "checkpoints"
OUTPUT_CONFIGS = OUTPUT_ROOT / "configs"
OUTPUT_TOKENIZER = OUTPUT_ROOT / "tokenizer"
OUTPUT_CACHE = OUTPUT_ROOT / "cache"
OUTPUT_CACHE_TOKENIZER = OUTPUT_CACHE / "tokenizer"

DATA_DIR = PROJECT_ROOT / "data"
SETUP_DIR = PROJECT_ROOT / "setup"

DEFAULT_CONFIG_PATH = OUTPUT_CONFIGS / "training_config.json"
DEFAULT_CHECKPOINT_DIR = OUTPUT_CHECKPOINTS / "run1"
DEFAULT_TRAINING_LOG = OUTPUT_LOGS / "training.log"
DEFAULT_TRAINING_PLOT = OUTPUT_LOGS / "training_plot_latest.png"
DEFAULT_LANDSCAPE_PLOT = OUTPUT_LOGS / "loss_landscape_latest.png"

DEFAULT_CHAT_URL = "http://127.0.0.1:7860"
DEFAULT_VIEWER_URL = "http://127.0.0.1:7861"
_WEIGHT_SUFFIXES = {".npz", ".npy", ".npx"}

# Legacy locations (pre-output/ migration) — still honored when passed explicitly.
LEGACY_LOGS_DIR = PROJECT_ROOT / "logs"
LEGACY_MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_CONFIG_PATH = SETUP_DIR / "training_config.json"

QUARTER_FRACTIONS = (0.25, 0.50, 0.75, 1.0)
QUARTER_NAMES = ("quarter_25", "quarter_50", "quarter_75", "quarter_100")
BEST_DIR_NAME = "best"


def ensure_output_dirs() -> None:
    """Create the output/ tree if missing."""
    for d in (
        OUTPUT_LOGS,
        OUTPUT_CHECKPOINTS,
        OUTPUT_CONFIGS,
        OUTPUT_TOKENIZER,
        OUTPUT_CACHE_TOKENIZER,
    ):
        d.mkdir(parents=True, exist_ok=True)


def resolve_checkpoints_dir(path: Optional[Union[str, Path]] = None) -> Path:
    """Return checkpoints root, defaulting to output/checkpoints."""
    if path is None:
        return OUTPUT_CHECKPOINTS
    return Path(path)


def resolve_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return config file path, defaulting to output/configs/training_config.json."""
    if path is None:
        return DEFAULT_CONFIG_PATH
    return Path(path)


def relative_to_project(path: Union[str, Path]) -> str:
    """Project-relative posix path. Absolute paths outside the repo stay as given."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _is_weight_file(path: Path) -> bool:
    return path.suffix.lower() in _WEIGHT_SUFFIXES


def checkpoint_weights_relpath(checkpoint: Union[str, Path]) -> str:
    """Relative weights.npz for a checkpoint dir, or the file if one was passed."""
    path = Path(checkpoint)
    weights = path if _is_weight_file(path) else path / "weights.npz"
    return relative_to_project(weights)


def _checkpoint_dir(checkpoint: Union[str, Path]) -> Path:
    path = Path(checkpoint)
    return path.parent if _is_weight_file(path) else path


def viewer_open_url(base: str, checkpoint: Union[str, Path]) -> str:
    """npzviewer URL that opens this checkpoint's weights."""
    from urllib.parse import quote

    root = str(base).rstrip("/")
    raw = str(checkpoint or "").strip()
    if not raw:
        return root + "/"
    return f"{root}/?path={quote(checkpoint_weights_relpath(raw), safe='/')}"


def chat_open_url(base: str, checkpoint: Union[str, Path]) -> str:
    """Chat URL for the checkpoint directory, not the weights file."""
    from urllib.parse import quote

    root = str(base).rstrip("/")
    raw = str(checkpoint or "").strip()
    if not raw:
        return root + "/"
    rel = relative_to_project(_checkpoint_dir(raw))
    return f"{root}/?checkpoint={quote(rel, safe='/')}"


def checkpoint_vocab_sidecar(checkpoint_dir: Union[str, Path]) -> Path:
    """Path for a vocab.json copy under output/tokenizer/ for a checkpoint run."""
    stem = Path(checkpoint_dir).name
    return OUTPUT_TOKENIZER / f"{stem}_vocab.json"


def quarter_name_for_fraction(fraction: float) -> str:
    """Map 0.25/0.50/0.75/1.0 → quarter_25/…/quarter_100."""
    pct = int(round(fraction * 100))
    return f"quarter_{pct}"


def quarter_dir(run_dir: Union[str, Path], fraction: float) -> Path:
    return Path(run_dir) / quarter_name_for_fraction(fraction)


def best_dir(run_dir: Union[str, Path]) -> Path:
    return Path(run_dir) / BEST_DIR_NAME


def run_root_for_checkpoint(checkpoint_dir: Union[str, Path]) -> Path:
    """If checkpoint is a nested quarter_*/best under a run, return the run root;
    otherwise return the checkpoint dir itself."""
    path = Path(checkpoint_dir)
    # Peel nested quarter_*/best even when the parent run root is not yet fully
    # materialized (fresh promote / partial trees).
    if path.name in QUARTER_NAMES or path.name == BEST_DIR_NAME:
        return path.parent
    return path


def list_quarter_dirs(run_dir: Union[str, Path]) -> List[Path]:
    """Return existing quarter_* dirs under a run, in milestone order."""
    root = Path(run_dir)
    found = []
    for name in QUARTER_NAMES:
        d = root / name
        if d.is_dir() and (d / "config.json").exists():
            found.append(d)
    return found
