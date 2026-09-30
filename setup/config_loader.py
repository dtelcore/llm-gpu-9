"""Load training config and resolve dataset corpus without embedding it in JSON."""

from pathlib import Path
from typing import Dict, List

from paths import DATA_DIR, PROJECT_ROOT
from setup.dataset_setup import COMBINED_DATASET_NAME, DatasetLoader


def is_combined_dataset(dataset_cfg: Dict) -> bool:
    """True when config asks to concatenate every data/*.txt file."""
    if dataset_cfg.get("combine"):
        return True
    return dataset_cfg.get("name") == COMBINED_DATASET_NAME


def resolve_dataset_corpus(dataset_cfg: Dict, data_dir: str = None) -> List[str]:
    """Return corpus text from inline config, combined data dir, or dataset name.

    Combined-directory mode (`combine: true` or `name: data_dir`) loads every
    non-empty line from sorted `*.txt` files under `data_dir`. Inline `corpus`
    still wins so tests and baked-in configs keep working. Single-file stems
    and built-in names are unchanged.
    """
    corpus = dataset_cfg.get("corpus")
    if corpus:
        return corpus

    path = dataset_cfg.get("path") or dataset_cfg.get("dataset_path")
    if path:
        target = Path(path)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        loader = DatasetLoader(data_dir=str(target.parent if target.is_file() else target))
        if target.is_file():
            return loader.load_from_file(str(target), dataset_cfg.get("name"))
        if target.is_dir() and is_combined_dataset(dataset_cfg):
            return loader.load_combined_directory(str(target))
        if target.is_dir():
            return loader.load_by_name(dataset_cfg.get("name", "minimal"))
        raise FileNotFoundError(f"Dataset path not found: {target}")

    resolved_dir = data_dir or str(DATA_DIR)
    loader = DatasetLoader(data_dir=resolved_dir)
    if is_combined_dataset(dataset_cfg):
        return loader.load_combined_directory(resolved_dir)

    name = dataset_cfg.get("name", "minimal")
    return loader.load_by_name(name)
