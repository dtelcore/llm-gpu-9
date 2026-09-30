"""Load training config and resolve dataset corpus without embedding it in JSON."""

from typing import Dict, List

from paths import DATA_DIR
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

    resolved_dir = data_dir or str(DATA_DIR)
    loader = DatasetLoader(data_dir=resolved_dir)
    if is_combined_dataset(dataset_cfg):
        return loader.load_combined_directory(resolved_dir)

    name = dataset_cfg.get("name", "minimal")
    return loader.load_by_name(name)
