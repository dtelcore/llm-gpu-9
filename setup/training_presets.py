"""
Scaling presets: bundled model architecture + training hyperparameters.

Used by the --menu wizard in train.py and auto_train.py.
"""

from typing import Dict, List, Tuple

from setup.dataset_setup import COMBINED_DATASET_NAME
from setup.model_config import PRESETS, estimate_vram_footprint

# Bundled presets: model + hyperparameters + default dataset
SCALE_PRESETS: Dict[str, Dict] = {
    'toy': {
        'name': 'Toy Run',
        'tagline': 'Quick smoke test',
        'model_key': 'toy',
        'dataset': 'minimal',
        'hyperparameters': {
            'name': 'Toy Run',
            'learning_rate': 0.01,
            'weight_decay': 0.01,
            'batch_size': 64,
            'num_epochs': 10,
            'warmup_steps': 100,
            'gradient_clip': 1.0,
            'gradient_accumulation_steps': 1,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
    },
    'story_sub1m': {
        'name': 'Fast Stories (<1M)',
        'tagline': 'Story training under 1M params (C=128, L=4, T=128)',
        'model_key': 'story_sub1m',
        'dataset': COMBINED_DATASET_NAME,
        'hyperparameters': {
            'name': 'Fast Stories Run',
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'batch_size': 8,
            'num_epochs': 1,
            'warmup_steps': 500,
            'gradient_clip': 1.0,
            'gradient_accumulation_steps': 2,
            'window_stride': 64,
            'min_lr_ratio': 0.1,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
    },
    'tiny_stories': {
        'name': 'Tiny Stories (real run)',
        'tagline': 'TinyStories-capable ~3M params (C=256, L=4, T=128)',
        'model_key': 'tiny_stories',
        'dataset': 'tiny_stories',
        'hyperparameters': {
            'name': 'Tiny Stories Run',
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'batch_size': 4,
            'num_epochs': 1,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'gradient_accumulation_steps': 4,
            'window_stride': 64,
            'min_lr_ratio': 0.1,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
    },
    'chat_5m': {
        'name': 'Chat 5M',
        'tagline': 'Ultra-tiny dialogue ~5M params (C=256, L=6, T=128)',
        'model_key': 'chat_5m',
        'dataset': COMBINED_DATASET_NAME,
        'hyperparameters': {
            'name': 'Chat 5M Run',
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'batch_size': 4,
            'num_epochs': 1,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'gradient_accumulation_steps': 4,
            'window_stride': 64,
            'min_lr_ratio': 0.1,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
    },
}

# Menu order (by size). Custom is appended after these.
SCALE_PRESET_MENU_ORDER: List[str] = ['toy', 'story_sub1m', 'tiny_stories', 'chat_5m']


def model_from_preset(preset_key: str, vocab_size: int = 100) -> Dict:
    """Build a model config dict from a PRESETS key."""
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown model preset: {preset_key}")
    cfg = PRESETS[preset_key].copy()
    cfg['vocab_size'] = vocab_size
    return cfg


def apply_scale_preset(scale_key: str, vocab_size: int = 100) -> Tuple[Dict, Dict, str]:
    """Return (model_config, hyperparameters, dataset_name) for a scale preset."""
    if scale_key not in SCALE_PRESETS:
        raise ValueError(f"Unknown scale preset: {scale_key}")
    scale = SCALE_PRESETS[scale_key]
    model = model_from_preset(scale['model_key'], vocab_size=vocab_size)
    hyperparams = scale['hyperparameters'].copy()
    return model, hyperparams, scale['dataset']


def _param_estimate(model_key: str, vocab_size: int = 110) -> int:
    cfg = model_from_preset(model_key, vocab_size=vocab_size)
    return estimate_vram_footprint(cfg)['total_params']


def _dataset_menu_note(scale_key: str) -> str:
    dataset = SCALE_PRESETS[scale_key]['dataset']
    if dataset == COMBINED_DATASET_NAME:
        return f"{dataset} (combine all data/*.txt)"
    if scale_key == 'toy':
        return f"{dataset} (built-in)"
    return f"{dataset} (data/*.txt)"


def print_scale_preset_menu(vocab_size: int = 110) -> None:
    """Print the scaling preset table for the training wizard."""
    print("\nScaling presets (model + hyperparameters + recommended dataset):")
    print("-" * 70)
    for i, key in enumerate(SCALE_PRESET_MENU_ORDER, 1):
        scale = SCALE_PRESETS[key]
        model = PRESETS[scale['model_key']]
        n = _param_estimate(scale['model_key'], vocab_size)
        hp = scale['hyperparameters']
        accum = hp.get('gradient_accumulation_steps', 1)
        print(f"  {i}. {scale['name']} — {scale['tagline']}")
        print(
            f"       embed={model['embedding_dim']}  heads={model['num_heads']}  "
            f"layers={model['num_layers']}  seq={model['max_len']}  "
            f"batch={hp['batch_size']}"
            + (f"  accum={accum}" if accum > 1 else "")
            + f"  ~{n:,} params"
        )
        extra = []
        if 'min_lr_ratio' in hp:
            extra.append(
                f"LR={hp['learning_rate']}  warmup={hp['warmup_steps']}  "
                f"cosine min_lr_ratio={hp['min_lr_ratio']}  "
                f"window_stride={hp.get('window_stride', 1)}"
            )
        if extra:
            print(f"       {extra[0]}")
            print("       rmsnorm+rope+tied  dropout=0 (unimplemented if >0)")
        print(f"       Dataset: {_dataset_menu_note(key)}")
        print()
    custom_n = len(SCALE_PRESET_MENU_ORDER) + 1
    print(f"  {custom_n}. Custom (pick model + hyperparameters separately)")
    print("-" * 70)


def prompt_scale_preset() -> str:
    """Interactively choose a bundled scale preset or custom. Returns preset key."""
    print_scale_preset_menu()
    custom_n = len(SCALE_PRESET_MENU_ORDER) + 1
    choice = input(f"\nSelect scaling preset (1-{custom_n}) [default=1]: ").strip()
    if choice == str(custom_n):
        return 'custom'
    try:
        idx = int(choice)
    except ValueError:
        return 'toy'
    if 1 <= idx <= len(SCALE_PRESET_MENU_ORDER):
        return SCALE_PRESET_MENU_ORDER[idx - 1]
    return 'toy'
