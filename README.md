# llm-gpu-9

**Version:** 0.0.1  
**Repo:** [github.com/dtelcore/llm-gpu-9](https://github.com/dtelcore/llm-gpu-9)  
**GPU:** NVIDIA GeForce GTX 1660 Ti (TU116, compute capability 7.5, `sm_75`)  
**Stack:** NumPy + PyCUDA GPT. Live path is GPU forward, GPU loss, GPU manual backward, GPU AdamW.

Ninth iteration of this training workspace. The model, checkpoints, and CLI come from [llm-gpu-8](https://github.com/dtelcore/llm-gpu-8). v0.0.1 is the machine port onto a 6 GB Turing card with CUDA 13.2 and Python 3.12. Kernels are still hand-written fp32 CUDA, JIT-compiled by nvcc for the GPU that is attached when the process starts.

The 1660 Ti is Turing without Tensor cores, so this release does not switch the train path to TF32 or tensor-core GEMM. fp16 in this tree is storage, not the math format of the training step.

---

## Hardware and software

| Component | This machine |
|-----------|----------------|
| GPU | GTX 1660 Ti, TU116, **CC 7.5** (`sm_75`) |
| VRAM | **6 GB** GDDR6 |
| Toolkit | CUDA **13.2** (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`) |
| Host compiler | MSVC **14.51** (Visual Studio 2026 Build Tools) |
| Windows SDK | **10.0.22621.0** |
| Python | **3.12** virtualenv at `venv/` |
| OS | Windows 10 (WDDM) |

`model/cuda/env.py` registers the CUDA 13.2 DLL directory, puts `nvcc` and `cl.exe` on `PATH`, and passes `-ccbin` plus the MSVC and SDK include paths into every `SourceModule`. Import `model.cuda.ops` (or call `model.cuda.env.configure()`) before any direct `pycuda` import.

A 2 GB process budget and a 5.5 GB soft guard are still enforced in `model/cuda/env.py`. The card has 6 GB; raising that cap means updating the tests that lock `PROCESS_BUDGET_BYTES`. The shared-memory GEMM tile is still 16, carried from the previous card.

---

## What v0.0.1 runs

| Piece | Behavior |
|-------|----------|
| Device | PyCUDA `SourceModule` JIT for the attached GPU (`sm_75` on this box) |
| Warp reductions | `__shfl_down_sync` when `__CUDA_ARCH__ >= 700` |
| Train step | Tiled GEMM, fused residual norm, causal attention, manual backward, AdamW |
| Norm / positions | New presets use RMSNorm + RoPE. Older checkpoints keep LayerNorm + learned positions |
| Host reference | NumPy analytic backward for parity |
| Tokenizer | BPE by default (200 merges). `--tokenizer char` opts into the character tokenizer |
| Checkpoints | `output/checkpoints/<run>/` with latest, `quarter_*`, and optional `best/` |
| Verification | `.\venv\Scripts\python.exe -m tests.parity.run_parity` |

Presets used by `train.py --menu`:

| Key | Model | Training |
|-----|--------|----------|
| `toy` | C=16, L=1, T=8 | LR 0.01, batch 64, built-in `minimal` corpus |
| `story_sub1m` | C=128, 8 heads, L=4, T=128 | ~0.83M params, LR 5e-4, batch 8, accum 2 |
| `tiny_stories` | C=256, 8 heads, L=4, T=128 | ~3M params, LR 3e-4, batch 4, accum 4 |
| `chat_5m` | C=256, 8 heads, L=6, T=128 | ~5M params, same recipe as TinyStories |

Configs: `setup/story_sub1m_config.json`, `setup/tiny_stories_v2_config.json`, `setup/chat_5m_config.json`. GPU training ignores dropout; leave `dropout_prob` at 0.

No 1660 Ti step-time baseline is checked in yet. Treat Kepler tok/s figures in `guide.md` and `oldREADME.md` as history from the GT 730, not as numbers for this card.

---

## Setup

Once per machine, from an elevated PowerShell prompt:

```powershell
cd C:\dev\llm-gpu-9
powershell -ExecutionPolicy Bypass -File .\setup\1_new_workspace_setup.ps1
```

That script expects Python 3.12, CUDA 13.2, and MSVC 14.51 already installed. It creates `venv\`, then builds PyCUDA against that toolkit.

Smoke-test the driver and a JIT vector-add:

```powershell
cd C:\dev\llm-gpu-9
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python setup\2_test_workspace.py
```

`Process` lasts for the current window. If activation is blocked, call the venv interpreter directly:

```powershell
.\venv\Scripts\python.exe setup\2_test_workspace.py
```

A good run prints the device name `NVIDIA GeForce GTX 1660 Ti`, compute capability `(7, 5)`, about 6144 MB, and a passing vector-add.

Install the rest of the train/UI dependencies after that smoke test:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Train and sample

Put story text in `data\` (one document per line). `data/` is gitignored.

```powershell
cd C:\dev\llm-gpu-9
.\venv\Scripts\python.exe train.py --config setup\story_sub1m_config.json `
  --checkpoint output\checkpoints\story_sub1m `
  --steps 2000 --no-prompt
```

Or `python train.py --menu` and pick a scaling preset. The same wizard is `auto_train.py --menu`. Sample with:

```powershell
.\venv\Scripts\python.exe generate.py --checkpoint output\checkpoints\story_sub1m `
  --prompt "once upon a" --max-new-tokens 256 `
  --temperature 0.6 --top-k 10 --top-p 0.9
```

Flag catalog: [`py_calls.md`](py_calls.md). Preset notes that still describe the old card: [`guide.md`](guide.md).

---

## Repository layout

```text
llm-gpu-9/
├── README.md                 ← this file
├── CHANGELOG.md              ← 1660 Ti releases, starting at 0.0.1
├── oldREADME.md              ← llm-gpu-8 write-up (GT 730, through 0.1.3)
├── oldCHANGELOG.md           ← changelog from that tree
├── VERSION / version.py      ← 0.0.1
├── train.py / auto_train.py / generate.py / interactive.py
├── model/cuda/               ← kernels, ops, env bootstrap for CUDA 13.2
├── training/                 ← dataset, loss, AdamW, checkpoints
├── tokenizer/                ← BPE default, character tokenizer optional
├── setup/                    ← presets, 1660 Ti workspace script, smoke test
├── tests/parity/             ← NumPy ↔ CUDA checks
├── data/                     ← corpora (not in git)
└── output/                   ← logs, checkpoints, configs (mostly not in git)
```

Runtime files live under `output/` as described in [`output/README.md`](output/README.md). `venv/`, `data/`, logs, checkpoints, and `setup/training_config.json` stay out of git (see `.gitignore`).

---

## Versioning

`VERSION` is the string `version.py` exposes as `__version__`. This repo starts at **0.0.1**, the 1660 Ti port. History through llm-gpu-8 **0.1.3** is in [`oldREADME.md`](oldREADME.md) and [`oldCHANGELOG.md`](oldCHANGELOG.md).
