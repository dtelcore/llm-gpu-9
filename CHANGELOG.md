# Changelog

Releases of **llm-gpu-9** on the GTX 1660 Ti. Older history (llm-gpu-8 through 0.1.3, Kepler GT 730) is in [`oldCHANGELOG.md`](oldCHANGELOG.md).

## 0.0.1 — 2026-09-30 — 1660 Ti port

First commit of this repo: [github.com/dtelcore/llm-gpu-9](https://github.com/dtelcore/llm-gpu-9), tag `v0.0.1`.

The training stack is the llm-gpu-8 workspace (NumPy reference, PyCUDA kernels, manual backward, AdamW, BPE, RMSNorm + RoPE presets). The machine target is a **GTX 1660 Ti (TU116, CC 7.5, 6 GB GDDR6)**.

- **Toolkit.** CUDA **13.2**, MSVC **14.51**, Windows SDK **10.0.22621.0**, Python **3.12**. `model/cuda/env.py` sets the DLL directory, `PATH`, and nvcc `-ccbin` / include flags before PyCUDA compiles. `setup/1_new_workspace_setup.ps1` builds that venv. `setup/2_test_workspace.py` JIT-compiles a vector-add and prints the device.
- **Kernels.** `SourceModule` JITs for the attached GPU, so this card gets `sm_75`. Warp reductions use `__shfl_down_sync` when `__CUDA_ARCH__ >= 700`.
- **Budget.** Process VRAM cap stays **2 GB**, with a **5.5 GB** soft guard. GEMM tile stays **16**. fp32 is still the train-step format. The 1660 Ti has no Tensor cores, and this release does not add a tensor-core or TF32 path.
- **Docs.** This changelog and [`README.md`](README.md) describe the 1660 Ti tree. The previous project write-up is kept as `oldREADME.md` / `oldCHANGELOG.md`.
- **Not in this tag.** No checked-in 1660 Ti tok/s baseline. `guide.md`, `py_calls.md`, and several module docstrings still name the GT 730.
