"""
model/cuda/env.py

CUDA/MSVC environment bootstrap for the GTX 1660 Ti (sm_75) port.

Mirrors the machine this tree is built on:
CUDA 13.2 toolkit + MSVC 14.51 (VS 2026 Build Tools) + Windows 10 SDK 10.0.22621.0.
Must be imported (and `configure()` called) BEFORE any `pycuda` import,
otherwise nvcc cannot find a compatible host compiler on Windows.
"""

import os

from logging_config import logger

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
MSVC_ROOT = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
MSVC_BIN = MSVC_ROOT + r"\bin\Hostx64\x64"
WINSDK_ROOT = r"C:\Program Files (x86)\Windows Kits\10"
WINSDK_VER = "10.0.22621.0"
WINSDK_BIN = f"{WINSDK_ROOT}\\bin\\{WINSDK_VER}\\x64"

NVCC_OPTIONS = [
    f"-ccbin={MSVC_BIN}\\cl.exe",
    f"-I{MSVC_ROOT}\\include",
    f"-I{WINSDK_ROOT}\\Include\\{WINSDK_VER}\\ucrt",
    f"-I{WINSDK_ROOT}\\Include\\{WINSDK_VER}\\um",
    f"-I{WINSDK_ROOT}\\Include\\{WINSDK_VER}\\shared",
]

_configured = False


def configure() -> None:
    """Wire up PATH / DLL search directories for CUDA 13.2 + MSVC 14.51.

    Idempotent: safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    if hasattr(os, "add_dll_directory") and os.path.exists(CUDA_BIN):
        os.add_dll_directory(CUDA_BIN)

    os.environ["PATH"] = f"{CUDA_BIN};{MSVC_BIN};{WINSDK_BIN};" + os.environ["PATH"]

    logger.debug("CUDA environment configured: CUDA_BIN=%s", CUDA_BIN)
    _configured = True


# Planner cap carried over from the Apple tree. Tests assert this exact value.
# The 1660 Ti has 6 GB; raise this only together with those tests.
PROCESS_BUDGET_BYTES = 2 * 1024 ** 3
SOFT_MACHINE_BYTES = int(5.5 * 1024 ** 3)
PEAK_TRANSIENT_BYTES = 64 * 1024 ** 2

_peak_bytes = 0


class MemoryBudgetError(RuntimeError):
    """Raised when process VRAM exceeds the budget or the soft machine guard."""


def get_active_memory() -> int:
    from model.cuda.ops import get_memory_usage

    return int(get_memory_usage()["process_used_bytes"])


def get_peak_memory() -> int:
    global _peak_bytes
    active = get_active_memory()
    if active > _peak_bytes:
        _peak_bytes = active
    return int(_peak_bytes)


def reset_peak_memory() -> None:
    global _peak_bytes
    _peak_bytes = get_active_memory()


def get_cache_memory() -> int:
    return 0


def process_budget_exceeded(active: int, peak: int, limit: int, transient: int) -> bool:
    """True if resident is over ``limit``, or peak is over ``limit + transient``."""
    return int(active) > int(limit) or int(peak) > int(limit) + int(transient)


def check_memory(where: str = "") -> dict:
    """Read process VRAM and abort if it exceeds the budget or the soft guard."""
    active = get_active_memory()
    peak = get_peak_memory()
    loc = f" ({where})" if where else ""
    if process_budget_exceeded(active, peak, PROCESS_BUDGET_BYTES, PEAK_TRANSIENT_BYTES):
        raise MemoryBudgetError(
            f"process VRAM exceeded 2 GB budget{loc}: "
            f"active={active / (1024 ** 2):.1f} MB peak={peak / (1024 ** 2):.1f} MB"
        )
    if process_budget_exceeded(active, peak, SOFT_MACHINE_BYTES, PEAK_TRANSIENT_BYTES):
        raise MemoryBudgetError(
            f"process VRAM exceeded 5.5 GB soft guard{loc}: "
            f"active={active / (1024 ** 2):.1f} MB peak={peak / (1024 ** 2):.1f} MB"
        )
    usage = {
        "process_used_bytes": int(active),
        "peak_bytes": int(peak),
        "cache_bytes": 0,
        "driver_free_bytes": 0,
        "driver_total_bytes": int(PROCESS_BUDGET_BYTES),
        "driver_used_bytes": int(active),
        "source": "cuda",
    }
    try:
        from model.cuda.ops import get_memory_usage

        usage.update(get_memory_usage())
        usage["peak_bytes"] = int(peak)
        usage["source"] = "cuda"
    except Exception:
        pass
    return usage
