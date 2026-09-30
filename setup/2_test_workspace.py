import os
import sys

# ------------------------------------------------------------------------------
# 1. Environment & Path Handshakes (Blockers 1 & 2 Fixes)
# ------------------------------------------------------------------------------
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
MSVC_BIN = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
WIN_SDK_BIN = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"

# Fix Python 3.8+ DLL loading isolation
if hasattr(os, "add_dll_directory") and os.path.exists(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)

os.environ["PATH"] = f"{CUDA_BIN};{MSVC_BIN};{WIN_SDK_BIN};" + os.environ["PATH"]

# ------------------------------------------------------------------------------
# 2. NVCC Options Definition
# ------------------------------------------------------------------------------
_msvc = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
_winsdk  = r"C:\Program Files (x86)\Windows Kits\10"
_sdk_ver = "10.0.22621.0"

_NVCC_OPTIONS = [
    f"-ccbin={_msvc}\\bin\\Hostx64\\x64\\cl.exe",
    f"-I{_msvc}\\include",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\ucrt",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\um",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\shared",
]

print("=" * 60)
print("RUNNING CUDA & PYCUDA WORKSPACE VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------------------------
# Test Stage A: Framework & Driver Interactivity
# ------------------------------------------------------------------------------
try:
    import pycuda
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    from pycuda.compiler import SourceModule
    
    device = cuda.Device(0)
    print(f"[SUCCESS] PyCUDA Module Bindings: Operational")
    print(f"[DEVICE]  Name: {device.name()}")
    print(f"[DEVICE]  Compute Capability: {device.compute_capability()}")
    print(f"[DEVICE]  Total Memory: {device.total_memory() // (1024**2)} MB")
except Exception as e:
    print(f"[FAILURE] Stage A (Driver/Initialization) failed.")
    print(f"ERROR: {e}")
    sys.exit(1)

print("-" * 60)

# ------------------------------------------------------------------------------
# Test Stage B: Real-Time Kernel Compilation & Math Execution (Header Patches Check)
# ------------------------------------------------------------------------------
try:
    print("Attempting runtime JIT compilation of vector addition kernel...")
    
    # Simple vector addition kernel
    mod = SourceModule("""
    __global__ void add(float *a, float *b, float *c) {
        int i = threadIdx.x;
        c[i] = a[i] + b[i];
    }
    """, options=_NVCC_OPTIONS)
    
    add_kernel = mod.get_function("add")
    
    # Generate mock data
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([10, 20, 30, 40], dtype=np.float32)
    c = np.zeros_like(a)
    
    # Run hardware compute pipeline
    add_kernel(cuda.In(a), cuda.In(b), cuda.Out(c), block=(4, 1, 1))
    
    print(f"Kernel Result Array: {c}")
    assert list(c) == [11.0, 22.0, 33.0, 44.0], f"Mathematical mismatch detected: {c}"
    
    print("-" * 60)
    print("[SUCCESS] Kernel JIT Compilation & STL bindings passed.")
    cc = device.compute_capability()
    print(f"[SUCCESS] {device.name()} (Compute Capability {cc[0]}.{cc[1]}) compiled and ran the kernel.")
    print("=" * 60)

except Exception as e:
    print(f"[FAILURE] Stage B (Kernel compilation/execution) failed.")
    print("This usually implies a broken compiler path or unapplied header patch.")
    print(f"ERROR: {e}")
    sys.exit(1)