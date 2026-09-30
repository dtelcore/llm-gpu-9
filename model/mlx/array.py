"""GPUArray under the old DeviceArray name."""

from model.cuda.env import configure

configure()

from pycuda import gpuarray  # noqa: E402

DeviceArray = gpuarray.GPUArray


def empty(shape, dtype=None):
    import numpy as np

    return gpuarray.empty(tuple(shape), dtype=dtype or np.float32)


def zeros(shape, dtype=None):
    import numpy as np

    return gpuarray.zeros(tuple(shape), dtype=dtype or np.float32)
