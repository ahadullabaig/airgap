"""Python access to K0 (kernels/k0_changemap) through ctypes, plus a NumPy
reference used to cross-check the hand-off.

Frames must be HWC uint8 and C-contiguous. A transposed NumPy view keeps its
old memory order, so a CHW buffer could otherwise be passed as if it were HWC
and give plausible but wrong maps; `changemap` refuses anything else.
"""

import ctypes
from pathlib import Path

import numpy as np

LIB_PATH = Path(__file__).resolve().parents[1] / "kernels" / "k0_changemap" / "build" / "libk0.so"

_U8P = ctypes.POINTER(ctypes.c_uint8)
_F32P = ctypes.POINTER(ctypes.c_float)


class K0:
    def __init__(self, path: Path = LIB_PATH, threads: int | None = None):
        if not path.exists():
            raise FileNotFoundError(f"{path} not built; run `make` in kernels/k0_changemap")
        self.lib = ctypes.CDLL(str(path))
        for name in ("k0_changemap_serial", "k0_changemap_omp"):
            fn = getattr(self.lib, name)
            fn.argtypes = [_U8P, _U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _F32P]
            fn.restype = ctypes.c_int
        self.lib.k0_set_num_threads.argtypes = [ctypes.c_int]
        self.lib.k0_get_max_threads.restype = ctypes.c_int
        if threads:
            self.lib.k0_set_num_threads(threads)

    @property
    def threads(self) -> int:
        return self.lib.k0_get_max_threads()

    def changemap(self, a: np.ndarray, b: np.ndarray, block: int = 16, impl: str = "omp") -> np.ndarray:
        for x in (a, b):
            if x.dtype != np.uint8 or x.ndim != 3 or not x.flags["C_CONTIGUOUS"]:
                raise ValueError(f"need C-contiguous HWC uint8, got {x.dtype} {x.shape} contiguous={x.flags['C_CONTIGUOUS']}")
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
        h, w, c = a.shape
        out = np.empty(((h + block - 1) // block, (w + block - 1) // block), dtype=np.float32)
        fn = self.lib.k0_changemap_omp if impl == "omp" else self.lib.k0_changemap_serial
        rc = fn(a.ctypes.data_as(_U8P), b.ctypes.data_as(_U8P), h, w, c, block, out.ctypes.data_as(_F32P))
        if rc != 0:
            raise RuntimeError(f"{fn.__name__} returned {rc}")
        return out


def changemap_numpy(a: np.ndarray, b: np.ndarray, block: int = 16) -> np.ndarray:
    """Reference block mean absolute difference (image sides must be multiples of block)."""
    h, w, c = a.shape
    if h % block or w % block:
        raise ValueError("numpy reference needs sides divisible by block")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float64)
    return d.reshape(h // block, block, w // block, block, c).mean(axis=(1, 3, 4))
