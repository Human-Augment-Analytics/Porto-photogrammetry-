import ctypes
from importlib.util import find_spec
from pathlib import Path


def _preload_torch_nvjitlink() -> None:
    """Prefer PyTorch's bundled nvJitLink over incompatible copies on LD_LIBRARY_PATH."""
    spec = find_spec("torch")
    if spec is None or spec.origin is None:
        return

    torch_dir = Path(spec.origin).resolve().parent
    candidate = torch_dir.parent / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
    if not candidate.exists():
        return

    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


_preload_torch_nvjitlink()

from . import tetranerf_cpp_extension as cpp

triangulate = cpp.triangulate
