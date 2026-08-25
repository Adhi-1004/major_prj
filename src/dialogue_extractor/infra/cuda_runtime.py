"""Making pip-installed CUDA libraries findable on Windows.

`nvidia-cublas-cu12` and `nvidia-cudnn-cu12` drop their DLLs under
`site-packages/nvidia/*/bin/`, which is not on the Windows DLL search path. So
installing them is not enough: CTranslate2 still reports
"Library cublas64_12.dll is not found or cannot be loaded" and inference falls
back to CPU, on a machine that has a perfectly good GPU.

Both PATH and `os.add_dll_directory` are used, because they are not
interchangeable here. Python 3.8+ ignores PATH for *extension-module* loading,
which is what `add_dll_directory` exists for -- but CTranslate2 loads cuBLAS
itself, at first inference, via its own `LoadLibrary` call, and that consults
PATH. Verified empirically on this stack: `add_dll_directory` alone still fails
with "Library cublas64_12.dll is not found", while prepending to PATH succeeds.

Must run before ctranslate2 loads its native library.

On Linux the wheels rely on RPATH and this is a no-op.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

_registered = False

# Every pip package that ships CUDA DLLs we might need.
_NVIDIA_SUBPACKAGES = (
    "cublas",
    "cudnn",
    "cuda_runtime",
    "cuda_nvrtc",
    "cufft",
)


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for sub in _NVIDIA_SUBPACKAGES:
            bin_dir = nvidia_root / sub / "bin"
            if bin_dir.is_dir():
                dirs.append(bin_dir)
    return dirs


def register_cuda_libraries() -> list[str]:
    """Add pip-installed CUDA DLL directories to the search path.

    Idempotent, and safe to call on machines with no GPU or no CUDA packages --
    it simply finds nothing. Returns the directories registered, for logging.
    """
    global _registered
    if _registered or sys.platform != "win32":
        return []

    registered: list[str] = []
    for directory in _candidate_dirs():
        text = str(directory)
        # The one that actually works for CTranslate2's runtime LoadLibrary.
        if text not in os.environ.get("PATH", ""):
            os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
        # Also register for extension-module loading, which ignores PATH.
        try:
            os.add_dll_directory(text)
        except (OSError, AttributeError) as exc:
            log.debug("cuda.dll_dir_failed", directory=text, error=type(exc).__name__)
        registered.append(text)

    if registered:
        log.debug("cuda.dll_dirs_registered", count=len(registered))
    _registered = True
    return registered


def reset() -> None:
    """Allow re-registration. Used by tests."""
    global _registered
    _registered = False
