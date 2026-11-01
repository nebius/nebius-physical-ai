"""Private, service-owned storage for OpenPI hardware and checkpoint receipts."""

from __future__ import annotations

import os
import inspect
from pathlib import Path
import stat

DIAGNOSTICS_DIRECTORY = Path("/workspace/openpi-server-cache/diagnostics")
_RECEIPTS = {
    "npa-openpi-server-hardware.json",
    "npa-openpi-loaded-checkpoint.json",
}


def diagnostics_directory() -> Path:
    """Require a real directory owned by this service with private permissions."""
    path = DIAGNOSTICS_DIRECTORY
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise RuntimeError(
            "OpenPI diagnostics require a private service-owned directory"
        )
    return path


def diagnostic_path(name: str) -> Path:
    """Resolve only service receipts and reject symlink destinations."""
    if name not in _RECEIPTS:
        raise ValueError("unknown OpenPI diagnostic receipt")
    path = diagnostics_directory() / name
    if path.is_symlink():
        raise RuntimeError("OpenPI diagnostic receipt cannot be a symlink")
    return path


def standalone_program() -> str:
    """Emit the same storage contract for upstream images without NPA installed."""
    return (
        "import os, stat\nfrom pathlib import Path\n"
        f"DIAGNOSTICS_DIRECTORY = Path({str(DIAGNOSTICS_DIRECTORY)!r})\n"
        f"_RECEIPTS = {tuple(sorted(_RECEIPTS))!r}\n"
        + inspect.getsource(diagnostics_directory)
        + "\n"
        + inspect.getsource(diagnostic_path)
    )
