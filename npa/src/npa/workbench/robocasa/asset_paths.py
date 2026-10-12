"""No-follow directory handles for RoboCasa's mutable asset publication paths."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class HeldAssetDirectory:
    """Keep every ancestor open and verify its namespace binding before writes."""

    path: Path
    descriptor: int
    bindings: tuple[tuple[int, str, int], ...]

    @property
    def anchored_path(self) -> Path:
        """Linux descriptor path for APIs that cannot accept a directory fd."""
        return Path("/proc/self/fd") / str(self.descriptor)

    def verify(self) -> None:
        """Reject a removed, renamed or replaced ancestor without following links."""
        for parent, name, descriptor in self.bindings:
            observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            held = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino, observed.st_mode) != (
                held.st_dev,
                held.st_ino,
                held.st_mode,
            ):
                raise OSError(f"RoboCasa asset directory binding changed: {self.path}")


@contextlib.contextmanager
def hold_asset_directory(
    path: Path, *, create: bool = False
) -> Iterator[HeldAssetDirectory]:
    """Open an absolute directory chain with O_NOFOLLOW, optionally creating it.

    Every mutation through the returned descriptor remains anchored even if an
    attacker replaces a pathname after verification. Callers must keep this
    context alive for all operations and use verify before publishing receipts.
    """
    absolute = path.absolute()
    if ".." in absolute.parts:
        raise OSError("RoboCasa asset directory cannot contain parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptors = [os.open(absolute.anchor, flags)]
    bindings = []
    try:
        for name in absolute.parts[1:]:
            parent = descriptors[-1]
            try:
                child = os.open(name, flags, dir_fd=parent)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    pass
                child = os.open(name, flags, dir_fd=parent)
            descriptors.append(child)
            bindings.append((parent, name, child))
        held = HeldAssetDirectory(absolute, descriptors[-1], tuple(bindings))
        held.verify()
        yield held
        held.verify()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
