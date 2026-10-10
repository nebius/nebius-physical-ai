"""Create deterministic archives for generated EmbodiedGen simulator assets."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_asset_tree(source: Path, destination: Path) -> dict[str, Any]:
    """Archive a generated asset tree with normalized gzip and tar metadata."""

    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"asset directory is missing or unsafe: {source}")
    root = source.resolve()
    temporary = destination.with_name(f".{destination.name}.tmp")
    members: list[dict[str, Any]] = []
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in sorted(root.rglob("*")):
                        if path.is_symlink():
                            raise RuntimeError(
                                "generated asset tree contains a symbolic link"
                            )
                        if path.is_dir():
                            continue
                        if not path.is_file():
                            raise RuntimeError(
                                "generated asset tree contains a non-regular file"
                            )
                        resolved = path.resolve()
                        if not resolved.is_relative_to(root):
                            raise RuntimeError(
                                "generated asset escapes its declared output directory"
                            )
                        member = archive.gettarinfo(
                            str(path), arcname=path.relative_to(root.parent).as_posix()
                        )
                        member.uid = member.gid = 0
                        member.uname = member.gname = "root"
                        member.mtime = 0
                        member.mode = 0o644
                        with path.open("rb") as handle:
                            archive.addfile(member, handle)
                        members.append(
                            {
                                "path": path.relative_to(root.parent).as_posix(),
                                "sha256": _sha256(path),
                                "bytes": path.stat().st_size,
                            }
                        )
        if not members:
            raise RuntimeError("generated asset tree is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": destination.name,
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "members": members,
    }
