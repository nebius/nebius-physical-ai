"""Extract upstream kitchen archives without following links or escaping assets."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile, ZipInfo


def _asset_target(root: Path, member: ZipInfo) -> Path:
    name = member.filename
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Kitchen archive contains an unsafe path")
    mode = stat.S_IFMT(member.external_attr >> 16)
    if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError("Kitchen archive contains a link or special file")
    target = root.joinpath(*relative.parts)
    if target == root:
        raise ValueError("Kitchen archive entry replaces its destination")
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError("Kitchen archive destination contains a symbolic link")
        if component == root:
            break
    if not target.resolve().is_relative_to(root):
        raise ValueError("Kitchen archive entry escapes its destination")
    return target


def extract_asset_archive(archive_path: Path, destination: Path) -> None:
    """Extract regular kitchen assets after validating every archive member.

    Args:
        archive_path: Downloaded, revision-pinned upstream ZIP archive.
        destination: Owned assets directory that will receive the archive.

    Returns:
        None.

    Raises:
        ValueError: An entry escapes the destination, repeats a path, or uses links.
        OSError: An asset cannot be read or written.
        zipfile.BadZipFile: Archive structure or a payload checksum is invalid.
    """
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("Kitchen asset root must not be a symbolic link")
    root = destination.resolve()
    with ZipFile(archive_path) as archive:
        entries = [(member, _asset_target(root, member)) for member in archive.infolist()]
        targets = [target for _member, target in entries]
        if len(targets) != len(set(targets)):
            raise ValueError("Kitchen archive repeats an asset path")
        for member, target in entries:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
