"""Extract USDZ into private audit directories without changing the rendered package."""

from __future__ import annotations

from pathlib import Path
import shutil
import stat
import zipfile

from .contract import _contained, _file_hashes


def _package_members(archive):
    members = archive.infolist()
    names = [member.filename for member in members]
    if not names or len(names) != len(set(names)):
        raise ValueError("USDZ requires nonempty, unique package members")
    _file_hashes(dict.fromkeys(names, "0" * 64))
    if Path(names[0]).suffix not in {".usd", ".usda", ".usdc"}:
        raise ValueError("USDZ first member must be its USD root layer")
    for member in members:
        mode = member.external_attr >> 16
        if (
            member.is_dir()
            or member.compress_type != zipfile.ZIP_STORED
            or member.flag_bits & 1
            or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
        ):
            raise ValueError("USDZ requires stored, unencrypted regular files")
    return members


def _extract_package(path, root):
    with zipfile.ZipFile(path) as archive:
        members = _package_members(archive)
        for member in members:
            target = _contained(root, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return [member.filename for member in members]
