"""Retain verified cuDNN runtime libraries and notices before a layer commits."""

from __future__ import annotations

import base64
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sysconfig


_DIST_INFO = "nvidia_cudnn_cu12-9.20.0.48.dist-info"
_GENERATED = {f"{_DIST_INFO}/{name}" for name in ("RECORD", "INSTALLER", "REQUESTED")}
_PATTERNS = {
    "omit_header": re.compile(r"nvidia/cudnn/include/cudnn\w*\.h"),
    "retain_runtime": re.compile(r"nvidia/cudnn/lib/libcudnn\w*\.so\.9"),
    "retain_metadata": re.compile(
        re.escape(_DIST_INFO)
        + r"/(?:METADATA|WHEEL|top_level\.txt|licenses/License\.txt)"
    ),
}


def _regular_member(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("cuDNN member escapes the installation")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("cuDNN members and parents must not be symlinks")
    if not path.is_file():
        raise ValueError("cuDNN member is not a regular file")
    return path


def _member_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_members(manifest: dict) -> dict[str, dict]:
    if (
        manifest.get("schema_version") != "npa.robocasa.cudnn-runtime-boundary.v1"
        or manifest.get("distribution") != "nvidia-cudnn-cu12"
        or manifest.get("version") != "9.20.0.48"
    ):
        raise ValueError("Unreviewed cuDNN manifest identity")
    members = {}
    for item in manifest["files"]:
        pattern = _PATTERNS.get(item["role"])
        if pattern is None or not pattern.fullmatch(item["path"]):
            raise ValueError("Unreviewed cuDNN member role or path")
        if item["path"] in members:
            raise ValueError("Duplicate cuDNN manifest member")
        members[item["path"]] = item
    if {item["role"] for item in members.values()} != set(_PATTERNS):
        raise ValueError("cuDNN runtime, notice and header boundaries are required")
    if f"{_DIST_INFO}/licenses/License.txt" not in members:
        raise ValueError("cuDNN license notice is required")
    return members


def _verify_inventory(root: Path, members: dict[str, dict]) -> list[list[str]]:
    distributions = list(metadata.distributions(path=[str(root)]))
    matches = [
        dist for dist in distributions if dist.metadata["Name"] == "nvidia-cudnn-cu12"
    ]
    if len(matches) != 1 or matches[0].version != "9.20.0.48":
        raise ValueError("Expected the exact pinned cuDNN distribution")
    expected = set(members) | _GENERATED
    actual = set()
    for directory in (root / "nvidia/cudnn", root / _DIST_INFO):
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("cuDNN inventory contains a symlink")
            if not path.is_dir():
                actual.add(str(path.relative_to(root)))
    if actual != expected:
        raise ValueError("cuDNN installed population differs from reviewed inventory")
    for name in expected:
        _regular_member(root, name)
    if (root / _DIST_INFO / "INSTALLER").read_bytes() != b"pip\n":
        raise ValueError("Unexpected cuDNN installer metadata")
    if (root / _DIST_INFO / "REQUESTED").read_bytes():
        raise ValueError("Unexpected cuDNN requested metadata")
    with (root / _DIST_INFO / "RECORD").open(newline="") as stream:
        records = list(csv.reader(stream))
    if any(len(row) != 3 for row in records):
        raise ValueError("Malformed cuDNN RECORD")
    if len(records) != len(expected) or {row[0] for row in records} != expected:
        raise ValueError("cuDNN RECORD does not bind its complete population")
    return records


def _verify_bytes(
    root: Path, members: dict[str, dict], records: list[list[str]]
) -> None:
    record_by_name = {row[0]: row for row in records}
    for name, item in members.items():
        path = _regular_member(root, name)
        if path.stat().st_size != item["bytes"] or _member_hash(path) != item["sha256"]:
            raise ValueError("cuDNN file differs from the pinned public wheel")
        encoded = (
            base64.urlsafe_b64encode(bytes.fromhex(item["sha256"])).decode().rstrip("=")
        )
        if record_by_name[name][1:] != [f"sha256={encoded}", str(item["bytes"])]:
            raise ValueError("cuDNN RECORD digest or size differs from verified bytes")
        if item["role"] == "retain_runtime":
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("cuDNN runtime is not an ELF library")


def filter_cudnn_runtime(site_packages: Path, manifest_path: Path) -> dict:
    """Remove only reviewed SDK headers after verifying the whole wheel population.

    Args:
        site_packages: Task-owned installed dependency root, not a shared environment.
        manifest_path: Committed exact public-wheel member hashes and roles.
    Returns:
        A receipt retaining the source wheel identity and every retained file hash.
    Raises:
        ValueError: Identity, population, paths, bytes or package metadata disagree.
        OSError: Reading or removing the selected installation fails.
    """
    if site_packages.is_symlink():
        raise ValueError("cuDNN installation root must not be a symlink")
    root = site_packages.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    members = _manifest_members(manifest)
    records = _verify_inventory(root, members)
    _verify_bytes(root, members, records)
    omitted = sorted(
        name for name, item in members.items() if item["role"] == "omit_header"
    )
    # Verification precedes every deletion. This build-only operation does not
    # repair an existing image: it must share the original pip-install RUN layer.
    for name in omitted:
        (root / name).unlink()
    with (root / _DIST_INFO / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(row for row in records if row[0] not in omitted)
    return {
        "schema_version": manifest["schema_version"],
        "wheel_sha256": manifest["wheel_sha256"],
        "omitted_headers": omitted,
        "retained_sha256": {
            name: item["sha256"]
            for name, item in members.items()
            if name not in omitted
        },
    }


if __name__ == "__main__":
    print(
        json.dumps(
            filter_cudnn_runtime(
                Path(sysconfig.get_paths()["purelib"]),
                Path(__file__).with_name("cudnn-runtime-boundary.json"),
            ),
            indent=2,
        )
    )
