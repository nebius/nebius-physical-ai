"""Retain only reviewed cuDNN shared libraries and notices before committing a layer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sysconfig

VERSION = "9.10.2.21"
DIST_INFO = f"nvidia_cudnn_cu12-{VERSION}.dist-info"
LICENSE_PATH = f"{DIST_INFO}/licenses/License.txt"
LICENSE_SHA256 = "49cf79bdb35734b52fe6203013b3bd759f81e998cd32aa2c65c51db9a88c61d2"
RUNTIME_PATH = re.compile(r"nvidia/cudnn/lib/libcudnn\w*\.so(?:\.\d+)*")
DEVELOPMENT_PATH = re.compile(r"nvidia/cudnn/(?:include/cudnn\w*\.h|lib/libcudnn\w*\.a)")
METADATA_PATHS = {
    f"{DIST_INFO}/{name}"
    for name in ("METADATA", "WHEEL", "RECORD", "top_level.txt", "INSTALLER", "REQUESTED", "licenses/License.txt")
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _recorded_files(root: Path) -> tuple[list[list[str]], set[str]]:
    distributions = list(root.glob("nvidia_cudnn_cu12-*.dist-info"))
    if distributions != [root / DIST_INFO]:
        raise ValueError("Review the exact cuDNN distribution before changing its version")
    with (root / DIST_INFO / "RECORD").open(newline="") as stream:
        rows = list(csv.reader(stream))
    paths = {row[0] for row in rows}
    if len(paths) != len(rows) or LICENSE_PATH not in paths:
        raise ValueError("cuDNN wheel inventory must contain unique paths and its license")
    if _sha256(root / LICENSE_PATH) != LICENSE_SHA256:
        raise ValueError("The installed cuDNN license differs from the reviewed wheel")
    return rows, paths


def _validate_files(root: Path, paths: set[str]) -> tuple[list[str], list[str]]:
    runtime, omitted = [], []
    for name in sorted(paths):
        path = root / name
        if not path.resolve().is_relative_to(root) or path.resolve() != path.absolute():
            raise ValueError("cuDNN file escapes its installed runtime or uses a symlink")
        if not path.is_file():
            raise ValueError("cuDNN wheel record references a missing regular file")
        if RUNTIME_PATH.fullmatch(name):
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("cuDNN shared runtime is not an ELF object")
            runtime.append(name)
        elif DEVELOPMENT_PATH.fullmatch(name):
            omitted.append(name)
        elif name not in METADATA_PATHS:
            raise ValueError("Unreviewed file in the cuDNN wheel")
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "nvidia/cudnn").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != {name for name in paths if name.startswith("nvidia/cudnn/")}:
        raise ValueError("Unrecorded file in the cuDNN namespace")
    if "nvidia/cudnn/lib/libcudnn.so.9" not in runtime:
        raise ValueError("The cuDNN shared runtime is missing")
    return runtime, omitted


def retain_cudnn_runtime(site_packages: Path) -> dict[str, object]:
    """Remove development files after validating the whole installed cuDNN package.

    Args:
        site_packages: The installed OpenPI Python environment's package directory.

    Returns:
        Removed paths and SHA-256 identities of unchanged libraries and license.

    Raises:
        ValueError: Version, license, path, inventory, or shared-library validation fails.
        OSError: The installed package cannot be read or updated.
    """
    root = site_packages.resolve()
    rows, paths = _recorded_files(root)
    runtime, omitted = _validate_files(root, paths)
    for name in omitted:
        (root / name).unlink()
    with (root / DIST_INFO / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(row for row in rows if row[0] not in omitted)
    return {
        "schema_version": "npa.openpi.cudnn-runtime.v1", "version": VERSION,
        "omitted_development_files": omitted,
        "retained_sha256": {name: _sha256(root / name) for name in [*runtime, LICENSE_PATH]},
        "license_url": "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html",
    }


if __name__ == "__main__":
    print(json.dumps(retain_cudnn_runtime(Path(sysconfig.get_paths()["purelib"])), indent=2))
