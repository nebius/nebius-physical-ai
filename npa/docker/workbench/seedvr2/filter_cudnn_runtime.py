"""Keep only the reviewed cuDNN shared-runtime payload before final-image copy."""

from __future__ import annotations

import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sysconfig


def filter_cudnn_runtime(site_packages: Path) -> dict:
    root = site_packages.resolve()
    distributions = [
        dist
        for dist in metadata.distributions(path=[str(root)])
        if re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower() == "nvidia-cudnn-cu13"
    ]
    if len(distributions) != 1:
        raise ValueError("expected exactly one installed cuDNN distribution")
    dist = distributions[0]
    if dist.version != "9.20.0.48":
        raise ValueError(
            "review the cuDNN payload boundary before changing its version"
        )
    dist_info = f"nvidia_cudnn_cu13-{dist.version}.dist-info"
    runtime_pattern = re.compile(r"nvidia/cudnn/lib/libcudnn\w*\.so(?:\.\d+)*")
    sdk_pattern = re.compile(r"nvidia/cudnn/(?:include/cudnn\w*\.h|lib/libcudnn\w*\.a)")
    allowed_metadata = {
        f"{dist_info}/{name}"
        for name in (
            "METADATA",
            "WHEEL",
            "RECORD",
            "top_level.txt",
            "INSTALLER",
            "REQUESTED",
            "licenses/License.txt",
        )
    }
    files = {str(path) for path in (dist.files or [])}
    if not files or f"{dist_info}/licenses/License.txt" not in files:
        raise ValueError("cuDNN wheel inventory and license must be present")
    runtime: list[str] = []
    omitted: list[str] = []
    for name in sorted(files):
        path = root / name
        if (
            path.is_symlink()
            or path.resolve().parent != path.absolute().parent
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("cuDNN payload must be regular and contained")
        if runtime_pattern.fullmatch(name):
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("cuDNN runtime library is not an ELF object")
            runtime.append(name)
        elif sdk_pattern.fullmatch(name):
            omitted.append(name)
        elif name not in allowed_metadata:
            raise ValueError("unreviewed file in cuDNN distribution")
    recorded_package = {name for name in files if name.startswith("nvidia/cudnn/")}
    actual_package = {
        str(path.relative_to(root))
        for path in (root / "nvidia/cudnn").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_package != recorded_package:
        raise ValueError("unrecorded file in cuDNN runtime namespace")
    allowed_unrecorded_metadata = {
        f"{dist_info}/INSTALLER",
        f"{dist_info}/REQUESTED",
        f"{dist_info}/direct_url.json",
    }
    actual_metadata = {
        str(path.relative_to(root))
        for path in (root / dist_info).rglob("*")
        if (path.is_file() or path.is_symlink()) and path.name != "RECORD"
    }
    recorded_metadata = {
        name
        for name in files
        if name.startswith(f"{dist_info}/") and name != f"{dist_info}/RECORD"
    }
    if actual_metadata - recorded_metadata - allowed_unrecorded_metadata:
        raise ValueError("unrecorded file in cuDNN distribution metadata")
    if "nvidia/cudnn/lib/libcudnn.so.9" not in runtime:
        raise ValueError("cuDNN shared runtime is missing")
    for name in omitted:
        (root / name).unlink()
    include_dir = root / "nvidia/cudnn/include"
    if include_dir.exists():
        include_dir.rmdir()
    record_path = root / dist_info / "RECORD"
    with record_path.open(newline="") as stream:
        records = [row for row in csv.reader(stream) if row[0] not in omitted]
    with record_path.open("w", newline="") as stream:
        csv.writer(stream).writerows(records)
    retained = {}
    for name in sorted(set(runtime) | {f"{dist_info}/licenses/License.txt"}):
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        retained[name] = digest.hexdigest()
    return {
        "schema_version": "npa.seedvr2.cudnn-runtime.v1",
        "version": dist.version,
        "omitted_sdk_files": omitted,
        "retained_sha256": retained,
        "license": (
            "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/"
            "reference/eula.html"
        ),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            filter_cudnn_runtime(Path(sysconfig.get_paths()["purelib"])),
            indent=2,
            sort_keys=True,
        )
    )
