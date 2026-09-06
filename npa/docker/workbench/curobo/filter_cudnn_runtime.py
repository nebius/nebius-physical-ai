"""Keep only the cuDNN runtime redistribution boundary before a layer commits.

Run in the same Docker RUN as the hash-locked pip install. Unknown wheel files
fail closed; retained runtime bytes and license notices remain unchanged.
"""

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
        raise ValueError("Expected exactly one installed cuDNN distribution")
    dist = distributions[0]
    if dist.version != "9.13.0.50":
        raise ValueError("Review the cuDNN payload boundary before changing its version")
    dist_info = f"nvidia_cudnn_cu13-{dist.version}.dist-info"
    runtime_pattern = re.compile(r"nvidia/cudnn/lib/libcudnn\w*\.so(?:\.\d+)*")
    omitted_pattern = re.compile(
        r"nvidia/cudnn/(?:include/cudnn\w*\.h|lib/libcudnn\w*\.a)"
    )
    allowed_metadata = {
        f"{dist_info}/{name}"
        for name in (
            "METADATA", "WHEEL", "RECORD", "top_level.txt", "INSTALLER",
            "REQUESTED", "licenses/License.txt",
        )
    }
    files = {str(path) for path in (dist.files or [])}
    if not files or f"{dist_info}/licenses/License.txt" not in files:
        raise ValueError("cuDNN wheel inventory and license must be present")
    runtime = []
    omitted = []
    for name in sorted(files):
        path = root / name
        if path.is_symlink() or path.resolve().parent != path.absolute().parent or not path.is_file():
            raise ValueError("cuDNN payload must consist of regular contained files")
        if not path.resolve().is_relative_to(root):
            raise ValueError("cuDNN payload escapes site-packages")
        if runtime_pattern.fullmatch(name):
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("cuDNN runtime library is not an ELF object")
            runtime.append(name)
        elif omitted_pattern.fullmatch(name):
            omitted.append(name)
        elif name not in allowed_metadata:
            raise ValueError("Unreviewed file in cuDNN distribution")
    # Also reject bytes under the vendor namespace omitted from wheel RECORD.
    recorded_package = {name for name in files if name.startswith("nvidia/cudnn/")}
    actual_package = {
        str(path.relative_to(root))
        for path in (root / "nvidia/cudnn").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_package != recorded_package:
        raise ValueError("Unrecorded file in cuDNN runtime namespace")
    if "nvidia/cudnn/lib/libcudnn.so.9" not in runtime:
        raise ValueError("cuDNN shared runtime is missing")
    # Validate everything before the first deletion. This script only removes
    # the inspected SDK headers/static files, never other dependencies/notices.
    for name in omitted:
        (root / name).unlink()
    record_path = root / dist_info / "RECORD"
    with record_path.open(newline="") as stream:
        records = [row for row in csv.reader(stream) if row[0] not in omitted]
    with record_path.open("w", newline="") as stream:
        csv.writer(stream).writerows(records)
    retained = {}
    for name in sorted(set(runtime) | {f"{dist_info}/licenses/License.txt"}):
        with (root / name).open("rb") as stream:
            retained[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "schema_version": "npa.curobo.cudnn-runtime.v1",
        "version": dist.version,
        "omitted_sdk_files": omitted,
        "retained_sha256": retained,
        "license": "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html",
    }


if __name__ == "__main__":
    print(json.dumps(filter_cudnn_runtime(Path(sysconfig.get_paths()["purelib"])), indent=2))
