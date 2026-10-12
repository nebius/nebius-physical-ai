#!/usr/bin/env python3
"""Remove NVSHMEM SDK payload while retaining reviewed shared runtimes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

OFFICIAL_LICENSE_REVISION = "v3.4.5-0"
OFFICIAL_LICENSE_SOURCE = (
    "https://raw.githubusercontent.com/NVIDIA/nvshmem/v3.4.5-0/License.txt"
)
OFFICIAL_LICENSE_SHA256 = (
    "1f5b7ada702926bc73327e6eb02dc2d41facc844cc4512ac900451bda06a459e"
)


def _distribution(site_packages: Path) -> importlib.metadata.Distribution:
    matches = [
        dist
        for dist in importlib.metadata.distributions(path=[str(site_packages)])
        if dist.metadata.get("Name", "").lower() == "nvidia-nvshmem-cu13"
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one nvidia-nvshmem-cu13 distribution")
    return matches[0]


def filter_nvshmem_runtime(site_packages: Path, official_license: Path) -> dict:
    root = site_packages.resolve(strict=True)
    official_license = official_license.resolve(strict=True)
    official_bytes = official_license.read_bytes()
    if (
        hashlib.sha256(official_bytes).hexdigest() != OFFICIAL_LICENSE_SHA256
        or b"Portions derived from DF-NVSHMEM-prototype" not in official_bytes
        or b"Portions derived from Sandia OpenSHMEM" not in official_bytes
    ):
        raise ValueError("official NVSHMEM product license differs from v3.4.5-0")
    dist = _distribution(root)
    dist_info = Path(dist._path).relative_to(root).as_posix()
    files = [Path(item).as_posix() for item in (dist.files or ())]
    if len(files) != len(set(files)) or not files:
        raise ValueError("NVSHMEM wheel inventory is empty or duplicated")
    runtime_pattern = re.compile(
        r"nvidia/nvshmem/lib/(?:libnvshmem_host|"
        r"nvshmem_(?:bootstrap|transport)_[^/]+)\.so\.\d+"
    )
    sdk_pattern = re.compile(r"nvidia/nvshmem/(?:include/.+|lib/[^/]+\.(?:a|bc))")
    allowed_metadata = {name for name in files if name.startswith(f"{dist_info}/")}
    license_path = f"{dist_info}/licenses/License.txt"
    if f"{dist_info}/RECORD" not in files or license_path not in files:
        raise ValueError("NVSHMEM wheel inventory and license must be present")
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
            raise ValueError("NVSHMEM payload must be regular and contained")
        if runtime_pattern.fullmatch(name):
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("NVSHMEM runtime library is not an ELF object")
            runtime.append(name)
        elif sdk_pattern.fullmatch(name):
            omitted.append(name)
        elif name not in allowed_metadata:
            raise ValueError("unreviewed file in NVSHMEM distribution")
    recorded_package = {name for name in files if name.startswith("nvidia/nvshmem/")}
    actual_package = {
        str(path.relative_to(root))
        for path in (root / "nvidia/nvshmem").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_package != recorded_package:
        raise ValueError("unrecorded file in NVSHMEM runtime namespace")
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
        raise ValueError("unrecorded file in NVSHMEM distribution metadata")
    if "nvidia/nvshmem/lib/libnvshmem_host.so.3" not in runtime:
        raise ValueError("NVSHMEM shared runtime is missing")
    for name in omitted:
        (root / name).unlink()
    include_dir = root / "nvidia/nvshmem/include"
    if include_dir.exists():
        for directory in sorted(include_dir.rglob("*"), reverse=True):
            if directory.is_dir():
                directory.rmdir()
        include_dir.rmdir()
    record_path = root / dist_info / "RECORD"
    with record_path.open(newline="") as stream:
        records = [row for row in csv.reader(stream) if row[0] not in omitted]
    with record_path.open("w", newline="") as stream:
        csv.writer(stream).writerows(records)
    retained = {}
    for name in sorted(set(runtime) | {license_path}):
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        retained[name] = digest.hexdigest()
    return {
        "schema_version": "npa.seedvr2.nvshmem-runtime.v1",
        "version": dist.version,
        "omitted_sdk_files": omitted,
        "retained_sha256": retained,
        "license": "LicenseRef-NVIDIA-Proprietary",
        "official_product_license": {
            "revision": OFFICIAL_LICENSE_REVISION,
            "source": OFFICIAL_LICENSE_SOURCE,
            "sha256": OFFICIAL_LICENSE_SHA256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/opt/seedvr2-venv/lib/python3.12/site-packages"),
    )
    parser.add_argument(
        "--official-license",
        type=Path,
        default=Path("/opt/npa-legal/NVSHMEM-License-v3.4.5-0.txt"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            filter_nvshmem_runtime(args.site_packages, args.official_license),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
