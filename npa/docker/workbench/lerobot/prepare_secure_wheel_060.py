"""Build the reviewed LeRobot 0.6.0 W&B security integration from pinned bytes."""

from __future__ import annotations

import argparse
import base64
from contextlib import closing
import csv
import hashlib
import http.client
import io
import json
from pathlib import Path
import tarfile
from urllib.parse import urlsplit
import zipfile


VERSION = "0.6.0+npa.secure4"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/5d/20/"
    "9a96311c19e9d256e65584ca83c49c5782d0f204836e84ceeb420d4d493e/"
    "lerobot-0.6.0-py3-none-any.whl"
)
WHEEL_SHA256 = "b38a564fbc441d98380576863bf68635dde5fc2c42ddc2a39d0486640dc9e9a8"
ARCHIVE_URL = (
    "https://files.pythonhosted.org/packages/1c/18/"
    "b2999bbb52399d404ddcf05645536d3902bd65575ba0ad6bb7bef990a723/"
    "lerobot-0.6.0.tar.gz"
)
ARCHIVE_SHA256 = "6cad660816fdb72570ea7345ecb23bf0f74d36324e2d7c816a260d31a0c29186"
SOURCE_ARCHIVE_ROOT = "lerobot-0.6.0"
METADATA_SHA256 = "fc40aa431b6af49645091369697ab14db3dbfe91d1801bcc8aca0550531945cd"
ORIGINAL_DIST_INFO = "lerobot-0.6.0.dist-info/"
DIST_INFO = f"lerobot-{VERSION}.dist-info/"
DEPENDENCIES = {
    "torch<2.12.0,>=2.7": "torch<2.14.0,>=2.13.0",
    "torchvision<0.27.0,>=0.22.0": "torchvision<0.29.0,>=0.28.0",
    "torchcodec<0.12.0,>=0.3.0": "torchcodec<0.17.0,>=0.16.0",
    "torchcodec<0.12.0,>=0.7.0": "torchcodec<0.17.0,>=0.16.0",
    "torchcodec<0.12.0,>=0.11.0": "torchcodec<0.17.0,>=0.16.0",
    "diffusers<0.36.0,>=0.27.2": "diffusers<0.39.0,>=0.38.0",
    "Pillow<13.0.0,>=10.0.0": "Pillow<14.0.0,>=13.0.0",
    "setuptools<81.0.0,>=71.0.0": "setuptools<85.0.0,>=84.1.0",
    "transformers<5.6.0,>=5.4.0": "transformers<5.11.0,>=5.10.0",
    "wandb<0.28.0,>=0.24.0": "wandb<0.31.0,>=0.30.0",
    "datasets<5.0.0,>=4.7.0": "datasets<5.1.0,>=5.0.1",
}
LOGGER_PATH = "lerobot/common/wandb_utils.py"
LOGGER_SHA256 = "99fe4a82e8fde848ec586f8261a892447a051e7fe8b79022886c965436fb5125"
LOGGER_MODIFICATION_NOTICE = (
    b"# Modified by Nebius: uses the supported W&B Run.url property.\n"
)


def _download(url: str, expected_hash: str) -> bytes:
    if url not in {WHEEL_URL, ARCHIVE_URL}:
        raise RuntimeError("unreviewed upstream download URL")
    parsed = urlsplit(url)
    with closing(
        http.client.HTTPSConnection(parsed.hostname, timeout=60)
    ) as connection:
        connection.request("GET", parsed.path)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("upstream artifact request failed")
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise RuntimeError("upstream artifact hash changed")
    return content


def _source_files(archive: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    source_prefix = f"{SOURCE_ARCHIVE_ROOT}/src/"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for member in source.getmembers():
            if not member.isfile() or not member.name.startswith(source_prefix):
                continue
            path = member.name.removeprefix(source_prefix)
            if not path.startswith("lerobot/"):
                continue
            if path in files:
                raise RuntimeError("duplicate upstream source member")
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError("unreadable upstream source member")
            files[path] = extracted.read()
        license_member = source.getmember(f"{SOURCE_ARCHIVE_ROOT}/LICENSE")
        if not license_member.isfile():
            raise RuntimeError("upstream license is not a regular file")
        extracted_license = source.extractfile(license_member)
        if extracted_license is None:
            raise RuntimeError("unreadable upstream license")
        files["LICENSE"] = extracted_license.read()
    return files


def _verify_wheel(content: bytes, source: dict[str, bytes]) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate upstream wheel member")
        files = {name: wheel.read(name) for name in names if not name.endswith("/")}
    modules = {
        name: value for name, value in files.items() if name.startswith("lerobot/")
    }
    if not modules or any(source.get(name) != value for name, value in modules.items()):
        raise RuntimeError("wheel package differs from the pinned upstream source")
    license_file = files.get(ORIGINAL_DIST_INFO + "licenses/LICENSE")
    metadata = files.get(ORIGINAL_DIST_INFO + "METADATA")
    if license_file is None or metadata is None:
        raise RuntimeError("unreviewed upstream wheel layout")
    if license_file != source["LICENSE"]:
        raise RuntimeError("upstream license differs between source and wheel")
    if hashlib.sha256(metadata).hexdigest() != METADATA_SHA256:
        raise RuntimeError("unreviewed upstream dependency metadata")
    return files


def _metadata(original: bytes) -> bytes:
    lines = original.decode().splitlines(keepends=True)
    if lines.count("Version: 0.6.0\n") != 1:
        raise RuntimeError("unexpected upstream package version")
    changes = {dependency: 0 for dependency in DEPENDENCIES}
    result = []
    for line in lines:
        if line == "Version: 0.6.0\n":
            line = f"Version: {VERSION}\n"
        for dependency, replacement in DEPENDENCIES.items():
            prefix = f"Requires-Dist: {dependency}"
            if line == prefix + "\n" or line.startswith(prefix + ";"):
                line = line.replace(prefix, "Requires-Dist: " + replacement, 1)
                changes[dependency] += 1
        result.append(line)
    if any(count != 1 for count in changes.values()):
        raise RuntimeError("upstream reviewed dependency edges changed")
    return "".join(result).encode()


def _logger_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != LOGGER_SHA256:
        raise RuntimeError("unreviewed upstream logger source")
    if original.count(b"wandb.run.get_url()") != 1:
        raise RuntimeError("reviewed upstream logger API changed")
    return LOGGER_MODIFICATION_NOTICE + original.replace(
        b"wandb.run.get_url()", b"wandb.run.url"
    )


def _native_model_source_unchanged(source_patches: dict[str, dict]) -> bool:
    """Derive the policy-source claim from the exact patched paths."""

    return not any(path.startswith("lerobot/policies/") for path in source_patches)


def _record(files: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for path, value in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode()
        writer.writerow((path, "sha256=" + digest.rstrip("="), len(value)))
    writer.writerow((DIST_INFO + "RECORD", "", ""))
    return output.getvalue().encode()


def _write_wheel(files: dict[str, bytes], original: bytes, output: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(original)) as upstream:
        permissions = {
            item.filename: item.external_attr for item in upstream.infolist()
        }
    with zipfile.ZipFile(output, "w") as built:
        for name, content in sorted(files.items()):
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.compress_type = zipfile.ZIP_DEFLATED
            original_name = name.replace(DIST_INFO, ORIGINAL_DIST_INFO, 1)
            item.external_attr = permissions.get(original_name, 0o100644 << 16)
            built.writestr(item, content)


def _prepare(wheel: bytes, archive: bytes, destination: Path) -> dict[str, object]:
    if hashlib.sha256(wheel).hexdigest() != WHEEL_SHA256:
        raise RuntimeError("unreviewed upstream wheel")
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("unreviewed upstream source archive")
    original = _verify_wheel(wheel, _source_files(archive))
    result = {
        name.replace(ORIGINAL_DIST_INFO, DIST_INFO, 1): value
        for name, value in original.items()
        if name != ORIGINAL_DIST_INFO + "RECORD"
    }
    result[DIST_INFO + "METADATA"] = _metadata(
        original[ORIGINAL_DIST_INFO + "METADATA"]
    )
    result[LOGGER_PATH] = _logger_source(original[LOGGER_PATH])
    source_patches = {
        LOGGER_PATH: {
            "original_sha256": LOGGER_SHA256,
            "patched_sha256": hashlib.sha256(result[LOGGER_PATH]).hexdigest(),
            "change": "added modification notice and replaced W&B Run.get_url() with supported Run.url property",
        }
    }
    receipt: dict[str, object] = {
        "integration_version": VERSION,
        "upstream_source_distribution": "lerobot-0.6.0.tar.gz",
        "upstream_wheel_sha256": WHEEL_SHA256,
        "upstream_source_sha256": ARCHIVE_SHA256,
        "package_source_unchanged": False,
        "native_model_source_unchanged": _native_model_source_unchanged(source_patches),
        "source_patches": source_patches,
        "license_retained": "Apache-2.0",
        "dependency_changes": {
            original: replacement for original, replacement in DEPENDENCIES.items()
        },
        "security_rationale": "candidate-only dependency closure replaces affected upstream ceilings with reviewed hash-locked security floors without changing acceptance of any prior image or permitting moving dependency resolution",
        "capability_qualification": "pending version-specific packaging, scanner, CPU, and GPU gates",
    }
    result[DIST_INFO + "npa-source-integration.json"] = json.dumps(
        receipt, indent=2
    ).encode()
    result[DIST_INFO + "RECORD"] = _record(result)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / f"lerobot-{VERSION}-py3-none-any.whl"
    _write_wheel(result, wheel, output)
    receipt["integration_wheel_sha256"] = hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    (destination / "source-integration.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = _prepare(
        _download(WHEEL_URL, WHEEL_SHA256),
        _download(ARCHIVE_URL, ARCHIVE_SHA256),
        args.output_dir,
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
