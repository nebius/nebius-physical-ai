"""Build an explicitly versioned NPA integration from verified upstream source."""

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


UPSTREAM_COMMIT = "1396b9fab7aecddd10006c33c47a487ffdcb54b4"
VERSION = "0.5.1+npa.secure1"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/60/8e/"
    "8e3763ad36ba0efb9ba0381ae06d328b54081587b45b0c1d049b26ed8d8c/"
    "lerobot-0.5.1-py3-none-any.whl"
)
WHEEL_SHA256 = "bbd11021023fde0947b6d1ff1c52fe91c86a28ab09a96359892f3ef7e8866862"
ARCHIVE_URL = (
    f"https://codeload.github.com/huggingface/lerobot/tar.gz/{UPSTREAM_COMMIT}"
)
ARCHIVE_SHA256 = "fe0b3079a517f0c60bdf325afc7832d81ae384bd1d022f4e802e7da8f73606dd"
METADATA_SHA256 = "88a5ac51b516cfdd66d12818b74325bb18f515679a522591faf205a1205275ed"
DEPENDENCIES = {
    "torch<2.11.0,>=2.7": "torch<2.14.0,>=2.13.0",
    "torchvision<0.26.0,>=0.22.0": "torchvision<0.29.0,>=0.28.0",
    "torchcodec<0.11.0,>=0.3.0": "torchcodec<0.17.0,>=0.16.0",
    "diffusers<0.36.0,>=0.27.2": "diffusers<0.39.0,>=0.38.0",
    "wandb<0.25.0,>=0.24.0": "wandb<0.31.0,>=0.30.0",
}
ORIGINAL_DIST_INFO = "lerobot-0.5.1.dist-info/"
DIST_INFO = f"lerobot-{VERSION}.dist-info/"
LOGGER_PATH = "lerobot/rl/wandb_utils.py"
LOGGER_SHA256 = "9328a2254e840e1447cb2464fa0498102fe640ec8d68edfefae4592ada22591b"


def _download(url: str, expected_hash: str) -> bytes:
    if url not in {WHEEL_URL, ARCHIVE_URL}:
        raise RuntimeError("unreviewed upstream download URL")
    parsed = urlsplit(url)
    with closing(http.client.HTTPSConnection(parsed.hostname)) as connection:
        connection.request("GET", parsed.path)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("upstream artifact request failed")
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise RuntimeError("upstream artifact hash changed")
    return content


def _source_files(archive: bytes) -> dict[str, bytes]:
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for member in source.getmembers():
            if not member.isfile() or "/src/lerobot/" not in member.name:
                continue
            path = member.name.split("/src/", 1)[1]
            if path in files:
                raise RuntimeError("duplicate upstream source member")
            files[path] = source.extractfile(member).read()
        license_member = source.getmember(f"lerobot-{UPSTREAM_COMMIT}/LICENSE")
        if not license_member.isfile():
            raise RuntimeError("upstream license is not a regular file")
        files["LICENSE"] = source.extractfile(license_member).read()
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
        raise RuntimeError("wheel package differs from the pinned upstream commit")
    if files[ORIGINAL_DIST_INFO + "licenses/LICENSE"] != source["LICENSE"]:
        raise RuntimeError("upstream license differs between source and wheel")
    if (
        hashlib.sha256(files[ORIGINAL_DIST_INFO + "METADATA"]).hexdigest()
        != METADATA_SHA256
    ):
        raise RuntimeError("unreviewed upstream dependency metadata")
    return files


def _metadata(original: bytes) -> bytes:
    lines = original.decode().splitlines(keepends=True)
    if lines.count("Version: 0.5.1\n") != 1:
        raise RuntimeError("unexpected upstream package version")
    changes = {original: 0 for original in DEPENDENCIES}
    result = []
    for line in lines:
        if line == "Version: 0.5.1\n":
            line = f"Version: {VERSION}\n"
        for original, replacement in DEPENDENCIES.items():
            prefix = f"Requires-Dist: {original}"
            if line == prefix + "\n" or line.startswith(prefix + ";"):
                line = line.replace(prefix, "Requires-Dist: " + replacement, 1)
                changes[original] += 1
        result.append(line)
    if any(count != 1 for count in changes.values()):
        raise RuntimeError("upstream reviewed dependency edges changed")
    return "".join(result).encode()


def _record(files: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for path, value in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(value).digest())
            .decode()
            .rstrip("=")
        )
        writer.writerow((path, "sha256=" + digest, len(value)))
    writer.writerow((DIST_INFO + "RECORD", "", ""))
    return output.getvalue().encode()


def _logger_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != LOGGER_SHA256:
        raise RuntimeError("unreviewed upstream logger source")
    if original.count(b"wandb.run.get_url()") != 1:
        raise RuntimeError("reviewed upstream logger API changed")
    return original.replace(b"wandb.run.get_url()", b"wandb.run.url")


def _integration(files: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    result = {
        name.replace(ORIGINAL_DIST_INFO, DIST_INFO, 1): value
        for name, value in files.items()
        if name != ORIGINAL_DIST_INFO + "RECORD"
    }
    result[DIST_INFO + "METADATA"] = _metadata(files[ORIGINAL_DIST_INFO + "METADATA"])
    result[LOGGER_PATH] = _logger_source(files[LOGGER_PATH])
    receipt = {
        "integration_version": VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_wheel_sha256": WHEEL_SHA256,
        "upstream_source_sha256": ARCHIVE_SHA256,
        "package_source_unchanged": False,
        "native_model_source_unchanged": True,
        "source_patches": {
            LOGGER_PATH: {
                "original_sha256": LOGGER_SHA256,
                "patched_sha256": hashlib.sha256(result[LOGGER_PATH]).hexdigest(),
                "change": "removed W&B Run.get_url() to supported Run.url property",
            }
        },
        "license_retained": "Apache-2.0",
        "dependency_changes": DEPENDENCIES,
        "capability_qualification": "pending native ACT, Diffusion, default decoder, server and offline logger gates",
    }
    result[DIST_INFO + "npa-source-integration.json"] = json.dumps(
        receipt, indent=2
    ).encode()
    result[DIST_INFO + "RECORD"] = _record(result)
    return result, receipt


def _prepare(wheel: bytes, archive: bytes, destination: Path) -> dict:
    if hashlib.sha256(wheel).hexdigest() != WHEEL_SHA256:
        raise RuntimeError("unreviewed upstream wheel")
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("unreviewed upstream source archive")
    files, receipt = _integration(_verify_wheel(wheel, _source_files(archive)))
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / f"lerobot-{VERSION}-py3-none-any.whl"
    _write_wheel(files, wheel, output)
    receipt["integration_wheel_sha256"] = hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    (destination / "source-integration.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


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
