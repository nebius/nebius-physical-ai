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
VERSION = "0.5.1+npa.secure2"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/60/8e/"
    "8e3763ad36ba0efb9ba0381ae06d328b54081587b45b0c1d049b26ed8d8c/"
    "lerobot-0.5.1-py3-none-any.whl"
)
WHEEL_SHA256 = "bbd11021023fde0947b6d1ff1c52fe91c86a28ab09a96359892f3ef7e8866862"
ARCHIVE_URL = (
    "https://files.pythonhosted.org/packages/3b/46/"
    "21c9531f77fc2c1a1fe3a4ea3e5b1633d9503738b85cc391f28ee4369f2a/"
    "lerobot-0.5.1.tar.gz"
)
ARCHIVE_SHA256 = "4a05a9d10f013bf8fc408c7bd9c32a261209250cb66987b796416911c96a6fd3"
SOURCE_ARCHIVE_ROOT = "lerobot-0.5.1"
METADATA_SHA256 = "88a5ac51b516cfdd66d12818b74325bb18f515679a522591faf205a1205275ed"
DEPENDENCIES = {
    "torch<2.11.0,>=2.7": "torch<2.14.0,>=2.13.0",
    "torchvision<0.26.0,>=0.22.0": "torchvision<0.29.0,>=0.28.0",
    "torchcodec<0.11.0,>=0.3.0": "torchcodec<0.17.0,>=0.16.0",
    "diffusers<0.36.0,>=0.27.2": "diffusers<0.39.0,>=0.38.0",
    "setuptools<81.0.0,>=71.0.0": "setuptools<84.0.0,>=83.0.0",
    "datasets<5.0.0,>=4.0.0": "datasets<5.1.0,>=5.0.1",
    "transformers==5.3.0": "transformers<5.11.0,>=5.10.0",
    "wandb<0.25.0,>=0.24.0": "wandb<0.31.0,>=0.30.0",
}
ORIGINAL_DIST_INFO = "lerobot-0.5.1.dist-info/"
DIST_INFO = f"lerobot-{VERSION}.dist-info/"
LOGGER_PATH = "lerobot/rl/wandb_utils.py"
LOGGER_SHA256 = "9328a2254e840e1447cb2464fa0498102fe640ec8d68edfefae4592ada22591b"
LOGGER_MODIFICATION_NOTICE = (
    b"# Modified by Nebius: uses the supported W&B Run.url property.\n"
)
GROOT_INIT_PATH = "lerobot/policies/groot/__init__.py"
GROOT_INIT_SHA256 = "40411dc2b519d34f96cd68bed78817d98f10905778eea58a18a0d539ce2cd79d"
GROOT_EAGER_IMPORTS = (
    b"from .configuration_groot import GrootConfig\n"
    b"from .modeling_groot import GrootPolicy\n"
    b"from .processor_groot import make_groot_pre_post_processors\n"
)
GROOT_LAZY_IMPORTS = (
    b"# Modified by Nebius: defer optional GR00T runtime imports so they cannot "
    b"block core policy initialization.\n"
    b"from .configuration_groot import GrootConfig\n\n\n"
    b"def __getattr__(name: str):\n"
    b'    if name == "GrootPolicy":\n'
    b"        from .modeling_groot import GrootPolicy\n\n"
    b"        return GrootPolicy\n"
    b'    if name == "make_groot_pre_post_processors":\n'
    b"        from .processor_groot import make_groot_pre_post_processors\n\n"
    b"        return make_groot_pre_post_processors\n"
    b'    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n'
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
    files = {}
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
            files[path] = source.extractfile(member).read()
        license_member = source.getmember(f"{SOURCE_ARCHIVE_ROOT}/LICENSE")
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
    return LOGGER_MODIFICATION_NOTICE + original.replace(
        b"wandb.run.get_url()", b"wandb.run.url"
    )


def _groot_init_source(original: bytes) -> bytes:
    """Defer optional GR00T imports without changing the model implementation."""

    if hashlib.sha256(original).hexdigest() != GROOT_INIT_SHA256:
        raise RuntimeError("unreviewed upstream GR00T package initializer")
    if original.count(GROOT_EAGER_IMPORTS) != 1:
        raise RuntimeError("reviewed upstream GR00T import surface changed")
    patched = original.replace(GROOT_EAGER_IMPORTS, GROOT_LAZY_IMPORTS)
    _validate_groot_import_surface(patched)
    return patched


def _validate_groot_import_surface(source: bytes) -> None:
    """Reject top-level optional model imports in the exact patched initializer."""

    eager_imports = (
        b"from .modeling_groot import GrootPolicy",
        b"from .processor_groot import make_groot_pre_post_processors",
    )
    if any(line in eager_imports for line in source.splitlines()):
        raise RuntimeError("eager optional GR00T model import remains")
    if GROOT_LAZY_IMPORTS not in source:
        raise RuntimeError("reviewed GR00T lazy import surface changed")


def _native_model_source_unchanged(source_patches: dict[str, dict]) -> bool:
    """Derive the model-implementation claim from the exact patched paths."""

    return not any(
        path.startswith("lerobot/policies/") and path != GROOT_INIT_PATH
        for path in source_patches
    )


def _integration(files: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    result = {
        name.replace(ORIGINAL_DIST_INFO, DIST_INFO, 1): value
        for name, value in files.items()
        if name != ORIGINAL_DIST_INFO + "RECORD"
    }
    result[DIST_INFO + "METADATA"] = _metadata(files[ORIGINAL_DIST_INFO + "METADATA"])
    result[LOGGER_PATH] = _logger_source(files[LOGGER_PATH])
    result[GROOT_INIT_PATH] = _groot_init_source(files[GROOT_INIT_PATH])
    source_patches = {
        LOGGER_PATH: {
            "original_sha256": LOGGER_SHA256,
            "patched_sha256": hashlib.sha256(result[LOGGER_PATH]).hexdigest(),
            "change": "added modification notice and replaced W&B Run.get_url() with supported Run.url property",
        },
        GROOT_INIT_PATH: {
            "original_sha256": GROOT_INIT_SHA256,
            "patched_sha256": hashlib.sha256(result[GROOT_INIT_PATH]).hexdigest(),
            "change": "deferred optional GR00T runtime imports so core policy imports do not execute the optional model path",
        },
    }
    receipt = {
        "integration_version": VERSION,
        "declared_upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_distribution": "lerobot-0.5.1.tar.gz",
        "upstream_wheel_sha256": WHEEL_SHA256,
        "upstream_source_sha256": ARCHIVE_SHA256,
        "package_source_unchanged": False,
        "native_model_source_unchanged": _native_model_source_unchanged(source_patches),
        "source_patches": source_patches,
        "license_retained": "Apache-2.0",
        "dependency_changes": DEPENDENCIES,
        "capability_qualification": "pending native ACT, Diffusion, default decoder, server and offline logger gates",
        "deferred_optional_policy_qualification": {
            "groot": "unqualified under the security-pinned transformers runtime; a request to instantiate GrootPolicy requires a separate runtime qualification"
        },
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
