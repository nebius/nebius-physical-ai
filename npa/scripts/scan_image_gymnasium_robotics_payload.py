#!/usr/bin/env python3
"""Inspect every config/history/layer/rootfs byte of a Docker-save archive.

The scanner is product-specific defense in depth. It cannot authorize release:
Phase A has no accepted manifest or native-byte policy, and incomplete locks are
always fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any
import zipfile

REQUIRED = {
    "opt/npa/gymnasium-robotics/source-lock.json",
    "opt/npa/gymnasium-robotics/apt-runtime.lock.json",
    "opt/npa/gymnasium-robotics/corresponding-source.lock.json",
    "opt/npa/gymnasium-robotics/asset-lock.json",
    "opt/npa/gymnasium-robotics/requirements.lock",
    "opt/npa/gymnasium-robotics/capability_smoke.py",
    "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md",
    "usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md",
    "usr/share/source/npa-gymnasium-robotics/final-rootfs-manifest.json",
}
FORBIDDEN_PATH = re.compile(
    r"(^|/)(\.git|\.cache|pip-cache|apt/lists|apt/archives|\.aws|\.docker|\.ssh)(/|$)|"
    r"(^|/)(workspace/byof-runs|root/\.cache)(/|$)|"
    r"(^|/)(usr/local/cuda|opt/nvidia)(/|$)|"
    r"(^|/)(libnvidia[^/]*|libcuda[^/]*)$|"
    r"\.(pt|pth|ckpt|safetensors|onnx|engine)$",
    re.IGNORECASE,
)
SECRET_TEXT = re.compile(
    rb"BEGIN (?:RSA |OPENSSH )?PRIVATE" rb" KEY|"
    rb"(?i:(?:api[_-]?key|secret[_-]?key|password)\s*[=:]\s*[^\s]{8,})"
)
VENDOR_TEXT = re.compile(
    rb"(?i:(?:nvcr\.io|isaacsim|omniverse[/\\]kit|accept_eula\s*[=:]\s*(?:1|yes|true)))"
)
MAX_NESTED_ARCHIVE = 512 * 1024 * 1024
EXPECTED_SOURCE = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO_COMMIT = "13827e9ee56f097f57acf69ae52b078f9839682d"
EXPECTED_SHADOW_COMMIT = "59d6bdf35bd9cf53185a20eb63413fdfe57fe77c"
EXPECTED_ASSET_LOCK = "e22eb62fc690a5e1d1ea931bab950392ca480caf3d51c7f16fd8cb4133d65568"
EXPECTED_BASE = {
    "image": "ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61",
    "config_digest": "sha256:b2b7ea366714195a1e1c5b2b578ece85c0b3920381a8654d038d9684f009613c",
    "layer_digest": "sha256:e51aee9c82ec5dd5ba2add49c45c6d85d460512757e2615b69bcdf9469c7cb58",
}
EXPECTED_SOURCE_FIELDS = {
    "farama_gymnasium_robotics": {
        "archive_sha256": "ad8771ed6e9dd772b1101a25310ea46dd0f6f0044fbd6ea9af01af7b7c52c2c7",
        "commit": EXPECTED_SOURCE,
        "license": "MIT",
        "license_sha256": "00668424e12956742815eb1d8e15c7be543192561511df5fde119ae1188315ef",
        "repository": "https://github.com/Farama-Foundation/Gymnasium-Robotics",
        "version": "1.4.2",
    },
    "mujoco": {
        "commit": EXPECTED_MUJOCO_COMMIT,
        "license": "Apache-2.0",
        "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "repository": "https://github.com/google-deepmind/mujoco",
        "third_party_notices_sha256": "aec5167579b94d6926340175b4f764b5159f4933657556d89bbfb8238d3b3eb8",
        "version": "3.12.0",
        "wheel_sha256": "7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0",
    },
    "shadow_sr_common": {
        "commit": EXPECTED_SHADOW_COMMIT,
        "license": "GPL-2.0-only",
        "license_sha256": "f9c375a1be4a41f7b70301dd83c91cb89e41567478859b77eef375a52d782505",
        "repository": "https://github.com/shadow-robot/sr_common",
    },
}
EXPECTED_PYTHON_DISTRIBUTIONS = {
    "absl-py",
    "boto3",
    "botocore",
    "cloudpickle",
    "etils",
    "farama-notifications",
    "fsspec",
    "glfw",
    "gymnasium",
    "imageio",
    "jinja2",
    "jmespath",
    "markupsafe",
    "mujoco",
    "numpy",
    "packaging",
    "pettingzoo",
    "pillow",
    "pyopengl",
    "python-dateutil",
    "s3transfer",
    "setuptools",
    "six",
    "typing-extensions",
    "urllib3",
    "zipp",
}
ROOTFS_MANIFEST = "usr/share/source/npa-gymnasium-robotics/final-rootfs-manifest.json"
ROOTFS_MANIFEST_EXEMPT = {
    ROOTFS_MANIFEST,
    "opt/npa/gymnasium-robotics/corresponding-source.lock.json",
}
ROOTFS_CLASSES = {
    "apt-runtime",
    "base",
    "corresponding-source",
    "farama-source",
    "license-notice",
    "npa-runtime",
    "python-runtime",
}


def _safe(name: str) -> str:
    original = PurePosixPath(name)
    if original.is_absolute() or ".." in original.parts:
        raise ValueError(f"unsafe archive path: {name}")
    normalized = name[2:] if name.startswith("./") else name
    path = PurePosixPath(normalized)
    if not path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return str(path)


def _json_member(archive: tarfile.TarFile, name: str) -> Any:
    member = archive.getmember(name)
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"missing archive member: {name}")
    return json.loads(stream.read())


def _nested_archive_members(path: str, content: bytes) -> int:
    """Inspect every named member of retained tar/zip Python/source archives."""

    lowered = path.lower()
    if lowered.endswith((".whl", ".zip")):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
                for name in names:
                    safe = _safe(name)
                    if FORBIDDEN_PATH.search(safe) or SECRET_TEXT.search(
                        archive.read(name)
                    ):
                        raise ValueError(
                            f"forbidden nested archive member: {path}:{safe}"
                        )
                return len(names)
        except zipfile.BadZipFile as error:
            raise ValueError(f"unreadable nested zip archive: {path}") from error
    if lowered.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
                count = 0
                for member in archive:
                    safe = _safe(member.name)
                    count += 1
                    if FORBIDDEN_PATH.search(safe):
                        raise ValueError(
                            f"forbidden nested archive member: {path}:{safe}"
                        )
                    if member.isfile():
                        stream = archive.extractfile(member)
                        if stream is None or SECRET_TEXT.search(stream.read()):
                            raise ValueError(
                                f"forbidden nested archive bytes: {path}:{safe}"
                            )
                return count
        except tarfile.TarError as error:
            raise ValueError(f"unreadable nested tar archive: {path}") from error
    return 0


def _layers(
    archive: tarfile.TarFile, names: list[str]
) -> tuple[dict[str, bytes], int, int]:
    rootfs: dict[str, bytes] = {}
    total = 0
    nested = 0
    for layer_name in names:
        member = archive.getmember(layer_name)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"unreadable layer: {layer_name}")
        raw = stream.read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as layer:
            for item in layer:
                path = _safe(item.name)
                total += 1
                leaf = PurePosixPath(path).name
                if leaf == ".wh..wh..opq":
                    parent = str(PurePosixPath(path).parent)
                    rootfs = {
                        key: value
                        for key, value in rootfs.items()
                        if not key.startswith(parent + "/")
                    }
                    continue
                if leaf.startswith(".wh."):
                    target = str(
                        PurePosixPath(path).with_name(leaf.removeprefix(".wh."))
                    )
                    rootfs = {
                        key: value
                        for key, value in rootfs.items()
                        if key != target and not key.startswith(target + "/")
                    }
                    continue
                if FORBIDDEN_PATH.search(path):
                    raise ValueError(f"forbidden image path: {path}")
                if item.isfile():
                    payload = layer.extractfile(item)
                    if payload is None:
                        raise ValueError(f"unreadable layer file: {path}")
                    content = payload.read()
                    if SECRET_TEXT.search(content):
                        raise ValueError(f"forbidden secret signature: {path}")
                    is_notice = path.startswith(
                        "usr/share/doc/npa-gymnasium-robotics/"
                    )
                    if not is_notice and VENDOR_TEXT.search(content):
                        raise ValueError(
                            f"forbidden vendor payload signature: {path}"
                        )
                    if path.lower().endswith(
                        (".whl", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz")
                    ):
                        if len(content) > MAX_NESTED_ARCHIVE:
                            raise ValueError(
                                f"nested archive exceeds scan bound: {path}"
                            )
                        nested += _nested_archive_members(path, content)
                    rootfs[path] = content
    return rootfs, total, nested


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not value or any(_missing(item) for item in value)
    if isinstance(value, dict):
        return not value or any(_missing(item) for item in value.values())
    return False


def _normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_python_distributions(raw: bytes) -> set[str]:
    text = raw.decode("utf-8")
    if "# status: complete" not in text:
        raise ValueError("Python lock is incomplete")
    logical: list[str] = []
    pending = ""
    for source_line in text.splitlines():
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        raise ValueError("Python lock ends with an incomplete continuation")
    distributions: set[str] = set()
    pattern = re.compile(
        r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==[^\s]+"
        r"(?:\s+--hash=sha256:[0-9a-f]{64})+"
    )
    for requirement in logical:
        match = pattern.fullmatch(requirement)
        if match is None:
            raise ValueError(f"unhashed or malformed Python lock entry: {requirement}")
        name = _normalize_distribution(match.group("name"))
        if name in distributions:
            raise ValueError(f"duplicate Python lock distribution: {name}")
        distributions.add(name)
    return distributions


def _classified_rootfs(rootfs: dict[str, bytes], expected_digest: str) -> None:
    raw = rootfs[ROOTFS_MANIFEST]
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("final rootfs classification manifest changed")
    manifest = json.loads(raw)
    if manifest.get("schema") != "npa.gymnasium-robotics.rootfs-manifest.v1":
        raise ValueError("final rootfs classification manifest schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("final rootfs classification manifest has no file map")
    expected_paths = set(rootfs) - ROOTFS_MANIFEST_EXEMPT
    if set(files) != expected_paths:
        raise ValueError("final rootfs contains missing or unclassified regular files")
    for path, record in files.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid rootfs classification: {path}")
        if record.get("class") not in ROOTFS_CLASSES or not record.get("source"):
            raise ValueError(f"invalid rootfs classification: {path}")
        if hashlib.sha256(rootfs[path]).hexdigest() != record.get("sha256"):
            raise ValueError(f"classified rootfs file hash changed: {path}")


def _locked_assets(rootfs: dict[str, bytes]) -> None:
    raw_lock = rootfs["opt/npa/gymnasium-robotics/asset-lock.json"]
    if hashlib.sha256(raw_lock).hexdigest() != EXPECTED_ASSET_LOCK:
        raise ValueError("approved asset lock bytes changed")
    lock = json.loads(raw_lock)
    if lock.get("source_commit") != EXPECTED_SOURCE:
        raise ValueError("asset lock source commit changed")
    groups = (lock.get("directly_loaded_xml"), lock.get("directly_loaded_mesh_texture"))
    if not all(isinstance(group, dict) for group in groups):
        raise ValueError("asset lock groups are missing")
    if len(groups[0]) != 5 or len(groups[1]) != 14:
        raise ValueError(
            "asset lock is not the exact 5 XML plus 14 mesh/texture closure"
        )
    prefix = "opt/venv/lib/python3.12/site-packages/gymnasium_robotics/envs/assets/"
    for group in groups:
        for name, digest in group.items():
            content = rootfs.get(prefix + name)
            if content is None or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"installed upstream asset hash changed: {name}")
    notice = rootfs.get(prefix + "LICENSE.md")
    if notice is None or hashlib.sha256(notice).hexdigest() != lock.get(
        "asset_notice_sha256"
    ):
        raise ValueError("installed Shadow Hand asset notice changed")


def _complete_locks(rootfs: dict[str, bytes]) -> None:
    parsed: dict[str, dict[str, Any]] = {}
    for name in (
        "source-lock.json",
        "apt-runtime.lock.json",
        "corresponding-source.lock.json",
    ):
        path = f"opt/npa/gymnasium-robotics/{name}"
        payload = json.loads(rootfs[path])
        parsed[name] = payload
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete evidence lock in image: {name}")
        if _missing(payload):
            raise ValueError(f"complete evidence lock contains missing values: {name}")
    components = parsed["source-lock.json"].get("components", {})
    if set(components) != set(EXPECTED_SOURCE_FIELDS):
        raise ValueError("source lock component closure changed")
    for name, expected in EXPECTED_SOURCE_FIELDS.items():
        if any(components[name].get(key) != value for key, value in expected.items()):
            raise ValueError(f"source lock identity changed: {name}")
    apt = parsed["apt-runtime.lock.json"]
    if apt.get("base") != EXPECTED_BASE:
        raise ValueError("Ubuntu base identity changed")
    if not apt.get("resolved_binary_packages") or not apt.get(
        "resolved_source_packages"
    ):
        raise ValueError("APT binary/source closure is empty")
    requirements = rootfs["opt/npa/gymnasium-robotics/requirements.lock"]
    if _locked_python_distributions(requirements) != EXPECTED_PYTHON_DISTRIBUTIONS:
        raise ValueError("Python distribution closure changed")
    annex = "usr/share/source/npa-gymnasium-robotics/"
    if not any(path.startswith(annex) for path in rootfs):
        raise ValueError("corresponding-source annex is empty")
    source_lock = parsed["source-lock.json"]
    shadow = components["shadow_sr_common"]
    for field in ("preferred_form_sha256", "transformation_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(shadow.get(field) or "")):
            raise ValueError(f"Shadow preferred-form closure is incomplete: {field}")
    deliveries = parsed["corresponding-source.lock.json"].get("deliveries", [])
    expected_deliveries = {
        "farama-gymnasium-robotics",
        "shadow-hand-xml-mesh-texture-assets",
        "ubuntu-runtime-closure",
    }
    if {delivery.get("binary_component") for delivery in deliveries} != expected_deliveries:
        raise ValueError("corresponding-source delivery closure changed")
    delivery_map = {delivery["binary_component"]: delivery for delivery in deliveries}
    farama_delivery = delivery_map["farama-gymnasium-robotics"]
    shadow_delivery = delivery_map["shadow-hand-xml-mesh-texture-assets"]
    ubuntu_delivery = delivery_map["ubuntu-runtime-closure"]
    if farama_delivery.get("archive_sha256") != source_lock["components"][
        "farama_gymnasium_robotics"
    ]["archive_sha256"]:
        raise ValueError("conveyed Farama source archive identity changed")
    shadow_fields = {
        "build_instructions_sha256": None,
        "preferred_form_archive_sha256": shadow["preferred_form_sha256"],
        "source_commit": EXPECTED_SHADOW_COMMIT,
        "transformation_manifest_sha256": shadow["transformation_manifest_sha256"],
    }
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(shadow_delivery.get(field) or ""))
        if expected is None
        else shadow_delivery.get(field) != expected
        for field, expected in shadow_fields.items()
    ):
        raise ValueError("Shadow preferred-form delivery mapping is incomplete")
    for field in (
        "binary_manifest_sha256",
        "build_materials_sha256",
        "source_manifest_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(ubuntu_delivery.get(field) or "")):
            raise ValueError(f"Ubuntu corresponding-source closure is incomplete: {field}")
    for delivery in deliveries:
        artifacts = delivery.get("artifacts")
        if not artifacts:
            raise ValueError("corresponding-source delivery has no exact artifacts")
        for artifact in artifacts:
            content = rootfs.get(annex + str(artifact.get("path") or ""))
            if content is None or hashlib.sha256(content).hexdigest() != artifact.get(
                "sha256"
            ):
                raise ValueError("corresponding-source annex artifact changed")
    if not any(
        artifact.get("sha256") == farama_delivery["archive_sha256"]
        for artifact in farama_delivery["artifacts"]
    ):
        raise ValueError("exact Farama source archive is absent from the annex")
    if not any(
        artifact.get("sha256") == shadow_delivery["preferred_form_archive_sha256"]
        for artifact in shadow_delivery["artifacts"]
    ):
        raise ValueError("exact Shadow preferred form is absent from the annex")
    manifest_digest = parsed["corresponding-source.lock.json"].get(
        "final_rootfs_manifest_sha256"
    )
    _classified_rootfs(rootfs, manifest_digest)
    _locked_assets(rootfs)


def scan(path: Path) -> dict[str, Any]:
    with tarfile.open(path, mode="r:*") as archive:
        manifest = _json_member(archive, "manifest.json")
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError("Docker save must contain exactly one image")
        entry = manifest[0]
        if not isinstance(entry, dict):
            raise ValueError("Docker save manifest entry must be an object")
        config_name = _safe(str(entry["Config"]))
        config = _json_member(archive, config_name)
        if not isinstance(config, dict):
            raise ValueError("Docker save config must be an object")
        runtime_config = config.get("config")
        if (
            not isinstance(runtime_config, dict)
            or runtime_config.get("User") != "ubuntu"
        ):
            raise ValueError("final image must declare the non-root ubuntu user")
        history = json.dumps(config.get("history", []), sort_keys=True).encode()
        if SECRET_TEXT.search(history) or VENDOR_TEXT.search(history):
            raise ValueError("forbidden payload signature in image history")
        layers = [_safe(str(name)) for name in entry.get("Layers", [])]
        if not layers:
            raise ValueError("Docker save contains no layers")
        rootfs, layer_members, nested_members = _layers(archive, layers)
    missing = sorted(REQUIRED - rootfs.keys())
    if missing:
        raise ValueError(f"required image files absent: {missing}")
    _complete_locks(rootfs)
    return {
        "schema": "npa.gymnasium-robotics.payload-scan.v1",
        "status": "passed",
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "layer_count": len(layers),
        "layer_member_count": layer_members,
        "nested_archive_member_count": nested_members,
        "final_regular_file_count": len(rootfs),
        "unresolved_findings": 0,
        "accepted_manifest_present": False,
        "release_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-save", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = scan(args.docker_save)
    except (
        KeyError,
        OSError,
        ValueError,
        tarfile.TarError,
        json.JSONDecodeError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
