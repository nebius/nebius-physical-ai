#!/usr/bin/env python3
"""Fail closed when a built SeedVR2 image layer contains runtime-only payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile


@dataclass(frozen=True)
class Finding:
    """One forbidden path or credential-shaped value in an immutable layer."""

    kind: str
    layer: str
    path: str


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "model_weight",
        re.compile(
            r"^(?!.*site-packages/[^/]+\.pth$).*\.(?:safetensors|ckpt|gguf|pth|pt)$",
            re.I,
        ),
    ),
    (
        "model_weight",
        re.compile(
            r"(?:^|/)(?:pytorch_model|diffusion_pytorch_model|model)-?\d*.*\.bin$", re.I
        ),
    ),
    (
        "populated_hf_cache",
        re.compile(r"(?:^|/)(?:\.cache/)?huggingface/(?:hub|datasets)/.+[^/]$", re.I),
    ),
    (
        "sensor_or_output_media",
        re.compile(r"^(?:opt/seedvr2|opt/npa-src)/.+\.(?:mp4|mov|avi|mkv|mcap)$", re.I),
    ),
    (
        "credential_file",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.git-credentials|\.docker/config\.json|"
            r"kubeconfig|ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
            re.I,
        ),
    ),
)

SECRET_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
)


def _is_application_content(name: str) -> bool:
    return name.startswith(("opt/seedvr2/", "opt/npa-src/")) or name in {
        "usr/local/bin/seedvr2-entrypoint",
        "usr/share/doc/npa-seedvr2/REDISTRIBUTION.md",
        "usr/share/doc/npa-seedvr2/THIRD_PARTY_NOTICES.md",
    }


def _scan_layer_archive(archive: tarfile.TarFile, *, layer: str) -> list[Finding]:
    findings: list[Finding] = []
    for member in archive:
        name = member.name.lstrip("./")
        if member.isdir():
            continue
        for kind, pattern in FORBIDDEN_PATHS:
            if pattern.search(name):
                findings.append(Finding(kind, layer, name))
        if (
            member.isfile()
            and _is_application_content(name)
            and member.size <= 16 * 1024**2
        ):
            stream = archive.extractfile(member)
            payload = stream.read() if stream is not None else b""
            if any(pattern.search(payload) for pattern in SECRET_CONTENT):
                findings.append(Finding("credential_content", layer, name))
    return findings


def scan_layer(path: Path, *, layer: str) -> list[Finding]:
    """Scan one layer archive, including bytes hidden by later whiteouts."""

    with tarfile.open(path) as archive:
        return _scan_layer_archive(archive, layer=layer)


def scan_saved_image(image_tar: Path) -> tuple[list[Finding], int]:
    """Scan config plus every immutable layer in one docker-save archive."""

    findings: list[Finding] = []
    layers = 0
    with tarfile.open(image_tar) as archive:
        manifest_stream = archive.extractfile("manifest.json")
        if manifest_stream is None:
            raise RuntimeError("docker-save archive has no manifest.json")
        manifests = json.loads(manifest_stream.read())
        if len(manifests) != 1:
            raise RuntimeError("expected exactly one image manifest")
        config_name = manifests[0].get("Config", "")
        config_stream = archive.extractfile(config_name) if config_name else None
        if config_stream is None:
            raise RuntimeError("docker-save archive has no image config")
        config_payload = config_stream.read()
        if any(pattern.search(config_payload) for pattern in SECRET_CONTENT):
            findings.append(Finding("credential_content", "image-config", config_name))
        for relative in manifests[0]["Layers"]:
            layers += 1
            layer_stream = archive.extractfile(relative)
            if layer_stream is None:
                raise RuntimeError(f"docker-save archive has no layer {relative}")
            with tarfile.open(fileobj=layer_stream, mode="r|*") as layer_archive:
                findings.extend(_scan_layer_archive(layer_archive, layer=relative))
    return findings, layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Local Docker image reference.")
    parser.add_argument("--tarball", type=Path, help="Existing docker-save tarball.")
    args = parser.parse_args()
    if bool(args.image) == bool(args.tarball):
        parser.error("provide exactly one image or --tarball")
    if args.tarball:
        source = str(args.tarball)
        findings, layers = scan_saved_image(args.tarball)
    else:
        source = str(args.image)
        with tempfile.TemporaryDirectory(prefix="npa-seedvr2-image-") as scratch:
            tarball = Path(scratch) / "image.tar"
            subprocess.run(["docker", "save", source, "-o", str(tarball)], check=True)
            findings, layers = scan_saved_image(tarball)
    report = {
        "format": "npa_seedvr2_payload_scan_v1",
        "source": source,
        "scan_complete": True,
        "layers_scanned": layers,
        "verdict": "clean" if not findings else "runtime-only-payload-detected",
        "findings": [asdict(item) for item in findings],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
