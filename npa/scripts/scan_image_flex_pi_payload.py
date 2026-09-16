#!/usr/bin/env python3
"""Fail closed when a flex-pi image layer contains runtime-only payload."""

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
    """One forbidden layer path or credential signature."""

    kind: str
    layer: str
    path: str


FORBIDDEN_PATHS = (
    ("model_weight", re.compile(r"\.(?:safetensors|ckpt|gguf)$", re.I)),
    ("model_weight", re.compile(r"(?:^|/)(?:pytorch_model|step_\d+|model)-?\d*.*\.(?:bin|pt|pth)$", re.I)),
    ("populated_model_cache", re.compile(
        r"^(?:workspace|root|home/[^/]+)/\.cache/(?:huggingface|modelscope|flex-pi)/.+[^/]$",
        re.I,
    )),
    ("packaged_checkpoint", re.compile(r"^opt/flex-pi/(?:checkpoints|weights|models)/.+[^/]$", re.I)),
    ("robotwin_payload", re.compile(r"^opt/flex-pi/(?!\.venv/).+\.(?:parquet|mp4|mkv)$", re.I)),
    ("credential_file", re.compile(r"(?:^|/)(?:\.aws/credentials|\.git-credentials|\.docker/config\.json|kubeconfig|ssh_host_(?:rsa|ecdsa|ed25519)_key)$", re.I)),
)
SECRET_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
)


def _application_content(name: str) -> bool:
    return name.startswith("opt/npa-src/") or name.startswith("opt/flex-pi/")


def _scan_archive(archive: tarfile.TarFile, *, layer: str) -> list[Finding]:
    findings = []
    for member in archive:
        name = member.name.lstrip("./")
        if member.isdir():
            continue
        for kind, pattern in FORBIDDEN_PATHS:
            if pattern.search(name):
                findings.append(Finding(kind, layer, name))
        if member.isfile() and _application_content(name) and member.size <= 16 * 1024**2:
            stream = archive.extractfile(member)
            payload = stream.read() if stream is not None else b""
            if any(pattern.search(payload) for pattern in SECRET_CONTENT):
                findings.append(Finding("credential_content", layer, name))
    return findings


def scan_layer(path: Path, *, layer: str) -> list[Finding]:
    """Scan one layer including bytes hidden by later whiteouts."""
    with tarfile.open(path) as archive:
        return _scan_archive(archive, layer=layer)


def scan_saved_image(image_tar: Path) -> tuple[list[Finding], int]:
    """Scan every image layer and the immutable image configuration."""
    findings = []
    with tarfile.open(image_tar) as archive:
        stream = archive.extractfile("manifest.json")
        if stream is None:
            raise RuntimeError("docker-save archive has no manifest")
        manifests = json.loads(stream.read())
        if len(manifests) != 1:
            raise RuntimeError("expected exactly one image manifest")
        config = archive.extractfile(manifests[0]["Config"])
        config_bytes = config.read() if config is not None else b""
        if any(pattern.search(config_bytes) for pattern in SECRET_CONTENT):
            findings.append(Finding("credential_content", "image-config", manifests[0]["Config"]))
        for relative in manifests[0]["Layers"]:
            layer_stream = archive.extractfile(relative)
            if layer_stream is None:
                raise RuntimeError("image layer is missing")
            with tarfile.open(fileobj=layer_stream, mode="r|*") as layer_archive:
                findings.extend(_scan_archive(layer_archive, layer=relative))
    return findings, len(manifests[0]["Layers"])


def main() -> int:
    """Scan a local image or existing docker-save tarball."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--tarball", type=Path)
    args = parser.parse_args()
    if bool(args.image) == bool(args.tarball):
        parser.error("provide exactly one image or --tarball")
    if args.tarball:
        source, result = str(args.tarball), scan_saved_image(args.tarball)
    else:
        with tempfile.TemporaryDirectory(prefix="npa-flex-pi-image-") as scratch:
            tarball = Path(scratch) / "image.tar"
            subprocess.run(["docker", "save", args.image, "-o", str(tarball)], check=True)
            source, result = args.image, scan_saved_image(tarball)
    findings, layers = result
    print(json.dumps({
        "format": "npa_flex_pi_payload_scan_v1", "source": source,
        "scan_complete": True, "layers_scanned": layers,
        "verdict": "clean" if not findings else "runtime-only-payload-detected",
        "findings": [asdict(item) for item in findings],
    }, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
