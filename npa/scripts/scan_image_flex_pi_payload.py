#!/usr/bin/env python3
"""Fail closed when a flex-pi image layer contains runtime-only payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
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
    ("model_weight", re.compile(r"\.(?:safetensors|ckpt|gguf|pt|pth)$", re.I)),
    ("model_weight", re.compile(
        r"(?:^|/)(?:pytorch_model|step_\d+|model|checkpoint|weights?|policy|encoder|decoder|adapter)"
        r"(?:[_.-].*|\d*)?\.bin$", re.I,
    )),
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
DEEPSPEED_LAUNCHER = "opt/conda/bin/deepspeed.pt"
# DeepSpeed 0.18.5's bin/deepspeed.pt is a Python launcher, not a checkpoint.
# Bind its exact source body; pip rewrites only the interpreter shebang.
# Source: https://pypi.org/project/deepspeed/0.18.5/
DEEPSPEED_LAUNCHER_BODY_SHA256 = "ec0f0f7dc8b59ded078668a97341535e382040ed7e6b928d119abd36a4a5fb6c"


def _application_content(name: str) -> bool:
    return name.startswith("opt/npa-src/") or name.startswith("opt/flex-pi/")


def _python_path_configuration(name: str, payload: bytes | None) -> bool:
    # Python packaging uses .pth for import hooks and module search paths.
    if not re.search(r"/(?:site|dist)-packages/[^/]+\.pth$", name):
        return False
    if payload is None or len(payload) > 64 * 1024:
        return False
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    return bool(lines) and all(
        not line.strip() or line.startswith(("#", "import ", "import\t"))
        or re.fullmatch(r"[\w./-]+", line, flags=re.ASCII) is not None
        for line in lines
    )


def _deepspeed_launcher(name: str, payload: bytes | None) -> bool:
    if name != DEEPSPEED_LAUNCHER or payload is None:
        return False
    shebang, separator, body = payload.partition(b"\n")
    return (
        bool(separator)
        and shebang in {
            b"#!/opt/conda/bin/python", b"#!/opt/conda/bin/python3",
            b"#!/opt/conda/bin/python3.11",
        }
        and hashlib.sha256(body).hexdigest() == DEEPSPEED_LAUNCHER_BODY_SHA256
    )


def _scan_archive(archive: tarfile.TarFile, *, layer: str) -> list[Finding]:
    findings = []
    for member in archive:
        name = member.name.lstrip("./")
        if member.isdir():
            continue
        payload = None
        if member.isfile() and member.size <= 16 * 1024**2:
            if _application_content(name) or name.endswith(".pth") or name == DEEPSPEED_LAUNCHER:
                stream = archive.extractfile(member)
                payload = stream.read() if stream is not None else None
        for kind, pattern in FORBIDDEN_PATHS:
            if pattern.search(name):
                if kind == "model_weight" and (
                    _python_path_configuration(name, payload) or _deepspeed_launcher(name, payload)
                ):
                    continue
                findings.append(Finding(kind, layer, name))
        if payload is not None and _application_content(name):
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
