#!/usr/bin/env python3
"""Fail closed when an Antioch adapter image contains restricted or private bytes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    detail: str


FORBIDDEN_PATHS = (
    ("vendor_state", re.compile(r"(?:^|/)(?:\.antioch|antioch[-_](?:config|cache|auth)|auth\.json|machines\.json|credentials\.json)(?:/|$)", re.I)),
    ("credential_file", re.compile(r"(?:^|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials|kubeconfig|ssh_host_(?:rsa|ecdsa|ed25519)_key)$", re.I)),
    ("checkpoint_or_weight", re.compile(r"(?:\.(?:safetensors|ckpt|onnx|gguf)$|(?:^|/)(?:weights?|checkpoints?|models?)/.*\.(?:bin|pt|pth)$)", re.I)),
    ("vendor_distribution", re.compile(r"(?:^|/)(?:antioch[-_]?sim|antioch_sim-.*\.(?:dist-info|egg-info))(?:/|$)", re.I)),
)
FORBIDDEN_HISTORY = (
    (
        "vendor_install",
        re.compile(
            r"(?:^|[;&|]\s*)[^;&|\n]*\bpip(?:3)?\s+(?:install|download)"
            r"(?:\s+--?[a-z0-9_-]+(?:[= ]\S+)?)?\s+antioch[-_]?sim(?:[=<>!~]|\s|$)",
            re.I,
        ),
    ),
    ("vendor_payload", re.compile(r"(?:COPY|ADD)[^\n]*(?:antioch[-_]?sim|omniverse|isaac[-_ ]?sim)", re.I)),
    ("cached_acceptance", re.compile(r"NPA_ANTIOCH_ACCEPT_TERMS\s*=\s*YES", re.I)),
)
SECRET_CONTENT = (
    re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n"
        rb"[A-Za-z0-9+/=\r\n]{64,}-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
)
SECRET_CONFIG = re.compile(
    r'(?i)(?:AWS_SECRET_ACCESS_KEY|ANTIOCH_(?:TOKEN|API_KEY)|HF_TOKEN)=[^$<"\s][^"\s]{7,}'
)
VENDOR_METADATA = re.compile(rb"(?im)^Name:\s*antioch[-_]?sim\s*$")
PROPRIETARY_BINARY = re.compile(rb"(?i)(?:antioch[-_]?sim|NVIDIA Omniverse|Isaac Sim)")


def _normal(path: str) -> str:
    return path.lstrip("./").lstrip("/")


def _inspect_file(path: str, payload: bytes) -> list[Finding]:
    findings: list[Finding] = []
    normalized = _normal(path)
    for kind, pattern in FORBIDDEN_PATHS:
        if pattern.search(normalized):
            findings.append(Finding(kind, normalized, "forbidden shipped path"))
    if payload.lstrip().startswith(b"-----BEGIN") and any(
        pattern.search(payload) for pattern in SECRET_CONTENT
    ):
        findings.append(Finding("credential_material", normalized, "secret or private-key signature"))
    if normalized.endswith(("METADATA", "PKG-INFO")) and VENDOR_METADATA.search(payload):
        findings.append(Finding("renamed_vendor_distribution", normalized, "distribution metadata identifies antioch-sim"))
    if payload.startswith(b"\x7fELF") and PROPRIETARY_BINARY.search(payload):
        findings.append(Finding("proprietary_binary", normalized, "ELF contains a proprietary runtime signature"))
    return findings


def scan_tarball(path: Path) -> dict[str, object]:
    findings: list[Finding] = []
    entries = layers = 0
    with tarfile.open(path, "r") as outer:
        manifest_member = outer.getmember("manifest.json")
        manifest_file = outer.extractfile(manifest_member)
        if manifest_file is None:
            raise ValueError("docker-save manifest is unreadable")
        manifest = json.load(manifest_file)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError("scanner requires one docker-save image")
        config_name = manifest[0].get("Config")
        layer_names = manifest[0].get("Layers")
        if not isinstance(config_name, str) or not isinstance(layer_names, list):
            raise ValueError("docker-save manifest schema is invalid")
        config_file = outer.extractfile(config_name)
        if config_file is None:
            raise ValueError("OCI config is unreadable")
        config = json.load(config_file)
        serialized_config = json.dumps(config.get("config") or {}, sort_keys=True)
        if SECRET_CONFIG.search(serialized_config):
            findings.append(
                Finding(
                    "credential_material",
                    config_name,
                    "secret-shaped value in OCI config",
                )
            )
        for kind, pattern in FORBIDDEN_HISTORY:
            if pattern.search(serialized_config):
                findings.append(Finding(kind, config_name, "forbidden OCI config value"))
        for item in config.get("history") or []:
            command = str(item.get("created_by") or "")
            for kind, pattern in FORBIDDEN_HISTORY:
                if pattern.search(command):
                    findings.append(Finding(kind, config_name, "forbidden layer history instruction"))
        for layer_name in layer_names:
            layer_file = outer.extractfile(layer_name)
            if layer_file is None:
                raise ValueError("image layer is unreadable")
            layers += 1
            with tarfile.open(fileobj=layer_file, mode="r|*") as layer:
                for member in layer:
                    entries += 1
                    if not member.isfile():
                        continue
                    handle = layer.extractfile(member)
                    if handle is None:
                        raise ValueError("image member is unreadable")
                    payload = handle.read()
                    findings.extend(_inspect_file(member.name, payload))
    unique = sorted({(f.kind, f.path, f.detail) for f in findings})
    return {
        "format": "npa_antioch_payload_scan_v2",
        "scan_complete": True,
        "entries_scanned": entries,
        "layers_scanned": layers,
        "findings": [asdict(Finding(*item)) for item in unique],
        "verdict": "clean" if not unique else "forbidden-payload-detected",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--tarball", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.tarball):
        parser.error("pass exactly one image or --tarball")
    if args.tarball:
        report = scan_tarball(args.tarball)
    else:
        with tempfile.TemporaryDirectory(prefix="npa-antioch-scan-") as directory:
            archive = Path(directory) / "image.tar"
            completed = subprocess.run(["docker", "save", "--output", str(archive), args.image], check=False)
            if completed.returncode:
                raise SystemExit("could not export built Antioch image")
            report = scan_tarball(archive)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    return 0 if report["verdict"] == "clean" else 1


if __name__ == "__main__":
    sys.exit(main())
