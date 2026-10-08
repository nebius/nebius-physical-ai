#!/usr/bin/env python3
"""Reject model, guardrail, cache, or credential payload in Cosmos3 Ray images."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_payload_credentials import (  # noqa: E402
    content_credential,
    normalise_member_name,
    path_credential,
)

FORBIDDEN_PATHS = (
    re.compile(r"(?i)(^|/)(?:\.cache/)?huggingface/hub/models--nvidia--"),
    re.compile(r"(?i)(^|/)models--nvidia--(?:cosmos|nemo-guardrails)"),
    re.compile(
        r"(?i)(^|/)(Cosmos3-Nano|Cosmos-Guardrail1|Cosmos-1\.0-Tokenizer-CV8x8x8)(/|$)"
    ),
    re.compile(
        r"(?i)(^|/)(\.aws/credentials|\.npa/credentials\.yaml|\.docker/config\.json|\.netrc)$"
    ),
)
FORBIDDEN_HISTORY = (
    re.compile(
        r"(?i)(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|NGC_API_KEY|AWS_SECRET_ACCESS_KEY)="
    ),
    re.compile(r"(?i)NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES"),
    re.compile(r"(?i)(huggingface-cli|hf)\s+download\s+nvidia/"),
)
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".gguf")
_PUBLIC_STDOUT_FIELDS = (
    "format",
    "scan_complete",
    "entries_scanned",
    "payload_hits",
    "history_hits",
    "credential_hits",
    "verdict",
)


@dataclass
class _PayloadFindings:
    """Keep legacy verdicts and safe evidence for each physical blocked member."""

    entries: int = 0
    payload: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    credentials: list[str] = field(default_factory=list)
    members: list[dict[str, object]] = field(default_factory=list)

    def report(self, archive_sha256: str, config_sha256: str) -> dict[str, object]:
        """Add byte identities without accepting or exposing credential values."""
        return {
            "format": "npa_cosmos3_ray_serve_payload_scan_v1",
            "report_scope": "full",
            "scan_complete": True,
            "archive_sha256": archive_sha256,
            "config_sha256": config_sha256,
            "entries_scanned": self.entries,
            "payload_hits": sorted(set(self.payload)),
            "history_hits": sorted(set(self.history)),
            "credential_hits": sorted(set(self.credentials)),
            "credential_members": self.members,
            "verdict": "restricted-payload-detected"
            if self.payload or self.history or self.credentials
            else "clean",
        }


def _stream_sha256(stream: BinaryIO) -> str:
    """Hash complete bytes with bounded reads, including after an early finding."""
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _history_findings(config: dict) -> list[str]:
    """Apply the existing config and history rules without returning their bytes."""
    commands = (
        json.dumps(config.get("config", {}), sort_keys=True)
        + "\n"
        + "\n".join(
            str(item.get("created_by", "")) for item in config.get("history", [])
        )
    )
    return [
        pattern.pattern for pattern in FORBIDDEN_HISTORY if pattern.search(commands)
    ]


def _credential_kind(
    layer: tarfile.TarFile, member: tarfile.TarInfo, name: str
) -> tuple[str, str] | None:
    """Preserve the path-first and existing member-content credential verdict."""
    kind = path_credential(name)
    if kind is not None:
        return kind, "path"
    payload = layer.extractfile(member)
    if payload is None:
        return None
    with payload:
        kind = content_credential(payload)
    if kind is None:
        return None
    return kind, "member-content"


def _credential_member(
    layer: tarfile.TarFile,
    member: tarfile.TarInfo,
    kind: str,
    detection_source: str,
    name: str,
    *,
    layer_index: int,
    layer_sha256: str,
    member_index: int,
) -> dict[str, object]:
    """Identify a blocked stored member without returning any content excerpt."""
    payload = layer.extractfile(member)
    if payload is None:
        raise RuntimeError("cannot hash blocked archive member")
    with payload:
        member_sha256 = _stream_sha256(payload)
    return {
        "kind": kind,
        "path": name,
        "size": member.size,
        "sha256": member_sha256,
        "layer_index": layer_index,
        "layer_sha256": layer_sha256,
        "member_index": member_index,
        "detection_reason": {
            "source": detection_source,
            "rule": kind,
        },
    }


def _scan_member(
    layer: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    member_index: int,
    layer_index: int,
    layer_sha256: str,
    findings: _PayloadFindings,
) -> None:
    """Preserve each existing finding while attaching physical member metadata."""
    findings.entries += 1
    name = normalise_member_name(member.name)
    if any(pattern.search(name) for pattern in FORBIDDEN_PATHS):
        findings.payload.append(name)
    if (
        member.isfile()
        and member.size >= 50 * 1024 * 1024
        and name.lower().endswith(MODEL_SUFFIXES)
    ):
        findings.payload.append(name)
    if not member.isfile():
        return
    credential = _credential_kind(layer, member, name)
    if credential is not None:
        kind, detection_source = credential
        findings.credentials.append(f"{kind}:{name}")
        findings.members.append(
            _credential_member(
                layer,
                member,
                kind,
                detection_source,
                name,
                layer_index=layer_index,
                layer_sha256=layer_sha256,
                member_index=member_index,
            )
        )


def _scan_layer(
    outer: tarfile.TarFile,
    layer_name: str,
    layer_index: int,
    findings: _PayloadFindings,
) -> None:
    """Scan every stored entry, retaining deleted or overwritten member evidence."""
    try:
        layer_member = outer.extractfile(layer_name)
    except KeyError as error:
        raise RuntimeError(f"missing layer {layer_name}") from error
    if layer_member is None:
        raise RuntimeError(f"layer {layer_name} is not a regular file")
    with layer_member, io.BytesIO(layer_member.read()) as layer_bytes:
        with layer_bytes.getbuffer() as layer_view:
            layer_sha256 = hashlib.sha256(layer_view).hexdigest()
        with tarfile.open(fileobj=layer_bytes) as layer:
            for member_index, member in enumerate(layer):
                _scan_member(
                    layer,
                    member,
                    member_index=member_index,
                    layer_index=layer_index,
                    layer_sha256=layer_sha256,
                    findings=findings,
                )


def _read_manifest_record(outer: tarfile.TarFile) -> tuple[str, list[str]]:
    """Read the one valid Docker-save record without leaking archive bytes."""
    try:
        manifest_member = outer.extractfile("manifest.json")
    except KeyError as error:
        raise RuntimeError("missing docker-save manifest") from error
    if manifest_member is None:
        raise RuntimeError("docker-save manifest is not a regular file")
    with manifest_member:
        try:
            manifest = json.load(manifest_member)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("malformed docker-save manifest") from error
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise RuntimeError("expected one image in docker-save archive")
    record = manifest[0]
    if not isinstance(record, dict):
        raise RuntimeError("malformed docker-save manifest")
    config_name = record.get("Config")
    layer_names = record.get("Layers")
    if not isinstance(config_name, str) or not (
        isinstance(layer_names, list)
        and all(isinstance(layer_name, str) for layer_name in layer_names)
    ):
        raise RuntimeError("malformed docker-save manifest")
    return config_name, layer_names


def _read_config(outer: tarfile.TarFile, config_name: str) -> tuple[bytes, dict]:
    """Read validated image configuration and retain fail-closed error messages."""
    try:
        config_member = outer.extractfile(config_name)
    except KeyError as error:
        raise RuntimeError("missing image config") from error
    if config_member is None:
        raise RuntimeError("image config is not a regular file")
    with config_member:
        config_bytes = config_member.read()
    try:
        config = json.loads(config_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("malformed image config") from error
    if not isinstance(config, dict):
        raise RuntimeError("malformed image config")
    return config_bytes, config


def scan_tarball(path: Path) -> dict[str, object]:
    """Scan stored layers and identify blocked bytes without exposing their values.

    Args:
        path: Docker-save archive to scan.
    Returns:
        Existing verdict and safe hashes, sizes and locations of blocked members.
    Raises:
        OSError: Archive bytes cannot be read.
        RuntimeError: Docker-save metadata is malformed or required image data is absent.
        tarfile.TarError: An archive is malformed.
    """
    findings = _PayloadFindings()
    with path.open("rb") as archive:
        archive_sha256 = _stream_sha256(archive)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as outer:
            config_name, layer_names = _read_manifest_record(outer)
            config_bytes, config = _read_config(outer, config_name)
            findings.history = _history_findings(config)
            for layer_index, layer_name in enumerate(layer_names):
                _scan_layer(outer, layer_name, layer_index, findings)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    return findings.report(archive_sha256, config_sha256)


def _stdout_report(report: dict[str, object]) -> dict[str, object]:
    """Keep public logs to the pre-existing blocking-summary fields."""
    summary = {key: report[key] for key in _PUBLIC_STDOUT_FIELDS}
    summary["report_scope"] = "redacted-summary"
    return summary


def _write_private_report(path: Path, rendered: str) -> None:
    """Write report metadata without inheriting broad permissions from a prior file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as report:
        os.fchmod(report.fileno(), 0o600)
        report.write(rendered + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--tarball", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--full-stdout", action="store_true")
    args = parser.parse_args()
    if bool(args.image) == bool(args.tarball):
        parser.error("provide exactly one of IMAGE or --tarball")
    if args.tarball:
        report = scan_tarball(args.tarball)
    else:
        with tempfile.TemporaryDirectory(prefix="npa-cosmos3-ray-scan-") as directory:
            saved = Path(directory) / "image.tar"
            subprocess.run(["docker", "pull", args.image], check=True)
            subprocess.run(
                ["docker", "save", "--output", saved, args.image], check=True
            )
            report = scan_tarball(saved)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        try:
            _write_private_report(args.json, rendered)
        except OSError:
            print(json.dumps({"error": "report-write-failed"}), file=sys.stderr)
            if args.full_stdout:
                print(rendered)
            else:
                print(json.dumps(_stdout_report(report), indent=2, sort_keys=True))
            return 2
    if not args.full_stdout:
        rendered = json.dumps(_stdout_report(report), indent=2, sort_keys=True)
    print(rendered)
    return 0 if report["verdict"] == "clean" else 1


if __name__ == "__main__":
    sys.exit(main())
