#!/usr/bin/env python3
"""Reject model, guardrail, cache, or credential payload in Cosmos3 Ray images."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

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


def scan_tarball(path: Path) -> dict[str, object]:
    hits: list[str] = []
    history_hits: list[str] = []
    entries = 0
    with tarfile.open(path) as outer:
        manifest = json.load(outer.extractfile("manifest.json"))  # type: ignore[arg-type]
        if len(manifest) != 1:
            raise RuntimeError("expected one image in docker-save archive")
        record = manifest[0]
        config = json.load(outer.extractfile(record["Config"]))  # type: ignore[arg-type]
        commands = (
            json.dumps(config.get("config", {}), sort_keys=True)
            + "\n"
            + "\n".join(
                str(item.get("created_by", "")) for item in config.get("history", [])
            )
        )
        history_hits.extend(
            pattern.pattern for pattern in FORBIDDEN_HISTORY if pattern.search(commands)
        )
        for layer_name in record["Layers"]:
            layer_member = outer.extractfile(layer_name)
            if layer_member is None:
                raise RuntimeError(f"missing layer {layer_name}")
            with tarfile.open(fileobj=io.BytesIO(layer_member.read())) as layer:
                for member in layer:
                    entries += 1
                    name = member.name.lstrip("./")
                    if any(pattern.search(name) for pattern in FORBIDDEN_PATHS):
                        hits.append(name)
                    if (
                        member.isfile()
                        and member.size >= 50 * 1024 * 1024
                        and name.lower().endswith(MODEL_SUFFIXES)
                    ):
                        hits.append(name)
    return {
        "format": "npa_cosmos3_ray_serve_payload_scan_v1",
        "scan_complete": True,
        "entries_scanned": entries,
        "payload_hits": sorted(set(hits)),
        "history_hits": sorted(set(history_hits)),
        "verdict": "clean"
        if not hits and not history_hits
        else "restricted-payload-detected",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--tarball", type=Path)
    parser.add_argument("--json", type=Path)
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
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["verdict"] == "clean" else 1


if __name__ == "__main__":
    sys.exit(main())
