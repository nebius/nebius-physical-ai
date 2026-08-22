#!/usr/bin/env python3
"""Reject restricted or gated payload in an exact Cosmos3 serving image tarball."""

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
    re.compile(r"(?i)(^|/)NGC-DL-CONTAINER-LICENSE$"),
    re.compile(r"(?i)(^|/)(huggingface|hf-cache)/hub/models--nvidia--"),
    re.compile(r"(?i)(^|/)(Cosmos3-Super|Cosmos-1\.0-Guardrail)(/|$)"),
    re.compile(r"(?i)(^|/)site-packages/(vllm|vllm_omni|torch|nvidia|cuda)(/|$)"),
    re.compile(r"(?i)(^|/)(?:usr/local/)?bin/(vllm|torchrun)$"),
    re.compile(r"(?i)(^|/)usr/local/cuda(?:-[^/]+)?(/|$)"),
    re.compile(r"(?i)(^|/)opt/nvidia(/|$)"),
    re.compile(r"(?i)(^|/)cuda_bindings-[^/]+\.dist-info/licenses/LICENSE$"),
)
FORBIDDEN_HISTORY = (
    re.compile(r"(?i)vllm/vllm-omni"),
    re.compile(r"(?i)nvcr\.io/"),
    re.compile(r"(?i)(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)="),
    re.compile(r"(?i)(ACCEPT_EULA|ISAACSIM_ACCEPT_EULA|OMNI_KIT_ACCEPT_EULA)=YES"),
    re.compile(r"(?i)NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES"),
    re.compile(r"(?i)FROM\s+[^\n]*(nvidia/cuda|pytorch/pytorch)"),
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
        serialized = json.dumps(config.get("config", {}), sort_keys=True)
        commands = (
            serialized
            + "\n"
            + "\n".join(
                str(item.get("created_by", "")) for item in config.get("history", [])
            )
        )
        for pattern in FORBIDDEN_HISTORY:
            if pattern.search(commands):
                history_hits.append(pattern.pattern)
        for layer_name in record["Layers"]:
            layer_member = outer.extractfile(layer_name)
            if layer_member is None:
                raise RuntimeError(f"missing layer {layer_name}")
            layer_bytes = layer_member.read()
            with tarfile.open(fileobj=io.BytesIO(layer_bytes)) as layer:
                for member in layer:
                    entries += 1
                    name = member.name.lstrip("./")
                    if any(pattern.search(name) for pattern in FORBIDDEN_PATHS):
                        hits.append(name)
                    if (
                        member.isfile()
                        and member.size >= 1024 * 1024
                        and name.lower().endswith(MODEL_SUFFIXES)
                    ):
                        hits.append(name)
    return {
        "format": "npa_cosmos3_serving_payload_scan_v1",
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
        with tempfile.TemporaryDirectory(prefix="npa-cosmos3-scan-") as directory:
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
