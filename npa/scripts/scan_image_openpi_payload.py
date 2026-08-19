#!/usr/bin/env python3
"""Fail when a built OpenPI image contains checkpoint/cache/credential payload."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tarfile


FORBIDDEN_PATHS: tuple[tuple[str, str], ...] = (
    (
        r"(?i)(^|/)(opt/npa-model-cache/openpi|workspace/openpi-cache|\.cache/openpi)/.+",
        "a populated OpenPI runtime cache",
    ),
    (r"(?i)(^|/)_CHECKPOINT_METADATA$", "an Orbax checkpoint root"),
    (r"(?i)(^|/)ocdbt\.process_[^/]+/", "Orbax checkpoint parameter shards"),
    (r"(?i)(^|/)(auth|credentials|token|cookies)\.json$", "credential material"),
)
FORBIDDEN_HISTORY: tuple[tuple[str, str], ...] = (
    (
        r"(?i)(ENV|ARG)\s+[^\n]*NPA_OPENPI_ACCEPT_GEMMA_TERMS\s*=\s*YES",
        "baked Gemma acceptance",
    ),
    (
        r"(?i)(ENV|ARG)\s+[^\n]*(HF_TOKEN|GOOGLE_APPLICATION_CREDENTIALS|PASSWORD)\s*=\s*\S+",
        "baked model-access credential",
    ),
)


def classify_path(path: str) -> str | None:
    normalized = path.lstrip("./").lstrip("/")
    for pattern, reason in FORBIDDEN_PATHS:
        if re.search(pattern, normalized):
            return reason
    return None


def classify_history(command: str) -> str | None:
    instructions = "\n".join(
        line for line in command.splitlines() if not line.lstrip().startswith("#")
    )
    for pattern, reason in FORBIDDEN_HISTORY:
        if re.search(pattern, instructions):
            return reason
    return None


def _scan_registry(image: str) -> tuple[list[dict[str, object]], int]:
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError("crane not found on PATH")
    process = subprocess.Popen([crane, "export", image, "-"], stdout=subprocess.PIPE)
    assert process.stdout is not None
    hits: list[dict[str, object]] = []
    count = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for member in archive:
                count += 1
                reason = classify_path(member.name)
                if reason:
                    hits.append({"path": member.name, "size": member.size, "reason": reason})
    finally:
        process.stdout.close()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, [crane, "export", image, "-"])
    return hits, count


def _history(image: str) -> tuple[list[dict[str, str]], str]:
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError("crane not found on PATH")
    config = subprocess.run([crane, "config", image], capture_output=True, text=True, check=True)
    payload = json.loads(config.stdout)
    hits = []
    for item in payload.get("history", []):
        command = str(item.get("created_by", ""))
        reason = classify_history(command)
        if reason:
            hits.append({"command": "<redacted build instruction>", "reason": reason})
    digest = subprocess.run(
        [crane, "digest", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError("registry returned an invalid image digest")
    return hits, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    payload_hits, count = _scan_registry(args.image)
    history_hits, digest = _history(args.image)
    clean = not payload_hits and not history_hits
    print(
        json.dumps(
            {
                "format": "npa_openpi_payload_scan_v1",
                "image": args.image,
                "digest": digest,
                "scan_complete": True,
                "entries_scanned": count,
                "payload_hits": payload_hits,
                "history_hits": history_hits,
                "verdict": "clean" if clean else "forbidden-payload-detected",
            },
            sort_keys=True,
        )
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
