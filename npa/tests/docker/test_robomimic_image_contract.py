from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
DOCKERFILE = IMAGE_ROOT / "Dockerfile"


def test_neutral_image_is_pinned_non_root_and_contains_no_cuda_install() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "FROM python:3.11.16-slim-bookworm@sha256:"
        "528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
    ) in text
    assert "USER ubuntu" in text
    assert 'ENTRYPOINT ["/opt/npa/robomimic/entrypoint.sh"]' in text
    assert "org.nebius.npa.skypilot-bootstrap-contract" in text
    normalized = re.sub(r"\\\n\s*", " ", text)
    assert re.search(r"\b(?:nvidia/cuda|pytorch/pytorch):", normalized, re.I) is None
    assert (
        re.search(r"\bpip\s+install\b[^\n]*(?:torch|nvidia-)", normalized, re.I) is None
    )
    assert "robomimic-runtime assert-refusal" in normalized
    assert "robomimic-runtime ensure" not in normalized
    assert "robomimic-runtime exec" not in normalized


def test_neutral_image_boundaries_and_locks_are_explicit() -> None:
    baked = (IMAGE_ROOT / "baked-requirements.lock").read_bytes()
    assert len(baked.splitlines()) == 48
    lowered = baked.lower()
    for forbidden in (b"torch==", b"torchvision==", b"triton==", b"nvidia-"):
        assert forbidden not in lowered
    runtime = json.loads((IMAGE_ROOT / "runtime-requirements.lock").read_text())
    assert runtime["schema"] == "npa.robomimic.runtime-lock.v1"
    assert runtime["packages"]["torch"] == "2.7.1+cu128"
    assert runtime["packages"]["nvidia-cudnn-cu12"] == "9.7.1.26"
    assert runtime["artifact_hash_closure"] == "required-in-runtime-inventory"
    for filename in (
        "REDISTRIBUTION.md",
        "THIRD_PARTY_NOTICES.md",
        "runtime_bootstrap.sh",
        "verify_image.py",
        "smoke.py",
    ):
        assert (IMAGE_ROOT / filename).is_file()


def test_no_local_consent_proxy_exists() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in IMAGE_ROOT.iterdir()
        if path.is_file()
    )
    assert "NPA_ROBOMIMIC_ACCEPT" not in combined
    assert "CUDA_ACCEPT" not in combined
    assert "CUDNN_ACCEPT" not in combined
