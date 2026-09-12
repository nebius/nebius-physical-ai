from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
DOCKERFILE = IMAGE_ROOT / "Dockerfile"
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_robomimic_image_contract", IMAGE_ROOT / "verify_image.py"
)
assert VERIFY_SPEC and VERIFY_SPEC.loader
VERIFIER = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFIER)


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
    baked_names = {
        line.split(b"==", 1)[0].decode("ascii")
        for line in baked.splitlines()
        if re.match(rb"^[A-Za-z0-9_.-]+==", line)
    }
    assert baked_names == {
        "anyio",
        "boto3",
        "botocore",
        "certifi",
        "charset-normalizer",
        "contourpy",
        "cycler",
        "diffusers",
        "filelock",
        "fonttools",
        "fsspec",
        "h11",
        "h5py",
        "hf-xet",
        "httpcore",
        "httpx",
        "huggingface-hub",
        "idna",
        "imageio",
        "importlib-metadata",
        "jmespath",
        "kiwisolver",
        "matplotlib",
        "numpy",
        "packaging",
        "pillow",
        "psutil",
        "pyparsing",
        "python-dateutil",
        "pyyaml",
        "regex",
        "requests",
        "s3transfer",
        "safetensors",
        "six",
        "termcolor",
        "tqdm",
        "typing-extensions",
        "urllib3",
        "zipp",
    }
    assert set(
        VERIFIER._locked_baked_packages(IMAGE_ROOT / "baked-requirements.lock")
    ) == (baked_names)
    # Import reachability at the pinned source revision is broader than the BC
    # code path itself: algo registration imports Diffusion Policy, while the
    # observation stack imports vis_utils and therefore matplotlib eagerly.
    assert {
        "diffusers",
        "huggingface-hub",
        "imageio",
        "matplotlib",
    } <= baked_names
    # These upstream install requirements are lazy and are not exercised by the
    # headless, non-language, non-rendering gate.
    assert {
        "egl-probe",
        "imageio-ffmpeg",
        "tensorboard",
        "tensorboardx",
        "transformers",
    }.isdisjoint(baked_names)
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
