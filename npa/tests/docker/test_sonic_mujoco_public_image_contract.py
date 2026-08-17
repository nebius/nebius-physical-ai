"""Static guards for the independently rebuilt public SONIC MuJoCo image."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = REPO_ROOT / "npa/docker/workbench/sonic"
DOCKERFILE = IMAGE_DIR / "Dockerfile.mujoco"
LOCK = IMAGE_DIR / "mujoco-requirements.lock"
MANIFEST = REPO_ROOT / "npa/src/npa/deploy/sonic_image_manifest.json"
ENTRYPOINT = IMAGE_DIR / "entrypoint.sh"
MUJOCO_EVAL = IMAGE_DIR / "mujoco_eval.py"


def test_independent_public_base_source_and_hashed_closure() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lock = LOCK.read_text(encoding="utf-8")
    assert re.search(r"ARG BASE_IMAGE=python:[^\s]+@sha256:[0-9a-f]{64}", text)
    assert "ARG DEBIAN_SNAPSHOT=20260817T000000Z" in text
    assert "libgnutls30 libssl3 openssl" in text
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in text
    assert "FROM npa-sonic" not in text
    assert "nvcr.io" not in text
    assert "vllm/vllm-omni" not in text
    assert "0a87181c9106d0e49293400714b157676e0ec664" in text
    assert "--require-hashes" in text
    assert "--hash=sha256:" in lock
    assert "torch==2.9.0" in lock
    assert "mujoco==3.11.0" in lock
    assert "su -s /bin/sh -c 'test -r" in text


def test_no_vendor_payload_consent_or_weight_fetch_is_baked() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    forbidden = re.compile(
        r"(?i)(NGC-DL-CONTAINER-LICENSE|ACCEPT_EULA=(?:Y|YES)|"
        r"OMNI_KIT_ACCEPT_EULA=(?:Y|YES)|ISAACSIM_ACCEPT_EULA=(?:Y|YES)|"
        r"git\s+lfs\s+pull|hf\s+download)"
    )
    assert not forbidden.search(text)
    assert "GIT_LFS_SKIP_SMUDGE=1" in text
    assert "!/decoupled_wbc/dexmg/**" in text
    assert "LicenseRef-NVIDIA-CUDA-Toolkit" in text


def test_mujoco_smoke_propagates_failures_and_uses_no_lfs_payload() -> None:
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    evaluator = MUJOCO_EVAL.read_text(encoding="utf-8")
    assert "mujoco-smoke|mujoco_smoke" in entrypoint
    assert 'run_mujoco_eval || return $?' in entrypoint
    assert 'mujoco_eval.py || return $?' in entrypoint
    assert "primitive-proxy-no-lfs-payload" in evaluator
    assert "git-lfs.github.com/spec" in evaluator


def test_manifest_records_exact_gpu_accepted_release() -> None:
    from npa.deploy.images import GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidate = next(
        item for item in manifest["images"] if item["id"] == "sonic-mujoco-runtime-fetch"
    )
    assert candidate["status"] == "active"
    assert candidate["redistribution"] == "public-runtime-fetch"
    assert candidate["digest"] == (
        "sha256:2388d9e97269afaa414966e83a27f676a3f44d4271e9828c57bc13fbdce80f57"
    )
    assert candidate["digest"] == GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS["sonic-mujoco"]
    assert candidate["base"]["image"].startswith("python:3.11.14-slim-bookworm@sha256:")
    assert candidate["base"]["cuda"] == "12.8 wheel closure"
