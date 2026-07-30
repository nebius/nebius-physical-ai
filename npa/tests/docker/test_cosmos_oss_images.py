"""The Cosmos OSS images must not redistribute model weights.

Both images bake an upstream source checkout (Apache-2.0, redistributable) but no
model weights: the curator's GPU stages need third-party and NVIDIA models under
their own licenses, and upstream's objects check needs EULA-gated Git-LFS objects.
These are static checks on the Dockerfiles and entrypoints, so they run in CI
without a build; the build itself repeats the assertion against the real filesystem
and each image's golden eval proves the checkout still works.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_ROOT = REPO_ROOT / "npa" / "docker" / "workbench"
IMAGES = ("cosmos-curate", "cosmos-evaluator")

# Fetching a model at build time is what would put a weight in a layer.
WEIGHT_FETCH_PATTERNS = (
    r"snapshot_download",
    r"hf_hub_download",
    r"huggingface-cli\s+download",
    r"\bhf\s+download\b",
    r"ngc\s+registry\s+model\s+download",
    r"git\s+lfs\s+(pull|fetch|install)",
)
WEIGHT_SUFFIXES = (".onnx", ".safetensors", ".pth", ".pt", ".bin", ".ckpt", ".gguf")


def _dockerfile(image: str) -> str:
    path = DOCKER_ROOT / image / "Dockerfile"
    assert path.is_file(), f"missing Dockerfile for {image}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("image", IMAGES)
def test_dockerfile_does_not_download_weights_at_build_time(image: str) -> None:
    body = _dockerfile(image)
    for pattern in WEIGHT_FETCH_PATTERNS:
        matches = [
            line.strip()
            for line in body.splitlines()
            # Comments explain the policy and legitimately name these commands.
            if not line.lstrip().startswith("#") and re.search(pattern, line)
        ]
        assert not matches, f"{image} Dockerfile fetches weights at build time: {matches}"


@pytest.mark.parametrize("image", IMAGES)
def test_dockerfile_never_copies_a_weight_file(image: str) -> None:
    body = _dockerfile(image)
    copied = [
        line.strip()
        for line in body.splitlines()
        if line.lstrip().upper().startswith(("COPY", "ADD"))
        and any(suffix in line.lower() for suffix in WEIGHT_SUFFIXES)
    ]
    assert not copied, f"{image} Dockerfile copies a weight file into the image: {copied}"


@pytest.mark.parametrize("image", IMAGES)
def test_upstream_checkout_skips_git_lfs_payloads(image: str) -> None:
    """LFS objects are how upstream ships its EULA-gated weights."""

    body = _dockerfile(image)
    assert "git fetch" in body, f"{image} Dockerfile does not fetch an upstream checkout"
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "git fetch" in stripped or "git checkout" in stripped:
            assert "GIT_LFS_SKIP_SMUDGE=1" in stripped, (
                f"{image}: `{stripped}` must set GIT_LFS_SKIP_SMUDGE=1 so LFS weights stay out"
            )


@pytest.mark.parametrize("image", IMAGES)
def test_build_asserts_no_weights_landed(image: str) -> None:
    body = _dockerfile(image)
    assert "must never be baked into this image" in body, (
        f"{image} Dockerfile has no build-time check that the image carries no weights"
    )


def test_curator_documents_the_runtime_weight_fetch() -> None:
    """The curator's GPU stages do need weights, so the runtime path must be wired."""

    body = _dockerfile("cosmos-curate")
    assert "NPA_COSMOS_CURATE_WEIGHTS_DIR=/config/models" in body
    assert 'VOLUME ["/config/models"]' in body, (
        "the weights directory must be a volume so downloads survive across runs"
    )
    entrypoint = (DOCKER_ROOT / "cosmos-curate" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "fetch-models" in entrypoint, "the image must expose a fetch-models mode"


def test_evaluator_needs_no_weights_at_all() -> None:
    """Nothing NPA wires in the evaluator loads a local model."""

    body = _dockerfile("cosmos-evaluator")
    assert "npa.model_weights=\"none" in body
    assert "GIT_LFS_SKIP_SMUDGE=1" in body


@pytest.mark.parametrize("image", IMAGES)
def test_entrypoint_is_mode_based_and_not_bash(image: str) -> None:
    """The packaging contract's job tier forbids a bare bash entrypoint."""

    contract = yaml.safe_load(
        (DOCKER_ROOT / "packaging-contract.yaml").read_text(encoding="utf-8")
    )
    assert contract["images"][image]["tier"] == "job"
    body = _dockerfile(image)
    assert f'ENTRYPOINT ["/opt/npa/docker/workbench/{image}/entrypoint.sh"]' in body
    entrypoint = DOCKER_ROOT / image / "entrypoint.sh"
    assert entrypoint.is_file(), f"missing entrypoint for {image}"
    text = entrypoint.read_text(encoding="utf-8")
    assert "smoke" in text, "the golden-eval smoke must be reachable as a mode"
    assert "engine" in text, "the availability report must be reachable as a mode"


@pytest.mark.parametrize("image", IMAGES)
def test_golden_eval_command_matches_the_image_smoke_script(image: str) -> None:
    manifest = yaml.safe_load(
        (REPO_ROOT / "npa" / "src" / "npa" / "smoke" / "golden_evals.yaml").read_text(encoding="utf-8")
    )
    entry = manifest["containers"][image]
    command = entry["golden_eval"]["command"]
    assert f"docker/workbench/{image}/smoke_functional.py" in command
    assert (DOCKER_ROOT / image / "smoke_functional.py").is_file()
    # These images are CPU-only, so a GPU requirement would be wrong.
    assert entry["golden_eval"]["gpu"] == "none"
