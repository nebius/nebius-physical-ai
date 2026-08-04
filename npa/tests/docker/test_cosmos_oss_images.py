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
def test_every_runtime_dependency_is_pinned(image: str) -> None:
    """An unpinned dependency makes the image a moving target.

    These images exist to be the one place a Cosmos tool's environment is known
    good, so rebuilding the same Dockerfile must not silently pick up whatever the
    index published since — including a bad release nobody chose.
    """

    body = _dockerfile(image)
    unpinned: list[str] = []
    for raw in body.splitlines():
        line = raw.strip().rstrip("\\").strip()
        # Only the quoted requirement arguments of a pip install continuation line.
        match = re.fullmatch(r'"([A-Za-z0-9_.\-]+(?:\[[a-z0-9,_-]+\])?)"', line)
        if match and "==" not in match.group(1):
            unpinned.append(match.group(1))
    assert not unpinned, f"{image} installs unpinned dependencies: {unpinned}"


@pytest.mark.parametrize("image", IMAGES)
def test_a_piped_download_is_checksummed(image: str) -> None:
    """Piping a download straight into a shell or tar executes whatever was served."""

    body = _dockerfile(image)
    piped = [
        line.strip()
        for line in body.splitlines()
        if not line.lstrip().startswith("#")
        and re.search(r"(curl|wget)[^|]*\|\s*(tar|bash|sh)\b", line)
    ]
    assert not piped, (
        f"{image} pipes a download into tar/sh without verifying it: {piped}; "
        "download to a file, check its sha256, then extract"
    )
    if "curl" in body and "micromamba" in body:
        assert "sha256sum -c" in body, f"{image} fetches micromamba without a checksum"


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


# Images the npa.workflow renderer pins for a Kubernetes stage. SkyPilot's
# Kubernetes provisioner supplies its keep-alive as container *args*, which
# Kubernetes hands to the image ENTRYPOINT — so an ENTRYPOINT that does not exec
# its arguments exits (126) and SkyPilot reports the misleading
# `container not found ("ray-node")` while setting up its runtime.
ORCHESTRATED_IMAGES = ("cosmos-curate", "cosmos-evaluator", "cosmos2-transfer")


@pytest.mark.parametrize("image", ORCHESTRATED_IMAGES)
def test_entrypoint_execs_its_arguments(image: str) -> None:
    body = _dockerfile(image)
    entrypoints = [
        line.strip()
        for line in body.splitlines()
        if line.lstrip().upper().startswith("ENTRYPOINT")
    ]
    assert entrypoints, f"{image} declares no ENTRYPOINT"
    assert len(entrypoints) == 1, f"{image} declares several ENTRYPOINTs: {entrypoints}"
    entrypoint = entrypoints[0]
    assert entrypoint != 'ENTRYPOINT ["/bin/bash"]', (
        f"{image}: a bare bash ENTRYPOINT swallows the args Kubernetes passes, so a "
        "SkyPilot-orchestrated pod exits 126; exec the arguments instead"
    )

    script = re.search(r'ENTRYPOINT \["([^"]+)"\]', entrypoint)
    assert script, f"{image}: ENTRYPOINT must be an exec-form script, got {entrypoint}"
    name = Path(script.group(1)).name
    local = DOCKER_ROOT / image / name
    assert local.is_file(), f"{image}: entrypoint script {name} is not in the image dir"
    text = local.read_text(encoding="utf-8")
    # Either the plain passthrough, or a mode dispatcher whose catch-all execs the
    # unrecognized first word plus the rest — both run an orchestrator's command.
    passes_through = 'exec "$@"' in text or 'exec "$MODE" "$@"' in text
    assert passes_through, (
        f'{image}/{name} must exec its arguments (`exec "$@"`, or a mode dispatcher '
        'whose catch-all branch does) so an orchestrator-supplied command runs'
    )


# SkyPilot's Kubernetes provisioner runs this per pod, as the image's own user:
#   prefix_cmd() { if [ $(id -u) -ne 0 ]; then echo "sudo"; else echo ""; fi; }
#   $(prefix_cmd) apt install openssh-server rsync -y
#   $(prefix_cmd) service ssh restart
# (sky/provision/kubernetes/instance.py). A non-root image without sudo, or without
# those packages, fails the script; the container exits and SkyPilot reports
# `container not found ("ray-node")`, which reads like a scheduling fault.
SKYPILOT_REQUIRED_PACKAGES = ("openssh-server", "rsync", "sudo")


@pytest.mark.parametrize("image", ORCHESTRATED_IMAGES)
def test_image_satisfies_skypilot_kubernetes_setup(image: str) -> None:
    body = _dockerfile(image)
    missing = [pkg for pkg in SKYPILOT_REQUIRED_PACKAGES if pkg not in body]
    assert not missing, (
        f"{image} does not install {missing}, which SkyPilot's Kubernetes setup shells "
        "out to; the pod will exit before SkyPilot can exec into it"
    )


@pytest.mark.parametrize("image", ORCHESTRATED_IMAGES)
def test_non_root_image_grants_passwordless_sudo(image: str) -> None:
    body = _dockerfile(image)
    if "USER ubuntu" not in body:
        pytest.skip(f"{image} does not drop to a non-root user")
    assert "NOPASSWD" in body, (
        f"{image} runs as a non-root user, so SkyPilot's `sudo apt install ...` needs "
        "passwordless sudo for that user"
    )


def test_transfer_image_keeps_its_interpreter_out_of_root() -> None:
    """The venv must be usable by the non-root user the image runs as."""

    body = _dockerfile("cosmos2-transfer")
    assert "UV_PYTHON_INSTALL_DIR=/opt/cosmos/uv-python" in body, (
        "uv installs its interpreter under /root by default, and /root is 0700, so the "
        "venv's bin/python symlink is unreadable by USER ubuntu (exit 126)"
    )
    assert "chown -R ubuntu:ubuntu" in body and "/opt/cosmos/uv-python" in body
    # And the build proves it rather than assuming it.
    assert "su ubuntu -s /bin/bash -c" in body, (
        "the build must verify the venv runs as ubuntu, not just that it exists"
    )
    assert "! grep -q '/root' .venv/pyvenv.cfg" in body, (
        "the build must fail if pyvenv.cfg points into the root-only tree"
    )
    assert "UV_LEGACY_PYTHON_DIR" not in body, (
        "a complete source build must never inherit or repair the opaque private parent's venv"
    )


def test_transfer_builds_a_fresh_rootless_venv_instead_of_relocating_one() -> None:
    """The clean-source build cannot regress to repairing an opaque parent's venv."""

    body = _dockerfile("cosmos2-transfer")
    assert "ARG BASE_IMAGE=npa-cosmos2-transfer" not in body
    assert 'uv python install "${COSMOS_PYTHON_VERSION}"' in body
    assert 'uv sync --locked --no-dev --no-editable --extra=cu128' in body
    assert 'readlink -f .venv/bin/python' in body
    assert 'find "${UV_PYTHON_INSTALL_DIR}"' in body


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
