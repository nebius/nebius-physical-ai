"""Guard the npa-cosmos3-serving image's contract and its preflight behavior.

Two halves, and the second is the point of the file.

The static half is the same shape as ``test_cosmos3_image_contract.py``: the
image is redistributable only because it carries no model weights, and the base
runtime is pinned by digest because every measured claim in the docs was taken
against that digest.

The behavioral half **runs the real entrypoint**. It puts a recording ``vllm``
and a configurable ``nvidia-smi`` on PATH and reads the argv the script actually
built, so the preflight branches and the serve-command assembly are executed
rather than restated. Asserting against a copy of the expected command would
pass while the script produced something else, which is exactly the gap the
review of the access-preflight change called out.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
NPA_ROOT = REPO_ROOT / "npa"
IMAGE_DIR = NPA_ROOT / "docker/workbench/cosmos3-serving"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
ENTRYPOINT = IMAGE_DIR / "entrypoint.sh"
VERIFY_ENV = IMAGE_DIR / "verify_env.py"
CONTRACT = NPA_ROOT / "docker/workbench/packaging-contract.yaml"

# The digest the serving numbers in docs/workbench/cosmos3-super-serving.md were
# measured against.
PINNED_DIGEST = "sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587"

# Anything that would pull weight bytes into a build layer.
WEIGHT_FETCH_PATTERNS = (
    r"hf\s+download",
    r"huggingface-cli\s+download",
    r"snapshot_download",
    r"git\s+lfs\s+pull",
)


def _load_verify_env():
    """Import the in-image build gate so its constants cannot drift from these tests.

    Bytecode writing is suppressed for the duration: the gate lives in the image's
    build context, and a ``__pycache__`` left beside it would ship into a layer.
    """
    spec = importlib.util.spec_from_file_location("cosmos3_serving_verify_env", VERIFY_ENV)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _dockerfile_instructions() -> str:
    lines = [
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


def test_image_files_exist_and_the_entrypoint_is_executable() -> None:
    assert DOCKERFILE.is_file()
    assert ENTRYPOINT.is_file()
    assert VERIFY_ENV.is_file()
    assert os.access(ENTRYPOINT, os.X_OK), "entrypoint.sh must be committed executable"


def test_base_runtime_is_pinned_by_digest() -> None:
    instructions = _dockerfile_instructions()

    assert PINNED_DIGEST in instructions, (
        "the base runtime must stay pinned to the digest the serving numbers were "
        "measured against; `vllm/vllm-omni:cosmos3` is a moving tag"
    )
    assert re.search(r"ARG BASE_IMAGE=\S+@sha256:[0-9a-f]{64}", instructions)


def test_dockerfile_never_downloads_model_weights() -> None:
    instructions = _dockerfile_instructions()

    for pattern in WEIGHT_FETCH_PATTERNS:
        assert not re.search(pattern, instructions, flags=re.IGNORECASE), (
            f"npa-cosmos3-serving Dockerfile matches {pattern!r}: weights must "
            "download at runtime with the operator's own credentials, never into a layer."
        )


def test_dockerfile_runs_the_build_gate_and_drops_to_a_non_root_user() -> None:
    instructions = _dockerfile_instructions()

    assert "verify_env.py" in instructions, "the build must run its own gate"
    users = re.findall(r"(?im)^\s*USER\s+(\S+)\s*$", instructions)
    assert users and users[-1] == "ubuntu"
    assert "EXPOSE 8000" in instructions


def test_dockerfile_sets_the_xet_workaround_this_base_image_needs() -> None:
    """Setting it here is correct precisely because the pins are baked in.

    `npa workbench cosmos3 generate` only warns, because it runs in whatever
    environment the operator built. This Dockerfile chose the base image, so it
    owns the pins, and verify_env.py fails the build once a bump moves off them.
    """
    instructions = _dockerfile_instructions()

    assert "HF_HUB_DISABLE_XET=1" in instructions
    verify_env = _load_verify_env()
    assert verify_env.XET_AFFECTED_PINS == {"hf-xet": "1.5.1", "huggingface_hub": "1.23.0"}


def test_healthcheck_allows_for_the_real_startup_time() -> None:
    """HSDP-sharded configs need minutes. A tight probe restarts healthy boots."""
    instructions = _dockerfile_instructions()

    match = re.search(r"--start-period=(\d+)s", instructions)
    assert match, "the healthcheck must declare a start-period"
    assert int(match.group(1)) >= 900, (
        "start-period must cover a cold-cache startup: HSDP configs reached ready "
        "in roughly 280-290 s warm and about 320 s longer cold"
    )


def test_the_image_is_deliberately_not_yet_a_registered_workbench_tool() -> None:
    """Pin the choice, so registering it later is a decision rather than a drift.

    Registration is one chain, not one edit: a ``packaging-contract.yaml`` entry
    obliges a datacenter Blackwell verdict, which obliges a
    ``CONTAINER_IMAGE_NAMES`` key and a supported-tools pin, which obliges a
    golden-eval manifest entry backed by a real 8-GPU capability smoke. This
    change ships the container and leaves that chain to the maintainer, and the
    docs page says so rather than leaving the absence to look like an oversight.
    """
    from npa.deploy.images import CONTAINER_IMAGE_NAMES

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert "cosmos3-serving" not in contract["images"]
    assert "cosmos3-serving" not in CONTAINER_IMAGE_NAMES

    docs = (REPO_ROOT / "docs/workbench/cosmos3-super-serving.md").read_text(encoding="utf-8")
    assert "not registered as a deployable workbench tool" in docs


# ---------------------------------------------------------------------------
# Behavioral: run the real entrypoint
# ---------------------------------------------------------------------------

RECORDING_VLLM = """#!/usr/bin/env bash
printf '%s\\n' "$*" > "${NPA_TEST_ARGV_FILE}"
exit 0
"""

FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
seq 0 $(( {gpus} - 1 ))
"""


@pytest.fixture
def harness(tmp_path):
    """PATH with a recording ``vllm``, plus a knob for the visible GPU count."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is present everywhere we run
        pytest.skip("bash is required to execute the entrypoint")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv.txt"

    vllm = bin_dir / "vllm"
    vllm.write_text(RECORDING_VLLM, encoding="utf-8")
    vllm.chmod(0o755)

    cache = tmp_path / "hf-cache"
    cache.mkdir()

    def run(gpus: int | None = 8, **env_overrides):
        if gpus is not None:
            smi = bin_dir / "nvidia-smi"
            smi.write_text(FAKE_NVIDIA_SMI.format(gpus=gpus), encoding="utf-8")
            smi.chmod(0o755)
        else:
            (bin_dir / "nvidia-smi").unlink(missing_ok=True)

        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "HF_HOME": str(cache),
            "NPA_TEST_ARGV_FILE": str(argv_file),
        }
        env.update({key: str(value) for key, value in env_overrides.items()})
        argv_file.unlink(missing_ok=True)
        result = subprocess.run(
            [str(ENTRYPOINT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        served = argv_file.read_text(encoding="utf-8").strip() if argv_file.exists() else None
        return result, served

    return run


def test_entrypoint_execs_the_pinned_eight_gpu_config(harness) -> None:
    result, served = harness(gpus=8, HF_TOKEN="hf_fake_token_for_test")

    assert result.returncode == 0, result.stderr
    assert served is not None, "the entrypoint never reached vllm"
    verify_env = _load_verify_env()
    for flag in verify_env.REQUIRED_SERVE_FLAGS:
        assert flag in served, f"{flag!r} missing from: {served}"
    assert served.startswith("serve nvidia/Cosmos3-Super")
    # Guardrails default to on, which means the flag that disables them is absent.
    assert "--no-guardrails" not in served
    assert "--init-timeout 1800" in served


def test_guardrails_off_passes_the_flag_and_needs_no_token(harness) -> None:
    result, served = harness(gpus=8, NPA_COSMOS3_SERVE_GUARDRAILS="off")

    assert result.returncode == 0, result.stderr
    assert served is not None
    assert "--no-guardrails" in served


def test_guardrails_on_without_a_token_fails_before_the_server_starts(harness) -> None:
    result, served = harness(gpus=8)

    assert result.returncode != 0
    assert served is None, "the server must not start when the guardrail fetch will 401"
    assert "nvidia/Cosmos-1.0-Guardrail" in result.stderr
    # The 401-vs-403 split is what tells an operator whether the token or the
    # license acceptance is the problem, so the message has to carry both.
    assert "401" in result.stderr and "403" in result.stderr


def test_a_gpu_count_that_does_not_fit_the_pinned_config_fails_fast(harness) -> None:
    result, served = harness(gpus=4, HF_TOKEN="hf_fake_token_for_test")

    assert result.returncode != 0
    assert served is None
    assert "found 4" in result.stderr
    assert "8 GPUs" in result.stderr


def test_a_matching_override_lets_a_different_gpu_count_through(harness) -> None:
    result, served = harness(
        gpus=4,
        HF_TOKEN="hf_fake_token_for_test",
        NPA_COSMOS3_SERVE_GPUS=4,
        NPA_COSMOS3_SERVE_EXTRA_ARGS="--tensor-parallel-size 4",
    )

    assert result.returncode == 0, result.stderr
    assert served is not None
    assert "--tensor-parallel-size 4" in served


def test_missing_nvidia_smi_fails_rather_than_starting_on_cpu(harness) -> None:
    result, served = harness(gpus=None, HF_TOKEN="hf_fake_token_for_test")

    assert result.returncode != 0
    assert served is None
    assert "nvidia-smi not found" in result.stderr


def test_an_unwritable_cache_mount_is_caught_before_the_download(harness, tmp_path) -> None:
    if os.geteuid() == 0:  # pragma: no cover - root ignores the mode bits
        pytest.skip("root can write to a read-only directory")
    locked = tmp_path / "locked-cache"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result, served = harness(gpus=8, HF_TOKEN="hf_fake_token_for_test", HF_HOME=str(locked))
    finally:
        locked.chmod(0o700)

    assert result.returncode != 0
    assert served is None
    assert "not writable" in result.stderr


def test_an_invalid_guardrail_value_is_rejected_rather_than_treated_as_off(harness) -> None:
    result, served = harness(gpus=8, NPA_COSMOS3_SERVE_GUARDRAILS="false")

    assert result.returncode != 0
    assert served is None
    assert "must be 'on' or 'off'" in result.stderr


def test_dry_run_prints_the_command_without_starting_the_server(harness) -> None:
    result, served = harness(
        gpus=None,
        NPA_COSMOS3_SERVE_GUARDRAILS="off",
        NPA_COSMOS3_SERVE_DRY_RUN=1,
        NPA_COSMOS3_SERVE_SKIP_GPU_CHECK=1,
    )

    assert result.returncode == 0, result.stderr
    assert served is None, "a dry run must not exec the server"
    assert "vllm serve nvidia/Cosmos3-Super" in result.stdout


def test_the_token_never_reaches_the_logs(harness) -> None:
    secret = "hf_do_not_log_this_value"
    result, _ = harness(gpus=8, HF_TOKEN=secret)

    assert secret not in result.stdout
    assert secret not in result.stderr
