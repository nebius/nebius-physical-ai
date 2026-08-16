"""Guard the single, run-scoped NVIDIA Isaac consent mechanism."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EULA_ENV = "ACCEPT_EULA"


def _isaac_tool_refs() -> tuple[str, ...]:
    from npa.orchestration.npa_workflow.skypilot_render import (
        ISAAC_IMAGE_TOOLS,
        TOOL_REF_IMAGE_TOOL,
    )

    return tuple(
        prefix
        for prefix, tool in TOOL_REF_IMAGE_TOOL.items()
        if tool in ISAAC_IMAGE_TOOLS
    )


@pytest.mark.parametrize("tool_ref", _isaac_tool_refs())
def test_renderer_defaults_acceptance_and_preserves_explicit_opt_out(tool_ref: str) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs(tool_ref) == {EULA_ENV: "Y"}
    assert isaac_eula_envs(tool_ref, accepted=True) == {EULA_ENV: "Y"}
    assert isaac_eula_envs(tool_ref, accepted=False) == {EULA_ENV: ""}


def test_renderer_does_not_add_consent_to_non_isaac_tasks() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import (
        isaac_eula_envs,
        tool_image_key,
    )

    assert isaac_eula_envs("workbench.lancedb.serve", accepted=True) == {}
    # Generic BYOF chooses its workload image from config.base_image in the
    # inner runner.  Treating the whole family as Isaac both selected the wrong
    # outer image and invented an Isaac consent requirement for public Wan.
    assert tool_image_key("workbench.byof.repo") is None
    assert isaac_eula_envs("workbench.byof.repo", accepted=True) == {}


def test_renderer_detects_isaac_identity_in_byof_base_config() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs(
        "workbench.byof.repo",
        config={"base_profile": "isaac-lab", "base_image": "tool://isaac-lab"},
    ) == {EULA_ENV: "Y"}
    assert isaac_eula_envs(
        "workbench.byof.repo",
        config={"base_profile": "ubuntu", "base_image": "ubuntu:22.04"},
    ) == {}


def test_renderer_gates_only_groot_isaac_simulation() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs("workbench.groot.finetune", config={}) == {}
    assert isaac_eula_envs(
        "workbench.groot.eval", config={"sim_backend": "isaac"}, accepted=False
    ) == {EULA_ENV: ""}


def test_serverless_forwarder_never_invents_acceptance(monkeypatch) -> None:
    from npa.serverless_common.env import build_serverless_job_env, isaac_eula_env

    monkeypatch.delenv(EULA_ENV, raising=False)
    assert isaac_eula_env() == {}
    assert EULA_ENV not in build_serverless_job_env(output_path="s3://bucket/run/")

    monkeypatch.setenv(EULA_ENV, "Y")
    assert isaac_eula_env() == {EULA_ENV: "Y"}
    assert EULA_ENV not in build_serverless_job_env(output_path="s3://bucket/run/")
    assert build_serverless_job_env(
        output_path="s3://bucket/run/", extra_env=isaac_eula_env()
    )[EULA_ENV] == "Y"


@pytest.mark.parametrize("value", ["Y", "YES", "yes", "true", "1"])
def test_preflight_migrates_legacy_affirmative_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from npa.serverless_common.env import require_isaac_eula_acceptance

    monkeypatch.setenv(EULA_ENV, value)
    assert require_isaac_eula_acceptance(
        context="test", resume_command="npa test"
    ) == "Y"


@pytest.mark.parametrize("value", ["", "N", "no", "false", "0"])
def test_preflight_preserves_recognized_opt_out(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from npa.serverless_common.env import (
        MissingIsaacEulaAcceptanceError,
        require_isaac_eula_acceptance,
    )

    monkeypatch.setenv(EULA_ENV, value)
    with pytest.raises(MissingIsaacEulaAcceptanceError, match="ACCEPT_EULA=Y"):
        require_isaac_eula_acceptance(context="test", resume_command="npa test")


def test_preflight_rejects_unrecognized_value_distinctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.serverless_common.env import (
        InvalidIsaacEulaValueError,
        require_isaac_eula_acceptance,
    )

    monkeypatch.setenv(EULA_ENV, "maybe")
    with pytest.raises(InvalidIsaacEulaValueError, match="Invalid ACCEPT_EULA"):
        require_isaac_eula_acceptance(context="test", resume_command="npa test")


def test_preflight_unset_default_is_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.serverless_common.env import require_isaac_eula_acceptance

    monkeypatch.delenv(EULA_ENV, raising=False)
    assert require_isaac_eula_acceptance(
        context="test", resume_command="npa test"
    ) == "Y"
    assert EULA_ENV not in __import__("os").environ


SIM2REAL_ISAAC_BUILDERS = (
    ("byo_isaac_eval", "build_isaac_eval_job_manifest"),
    ("byo_isaac_trainer", "build_isaac_job_manifest"),
    ("byo_isaac_policy_rollout", "build_isaac_rollout_job_manifest"),
)

_MANIFEST_KWARGS = {
    "byo_isaac_eval": dict(
        job_name="j",
        run_id="r",
        image="reg/isaac@sha256:" + "a" * 64,
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4,
        checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/p.json",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
    "byo_isaac_trainer": dict(
        job_name="j",
        run_id="r",
        image="reg/isaac@sha256:" + "a" * 64,
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4,
        iterations=1,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
    "byo_isaac_policy_rollout": dict(
        job_name="j",
        run_id="r",
        image="reg/isaac@sha256:" + "a" * 64,
        task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=2,
        steps_per_rollout=4,
        checkpoint_uri="s3://b/m.pt",
        out_s3_prefix="s3://b/o",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
}


def _job_env(module_name: str, builder_name: str) -> dict[str, str]:
    module = importlib.import_module(f"npa.workflows.sim2real.{module_name}")
    manifest = getattr(module, builder_name)(**_MANIFEST_KWARGS[module_name])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return {entry["name"]: entry["value"] for entry in container.get("env", [])}


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
def test_child_isaac_jobs_inherit_authorized_operation(
    monkeypatch, module_name, builder
) -> None:
    monkeypatch.setenv(EULA_ENV, "Y")
    assert _job_env(module_name, builder)[EULA_ENV] == "Y"


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
def test_child_isaac_jobs_apply_default_acceptance(
    monkeypatch, module_name, builder
) -> None:
    monkeypatch.delenv(EULA_ENV, raising=False)
    assert _job_env(module_name, builder)[EULA_ENV] == "Y"


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
@pytest.mark.parametrize("value", ["yes", "TRUE", "1"])
def test_child_isaac_jobs_normalize_affirmative_values(
    monkeypatch, module_name, builder, value
) -> None:
    monkeypatch.setenv(EULA_ENV, value)
    assert _job_env(module_name, builder)[EULA_ENV] == "Y"


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
def test_child_isaac_jobs_preserve_explicit_empty_opt_out(
    monkeypatch, module_name, builder
) -> None:
    monkeypatch.setenv(EULA_ENV, "")
    assert _job_env(module_name, builder)[EULA_ENV] == ""


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
def test_child_isaac_jobs_reject_invalid_value_before_manifest(
    monkeypatch, module_name, builder
) -> None:
    from npa.serverless_common.env import InvalidIsaacEulaValueError

    monkeypatch.setenv(EULA_ENV, "maybe")
    with pytest.raises(InvalidIsaacEulaValueError, match="Invalid ACCEPT_EULA"):
        _job_env(module_name, builder)


def test_no_user_facing_legacy_consent_or_privacy_defaults() -> None:
    forbidden = (
        "sonic_accept_nvidia_eula",
        '"PRIVACY_CONSENT": "Y"',
        "PRIVACY_CONSENT=Y",
    )
    roots = (REPO_ROOT / "npa" / "src", REPO_ROOT / "npa" / "workflows")
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                assert marker not in text, (
                    f"{path.relative_to(REPO_ROOT)} contains {marker}"
                )
