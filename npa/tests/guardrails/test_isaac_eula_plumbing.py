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
def test_renderer_fails_closed_and_forwards_only_scoped_consent(tool_ref: str) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs(tool_ref) == {EULA_ENV: ""}
    assert isaac_eula_envs(tool_ref, accepted=True) == {EULA_ENV: "Y"}


def test_renderer_does_not_add_consent_to_non_isaac_tasks() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs("workbench.lancedb.serve", accepted=True) == {}


def test_serverless_forwarder_never_invents_acceptance(monkeypatch) -> None:
    from npa.serverless_common.env import build_serverless_job_env, isaac_eula_env

    monkeypatch.delenv(EULA_ENV, raising=False)
    assert isaac_eula_env() == {}
    assert EULA_ENV not in build_serverless_job_env(output_path="s3://bucket/run/")

    monkeypatch.setenv(EULA_ENV, "Y")
    assert isaac_eula_env() == {EULA_ENV: "Y"}
    assert build_serverless_job_env(output_path="s3://bucket/run/")[EULA_ENV] == "Y"


@pytest.mark.parametrize("value", ["", "YES", "yes", "true", "1", "no"])
def test_preflight_accepts_only_nvidias_documented_exact_value(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from npa.serverless_common.env import (
        MissingIsaacEulaAcceptanceError,
        require_isaac_eula_acceptance,
    )

    monkeypatch.setenv(EULA_ENV, value)
    with pytest.raises(MissingIsaacEulaAcceptanceError, match="ACCEPT_EULA=Y"):
        require_isaac_eula_acceptance(context="test", resume_command="npa test")


def test_preflight_accepts_exact_y(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.serverless_common.env import require_isaac_eula_acceptance

    monkeypatch.setenv(EULA_ENV, "Y")
    require_isaac_eula_acceptance(context="test", resume_command="npa test")


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
def test_child_isaac_jobs_do_not_invent_acceptance(
    monkeypatch, module_name, builder
) -> None:
    monkeypatch.delenv(EULA_ENV, raising=False)
    assert EULA_ENV not in _job_env(module_name, builder)


def test_no_user_facing_legacy_consent_or_privacy_defaults() -> None:
    forbidden = (
        "sonic_accept_nvidia_eula",
        "--accept-nvidia-eula",
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
