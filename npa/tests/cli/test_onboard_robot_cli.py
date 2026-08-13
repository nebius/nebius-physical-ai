"""CLI tests for ``npa workbench sim2real onboard-robot`` (B4).

Validate-and-derive runs fully offline. The ``--smoke`` submit path is exercised
with the structured Kubernetes client monkeypatched, so no infrastructure is touched
(per repo policy). These guard that: the shipped Kinova example onboards, the
derived config is shown, an incompatible embodiment is rejected with a non-zero
exit, a malformed spec fails fast, and a failed smoke submit propagates a
non-zero exit code rather than printing a false success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workflows.sim2real.k8s_client import KubernetesJobClient
from npa.workflows.sim2real.isaac_job_payload import decode_compressed_bash_args

runner = CliRunner()

# tests/ is npa/tests; the onboarding examples live under npa/workflows/...
KINOVA_YAML = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "workbench"
    / "sim2real"
    / "onboarding"
    / "kinova-jaco2.yaml"
)


def _write_spec(tmp_path: Path, doc: str) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(doc, encoding="utf-8")
    return p


def test_onboard_kinova_example_validates_and_derives() -> None:
    """The shipped Kinova example onboards and prints its derived config."""
    result = runner.invoke(
        app, ["workbench", "sim2real", "onboard-robot", "--spec", str(KINOVA_YAML)]
    )
    assert result.exit_code == 0, result.output
    assert "kinova_j2n7s300" in result.output
    assert "action_scale" in result.output
    assert "compatible" in result.output.lower()


def test_onboard_json_output_is_machine_readable() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "onboard-robot",
            "--spec",
            str(KINOVA_YAML),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["robot"] == "kinova_j2n7s300"
    assert payload["compat"]["task_robot_compatible"] is True
    assert payload["derived"]["action_scale"] > 0


def test_onboard_invalid_spec_fails_fast(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, "schema: npa.sim2real.onboarding.v1\nrobot: {}\n")
    result = runner.invoke(
        app, ["workbench", "sim2real", "onboard-robot", "--spec", str(spec)]
    )
    assert result.exit_code == 1
    assert "validation failed" in result.output.lower()


def test_onboard_gripperless_arm_rejected(tmp_path: Path) -> None:
    """A bare arm with no gripper cannot lift — non-zero exit, clear reason."""
    spec = _write_spec(
        tmp_path,
        """
schema: npa.sim2real.onboarding.v1
robot:
  name: ur10_bare
  robot_source: byo_usd
  usd_path: https://example.com/ur10.usd
  ee_link: tool0
  base_link: base
  n_arm_joints: 6
  joint_names: [j1, j2, j3, j4, j5, j6]
  n_gripper_joints: 0
  gripper_joint_names: []
task:
  skill: lift
  success_threshold: 0.4
""",
    )
    result = runner.invoke(
        app, ["workbench", "sim2real", "onboard-robot", "--spec", str(spec)]
    )
    # Rejected with a non-zero exit and a gripper-specific reason (the gate fires
    # at spec validation; either way it must never print a green "compatible").
    assert result.exit_code == 1
    assert "gripper" in result.output.lower()
    assert (
        "compatible" not in result.output.lower()
        or "incompatible" in result.output.lower()
    )


def test_onboard_smoke_iterations_validated() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "onboard-robot",
            "--spec",
            str(KINOVA_YAML),
            "--smoke",
            "--smoke-iterations",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "smoke-iterations" in result.output.lower()


def test_onboard_smoke_requires_image(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ISAAC_IMAGE", "NPA_SIM2REAL_ISAAC_IMAGE"):
        monkeypatch.delenv(var, raising=False)
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "onboard-robot",
            "--spec",
            str(KINOVA_YAML),
            "--smoke",
        ],
    )
    assert result.exit_code == 1
    assert "isaac_image" in result.output.lower()


def _smoke_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_IMAGE", f"cr.example/npa-isaac-lab@sha256:{'a' * 64}")
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example")


def test_onboard_smoke_submits_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """--smoke reconciles a queued, immutable BYO trainer Job."""
    _smoke_env(monkeypatch)
    applied: list[dict] = []

    class _Client:
        def create_or_adopt(self, manifest, **_kwargs):
            applied.append(manifest)
            return "job-uid", False

    monkeypatch.setattr(
        KubernetesJobClient,
        "from_environment",
        classmethod(lambda _cls, **_kwargs: _Client()),
    )
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "onboard-robot",
            "--spec",
            str(KINOVA_YAML),
            "--smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "submitted" in result.output.lower()
    assert applied, "expected a structured API create with the job manifest"
    manifest = applied[0]
    assert manifest["kind"] == "Job"
    assert manifest["spec"]["suspend"] is True
    assert manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
    # The BYO-robot routing + B2-derived task config are baked into the container
    # command (the wrapper exports them in-container), not pod env. Confirm both
    # reach the job, plus the customer robot name, so the smoke job trains THIS arm.
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert "@sha256:" in container["image"]
    assert container["volumeMounts"] == [
        {
            "name": "isaac-runtime-cache",
            "mountPath": "/opt/isaac-cache",
            "readOnly": True,
        }
    ]
    script = decode_compressed_bash_args(container["args"])
    assert "NPA_BYO_ROBOT_SPEC_JSON" in script
    assert "NPA_BYO_TASK_CONFIG_JSON" in script
    assert "kinova_j2n7s300" in script


def test_onboard_smoke_apply_failure_is_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed API reconcile must exit non-zero, never print a false success."""
    _smoke_env(monkeypatch)

    class _Client:
        def create_or_adopt(self, manifest, **_kwargs):
            del manifest
            raise RuntimeError("forbidden")

    monkeypatch.setattr(
        KubernetesJobClient,
        "from_environment",
        classmethod(lambda _cls, **_kwargs: _Client()),
    )
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "onboard-robot",
            "--spec",
            str(KINOVA_YAML),
            "--smoke",
        ],
    )
    assert result.exit_code == 1
    assert "submitted" not in result.output.lower()
