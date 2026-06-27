"""CLI tests for ``npa workbench sim2real onboard-robot`` (B4).

Validate-and-derive runs fully offline. The ``--smoke`` submit path is exercised
with the cluster kubectl call monkeypatched, so no real infrastructure is touched
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
from npa.workflows.sim2real import byo_isaac_trainer as trainer

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
        ["workbench", "sim2real", "onboard-robot", "--spec", str(KINOVA_YAML), "--json"],
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
    assert "compatible" not in result.output.lower() or "incompatible" in result.output.lower()


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
        ["workbench", "sim2real", "onboard-robot", "--spec", str(KINOVA_YAML), "--smoke"],
    )
    assert result.exit_code == 1
    assert "isaac_image" in result.output.lower()


def _smoke_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_IMAGE", "cr.example/npa-isaac-lab:test")
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example")


def test_onboard_smoke_submits_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """--smoke builds + applies a BYO trainer job; success => exit 0."""
    _smoke_env(monkeypatch)
    applied: list[dict] = []

    class _OK:
        returncode = 0
        stderr = ""

    def _fake_kubectl(args, *, stdin=None, timeout=300):
        if args[:1] == ["apply"]:
            applied.append(json.loads(stdin))
        return _OK()

    monkeypatch.setattr(trainer, "_kubectl", _fake_kubectl)
    result = runner.invoke(
        app,
        ["workbench", "sim2real", "onboard-robot", "--spec", str(KINOVA_YAML), "--smoke"],
    )
    assert result.exit_code == 0, result.output
    assert "submitted" in result.output.lower()
    assert applied, "expected a kubectl apply with the job manifest"
    manifest = applied[0]
    assert manifest["kind"] == "Job"
    # The BYO-robot routing + B2-derived task config are baked into the container
    # command (the wrapper exports them in-container), not pod env. Confirm both
    # reach the job, plus the customer robot name, so the smoke job trains THIS arm.
    blob = json.dumps(manifest)
    assert "NPA_BYO_ROBOT_SPEC_JSON" in blob
    assert "NPA_BYO_TASK_CONFIG_JSON" in blob
    assert "kinova_j2n7s300" in blob


def test_onboard_smoke_apply_failure_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed kubectl apply must exit non-zero, never print a false success."""
    _smoke_env(monkeypatch)

    class _Fail:
        returncode = 1
        stderr = "forbidden"

    monkeypatch.setattr(trainer, "_kubectl", lambda *a, **k: _Fail())
    result = runner.invoke(
        app,
        ["workbench", "sim2real", "onboard-robot", "--spec", str(KINOVA_YAML), "--smoke"],
    )
    assert result.exit_code == 1
    assert "submitted" not in result.output.lower()
