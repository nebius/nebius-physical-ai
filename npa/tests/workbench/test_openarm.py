"""OpenArm CLI, SDK, service, workflow, and packaging contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import argv_for_tool
from npa.sdk.workbench import openarm as sdk
from npa.workbench.openarm.runtime import (
    OpenArmError,
    _all_finite,
    _render_video,
    _step_mujoco,
    _validate_video,
    _validate_qualification_tree,
)
from npa.workbench.openarm.schemas import OpenArmRunRequest, OpenArmStatusResponse
from npa.workbench.openarm.service import RunRegistry, create_app

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/openarm-simulators.yaml"


def test_cli_registered() -> None:
    result = CliRunner().invoke(app, ["workbench", "openarm", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("run", "qualify", "deploy", "delete", "status", "system-info"):
        assert command in result.output


def test_light_image_cli_imports_only_openarm_sdk() -> None:
    env = dict(os.environ)
    env.update(NPA_SKIP_EAGER_IMPORTS="1", NPA_LIGHT_WORKBENCH_TOOL="openarm")
    probe = """
import importlib.abc
import sys

class BlockFullCli(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "npa.cli.main":
            raise RuntimeError("full platform CLI import attempted")
        return None

sys.meta_path.insert(0, BlockFullCli())
sys.argv = ["npa", "workbench", "openarm", "--help"]
from npa.cli.entry import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Enactic OpenArm" in result.stdout


def test_request_rejects_non_s3_output() -> None:
    with pytest.raises(ValueError, match="S3"):
        OpenArmRunRequest(simulator="mujoco", output_uri="file:///local/output")


def test_mujoco_rollout_commands_only_bimanual_actuators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        *(f"right_joint{index}_ctrl" for index in range(1, 8)),
        "right_finger1_ctrl",
        *(f"left_joint{index}_ctrl" for index in range(1, 8)),
        "left_finger1_ctrl",
    ]
    actuator_ids = {name: index + 1 for index, name in enumerate(names)}
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_ACTUATOR=object()),
        mj_name2id=lambda _model, _kind, name: actuator_ids.get(name, -1),
        mj_step=lambda _model, data: setattr(data, "time", data.time + 0.001),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    model = SimpleNamespace(
        actuator_ctrlrange=np.asarray([[-1.0, 1.0]] * 17),
    )
    data = SimpleNamespace(
        ctrl=np.zeros(17, dtype=float),
        qpos=np.zeros(16, dtype=float),
        qvel=np.zeros(16, dtype=float),
        time=0.0,
    )

    class Resolver:
        def set_ctrl(self, ctrl, values, segment):
            selected = range(1, 9) if segment == "right" else range(9, 17)
            ctrl[list(selected)] = values

        def get_driver(self, _qpos, _segment):
            return np.zeros(7), 0.0

    commands, _samples, _energies = _step_mujoco(model, data, Resolver(), 3)

    assert np.asarray(commands).shape == (3, 16)
    assert data.ctrl[0] == 0.0
    assert np.count_nonzero(data.ctrl[1:]) > 0


def test_mujoco_renderer_replays_only_bimanual_actuators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = [
        *(f"right_joint{index}_ctrl" for index in range(1, 8)),
        "right_finger1_ctrl",
        *(f"left_joint{index}_ctrl" for index in range(1, 8)),
        "left_finger1_ctrl",
    ]
    actuator_ids = {name: index + 1 for index, name in enumerate(names)}

    class Renderer:
        def __init__(self, _model, *, height, width):
            assert (height, width) == (480, 640)

        def update_scene(self, _data):
            return None

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            return None

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.mjtObj = SimpleNamespace(mjOBJ_ACTUATOR=object())
    fake_mujoco.mj_name2id = (
        lambda _model, _kind, name: actuator_ids.get(name, -1)
    )
    fake_mujoco.Renderer = Renderer
    fake_mujoco.mj_resetData = lambda _model, data: data.ctrl.fill(0.0)
    fake_mujoco.mj_step = lambda _model, _data: None
    fake_imageio = ModuleType("imageio.v2")
    written = {}
    fake_imageio.mimwrite = lambda path, frames, **kwargs: written.update(
        path=path, frames=frames, kwargs=kwargs
    )
    fake_imageio_package = ModuleType("imageio")
    fake_imageio_package.v2 = fake_imageio
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "imageio", fake_imageio_package)
    monkeypatch.setitem(sys.modules, "imageio.v2", fake_imageio)
    model = SimpleNamespace(actuator_ctrlrange=np.asarray([[-1.0, 1.0]] * 17))
    data = SimpleNamespace(ctrl=np.zeros(17, dtype=float))

    _render_video(model, data, [np.ones(16)], tmp_path / "rollout.mp4")

    assert data.ctrl[0] == 0.0
    assert np.array_equal(data.ctrl[1:], np.ones(16))
    assert len(written["frames"]) == 1


def test_sdk_service_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_request(method, endpoint, path, **kwargs):
        captured.update(method=method, endpoint=endpoint, path=path, **kwargs)
        return {
            "run_id": "run-1",
            "status": "running",
            "simulator": "isaac-lab",
            "output_uri": "s3://bucket/openarm/",
            "manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr(sdk, "_request", fake_request)
    response = sdk.run(
        simulator="isaac-lab",
        output_path="s3://bucket/openarm/",
        mode="service",
        endpoint="https://openarm.example",
        task="Isaac-Reach-OpenArm-v0",
    )
    assert response.run_id == "run-1"
    assert captured["payload"]["simulator"] == "isaac-lab"
    assert captured["payload"]["task"] == "Isaac-Reach-OpenArm-v0"


def test_service_auth_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARM_AUTH_MODE", "token")
    monkeypatch.setenv("OPENARM_TOKEN", "test-token")
    registry = RunRegistry()
    registry.put(
        OpenArmStatusResponse(
            run_id="known",
            status="completed",
            simulator="mujoco",
            output_uri="s3://bucket/result/",
            result={"ok": True},
        )
    )
    client = TestClient(create_app(registry=registry))
    assert client.get("/health").status_code == 200
    assert client.get("/runs").status_code == 401
    response = client.get(
        "/status",
        params={"run_id": "known"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    assert response.json()["result"] == {"ok": True}


def test_service_registry_claims_one_concurrent_execution() -> None:
    registry = RunRegistry()
    pending = OpenArmStatusResponse(
        run_id="same-run",
        status="running",
        simulator="mujoco",
        output_uri="s3://bucket/result/",
    )

    with ThreadPoolExecutor(max_workers=16) as executor:
        claims = list(executor.map(lambda _index: registry.claim(pending)[1], range(64)))

    assert claims.count(True) == 1
    assert registry.get("same-run") == pending


def test_workflow_and_toolrefs_are_real_and_routed() -> None:
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    plan = build_plan(spec, run_id="test")
    assert [step.state for step in plan.steps] == [
        "mujoco-rollout",
        "isaac-rollout",
        "isaac-training",
        "qualify-artifacts",
    ]
    for ref in (
        "workbench.openarm.mujoco_rollout",
        "workbench.openarm.isaac_rollout",
        "workbench.openarm.isaac_train",
    ):
        argv = argv_for_tool(ref)
        assert argv[:4] == ["npa", "workbench", "openarm", "run"]
        assert "--output-path" in argv
    assert argv_for_tool("workbench.openarm.qualify")[:4] == [
        "npa",
        "workbench",
        "openarm",
        "qualify",
    ]
    assert "--render" in argv_for_tool("workbench.openarm.mujoco_rollout")
    assert "--max-iterations" in argv_for_tool("workbench.openarm.isaac_train")
    assert "--input-path" in argv_for_tool("workbench.openarm.qualify")


def test_qualification_validates_real_artifact_tree(tmp_path: Path) -> None:
    stages = {
        "mujoco": ("npa.openarm.mujoco_rollout.v1", "trace.npz"),
        "isaac-rollout": ("npa.openarm.isaac_lab_rollout.v1", "rollout.npz"),
    }
    for stage, (schema, artifact) in stages.items():
        root = tmp_path / stage
        root.mkdir()
        arrays = (
            {
                "joint_position": np.ones((2, 16)),
                "command": np.ones((3, 16)),
                "velocity_energy": np.ones(2),
            }
            if stage == "mujoco"
            else {"reward": np.ones(3), "policy_observation": np.ones((3, 8))}
        )
        np.savez(root / artifact, **arrays)
        artifact_path = root / artifact
        (root / "result.json").write_text(
            json.dumps(
                {
                    "schema": schema,
                    "status": "completed",
                    "finite_metrics": True,
                    "artifact": {
                        "path": artifact,
                        "bytes": artifact_path.stat().st_size,
                    },
                }
            ),
            encoding="utf-8",
        )
    training = tmp_path / "isaac-training"
    training.mkdir()
    with zipfile.ZipFile(training / "model_0.pt", "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"serialized-state")
    (training / "result.json").write_text(
        json.dumps(
            {
                "schema": "npa.openarm.isaac_lab_training.v1",
                "status": "completed",
                "finite_metrics": True,
                "checkpoints": ["model_0.pt"],
            }
        ),
        encoding="utf-8",
    )
    artifacts = _validate_qualification_tree(tmp_path)
    assert {row["stage"] for row in artifacts} == {
        "mujoco",
        "isaac-rollout",
        "isaac-training",
    }
    assert all(len(row["sha256"]) == 64 for row in artifacts)


def test_qualification_rejects_nonfinite_npz_despite_result_claim(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "mujoco"
    stage.mkdir()
    np.savez(
        stage / "trace.npz",
        joint_position=np.asarray([[np.nan]]),
        command=np.ones((1, 16)),
        velocity_energy=np.ones(1),
    )
    (stage / "result.json").write_text(
        json.dumps(
            {
                "schema": "npa.openarm.mujoco_rollout.v1",
                "status": "completed",
                "finite_metrics": True,
                "artifact": {"path": "trace.npz"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OpenArmError, match="invalid joint_position values"):
        _validate_qualification_tree(tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        {"nested": {"metric": float("nan")}},
        {"nested": [1.0, (2.0, float("inf"))]},
        np.asarray([[1.0, -np.inf]]),
        np.asarray([{"metric": np.float32("nan")}], dtype=object),
    ],
)
def test_finite_metrics_rejects_nested_nonfinite_values(value: object) -> None:
    assert _all_finite(value) is False


def test_missing_ffprobe_is_clean_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failure = FileNotFoundError("ffprobe executable is unavailable")
    monkeypatch.setattr(
        "npa.workbench.openarm.runtime.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(OpenArmError, match="cannot probe MuJoCo video") as caught:
        _validate_video(tmp_path / "rollout.mp4")

    assert caught.value.__cause__ is failure


def test_qualify_cli_reports_missing_ffprobe_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_qualification(**_kwargs: object) -> None:
        try:
            raise FileNotFoundError("ffprobe executable is unavailable")
        except OSError as exc:
            raise OpenArmError("cannot probe MuJoCo video") from exc

    monkeypatch.setattr(sdk, "qualify", fail_qualification)

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "openarm",
            "qualify",
            "--input-path",
            "s3://input/root/",
            "--output-path",
            "s3://output/root/",
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == "cannot probe MuJoCo video"
    assert "Traceback" not in result.output


def test_packaging_pins_and_excludes_isaac_payload() -> None:
    dockerfile = (ROOT / "npa/docker/workbench/openarm/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "a8c979629f2591ad035d99d338ce114969e6cddc" in dockerfile
    assert "bad82e23716e6941c2de78ccb978f57c78b37734" in dockerfile
    assert "install_isaac_runtime_base.sh" in dockerfile
    assert "isaac-bootstrap ensure" not in dockerfile
    assert "nvcr.io/nvidia/isaac" not in dockerfile
    assert "--require-hashes" in dockerfile
    assert 'm.version("GitPython") == "3.1.62"' in dockerfile
    assert 'm.version("fastapi") == "0.136.1"' in dockerfile
    assert 'm.version("starlette") == "1.6.0"' in dockerfile
    assert '"serve", "--host", "0.0.0.0"' in dockerfile
    assert "OPENARM_GNUPG_VERSION=2.2.27-3ubuntu2.5" in dockerfile
    assert "OPENARM_OPENSSL_VERSION=3.0.2-0ubuntu1.26" in dockerfile
    assert "OPENARM_LINUX_LIBC_DEV_VERSION=5.15.0-190.200" in dockerfile
    assert '"gnupg2=${OPENARM_GNUPG_VERSION}"' in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile
    service_lock = (
        ROOT / "npa/docker/workbench/openarm/mujoco-requirements.txt"
    ).read_text(encoding="utf-8")
    assert "starlette==1.6.0" in service_lock
    assert "starlette==0.45.3" not in service_lock
    security_lock = (
        ROOT / "npa/docker/workbench/openarm/security-requirements.txt"
    ).read_text(encoding="utf-8")
    assert "gitpython==3.1.62" in security_lock
    assert "gitpython==3.1.57" not in security_lock
    assert (ROOT / "npa/docker/workbench/openarm/THIRD_PARTY_NOTICES.md").is_file()
    assert "prune_python_vendor_devel.py" in dockerfile
    assert (ROOT / "npa/docker/workbench/openarm/ONBOARDING_CONTRACT.md").is_file()
    components = json.loads(
        (ROOT / "npa/docker/workbench/openarm/components.json").read_text(
            encoding="utf-8"
        )
    )
    baked = {row["name"] for row in components["components"] if row["baked"]}
    assert "enactic/openarm_mujoco" in baked
    assert "enactic/openarm_isaac_lab" in baked
    assert "NVIDIA Isaac Sim and Isaac Lab" not in baked
    sources = {row["name"]: row for row in components["components"]}
    assert len(sources["enactic/openarm_mujoco"]["license_sha256"]) == 64
    assert len(sources["enactic/openarm_isaac_lab"]["license_sha256"]) == 64
    assert sources["OpenArm HTTP service stack"]["version"] == (
        "FastAPI 0.136.1 / Starlette 1.6.0 / Uvicorn 0.53.0"
    )
    assert "docker/workbench/openarm/security-requirements.txt" in components[
        "transitive_inventory"
    ]["python_lock"]


def test_serve_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: observed.update(app=app, **kwargs)
    )

    result = CliRunner().invoke(app, ["workbench", "openarm", "serve"])

    assert result.exit_code == 0, result.output
    assert observed == {
        "app": "npa.workbench.openarm.service:app",
        "host": "127.0.0.1",
        "port": 8792,
    }


def test_deploy_dry_run_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARM_TOKEN", "do-not-print")
    monkeypatch.setattr(
        "npa.cli.workbench.openarm.load_credentials", lambda: type("C", (), {})()
    )
    monkeypatch.setattr(
        "npa.cli.workbench.openarm.apply_shared_credential_env",
        lambda env, credentials: None,
    )
    result = CliRunner().invoke(
        app,
        ["workbench", "openarm", "deploy", "--dry-run", "--output-format", "json"],
    )
    assert result.exit_code == 0, result.output
    assert "do-not-print" not in result.output
    payload = json.loads(result.output)
    assert payload["items"][0]["data"]["OPENARM_TOKEN"] == "<redacted>"
    pod_spec = payload["items"][1]["spec"]["template"]["spec"]
    assert "args" not in pod_spec["containers"][0]
    assert pod_spec["nodeSelector"] == {
        "node.kubernetes.io/instance-type": "gpu-rtx6000"
    }
