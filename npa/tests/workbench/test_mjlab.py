"""Verify shared MJLab requests, safe execution, publication and API boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from npa.sdk.workbench import mjlab as sdk
from npa.workbench.mjlab import runtime
from npa.workbench.mjlab.client import invoke
from npa.workbench.mjlab.deployment import DeployRequest, deploy
from npa.workbench.mjlab.schemas import (
    EvalRequest,
    ExportRequest,
    MjlabError,
    TrainRequest,
)
from npa.workbench.mjlab.service import create_app
from npa.workbench.storage_scope import StorageScope, use_storage_scope

OUTPUT = "s3://fixture/runs/mjlab"
CHECKPOINT = "s3://fixture/inputs/model.pt"


@pytest.mark.parametrize(
    "fields",
    [
        {"iterations": 0},
        {"num_envs": -1},
        {"gpu_count": 0},
        {"learning_rate": float("nan")},
        {"output_path": "/tmp/output"},
        {"output_path": "https://example.org/a"},
        {"output_path": "s3://fixture/a/../b"},
        {"output_path": "s3://fixture/a?secret=x"},
        {"input_path": "s3://fixture/retargeted/"},
        {"task": "module:callable"},
        {"task": "Mjlab-Tracking-Flat-Unitree-G1"},
        {"score": 0.9},
    ],
)
def test_invalid_requests_fail_before_execution(fields):
    with pytest.raises(ValueError):
        TrainRequest(**{"output_path": OUTPUT, **fields})


def test_tracking_accepts_native_motion_npz():
    request = TrainRequest(
        task="Mjlab-Tracking-Flat-Unitree-G1",
        input_path="s3://fixture/motion.npz",
        output_path=OUTPUT,
    )
    assert request.input_path.endswith(".npz")


@pytest.mark.parametrize(
    "operation,model",
    [("train", TrainRequest), ("eval", EvalRequest), ("export", ExportRequest)],
)
def test_sdk_calls_the_same_implementation(monkeypatch, operation, model):
    request = model(
        output_path=OUTPUT,
        **({"checkpoint": CHECKPOINT} if operation != "train" else {}),
    )
    captured = []
    name = "evaluate" if operation == "eval" else operation
    monkeypatch.setattr(
        runtime,
        name,
        lambda value, **kwargs: captured.append(value) or {"operation": operation},
    )
    assert getattr(sdk, operation)(request) == {"operation": operation}
    assert captured == [request]


def test_worker_forces_safe_loading_and_disables_wandb(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0),
    )
    runtime._worker("train", {}, tmp_path)
    assert calls[0][1]["env"]["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in calls[0][1]["env"]
    assert calls[0][1]["env"]["WANDB_MODE"] == "disabled"
    assert calls[0][0][1:3] == ["-m", "npa.workbench.mjlab.worker"]


def test_worker_failure_never_publishes(monkeypatch):
    monkeypatch.setattr(runtime, "system_info", lambda: {"installed": True})
    client = SimpleNamespace(upload_file=lambda *a: pytest.fail("published failed run"))
    monkeypatch.setattr(
        runtime.StorageClient, "from_environment", lambda **kwargs: client
    )
    monkeypatch.setattr(
        runtime, "_worker", lambda *a: (_ for _ in ()).throw(MjlabError("failed"))
    )
    with pytest.raises(MjlabError, match="failed"):
        runtime.train(TrainRequest(output_path=OUTPUT))


def test_publish_manifest_is_last_and_hashes_actual_bytes(monkeypatch, tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/checkpoint.pt").write_bytes(b"native-checkpoint-test")
    uploaded = []
    client = SimpleNamespace(
        upload_file=lambda path, uri: uploaded.append((uri, Path(path).read_bytes()))
    )
    report = runtime._publish(
        client,
        "train",
        TrainRequest(output_path=OUTPUT),
        tmp_path,
        {"status": "completed"},
        {},
    )
    assert uploaded[-1][0] == OUTPUT + "/mjlab_train.json"
    assert json.loads(uploaded[-1][1]) == report
    import hashlib

    assert (
        report["artifacts"]["checkpoint.pt"]["sha256"]
        == hashlib.sha256(b"native-checkpoint-test").hexdigest()
    )


def test_no_outputs_cannot_be_success(tmp_path):
    with pytest.raises(MjlabError, match="no artifacts"):
        runtime._publish(
            None, "train", TrainRequest(output_path=OUTPUT), tmp_path, {}, {}
        )


def test_service_rechecks_scope_after_model_creation(monkeypatch):
    request = TrainRequest(output_path="s3://forbidden/out")
    monkeypatch.setattr(
        runtime.StorageClient,
        "from_environment",
        lambda **kwargs: pytest.fail("unauthorized I/O"),
    )
    with use_storage_scope(StorageScope.from_config(s3_roots=[OUTPUT])):
        with pytest.raises(ValueError, match="outside"):
            runtime.train(request)


@pytest.mark.parametrize("route", ["/train", "/eval", "/export"])
def test_service_requires_authentication(route):
    client = TestClient(create_app(token="test-token", allowed_s3_roots=[OUTPUT]))
    response = client.post(
        route, json={"output_path": OUTPUT, "checkpoint": CHECKPOINT}
    )
    assert response.status_code == 401


def test_service_missing_token_fails_closed():
    client = TestClient(create_app(token="", allowed_s3_roots=[OUTPUT]))
    assert client.get("/health").status_code == 200
    assert client.get("/status").status_code == 503


@pytest.mark.parametrize(
    "route,model",
    [("train", TrainRequest), ("eval", EvalRequest), ("export", ExportRequest)],
)
def test_service_calls_shared_runtime_and_reports_busy(monkeypatch, route, model):
    application = create_app(token="test-token", allowed_s3_roots=[OUTPUT, CHECKPOINT])
    client = TestClient(application)
    request = model(
        output_path=OUTPUT, **({"checkpoint": CHECKPOINT} if route != "train" else {})
    )
    name = "evaluate" if route == "eval" else route
    monkeypatch.setattr(runtime, name, lambda req: {"task": req.task})
    headers = {"Authorization": "Bearer test-token"}
    response = client.post("/" + route, json=request.model_dump(), headers=headers)
    assert response.status_code == 200
    assert response.json()["task"] == request.task
    application.state.mjlab.lock.acquire()
    try:
        assert (
            client.post(
                "/" + route, json=request.model_dump(), headers=headers
            ).status_code
            == 409
        )
    finally:
        application.state.mjlab.lock.release()


def test_service_rejects_foreign_storage_before_worker(monkeypatch):
    client = TestClient(create_app(token="test-token", allowed_s3_roots=[OUTPUT]))
    monkeypatch.setattr(
        runtime, "_worker", lambda *a: pytest.fail("unauthorized worker")
    )
    response = client.post(
        "/train",
        json={"output_path": "s3://foreign/out"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://remote.example",
        "https://user:pass@example.com",
        "https://example.com?token=x",
    ],
)
def test_client_rejects_unsafe_transport(endpoint):
    with pytest.raises(MjlabError, match="HTTPS"):
        invoke("status", endpoint=endpoint)


def test_client_no_redirect_or_operation_timeout(monkeypatch):
    calls = []
    monkeypatch.setenv("MJLAB_TOKEN", "test-token")
    monkeypatch.setattr(
        "npa.workbench.mjlab.client.httpx.request",
        lambda *a, **kw: (
            calls.append((a, kw))
            or SimpleNamespace(status_code=200, json=lambda: {"busy": False})
        ),
    )
    assert invoke("status", endpoint="http://localhost:8080") == {"busy": False}
    assert calls[0][1]["follow_redirects"] is False
    assert calls[0][1]["timeout"] is None


def test_deploy_renders_private_service_without_credentials(monkeypatch):
    monkeypatch.setattr(
        "npa.workbench.mjlab.deployment.subprocess.run",
        lambda *a, **kw: pytest.fail("dry-run applied"),
    )
    request = DeployRequest(
        image="registry.example/mjlab:dev",
        token_secret="mjlab-auth",
        storage_secret="storage",
        allowed_s3_roots=OUTPUT,
        accelerator="test-gpu",
    )
    result = deploy(request, dry_run=True)
    deployment, service = result["manifest"]["items"]
    assert service["spec"]["type"] == "ClusterIP"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["env"][0]["valueFrom"]["secretKeyRef"]["name"] == "mjlab-auth"


def test_episode_quotas_do_not_favor_fast_failures(tmp_path):
    torch = pytest.importorskip("torch")
    from npa.workbench.mjlab.rollout import measure_episodes

    class Environment:
        num_envs = 2
        max_episode_length = 2
        steps = 0

        def __init__(self):
            self.unwrapped = self

        def get_observations(self):
            return torch.zeros((2, 1))

        def step(self, actions):
            self.steps += 1
            self.reset_terminated = torch.tensor([True, False])
            self.reset_time_outs = torch.tensor([False, self.steps % 2 == 0])
            return (
                self.get_observations(),
                torch.tensor([1.0, 2.0]),
                torch.tensor([True, self.steps % 2 == 0]),
                {},
            )

    env = Environment()
    request = EvalRequest(
        checkpoint=CHECKPOINT, output_path=OUTPUT, num_envs=2, episodes=3
    )
    report = measure_episodes(env, lambda obs: torch.zeros((2, 1)), request, tmp_path)
    assert report["episodes_completed"] == 3
    assert report["score"] == pytest.approx(1 / 3)
    assert [row["return"] for row in report["episodes"]] == [1.0, 1.0, 4.0]
    assert [row["environment"] for row in report["episodes"]] == [0, 0, 1]


def test_rollout_rejects_nonfinite_rewards():
    from npa.workbench.mjlab.rollout import _record_step

    with pytest.raises(MjlabError, match="non-finite"):
        _record_step(
            SimpleNamespace(num_envs=1), [float("nan")], [False], [1], [0], [0], [0], []
        )


def test_rollout_rejects_missing_horizon_termination():
    from npa.workbench.mjlab.rollout import _record_step

    with pytest.raises(MjlabError, match="horizon"):
        _record_step(
            SimpleNamespace(num_envs=1, max_episode_length=2),
            [1.0],
            [False],
            [1],
            [0],
            [0],
            [2],
            [],
        )
