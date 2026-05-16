from __future__ import annotations

import base64
import contextlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeDevice:
    def __init__(self, value: str = "cpu") -> None:
        self.type = value

    def __str__(self) -> str:
        return self.type


class FakeTensor:
    pass


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None


@pytest.fixture()
def server_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        device=FakeDevice,
        Tensor=FakeTensor,
        cuda=FakeCuda(),
        inference_mode=contextlib.nullcontext,
        autocast=lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("NPA_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("NPA_JOB_STATUS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("NPA_LOG_DIR", str(tmp_path / "logs"))
    sys.modules.pop("npa.server.app", None)
    module = importlib.import_module("npa.server.app")
    yield module
    sys.modules.pop("npa.server.app", None)


def test_parse_observation_converts_supported_shapes(server_module) -> None:
    encoded = base64.b64encode(bytes([1, 2, 3])).decode()

    observation = server_module._parse_observation(
        {
            "image_b64": encoded,
            "image_list": [[[0, 255, 1]]],
            "state": [1.0, 2.0],
            "scalar": 3,
        }
    )

    assert observation["image_b64"].dtype == np.uint8
    assert observation["image_b64"].tolist() == [1, 2, 3]
    assert observation["image_list"].dtype == np.uint8
    assert observation["state"].dtype == np.float32
    assert observation["scalar"].tolist() == [3.0]


def test_parse_observation_rejects_unsupported_type(server_module) -> None:
    with pytest.raises(ValueError, match="Unsupported observation type"):
        server_module._parse_observation({"bad": {"nested": "dict"}})


def test_resolve_checkpoint_prefers_existing_local_paths(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    absolute = tmp_path / "absolute"
    absolute.mkdir()
    cache_root = tmp_path / "checkpoints"
    relative = cache_root / "relative"
    relative.mkdir(parents=True)
    monkeypatch.setattr(server_module, "CHECKPOINT_DIR", str(cache_root))

    assert server_module._resolve_checkpoint(str(absolute)) == str(absolute)
    assert server_module._resolve_checkpoint("relative") == str(relative)
    assert server_module._resolve_checkpoint("hf/repo") == "hf/repo"


def test_pull_from_s3_downloads_objects(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch, mock_s3
) -> None:
    monkeypatch.setattr(server_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "models/run/config.json"}, {"Key": "models/run/weights.bin"}]}
    ]

    local = Path(server_module._pull_from_s3("s3://bucket/models/run"))

    assert local.name == "bucket_models_run"
    assert mock_s3.download_file.call_count == 2
    mock_s3.get_paginator.assert_called_once_with("list_objects_v2")


def test_read_jobs_ignores_bad_json(tmp_path: Path, server_module, monkeypatch) -> None:
    status_dir = tmp_path / "jobs"
    status_dir.mkdir()
    (status_dir / "good.json").write_text(json.dumps({"name": "job", "status": "ok"}))
    (status_dir / "bad.json").write_text("{bad json")
    monkeypatch.setattr(server_module, "JOB_STATUS_DIR", str(status_dir))

    assert server_module._read_jobs() == [{"name": "job", "status": "ok"}]


def test_health_and_status_endpoints(server_module) -> None:
    with TestClient(server_module.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        status = client.get("/status").json()

    assert status["policy_server"]["running"] is False
    assert status["checkpoint_dir"] == server_module.CHECKPOINT_DIR


def test_infer_without_loaded_policy_returns_conflict(server_module) -> None:
    with TestClient(server_module.app) as client:
        response = client.post("/infer", json={"observation.state": [1.0]})

    assert response.status_code == 409
    assert "No policy loaded" in response.json()["detail"]


def test_serve_and_stop_endpoints_use_policy_state(server_module, monkeypatch) -> None:
    class FakePolicyState:
        def __init__(self) -> None:
            self.policy = object()
            self.device = FakeDevice("cpu")
            self.loaded = False
            self.checkpoint = ""
            self.loaded_at = 0.0
            self.load_calls: list[tuple[str, str | None, str | None]] = []
            self.unloaded = False

        def load(self, checkpoint: str, env_type=None, env_task=None) -> None:
            self.loaded = True
            self.checkpoint = checkpoint
            self.load_calls.append((checkpoint, env_type, env_task))

        def unload(self) -> None:
            self.loaded = False
            self.unloaded = True

    state = FakePolicyState()
    monkeypatch.setattr(server_module, "policy_state", state)
    monkeypatch.setattr(server_module, "_resolve_checkpoint", lambda checkpoint: "/resolved")

    with TestClient(server_module.app) as client:
        serve_response = client.post(
            "/serve",
            json={"checkpoint": "s3://bucket/model", "env_type": "aloha", "env_task": "task"},
        )
        stop_response = client.delete("/serve")

    assert serve_response.status_code == 200
    assert serve_response.json()["checkpoint"] == "/resolved"
    assert state.load_calls == [("/resolved", "aloha", "task")]
    assert stop_response.json() == {"status": "stopped"}
    assert state.unloaded is True


def test_infer_endpoint_parses_observation_and_returns_actions(
    server_module, monkeypatch
) -> None:
    class FakePolicyState:
        loaded = True
        checkpoint = "/checkpoint"

        def predict(self, observation):
            assert "observation.state" in observation
            return [0.1, 0.2]

        def unload(self) -> None:
            return None

    monkeypatch.setattr(server_module, "policy_state", FakePolicyState())

    with TestClient(server_module.app) as client:
        response = client.post("/infer", json={"observation.state": [1.0, 2.0]})

    assert response.status_code == 200
    assert response.json()["actions"] == [0.1, 0.2]
    assert response.json()["checkpoint"] == "/checkpoint"
