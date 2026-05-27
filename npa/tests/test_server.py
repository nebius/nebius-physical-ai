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


def test_parse_observation_float_image_branch(server_module) -> None:
    """List-shaped image with values <= 1.0 falls through to float32 cast."""
    obs = server_module._parse_observation({"image": [[[0.1, 0.2, 0.3]]]})

    assert obs["image"].dtype == np.float32
    assert obs["image"].shape == (1, 1, 3)


def test_resolve_checkpoint_s3_uri_delegates_to_pull(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_pull(uri: str) -> str:
        calls.append(uri)
        return "/cached/path"

    monkeypatch.setattr(server_module, "_pull_from_s3", fake_pull)

    assert server_module._resolve_checkpoint("s3://bucket/key") == "/cached/path"
    assert calls == ["s3://bucket/key"]


def test_pull_from_s3_uses_cached_dir_when_populated(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch, mock_s3
) -> None:
    """If the cache dir already has files, skip the boto3 download entirely."""
    monkeypatch.setattr(server_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    cache_dir = (
        tmp_path / "checkpoints" / "s3_cache" / "bucket_models_run"
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / "weights.bin").write_bytes(b"cached")

    result = server_module._pull_from_s3("s3://bucket/models/run")

    assert result == str(cache_dir)
    mock_s3.get_paginator.assert_not_called()
    mock_s3.download_file.assert_not_called()


def test_pull_from_s3_passes_credentials_to_boto3(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """boto3.client must receive endpoint + credentials from module env."""
    monkeypatch.setattr(server_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(server_module, "AWS_ENDPOINT_URL", "https://s3.example")
    monkeypatch.setattr(server_module, "AWS_ACCESS_KEY_ID", "ak")
    monkeypatch.setattr(server_module, "AWS_SECRET_ACCESS_KEY", "sk")

    fake_client = mocker.MagicMock()
    fake_client.get_paginator.return_value.paginate.return_value = []
    boto3_client = mocker.patch("boto3.client", return_value=fake_client)

    server_module._pull_from_s3("s3://bucket/prefix")

    boto3_client.assert_called_once_with(
        "s3",
        endpoint_url="https://s3.example",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )


def test_pull_from_s3_skips_empty_relative_key(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch, mock_s3
) -> None:
    """An object key equal to the prefix itself yields rel='' and must be skipped."""
    monkeypatch.setattr(server_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "models/run/"},
                {"Key": "models/run/weights.bin"},
            ]
        }
    ]

    server_module._pull_from_s3("s3://bucket/models/run")

    assert mock_s3.download_file.call_count == 1


def test_pull_from_s3_handles_prefix_with_trailing_slash(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch, mock_s3
) -> None:
    """A trailing slash on the source URI must not double-slash the prefix."""
    monkeypatch.setattr(server_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = []

    server_module._pull_from_s3("s3://bucket/models/run/")

    paginator.paginate.assert_called_once_with(
        Bucket="bucket", Prefix="models/run/"
    )


def test_read_jobs_returns_empty_when_dir_missing(
    tmp_path: Path, server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server_module, "JOB_STATUS_DIR", str(tmp_path / "does-not-exist")
    )

    assert server_module._read_jobs() == []


def test_status_endpoint_reports_loaded_policy_fields(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePolicy:
        pass

    class FakeLoadedState:
        policy = FakePolicy()
        device = FakeDevice("cpu")
        checkpoint = "/ckpt"
        loaded_at = 0.0
        loaded = True

        def unload(self) -> None:
            return None

    monkeypatch.setattr(server_module, "policy_state", FakeLoadedState())

    with TestClient(server_module.app) as client:
        body = client.get("/status").json()

    assert body["policy_server"]["running"] is True
    assert body["policy_server"]["checkpoint"] == "/ckpt"
    assert body["policy_server"]["policy_class"] == "FakePolicy"
    assert body["policy_server"]["device"] == "cpu"
    assert body["policy_server"]["uptime_seconds"] >= 0


def test_serve_returns_400_when_resolve_fails(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_checkpoint: str) -> str:
        raise RuntimeError("missing")

    monkeypatch.setattr(server_module, "_resolve_checkpoint", boom)

    with TestClient(server_module.app) as client:
        response = client.post("/serve", json={"checkpoint": "s3://bucket/x"})

    assert response.status_code == 400
    assert "Checkpoint resolution failed" in response.json()["detail"]


def test_serve_returns_500_when_load_fails(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingState:
        loaded = False

        def load(self, *_args, **_kwargs) -> None:
            raise RuntimeError("bad weights")

        def unload(self) -> None:
            return None

    monkeypatch.setattr(server_module, "_resolve_checkpoint", lambda c: "/resolved")
    monkeypatch.setattr(server_module, "policy_state", FailingState())

    with TestClient(server_module.app) as client:
        response = client.post("/serve", json={"checkpoint": "s3://bucket/x"})

    assert response.status_code == 500
    assert "Failed to load policy" in response.json()["detail"]


def test_infer_returns_400_on_invalid_observation(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LoadedState:
        loaded = True

        def unload(self) -> None:
            return None

    monkeypatch.setattr(server_module, "policy_state", LoadedState())

    with TestClient(server_module.app) as client:
        response = client.post("/infer", json={"bad": {"nested": "dict"}})

    assert response.status_code == 400
    assert "Invalid observation payload" in response.json()["detail"]


def test_infer_returns_500_when_predict_raises(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingState:
        loaded = True
        checkpoint = "/ckpt"

        def predict(self, _obs):
            raise RuntimeError("kaboom")

        def unload(self) -> None:
            return None

    monkeypatch.setattr(server_module, "policy_state", ExplodingState())

    with TestClient(server_module.app) as client:
        response = client.post("/infer", json={"observation.state": [1.0]})

    assert response.status_code == 500
    assert "Inference failed" in response.json()["detail"]


def test_stop_serve_is_idempotent_when_nothing_loaded(server_module) -> None:
    with TestClient(server_module.app) as client:
        response = client.delete("/serve")

    assert response.status_code == 200
    assert response.json() == {"status": "stopped"}


def test_lifespan_creates_runtime_directories_and_unloads_on_shutdown(
    server_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    unload_calls: list[bool] = []

    class TrackingState:
        loaded = False

        def unload(self) -> None:
            unload_calls.append(True)

    monkeypatch.setattr(server_module, "policy_state", TrackingState())

    with TestClient(server_module.app) as client:
        client.get("/health")
        assert Path(server_module.LOG_DIR).is_dir()
        assert Path(server_module.JOB_STATUS_DIR).is_dir()
        assert Path(server_module.CHECKPOINT_DIR).is_dir()

    assert unload_calls == [True]


def test_aws_endpoint_url_falls_back_to_nebius_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Module-level config must honour NEBIUS_S3_ENDPOINT when AWS_ENDPOINT_URL is unset."""
    fake_torch = SimpleNamespace(
        device=FakeDevice,
        Tensor=FakeTensor,
        cuda=FakeCuda(),
        inference_mode=contextlib.nullcontext,
        autocast=lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("NEBIUS_S3_ENDPOINT", "https://nebius.example")
    monkeypatch.setenv("NPA_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("NPA_JOB_STATUS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("NPA_LOG_DIR", str(tmp_path / "logs"))

    sys.modules.pop("npa.server.app", None)
    try:
        module = importlib.import_module("npa.server.app")
        assert module.AWS_ENDPOINT_URL == "https://nebius.example"
    finally:
        sys.modules.pop("npa.server.app", None)
