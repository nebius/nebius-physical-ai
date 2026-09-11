"""Exercise the actual HTTP boundary without fetching model or dataset payloads."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import stat
import threading
import time

from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

from npa.workbench.alpamayo2_super import service
from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperError,
    Alpamayo2SuperRequest,
    _runtime_env,
    run_inference,
)

TOKEN = "test-inference-credential"
HEADERS = {"Authorization": "Bearer " + TOKEN}


@pytest.fixture
def configured(tmp_path):
    source = tmp_path / "operator-manifest.json"
    source.write_text(json.dumps({"samples": [{"clip_id": "sample", "t0_us": 1}]}))
    output = tmp_path / "outputs"
    app = service.create_app(token=TOKEN, output_root=str(output), manifest=str(source))
    return app, source, output


@pytest.mark.parametrize("path", ["/health", "/status", "/system-info", "/list", "/run"])
def test_all_service_operations_require_authentication(configured, path):
    app, _, output = configured
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer unrelated-credential"}):
            response = client.post(path, json={}, headers=headers) if path == "/run" else client.get(path, headers=headers)
            assert response.status_code == 401
        assert list(output.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("model_id", "untrusted/model"), ("model_revision", "main"),
    ("dataset_revision", "main"), ("manifest", "/tmp/manifest.json"),
    ("runtime_image", "untrusted-runtime"), ("output_path", "../escape"),
    ("output_path", "/tmp/escape"), ("output_path", "s3://outside/results"),
    ("output_path", "nested/result"), ("output_path", "result\\escape"),
    ("output_path", "x" * 129), ("sample_index", -1), ("sample_index", 1),
])
def test_http_cannot_change_deployment_code_inputs_or_output_authority(configured, field, value):
    app, _, output = configured
    with TestClient(app) as client:
        response = client.post("/run", json={field: value, "dry_run": True}, headers=HEADERS)
        assert response.status_code == 422
        assert list(output.iterdir()) == []


def test_manifest_snapshot_is_private_immutable_and_removed_on_shutdown(configured):
    app, source, output = configured
    initial = source.read_bytes()
    with TestClient(app) as client:
        source.write_text('{"samples": []}')
        response = client.post("/run", json={"output_path": "sample", "dry_run": True}, headers=HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == {"id": DEFAULT_MODEL_ID, "revision": DEFAULT_MODEL_REVISION}
        assert body["dataset"]["revision"] == DEFAULT_DATASET_REVISION
        snapshot = Path(body["request"]["manifest"])
        assert snapshot.read_bytes() == initial
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400
        assert stat.S_IMODE(snapshot.parent.stat().st_mode) == 0o700
        assert Path(body["request"]["output_path"]).parent == output
        assert list(output.iterdir()) == []  # Preparation does not leave empty results.
    assert not snapshot.exists()


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_service_rejects_output_root_accessible_to_other_users(tmp_path, mode):
    output = tmp_path / "outputs"
    output.mkdir(mode=mode)
    output.chmod(mode)
    with pytest.raises(ValueError, match="0700"):
        with TestClient(service.create_app(token=TOKEN, output_root=str(output))):
            pass


def test_service_rejects_symlink_output_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "outputs"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="0700"):
        with TestClient(service.create_app(token=TOKEN, output_root=str(link))):
            pass


def test_missing_admission_credential_fails_startup(monkeypatch):
    monkeypatch.delenv("NPA_ALPAMAYO2_SUPER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TOKEN is required"):
        with TestClient(service.create_app()):
            pass


def test_upstream_errors_do_not_reflect_sensitive_diagnostics(configured, monkeypatch):
    def fail(_request):
        raise Alpamayo2SuperError("synthetic-private-diagnostic")
    monkeypatch.setattr(service, "run_inference", fail)
    app, _, output = configured
    with TestClient(app) as client:
        response = client.post("/run", json={}, headers=HEADERS)
    assert response.status_code == 422
    assert "synthetic-private-diagnostic" not in response.text
    assert list(output.iterdir()) == []


def test_s3_outputs_get_unique_prefixes_inside_configured_authority(configured):
    _, source, _ = configured
    app = service.create_app(token=TOKEN, output_root="s3://operator-results/inference", manifest=str(source))
    with TestClient(app) as client:
        first, second = [client.post("/run", json={"dry_run": True}, headers=HEADERS).json() for _ in range(2)]
    a, b = first["request"]["output_path"], second["request"]["output_path"]
    assert a.startswith("s3://operator-results/inference/run-")
    assert b.startswith("s3://operator-results/inference/run-")
    assert a != b


def test_trusted_operator_cli_contract_keeps_custom_model_support(tmp_path, monkeypatch):
    request = Alpamayo2SuperRequest(output_path=str(tmp_path), model_id="operator/custom-model", dry_run=True)
    result = run_inference(request)
    assert result["model"]["id"] == "operator/custom-model"
    monkeypatch.setenv("NPA_ALPAMAYO2_SUPER_TOKEN", TOKEN)
    assert "NPA_ALPAMAYO2_SUPER_TOKEN" not in _runtime_env(request)


@contextmanager
def live_service(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    address = listener.getsockname()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on", ws="none"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("HTTP service exited before readiness")
            time.sleep(0.01)
        with httpx.Client(base_url=f"http://127.0.0.1:{address[1]}") as client:
            yield client
    finally:
        server.should_exit = True
        thread.join()
        listener.close()


def test_actual_http_authorized_component_preparation_and_admission(configured):
    app, _, output = configured
    with live_service(app) as client:
        assert client.post("/run", json={"dry_run": True}).status_code == 401
        assert client.post("/run", json={"model_id": "untrusted/model"}, headers=HEADERS).status_code == 422
        assert client.get("/health", headers=HEADERS).status_code == 200
        response = client.post("/run", json={"dry_run": True}, headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "dry_run"
        assert data["argv"][1:3] == ["-m", "alpamayo2_super.inference_smoke"]
        assert data["artifacts"] == {}
    assert list(output.iterdir()) == []
