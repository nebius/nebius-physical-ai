"""Route-level tests for the shipped Foxglove backend module.

The `/api/foxglove/*` routes are registered by
``npa.agent_backend.foxglove_routes.register_foxglove_routes`` on the agent VM.
Because every backend dependency is injected, the routes can be exercised here
against a real FastAPI app with fakes — no agent VM, no object storage, no SSH.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from npa.agent_backend.foxglove import (  # noqa: E402
    FOXGLOVE_SDK_FILES,
    resolve_foxglove_config,
)
from npa.agent_backend.foxglove_routes import (  # noqa: E402
    FoxgloveDeps,
    register_foxglove_routes,
)


class _Summary:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


@pytest.fixture()
def harness(tmp_path: Path):
    """A FastAPI app wired to in-memory fakes, plus the captured state."""
    assets = tmp_path / "sdk"
    assets.mkdir()
    for name in FOXGLOVE_SDK_FILES:
        (assets / name).write_text("export {};", encoding="utf-8")

    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    state: dict = {"sim_viz": {"run_id": "run-1"}}
    saved: list[dict] = []
    recorded: list[dict] = []
    env: dict[str, str] = {"NPA_FOXGLOVE_EMBED_SRC": "https://embed.foxglove.dev/"}
    convert_calls: list[dict] = []
    cloud_calls: list[tuple[Path, str]] = []

    def _config(current: dict | None = None) -> dict:
        return resolve_foxglove_config(
            env,
            assets_dir=assets,
            origin=env.get("_origin", "https://agent.example"),
            sim_viz=(current or state).get("sim_viz", {}),
            self_hosted_ready=env.get("_self_hosted_ready") == "1",
        )

    def _convert_run(**kwargs):
        convert_calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"\x89MCAP0\r\n" + b"x" * 32)
        return _Summary(
            {
                "frames": 3,
                "metrics": 1,
                "logs": 2,
                "message_count": 6,
                "fps": kwargs["fps"],
            }
        )

    loaded: list[dict] = []

    def _load_artifact(body: dict):
        loaded.append(body)
        return {"ok": True, "render": "mcap", "sim_viz": state["sim_viz"]}

    def _validate_run_id(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("invalid run_id")
        return value

    def _cloud(path: Path, run_id: str, **_kwargs) -> dict:
        cloud_calls.append((path, run_id))
        return {
            "recording_id": "rec_test123",
            "recording_key": "npa-digest",
            "import_status": "complete",
            "size_bytes": path.stat().st_size,
            "uploaded": not bool(cloud_calls[:-1]),
            "reused": bool(cloud_calls[:-1]),
            "layout": {
                "layout_id": "layout-rich",
                "available": True,
                "created": not bool(cloud_calls[:-1]),
                "updated": False,
                "reused": bool(cloud_calls[:-1]),
            },
        }

    app = FastAPI()
    register_foxglove_routes(
        app,
        FoxgloveDeps(
            load_state=lambda: state,
            save_state=lambda new: saved.append(dict(new)),
            record_run=lambda st, viz: recorded.append(dict(viz)),
            foxglove_config=_config,
            load_artifact=_load_artifact,
            convert_run=_convert_run,
            now_iso=lambda: "2026-07-31T00:00:00+00:00",
            validate_run_id=_validate_run_id,
            data_dir=data_dir,
            runs_dir=runs_dir,
            keep_published=2,
            ensure_cloud_recording=_cloud,
        ),
        HTTPException,
    )
    return {
        "client": TestClient(app),
        "state": state,
        "saved": saved,
        "recorded": recorded,
        "env": env,
        "runs_dir": runs_dir,
        "data_dir": data_dir,
        "convert_calls": convert_calls,
        "loaded": loaded,
        "cloud_calls": cloud_calls,
    }


def test_config_and_status_report_the_selected_backend(harness) -> None:
    client = harness["client"]

    config = client.get("/foxglove/config").json()
    assert config["viewer_backend"] == "foxglove-sdk"
    assert config["available"] is True

    status = client.get("/foxglove/status").json()
    assert status["viewer_backend"] == "foxglove-sdk"
    assert status["available"] is True


def test_config_falls_back_to_the_self_hosted_viewer(harness) -> None:
    # No embed source, but the OSS viewer is healthy: the pane must still render.
    harness["env"]["NPA_FOXGLOVE_EMBED_SRC"] = "not-a-url"
    harness["env"]["_self_hosted_ready"] = "1"
    harness["state"]["sim_viz"]["mcap_uri"] = (
        "file:///opt/npa-agent/recordings/sim2real.mcap"
    )

    config = harness["client"].get("/foxglove/config").json()

    assert config["viewer_backend"] == "self-hosted"
    assert config["available"] is True
    assert config["self_hosted_url"].startswith("/lichtblick/?ds=remote-file")
    assert "sim2real.mcap" in config["self_hosted_url"]


def test_config_explains_when_no_viewer_can_render(harness, tmp_path: Path) -> None:
    harness["env"]["NPA_FOXGLOVE_EMBED_SRC"] = ""
    harness["env"]["NPA_FOXGLOVE_ENABLED"] = "0"

    config = harness["client"].get("/foxglove/config").json()

    assert config["available"] is False
    assert "disabled" in config["reason"]
    assert config["viewer_backend"] == ""


def test_load_artifact_rejects_non_recordings(harness) -> None:
    response = harness["client"].post(
        "/foxglove/load-artifact", json={"key": "run-1/report.json"}
    )

    assert response.status_code == 400
    assert "recordings" in response.json()["detail"]
    assert harness["loaded"] == []


def test_load_artifact_delegates_and_attaches_status(harness) -> None:
    response = harness["client"].post(
        "/foxglove/load-artifact",
        json={"run_id": "run-1", "key": "run-1/reports/x.mcap"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["foxglove"]["viewer_backend"] == "foxglove-sdk"
    assert harness["loaded"] == [{"run_id": "run-1", "key": "run-1/reports/x.mcap"}]


def test_convert_run_requires_local_artifacts(harness) -> None:
    response = harness["client"].post(
        "/foxglove/convert-run", json={"run_id": "missing-run"}
    )

    assert response.status_code == 404
    assert "no local artifacts" in response.json()["detail"]
    assert harness["convert_calls"] == []


def test_convert_run_writes_publishes_and_updates_state(harness) -> None:
    (harness["runs_dir"] / "run-1").mkdir()

    response = harness["client"].post(
        "/foxglove/convert-run", json={"run_id": "run-1", "fps": 4}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["frames"] == 3
    assert harness["convert_calls"][0]["fps"] == 4.0
    published = list(harness["data_dir"].glob("*.mcap"))
    assert len(published) == 1
    assert published[0].read_bytes().startswith(b"\x89MCAP0\r\n")
    # State is persisted and the run history records it.
    assert body["sim_viz"]["foxglove_url"].endswith(published[0].name)
    assert body["sim_viz"]["artifact_render"] == "mcap"
    assert harness["saved"] and harness["recorded"]


def test_export_reuses_active_mcap_and_returns_open_links(harness) -> None:
    harness["data_dir"].mkdir()
    (harness["data_dir"] / "random-active.mcap").write_bytes(b"\x89MCAP0\r\nbody")
    harness["state"]["sim_viz"].update(
        {
            "foxglove_url": "/foxglove/data/random-active.mcap",
            "foxglove_ready": True,
        }
    )

    response = harness["client"].post("/foxglove/export", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["converted"] is False
    assert body["export"]["recording_url"] == (
        "https://agent.example/foxglove/data/random-active.mcap"
    )
    assert "web_url" not in body["export"]
    assert harness["convert_calls"] == []


def test_export_converts_active_run_when_no_mcap_is_published(harness) -> None:
    (harness["runs_dir"] / "run-1").mkdir()

    response = harness["client"].post("/foxglove/export", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["converted"] is True
    assert body["summary"]["message_count"] == 6
    assert body["export"]["available"] is True
    assert harness["convert_calls"]


def test_export_open_web_returns_selected_remote_file_link_without_cloud_upload(
    harness,
) -> None:
    (harness["runs_dir"] / "run-1").mkdir()

    response = harness["client"].post("/foxglove/export", json={"open_web": True})

    assert response.status_code == 200
    exported = response.json()["export"]
    assert exported["data_source"] == "remote-file"
    assert exported["web_open_mode"] == "remote-file"
    assert "ds=remote-file" in exported["web_url"]
    assert "ds.url=https%3A%2F%2Fagent.example%2Ffoxglove%2Fdata%2F" in exported["web_url"]
    assert "cloud" not in exported
    assert "openIn" not in exported["web_url"]
    assert harness["cloud_calls"] == []


def test_export_different_safe_run_converts_that_run(harness) -> None:
    (harness["runs_dir"] / "run-2").mkdir()

    response = harness["client"].post("/foxglove/export", json={"run_id": "run-2"})

    assert response.status_code == 200
    assert response.json()["converted"] is True
    assert response.json()["run_id"] == "run-2"
    assert harness["convert_calls"][-1]["run_id"] == "run-2"


def test_export_uses_one_canonical_s3_contract_for_viewers_download_and_cloud(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "sdk"
    assets.mkdir()
    for name in FOXGLOVE_SDK_FILES:
        (assets / name).write_text("export {};", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical = tmp_path / "canonical.mcap"
    canonical.write_bytes(b"\x89MCAP0\r\ncanonical-bytes")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    key = "runs/run-1/reports/sim2real.mcap"
    uri = f"s3://bucket/{key}"
    state = {"sim_viz": {"run_id": "run-1"}}
    prepare_calls: list[dict] = []
    cloud_calls: list[Path] = []

    def save_state(new: dict) -> None:
        snapshot = dict(new)
        state.clear()
        state.update(snapshot)

    def prepare(**kwargs) -> dict:
        prepare_calls.append(kwargs)
        source = (
            "generated-from-s3-artifacts"
            if len(prepare_calls) == 1
            else "native-reused"
        )
        provenance = {
            "schema": "npa.canonical-mcap.v1",
            "canonical_s3_uri": uri,
            "sha256": digest,
            "source": source,
        }
        return {
            "artifact_key": key,
            "s3_uri": uri,
            "local_path": str(canonical),
            "sha256": digest,
            "size_bytes": canonical.stat().st_size,
            "source": source,
            "created": len(prepare_calls) == 1,
            "provenance": provenance,
            "summary": {"message_count": 1, "channels": {"/camera": 1}},
        }

    def load_artifact(body: dict) -> dict:
        assert body == {"run_id": "run-1", "run_ref": "npa1_exact", "key": key}
        published = data_dir / "transport.mcap"
        published.write_bytes(canonical.read_bytes())
        state["sim_viz"].update(
            {
                "run_id": "run-1",
                "artifact_key": key,
                "artifact_render": "mcap",
                "foxglove_url": "/foxglove/data/transport.mcap",
                "lichtblick_ready": True,
                "lichtblick_iframe_url": "/lichtblick/?ds=remote-file",
            }
        )
        return {"ok": True, "sim_viz": state["sim_viz"], "render": "mcap"}

    def config(current: dict | None = None) -> dict:
        return resolve_foxglove_config(
            {"NPA_FOXGLOVE_EMBED_SRC": "https://embed.foxglove.dev/"},
            assets_dir=assets,
            origin="https://agent.example",
            sim_viz=(current or state)["sim_viz"],
            self_hosted_ready=True,
        )

    def cloud(path: Path, _run_id: str, **_kwargs) -> dict:
        cloud_calls.append(path)
        assert path.read_bytes() == canonical.read_bytes()
        return {
            "recording_id": "rec_exact",
            "recording_key": f"npa-{digest}",
            "import_status": "complete",
            "size_bytes": path.stat().st_size,
            "uploaded": len(cloud_calls) == 1,
            "reused": len(cloud_calls) > 1,
        }

    app = FastAPI()
    register_foxglove_routes(
        app,
        FoxgloveDeps(
            load_state=lambda: state,
            save_state=save_state,
            record_run=lambda _state, _viz: None,
            foxglove_config=config,
            load_artifact=load_artifact,
            convert_run=lambda **_kwargs: None,
            now_iso=lambda: "2026-08-09T00:00:00+00:00",
            validate_run_id=lambda value: value,
            data_dir=data_dir,
            runs_dir=tmp_path / "runs",
            prepare_canonical_mcap=prepare,
            ensure_cloud_recording=cloud,
        ),
        HTTPException,
    )
    client = TestClient(app)

    request = {"open_web": True, "cloud_import": True, "run_ref": "npa1_exact"}
    first = client.post("/foxglove/export", json=request)
    second = client.post("/foxglove/export", json=request)

    assert first.status_code == second.status_code == 200
    assert [call["run_ref"] for call in prepare_calls] == [
        "npa1_exact",
        "npa1_exact",
    ]
    assert first.json()["converted"] is True
    assert second.json()["converted"] is False
    for response in (first.json(), second.json()):
        assert response["artifact_key"] == key
        assert response["export"]["canonical_s3_uri"] == uri
        assert response["export"]["sha256"] == digest
        assert response["size_bytes"] == canonical.stat().st_size
        assert response["sim_viz"]["canonical_mcap_sha256"] == digest
        assert response["export"]["cloud"]["recording_key"] == f"npa-{digest}"
    assert state["sim_viz"]["foxglove_cloud"]["import_status"] == "complete"


def test_canonical_s3_failure_is_not_concealed(tmp_path: Path) -> None:
    app = FastAPI()

    def fail(**_kwargs):
        raise RuntimeError("S3 put failed")

    register_foxglove_routes(
        app,
        FoxgloveDeps(
            load_state=lambda: {"sim_viz": {"run_id": "run-1"}},
            save_state=lambda _state: None,
            record_run=lambda _state, _viz: None,
            foxglove_config=lambda _state=None: {},
            load_artifact=lambda _body: {},
            convert_run=lambda **_kwargs: None,
            now_iso=lambda: "",
            validate_run_id=lambda value: value,
            data_dir=tmp_path / "data",
            runs_dir=tmp_path / "runs",
            prepare_canonical_mcap=fail,
        ),
        HTTPException,
    )

    response = TestClient(app).post("/foxglove/export", json={})

    assert response.status_code == 422
    assert "S3 put failed" in response.json()["detail"]


def test_export_rejects_traversal_without_writing(harness) -> None:
    before = list(harness["data_dir"].glob("**/*"))

    response = harness["client"].post(
        "/foxglove/export", json={"run_id": "../../etc", "force_convert": True}
    )

    assert response.status_code == 400
    assert "invalid run_id" in response.json()["detail"]
    assert list(harness["data_dir"].glob("**/*")) == before


@pytest.mark.parametrize("origin", ["", "http://agent.example", "https://127.0.0.1"])
def test_export_returns_409_without_secure_public_origin(harness, origin: str) -> None:
    (harness["runs_dir"] / "run-1").mkdir()
    harness["env"]["_origin"] = origin

    response = harness["client"].post("/foxglove/export", json={})

    assert response.status_code == 409
    assert "HTTPS" in response.json()["detail"]


def test_export_reports_missing_active_run(harness) -> None:
    harness["state"]["sim_viz"] = {}

    response = harness["client"].post("/foxglove/export", json={})

    assert response.status_code == 400
    assert "no active run" in response.json()["detail"]


def test_live_route_validates_and_uses_session_state(harness, monkeypatch) -> None:
    import os

    monkeypatch.delenv("NPA_FOXGLOVE_LIVE_URL", raising=False)
    client = harness["client"]

    bad = client.post("/foxglove/live", json={"url": "ws://127.0.0.1:8765"})
    assert bad.status_code == 400
    assert "public ws" in bad.json()["detail"]

    good = client.post("/foxglove/live", json={"url": "wss://robot.example.com:8765"})
    assert good.status_code == 200
    assert good.json()["data_source"] == {
        "type": "live",
        "protocol": "foxglove-websocket",
        "url": "wss://robot.example.com:8765",
    }
    # Session state, not the process environment (the review's ask).
    assert (
        harness["state"]["sim_viz"]["foxglove_live_url"]
        == "wss://robot.example.com:8765"
    )
    assert "NPA_FOXGLOVE_LIVE_URL" not in os.environ
    assert harness["saved"]


def test_convert_run_surfaces_writer_failures(harness) -> None:
    (harness["runs_dir"] / "run-1").mkdir()

    def _boom(**_kwargs):
        raise RuntimeError("nothing to convert")

    # Re-register with a failing converter to prove the 422 path.
    app = FastAPI()
    register_foxglove_routes(
        app,
        FoxgloveDeps(
            load_state=lambda: harness["state"],
            save_state=lambda new: None,
            record_run=lambda st, viz: None,
            foxglove_config=lambda current=None: {"available": True},
            load_artifact=lambda body: {"ok": False},
            convert_run=_boom,
            now_iso=lambda: "2026-07-31T00:00:00+00:00",
            validate_run_id=lambda value: value,
            data_dir=harness["data_dir"],
            runs_dir=harness["runs_dir"],
        ),
        HTTPException,
    )
    response = TestClient(app).post("/foxglove/convert-run", json={"run_id": "run-1"})

    assert response.status_code == 422
    assert "MCAP conversion failed" in response.json()["detail"]


def test_routes_are_json_serializable(harness) -> None:
    # Guards against a payload that FastAPI cannot encode (e.g. a Path object).
    for path in ("/foxglove/config", "/foxglove/status"):
        json.dumps(harness["client"].get(path).json())
