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
from urllib.parse import parse_qs, urlparse

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from npa.agent_backend.foxglove import (  # noqa: E402
    FOXGLOVE_SDK_FILES,
    resolve_foxglove_config,
)
from npa.agent_backend.foxglove_cloud import (  # noqa: E402
    FoxgloveCloudTimeoutError,
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
    layout_calls: list[dict] = []

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
        key = str(body.get("key") or "")
        if key.lower().endswith(".mcap"):
            data_dir.mkdir(parents=True, exist_ok=True)
            published = data_dir / "selected-native.mcap"
            published.write_bytes(b"\x89MCAP0\r\nselected-exact-bytes")
            loaded_key = (
                "runs/run-1/other.mcap" if key.endswith("requested.mcap") else key
            )
            state["sim_viz"].update(
                {
                    "run_id": str(body.get("run_id") or "run-1"),
                    "artifact_run_ref": str(body.get("run_ref") or ""),
                    "artifact_key": loaded_key,
                    "artifact_uri": str(body.get("s3_uri") or f"s3://bucket/{key}"),
                    "artifact_render": "mcap",
                    "bucket": str(body.get("bucket") or "bucket"),
                    "project_id": "project-1",
                    "resolved_prefix": "runs",
                    "foxglove_url": "/foxglove/data/selected-native.mcap",
                    "foxglove_ready": True,
                }
            )
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

    def _layout(*, provenance: dict) -> dict:
        layout_calls.append(dict(provenance))
        return {
            "layout_id": "layout-rich",
            "available": True,
            "created": not bool(layout_calls[:-1]),
            "updated": False,
            "reused": bool(layout_calls[:-1]),
            "reason": "",
        }

    app = FastAPI()
    deps = FoxgloveDeps(
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
        ensure_cloud_layout=_layout,
    )
    register_foxglove_routes(
        app,
        deps,
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
        "layout_calls": layout_calls,
        "deps": deps,
    }


def test_config_and_status_report_the_selected_backend(harness) -> None:
    client = harness["client"]

    config = client.get("/foxglove/config").json()
    assert config["viewer_backend"] == "foxglove-sdk"
    assert config["available"] is True
    assert config["cloud_import_timeout_seconds"] == 300.0

    status = client.get("/foxglove/status").json()
    assert status["viewer_backend"] == "foxglove-sdk"
    assert status["available"] is True


def test_config_advertises_the_validated_cloud_import_deadline(harness) -> None:
    harness["env"]["NPA_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS"] = "425.5"
    config = harness["client"].get("/foxglove/config").json()
    assert config["cloud_import_timeout_seconds"] == 425.5

    # Deploy/bootstrap reject this state. A hand-edited remote environment still
    # receives a finite browser fallback while the Cloud client returns its
    # typed invalid-configuration error.
    harness["env"]["NPA_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS"] = "nan"
    fallback = harness["client"].get("/foxglove/config").json()
    assert fallback["cloud_import_timeout_seconds"] == 300.0


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


def test_cloud_timeout_maps_to_504_and_releases_export_lock(harness) -> None:
    harness["data_dir"].mkdir()
    active = harness["data_dir"] / "active.mcap"
    active.write_bytes(b"\x89MCAP0\r\nbody")
    harness["state"]["sim_viz"].update(
        {
            "foxglove_url": "/foxglove/data/active.mcap",
            "foxglove_ready": True,
        }
    )

    def timeout(_path: Path, _run_id: str, **_kwargs) -> dict:
        raise FoxgloveCloudTimeoutError(
            recording_key="npa-" + "a" * 64,
            recording_status="missing",
            import_status="processing",
            elapsed_seconds=300,
        )

    harness["deps"].ensure_cloud_recording = timeout
    failed = harness["client"].post("/foxglove/export", json={"cloud_import": True})

    assert failed.status_code == 504
    assert "server deadline" in failed.json()["detail"]

    # The timeout escaped a route wrapped by the process-wide export lock. A
    # subsequent conversion must still enter the same lock and complete.
    harness["runs_dir"].mkdir(exist_ok=True)
    (harness["runs_dir"] / "run-1").mkdir(exist_ok=True)
    recovered = harness["client"].post(
        "/foxglove/convert-run", json={"run_id": "run-1"}
    )
    assert recovered.status_code == 200
    assert recovered.json()["ok"] is True


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
    assert (
        "ds.url=https%3A%2F%2Fagent.example%2Ffoxglove%2Fdata%2F" in exported["web_url"]
    )
    assert "layoutId=layout-rich" in exported["web_url"]
    assert exported["layout"] == {
        "layout_id": "layout-rich",
        "available": True,
        "created": True,
        "updated": False,
        "reused": False,
        "reason": "",
    }
    assert "canonical shared NPA layout" in exported["layout_note"]
    assert "cloud" not in exported
    assert "openIn" not in exported["web_url"]
    assert harness["cloud_calls"] == []
    assert harness["layout_calls"] == [{}]


def test_export_exact_discovered_mcap_preserves_selection_and_once_encoded_url(
    harness,
) -> None:
    key = "runs/run-1/stages/native camera.mcap"
    uri = f"s3://bucket/{key}"
    request = {
        "run_id": "run-1",
        "run_ref": "npa1_exact_source",
        "key": key,
        "resource_bucket": "bucket",
        "project_id": "project-1",
        "resolved_prefix": "runs",
        "s3_uri": uri,
        "open_web": True,
    }
    # A prior same-run selection must not supply this request's transport URL.
    harness["state"]["sim_viz"]["foxglove_selected_artifact"] = {
        "run_id": "run-1",
        "run_ref": "npa1_exact_source",
        "key": "runs/run-1/stages/older.mcap",
        "recording_url": "https://agent.example/foxglove/data/older.mcap",
    }

    response = harness["client"].post("/foxglove/export", json=request)

    assert response.status_code == 200
    body = response.json()
    assert harness["loaded"][-1] == {
        "run_id": "run-1",
        "run_ref": "npa1_exact_source",
        "key": key,
        "bucket": "bucket",
        "project_id": "project-1",
        "resolved_prefix": "runs",
        "s3_uri": uri,
    }
    assert body["artifact_key"] == key
    assert body["selected_artifact"] == body["export"]["selected_artifact"]
    assert body["selected_artifact"]["key"] == key
    assert body["selected_artifact"]["run_ref"] == "npa1_exact_source"
    assert body["selected_artifact"]["s3_uri"] == uri
    assert body["selected_artifact"]["bucket"] == "bucket"
    assert body["selected_artifact"]["resource_bucket"] == "bucket"
    assert body["selected_artifact"]["project_id"] == "project-1"
    assert body["selected_artifact"]["resolved_prefix"] == "runs"
    assert re.fullmatch(r"[0-9a-f]{64}", body["selected_artifact"]["sha256"])
    assert body["selected_artifact"]["recording_url"] == body["export"]["recording_url"]
    query = parse_qs(urlparse(body["export"]["web_url"]).query)
    assert query["ds"][0] == "remote-file"
    assert query["ds.url"][0] == (
        "https://agent.example/foxglove/data/selected-native.mcap"
    )
    assert "%253A" not in body["export"]["web_url"]
    assert "layoutId" not in query
    assert "default topic browser" in body["export"]["layout_note"]
    assert harness["layout_calls"] == []
    assert (
        harness["state"]["sim_viz"]["foxglove_selected_artifact"]
        == body["selected_artifact"]
    )
    config = harness["client"].get("/foxglove/config").json()
    status = harness["client"].get("/foxglove/status").json()
    for payload in (config, status):
        assert payload["run_id"] == "run-1"
        assert payload["artifact_run_ref"] == "npa1_exact_source"
        assert payload["artifact_key"] == key
        assert payload["artifact_uri"] == uri
        assert payload["project_id"] == "project-1"
        assert payload["resource_bucket"] == "bucket"
        assert payload["resolved_prefix"] == "runs"
        assert payload["artifact_sha256"] == body["selected_artifact"]["sha256"]
        assert payload["selected_artifact"] == body["selected_artifact"]

    # A later same-run preview may mutate the general sim-viz aliases. The
    # selected Foxglove transport remains authoritative until another exact
    # card is prepared.
    harness["state"]["sim_viz"].update(
        {
            "artifact_key": "runs/run-1/other.mcap",
            "artifact_uri": "s3://bucket/runs/run-1/other.mcap",
            "foxglove_url": "/foxglove/data/background-other.mcap",
        }
    )
    refreshed = harness["client"].get("/foxglove/config").json()
    assert refreshed["artifact_key"] == key
    assert refreshed["recording_url"] == body["export"]["recording_url"]
    assert refreshed["data_source"]["urls"] == [body["export"]["recording_url"]]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("project_id", "project-other", "project id"),
        ("resolved_prefix", "other-prefix", "resolved prefix"),
        ("resolved_prefix", "", "resolved prefix"),
    ],
)
def test_export_exact_artifact_rejects_conflicting_source_provenance(
    harness, field: str, value: str, message: str
) -> None:
    key = "runs/run-1/stages/native.mcap"
    request = {
        "run_id": "run-1",
        "run_ref": "npa1_exact_source",
        "key": key,
        "resource_bucket": "bucket",
        "project_id": "project-1",
        "resolved_prefix": "runs",
        "s3_uri": f"s3://bucket/{key}",
        field: value,
    }

    response = harness["client"].post("/foxglove/export", json=request)

    assert response.status_code == 409
    assert message in response.json()["detail"]


def test_export_exact_artifact_rejects_non_mcap_before_publication(harness) -> None:
    response = harness["client"].post(
        "/foxglove/export",
        json={"run_id": "run-1", "key": "runs/run-1/reports/report.json"},
    )

    assert response.status_code == 400
    assert "exact .mcap" in response.json()["detail"]
    assert harness["loaded"] == []


def test_export_exact_artifact_rejects_selection_race(harness) -> None:
    response = harness["client"].post(
        "/foxglove/export",
        json={"run_id": "run-1", "key": "runs/run-1/requested.mcap"},
    )

    assert response.status_code == 409
    assert "selection changed" in response.json()["detail"]


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
            "schema": "npa.canonical-mcap.v2",
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
    assert [call["run_ref"] for call in prepare_calls] == ["npa1_exact"]
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

    # A byte mismatch invalidates the fast path and repairs the public transport
    # from the authoritative canonical S3 contract on the next click.
    (data_dir / "transport.mcap").write_bytes(b"corrupt-public-cache")
    repaired = client.post("/foxglove/export", json=request)
    assert repaired.status_code == 200
    assert len(prepare_calls) == 2
    assert (data_dir / "transport.mcap").read_bytes() == canonical.read_bytes()
    assert repaired.json()["export"]["sha256"] == digest


def test_exact_canonical_fast_path_reuses_version_and_invalidates_change(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "sdk"
    assets.mkdir()
    for name in FOXGLOVE_SDK_FILES:
        (assets / name).write_text("export {};", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical = tmp_path / "canonical.mcap"
    transport = data_dir / "transport.mcap"
    key = "runs/run-1/reports/sim2real.mcap"
    uri = f"s3://bucket/{key}"
    request = {
        "run_id": "run-1",
        "run_ref": "npa1_exact",
        "key": key,
        "resource_bucket": "bucket",
        "project_id": "project-1",
        "resolved_prefix": "runs",
        "s3_uri": uri,
    }
    source = {
        "fingerprint": "a" * 64,
        "bytes": b"\x89MCAP0\r\nfirst-version",
    }
    state = {"sim_viz": {"run_id": "run-1"}}
    calls = {"resolve": 0, "load": 0, "prepare": 0, "apply": 0}

    def save_state(new: dict) -> None:
        snapshot = dict(new)
        state.clear()
        state.update(snapshot)

    def resolve(_body: dict) -> dict:
        calls["resolve"] += 1
        return {
            "run_id": "run-1",
            "run_ref": "npa1_exact",
            "key": key,
            "s3_uri": uri,
            "bucket": "bucket",
            "resource_bucket": "bucket",
            "project_id": "project-1",
            "resolved_prefix": "runs",
            "source_fingerprint": source["fingerprint"],
            "source_size_bytes": len(source["bytes"]),
            "source_last_modified": "2026-08-16T00:00:00+00:00",
        }

    def load_artifact(_body: dict) -> dict:
        calls["load"] += 1
        raise AssertionError("canonical exact selection must not pre-download")

    def prepare(**_kwargs) -> dict:
        calls["prepare"] += 1
        canonical.write_bytes(source["bytes"])
        digest = hashlib.sha256(source["bytes"]).hexdigest()
        provenance = {
            "schema": "npa.canonical-mcap.v2",
            "canonical_s3_uri": uri,
            "sha256": digest,
            "source": "native-reused",
        }
        return {
            "artifact_key": key,
            "s3_uri": uri,
            "local_path": str(canonical),
            "sha256": digest,
            "size_bytes": canonical.stat().st_size,
            "source": "native-reused",
            "created": False,
            "provenance": provenance,
            "summary": {"message_count": 1},
        }

    def apply_prepared(*, canonical: dict, run_id: str, run_ref: str) -> dict:
        calls["apply"] += 1
        transport.write_bytes(Path(canonical["local_path"]).read_bytes())
        state["sim_viz"].update(
            {
                "run_id": run_id,
                "artifact_run_ref": run_ref,
                "artifact_key": key,
                "artifact_uri": uri,
                "artifact_render": "mcap",
                "bucket": "bucket",
                "project_id": "project-1",
                "resolved_prefix": "runs",
                "artifact_source_fingerprint": source["fingerprint"],
                "foxglove_url": "/foxglove/data/transport.mcap",
                "foxglove_ready": True,
            }
        )
        return {"ok": True, "render": "mcap", "sim_viz": state["sim_viz"]}

    def config(current: dict | None = None) -> dict:
        return resolve_foxglove_config(
            {"NPA_FOXGLOVE_EMBED_SRC": "https://embed.foxglove.dev/"},
            assets_dir=assets,
            origin="https://agent.example",
            sim_viz=(current or state)["sim_viz"],
        )

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
            now_iso=lambda: "2026-08-16T00:00:00+00:00",
            validate_run_id=lambda value: value,
            data_dir=data_dir,
            runs_dir=tmp_path / "runs",
            prepare_canonical_mcap=prepare,
            resolve_artifact=resolve,
            apply_prepared_canonical=apply_prepared,
        ),
        HTTPException,
    )
    client = TestClient(app)

    first = client.post("/foxglove/export", json=request)
    second = client.post("/foxglove/export", json=request)

    assert first.status_code == second.status_code == 200
    assert first.json()["cache_reused"] is False
    assert second.json()["cache_reused"] is True
    assert second.json()["sim_viz"]["transport_state"] == "published-selected-cache"
    assert calls == {"resolve": 2, "load": 0, "prepare": 1, "apply": 1}

    source.update(
        {
            "fingerprint": "b" * 64,
            "bytes": b"\x89MCAP0\r\nchanged-version",
        }
    )
    changed = client.post("/foxglove/export", json=request)

    assert changed.status_code == 200
    assert changed.json()["cache_reused"] is False
    assert changed.json()["selected_artifact"]["source_fingerprint"] == "b" * 64
    assert changed.json()["export"]["sha256"] != first.json()["export"]["sha256"]
    assert calls == {"resolve": 3, "load": 0, "prepare": 2, "apply": 2}


def test_exact_native_cache_rechecks_object_identity_and_retries_race(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "sdk"
    assets.mkdir()
    for name in FOXGLOVE_SDK_FILES:
        (assets / name).write_text("export {};", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    transport = data_dir / "native.mcap"
    key = "runs/run-1/recordings/native.mcap"
    uri = f"s3://bucket/{key}"
    request = {
        "run_id": "run-1",
        "run_ref": "npa1_exact",
        "key": key,
        "resource_bucket": "bucket",
        "project_id": "project-1",
        "resolved_prefix": "runs",
        "s3_uri": uri,
    }
    source = {
        "fingerprint": "a" * 64,
        "bytes": b"\x89MCAP0\r\nnative-v1",
        "race_to": None,
    }
    state = {"sim_viz": {"run_id": "run-1"}}
    calls = {"resolve": 0, "load": 0}

    def save_state(new: dict) -> None:
        snapshot = dict(new)
        state.clear()
        state.update(snapshot)

    def resolve(_body: dict) -> dict:
        calls["resolve"] += 1
        return {
            "run_id": "run-1",
            "run_ref": "npa1_exact",
            "key": key,
            "s3_uri": uri,
            "bucket": "bucket",
            "resource_bucket": "bucket",
            "project_id": "project-1",
            "resolved_prefix": "runs",
            "source_fingerprint": source["fingerprint"],
            "source_size_bytes": len(source["bytes"]),
            "source_last_modified": "2026-08-16T00:00:00+00:00",
        }

    def load_artifact(_body: dict) -> dict:
        calls["load"] += 1
        transport.write_bytes(source["bytes"])
        state["sim_viz"].update(
            {
                "run_id": "run-1",
                "artifact_run_ref": "npa1_exact",
                "artifact_key": key,
                "artifact_uri": uri,
                "artifact_render": "mcap",
                "bucket": "bucket",
                "project_id": "project-1",
                "resolved_prefix": "runs",
                "foxglove_url": "/foxglove/data/native.mcap",
                "foxglove_ready": True,
            }
        )
        raced = source.pop("race_to", None)
        if raced is not None:
            source.update(raced)
        return {"ok": True, "sim_viz": state["sim_viz"]}

    def config(current: dict | None = None) -> dict:
        return resolve_foxglove_config(
            {"NPA_FOXGLOVE_EMBED_SRC": "https://embed.foxglove.dev/"},
            assets_dir=assets,
            origin="https://agent.example",
            sim_viz=(current or state)["sim_viz"],
        )

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
            now_iso=lambda: "2026-08-16T00:00:00+00:00",
            validate_run_id=lambda value: value,
            data_dir=data_dir,
            runs_dir=tmp_path / "runs",
            resolve_artifact=resolve,
        ),
        HTTPException,
    )
    client = TestClient(app)

    first = client.post("/foxglove/export", json=request)
    reused = client.post("/foxglove/export", json=request)
    assert first.status_code == reused.status_code == 200
    assert first.json()["cache_reused"] is False
    assert reused.json()["cache_reused"] is True
    assert calls == {"resolve": 3, "load": 1}

    source.update({"fingerprint": "b" * 64, "bytes": b"\x89MCAP0\r\nnative-v2"})
    changed = client.post("/foxglove/export", json=request)
    assert changed.status_code == 200
    assert changed.json()["cache_reused"] is False
    assert changed.json()["export"]["sha256"] != first.json()["export"]["sha256"]
    assert calls == {"resolve": 5, "load": 2}

    source.update(
        {
            "fingerprint": "c" * 64,
            "bytes": b"\x89MCAP0\r\nnative-racing",
            "race_to": {
                "fingerprint": "d" * 64,
                "bytes": b"\x89MCAP0\r\nnative-after-race",
            },
        }
    )
    raced = client.post("/foxglove/export", json=request)
    assert raced.status_code == 409
    assert "identity changed" in raced.json()["detail"]
    retried = client.post("/foxglove/export", json=request)
    assert retried.status_code == 200
    assert retried.json()["selected_artifact"]["source_fingerprint"] == "d" * 64
    assert calls == {"resolve": 9, "load": 4}


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
