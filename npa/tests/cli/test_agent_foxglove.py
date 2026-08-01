"""Tier-0 tests for the agent's Foxglove embedded-viewer helpers (0 tokens).

Covers config resolution (including every honest "unavailable" path), the
Foxglove data-source shapes documented by the embedding SDK, the publish-path
safety rules for the unauthenticated CORS data directory, and the text-only
"Describe this" context used because the embed is a cross-origin iframe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.cli.agent_foxglove import (
    FOXGLOVE_ARTIFACT_EXTENSIONS,
    FOXGLOVE_DATA_URL_PREFIX,
    FOXGLOVE_DEFAULT_EMBED_SRC,
    FOXGLOVE_HOST_MODULE_URL,
    FOXGLOVE_SDK_FILES,
    FOXGLOVE_SDK_URL,
    MCAP_MAGIC,
    data_source_for_state,
    describe_foxglove_context,
    foxglove_status_payload,
    is_foxglove_artifact,
    live_data_source,
    live_url_allowed,
    looks_like_mcap,
    prune_published,
    published_data_name,
    remote_file_data_source,
    resolve_foxglove_config,
    sdk_assets_state,
)
from npa.workbench import foxglove as foxglove_pkg


def _install_assets(tmp_path: Path, *, manifest: bool = True) -> Path:
    assets = tmp_path / "sdk"
    assets.mkdir(parents=True, exist_ok=True)
    for name in FOXGLOVE_SDK_FILES:
        (assets / name).write_text("export {};\n", encoding="utf-8")
    if manifest:
        (assets / "npa-sdk-manifest.json").write_text(
            '{"package": "@foxglove/embed", "version": "9.9.9", '
            '"integrity": "sha512-test", "source": "https://registry.example/x.tgz"}',
            encoding="utf-8",
        )
    return assets


# --------------------------------------------------------------------------- #
# constants stay in sync with the shipped workbench package
# --------------------------------------------------------------------------- #


def test_embedded_constants_match_workbench_package() -> None:
    # agent_foxglove is inlined into the agent backend and cannot import npa,
    # so its copies of these constants must be checked here.
    assert FOXGLOVE_SDK_FILES == foxglove_pkg.SDK_FILES
    assert FOXGLOVE_ARTIFACT_EXTENSIONS == foxglove_pkg.FOXGLOVE_ARTIFACT_EXTENSIONS
    assert MCAP_MAGIC == foxglove_pkg.MCAP_MAGIC
    assert FOXGLOVE_DEFAULT_EMBED_SRC == foxglove_pkg.DEFAULT_FOXGLOVE_EMBED_SRC


# --------------------------------------------------------------------------- #
# artifact recognition + publish safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key,expected",
    [
        ("runs/x/reports/session.mcap", True),
        ("runs/x/ROSBAG.BAG", True),
        ("runs/x/data.db3", True),
        ("flight.ulg", True),
        ("flight.ulog", True),
        ("runs/x/reports/sim2real.rrd", False),
        ("runs/x/video.mp4", False),
        ("", False),
    ],
)
def test_is_foxglove_artifact(key: str, expected: bool) -> None:
    assert is_foxglove_artifact(key) is expected


def test_looks_like_mcap() -> None:
    assert looks_like_mcap(MCAP_MAGIC + b"rest")
    assert not looks_like_mcap(b"RIFF0000")
    assert not looks_like_mcap(b"")
    assert not looks_like_mcap(None)


def test_published_data_name_is_random_and_traversal_safe() -> None:
    name = published_data_name("../../etc/passwd/../run one.mcap")
    assert "/" not in name and ".." not in name
    assert name.endswith(".mcap")
    # Unguessable prefix: the path is served without authentication.
    other = published_data_name("../../etc/passwd/../run one.mcap")
    assert name != other
    assert published_data_name("weird.name.bag", token="abc123").startswith("abc123-")
    assert published_data_name("weird.name.bag", token="abc123").endswith(".bag")
    # A non-recording extension still yields a safe, .mcap-suffixed name.
    assert published_data_name("mystery").endswith(".mcap")
    # Token sanitization strips path characters.
    assert "/" not in published_data_name("x.mcap", token="../../evil")


def test_prune_published_keeps_newest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    names = []
    for index in range(5):
        path = data_dir / f"rec{index}.mcap"
        path.write_bytes(MCAP_MAGIC)
        # Deterministic ordering by mtime.
        import os

        os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))
        names.append(path.name)
    (data_dir / "notes.txt").write_text("keep me", encoding="utf-8")

    removed = prune_published(data_dir, keep=2)

    assert sorted(removed) == sorted(names[:3])
    assert {p.name for p in data_dir.iterdir()} == {names[3], names[4], "notes.txt"}
    assert prune_published(tmp_path / "missing", keep=1) == []


# --------------------------------------------------------------------------- #
# data sources (shapes documented by the Foxglove embedding SDK)
# --------------------------------------------------------------------------- #


def test_remote_file_data_source_shape() -> None:
    assert remote_file_data_source(["https://x/a.mcap"]) == {
        "type": "remote-file",
        "urls": ["https://x/a.mcap"],
    }
    assert remote_file_data_source(
        ["https://x/a.mcap", " "], autoplay=True, start_time=12.5
    ) == {
        "type": "remote-file",
        "urls": ["https://x/a.mcap"],
        "autoplay": True,
        "startTime": 12.5,
    }
    assert remote_file_data_source([]) is None


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("wss://foxglove.example.com:8765", True),
        # A public IP literal is fine (8.8.8.8 is globally routable; RFC 5737
        # documentation ranges are classified private by Python 3.12+).
        ("ws://8.8.8.8:8765", True),
        ("http://foxglove.example.com", False),
        ("ws://localhost:8765", False),
        ("ws://127.0.0.1:8765", False),
        ("ws://10.0.0.5:8765", False),
        ("ws://169.254.169.254", False),
        ("ws://metadata.google.internal", False),
        ("ws://cluster.internal", False),
        ("wss://", False),
        ("", False),
    ],
)
def test_live_url_allowed(url: str, allowed: bool) -> None:
    assert live_url_allowed(url) is allowed


def test_live_data_source_defaults_and_rejection() -> None:
    assert live_data_source("wss://robot.example.com:8765") == {
        "type": "live",
        "protocol": "foxglove-websocket",
        "url": "wss://robot.example.com:8765",
    }
    assert live_data_source(
        "wss://robot.example.com:9090", protocol="rosbridge-websocket"
    )["protocol"] == "rosbridge-websocket"
    # Unknown protocol falls back to the Foxglove WebSocket, never crashes.
    assert live_data_source("wss://a.example.com", protocol="nonsense")["protocol"] == (
        "foxglove-websocket"
    )
    assert live_data_source("ws://127.0.0.1:8765") is None


def test_data_source_for_state_prefers_published_recording() -> None:
    state = {"foxglove_url": "/foxglove/data/abc-run.mcap"}
    source = data_source_for_state(state, origin="https://agent.example", env={})
    assert source == {
        "type": "remote-file",
        "urls": ["https://agent.example/foxglove/data/abc-run.mcap"],
    }

    # No recording: fall back to a configured live URL.
    live = data_source_for_state({}, origin="https://agent.example", env={
        "NPA_FOXGLOVE_LIVE_URL": "wss://robot.example.com:8765"
    })
    assert live["type"] == "live"

    assert data_source_for_state({}, origin="https://agent.example", env={}) is None
    # A published recording wins over a live URL (the operator loaded it explicitly).
    both = data_source_for_state(
        state,
        origin="https://agent.example",
        env={"NPA_FOXGLOVE_LIVE_URL": "wss://robot.example.com:8765"},
    )
    assert both["type"] == "remote-file"


# --------------------------------------------------------------------------- #
# config resolution
# --------------------------------------------------------------------------- #


def test_sdk_assets_state_reports_missing_and_incomplete(tmp_path: Path) -> None:
    state = sdk_assets_state(tmp_path / "nope")
    assert not state["ready"] and "not installed" in state["reason"]

    partial = tmp_path / "sdk"
    partial.mkdir()
    (partial / "index.js").write_text("export {};", encoding="utf-8")
    state = sdk_assets_state(partial)
    assert not state["ready"] and "incomplete" in state["reason"]

    assets = _install_assets(tmp_path)
    state = sdk_assets_state(assets)
    assert state["ready"] and state["version"] == "9.9.9"
    assert state["integrity"] == "sha512-test"


def test_resolve_config_available_with_assets(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)
    config = resolve_foxglove_config(
        {
            "NPA_FOXGLOVE_ORG_SLUG": "acme-robotics",
            "NPA_FOXGLOVE_EMBED_SRC": FOXGLOVE_DEFAULT_EMBED_SRC,
        },
        assets_dir=assets,
        origin="https://agent.example",
        sim_viz={"foxglove_url": "/foxglove/data/tok-run.mcap", "run_id": "run-7"},
    )
    assert config["available"] is True
    assert config["reason"] == ""
    assert config["embed_src"] == FOXGLOVE_DEFAULT_EMBED_SRC
    assert config["org_slug"] == "acme-robotics"
    assert config["sdk_url"] == FOXGLOVE_SDK_URL
    assert config["host_module_url"] == FOXGLOVE_HOST_MODULE_URL
    assert config["sdk_version"] == "9.9.9"
    assert config["color_scheme"] == "dark"
    assert config["layout_storage_key"] == "npa-agent-foxglove"
    assert config["data_source"]["urls"] == [
        "https://agent.example/foxglove/data/tok-run.mcap"
    ]
    assert config["recording_url"].startswith("https://agent.example")
    assert config["run_id"] == "run-7"


def test_resolve_config_unavailable_paths(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)

    missing = resolve_foxglove_config({}, assets_dir=tmp_path / "absent")
    assert missing["available"] is False
    assert "not installed" in missing["reason"]
    assert missing["sdk_ready"] is False

    disabled = resolve_foxglove_config({"NPA_FOXGLOVE_ENABLED": "0"}, assets_dir=assets)
    assert disabled["available"] is False
    assert "disabled" in disabled["reason"]

    no_src = resolve_foxglove_config(
        {"NPA_FOXGLOVE_EMBED_SRC": "not-a-url"}, assets_dir=assets
    )
    assert no_src["available"] is False
    assert "embed source" in no_src["reason"]


def test_resolve_config_drops_unsafe_live_url(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)
    config = resolve_foxglove_config(
        {"NPA_FOXGLOVE_LIVE_URL": "ws://127.0.0.1:8765"}, assets_dir=assets
    )
    assert config["live_url"] == ""
    assert config["data_source"] is None


def test_resolve_config_never_echoes_secrets(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)
    config = resolve_foxglove_config(
        {
            "NEBIUS_TOKEN_FACTORY_KEY": "v1.super-secret-token",
            "AWS_SECRET_ACCESS_KEY": "secret-key-value",
            "NPA_FOXGLOVE_ORG_SLUG": "acme",
        },
        assets_dir=assets,
    )
    blob = repr(config)
    assert "super-secret-token" not in blob
    assert "secret-key-value" not in blob


def test_status_payload_and_describe_context(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)
    sim_viz = {
        "foxglove_url": "/foxglove/data/tok-run.mcap",
        "foxglove_ready": True,
        "run_id": "run-7",
        "artifact_key": "run-7/reports/session.mcap",
        "artifact_render": "mcap",
        "mcap_updated_at": "2026-07-30T00:00:00+00:00",
    }
    config = resolve_foxglove_config(
        {"NPA_FOXGLOVE_EMBED_SRC": FOXGLOVE_DEFAULT_EMBED_SRC},
        assets_dir=assets,
        origin="https://agent.example",
        sim_viz=sim_viz,
    )
    status = foxglove_status_payload(config, sim_viz)
    assert status["available"] is True
    assert status["foxglove_ready"] is True
    assert status["data_source_type"] == "remote-file"
    assert status["artifact_key"] == "run-7/reports/session.mcap"
    assert status["recording_url"].endswith("/foxglove/data/tok-run.mcap")

    context = describe_foxglove_context(config, sim_viz)
    assert "cross-origin" in context
    assert "no pixel capture" in context
    assert "run-7" in context
    assert "remote-file" in context


def test_data_url_prefix_is_the_public_path() -> None:
    assert FOXGLOVE_DATA_URL_PREFIX == "/foxglove/data/"
    assert FOXGLOVE_SDK_URL.startswith("/foxglove/")
    assert FOXGLOVE_HOST_MODULE_URL.startswith("/foxglove/")


# --------------------------------------------------------------------------- #
# viewer backend selection (official Foxglove app vs self-hosted OSS viewer)
# --------------------------------------------------------------------------- #


def test_select_viewer_backend_prefers_the_official_app_then_the_oss_viewer() -> None:
    from npa.agent_backend.foxglove import select_viewer_backend

    backend, reason = select_viewer_backend(
        {}, sdk_ready=True, embed_src="https://embed.foxglove.dev/", self_hosted_ready=True
    )
    assert (backend, reason) == ("foxglove-sdk", "")

    # No embed source: fall back to the viewer that can actually render.
    backend, reason = select_viewer_backend(
        {}, sdk_ready=True, embed_src="", self_hosted_ready=True
    )
    assert (backend, reason) == ("self-hosted", "")

    # SDK assets missing but the OSS viewer is up — still a working viewer.
    backend, reason = select_viewer_backend(
        {}, sdk_ready=False, embed_src="https://embed.foxglove.dev/", self_hosted_ready=True
    )
    assert (backend, reason) == ("self-hosted", "")

    # Nothing usable: explain, never pretend.
    backend, reason = select_viewer_backend(
        {}, sdk_ready=False, embed_src="", self_hosted_ready=False
    )
    assert backend == ""
    assert "not installed" in reason


def test_select_viewer_backend_honors_an_operator_override() -> None:
    from npa.agent_backend.foxglove import select_viewer_backend

    forced, _ = select_viewer_backend(
        {"NPA_FOXGLOVE_VIEWER_BACKEND": "self-hosted"},
        sdk_ready=True,
        embed_src="https://embed.foxglove.dev/",
        self_hosted_ready=True,
    )
    assert forced == "self-hosted"

    # An override that cannot be served is ignored rather than breaking the pane.
    fallback, _ = select_viewer_backend(
        {"NPA_FOXGLOVE_VIEWER_BACKEND": "self-hosted"},
        sdk_ready=True,
        embed_src="https://embed.foxglove.dev/",
        self_hosted_ready=False,
    )
    assert fallback == "foxglove-sdk"


def test_self_hosted_viewer_url_uses_the_remote_file_contract() -> None:
    from npa.agent_backend.foxglove import self_hosted_viewer_url

    url = self_hosted_viewer_url("/lichtblick/recordings/sim2real.mcap")
    assert url.startswith("/lichtblick/?ds=remote-file&ds.url=")
    assert "sim2real.mcap" in url
    # No recording yet -> plain viewer, not a malformed data source.
    assert self_hosted_viewer_url("") == "/lichtblick/"


def test_config_exposes_the_self_hosted_backend(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)
    sim_viz = {"mcap_uri": "file:///opt/npa-agent/recordings/sim2real.mcap", "run_id": "run-9"}

    config = resolve_foxglove_config(
        {"NPA_FOXGLOVE_EMBED_SRC": ""},
        assets_dir=assets,
        origin="https://agent.example",
        sim_viz=sim_viz,
        self_hosted_ready=True,
    )

    assert config["available"] is True
    assert config["viewer_backend"] == "self-hosted"
    assert config["self_hosted_ready"] is True
    assert config["self_hosted_url"].startswith("/lichtblick/?ds=remote-file")
    assert config["reason"] == ""


def test_describe_context_names_the_backend_and_capture_ability(tmp_path: Path) -> None:
    assets = _install_assets(tmp_path)
    sim_viz = {"mcap_uri": "file:///opt/npa-agent/recordings/sim2real.mcap", "run_id": "run-9"}

    self_hosted = resolve_foxglove_config(
        {"NPA_FOXGLOVE_EMBED_SRC": ""},
        assets_dir=assets,
        sim_viz=sim_viz,
        self_hosted_ready=True,
    )
    text = describe_foxglove_context(self_hosted, sim_viz)
    assert "viewer_backend: `self-hosted`" in text
    assert "frame capture is possible" in text

    official = resolve_foxglove_config(
        {"NPA_FOXGLOVE_EMBED_SRC": FOXGLOVE_DEFAULT_EMBED_SRC},
        assets_dir=assets,
        sim_viz=sim_viz,
    )
    text = describe_foxglove_context(official, sim_viz)
    assert "viewer_backend: `foxglove-sdk`" in text
    assert "cross-origin" in text


def test_stock_deploy_has_no_implicit_hosted_app(tmp_path: Path) -> None:
    """An unset embed source must not silently point at the account-gated app."""
    assets = _install_assets(tmp_path)

    unset = resolve_foxglove_config({}, assets_dir=assets, sim_viz={}, self_hosted_ready=False)
    assert unset["embed_src"] == ""
    assert unset["viewer_backend"] == ""
    assert "NPA_FOXGLOVE_EMBED_SRC" in unset["reason"]

    # ...but with the OSS viewer running the pane still renders.
    with_oss = resolve_foxglove_config({}, assets_dir=assets, sim_viz={}, self_hosted_ready=True)
    assert with_oss["viewer_backend"] == "self-hosted"
    assert with_oss["available"] is True
