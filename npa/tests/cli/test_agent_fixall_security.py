"""Regression tests for the agent fix-all security / concurrency / UX batch."""

from __future__ import annotations

import threading
from pathlib import Path

from npa.cli.agent_rrd_proxy import (
    file_uri_path_allowed,
    resolve_rrd_proxy_target,
    rrd_proxy_uri_allowed,
)
from npa.cli.agent_s3_guard import (
    configured_agent_s3_buckets,
    s3_uri_in_configured_buckets,
)
from npa.cli.agent_state import StateStore, preserve_latest_namespaces
from npa.cli.agent_workflow import (
    author_workflow_from_goal,
    extract_workflow_name,
    format_workflow_chat_reply,
    _desired_step_count,
)


AGENT_PY = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"
UI_HTML = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"


def test_escape_html_escapes_quotes() -> None:
    source = UI_HTML.read_text(encoding="utf-8")
    assert '.replace(/"/g, "&quot;")' in source
    assert ".replace(/'/g, \"&#39;\")" in source or ".replace(/'/g, '&#39;')" in source


def test_configured_s3_buckets_exclude_listbuckets_noise() -> None:
    buckets = configured_agent_s3_buckets("agent-bucket", "extra-a, extra-b")
    assert buckets == {"agent-bucket", "extra-a", "extra-b"}
    ok, reason = s3_uri_in_configured_buckets(
        "s3://other-bucket/secret",
        primary="agent-bucket",
        extras_csv="extra-a",
    )
    assert ok is False
    assert "configured agent bucket" in reason
    ok2, _ = s3_uri_in_configured_buckets(
        "s3://agent-bucket/path/obj.rrd",
        primary="agent-bucket",
    )
    assert ok2 is True


def test_s3_uri_prefix_gate_optional() -> None:
    ok, reason = s3_uri_in_configured_buckets(
        "s3://agent-bucket/other/key",
        primary="agent-bucket",
        prefix="checkpoints",
    )
    assert ok is False
    assert "prefix" in reason


def test_agent_source_uses_configured_bucket_assert() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert "_assert_s3_uri_in_agent_bucket" in source
    assert "configured_agent_s3_buckets" in source
    assert '_AGENT_S3_GUARD_EMBED' in source


def test_soperator_rejects_spec_path() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert 'detail="Provide spec_yaml, yaml, or spec"' in source
    # Network handler must not read arbitrary paths.
    assert 'body.get("spec_path")' not in source.split("def _soperator_spec_text_from_payload")[1].split(
        "def _soperator_validate_payload"
    )[0]


def test_allow_provision_defaults_false() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert 'allow_provision = bool(body.get("allow_provision", False))' in source
    assert 'dry_run = bool(body.get("dry_run", True))' in source.split("def provision_infra")[1].split(
        "def validate_soperator"
    )[0]
    assert "needs_confirmation" in source.split("def provision_infra")[1].split("def validate_soperator")[0]


def test_state_store_concurrent_distinct_keys(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path, default_factory=lambda: {"a": 0, "b": 0})

    def bump(key: str, n: int = 40) -> None:
        for _ in range(n):
            def _fn(state, k=key):
                state[k] = int(state.get(k) or 0) + 1
            store.mutate(_fn)

    t1 = threading.Thread(target=bump, args=("a",))
    t2 = threading.Thread(target=bump, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    final = store.read()
    assert final["a"] == 40
    assert final["b"] == 40


def test_agent_embeds_state_lock() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert "_STATE_LOCK" in source
    assert "_mutate_state" in source
    assert "_append_chat_turn" in source
    assert '_AGENT_STATE_EMBED' in source


def test_legacy_state_save_preserves_latest_atomic_leisaac_namespace() -> None:
    stale = {
        "chat_history": [{"role": "user", "content": "done"}],
        "leisaac": {
            "run_id": "live-run",
            "bundle_selection": {"robot": {"name": "old-robot"}},
        },
    }
    latest = {
        "chat_history": [],
        "leisaac": {
            "run_id": "live-run",
            "bundle_selection": {
                "robot": {"name": "new-robot"},
                "scene": {"name": "table"},
                "device": {"name": "keyboard"},
            },
        },
    }

    merged = preserve_latest_namespaces(stale, latest, ("leisaac",))

    assert merged["chat_history"] == stale["chat_history"]
    assert merged["leisaac"] == latest["leisaac"]
    assert merged["leisaac"] is not latest["leisaac"]


def test_rendered_agent_legacy_save_preserves_atomic_leisaac_namespace() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    block = source.split("def _save_state(state: dict) -> None:", 1)[1].split(
        "def _mutate_state", 1
    )[0]
    assert "preserve_latest_namespaces(state, latest, (\"leisaac\",))" in block


def test_resolve_workflow_yaml_no_draft_fallback() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    block = source.split("def _resolve_workflow_yaml")[1].split("def _agent_npa_ready")[0]
    assert "_workflow_draft_from_state" not in block
    assert 'payload.get("yaml")' in block


def test_subprocess_timeout_handled() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert "subprocess.TimeoutExpired" in source
    assert "live sim2real submit timed out" in source
    assert "NPA command timed out" in source


def test_empty_llm_reply_message() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert "Model returned no content." in source


def test_extract_workflow_name() -> None:
    assert extract_workflow_name("Name it cosmos-video-aug please") == "cosmos-video-aug"
    assert extract_workflow_name("create a workflow called my_pipe") == "my-pipe"
    assert extract_workflow_name("just make something") == ""


def test_author_workflow_honors_name_and_stages() -> None:
    catalog = frozenset(
        {
            "workbench.cosmos2.transfer",
            "workbench.token_factory.generate",
            "workbench.lancedb.import_bdd100k",
            "workbench.sim2real_envgen.raw_shard",
            "workbench.dataset.ingest",
        }
    )
    result = author_workflow_from_goal(
        "generate → augment → ingest into LanceDB; name it cosmos-video-aug",
        tool_refs=catalog,
    )
    assert result.get("name") == "cosmos-video-aug"
    assert "cosmos-video-aug" in (result.get("yaml") or "")
    assert int(result.get("desired_steps") or 0) >= 3
    states = result.get("states") or []
    tool_refs = result.get("tool_refs") or []
    note = str(result.get("dropped_stages_note") or "")
    assert len(states) >= 3 or len(tool_refs) >= 3 or note
    reply = format_workflow_chat_reply(
        str(result.get("yaml") or "apiVersion: npa.workflow/v0.0.1\nmetadata:\n  name: cosmos-video-aug\n"),
        result.get("validation")
        or {"ok": True, "name": "cosmos-video-aug", "states": ["generate", "augment", "ingest"]},
        template="catalog-composed",
        plan=result.get("plan") or {"ok": True, "steps": [1, 2, 3]},
        runnable=bool(result.get("runnable")),
        dropped_stages_note=note,
    )
    header = reply.split("\n", 1)[0]
    assert "Sim2Real" not in header
    assert "pipeline" in header.lower() or "step" in header.lower()


def test_desired_step_count_infers_arrows() -> None:
    assert _desired_step_count("generate → augment → ingest into LanceDB") >= 3


def test_load_run_rejects_unknown_ids() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    block = source.split('@app.post("/sim-viz/load-run")')[1].split('@app.get("/sim-viz/recordings")')[0]
    assert "run_id not found" in block
    assert "Never invent phantom run ids" in block or "sim2real_runs" in block
    assert "if artifacts:" in block
    assert '"stage": "artifacts_available"' in block
    assert "conditional tabs (LeIsaac in" in block


def test_franka_wire_preserves_selection() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    block = source.split("def _wire_franka_demo")[1].split("def _wire_sim2real_run_preview")[0]
    assert "Preserve operator-posted custom URIs" in block
    assert 'state["selection"] = _stock_franka_selection()' not in block or "if not current" in block


def test_placeholder_chat_title_and_session_404() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert "_is_placeholder_chat_title" in source
    assert "_lookup_chat_session" in source
    assert 'detail=f"chat session not found:' in source
    assert '"model": "grounded"' in source
    assert 'lower() in {{"", "auto"}}' in source or 'in {"", "auto"}' in source or 'in {{"", "auto"}}' in source


def test_ui_frontend_fixes() -> None:
    ui = UI_HTML.read_text(encoding="utf-8")
    assert 'classList.toggle("mobile-agent", detectMobileLayout())' in ui
    assert "activeMainTab === \"rerun\"" in ui.split("function startPeriodicRefresh")[1].split("function startApp")[0]
    assert "selectedRunIdFromUi()" in ui.split("async function loadRunData")[1].split("async function selectCamera")[0]
    assert "MAX_CLIENT_CHAT_TURNS" in ui
    assert "chatHistory.slice(-MAX_CLIENT_CHAT_TURNS)" in ui
    assert "Warning: Rerun recording/iframe did not reach SUCCESS" in ui
    assert "event.origin !== window.location.origin" in ui
    assert "function renderCameraCards" not in ui
    assert "async function previewCamera" not in ui


def test_rrd_resolve_once_and_file_jail(tmp_path: Path) -> None:
    public = ".".join(("1", "1", "1", "1"))

    def _gai(host, port, *args, **kwargs):
        if host == "cdn.example.test":
            return [(0, 0, 0, "", (public, port))]
        raise OSError("nxdomain")

    from unittest.mock import patch

    with patch("npa.cli.agent_rrd_proxy.socket.getaddrinfo", side_effect=_gai):
        allowed, fetch_url, host = resolve_rrd_proxy_target("https://cdn.example.test/a.rrd")
        assert allowed is True
        assert host == "cdn.example.test"
        assert public in fetch_url
        assert rrd_proxy_uri_allowed("https://cdn.example.test/a.rrd")

    recordings = tmp_path / "recordings"
    recordings.mkdir()
    good = recordings / "sim2real.rrd"
    good.write_bytes(b"rrd")
    outside = tmp_path / "secret.rrd"
    outside.write_bytes(b"nope")
    assert file_uri_path_allowed(f"file://{good}", allowed_paths=(str(recordings),))
    assert not file_uri_path_allowed(f"file://{outside}", allowed_paths=(str(recordings),))


def test_embedded_helpers_in_render_chain() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")
    assert "_embedded_agent_state_source" in source
    assert "_embedded_agent_s3_guard_source" in source
    assert "resolve_rrd_proxy_target" in source
    assert "file_uri_path_allowed" in source
