"""Exercise improvement hooks in the actual rendered/shipped backend."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

from .test_agent_backend_render import _clear_rendered_agent_backend_modules, _import_rendered_backend


@pytest.fixture
def backend(monkeypatch, tmp_path):
    _clear_rendered_agent_backend_modules()
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name="npa_improvement_hook_backend")
    state = {}
    monkeypatch.setattr(module, "_load_state", lambda: state)
    monkeypatch.setattr(module, "_save_state", lambda value: state.update(value))
    monkeypatch.setattr(module, "TRACE_DIR", tmp_path / "traces")
    import agent_backend.trajectory as trajectory
    records = []
    monkeypatch.setattr(trajectory, "flush_outbox", lambda **kwargs: None)
    monkeypatch.setattr(trajectory, "emit_trajectory", lambda **kwargs: records.append(kwargs))
    from agent_backend.improvements import ImprovementScope, ImprovementStore
    from agent_backend.improvement_routes import ImprovementRuntime
    repository = tmp_path / "source"
    repository.mkdir()
    scopes = [ImprovementScope(scope_id=name, component=name, files=(name + ".py",), base_revision="b" * 40,
                               required_checks=("reproducer",)) for name in ("retrieval_search", "sim2real-drive")]
    store = ImprovementStore(tmp_path / "queue", repository=repository, evidence_directory=tmp_path / "evidence",
                             scopes=scopes, reviewers=("independent-reviewer",))
    monkeypatch.setattr(module, "_agent_improvements", ImprovementRuntime(lambda: store))
    yield module, store, records
    sys.modules.pop("npa_improvement_hook_backend", None)
    _clear_rendered_agent_backend_modules()


def _failed_action(*args, **kwargs):
    return {"ok": False, "steps": [{"phase": "call", "tool": "retrieval_search", "status": "error",
                                    "args": {"query": "synthetic input"}, "observation": {"error": "synthetic failure"}}],
            "reply": "A synthetic tool failure occurred.", "usage": {"total_tokens": 0}}


def _assert_bound_observation(store, records):
    assert len(records) == 1
    items = store.list_items()
    assert len(items) == 1
    occurrence = store.history(items[0]["id"])["occurrences"][0]
    expected = hashlib.sha256(json.dumps(records[0]["episode_id"]).encode()).hexdigest()
    assert occurrence["episode_ref"] == expected
    assert occurrence["session_ref"]
    return items[0]


def test_direct_action_hook_preserves_episode_and_original_failure(backend, monkeypatch):
    module, store, records = backend
    seen = {}

    def action(*args, **kwargs):
        seen.update(kwargs)
        return _failed_action()

    monkeypatch.setattr(module, "run_action_loop", action)
    result = module.agent_act({"goal": "inspect retrieval_search", "session_id": "same-parent"})
    assert result["ok"] is False
    assert result["improvements"]["status"] == "recorded"
    assert "live_context" in seen
    _assert_bound_observation(store, records)


def test_semantic_action_chat_hook_preserves_trace_and_episode(backend, monkeypatch):
    module, store, records = backend
    monkeypatch.setattr(module, "_agent_chat_with_tools", lambda **kwargs: None)
    monkeypatch.setattr(module, "_maybe_origin_reply", lambda *args, **kwargs: ("", []))
    monkeypatch.setattr(module, "match_chat_intent", lambda *args: None)
    monkeypatch.setattr(module, "_semantic_route", lambda *args: {"mode": "action", "tokens": 0})
    monkeypatch.setattr(module, "run_chat_action_loop", _failed_action)
    result = module.chat({"messages": [{"role": "user", "content": "inspect retrieval_search"}], "session_id": "same-parent"})
    assert result["ok"] is False
    assert result["tier"] == "semantic-action"
    assert result["improvements"]["status"] == "recorded"
    assert len(module._recent_agent_traces()) == 1
    _assert_bound_observation(store, records)


def test_drive_hook_preserves_episode_and_does_not_repeat_launch(backend, monkeypatch):
    module, store, records = backend
    calls = []

    def drive(*args, **kwargs):
        calls.append(1)
        return {"ok": False, "iterations": [{"status": "error", "error": "synthetic launch failure"}],
                "reply": "Failed", "stopped_reason": "error"}

    monkeypatch.setattr(module, "drive_sim2real_loop", drive)
    handler = next(route.endpoint for route in module.app.routes if route.path == "/agent/sim2real/drive")
    result = handler({"goal": "diagnose", "session_id": "same-parent"})
    assert result["ok"] is False and calls == [1]
    assert result["improvements"]["status"] == "recorded"
    _assert_bound_observation(store, records)


def test_rendered_context_uses_known_lesson_without_model_call(backend, monkeypatch):
    module, store, records = backend
    from agent_backend.improvement_routes import ImprovementRuntime

    class VerifiedStore:
        scopes = {"trajectory": None}

        def matching_verified_lessons(self, targets):
            return [{"item_id": "a" * 64, "lesson_key": "trajectory_observation_conservation"}] if "trajectory" in targets else []

    monkeypatch.setattr(module, "_agent_improvements", ImprovementRuntime(lambda: VerifiedStore()))
    monkeypatch.setattr(module, "_skill_excerpt", lambda name: "")
    names, context = module._resolve_skill_context(user_text="inspect trajectory", intent=None)
    assert "Preserve action phase, status and args" in context
    assert "verified-lesson" in context
    assert module._resolve_skill_context(user_text="select GPU", intent=None)[1] == ""
