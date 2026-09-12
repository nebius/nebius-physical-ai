"""Deterministic tests for Cosmos 3's native fail-closed guardrail wrapper."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from npa.workbench.cosmos import guarded_inference as guarded


class Blocklist:
    def is_safe(self, value):
        return True, "safe"


class Qwen3Guard:
    def is_safe(self, value):
        return True, "safe"


class VideoContentSafetyFilter:
    def __init__(self, *, offload_model_to_cpu=False):
        self.offload_model_to_cpu = offload_model_to_cpu

    def is_safe(self, value):
        return True, "safe"


def _runner(models):
    return SimpleNamespace(
        safety_models=models,
        generic_block_msg="",
        generic_safe_msg="safe",
    )


def _reset(monkeypatch, tmp_path, *, requested: bool = True):
    receipt = tmp_path / "guardrails.json"
    monkeypatch.setenv(guarded.GUARDRAIL_RECEIPT_ENV, str(receipt))
    monkeypatch.setattr(guarded, "_STATE", guarded._new_state(requested))
    return receipt


def test_empty_safety_model_runner_fails_closed(monkeypatch, tmp_path) -> None:
    receipt = _reset(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="no safety model was discovered"):
        guarded._guarded_safety_check(_runner([]), "a benign robotics prompt")

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["requested"] is True
    assert state["discovered"]["prompt_input"] == []
    assert state["evaluated"]["prompt_input"] == []
    assert state["effective"] is False
    assert state["status"] == "ineffective"
    assert state["failure"] == "prompt_input:no_safety_models"


def test_empty_upstream_media_runner_restores_shipped_safety_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        guarded, "_media_safety_model_class", lambda: VideoContentSafetyFilter
    )
    restored = guarded._restore_media_safety(
        lambda *, offload_model_to_cpu=False: _runner([])
    )(offload_model_to_cpu=True)

    assert len(restored.safety_models) == 1
    assert isinstance(restored.safety_models[0], VideoContentSafetyFilter)
    assert restored.safety_models[0].offload_model_to_cpu is True


def test_nonempty_upstream_media_runner_is_not_replaced(monkeypatch) -> None:
    original = VideoContentSafetyFilter()
    monkeypatch.setattr(
        guarded,
        "_media_safety_model_class",
        lambda: pytest.fail("fallback must not load for a nonempty upstream runner"),
    )
    restored = guarded._restore_media_safety(
        lambda *, offload_model_to_cpu=False: _runner([original])
    )()

    assert restored.safety_models == [original]


def test_receipt_is_effective_only_after_both_safety_stages_run(
    monkeypatch, tmp_path
) -> None:
    receipt = _reset(monkeypatch, tmp_path)

    assert guarded._guarded_safety_check(
        _runner([Blocklist(), Qwen3Guard()]), "a benign robotics prompt"
    )[0]
    assert guarded._guarded_safety_check(
        _runner([VideoContentSafetyFilter()]), object()
    )[0]
    guarded._finalize_state()

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["discovered"] == state["evaluated"]
    assert state["effective"] is True
    assert state["status"] == "passed"


def test_invalid_safety_decision_is_not_counted_as_evaluated(
    monkeypatch, tmp_path
) -> None:
    class BrokenSafetyModel:
        def is_safe(self, value):
            return "yes", "not a boolean"

    receipt = _reset(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="non-boolean"):
        guarded._guarded_safety_check(_runner([BrokenSafetyModel()]), object())

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["discovered"]["generated_media"] == ["BrokenSafetyModel"]
    assert state["evaluated"]["generated_media"] == []
    assert state["effective"] is False


def test_explicit_opt_out_is_auditable_but_never_called_effective(
    monkeypatch, tmp_path
) -> None:
    receipt = _reset(monkeypatch, tmp_path, requested=False)
    guarded._finalize_state()

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["requested"] is False
    assert state["effective"] is False
    assert state["status"] == "explicit_opt_out"
