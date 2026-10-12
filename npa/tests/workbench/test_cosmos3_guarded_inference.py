"""Deterministic tests for Cosmos 3's native fail-closed guardrail wrapper."""

from __future__ import annotations

import json
import os
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

    def __infer(self, value):
        return value

    def is_safe(self, value):
        self.__infer(value)
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
    with pytest.raises(RuntimeError, match="invalid safety decision"):
        guarded._guarded_safety_check(_runner([BrokenSafetyModel()]), object())

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["discovered"]["generated_media"] == ["BrokenSafetyModel"]
    assert state["evaluated"]["generated_media"] == []
    assert state["effective"] is False


def test_upstream_fail_open_text_error_is_rejected(monkeypatch, tmp_path) -> None:
    class FailOpenQwen3Guard:
        def is_safe(self, value):
            return True, "Unexpected error occurred when running Qwen3Guard guardrail."

    receipt = _reset(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="internal evaluation error"):
        guarded._guarded_safety_check(
            _runner([Blocklist(), FailOpenQwen3Guard()]), "a benign prompt"
        )

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["discovered"]["prompt_input"] == [
        "Blocklist",
        "FailOpenQwen3Guard",
    ]
    assert state["evaluated"]["prompt_input"] == ["Blocklist"]
    assert state["evaluation_details"]["prompt_input"]["FailOpenQwen3Guard"] == {
        "decision": "error"
    }
    assert state["failure"].endswith("evaluation_error")


def test_media_filter_fails_closed_when_a_frame_classifier_error_is_swallowed(
    monkeypatch, tmp_path
) -> None:
    class FailOpenVideoContentSafetyFilter:
        def __infer(self, value):
            if value == "bad-frame":
                raise RuntimeError("classifier failed")
            return value

        def is_safe(self, values):
            for value in values:
                try:
                    self.__infer(value)
                except RuntimeError:
                    pass
            return True, "safe frames detected"

    model = guarded._instrument_media_model(FailOpenVideoContentSafetyFilter())
    receipt = _reset(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="every safety input"):
        guarded._guarded_safety_check(_runner([model]), ["good-frame", "bad-frame"])

    state = json.loads(receipt.read_text(encoding="utf-8"))
    name = "FailOpenVideoContentSafetyFilter"
    assert state["evaluated"]["generated_media"] == []
    assert state["evaluation_details"]["generated_media"][name] == {
        "decision": "safe",
        "attempted_inputs": 2,
        "successful_inputs": 1,
    }
    assert state["failure"].endswith("ineffective_evaluation")


def test_instrumented_media_filter_records_all_successful_frames(
    monkeypatch, tmp_path
) -> None:
    model = guarded._instrument_media_model(VideoContentSafetyFilter())
    receipt = _reset(monkeypatch, tmp_path)

    assert guarded._guarded_safety_check(_runner([model]), ["frame"])[0]

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["evaluation_details"]["generated_media"][
        "VideoContentSafetyFilter"
    ] == {
        "decision": "safe",
        "attempted_inputs": 1,
        "successful_inputs": 1,
    }


def test_explicit_opt_out_is_auditable_but_never_called_effective(
    monkeypatch, tmp_path
) -> None:
    receipt = _reset(monkeypatch, tmp_path, requested=False)
    guarded._finalize_state()

    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["requested"] is False
    assert state["effective"] is False
    assert state["status"] == "explicit_opt_out"


def test_guardrail_tokenizer_is_materialized_before_upstream_import(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench.cosmos import transfer

    cache = tmp_path / "hf-cache"
    previous = tmp_path / "existing-nltk-data"
    observed = {}
    prepared = []
    _reset(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("NLTK_DATA", str(previous))

    def prepare(*, hf_home):
        prepared.append(hf_home)

    monkeypatch.setattr(transfer, "prepare_guardrail_nltk_data", prepare)

    class NativeModule:
        @staticmethod
        def main():
            return None

    def import_native(module_name):
        observed["module"] = module_name
        observed["nltk_data"] = os.environ["NLTK_DATA"]
        return NativeModule

    monkeypatch.setattr(guarded.importlib, "import_module", import_native)
    monkeypatch.setattr(guarded, "_install_fail_closed_guardrails", lambda: None)

    guarded.main()

    roots = observed["nltk_data"].split(os.pathsep)
    assert prepared == [str(cache)]
    assert observed["module"] == guarded.DEFAULT_INFERENCE_MODULE
    assert roots == [
        str(transfer._guardrail_nltk_data_path(str(cache))),
        str(previous),
    ]


def test_action_policy_server_is_allowlisted_for_fail_closed_guardrails() -> None:
    assert guarded.ACTION_POLICY_SERVER_MODULE in guarded.ALLOWED_INFERENCE_MODULES


def _install_action_service_adapter(
    monkeypatch, events: list[str], *, decode_video: bool = True
):
    """Install the adapter around a lightweight copy of the selected direct seam."""

    class PolicyModel:
        input_caption_key = "ai_caption"

        def generate_samples_from_batch(self, data_batch, **_kwargs):
            events.append("generate")
            return {"action": "native-action"}

    class PolicyPipe:
        guardrails = object()

        def _run_text_guardrail(self, _name, prompt):
            events.append("prompt")
            if prompt == "block-prompt":
                raise ValueError("prompt blocked")

        def _run_video_guardrail(self, _name, video):
            events.append("media")
            if video != "decoded-cthw":
                raise ValueError("unexpected decoded video")
            if events[-2] != "generate":
                raise ValueError("media guard ran before native generation")
            if self.block_media:
                raise ValueError("media blocked")
            return video

        block_media = False

    class RobolabPolicyService:
        def __init__(self):
            self.cfg = SimpleNamespace(decode_video=decode_video)
            self.pipe = PolicyPipe()
            self.model = PolicyModel()

        def infer(self, obs):
            data_batch = {"ai_caption": [obs["prompt"]]}
            self.model.generate_samples_from_batch(data_batch)
            outputs = {"action": "native-action"}
            if self.cfg.decode_video:
                outputs["video"] = "native-decoded-thwc"
            return outputs

    source = "selected direct service infer seam"
    monkeypatch.setattr(guarded.inspect, "getsource", lambda _method: source)
    monkeypatch.setattr(
        guarded,
        "ROBOLAB_POLICY_SERVICE_INFER_SHA256",
        guarded._source_sha256(source),
    )
    module = SimpleNamespace(RobolabPolicyService=RobolabPolicyService)
    monkeypatch.setattr(
        guarded,
        "_decoded_video_to_cthw",
        lambda video: "decoded-cthw" if video == "native-decoded-thwc" else None,
    )
    monkeypatch.setattr(
        guarded,
        "_guardrail_video_to_thwc",
        lambda video: "guarded-thwc" if video == "decoded-cthw" else None,
    )
    guarded._install_action_policy_service_guardrails(module)
    return RobolabPolicyService


def test_action_service_adapter_checks_native_prompt_then_decoded_media_and_publishes(
    monkeypatch, tmp_path
) -> None:
    receipt = _reset(monkeypatch, tmp_path)
    events: list[str] = []
    service = _install_action_service_adapter(monkeypatch, events)()

    response = service.infer({"prompt": "move the cube"})

    assert events == ["prompt", "generate", "media"]
    assert response["action"] == "native-action"
    assert response["video"] == "guarded-thwc"
    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["status"] == "passed"
    assert state["effective"] is True
    assert state["action_policy_requests"] == {
        "started": 1,
        "completed": 1,
        "failed": 0,
        "last": {
            "id": 1,
            "status": "passed",
            "prompt_input": True,
            "generated_media": True,
        },
    }


def test_action_service_prompt_refusal_prevents_model_execution(
    monkeypatch, tmp_path
) -> None:
    receipt = _reset(monkeypatch, tmp_path)
    events: list[str] = []
    service = _install_action_service_adapter(monkeypatch, events)()

    with pytest.raises(ValueError, match="prompt blocked"):
        service.infer({"prompt": "block-prompt"})

    assert events == ["prompt"]
    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["effective"] is False
    assert state["action_policy_requests"]["completed"] == 0
    assert state["action_policy_requests"]["failed"] == 1
    assert state["action_policy_requests"]["last"]["status"] == "failed"


def test_action_service_media_refusal_prevents_response_and_marks_request_failed(
    monkeypatch, tmp_path
) -> None:
    receipt = _reset(monkeypatch, tmp_path)
    events: list[str] = []
    service = _install_action_service_adapter(monkeypatch, events)()
    service.pipe.block_media = True

    with pytest.raises(ValueError, match="media blocked"):
        service.infer({"prompt": "move the cube"})

    assert events == ["prompt", "generate", "media"]
    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["effective"] is False
    assert state["action_policy_requests"]["completed"] == 0
    assert state["action_policy_requests"]["failed"] == 1
    assert state["action_policy_requests"]["last"]["generated_media"] is False


def test_action_service_adapter_refuses_a_changed_upstream_infer_seam(
    monkeypatch,
) -> None:
    class ChangedService:
        def infer(self, _obs):
            return {"action": "different"}

    with pytest.raises(RuntimeError, match="differs from the pinned source"):
        guarded._install_action_policy_service_guardrails(
            SimpleNamespace(RobolabPolicyService=ChangedService)
        )


def test_action_service_adapter_requires_actual_video_decode(monkeypatch) -> None:
    events: list[str] = []
    service_class = _install_action_service_adapter(
        monkeypatch, events, decode_video=False
    )

    with pytest.raises(RuntimeError, match="require --decode-video True"):
        service_class()


def test_missing_token_refuses_before_upstream_import(monkeypatch, tmp_path) -> None:
    _reset(monkeypatch, tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        guarded.importlib,
        "import_module",
        lambda _name: pytest.fail("upstream must not import without model access"),
    )

    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        guarded.main()
