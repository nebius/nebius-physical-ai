"""Reject model policies incompatible with the shared paired-judge request."""

import pytest

from npa.clients.token_factory import TokenFactoryChatProfile
from npa.live_verification.vlm_audit_controls import PAIRED_MODELS
from npa.workbench import vlm_eval


def test_scheduled_pair_intentionally_uses_common_not_scalar_profile_body():
    profiles = [vlm_eval.token_factory_chat_profile(model) for model in PAIRED_MODELS]
    assert all(profile.include_temperature for profile in profiles)
    assert profiles[0].default_extra() != profiles[1].default_extra()
    assert profiles[0].use_vlm_response_format != profiles[1].use_vlm_response_format
    assert vlm_eval._comparison_models(*PAIRED_MODELS) == PAIRED_MODELS

    common = vlm_eval._common_hosted_request(prompt="frozen prompt", frames=[])
    assert common == {
        "temperature": 0,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "frozen prompt"}]}
        ],
    }
    requests = [vlm_eval._request_for_model(common, model) for model in PAIRED_MODELS]
    assert all(
        set(request) == {"model", "temperature", "messages"} for request in requests
    )
    assert vlm_eval._assert_model_only_request_difference(
        requests
    ) == vlm_eval._sha256_json(common)


@pytest.mark.parametrize("kimi_first", [True, False])
def test_judge_profile_validation_precedes_preparation(
    monkeypatch, tmp_path, kimi_first
):
    def unexpected(*_args, **_kwargs):
        pytest.fail("incompatible paired model reached a preparation boundary")

    for name in ("_materialized_input", "_resolve_api_key", "_post_comparison_request"):
        monkeypatch.setattr(vlm_eval, name, unexpected)
    models = ["moonshotai/Kimi-K3", "MiniMaxAI/MiniMax-M3"]
    if not kimi_first:
        models.reverse()
    request = vlm_eval.VlmJudgeComparisonRequest(
        input_path=str(tmp_path / "input"),
        output_path=str(tmp_path / "report"),
        primary_model=models[0],
        secondary_model=models[1],
    )
    with pytest.raises(vlm_eval.VlmEvalError, match="shared temperature request"):
        vlm_eval.compare_vlm_judges(request)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("incompatible_first", [True, False])
def test_shared_profile_incompatibility_precedes_preparation(
    monkeypatch, tmp_path, incompatible_first
):
    def unexpected(*_args, **_kwargs):
        pytest.fail("incompatible profile reached a preparation boundary")

    for name in ("_materialized_input", "_resolve_api_key", "_post_comparison_request"):
        monkeypatch.setattr(vlm_eval, name, unexpected)
    profiles = {
        "compatible-vision": TokenFactoryChatProfile(),
        "incompatible-vision": TokenFactoryChatProfile(include_temperature=False),
    }
    monkeypatch.setattr(vlm_eval, "token_factory_chat_profile", profiles.__getitem__)
    models = ["incompatible-vision", "compatible-vision"]
    if not incompatible_first:
        models.reverse()
    request = vlm_eval.VlmJudgeComparisonRequest(
        input_path=str(tmp_path / "input"),
        output_path=str(tmp_path / "report"),
        primary_model=models[0],
        secondary_model=models[1],
    )
    with pytest.raises(vlm_eval.VlmEvalError, match="shared temperature request"):
        vlm_eval.compare_vlm_judges(request)
    assert not (tmp_path / "report").exists()
