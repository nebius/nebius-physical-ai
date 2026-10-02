"""Reject model policies incompatible with the shared paired-judge request."""

import pytest

from npa.workbench import vlm_eval


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
