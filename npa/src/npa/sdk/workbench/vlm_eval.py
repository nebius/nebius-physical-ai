"""Python SDK surfaces for Workbench VLM evaluation."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
from npa.workbench.vlm_eval import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_FRAME_SELECTION,
    DEFAULT_MAX_FRAMES,
    DEFAULT_TIMEOUT_S,
    DEFAULT_VISUAL_REVIEW_RUBRIC,
    VlmJudgeComparisonRequest,
    VlmPreferenceComparisonRequest,
    VlmVisualReviewError,
    VlmVisualReviewReport,
    VlmVisualReviewRequest,
    benchmark_vlm_eval,
    compare_vlm_preference,
    compare_vlm_judges,
    review_visual as _review_visual_backend,
)

run = make_cli_wrapper("npa.cli.workbench.vlm_eval", "run_cmd", "Run VLM evaluation.")
benchmark = benchmark_vlm_eval
compare_judges = compare_vlm_judges
compare_preference = compare_vlm_preference
status = make_cli_wrapper(
    "npa.cli.workbench.vlm_eval", "status_cmd", "Show VLM eval status."
)
list = make_cli_wrapper(
    "npa.cli.workbench.vlm_eval", "list_cmd", "List VLM eval backends."
)
workflow = make_cli_wrapper(
    "npa.cli.workbench.vlm_eval", "workflow_cmd", "Show VLM eval workflow."
)


def review_visual(
    *,
    input_path: str,
    output_path: str,
    model: str,
    task: str,
    baseline_path: str = "",
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    endpoint_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    rubric: str = DEFAULT_VISUAL_REVIEW_RUBRIC,
    rubric_path: str = "",
    objective_evidence_path: str = "",
    matched_view_map_path: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> VlmVisualReviewReport:
    """Run and finalize one private, audit-only rich visual review.

    Args:
        input_path: Current visual source; output_path: private report destination.
        model: Exact hosted model; task: neutral visible-review instruction.
        baseline_path: Optional comparison source; frame_selection: sampling strategy.
        max_frames: Per-source frame bound; endpoint_url: optional explicit endpoint.
        api_key_env: Exact credential environment; rubric: review instructions.
        rubric_path: Optional rubric file; objective_evidence_path: reference JSON.
        matched_view_map_path: Optional matched-view JSON; timeout_s: request timeout.
    Returns:
        The report after its canonical private artifact is finalized.
    Raises:
        VlmVisualReviewError: A bounded review failure with private details omitted.
    """
    try:
        request = VlmVisualReviewRequest(**locals())
        return _review_visual_backend(request)
    except Exception:
        raise VlmVisualReviewError(
            "Visual review failed; inspect private evidence."
        ) from None


__all__ = [
    "VlmJudgeComparisonRequest",
    "VlmPreferenceComparisonRequest",
    "VlmVisualReviewReport",
    "VlmVisualReviewRequest",
    "benchmark",
    "compare_judges",
    "compare_preference",
    "list",
    "review_visual",
    "run",
    "status",
    "workflow",
]
