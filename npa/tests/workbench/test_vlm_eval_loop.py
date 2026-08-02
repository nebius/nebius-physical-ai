"""Rollout-SET scoring: the capability `sim-to-real-loop.yaml` implemented in bash.

`vlm-eval run` scores ONE rollout — it discovers frames recursively, so pointing it at a
prefix holding many rollouts blends them into a single score. The retired template therefore
enumerated the rollout directories, called `run` per rollout, and aggregated the results with
`jq` into ``task_success_report.json``. `tests/workbench/test_vlm_eval_loop_e2e.py`
re-implemented the same loop in Python. Neither is reachable from a spec, so the behaviour
moved into the tool; these tests pin the parts the template's `jq` guaranteed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from npa.workbench.vlm_eval import (
    LOOP_REPORT_FILENAME,
    VlmEvalError,
    VlmLoopRollout,
    aggregate_loop_report,
    discover_rollouts,
    evaluate_rollout_set,
    loop_report_uri_for,
)


def _write_rollout(root: Path, name: str, frames: int = 2) -> Path:
    rollout = root / name
    rollout.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        Image.new("RGB", (32, 24), (10 * index, 20, 30)).save(rollout / f"frame_{index:03d}.png")
    return rollout


# ------------------------------------------------------------------------- discovery


def test_discovery_lists_one_entry_per_rollout_directory(tmp_path: Path) -> None:
    root = tmp_path / "rollouts"
    _write_rollout(root, "episode_001")
    _write_rollout(root, "episode_000")

    found = discover_rollouts(str(root))

    # Sorted, like the template's `find ... | sort`.
    assert [Path(item).name for item in found] == ["episode_000", "episode_001"]


def test_discovery_falls_back_to_the_prefix_itself(tmp_path: Path) -> None:
    """The template's `if [[ ${#rollout_paths[@]} -eq 0 ]]` branch: a flat rollout."""

    root = _write_rollout(tmp_path, "single")

    assert discover_rollouts(str(root)) == [str(root)]


def test_discovery_rejects_a_missing_or_empty_input(tmp_path: Path) -> None:
    with pytest.raises(VlmEvalError, match="not found"):
        discover_rollouts(str(tmp_path / "nope"))
    with pytest.raises(VlmEvalError, match="required"):
        discover_rollouts("  ")


# ------------------------------------------------------------------------ aggregation


def _rollout(rollout_id: str, score: float, success: bool) -> VlmLoopRollout:
    return VlmLoopRollout(
        rollout_id=rollout_id,
        success=success,
        score=score,
        rationale="because",
        status="passed" if success else "needs_iteration",
        frame_count=2,
        result_uri=f"s3://b/{rollout_id}/vlm_eval_stub.json",
    )


def test_aggregate_matches_the_templates_jq_report() -> None:
    report = aggregate_loop_report(
        [_rollout("a", 0.9, True), _rollout("b", 0.7, False)],
        model="Qwen/Qwen2-VL-7B-Instruct",
        frame_selection="keyframes",
        success_threshold=0.8,
        output_dir="s3://b/scores/",
    )

    assert report["status"] == "completed"
    assert report["total_rollouts"] == 2
    assert report["passed_rollouts"] == 1
    assert report["success_rate"] == 0.5
    assert report["mean_score"] == pytest.approx(0.8)
    # The gate is the MEAN score, not the pass rate: 0.8 >= 0.8.
    assert report["task_success"] is True
    assert [item["rollout_id"] for item in report["rollouts"]] == ["a", "b"]


def test_aggregate_gate_is_the_mean_not_the_pass_rate() -> None:
    """Every rollout can pass its own threshold while the mean still fails, and vice versa."""

    low_mean = aggregate_loop_report(
        [_rollout("a", 0.5, True), _rollout("b", 0.5, True)],
        model="m",
        frame_selection="keyframes",
        success_threshold=0.8,
        output_dir="out",
    )

    assert low_mean["success_rate"] == 1.0
    assert low_mean["task_success"] is False


def test_aggregate_handles_an_empty_set_without_dividing_by_zero() -> None:
    report = aggregate_loop_report(
        [], model="m", frame_selection="keyframes", success_threshold=0.8, output_dir="out"
    )

    assert report["total_rollouts"] == 0
    assert report["success_rate"] == 0.0
    assert report["mean_score"] == 0.0
    assert report["task_success"] is False


# ----------------------------------------------------------------------- report path


@pytest.mark.parametrize(
    ("output_path", "expected"),
    [
        ("s3://b/scores/", f"s3://b/scores/{LOOP_REPORT_FILENAME}"),
        ("s3://b/scores", f"s3://b/scores/{LOOP_REPORT_FILENAME}"),
        ("s3://b/scores/custom.json", "s3://b/scores/custom.json"),
    ],
)
def test_report_uri_for(output_path: str, expected: str) -> None:
    assert loop_report_uri_for(output_path) == expected


# ------------------------------------------------------------------------ end to end


def test_loop_scores_every_rollout_and_writes_both_artifact_levels(tmp_path: Path) -> None:
    """The stub backend needs no GPU, so the whole loop is checkable offline."""

    root = tmp_path / "rollouts"
    for name in ("episode_000", "episode_001", "episode_002"):
        _write_rollout(root, name)
    scores = tmp_path / "scores"

    report = evaluate_rollout_set(
        input_path=str(root),
        output_path=str(scores),
        backend="stub",
        success_threshold=0.8,
    )

    assert report["total_rollouts"] == 3
    assert {item["rollout_id"] for item in report["rollouts"]} == {
        "episode_000",
        "episode_001",
        "episode_002",
    }
    # One result per rollout ...
    for name in ("episode_000", "episode_001", "episode_002"):
        assert (scores / "rollouts" / name).is_dir()
        written = list((scores / "rollouts" / name).glob("*.json"))
        assert written, f"no per-rollout result for {name}"
    # ... plus the aggregate report the sim-to-real loop gates on.
    report_path = scores / LOOP_REPORT_FILENAME
    assert report_path.is_file()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["total_rollouts"] == 3
    assert report["report_uri"] == str(report_path)
    assert "latency_s" in report


def test_loop_scores_each_rollout_against_its_own_input(tmp_path: Path) -> None:
    """Scoring the prefix as one rollout would produce a single blended score."""

    root = tmp_path / "rollouts"
    _write_rollout(root, "episode_000", frames=1)
    _write_rollout(root, "episode_001", frames=4)
    scores = tmp_path / "scores"

    evaluate_rollout_set(input_path=str(root), output_path=str(scores), backend="stub")

    scored_inputs = {
        json.loads(path.read_text(encoding="utf-8"))["input_path"]
        for path in (scores / "rollouts").rglob("*.json")
    }
    assert scored_inputs == {str(root / "episode_000"), str(root / "episode_001")}
    # The prefix itself was never handed to the scorer.
    assert str(root) not in scored_inputs


# --------------------------------------------------- lerobot eval checkpoint resolution


def test_eval_checkpoint_uri_is_not_mangled_into_a_repo_id() -> None:
    """`--checkpoint-path` must stay a string: Path() collapses `s3://` to `s3:/`.

    Live job 259 fell through to the Hugging Face branch and raised
    `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
    's3:/lerobot-…/policy'` — the double slash had already been eaten by argparse's `type=Path`.
    """

    from npa.workbench.lerobot.policy_container import build_parser

    args = build_parser().parse_args(
        [
            "eval",
            "--checkpoint-path",
            "s3://bucket/prefix/policy/",
            "--output-dir",
            "/tmp/out",
        ]
    )

    assert args.checkpoint_path == "s3://bucket/prefix/policy/"
    assert isinstance(args.checkpoint_path, str)
