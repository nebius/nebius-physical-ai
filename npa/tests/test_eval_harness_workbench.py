"""Tests for the eval_harness workbench surface.

Uses an in-memory fake task (no MuJoCo, no network); the VLM judge's real
client path is exercised with the underlying vlm_eval client monkeypatched,
and its guard rails test the real error paths.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from npa.cli.workbench import eval_harness as eval_harness_cli
from npa.workbench import eval_harness as wb
from npa.workbench.eval_harness import metrics
from npa.workbench.eval_harness.compare import _paired_bootstrap_ci
from npa.workbench.eval_harness.judges import (
    HeuristicJudge,
    VLMJudge,
    make_judge,
)
from npa.workbench.eval_harness.policies import make_policy_factory
from npa.workbench.eval_harness.runner import EpisodeRecord
from npa.workbench.eval_harness.tasks import (
    EvalHarnessError,
    get_task,
    register_task,
)
from npa.workflows.byof import eval_harness_pipeline as pipe


class _FakeEnv:
    """Minimal ManipEnv: succeeds after _succeed_after control steps."""

    def __init__(self, succeed_after: int = 3) -> None:
        self._succeed_after = succeed_after
        self._step = 0
        self.max_steps = 50

    @property
    def action_space(self):
        return {"shape": (2,)}

    @property
    def observation_space(self):
        return {"shape": (3,)}

    def reset(self, seed=None):
        self._step = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action):
        self._step += 1
        success = self._step >= self._succeed_after
        done = success or self._step >= self.max_steps
        return (
            np.zeros(3, dtype=np.float32),
            1.0,
            done,
            {"success": success},
        )


def _fake_policy(obs):
    return np.zeros(2, dtype=np.float32)


_FAKE_TASK = "test_eval_harness_fake_task"
_FAKE_POLICY = f"python:{__name__}:_fake_policy"

register_task(_FAKE_TASK, lambda: _FakeEnv())


def test_sdk_surface_is_workbench_first():
    assert wb.run is pipe.run
    assert wb.compare is pipe.compare
    assert wb.EvalHarnessPipelineError is pipe.EvalHarnessPipelineError
    for name in ("run", "compare", "metrics", "run_policy", "compare_policies"):
        assert name in wb.__all__
    assert not hasattr(wb, "__npa_cli_module__")


def test_cli_registers_run_and_compare():
    names = {c.name for c in eval_harness_cli.app.registered_commands}
    assert "run" in names
    assert "compare" in names


def test_get_task_and_register_roundtrip():
    factory = get_task(_FAKE_TASK)
    env = factory()
    assert isinstance(env, _FakeEnv)


def test_get_task_unknown_raises():
    with pytest.raises(EvalHarnessError, match="unknown eval_harness task"):
        get_task("no.such.task")


def test_wilson_interval_properties():
    lo, hi = metrics.wilson_interval(8, 10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0
    assert metrics.wilson_interval(0, 0) == (0.0, 1.0)
    lo_all, hi_all = metrics.wilson_interval(10, 10)
    assert lo_all > 0.7 and hi_all == 1.0


def test_paired_bootstrap_ci_properties():
    diffs = np.array([1.0, 1.0, 0.0, 0.0])
    lo, hi = _paired_bootstrap_ci(diffs, n_bootstrap=500, seed=0)
    assert lo <= 0.5 <= hi
    assert _paired_bootstrap_ci(np.array([]), n_bootstrap=100, seed=0) == (
        0.0,
        0.0,
    )


def test_summarize_counts_and_wilson():
    records = [
        {
            "index": i,
            "seed": i,
            "success": i % 2 == 0,
            "steps": 5,
            "time_to_success": 3 if i % 2 == 0 else None,
            "total_reward": 5.0,
            "judge": "heuristic",
            "judge_score": 1.0,
            "frames_captured": 0,
        }
        for i in range(4)
    ]
    summary = metrics.summarize(records)
    assert summary["episodes"] == 4
    assert summary["successes"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    lo, hi = summary["wilson_95"]["lo"], summary["wilson_95"]["hi"]
    assert lo <= 0.5 <= hi
    assert summary["median_time_to_success"] == pytest.approx(3.0)


def test_episode_record_as_dict_roundtrip():
    record = EpisodeRecord(
        index=0,
        seed=7,
        success=True,
        steps=3,
        time_to_success=3,
        total_reward=3.0,
        judge="heuristic",
        judge_score=1.0,
        frames_captured=0,
    )
    payload = record.as_dict()
    assert payload["seed"] == 7
    assert metrics.summarize([payload])["success_rate"] == pytest.approx(1.0)


def test_heuristic_judge_passes_through():
    judge = HeuristicJudge()
    assert judge.needs_frames is False
    result = judge.judge(task="t", env_success=True, frames=[])
    assert result.success is True
    assert result.judge == "heuristic"
    assert result.score == 1.0


def test_make_judge_variants():
    assert isinstance(make_judge("heuristic"), HeuristicJudge)
    assert isinstance(make_judge("vlm"), VLMJudge)
    with pytest.raises(EvalHarnessError, match="unknown judge"):
        make_judge("oracle")


def test_vlm_judge_requires_endpoint():
    judge = VLMJudge()
    with pytest.raises(EvalHarnessError, match="vlm-endpoint-url"):
        judge.judge(task="t", env_success=True, frames=[])


def test_vlm_judge_requires_frames():
    judge = VLMJudge(endpoint_url="https://example.invalid/vlm")
    with pytest.raises(EvalHarnessError, match="frames"):
        judge.judge(task="t", env_success=True, frames=[])


def test_vlm_judge_real_client_path(monkeypatch):
    from npa.workbench import vlm_eval

    calls = {}

    def fake_evaluate_vlm(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(score=0.9)

    monkeypatch.setattr(vlm_eval, "evaluate_vlm", fake_evaluate_vlm)
    judge = VLMJudge(endpoint_url="https://example.invalid/vlm")
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    result = judge.judge(task="peg", env_success=False, frames=frames)
    assert result.success is True
    assert result.score == pytest.approx(0.9)
    assert calls["task"] == "eval_harness:peg"


def test_run_policy_end_to_end():
    report = wb.run_policy(
        task=_FAKE_TASK,
        policy=_FAKE_POLICY,
        episodes=3,
        seed=7,
        judge="heuristic",
        max_steps=50,
    )
    assert report.task == _FAKE_TASK
    assert len(report.records) == 3
    assert [r.seed for r in report.records] == [7, 8, 9]
    assert report.summary["success_rate"] == pytest.approx(1.0)
    assert report.summary["successes"] == 3
    assert report.records[0].time_to_success == 3


def test_run_policy_rejects_bad_counts():
    with pytest.raises(EvalHarnessError, match="episodes"):
        wb.run_policy(task=_FAKE_TASK, policy=_FAKE_POLICY, episodes=0)


def test_compare_policies_paired():
    report = wb.compare_policies(
        task=_FAKE_TASK,
        policy_a=_FAKE_POLICY,
        policy_b=_FAKE_POLICY,
        episodes=4,
        seed=11,
        judge="heuristic",
        n_bootstrap=200,
    )
    assert [e["seed"] for e in report.per_episode] == [11, 12, 13, 14]
    assert report.success_rate_diff == pytest.approx(0.0)
    lo, hi = report.bootstrap_95["lo"], report.bootstrap_95["hi"]
    assert lo <= 0.0 <= hi
    assert len(report.per_episode) == 4


def test_pipeline_run_writes_reports(tmp_path):
    out = tmp_path / "reports"
    result = pipe.run(
        task=_FAKE_TASK,
        policy=_FAKE_POLICY,
        output_uri=str(out),
        episodes=2,
        seed=0,
    )
    assert result["status"] == "completed"
    assert result["schema"] == pipe.SCHEMA_RUN
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["success_rate"] == pytest.approx(1.0)
    assert report["records"][0]["judge"] == "heuristic"
    assert (out / "report.md").read_text().startswith("# eval_harness run")
    assert result["artifacts"]["report_json"].endswith("report.json")


def test_pipeline_compare_writes_reports(tmp_path):
    out = tmp_path / "compare"
    result = pipe.compare(
        task=_FAKE_TASK,
        policy_a=_FAKE_POLICY,
        policy_b=_FAKE_POLICY,
        output_uri="file://" + str(out),
        episodes=2,
        seed=0,
    )
    assert result["status"] == "completed"
    assert result["schema"] == pipe.SCHEMA_COMPARE
    compare = json.loads((out / "compare.json").read_text())
    assert compare["success_rate_diff"] == pytest.approx(0.0)
    assert (out / "compare.md").read_text().startswith("# eval_harness compare")


def test_pipeline_run_rejects_bad_judge(tmp_path):
    with pytest.raises(pipe.EvalHarnessPipelineError, match="judge"):
        pipe.run(
            task=_FAKE_TASK,
            policy=_FAKE_POLICY,
            output_uri=str(tmp_path / "x"),
            judge="oracle",
        )


def test_cli_run_end_to_end(tmp_path):
    runner = CliRunner()
    out = tmp_path / "cli-reports"
    result = runner.invoke(
        eval_harness_cli.app,
        [
            "run",
            "--task",
            _FAKE_TASK,
            "--policy",
            _FAKE_POLICY,
            "--episodes",
            "2",
            "--output-uri",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "report.json").exists()


def test_policy_spec_errors():
    with pytest.raises(EvalHarnessError, match="non-empty"):
        make_policy_factory("", _FAKE_TASK)
    with pytest.raises(EvalHarnessError, match="python:"):
        make_policy_factory("python:onlyone", _FAKE_TASK)
    with pytest.raises(EvalHarnessError, match="import"):
        make_policy_factory("python:definitely.not.a.module:fn", _FAKE_TASK)
    with pytest.raises(EvalHarnessError, match="unknown policy spec"):
        make_policy_factory("oracle:yes", _FAKE_TASK)
