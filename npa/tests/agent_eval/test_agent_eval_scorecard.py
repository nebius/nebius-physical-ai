"""Agent task-eval scorecard test (Phase E).

Runs the mocked task-completion suite (0 tokens, CI-safe), asserts a competitive
bar, and defends a committed scorecard baseline. A live submit → ingest → ask
variant is gated behind ``NPA_AGENT_CHAT_LIVE=1`` (Tier-2 convention).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from agent_eval.harness import (
    assert_scorecard_not_regressed,
    run_operate_eval,
    run_suite,
    scorecard_regressions,
)
from agent_eval.scenarios import SCENARIOS

ARTIFACT_DIR = Path(__file__).with_name("_artifacts")
SCORECARD_PATH = ARTIFACT_DIR / "scorecard.json"


def test_agent_eval_scorecard_meets_bar():
    report = run_suite()
    scorecard = report["scorecard"]

    # Competitive bar: the mocked suite must fully pass and stay cheap.
    assert scorecard["total"] == len(SCENARIOS)
    assert scorecard["success_rate"] >= 0.9, report["results"]
    # avg_tokens is simulated (mocked planner); it must stay small.
    assert scorecard["avg_tokens"] <= 5.0, scorecard


def _baseline_scorecard() -> dict:
    return json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))["scorecard"]


def test_agent_eval_scorecard_does_not_regress_from_committed_baseline():
    current = run_suite()["scorecard"]
    assert_scorecard_not_regressed(current, _baseline_scorecard())


def test_agent_eval_regression_gate_rejects_negative_control():
    baseline = _baseline_scorecard()
    degraded = {
        **baseline,
        "success_rate": baseline["success_rate"] - 0.1,
        "avg_steps": baseline["avg_steps"] + 1.0,
        "avg_tokens": baseline["avg_tokens"] + 1.0,
    }
    regressions = scorecard_regressions(degraded, baseline)
    assert {item.split("=", 1)[0] for item in regressions} == {
        "success_rate",
        "avg_steps",
        "avg_tokens",
    }
    with pytest.raises(AssertionError, match="scorecard regressed"):
        assert_scorecard_not_regressed(degraded, baseline)


def test_every_scenario_kind_is_exercised():
    report = run_suite()
    kinds = {r["kind"] for r in report["results"]}
    assert {"grounded", "workflow", "action_loop", "sim2real_loop", "semantic"} <= kinds


def test_no_task_crashes():
    report = run_suite()
    crashed = [r for r in report["results"] if str(r.get("detail", "")).startswith("error:")]
    assert not crashed, crashed


def test_operate_eval_mocked_round_trip_is_grounded(tmp_path: Path):
    from npa.cli.agent_actions import summarize_observations
    from npa.workbench.insights.analytics import query_metrics
    from npa.workbench.insights.schemas import IngestRunRequest, QueryRequest
    from npa.workbench.insights.store import ingest_run

    run_id = "operate-observed"
    fixture = tmp_path / "fixture"
    store = str(tmp_path / "store")
    empty_store = str(tmp_path / "empty-store")

    def submit(observed_run_id: str) -> dict:
        fixture.mkdir()
        (fixture / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "npa.dataset.manifest.v1",
                    "run_id": observed_run_id,
                    "dataset_id": "operate-eval",
                    "version": "v1",
                    "record_count": 2,
                    "quality_stats": {
                        "record_count": 2,
                        "mean_completeness": 1.0,
                        "corrupt_count": 0,
                        "modalities": ["camera"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return {"status": "completed", "run_id": observed_run_id, "run_uri": str(fixture)}

    def ingest(submission: dict, output_uri: str, observed_run_id: str) -> dict:
        response = ingest_run(
            IngestRunRequest(
                input_uri=str(submission["run_uri"]),
                output_uri=output_uri,
                workflow_run=observed_run_id,
            )
        )
        return response.model_dump(mode="json")

    def observe(input_uri: str) -> dict:
        return query_metrics(QueryRequest(input_uri=input_uri)).model_dump(mode="json")

    def ask(input_uri: str) -> dict:
        result = observe(input_uri)
        reply = summarize_observations([{"tool": "insights_query", "result": result}])
        return {"reply": reply, "tools_used": ["insights_query"], "usage": {"total_tokens": 0}}

    report = run_operate_eval(
        run_id=run_id,
        empty_store_uri=empty_store,
        store_uri=store,
        submit=submit,
        ingest=ingest,
        observe=observe,
        ask=ask,
    )
    assert report["success"], report
    assert "no runs found" in report["empty"]["reply"].lower()
    assert report["observed_run_ids"] == [run_id]
    assert run_id in report["populated"]["reply"]
    assert report["scorecard"]["avg_tokens"] == 0.0


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_CHAT_LIVE") != "1",
    reason="live agent eval gated behind NPA_AGENT_CHAT_LIVE=1 (cheapest pinned model)",
)
def test_agent_eval_live_operate_round_trip():  # pragma: no cover - opt-in live variant
    """Submit CPU workflow → ingest Insights → ask deployed agent → verify evidence."""
    import httpx
    from npa.sdk.workbench.insights import query

    base = os.environ.get("NPA_AGENT_URL", "").rstrip("/")
    bucket = os.environ.get("NPA_AGENT_EVAL_BUCKET", "").strip()
    fixture_uri = os.environ.get("NPA_AGENT_EVAL_FIXTURE_URI", "").strip()
    missing = [
        name
        for name, value in (
            ("NPA_AGENT_URL", base),
            ("NPA_AGENT_EVAL_BUCKET", bucket),
            ("NPA_AGENT_EVAL_FIXTURE_URI", fixture_uri),
        )
        if not value
    ]
    if missing:
        pytest.skip("live operate-eval requires " + ", ".join(missing))

    run_id = f"agent-eval-{uuid.uuid4().hex[:12]}"
    prefix = f"agent-eval/{run_id}"
    store_uri = f"s3://{bucket}/{prefix}/store/"
    empty_store_uri = f"s3://{bucket}/{prefix}/empty/"
    workflow_path = Path(__file__).parents[2] / "workflows/workbench/npa-workflows/insights-smoke.yaml"

    def submit(observed_run_id: str) -> dict:
        npa_executable = str(Path(sys.executable).with_name("npa"))
        command = [
            npa_executable,
            "workbench",
            "workflow",
            "submit",
            str(workflow_path),
            "--runtime",
            "--run-id",
            observed_run_id,
            "--output-format",
            "json",
        ]
        variables = {
            "bucket": bucket,
            "prefix": prefix,
            "run_prefix_uri": fixture_uri,
            "insights_store_uri": store_uri,
            "comparison_uri": f"s3://{bucket}/{prefix}/comparison/",
            "dashboard_uri": f"s3://{bucket}/{prefix}/dashboard/",
        }
        for key, value in variables.items():
            command.extend(["--var", f"{key}={value}"])
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
            if os.environ.get(name):
                command.extend(["--secret-env", name])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise AssertionError(completed.stderr or completed.stdout)
        return {"status": "completed", "run_id": observed_run_id, "stdout": completed.stdout}

    def ingest(submission: dict, output_uri: str, observed_run_id: str) -> dict:
        # The submitted workflow's first real state is insights ingest-run. Query
        # the resulting store to make that observation explicit before asking.
        observed = query(input_uri=output_uri)
        return {
            "status": submission.get("status"),
            "run_id": observed_run_id,
            "recorded_count": observed.count,
        }

    def observe(input_uri: str) -> dict:
        return query(input_uri=input_uri).model_dump(mode="json")

    user = os.environ.get("AGENT_USER", "")
    password = os.environ.get("AGENT_PASSWORD", "")

    def ask(input_uri: str) -> dict:
        response = httpx.post(
            f"{base}/api/agent/act",
            json={
                "goal": (
                    "Use insights_query with input_uri "
                    f"{input_uri} and report only run ids returned by the store."
                )
            },
            auth=(user, password) if user else None,
            timeout=None,
            verify=os.environ.get("NPA_AGENT_TLS_VERIFY") == "1",
        )
        response.raise_for_status()
        return response.json()

    report = run_operate_eval(
        run_id=run_id,
        empty_store_uri=empty_store_uri,
        store_uri=store_uri,
        submit=submit,
        ingest=ingest,
        observe=observe,
        ask=ask,
    )
    assert report["success"], report


if __name__ == "__main__":  # pragma: no cover
    report = run_suite()
    print(json.dumps(report["scorecard"], indent=2))
