from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.predicates import evaluate_predicate
from npa.orchestration.npa_workflow.run_state import RunManifest, RunStateStore
from npa.orchestration.npa_workflow.spec import LoopSpec, StateSpec, validate_spec

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
RUNNER = CliRunner()


def test_promote_checkpoint_plans_finalize_once() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    plan = build_plan(spec, run_id="audit-promote", assume_decision="promote_checkpoint")
    states = [step.state for step in plan.steps]
    assert states.count("finalize") == 1
    assert len(plan.steps) == 11


def test_loop_back_assume_normalizes_for_plan_predicates() -> None:
    context = {"last_decision": "loop_back"}
    assert evaluate_predicate("loop_back", context) is True


def test_unbounded_cycle_rejected_at_validate() -> None:
    spec = load_spec(SPECS / "tokenfactory-rollout-judge.yaml")
    spec.states["reason-scene"].next = "judge-rollouts"
    spec.states["judge-rollouts"].terminal = False
    spec.states["judge-rollouts"].next = "reason-scene"
    spec.states["done"] = StateSpec(name="done", terminal=True)
    with pytest.raises(NpaWorkflowError, match="unbounded control-flow cycle"):
        validate_spec(spec)


def test_transition_cycle_not_whitelisted_by_loop_block() -> None:
    spec = load_spec(SPECS / "tokenfactory-rollout-judge.yaml")
    spec.states["reason-scene"].loop = LoopSpec(max=3)
    spec.states["reason-scene"].next = "judge-rollouts"
    spec.states["judge-rollouts"].terminal = False
    spec.states["judge-rollouts"].next = "reason-scene"
    spec.states["done"] = StateSpec(name="done", terminal=True)
    with pytest.raises(NpaWorkflowError, match="unbounded control-flow cycle"):
        validate_spec(spec)


def test_zero_loop_max_rejected_at_validate() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    spec.config["inner_iterations"] = 0
    with pytest.raises(NpaWorkflowError, match="loop.max must be >= 1"):
        validate_spec(spec)


def test_missing_config_token_rejected_at_validate() -> None:
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    spec.states["score-rollouts"].outputs = [
        __import__(
            "npa.orchestration.npa_workflow.spec", fromlist=["ArtifactSpec"]
        ).ArtifactSpec(uri="{{config.does_not_exist}}")
    ]
    with pytest.raises(NpaWorkflowError, match="does_not_exist"):
        validate_spec(spec)


def test_non_int_loop_max_rejected_at_validate() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    spec.config["inner_iterations"] = "not-an-int"
    with pytest.raises(NpaWorkflowError, match="must be an integer loop bound"):
        validate_spec(spec)


def test_state_output_token_resolves_during_plan(tmp_path: Path) -> None:
    path = tmp_path / "state-token.yaml"
    path.write_text(
        """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: state-token-plan
config:
  bucket: example-bucket
initial: produce
states:
  produce:
    run:
      shell: echo ok
    outputs:
      - uri: s3://{{config.bucket}}/artifact.json
    next: consume
  consume:
    run:
      shell: cat {{state.produce.uri}}
    terminal: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = load_spec(path)
    plan = build_plan(spec, run_id="state-token-run")
    consume = next(step for step in plan.steps if step.state == "consume")
    assert consume.shell == "cat s3://example-bucket/artifact.json"


def test_execution_step_limit_guard(monkeypatch) -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execution_step_limit",
        lambda _spec: 4,
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: {"state": step.state, "status": "ok"},
    )
    with pytest.raises(NpaWorkflowError, match="execution exceeded step limit"):
        run_workflow(
            spec,
            run_id="depth-guard",
            execute=True,
            assume_decision="loop_back",
        )


def test_failed_execute_persists_failed_manifest(monkeypatch) -> None:
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    store: dict[tuple[str, str], bytes] = {}

    def writer(bucket: str, key: str, body: bytes) -> None:
        store[(bucket, key)] = body

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/fail-demo",
        writer=writer,
        reader=lambda _b, _k: (_ for _ in ()).throw(FileNotFoundError()),
    )

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: (_ for _ in ()).throw(
            NpaWorkflowError(f"boom at {step.state}")
        ),
    )

    with pytest.raises(NpaWorkflowError, match="boom"):
        run_workflow(
            spec,
            run_id="fail-demo",
            execute=True,
            state_store=state_store,
        )

    manifest = RunManifest.from_dict(
        json.loads(store[("bucket", "runs/fail-demo/npa-workflow/manifest.json")])
    )
    assert manifest.status == "failed"
    assert manifest.steps
    assert manifest.steps[0]["status"] == "failed"


def test_persist_failure_does_not_mask_workflow_error(monkeypatch) -> None:
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    calls = {"n": 0}

    def flaky_writer(bucket: str, key: str, body: bytes) -> None:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("s3 down")

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/persist-mask",
        writer=flaky_writer,
        reader=lambda _b, _k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: (_ for _ in ()).throw(
            NpaWorkflowError("step failed first")
        ),
    )

    with pytest.raises(NpaWorkflowError, match="step failed first"):
        run_workflow(
            spec,
            run_id="persist-mask",
            execute=True,
            state_store=state_store,
        )


def test_persist_failure_surfaces_when_no_step_error(monkeypatch) -> None:
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    calls = {"n": 0}

    def flaky_writer(bucket: str, key: str, body: bytes) -> None:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("s3 down")

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/persist-only",
        writer=flaky_writer,
        reader=lambda _b, _k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: {"state": step.state, "status": "ok"},
    )

    with pytest.raises(NpaWorkflowError, match="failed to persist workflow manifest"):
        run_workflow(
            spec,
            run_id="persist-only",
            execute=True,
            state_store=state_store,
        )


def test_cli_validate_spec_rejects_missing_config_token(tmp_path: Path) -> None:
    path = tmp_path / "bad-token.yaml"
    path.write_text(
        (SPECS / "vlm-eval-single.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "{{config.scores_uri}}report.json",
            "{{config.does_not_exist}}",
        ),
        encoding="utf-8",
    )
    result = RUNNER.invoke(
        app,
        ["workbench", "workflow", "validate-spec", str(path)],
    )
    assert result.exit_code == 1, result.output
    assert "does_not_exist" in result.output


def test_cli_plan_spec_rejects_unbounded_cycle(tmp_path: Path) -> None:
    path = tmp_path / "cycle.yaml"
    path.write_text(
        """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: cycle
config: {}
initial: a
states:
  a:
    run:
      shell: echo a
    next: b
  b:
    run:
      shell: echo b
    next: a
  done:
    terminal: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    validate = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path)])
    assert validate.exit_code == 1, validate.output
    assert "cycle" in validate.output.lower()

    plan = RUNNER.invoke(
        app,
        ["workbench", "workflow", "plan-spec", str(path), "--run-id", "cycle-plan"],
    )
    assert plan.exit_code == 1, plan.output
    assert "cycle" in plan.output.lower() or "step limit" in plan.output.lower()


def test_malformed_yaml_raises_npa_workflow_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nconfig: [\n",
        encoding="utf-8",
    )
    with pytest.raises(NpaWorkflowError, match="not valid YAML"):
        load_spec(path)


def test_cli_malformed_yaml_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("metadata:\n  name: x\n  bad indent\nfoo:\n", encoding="utf-8")
    result = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path)])
    assert result.exit_code == 1, result.output
    assert "not valid YAML" in result.output or "Error:" in result.output
    assert "Unexpected error" not in result.output


def test_persist_state_without_bucket_raises(tmp_path: Path) -> None:
    path = tmp_path / "no-bucket.yaml"
    path.write_text(
        """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: no-bucket
config:
  prefix: runs/test
initial: x
states:
  x:
    run:
      shell: echo ok
    terminal: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = load_spec(path)
    with pytest.raises(NpaWorkflowError, match="persist_state requires config.bucket"):
        run_workflow(spec, run_id="no-bucket-run", persist_state=True)


def test_cli_persist_state_without_bucket_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "no-bucket.yaml"
    path.write_text(
        """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: no-bucket
config:
  prefix: runs/test
initial: x
states:
  x:
    run:
      shell: echo ok
    terminal: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(path),
            "--run-id",
            "cli-no-bucket",
            "--plan-only",
            "--persist-state",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "persist_state requires config.bucket" in result.output
