from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
from argparse import Namespace

import pytest
import yaml
from typer.testing import CliRunner

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.detect import detect_submit_format
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_setup_for_tool,
    render_skypilot_yaml,
)
from npa.workflows.sim2real.workflow_io import (
    declared_loop_uri,
    write_loop_output,
)
from npa.workflows.sim2real.capture import DEFAULT_PPO_ITERATIONS
from npa.workflows.sim2real.workflow_stage import _authoritative_scene_args
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.workflows.sim2real.workflow_stage import (
    _stage11,
    _stage14,
    _stage14_download_plan,
    _stage8,
    _stage9,
    _stage9_existing_replay,
    _validate_stage7_cosmos3_coverage,
)


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "sim2real.yaml"


def test_canonical_is_one_standard_compositional_workflow() -> None:
    payload = yaml.safe_load(SPEC.read_text())
    assert payload["apiVersion"] == "npa.workflow/v0.0.1"
    assert payload["kind"] == "Workflow"
    assert detect_submit_format(SPEC) == "npa.workflow"
    assert not (ROOT / "npa" / "workflows" / "sim2real.yaml").exists()
    assert not (
        ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "sim2real-vlm-rl.yaml"
    ).exists()

    leaf_states = [state for state in payload["states"].values() if state.get("run")]
    assert len(leaf_states) >= 14
    rendered_commands = "\n".join(
        " ".join(state["run"].get("argv") or []) for state in leaf_states
    )
    for forbidden in (
        "run_preamble",
        "run_inner_loop",
        "run_single_outer_iteration",
        "run_finalize",
        "k8s_submit",
    ):
        assert forbidden not in rendered_commands

    for state in (f"stage-04-shard-{index}" for index in range(8)):
        assert any(
            "/components/lanes/stage_04/" in output["uri"]
            for output in payload["states"][state]["outputs"]
        )
    assert "stage-08-wave" not in payload["states"]
    assert "stage-08-reason2" not in payload["states"]
    assert "reason2_model" not in payload["config"]
    assert "reason_image" not in payload["config"]
    assert "reason-gpu" not in payload["resources"]

    viewer = payload["resources"]["viewer-cpu"]["kubernetes"]["pod_config"]["spec"][
        "containers"
    ][0]["resources"]
    assert viewer["requests"]["ephemeral-storage"] == "24Gi"
    assert viewer["limits"]["ephemeral-storage"] == "48Gi"
    cosmos3 = payload["states"]["stage-08-cosmos3"]
    assert cosmos3["resources"] == "stage8-cpu"
    assert "accelerators" not in payload["resources"]["stage8-cpu"]
    assert payload["config"]["cosmos3_model"] == "nvidia/Cosmos3-Super-Reasoner"
    assert "--reason-lane" not in cosmos3["run"]["argv"]
    assert "--reason-backend" not in cosmos3["run"]["argv"]
    assert int(payload["config"]["ppo_iterations"]) == DEFAULT_PPO_ITERATIONS
    assert DEFAULT_PPO_ITERATIONS >= 2_000


def test_retired_monolithic_toolrefs_are_not_catalog_surfaces() -> None:
    for tool_ref in (
        "workbench.sim2real.run",
        "workbench.sim2real.policy_rollouts",
        "workbench.sim2real.heldout_eval",
        "workbench.sim2real.finalize",
    ):
        assert tool_ref not in TOOL_CATALOG


def test_stage8_scores_every_rollout_once_with_hosted_cosmos3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.cosmos import reason
    from npa.workflows.sim2real import stage8_cosmos3

    work = tmp_path / "stage8"
    work.mkdir()

    class Store:
        def download_directory(self, _source, destination):
            destination = Path(destination)
            for index in range(2):
                rollout = destination / f"rollout-{index:04d}"
                rollout.mkdir(parents=True)
                (rollout / "camera-000.png").write_bytes(b"synthetic-public-frame")
                (rollout / "manifest.json").write_text(
                    json.dumps(
                        {
                            "rollout_id": f"rollout-{index:04d}",
                            "task_description": "strict cube grasp",
                            "camera_observations": ["camera-000.png"],
                            "actions": [{"step": 0, "action": [0.0]}],
                        }
                    )
                )

    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs["rollout_id"])
        return {
            "schema": "npa.sim2real.vlm_eval.v3",
            "rollout_id": kwargs["rollout_id"],
            "model": kwargs["model_id"],
            "provider": "nebius",
            "backend": "token_factory",
            "action_count": 1,
            "per_step": [{"step": 0}],
            "request": {
                "request_id": f"request-{len(calls)}",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "latency_seconds": 0.25,
                "retries": 0,
                "cost_usd": None,
            },
        }

    writes = []
    records = []
    monkeypatch.setattr(
        stage8_cosmos3.tempfile, "mkdtemp", lambda **_kwargs: str(work)
    )
    monkeypatch.setattr(stage8_cosmos3, "storage", lambda: Store())
    monkeypatch.setattr(reason, "run_token_factory_rollout_vlm", evaluate)
    monkeypatch.setattr(
        stage8_cosmos3,
        "image_provenance",
        lambda **kwargs: {"gpu_required": kwargs["require_gpu"]},
    )
    monkeypatch.setattr(
        stage8_cosmos3,
        "write_loop_output",
        lambda uri, payload, *_args: writes.append((uri, payload)),
    )
    monkeypatch.setattr(
        stage8_cosmos3,
        "publish_component_record",
        lambda **kwargs: records.append(kwargs),
    )

    _stage8(
        Namespace(
            root_uri="s3://unit/run",
            outer_iteration=1,
            inner_iteration=1,
            reason_model="nvidia/Cosmos3-Super-Reasoner",
            threshold=0.5,
        )
    )

    assert calls == ["rollout-0000", "rollout-0001"]
    payload = writes[0][1]
    assert payload["schema"] == "npa.sim2real.cosmos3_evaluator.v1"
    assert payload["source_rollout_ids"] == calls
    assert payload["evaluator_usage"]["request_count"] == 2
    assert payload["evaluator_usage"]["total_tokens"] == 30
    assert payload["evaluator_usage"]["cost_usd"] is None
    assert payload["provenance"]["gpu_required"] is False
    assert records[0]["require_gpu"] is False


def test_reduced_plan_preserves_all_real_solution_boundaries() -> None:
    spec = load_spec(SPEC)
    spec.config.update(
        {
            "outer_iterations": "1",
            "inner_iterations": "1",
            "env_count": "12",
            "rollout_count": "1",
            "ppo_iterations": "5",
            "validation_count": "3",
            "gold_count": "3",
        }
    )
    plan = build_plan(spec, run_id="compose-1x1", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    expected = {
        "stage-01-trigger",
        "stage-02-assets",
        "stage-03-transfer",
        *(f"stage-04-shard-{index}" for index in range(8)),
        "stage-05-split",
        "stage-06-tokens",
        "stage-07-rollouts",
        "stage-08-cosmos3",
        "stage-09-ppo",
        "stage-10-gold",
        "stage-11-decision",
        "stage-12-external-seam",
        "stage-13-retrigger",
        "stage-14-visualize",
    }
    assert set(states) == expected
    assert states.count("stage-07-rollouts") == 1
    assert states.count("stage-09-ppo") == 1


@pytest.mark.parametrize(
    ("stage7_ids", "source_ids", "evaluation_ids"),
    [
        (["rollout-1", "rollout-2"], ["rollout-1"], ["rollout-1"]),
        (["rollout-1"], ["rollout-1", "rollout-1"], ["rollout-1", "rollout-1"]),
        (["rollout-1"], ["rollout-1", "rollout-extra"], ["rollout-1", "rollout-extra"]),
    ],
    ids=["missing", "duplicate", "extra"],
)
def test_stage9_rejects_inexact_single_evaluator_coverage(
    stage7_ids: list[str], source_ids: list[str], evaluation_ids: list[str]
) -> None:
    stage7 = {
        "schema": "npa.sim2real.policy_rollouts.v1",
        "rollout_dirs": [f"/tmp/actions/{item}" for item in stage7_ids],
    }
    cosmos3 = {
        "source_rollout_ids": source_ids,
        "evaluations": [{"rollout_id": item} for item in evaluation_ids],
    }

    with pytest.raises(RuntimeError, match="exactly cover Stage 7"):
        _validate_stage7_cosmos3_coverage(stage7, cosmos3)


def test_stage_adapters_do_not_submit_hidden_kubernetes_jobs() -> None:
    source = (
        ROOT / "npa" / "src" / "npa" / "workflows" / "sim2real" / "workflow_stage.py"
    ).read_text()
    source += (
        ROOT
        / "npa"
        / "src"
        / "npa"
        / "workflows"
        / "sim2real"
        / "isaac_stage_contract.py"
    ).read_text()
    assert "KubernetesJobClient" not in source
    assert "run_gpu_job_with_fallback" not in source
    assert "submit_sim2real" not in source
    assert "sim2real.engine" not in source
    assert "NPA_SIM2REAL_INLINE_TASK" in source


def test_environment_generation_and_split_share_stage_two_contract() -> None:
    root = "s3://bucket/runs/exact-run"
    expected = [
        "--scene-spec-uri",
        root + "/stage_02_assets/consumed_scene_spec.json",
    ]
    source = (
        ROOT / "npa" / "src" / "npa" / "workflows" / "sim2real" / "workflow_stage.py"
    ).read_text()

    assert _authoritative_scene_args(root) == expected
    assert source.count("*_authoritative_scene_args(root)") == 2


def test_loop_outputs_preserve_canonical_lineage_and_runtime_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    canonical = "s3://bucket/run/actions/train/outer-01/iter-01/rollouts-result.json"
    declared = "s3://bucket/run/actions/train/outer-1/iter-1/rollouts-result.json"
    writes: list[tuple[str, dict[str, str], Path]] = []

    def record(uri: str, payload: dict[str, str], *, directory: Path) -> str:
        writes.append((uri, payload, directory))
        return uri

    monkeypatch.setattr(
        "npa.workflows.sim2real.workflow_io.write_json",
        record,
    )
    payload = {"schema": "npa.sim2real.policy_rollouts.v1"}

    result = write_loop_output(
        canonical,
        payload,
        tmp_path,
        1,
        1,
    )

    assert result == canonical
    assert [item[0] for item in writes] == [canonical, declared]
    assert all(item[1] == payload for item in writes)
    assert (
        declared_loop_uri(
            "s3://bucket/run/inner_loop/outer-01/evidence.json",
            1,
        )
        == "s3://bucket/run/inner_loop/outer-1/evidence.json"
    )
    with pytest.raises(ValueError, match="expected segment"):
        declared_loop_uri(
            "s3://bucket/run/inner_loop/evidence.json",
            1,
        )


@pytest.mark.parametrize(
    ("state", "canonical", "outer_iteration", "inner_iteration"),
    [
        (
            "stage-07-rollouts",
            "s3://unit/run/actions/train/outer-01/iter-01/rollouts-result.json",
            1,
            1,
        ),
        (
            "stage-08-cosmos3",
            "s3://unit/run/vlm_eval/train/outer-01/iter-01/cosmos3.json",
            1,
            1,
        ),
        (
            "stage-09-ppo",
            "s3://unit/run/inner_loop/outer-01/evidence.json",
            1,
            None,
        ),
        (
            "stage-10-gold",
            "s3://unit/run/eval/gold-heldout/outer-01/report.json",
            1,
            None,
        ),
    ],
)
def test_each_runtime_loop_output_is_published_by_write_loop_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: str,
    canonical: str,
    outer_iteration: int,
    inner_iteration: int | None,
) -> None:
    payload = yaml.safe_load(SPEC.read_text())
    declared = (
        payload["states"][state]["outputs"][0]["uri"]
        .replace("{{config.root_uri}}", "s3://unit/run")
        .replace("{{loop.outer-loop}}", "1")
        .replace("{{loop.inner-loop}}", "1")
    )
    writes: list[str] = []

    def record(uri: str, _payload: dict, *, directory: Path) -> str:
        assert directory.is_relative_to(tmp_path)
        writes.append(uri)
        return uri

    monkeypatch.setattr("npa.workflows.sim2real.workflow_io.write_json", record)
    write_loop_output(
        canonical,
        {"state": state},
        tmp_path,
        outer_iteration,
        inner_iteration,
    )

    assert writes == [canonical, declared]


@pytest.mark.parametrize("shard_count", [1, 3, 17])
def test_shard_count_override_fails_before_plan_or_submit(shard_count: int) -> None:
    spec = load_spec(SPEC)
    with pytest.raises(
        NpaWorkflowError,
        match=rf"parallelCount resolves to {shard_count}.*8 members",
    ):
        merge_config_overrides(spec, {"shard_count": str(shard_count)})


def test_shard_count_mismatch_fails_static_spec_validation(tmp_path: Path) -> None:
    payload = yaml.safe_load(SPEC.read_text())
    payload["config"]["shard_count"] = "3"
    invalid = tmp_path / "sim2real-invalid-shards.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(
        NpaWorkflowError, match="parallelCount resolves to 3.*8 members"
    ):
        load_spec(invalid)


def test_default_shard_count_matches_declared_parallel_lanes() -> None:
    spec = merge_config_overrides(load_spec(SPEC), {"shard_count": "8"})
    assert spec.states["stage-04-wave"].parallel_count == "{{config.shard_count}}"
    assert len(spec.states["stage-04-wave"].parallel) == 8


@pytest.mark.parametrize(
    ("enabled", "success_rate", "expected"),
    [(True, 0.8, "promote_checkpoint"), (False, 0.8, "loop_back_to_inner_loop")],
)
def test_stage11_honors_configurable_early_exit(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    success_rate: float,
    expected: str,
) -> None:
    decisions: list[dict] = []
    monkeypatch.setattr(
        "npa.workflows.sim2real.workflow_stage.read_json",
        lambda *_args, **_kwargs: {"success_rate": success_rate},
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.workflow_stage.write_json",
        lambda _uri, payload, **_kwargs: decisions.append(payload) or _uri,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.workflow_stage.publish_component_record",
        lambda **_kwargs: {},
    )
    args = Namespace(
        root_uri="s3://unit/run",
        outer_iteration=1,
        threshold=0.5,
        allow_early_exit=enabled,
        run_id="early-exit",
    )

    _stage11(args)

    assert decisions[0]["decision"] == expected
    assert decisions[0]["early_exit_enabled"] is enabled


def _stage9_replay_fixture() -> tuple[dict, dict, dict, dict]:
    validation = {"success_rate": 0.75, "per_env": [{"env_id": "env-1"}]}
    candidate = {
        "evaluation_split": "validation",
        "outer_iteration": 1,
        "inner_iteration": 1,
        "training_iteration": 2,
        "checkpoint_uri": "s3://unit/run/checkpoint.pt",
        "checkpoint_sha256": "a" * 64,
        "validation_report_uri": "s3://unit/run/validation.json",
        "validation_report": validation,
    }
    sample_eval = {
        "schema": "npa.sim2real.vlm_eval.v3",
        "rollout_id": "rollout-1",
        "score": 0.8,
        "threshold": 0.5,
        "model": "nvidia/Cosmos3-Super-Reasoner",
        "provider": "nebius",
        "backend": "token_factory",
        "request": {
            "request_id": "request-1",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "latency_seconds": 0.25,
            "retries": 0,
            "cost_usd": None,
        },
        "action_count": 1,
        "per_step": [{"step": 0}],
    }
    sample_signal = {"rollout_id": "rollout-1", "weight": 1.0}
    iteration = {
        "iteration": 1,
        "actions_uri": "s3://unit/run/actions/",
        "vlm_eval_uri": "s3://unit/run/evaluations/",
        "signal_uri": "s3://unit/run/signals/",
        "trainer_component_invocation": {"mode": "npa_workflow_skypilot_task"},
        "update": {"checkpoint_path": candidate["checkpoint_uri"]},
        "sample_vlm_eval": sample_eval,
        "sample_signal": sample_signal,
    }
    from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint

    selection = select_best_checkpoint([candidate])
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": 1,
        "iterations": [iteration],
        "checkpoint_candidates": [candidate],
        "selected_checkpoint_uri": candidate["checkpoint_uri"],
        "final_checkpoint_uri": candidate["checkpoint_uri"],
        "checkpoint_selection": selection,
        "selected_validation_report": validation,
    }
    return evidence, candidate, sample_eval, sample_signal


def test_stage9_exact_same_iteration_replay_is_idempotent() -> None:
    evidence, candidate, sample_eval, sample_signal = _stage9_replay_fixture()
    result = _stage9_existing_replay(
        prior=evidence,
        outer_iteration=1,
        inner_iteration=1,
        actions_uri="s3://unit/run/actions/",
        evaluation_uri="s3://unit/run/evaluations/",
        signal_uri="s3://unit/run/signals/",
        sample_vlm_eval=sample_eval,
        sample_signal=sample_signal,
    )
    assert result is not None
    assert result[0] == candidate
    assert len(evidence["iterations"]) == len(evidence["checkpoint_candidates"]) == 1


def test_stage9_conflicting_same_iteration_replay_fails_closed() -> None:
    evidence, _candidate, sample_eval, sample_signal = _stage9_replay_fixture()
    with pytest.raises(RuntimeError, match="conflicts with durable evidence"):
        _stage9_existing_replay(
            prior=evidence,
            outer_iteration=1,
            inner_iteration=1,
            actions_uri="s3://unit/run/different-actions/",
            evaluation_uri="s3://unit/run/evaluations/",
            signal_uri="s3://unit/run/signals/",
            sample_vlm_eval=sample_eval,
            sample_signal=sample_signal,
        )


def test_stage9_retry_republishes_exact_evidence_without_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workflows.sim2real import byo_isaac_trainer, temporal_credit
    from npa.workflows.sim2real import workflow_stage

    root = "s3://unit/run"
    evidence, _candidate, sample_eval, sample_signal = _stage9_replay_fixture()
    iteration = evidence["iterations"][0]
    iteration.update(
        {
            "actions_uri": f"{root}/actions/train/outer-01/iter-01/",
            "vlm_eval_uri": f"{root}/vlm_eval/train/outer-01/iter-01/evaluations/",
            "signal_uri": f"{root}/vlm_eval/train/outer-01/iter-01/signals/",
        }
    )
    lane_base = f"{root}/vlm_eval/train/outer-01/iter-01/"
    lanes = {
        f"{root}/actions/train/outer-01/iter-01/rollouts-result.json": {
            "schema": "npa.sim2real.policy_rollouts.v1",
            "rollout_dirs": ["/tmp/actions/rollout-1"],
        },
        f"{root}/components/stage_08.json": {
            "stage": 8,
            "name": "stage_08_vlm_eval_train",
            "artifacts": {
                "result": lane_base + "cosmos3.json",
                "backend": "token_factory",
                "outer_iteration": 1,
                "inner_iteration": 1,
            },
        },
        lane_base + "cosmos3.json": {
            "schema": "npa.sim2real.cosmos3_evaluator.v1",
            "evaluator": "cosmos3",
            "model": "nvidia/Cosmos3-Super-Reasoner",
            "provider": "nebius",
            "backend": "token_factory",
            "provenance": {"image": "cosmos3"},
            "evaluator_usage": {
                "request_count": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "per_request_latency_seconds": [0.25],
            },
            "source_rollout_ids": ["rollout-1"],
            "evaluations": [sample_eval],
        },
    }

    work = tmp_path / "stage9"
    work.mkdir()
    monkeypatch.setattr(workflow_stage, "_work", lambda _stage: work)
    monkeypatch.setattr(workflow_stage, "list_prefix", lambda _uri: [{"Size": 1}])
    monkeypatch.setattr(
        workflow_stage,
        "read_json",
        lambda uri, **_kwargs: (
            evidence if uri.endswith("/evidence.json") else lanes[uri]
        ),
    )
    monkeypatch.setattr(
        temporal_credit, "convert_evaluation", lambda _item: sample_signal
    )
    monkeypatch.setattr(
        byo_isaac_trainer,
        "main",
        lambda: pytest.fail("an exact replay must not run PPO again"),
    )
    writes: list[tuple[str, dict, int]] = []
    records: list[dict] = []
    monkeypatch.setattr(
        workflow_stage,
        "write_loop_output",
        lambda uri, payload, _directory, outer: writes.append((uri, payload, outer)),
    )
    monkeypatch.setattr(
        workflow_stage,
        "publish_component_record",
        lambda **kwargs: records.append(kwargs),
    )

    _stage9(
        Namespace(
            root_uri=root,
            outer_iteration=1,
            inner_iteration=1,
            threshold=0.5,
            ppo_iterations=2,
        )
    )

    assert writes == [(f"{root}/inner_loop/outer-01/evidence.json", evidence, 1)]
    assert len(evidence["iterations"]) == len(evidence["checkpoint_candidates"]) == 1
    assert records[0]["artifacts"]["idempotent_replay"] is True


def test_stage14_selects_only_consumed_artifacts_and_cleans_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = "s3://unit/runs/finalize"
    evidence = {
        "iterations": [
            {
                "iteration": 1,
                "actions_uri": f"{root}/actions/train/outer-01/iter-01/",
                "vlm_eval_uri": f"{root}/vlm_eval/train/outer-01/iter-01/evaluations/",
                "signal_uri": f"{root}/vlm_eval/train/outer-01/iter-01/signals/",
            }
        ]
    }
    gold = {
        "render_lineage": {
            "canonical_s3_uri": f"{root}/eval/gold-heldout/outer-01/renders/",
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        }
    }
    plan = _stage14_download_plan(
        root=root, outer_iteration=1, evidence=evidence, gold=gold
    )
    assert plan
    assert all(source != root + "/" for source, _destination, _prefix in plan)
    assert {destination for _source, destination, _prefix in plan} >= {
        "augment/frames",
        "actions/train/outer-01/iter-01",
        "eval/gold-heldout/outer-01/renders",
    }

    workspaces: list[Path] = []

    def capture(_args: Namespace, *, root: str, work: Path) -> None:
        assert root == "s3://unit/run"
        workspaces.append(work)
        (work / "proof").write_text("bounded")

    monkeypatch.setattr(
        "npa.workflows.sim2real.workflow_stage._stage14_in_work", capture
    )
    _stage14(Namespace(root_uri="s3://unit/run"))
    assert workspaces and not workspaces[0].exists()


def test_retired_materialize_command_gives_actionable_migration() -> None:
    from npa.cli.workbench.sim2real import app

    result = CliRunner().invoke(app, ["materialize"])
    assert result.exit_code == 2
    assert "workflow submit" in result.output
    assert "sim2real.yaml --runtime skypilot" in result.output

    representative_old_call = CliRunner().invoke(
        app,
        [
            "materialize",
            "legacy-runbook.yaml",
            "--run-id",
            "old-run",
            "--image",
            "registry.example/old:tag",
        ],
    )
    assert representative_old_call.exit_code == 2
    assert "workflow submit" in representative_old_call.output


def test_stage_adapter_import_does_not_load_legacy_controller() -> None:
    script = """
import sys
import npa.workflows.sim2real.workflow_stage
for name in (
    'npa.workflows.sim2real.engine',
    'npa.workflows.sim2real.legacy_orchestration',
    'npa.workflows.sim2real.runner',
    'npa.workflows.sim2real.scheduler',
    'npa.workflows.sim2real.stage_execution',
):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_legacy_facade_has_a_finite_compatibility_contract() -> None:
    from npa.workflows.sim2real import engine

    assert engine.LEGACY_COMPATIBILITY_UNTIL == "2027-02-01"
    assert engine.LEGACY_REMOVAL_VERSION == "0.5.0"
    assert engine.LEGACY_SCOPE == (
        "pre-standard-runtime callers and archived artifact replay"
    )
    canonical = SPEC.read_text()
    for legacy_surface in ("stage_execution", "sim2real.scheduler", "run_staged"):
        assert legacy_surface not in canonical


def test_baked_setup_executes_and_records_the_declared_interpreter(
    tmp_path: Path,
) -> None:
    record = tmp_path / "npa-python"
    setup = render_setup_for_tool(
        "run.shell",
        config={"require_baked_npa": "1"},
        options=SkypilotRenderOptions(),
    ).replace("/tmp/npa-python", str(record))
    source_sha = "a" * 40
    environment = {
        "PATH": "/usr/bin:/bin",
        "NPA_BAKED_PYTHON": sys.executable,
        "NPA_IMAGE_SOURCE_SHA": source_sha,
        "NPA_SIM2REAL_SOURCE_SHA": source_sha,
    }

    subprocess.run(["bash", "-c", setup], check=True, env=environment)

    assert record.read_text(encoding="utf-8").strip() == sys.executable
    environment["NPA_BAKED_PYTHON"] = "relative/python"
    failed = subprocess.run(
        ["bash", "-c", setup], check=False, capture_output=True, env=environment
    )
    assert failed.returncode == 68
    assert b"must be an absolute path" in failed.stderr


def test_baked_setup_rejects_an_unsafe_import_module() -> None:
    with pytest.raises(
        NpaWorkflowError,
        match="config.baked_npa_import must be a dotted Python module name",
    ):
        render_setup_for_tool(
            "run.shell",
            config={
                "require_baked_npa": "1",
                "baked_npa_import": "npa.cli.main; raise SystemExit(0)",
            },
            options=SkypilotRenderOptions(),
        )


def test_baked_raw_module_setup_probes_the_executed_module() -> None:
    setup = render_setup_for_tool(
        "",
        config={"require_baked_npa": "1"},
        options=SkypilotRenderOptions(),
        command=["python3", "-m", "npa.workflows.sim2real.workflow_stage"],
    )

    assert (
        "importlib.import_module('npa.workflows.sim2real.workflow_stage')" in setup
    )
    assert "npa.cli.main" not in setup


def test_exact_source_and_per_state_immutable_images_reach_rendered_tasks() -> None:
    spec = load_spec(SPEC)
    source_sha = "a" * 40
    image = "cr.example/npa/runtime@sha256:" + "b" * 64
    spec.config.update(
        {
            "source_sha": source_sha,
            "outer_iterations": "1",
            "inner_iterations": "1",
            "controller_image": image,
            "transfer_image": image,
            "envgen_image": image,
            "isaac_image": image,
            "viewer_image": image,
            "isaac_cache_pvc": "isaac-cache",
        }
    )
    plan = build_plan(spec, run_id="render-1x1", assume_decision="loop_back")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="render-1x1",
        options=SkypilotRenderOptions(
            materialize_registry_secrets=False,
            accept_eula=True,
        ),
    )
    documents = [item for item in yaml.safe_load_all(rendered) if item]
    tasks = [item for item in documents if item.get("envs")]
    assert tasks
    for task in tasks:
        assert task["envs"]["NPA_SIM2REAL_SOURCE_SHA"] == source_sha
        assert task["envs"]["NPA_TASK_IMAGE"] == image
        assert "immutable baked NPA runtime verified" in task["setup"]
        assert (
            "importlib.import_module('npa.workflows.sim2real.workflow_stage')"
            in task["setup"]
        )
        assert "NPA_BAKED_PYTHON" in task["setup"]
        assert "/tmp/npa-python" in task["setup"]
        assert "/opt/npa/src" in task["setup"]
        assert "/tmp/npa-baked-pythonpath" in task["setup"]
        assert "/tmp/npa-baked-pythonpath" in task["run"]
        assert "baked NPA interpreter must be an absolute path" in task["setup"]
        assert "baked NPA interpreter is not executable" in task["setup"]
        assert "pip install" not in task["setup"]
        assert "NPA_SRC_S3_URI" not in task["envs"]
        pod_containers = task["config"]["kubernetes"]["pod_config"]["spec"][
            "containers"
        ]
        ray_node = next(
            container for container in pod_containers if container["name"] == "ray-node"
        )
        bootstrap_env = {
            item["name"]: item["value"] for item in ray_node.get("env", [])
        }
        assert bootstrap_env["XDG_CACHE_HOME"] == "/tmp/npa-skypilot-xdg-cache"
        assert bootstrap_env["UV_CACHE_DIR"] == "/tmp/npa-skypilot-uv-cache"

    gpu_tasks = [task for task in tasks if task["resources"].get("accelerators")]
    assert gpu_tasks
    for task in gpu_tasks:
        pod_config = task["config"]["kubernetes"]["pod_config"]
        assert pod_config["metadata"]["labels"] == {
            "kueue.x-k8s.io/queue-name": "sim2real-gpu"
        }
        assert pod_config["spec"]["priorityClassName"] == "sim2real-production"

    transfer_pod = spec.resources["transfer-gpu"]["kubernetes"]["pod_config"][
        "spec"
    ]
    transfer_env = {
        item["name"]: item["value"]
        for item in transfer_pod["containers"][0]["env"]
    }
    assert transfer_env == {
        "UV_CACHE_DIR": "/tmp/npa-skypilot-uv-cache",
        "XDG_CACHE_HOME": "/tmp/npa-skypilot-xdg-cache",
    }
    rendered_transfer_tasks = [
        task
        for task in tasks
        if task["envs"]["NPA_WORKFLOW_STATE"] == "stage-03-transfer"
    ]
    assert len(rendered_transfer_tasks) == 1
    assert (
        rendered_transfer_tasks[0]["config"]["kubernetes"]["pod_config"]["spec"][
            "containers"
        ][0]["env"]
        == transfer_pod["containers"][0]["env"]
    )

    isaac_pod = spec.resources["isaac-gpu"]["kubernetes"]["pod_config"]["spec"]
    assert isaac_pod.get("securityContext", {}).get("runAsUser") != 0
    assert isaac_pod["containers"][0].get("securityContext", {}).get("runAsUser") != 0
    isaac_env = isaac_pod["containers"][0]["env"]
    isaac_env_by_name = {item["name"]: item["value"] for item in isaac_env}
    assert isaac_env_by_name["NPA_BAKED_PYTHON"] == "/opt/npa/sim/venv/bin/python"
    rendered_isaac_tasks = [task for task in tasks if "ACCEPT_EULA" in task["envs"]]
    for task in rendered_isaac_tasks:
        assert task["envs"]["ACCEPT_EULA"] == "Y"
    assert len(rendered_isaac_tasks) == 3


def test_canonical_isaac_eula_acceptance_is_operator_supplied_and_fail_closed() -> None:
    payload = yaml.safe_load(SPEC.read_text())
    assert "omni_kit_accept_eula" not in payload["config"]
    assert "isaacsim_accept_eula" not in payload["config"]
    env = payload["resources"]["isaac-gpu"]["kubernetes"]["pod_config"]["spec"][
        "containers"
    ][0]["env"]
    by_name = {item["name"]: item["value"] for item in env}
    assert "ACCEPT_EULA" not in by_name
