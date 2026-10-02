from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from npa.workflows.sim2real import (
    component_authority,
    stage10_authority,
    workflow_io,
    workflow_stage,
)
from npa.workflows.sim2real.byo_isaac_trainer import artifact_tag, k8s_job_name
from npa.workflows.sim2real.decision_authority import (
    gold_report_sha256,
    validate_stage11_decision,
)
from npa.workflows.sim2real.models import Sim2RealLoopConfig
import npa.workflows.sim2real.stage14_finalize as stage14
import npa.workflows.sim2real_rerun_regen as regen


RUN_ID = "run-a"
ROOT = f"s3://demo-bucket/sim2real/{RUN_ID}"
SOURCE_SHA = "b" * 40
DIGEST = "a" * 64
IMAGE_DIGEST = "c" * 64
ATTEMPT = "gold_heldout-outer-01-attempt-" + "d" * 32
TRAIN_JOB = k8s_job_name("s2r-byo-isaac-train", RUN_ID)
CHECKPOINT = (
    f"{ROOT}/byo-trainer/{TRAIN_JOB}/{artifact_tag('outer-01-iter-01')}/model_latest.pt"
)


def _identity() -> dict[str, Any]:
    return {
        "checkpoint_uri": CHECKPOINT,
        "checkpoint_sha256": DIGEST,
        "checkpoint_size_bytes": 128,
        "generator_policy_sha256": DIGEST,
    }


def _candidate() -> dict[str, Any]:
    return {
        **_identity(),
        "evaluation_split": "validation",
        "outer_iteration": 1,
        "inner_iteration": 1,
        "training_iteration": 10,
        "validation_report_uri": f"{ROOT}/eval/validation/report.json",
    }


def _evidence() -> dict[str, Any]:
    candidate = _candidate()
    return {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "run_id": RUN_ID,
        "outer_iteration": 1,
        "iterations": [
            {
                "iteration": 1,
                "actions_uri": f"{ROOT}/actions/train/outer-01/iter-01/",
                "vlm_eval_uri": (
                    f"{ROOT}/vlm_eval/train/outer-01/iter-01/evaluations/"
                ),
                "signal_uri": (f"{ROOT}/vlm_eval/train/outer-01/iter-01/signals/"),
            }
        ],
        "selected_checkpoint_uri": CHECKPOINT,
        "final_checkpoint_uri": CHECKPOINT,
        "checkpoint_candidates": [dict(candidate)],
        "checkpoint_selection": dict(candidate),
    }


def _producer_render_uri(attempt: str = ATTEMPT) -> str:
    job = k8s_job_name(
        "s2r-byo-isaac-eval",
        RUN_ID,
        artifact_tag(attempt),
    )
    return f"{ROOT}/byo-eval/{job}/renders/"


def _gold() -> dict[str, Any]:
    manifest = {
        "evaluation_attempt_tag": ATTEMPT,
        "renders_s3_uri": _producer_render_uri(),
        "policy_checkpoint": {
            "uri": CHECKPOINT,
            "sha256": DIGEST,
            "size_bytes": 128,
        },
        "episodes": [{"env_id": "env-1", "frames": ["camera-000.png"]}],
    }
    return {
        "schema": "npa.sim2real.heldout_eval.v1",
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "evaluation_attempt_tag": ATTEMPT,
        "success_rate": 0.2,
        "deployable_policy_eval": True,
        "policy_checkpoint_uri": CHECKPOINT,
        "policy_checkpoint_sha256": DIGEST,
        "policy_checkpoint_size_bytes": 128,
        "per_env": [{"env_id": "env-1", "success": False}],
        "policy_inference_provenance": {
            **_identity(),
            "loaded_for_inference": True,
            "stock_or_scripted_policy": False,
            "actor_is_learned": True,
            "scripted_post_actor_controller": False,
            "policy_composition": "learned_actor_only",
            "post_actor_controller": None,
        },
        "component_invocation": {
            "output_uri": f"{ROOT}/mutable-other-producer/result.json"
        },
        "render_manifest": manifest,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "evaluation_attempt_tag": ATTEMPT,
            "source_s3_uri": _producer_render_uri(),
            "canonical_s3_uri": (
                f"{ROOT}/eval/gold-heldout/outer-01/attempts/{ATTEMPT}/renders/"
            ),
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }


def _provenance(*, parallel: bool = False) -> dict[str, Any]:
    base: dict[str, Any] = {
        "image": f"ghcr.io/example/npa@sha256:{IMAGE_DIGEST}",
        "image_digest": f"sha256:{IMAGE_DIGEST}",
        "source_sha": SOURCE_SHA,
        "gpu_products": ["NVIDIA H100"],
    }
    if parallel:
        return {
            **base,
            "execution_mode": "standard_npa_workflow_parallel_join",
            "workflow_jobs": ["lane-0", "lane-1"],
            "lane_count": 2,
        }
    return {
        **base,
        "execution_mode": "standard_npa_workflow_skypilot",
        "workflow_job": "job",
    }


def _rehash(record: dict[str, Any]) -> dict[str, Any]:
    material = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def _component(stage: int) -> dict[str, Any]:
    name, tier, required = component_authority.COMPONENT_CONTRACTS[stage]
    authority = component_authority._authority_uris(ROOT, _evidence(), _gold())
    artifacts: dict[str, Any] = {"source_sha": SOURCE_SHA}
    for key in required:
        artifacts[key] = authority[stage].get(key, f"stage-{stage}-{key}")
    if stage == 10:
        artifacts["render_lineage"] = _gold()["render_lineage"]
        artifacts["gold_report_sha256"] = gold_report_sha256(_gold())
    if stage == 4:
        artifacts.update(
            {
                "shard_count": 2,
                "shard_provenance": [{}, {}],
                "lane_records": [{}, {}],
            }
        )
    if stage != 12:
        artifacts.update(_provenance(parallel=stage == 4))
    record = {
        "schema": "npa.sim2real.component_record.v1",
        "stage": stage,
        "name": name,
        "tier": tier,
        "evidence": f"stage {stage} complete",
        "artifacts": artifacts,
        "next_action": "CONTINUE",
    }
    return _rehash(record)


def _stage14_component(
    *,
    rrd_uri: str,
    mcap_uri: str,
    report_uri: str,
) -> dict[str, Any]:
    record = {
        "schema": "npa.sim2real.component_record.v1",
        "stage": 14,
        "name": "stage_14_rerun_viz",
        "tier": "WORKS",
        "evidence": "stage 14 complete",
        "artifacts": {
            **_provenance(),
            "rrd": rrd_uri,
            "mcap": mcap_uri,
            "report": report_uri,
        },
        "next_action": "CONTINUE",
    }
    return _rehash(record)


def _decision(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.threshold_decision.v1",
        "run_id": RUN_ID,
        "outer_iteration": 1,
        "decision": "loop_back_to_inner_loop",
        "success_rate": report["success_rate"],
        "threshold": 0.5,
        "strict_success_distance_m": 0.05,
        "placement_stability_required": True,
        "early_exit_enabled": False,
        "checkpoint_uri": CHECKPOINT,
        "gold_report_uri": f"{ROOT}/eval/gold-heldout/outer-01/report.json",
        "gold_report_sha256": gold_report_sha256(report),
        "candidate": _candidate(),
    }


def _config() -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id=RUN_ID,
        s3_bucket="demo-bucket",
        s3_prefix="sim2real",
        threshold=0.5,
        early_exit=False,
    )


def test_parallel_stage4_join_provenance_is_a_valid_component_record() -> None:
    record = workflow_io.build_component_record(
        stage=4,
        name="stage_04_envs_raw",
        tier="WORKS",
        evidence="joined every declared shard",
        artifacts={"raw_envs": f"{ROOT}/envs/raw/", "shard_provenance": [{}]},
        require_gpu=True,
        execution_provenance=_provenance(parallel=True),
    )
    workflow_io.validate_component_record(
        record,
        expected_stage=4,
        expected_name="stage_04_envs_raw",
        expected_tier="WORKS",
        required_artifacts=("raw_envs", "shard_provenance"),
        expected_source_sha=SOURCE_SHA,
    )


@pytest.mark.parametrize(
    "field",
    ["shard_count", "lane_count", "shard_provenance", "lane_records"],
)
def test_parallel_stage4_join_requires_complete_lane_evidence(field: str) -> None:
    components = [_component(stage) for stage in range(1, 14)]
    artifacts = components[3]["artifacts"]
    if field in {"shard_count", "lane_count"}:
        artifacts[field] = 1
    else:
        artifacts[field] = [{}]
    _rehash(components[3])
    with pytest.raises(ValueError):
        component_authority.validate_component_records(
            components,
            root=ROOT,
            evidence=_evidence(),
            gold=_gold(),
            expected_source_sha=SOURCE_SHA,
        )


def test_fresh_stage9_evidence_carries_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import byo_isaac_trainer, temporal_credit

    captured_env: dict[str, Any] = {}
    written: list[dict[str, Any]] = []
    validation_inputs: list[dict[str, Any]] = []
    sample_eval = {"rollout_id": "rollout-1"}
    update = {
        "checkpoint_path": CHECKPOINT,
        "component_invocation": {"mode": "npa_workflow_skypilot_task"},
    }

    def trainer() -> int:
        Path(captured_env["NPA_SIM2REAL_OUTPUT_JSON"]).write_text(json.dumps(update))
        return 0

    def run_eval(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        validation_inputs.append(kwargs["evidence"])
        return {
            "report_uri": f"{ROOT}/eval/validation/report.json",
            "success_rate": 0.4,
            "per_env": [{"env_id": "env-1"}],
            "policy_checkpoint_sha256": DIGEST,
            "policy_checkpoint_size_bytes": 128,
            "policy_inference_provenance": {"generator_policy_sha256": DIGEST},
        }

    monkeypatch.setattr(workflow_stage, "_work", lambda _stage: tmp_path)
    monkeypatch.setattr(workflow_stage, "_common_isaac_env", lambda *_a, **_k: {})
    monkeypatch.setattr(
        workflow_stage, "_set_env", lambda values: captured_env.update(values)
    )
    monkeypatch.setattr(workflow_stage, "source_sha", lambda: SOURCE_SHA)
    monkeypatch.setattr(workflow_stage, "list_prefix", lambda _uri: [])
    monkeypatch.setattr(
        workflow_stage,
        "read_json",
        lambda uri, **_kwargs: (
            {"evaluator_usage": {}} if uri.endswith("cosmos3.json") else {}
        ),
    )
    monkeypatch.setattr(
        workflow_stage,
        "validate_hosted_evaluator",
        lambda **_kwargs: {"rollout-1": sample_eval},
    )
    monkeypatch.setattr(
        temporal_credit,
        "convert_evaluation",
        lambda _evaluation: {"rollout_id": "rollout-1", "weight": 1.0},
    )
    monkeypatch.setattr(byo_isaac_trainer, "main", trainer)
    monkeypatch.setattr(workflow_stage, "_run_eval", run_eval)
    monkeypatch.setattr(
        workflow_stage, "_assert_embodiment_evidence", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        workflow_stage,
        "storage",
        lambda: SimpleNamespace(upload_directory=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(workflow_stage, "write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(
        workflow_stage,
        "write_loop_output",
        lambda _uri, payload, *_a: written.append(payload),
    )
    monkeypatch.setattr(
        workflow_stage, "publish_component_record", lambda **_kwargs: {}
    )
    workflow_stage._stage9(
        Namespace(
            root_uri=ROOT,
            run_id=RUN_ID,
            reason_model="reasoner",
            outer_iteration=1,
            inner_iteration=1,
            threshold=0.5,
            ppo_num_envs=1,
            ppo_iterations=10,
            ppo_steps_per_env=1,
            validation_count=1,
        )
    )

    assert validation_inputs[0]["run_id"] == RUN_ID
    assert written[0]["run_id"] == RUN_ID


@pytest.mark.parametrize("field", ["schema", "run_id"])
def test_stage10_rejects_stripped_modern_authority(field: str) -> None:
    evidence = _evidence()
    evidence.pop(field)
    with pytest.raises(RuntimeError):
        stage10_authority.validate_stage10_input_scope(
            Namespace(run_id=RUN_ID, outer_iteration=1),
            root=ROOT,
            evidence=evidence,
        )


@pytest.mark.parametrize("defect", ["schema", "arbitrary-checkpoint"])
def test_heldout_only_requires_canonical_trainer_authority(defect: str) -> None:
    evidence = _evidence()
    if defect == "schema":
        evidence.pop("schema")
    else:
        replacement = f"{ROOT}/imports/arbitrary.pt"
        evidence["selected_checkpoint_uri"] = replacement
        evidence["final_checkpoint_uri"] = replacement
        evidence["checkpoint_selection"]["checkpoint_uri"] = replacement
        evidence["checkpoint_candidates"][0]["checkpoint_uri"] = replacement
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._resolve_heldout_eval_checkpoint(
            _config(),
            ROOT + "/",
            evidence,
            outer_iteration=1,
        )


def test_heldout_only_downloads_exact_manifest_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _gold()
    downloaded: list[str] = []

    def download(
        _storage: Any,
        source: str,
        destination: Path,
        **_kwargs: Any,
    ) -> bool:
        downloaded.append(source)
        frame = Path(destination) / "env-1/camera-000.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"png")
        return True

    monkeypatch.setattr(regen, "_download_render_tree", download)
    regen._sync_heldout_eval_renders(
        _config(), tmp_path, object(), report, _candidate()
    )
    assert downloaded == [_producer_render_uri()]


@pytest.mark.parametrize("defect", ["missing-uri", "wrong-frame", "checkpoint"])
def test_heldout_only_rejects_unbound_render_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    defect: str,
) -> None:
    report = _gold()
    if defect == "missing-uri":
        report["render_manifest"]["renders_s3_uri"] = ""
    elif defect == "checkpoint":
        report["render_manifest"]["policy_checkpoint"]["sha256"] = "f" * 64

    def download(
        _storage: Any,
        _source: str,
        destination: Path,
        **_kwargs: Any,
    ) -> bool:
        name = "camera-999.png" if defect == "wrong-frame" else "camera-000.png"
        frame = Path(destination) / "env-1" / name
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"png")
        return True

    monkeypatch.setattr(regen, "_download_render_tree", download)
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._sync_heldout_eval_renders(
            _config(), tmp_path, object(), report, _candidate()
        )


def test_stage11_decision_is_bound_to_config_and_exact_report_bytes() -> None:
    report = _gold()
    decision = _decision(report)
    validate_stage11_decision(
        decision,
        run_id=RUN_ID,
        root=ROOT,
        outer_iteration=1,
        gold_report=report,
        checkpoint_uri=CHECKPOINT,
        expected_threshold=0.5,
        expected_early_exit=False,
        gold_report_bytes_sha256=gold_report_sha256(report),
    )
    mutated = json.loads(json.dumps(report))
    mutated["per_env"][0]["success"] = True
    with pytest.raises(ValueError):
        validate_stage11_decision(
            decision,
            run_id=RUN_ID,
            root=ROOT,
            outer_iteration=1,
            gold_report=mutated,
            checkpoint_uri=CHECKPOINT,
            expected_threshold=0.5,
            expected_early_exit=False,
            gold_report_bytes_sha256=gold_report_sha256(mutated),
        )
    with pytest.raises(ValueError):
        validate_stage11_decision(
            decision,
            run_id=RUN_ID,
            root=ROOT,
            outer_iteration=1,
            gold_report=report,
            checkpoint_uri=CHECKPOINT,
            expected_threshold=0.9,
            expected_early_exit=True,
            gold_report_bytes_sha256=gold_report_sha256(report),
        )


def test_stage11_rejects_semantically_equal_but_replaced_report_bytes() -> None:
    report = _gold()
    decision = _decision(report)
    replacement = json.dumps(report, sort_keys=True).encode() + b"\n\n"
    with pytest.raises(ValueError):
        validate_stage11_decision(
            decision,
            run_id=RUN_ID,
            root=ROOT,
            outer_iteration=1,
            gold_report=report,
            checkpoint_uri=CHECKPOINT,
            expected_threshold=0.5,
            expected_early_exit=False,
            gold_report_bytes_sha256=hashlib.sha256(replacement).hexdigest(),
        )


def test_stage14_rejects_pointer_history_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    components = [_component(stage) for stage in range(1, 14)]

    def reader(uri: str, **_kwargs: Any) -> dict[str, Any]:
        stage = int(uri.rsplit("stage_", 1)[1][:2])
        record = json.loads(json.dumps(components[stage - 1]))
        if "/history/" in uri and stage == 10:
            record["artifacts"]["report"] = f"{ROOT}/unrelated/report.json"
            _rehash(record)
        return record

    state = stage14._Stage14State(
        args=Namespace(
            run_id=RUN_ID,
            outer_iteration=1,
            threshold=0.5,
            allow_early_exit=False,
        ),
        root=ROOT,
        work=tmp_path,
        local=tmp_path / "run",
        evidence=_evidence(),
        gold=_gold(),
        rrd_uri=f"{ROOT}/reports/generation/rrd",
        mcap_uri=f"{ROOT}/reports/generation/mcap",
        report_uri=f"{ROOT}/reports/generation/report",
    )
    monkeypatch.setattr(stage14, "source_sha", lambda: SOURCE_SHA)
    monkeypatch.setattr(stage14, "read_json", reader)
    with pytest.raises(RuntimeError, match="invalid ComponentRecord"):
        stage14._load_component_records(state)


@pytest.mark.parametrize(
    "defect",
    [
        "foreign-artifact-uri",
        "missing-source-sha",
        "missing-architecture",
        "missing-schema",
        "missing-run-id",
        "legacy-component-alias",
    ],
)
def test_regeneration_revalidates_embedded_component_authority(
    tmp_path: Path,
    defect: str,
) -> None:
    report_uri = f"{ROOT}/reports/generations/generation/sim2real-report.json"
    rrd_uri = f"{ROOT}/reports/generations/generation/sim2real.rrd"
    mcap_uri = f"{ROOT}/reports/generations/generation/sim2real.mcap"
    components = [_component(stage) for stage in range(1, 14)]
    components.append(
        _stage14_component(
            rrd_uri=rrd_uri,
            mcap_uri=mcap_uri,
            report_uri=report_uri,
        )
    )
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "run_id": RUN_ID,
        "architecture": "npa.workflow/v0.0.1_compositional_standard_runtime",
        "source_sha": SOURCE_SHA,
        "rrd_uri": rrd_uri,
        "mcap_uri": mcap_uri,
        "report_uri": report_uri,
        "component_records": components,
    }
    if defect == "foreign-artifact-uri":
        components[9]["artifacts"]["report"] = f"{ROOT}/unrelated/report.json"
        _rehash(components[9])
    elif defect == "missing-source-sha":
        report.pop("source_sha")
    elif defect == "missing-architecture":
        report.pop("architecture")
    elif defect == "missing-schema":
        report.pop("schema")
    elif defect == "missing-run-id":
        report.pop("run_id")
    elif defect == "legacy-component-alias":
        report["components"] = report.pop("component_records")
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(_gold()))
    inputs = regen._RegenInputs(
        inner_evidence=_evidence(),
        heldout_report=_gold(),
        heldout_path=heldout_path,
        report_path=tmp_path / "report.json",
        report=report,
        current_decision=_decision(_gold()),
    )
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._validate_regen_component_authority(
            _config(),
            tmp_path,
            object(),
            inputs,
            sync_inputs=False,
        )


def test_stage14_does_not_update_aliases_after_immutable_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = Namespace(run_id=RUN_ID, outer_iteration=1)
    state = stage14._stage14_state(
        args,
        ROOT,
        tmp_path,
        tmp_path / "run",
        _evidence(),
        _gold(),
    )
    reports = state.local / "reports"
    reports.mkdir(parents=True)
    (reports / "sim2real.rrd").write_bytes(b"rrd")
    (reports / "sim2real.mcap").write_bytes(b"mcap")
    uploaded: list[str] = []

    class Storage:
        def upload_file(self, _source: str, destination: str) -> str:
            uploaded.append(destination)
            if destination == state.mcap_uri:
                raise RuntimeError("immutable MCAP failed")
            return destination

    monkeypatch.setattr(stage14, "storage", lambda: Storage())
    result = SimpleNamespace(
        heldout_frame_count=1,
        to_dict=lambda: {"heldout_frame_count": 1},
    )
    report = {
        "component_records": [
            _stage14_component(
                rrd_uri=state.rrd_uri,
                mcap_uri=state.mcap_uri,
                report_uri=state.report_uri,
            )
        ]
    }
    with pytest.raises(RuntimeError, match="immutable MCAP failed"):
        stage14._publish_stage14_outputs(state, report, reports, result, result)
    assert state.canonical_rrd_uri not in uploaded
    assert state.canonical_mcap_uri not in uploaded
    assert state.canonical_report_uri not in uploaded


def test_disabled_mcap_removes_every_stale_uri(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    component = _stage14_component(
        rrd_uri="s3://old/rrd",
        mcap_uri="s3://old/mcap",
        report_uri="s3://old/report",
    )
    report = {
        "mcap_uri": "s3://old/mcap",
        "canonical_mcap_uri": "s3://old/canonical.mcap",
        "visualization": {"mcap_s3_uri": "s3://old/mcap"},
        "recording_summaries": {"mcap": {"status": "written"}},
        "component_records": [component],
    }
    path.write_text(json.dumps(report))
    regen._seal_regen_publication_report(
        path,
        publication_id="e" * 64,
        rrd_uri="s3://new/rrd",
        mcap_uri="",
        report_uri="s3://new/report",
    )
    sealed = json.loads(path.read_text())
    assert "mcap_uri" not in sealed
    assert "canonical_mcap_uri" not in sealed
    assert "mcap_s3_uri" not in sealed["visualization"]
    assert "mcap" not in sealed["recording_summaries"]
    assert "mcap" not in sealed["component_records"][0]["artifacts"]


def test_zero_byte_rrd_is_never_uploaded(tmp_path: Path) -> None:
    rrd = tmp_path / "reports/sim2real.rrd"
    rrd.parent.mkdir(parents=True)
    rrd.write_bytes(b"")
    uploads: list[str] = []
    client = SimpleNamespace(
        upload_file=lambda _source, uri: uploads.append(uri) or uri,
        upload_directory=lambda *_a, **_k: None,
    )
    with pytest.raises(regen.Sim2RealRerunRegenError, match="non-empty"):
        regen.publish_regen_outputs(_config(), tmp_path, client=client)
    assert uploads == []


def test_legacy_render_manifest_rejects_duplicate_json_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    renders = tmp_path / "renders"
    renders.mkdir()
    monkeypatch.setattr(
        regen,
        "_list_common_prefixes",
        lambda *_args: ["sim2real/run-a/byo-eval/attempt/"],
    )
    monkeypatch.setattr(regen, "_download_render_tree", lambda *_a, **_k: True)

    def download(
        _storage: Any,
        _uri: str,
        target: Path,
        **_kwargs: Any,
    ) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"episodes": [], "episodes": [{"env_id": "evil"}]}')
        return True

    monkeypatch.setattr(regen, "_download_if_exists", download)
    with pytest.raises(ValueError, match="duplicate JSON field"):
        regen._sync_legacy_render_source(
            object(),
            f"{ROOT}/byo-eval/",
            renders,
            render_suffix="renders/",
            manifest_suffix="render-manifest.json",
            manifest_name="manifest.json",
            config=_config(),
            local_dir=tmp_path,
            heldout_report={},
        )
