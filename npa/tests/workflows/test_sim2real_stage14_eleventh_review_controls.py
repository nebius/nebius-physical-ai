from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from npa.workflows.sim2real import engine, workflow_stage
from npa.workflows.sim2real.byo_isaac_trainer import artifact_tag, k8s_job_name
from npa.workflows.sim2real.models import Sim2RealLoopConfig
import npa.workflows.sim2real.stage14_finalize as stage14
import npa.workflows.sim2real_rerun_regen as regen


RUN_ID = "run-a"
ROOT = f"s3://demo-bucket/sim2real-b/{RUN_ID}"
CHECKPOINT = f"{ROOT}/byo-trainer/job/model_latest.pt"
DIGEST = "a" * 64


def _config() -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id=RUN_ID,
        s3_bucket="demo-bucket",
        s3_prefix="sim2real-b",
    )


def _identity(uri: str = CHECKPOINT) -> dict[str, Any]:
    return {
        "checkpoint_uri": uri,
        "checkpoint_sha256": DIGEST,
        "checkpoint_size_bytes": 128,
        "generator_policy_sha256": DIGEST,
    }


def _evidence(uri: str = CHECKPOINT) -> dict[str, Any]:
    identity = _identity(uri)
    return {
        "outer_iteration": 1,
        "iterations": [],
        "selected_checkpoint_uri": uri,
        "final_checkpoint_uri": uri,
        "checkpoint_selection": dict(identity),
        "checkpoint_candidates": [dict(identity)],
    }


def _learned_gold(uri: str = CHECKPOINT) -> dict[str, Any]:
    return {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "policy_checkpoint_sha256": DIGEST,
        "policy_checkpoint_size_bytes": 128,
        "deployable_policy_eval": True,
        "policy_inference_provenance": {
            **_identity(uri),
            "loaded_for_inference": True,
            "stock_or_scripted_policy": False,
            "actor_is_learned": True,
            "scripted_post_actor_controller": False,
            "policy_composition": "learned_actor_only",
            "post_actor_controller": None,
        },
    }


def _canonical_gold_lineage() -> dict[str, Any]:
    return {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": f"{ROOT}/eval/gold-heldout/outer-01/renders/",
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }


def _state(
    tmp_path: Path,
    *,
    evidence: dict[str, Any] | None = None,
    gold: dict[str, Any] | None = None,
) -> stage14._Stage14State:
    return stage14._Stage14State(
        args=Namespace(run_id=RUN_ID, outer_iteration=1),
        root=ROOT,
        work=tmp_path,
        local=tmp_path / "run",
        evidence=evidence or {},
        gold=gold or {},
        rrd_uri=f"{ROOT}/reports/sim2real.rrd",
        mcap_uri=f"{ROOT}/reports/sim2real.mcap",
        report_uri=f"{ROOT}/reports/sim2real-report.json",
    )


def test_stage10_accepts_exact_byo_render_producer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = k8s_job_name(
        "s2r-byo-isaac-eval",
        RUN_ID,
        artifact_tag("gold-o01"),
    )
    producer = f"{ROOT}/byo-eval/{job}/renders"
    report = _learned_gold()
    report.update(
        {
            "component_invocation": {"mode": "npa_workflow_skypilot_task"},
            "render_manifest": {
                "renders_s3_uri": producer,
                "episodes": [{"env_id": "env-1", "frames": ["camera-000.png"]}],
            },
        }
    )
    canonical_uploads: list[str] = []

    class Storage:
        def download_directory(self, source: str, destination: str) -> None:
            assert source.rstrip("/") == producer
            frame = Path(destination) / "env-1/camera-000.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"png")

        def upload_directory(self, _source: str, destination: str) -> None:
            canonical_uploads.append(destination)

    monkeypatch.setattr(workflow_stage, "_root", lambda _args: ROOT)
    monkeypatch.setattr(workflow_stage, "_work", lambda _stage: tmp_path)
    monkeypatch.setattr(workflow_stage, "read_json", lambda *_a, **_k: _evidence())
    monkeypatch.setattr(workflow_stage, "_run_eval", lambda *_a, **_k: report)
    monkeypatch.setattr(
        workflow_stage,
        "_assert_embodiment_evidence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(workflow_stage, "storage", lambda: Storage())
    monkeypatch.setattr(workflow_stage, "write_loop_output", lambda *_a, **_k: "")
    monkeypatch.setattr(
        workflow_stage,
        "publish_component_record",
        lambda **_kwargs: {},
    )

    workflow_stage._stage10(Namespace(run_id=RUN_ID, outer_iteration=1, gold_count=1))

    assert canonical_uploads == [f"{ROOT}/eval/gold-heldout/outer-01/renders/"]


def test_stage10_rejects_foreign_checkpoint_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence("s3://foreign-bucket/other-run/model.pt")
    evidence["run_id"] = "other-run"
    reached: list[str] = []
    monkeypatch.setattr(workflow_stage, "_root", lambda _args: ROOT)
    monkeypatch.setattr(workflow_stage, "_work", lambda _stage: tmp_path)
    monkeypatch.setattr(workflow_stage, "read_json", lambda *_a, **_k: evidence)
    monkeypatch.setattr(
        workflow_stage,
        "_run_eval",
        lambda *_a, **_k: reached.append("eval") or {},
    )

    try:
        workflow_stage._stage10(
            Namespace(run_id=RUN_ID, outer_iteration=1, gold_count=1)
        )
    except RuntimeError:
        pass
    assert reached == []


def test_heldout_only_rejects_foreign_checkpoint_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence("s3://foreign-bucket/other-run/model.pt")
    evidence["run_id"] = "other-run"
    reached: list[str] = []

    class EvaluationBoundaryReached(Exception):
        pass

    def evaluation_boundary(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        reached.append("eval")
        raise EvaluationBoundaryReached

    monkeypatch.setattr(
        regen,
        "_sync_heldout_eval_inputs",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        engine,
        "run_heldout_eval",
        evaluation_boundary,
    )

    try:
        regen.rerun_heldout_eval_only(
            _config(),
            local_dir=tmp_path,
            publish=False,
            client=object(),
        )
    except (regen.Sim2RealRerunRegenError, EvaluationBoundaryReached):
        pass
    assert reached == []


def test_canonical_stage14_rejects_invalid_policy_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(
        tmp_path,
        evidence=_evidence(),
        gold=_learned_gold(f"{ROOT}/byo-trainer/job/other.pt"),
    )
    decision = state.local / "outer_loop/decision.json"
    robot = state.local / "stage_02_assets/consumed_robot_spec.json"
    decision.parent.mkdir(parents=True)
    robot.parent.mkdir(parents=True)
    decision.write_text(
        json.dumps({"decision": "promote_checkpoint", "checkpoint_uri": CHECKPOINT}),
        encoding="utf-8",
    )
    robot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(stage14, "_materialize_stage14", lambda *_a, **_k: state)
    publication_boundary: list[str] = []

    class PublicationBoundaryReached(Exception):
        pass

    def component_publication_boundary(
        _state: stage14._Stage14State,
    ) -> list[dict[str, Any]]:
        publication_boundary.append("component")
        raise PublicationBoundaryReached

    monkeypatch.setattr(
        stage14,
        "_load_component_records",
        component_publication_boundary,
    )
    monkeypatch.setattr(stage14, "source_sha", lambda: "b" * 40)

    try:
        stage14.finalize_in_work(state.args, root=ROOT, work=tmp_path)
    except (RuntimeError, PublicationBoundaryReached):
        pass
    assert publication_boundary == []


def test_heldout_only_rejects_scripted_policy_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _learned_gold()
    report["deployable_policy_eval"] = False
    report["policy_inference_provenance"].update(
        {
            "stock_or_scripted_policy": True,
            "actor_is_learned": False,
            "scripted_post_actor_controller": True,
            "policy_composition": "scripted_policy",
            "post_actor_controller": "oracle",
        }
    )
    uploads: list[str] = []

    class Storage:
        def upload_file(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

    monkeypatch.setattr(
        regen,
        "_sync_heldout_eval_inputs",
        lambda *_args, **_kwargs: _evidence(),
    )
    monkeypatch.setattr(engine, "run_heldout_eval", lambda *_a, **_k: report)
    monkeypatch.setattr(
        regen,
        "_sync_heldout_eval_renders",
        lambda *_args, **_kwargs: None,
    )
    publication_boundary: list[str] = []

    class PublicationBoundaryReached(Exception):
        pass

    def publication_boundary_reached(*_args: Any, **_kwargs: Any) -> None:
        publication_boundary.append("publish")
        raise PublicationBoundaryReached

    monkeypatch.setattr(
        regen,
        "_publish_heldout_eval_outputs",
        publication_boundary_reached,
    )

    try:
        regen.rerun_heldout_eval_only(
            _config(),
            local_dir=tmp_path,
            publish=True,
            client=Storage(),
        )
    except (regen.Sim2RealRerunRegenError, PublicationBoundaryReached):
        pass
    assert uploads == []
    assert publication_boundary == []


def test_canonical_stage14_rejects_zero_heldout_frames_before_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    reports = state.local / "reports"
    reports.mkdir(parents=True)
    (reports / "sim2real.rrd").write_bytes(b"rrd")
    (reports / "sim2real.mcap").write_bytes(b"mcap")
    uploads: list[str] = []

    class Storage:
        def upload_file(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

    zero = SimpleNamespace(
        heldout_frame_count=0,
        to_dict=lambda: {"heldout_frame_count": 0},
    )
    monkeypatch.setattr(stage14, "storage", lambda: Storage())

    try:
        stage14._publish_stage14_outputs(
            state,
            {"component_records": [{"stage": 14}]},
            reports,
            zero,
            zero,
        )
    except RuntimeError:
        pass
    assert uploads == []


def test_remote_pair_is_bound_to_payload_outer_iteration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_sync(
        _config: Sim2RealLoopConfig,
        local_dir: Path,
        **_kwargs: Any,
    ) -> tuple[Path, Path]:
        evidence = local_dir / "inner_loop/outer-01/evidence.json"
        report = local_dir / "eval/gold-heldout/outer-01/report.json"
        evidence.parent.mkdir(parents=True)
        report.parent.mkdir(parents=True)
        evidence.write_text(
            json.dumps({"outer_iteration": 2, "iterations": []}),
            encoding="utf-8",
        )
        report.write_text(
            json.dumps({"evaluation_split": "gold_heldout", "outer_iteration": 2}),
            encoding="utf-8",
        )
        return evidence, report

    monkeypatch.setattr(regen, "sync_regen_inputs", fake_sync)
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._load_regen_state(
            _config(),
            tmp_path,
            object(),
            sync_inputs=True,
        )


def test_stage14_rejects_cross_iteration_artifact_sources() -> None:
    evidence = {
        "outer_iteration": 1,
        "iterations": [
            {
                "iteration": 1,
                "actions_uri": f"{ROOT}/actions/train/outer-99/iter-07/",
                "vlm_eval_uri": (
                    f"{ROOT}/vlm_eval/train/outer-99/iter-07/evaluations/"
                ),
                "signal_uri": (f"{ROOT}/vlm_eval/train/outer-99/iter-07/signals/"),
            }
        ],
    }

    with pytest.raises(RuntimeError, match="do not match"):
        stage14.download_plan(
            root=ROOT,
            outer_iteration=1,
            evidence=evidence,
            gold=_canonical_gold_lineage(),
        )


@pytest.mark.parametrize(
    "evidence",
    [
        {"iterations": []},
        {"outer_iteration": 1, "iterations": {}},
        {
            "outer_iteration": 1,
            "iterations": [
                {
                    "iteration": 1,
                    "actions_uri": f"{ROOT}/actions/train/outer-01/iter-01/",
                    "vlm_eval_uri": (
                        f"{ROOT}/vlm_eval/train/outer-01/iter-01/evaluations/"
                    ),
                    "signal_uri": (f"{ROOT}/vlm_eval/train/outer-01/iter-01/signals/"),
                },
                {
                    "iteration": 1,
                    "actions_uri": f"{ROOT}/actions/train/outer-01/iter-01/",
                    "vlm_eval_uri": (
                        f"{ROOT}/vlm_eval/train/outer-01/iter-01/evaluations/"
                    ),
                    "signal_uri": (f"{ROOT}/vlm_eval/train/outer-01/iter-01/signals/"),
                },
            ],
        },
    ],
)
def test_stage14_requires_one_canonical_outer_and_unique_iteration_list(
    evidence: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError):
        stage14.download_plan(
            root=ROOT,
            outer_iteration=1,
            evidence=evidence,
            gold=_canonical_gold_lineage(),
        )


def test_regen_rejects_foreign_run_uris_instead_of_localizing(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "inner_loop/outer-01/evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "outer_iteration": 1,
                "iterations": [
                    {
                        "iteration": 1,
                        "actions_uri": (
                            "s3://foreign/other/actions/train/outer-99/iter-07"
                        ),
                        "vlm_eval_uri": (
                            "s3://foreign/other/vlm_eval/train/"
                            "outer-99/iter-07/evaluations"
                        ),
                        "signal_uri": (
                            "s3://foreign/other/vlm_eval/train/outer-99/iter-07/signals"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        regen.Sim2RealRerunRegenError,
        match="artifact root|authority",
    ):
        regen._rewrite_inner_evidence_paths(tmp_path, evidence)


def test_regen_publication_rejects_symlinked_render_ancestor(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    evidence = run / "inner_loop/outer-01/evidence.json"
    report = run / "eval/gold-heldout/outer-01/report.json"
    evidence.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    report.write_text(
        json.dumps({"local_renders_dir": "eval/heldout/renders"}),
        encoding="utf-8",
    )
    outside = tmp_path / "outside-heldout"
    frame = outside / "renders/env-1/camera-000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"secret")
    (run / "eval/heldout").symlink_to(outside, target_is_directory=True)
    recording = run / "reports/sim2real.rrd"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"rrd")
    uploads: list[str] = []

    class Storage:
        def upload_file(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

    with pytest.raises(regen.Sim2RealRerunRegenError, match="symlinked ancestor"):
        regen.publish_regen_outputs(_config(), run, client=Storage())
    assert uploads == []


def test_archived_pre_identity_replay_disables_policy_access(
    tmp_path: Path,
) -> None:
    old_candidate = {
        "checkpoint_uri": CHECKPOINT,
        "checkpoint_sha256": DIGEST,
        "validation_report_uri": f"{ROOT}/eval/validation/report.json",
        "validation_report": {"success_rate": 1.0},
    }
    evidence = tmp_path / "inner_loop/outer-01/evidence.json"
    heldout = tmp_path / "eval/heldout/report.json"
    final_report = tmp_path / "reports/sim2real-report.json"
    evidence.parent.mkdir(parents=True)
    heldout.parent.mkdir(parents=True)
    final_report.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "outer_iteration": 1,
                "iterations": [],
                "selected_checkpoint_uri": CHECKPOINT,
                "final_checkpoint_uri": CHECKPOINT,
                "checkpoint_selection": dict(old_candidate),
                "checkpoint_candidates": [dict(old_candidate)],
            }
        ),
        encoding="utf-8",
    )
    heldout.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
    final_report.write_text(json.dumps({"components": []}), encoding="utf-8")

    state = regen._load_regen_state(
        _config(),
        tmp_path,
        object(),
        sync_inputs=False,
    )

    assert state.report["components"] == []
    assert state.policy_access["deployable_policy"] is False
    assert state.policy_access["authenticated_download_command"] == ""


def test_modern_evidence_without_checkpoint_identity_fails_closed() -> None:
    with pytest.raises(regen.Sim2RealRerunRegenError, match="identity is missing"):
        regen._current_checkpoint_sources(
            {
                "schema": "npa.sim2real.inner_loop_evidence.v1",
                "outer_iteration": 1,
                "iterations": [],
            }
        )
