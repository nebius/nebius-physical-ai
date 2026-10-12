from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from npa.workflows.sim2real import (
    byo_isaac_eval,
    component_authority,
    stage10_authority,
    workflow_stage,
)
from npa.workflows.sim2real.byo_isaac_trainer import artifact_tag, k8s_job_name
from npa.workflows.sim2real.decision_authority import gold_report_sha256
from npa.workflows.sim2real.models import Sim2RealLoopConfig
import npa.workflows.sim2real.stage14_finalize as stage14
import npa.workflows.sim2real_rerun_regen as regen

pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")


RUN_ID = "run-a"
ROOT = f"s3://demo-bucket/sim2real-b/{RUN_ID}"
TRAIN_JOB = k8s_job_name("s2r-byo-isaac-train", RUN_ID)
TRAIN_TAG = artifact_tag("outer-01-iter-01")
CHECKPOINT = f"{ROOT}/byo-trainer/{TRAIN_JOB}/{TRAIN_TAG}/model_latest.pt"
DIGEST = "a" * 64
SOURCE_SHA = "b" * 40
IMAGE_DIGEST = "c" * 64
EVAL_ATTEMPT_TAG = "gold-o01-attempt-" + "d" * 32
FRAME_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
    "0000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
)
COMPONENT_NAMES = {
    1: "stage_01_trigger",
    2: "stage_02_assets",
    3: "stage_03_augment",
    4: "stage_04_envs_raw",
    5: "stage_05_envs_train",
    6: "stage_06_tokens",
    7: "stage_07_actions_train",
    8: "stage_08_vlm_eval_train",
    9: "stage_09_training_signal",
    10: "stage_10_eval_heldout",
    11: "stage_11_outer_loop",
    12: "stage_12_external_validation",
    13: "stage_13_retrigger",
    14: "stage_14_rerun_viz",
}
REQUIRED_ARTIFACTS = {
    stage: required
    for stage, (_name, _tier, required) in stage14._COMPONENT_CONTRACTS.items()
}
REQUIRED_ARTIFACTS[14] = ("rrd", "mcap", "report")


@pytest.fixture(autouse=True)
def _exact_stage_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage14, "source_sha", lambda: SOURCE_SHA)


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
    candidate = {
        **_identity(uri),
        "outer_iteration": 1,
        "inner_iteration": 1,
        "training_iteration": 10,
        "validation_report_uri": f"{ROOT}/eval/validation/report.json",
    }
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
        "selected_checkpoint_uri": uri,
        "final_checkpoint_uri": uri,
        "checkpoint_selection": dict(candidate),
        "checkpoint_candidates": [dict(candidate)],
    }


def _gold(uri: str = CHECKPOINT) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.heldout_eval.v1",
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "policy_checkpoint": uri,
        "policy_checkpoint_sha256": DIGEST,
        "policy_checkpoint_size_bytes": 128,
        "deployable_policy_eval": True,
        "success_rate": 1.0,
        "per_env": [{"env_id": "env-1", "success": True}],
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "evaluation_attempt_tag": EVAL_ATTEMPT_TAG,
            "source_s3_uri": (
                f"{ROOT}/byo-eval/"
                f"{k8s_job_name('s2r-byo-isaac-eval', RUN_ID, artifact_tag(EVAL_ATTEMPT_TAG))}/"
                "renders/"
            ),
            "canonical_s3_uri": (
                f"{ROOT}/eval/gold-heldout/outer-01/attempts/"
                f"{EVAL_ATTEMPT_TAG}/renders/"
            ),
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
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


def _stage4_lane(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = {
        "image": f"ghcr.io/example/npa@sha256:{IMAGE_DIGEST}",
        "image_digest": f"sha256:{IMAGE_DIGEST}",
        "source_sha": SOURCE_SHA,
        "execution_mode": "standard_npa_workflow_skypilot",
        "workflow_job": f"job-4-{index}",
        "gpu_products": ["NVIDIA H100"],
        "gpu_rows": ["NVIDIA H100, GPU-test-0000"],
    }
    proof = {
        "schema": "npa.sim2real.envgen_shard_execution.v1",
        "shard_index": index,
        "shard_count": 2,
        "provenance": provenance,
    }
    lane = {
        "schema": "npa.sim2real.component_lane_record.v1",
        "stage": 4,
        "lane": f"shard-{index:05d}",
        "tier": "WORKS",
        "evidence": f"generated shard {index}",
        "artifacts": {
            "raw_envs": f"{ROOT}/envs/raw/",
            "shard_index": index,
            "shard_count": 2,
            "provenance": f"{ROOT}/envs/raw/provenance-{index:05d}.json",
            **provenance,
        },
    }
    return proof, _rehash_component(lane)


def _component_record(stage: int) -> dict[str, Any]:
    authority = component_authority._authority_uris(ROOT, _evidence(), _gold())
    artifacts: dict[str, Any] = {"source_sha": SOURCE_SHA}
    artifacts.update(
        {
            key: (
                authority[stage][key]
                if key in stage14._COMPONENT_URI_KEYS.get(stage, ())
                else f"stage-{stage}-{key}"
            )
            for key in REQUIRED_ARTIFACTS[stage]
        }
    )
    if stage == 10:
        artifacts["render_lineage"] = _gold()["render_lineage"]
        artifacts["gold_report_sha256"] = gold_report_sha256(_gold())
    if stage == 4:
        lanes = [_stage4_lane(index) for index in range(2)]
        artifacts.update(
            {
                "shard_count": 2,
                "shard_provenance": [proof for proof, _record in lanes],
                "lane_records": [record for _proof, record in lanes],
            }
        )
    if stage != 12:
        provenance = {
            "image": f"ghcr.io/example/npa@sha256:{IMAGE_DIGEST}",
            "image_digest": f"sha256:{IMAGE_DIGEST}",
        }
        if stage == 4:
            provenance.update(
                {
                    "execution_mode": "standard_npa_workflow_parallel_join",
                    "workflow_jobs": ["job-4-0", "job-4-1"],
                    "lane_count": 2,
                    "gpu_products": ["NVIDIA H100"],
                    "gpu_rows": ["NVIDIA H100, GPU-test-0000"],
                }
            )
        else:
            provenance.update(
                {
                    "execution_mode": "standard_npa_workflow_skypilot",
                    "workflow_job": f"job-{stage}",
                }
            )
            if stage in {3, 7, 9, 10}:
                provenance.update(
                    {
                        "gpu_products": ["NVIDIA H100"],
                        "gpu_rows": ["NVIDIA H100, GPU-test-0000"],
                    }
                )
        artifacts.update(provenance)
    payload = {
        "schema": "npa.sim2real.component_record.v1",
        "stage": stage,
        "name": COMPONENT_NAMES[stage],
        "tier": "SEAM" if stage == 12 else "WORKS",
        "evidence": f"stage {stage} completed",
        "artifacts": artifacts,
        "next_action": "CONTINUE",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}


def _rehash_component(record: dict[str, Any]) -> dict[str, Any]:
    material = {key: value for key, value in record.items() if key != "content_sha256"}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    record["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def _state(tmp_path: Path) -> stage14._Stage14State:
    return stage14._Stage14State(
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
        rrd_uri=f"{ROOT}/reports/sim2real.rrd",
        mcap_uri=f"{ROOT}/reports/sim2real.mcap",
        report_uri=f"{ROOT}/reports/sim2real-report.json",
    )


def _record_reader(uri: str, **_kwargs: Any) -> dict[str, Any]:
    if "/envs/raw/provenance-" in uri:
        index = int(uri.rsplit("provenance-", 1)[1][:5])
        return _component_record(4)["artifacts"]["shard_provenance"][index]
    if "/components/lanes/stage_04/shard-" in uri:
        index = int(uri.rsplit("/shard-", 1)[1][:5])
        return _component_record(4)["artifacts"]["lane_records"][index]
    stage = int(uri.rsplit("stage_", 1)[1][:2])
    return _component_record(stage)


def test_stage14_encoder_failure_does_not_publish_works_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published: list[int] = []
    monkeypatch.setattr(
        stage14, "_materialize_stage14", lambda *_a, **_k: _state(tmp_path)
    )
    monkeypatch.setattr(
        stage14,
        "_capture_stage14_publication_snapshots",
        lambda state: state,
    )
    monkeypatch.setattr(
        stage14, "_assert_stage14_publication_preconditions", lambda *_a: None
    )
    monkeypatch.setattr(
        stage14,
        "_validate_stage14_materialized_frames",
        lambda *_a: None,
    )
    monkeypatch.setattr(stage14, "read_json", _record_reader)
    monkeypatch.setattr(stage14, "source_sha", lambda: SOURCE_SHA)
    for publisher in (
        "publish_built_component_history",
        "publish_built_component_pointer",
    ):
        monkeypatch.setattr(
            stage14,
            publisher,
            lambda **kwargs: published.append(kwargs["expected_stage"]),
        )
    monkeypatch.setattr(
        stage14,
        "_build_stage14_report",
        lambda *_a: ({}, {}, tmp_path / "reports"),
    )
    monkeypatch.setattr(
        stage14,
        "_emit_stage14_outputs",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("encoder failed")),
    )

    with pytest.raises(RuntimeError, match="encoder failed"):
        stage14.finalize_in_work(Namespace(), root=ROOT, work=tmp_path)
    assert published == []


@pytest.mark.parametrize("defect", ["digest", "name"])
def test_stage14_rejects_tampered_component_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    defect: str,
) -> None:
    def reader(uri: str, **_kwargs: Any) -> dict[str, Any]:
        record = _record_reader(uri)
        if record["stage"] == 5:
            record["content_sha256" if defect == "digest" else "name"] = "tampered"
        return record

    monkeypatch.setattr(stage14, "read_json", reader)
    with pytest.raises(RuntimeError):
        stage14._load_component_records(_state(tmp_path))


@pytest.mark.parametrize("defect", ["missing", "foreign", "stale-source"])
def test_stage14_rejects_rehashed_incomplete_or_foreign_components(
    tmp_path: Path,
    defect: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reader(uri: str, **_kwargs: Any) -> dict[str, Any]:
        record = _record_reader(uri)
        if record["stage"] != 5:
            return record
        if defect == "missing":
            record["artifacts"].pop("gold_envs")
        elif defect == "foreign":
            record["artifacts"]["gold_envs"] = "s3://foreign/run/envs.jsonl"
        else:
            record["artifacts"]["source_sha"] = "f" * 40
        return _rehash_component(record)

    monkeypatch.setattr(stage14, "read_json", reader)
    with pytest.raises(RuntimeError, match="invalid ComponentRecord"):
        stage14._load_component_records(_state(tmp_path))


def test_stage14_rejects_stale_decision_authority() -> None:
    gold = _gold()
    decision = {
        "schema": "npa.sim2real.threshold_decision.v1",
        "run_id": "other-run",
        "outer_iteration": 99,
        "decision": "promote_checkpoint",
        "checkpoint_uri": CHECKPOINT,
        "gold_report_uri": f"{ROOT}/eval/gold-heldout/outer-99/report.json",
    }

    with pytest.raises(RuntimeError):
        stage14._stage14_policy_metadata(
            _evidence(),
            decision,
            gold,
            run_root=ROOT,
            run_id=RUN_ID,
            expected_threshold=0.5,
            expected_early_exit=False,
            expected_gold_report_sha256=gold_report_sha256(gold),
        )


def test_regen_rejects_stale_current_decision_before_policy_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inner = tmp_path / "inner_loop/outer-01/evidence.json"
    heldout = tmp_path / "eval/gold-heldout/outer-01/report.json"
    decision = tmp_path / "outer_loop/decision.json"
    for path, payload in (
        (inner, _evidence()),
        (heldout, _gold()),
        (
            decision,
            {
                "schema": "npa.sim2real.threshold_decision.v1",
                "run_id": RUN_ID,
                "outer_iteration": 2,
                "decision": "promote_checkpoint",
                "checkpoint_uri": CHECKPOINT,
                "gold_report_uri": f"{ROOT}/eval/gold-heldout/outer-02/report.json",
            },
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    reached: list[str] = []
    monkeypatch.setattr(
        regen,
        "_ensure_policy_access_metadata",
        lambda *_a, **_k: reached.append("access") or {},
    )
    monkeypatch.setattr(
        regen,
        "_validate_regen_component_authority",
        lambda *_a, **_k: None,
    )

    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._load_regen_state(_config(), tmp_path, object(), sync_inputs=False)
    assert reached == []


def test_heldout_only_syncs_canonical_gold_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloads: list[str] = []

    def download_tree(_storage: Any, uri: str, *_args: Any, **_kwargs: Any) -> bool:
        downloads.append(uri)
        return True

    def download_evidence(
        _storage: Any,
        _uri: str,
        target: Path,
        **_kwargs: Any,
    ) -> bool:
        target.write_text(json.dumps(_evidence()), encoding="utf-8")
        return True

    monkeypatch.setattr(regen, "_download_directory_fresh", download_tree)
    monkeypatch.setattr(regen, "_download_if_exists", download_evidence)
    regen._sync_heldout_eval_inputs(
        _config(),
        tmp_path,
        object(),
        f"{ROOT}/",
        outer_iteration=1,
    )
    assert downloads == [f"{ROOT}/envs/gold-heldout/"]


def test_heldout_only_publishes_only_immutable_attempt_evidence(tmp_path: Path) -> None:
    renders = tmp_path / "eval/gold-heldout/outer-01/renders/env-1"
    renders.mkdir(parents=True)
    (renders / "camera-000.png").write_bytes(b"png")
    uploads: list[tuple[bytes, str]] = []

    class Storage:
        def read_bytes_with_etag(self, _destination: str) -> None:
            return None

        def put_bytes_conditional(
            self,
            payload: bytes,
            destination: str,
            **_kwargs: object,
        ) -> str:
            uploads.append((payload, destination))
            return '"etag"'

        def upload_file(self, source: str, destination: str) -> str:
            uploads.append((Path(source).read_bytes(), destination))
            return destination

    report = _gold()
    attempt_tag = "gold_heldout-outer-01-attempt-" + "d" * 32
    report["evaluation_attempt_tag"] = attempt_tag
    report["render_manifest"] = {
        "evaluation_attempt_tag": attempt_tag,
        "renders_s3_uri": report["render_lineage"]["source_s3_uri"],
    }
    regen._publish_heldout_eval_outputs(
        _config(),
        tmp_path,
        report,
        outer_iteration=1,
        storage=Storage(),
    )
    published = json.loads(uploads[-1][0])
    attempt_root = f"{ROOT}/eval/gold-heldout/outer-01/attempts/{attempt_tag}"
    assert [destination for _source, destination in uploads] == [
        f"{attempt_root}/renders/env-1/camera-000.png",
        f"{attempt_root}/report.json",
    ]
    assert published["render_lineage"]["canonical_s3_uri"] == (
        f"{attempt_root}/renders/"
    )
    assert published["report_uri"] == f"{attempt_root}/report.json"


def test_regen_transaction_fences_convenience_aliases(tmp_path: Path) -> None:
    inner = tmp_path / "inner_loop/outer-01/evidence.json"
    report = tmp_path / "eval/gold-heldout/outer-01/report.json"
    renders = report.parent / "renders/env-1"
    final_report = tmp_path / "reports/sim2real-report.json"
    rrd = tmp_path / "reports/sim2real.rrd"
    for path, payload in ((inner, _evidence()), (report, _gold())):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    renders.mkdir(parents=True)
    (renders / "camera-000.png").write_bytes(b"png")
    final_report.parent.mkdir(parents=True, exist_ok=True)
    final_report.write_text(
        json.dumps(
            {
                "source_sha": SOURCE_SHA,
                "component_records": [_component_record(14)],
            }
        ),
        encoding="utf-8",
    )
    rrd.write_bytes(b"rrd")
    uploads: list[str] = []

    class Storage:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def read_bytes_with_etag(
            self,
            destination: str,
        ) -> tuple[bytes, str] | None:
            payload = self.objects.get(destination)
            return None if payload is None else (payload, '"etag"')

        def put_bytes_conditional(
            self,
            payload: bytes,
            destination: str,
            **_kwargs: object,
        ) -> str:
            uploads.append(destination)
            self.objects[destination] = bytes(payload)
            return '"etag"'

        def upload_file(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

    storage = Storage()
    regen.publish_regen_outputs(
        _config(),
        tmp_path,
        rrd_path=rrd,
        client=storage,
        snapshots=regen._capture_regen_publication_snapshots(_config(), storage),
    )
    canonical_report = f"{ROOT}/reports/sim2real-report.json"
    canonical_rrd = f"{ROOT}/reports/sim2real.rrd"
    assert uploads.index(canonical_rrd) < uploads.index(canonical_report)
    assert all(
        uploads.index(uri) < uploads.index(canonical_rrd)
        for uri in uploads
        if "/reports/generations/" in uri
    )
    assert uploads[-1] == f"{ROOT}/reports/.sim2real-publication.json"


def test_disabled_mcap_does_not_upload_stale_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "reports/sim2real.rrd"
    mcap = tmp_path / "reports/sim2real.mcap"
    rrd.parent.mkdir(parents=True)
    rrd.write_bytes(b"rrd")
    mcap.write_bytes(b"stale")
    uploads: list[str] = []

    class Storage:
        def upload_file(self, _source: str, destination: str) -> str:
            uploads.append(destination)
            return destination

    monkeypatch.setattr(
        regen,
        "emit_sim2real_mcap_if_enabled",
        lambda **_kwargs: {"status": "disabled"},
    )
    monkeypatch.setattr(
        regen,
        "publish_regen_outputs",
        lambda *_a, **_k: f"{ROOT}/reports/sim2real.rrd",
    )
    result = SimpleNamespace(
        heldout_frame_count=1,
        output_rrd_path=str(rrd),
        rollout_count=0,
        frame_count=1,
        synthetic_frame_count=0,
    )
    state = regen._RegenState({}, {}, tmp_path / "report.json", {}, {})
    regen._finalize_regen_result(
        _config(),
        tmp_path,
        Storage(),
        state,
        result,
        upload=True,
        output_rrd=rrd,
    )
    assert f"{ROOT}/reports/sim2real.mcap" not in uploads


def test_byo_eval_propagates_inprocess_render_upload_failure() -> None:
    assert "_upload_succeeded = False" in byo_isaac_eval.ISAAC_EVAL_SCRIPT
    assert "os._exit(0 if _upload_succeeded else 1)" in byo_isaac_eval.ISAAC_EVAL_SCRIPT


def test_heldout_only_rejects_foreign_render_producer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt = "gold_heldout-outer-01-attempt-" + "d" * 32
    report = {
        **_gold(),
        "evaluation_attempt_tag": attempt,
        "component_invocation": {"output_uri": f"{ROOT}/component-output/report.json"},
        "render_manifest": {
            "schema": "npa.sim2real.heldout_renders.v2",
            "evaluation_attempt_tag": attempt,
            "renders_s3_uri": "s3://foreign-bucket/other-run/renders/",
            "policy_checkpoint": {
                "uri": CHECKPOINT,
                "sha256": DIGEST,
                "size_bytes": 128,
            },
            "episodes": [{"env_id": "env-1", "frames": ["camera-000.png"]}],
            "frame_artifacts": {
                "env-1/camera-000.png": {
                    "sha256": hashlib.sha256(FRAME_BYTES).hexdigest(),
                    "size_bytes": len(FRAME_BYTES),
                }
            },
        },
    }

    def download(
        _storage: Any,
        _source: str,
        destination: Path,
        **_kwargs: Any,
    ) -> bool:
        frame = Path(destination) / "env-1/camera-000.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(FRAME_BYTES)
        return True

    monkeypatch.setattr(regen, "_download_render_tree", download)
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._sync_heldout_eval_renders(
            _config(), tmp_path, object(), report, _identity()
        )


def test_stage10_rejects_nontrainer_checkpoint_under_run_root() -> None:
    evidence = _evidence(f"{ROOT}/imports/unrelated.pt")
    with pytest.raises(RuntimeError):
        stage10_authority.validate_stage10_input_scope(
            Namespace(run_id=RUN_ID, outer_iteration=1),
            root=ROOT,
            evidence=evidence,
        )


def _run_stage10_with_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report: dict[str, Any],
    frame_paths: list[str],
) -> None:
    class Storage:
        def read_bytes_with_etag(self, _uri: str) -> None:
            return None

        def put_bytes_conditional(
            self,
            _payload: bytes,
            _uri: str,
            **_kwargs: object,
        ) -> str:
            return '"etag"'

        def download_directory(self, _source: str, destination: str) -> None:
            for relative in frame_paths:
                frame = Path(destination) / relative
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(FRAME_BYTES)

        def upload_directory(self, _source: str, _destination: str) -> None:
            return None

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


def _stage10_report() -> dict[str, Any]:
    producer = stage10_authority.expected_byo_render_prefix(
        root=ROOT,
        run_id=RUN_ID,
        outer_iteration=1,
        evaluation_tag=EVAL_ATTEMPT_TAG,
    )
    report = _gold()
    report["component_invocation"] = {"mode": "npa_workflow_skypilot_task"}
    report["render_manifest"] = {
        "schema": "npa.sim2real.heldout_renders.v2",
        "evaluation_attempt_tag": EVAL_ATTEMPT_TAG,
        "renders_s3_uri": producer,
        "policy_checkpoint": {
            "uri": CHECKPOINT,
            "sha256": DIGEST,
            "size_bytes": 128,
        },
        "episodes": [{"env_id": "env-1", "frames": ["camera-000.png"]}],
        "frame_artifacts": {
            "env-1/camera-000.png": {
                "sha256": hashlib.sha256(FRAME_BYTES).hexdigest(),
                "size_bytes": len(FRAME_BYTES),
            }
        },
    }
    return report


def test_stage10_rejects_render_manifest_checkpoint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _stage10_report()
    report["render_manifest"]["policy_checkpoint"]["uri"] = f"{ROOT}/imports/other.pt"
    with pytest.raises(RuntimeError):
        _run_stage10_with_frames(
            monkeypatch,
            tmp_path,
            report,
            ["env-1/camera-000.png"],
        )


def test_stage10_rejects_undeclared_or_missing_render_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _stage10_report()
    with pytest.raises(RuntimeError):
        _run_stage10_with_frames(
            monkeypatch,
            tmp_path,
            report,
            ["old-env/camera-000.png"],
        )


def test_stage10_rejects_mutable_render_producer_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _stage10_report()
    report["render_manifest"].pop("evaluation_attempt_tag")
    report["render_manifest"]["renders_s3_uri"] = (
        stage10_authority.expected_byo_render_prefix(
            root=ROOT,
            run_id=RUN_ID,
            outer_iteration=1,
        )
    )
    with pytest.raises(RuntimeError, match="immutable attempt"):
        _run_stage10_with_frames(
            monkeypatch,
            tmp_path,
            report,
            ["env-1/camera-000.png"],
        )


def test_legacy_heldout_report_outer_must_match_selected_inner(tmp_path: Path) -> None:
    inner = tmp_path / "inner_loop/outer-01/evidence.json"
    heldout = tmp_path / "eval/heldout/report.json"
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._assert_selected_pair_outer_iteration(
            inner,
            heldout,
            _evidence(),
            {"outer_iteration": 2},
        )


def test_stage14_uses_generation_uris_and_commits_journal_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(stage14.secrets, "token_hex", lambda _size: "e" * 32)
    state = stage14._stage14_state(
        Namespace(run_id=RUN_ID, outer_iteration=1),
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
    events: list[str] = []

    class Storage:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def read_bytes_with_etag(
            self,
            destination: str,
        ) -> tuple[bytes, str] | None:
            payload = self.objects.get(destination)
            return None if payload is None else (payload, '"etag"')

        def put_bytes_conditional(
            self,
            payload: bytes,
            destination: str,
            **_kwargs: object,
        ) -> str:
            events.append(destination)
            self.objects[destination] = bytes(payload)
            return '"etag"'

        def upload_file(self, _source: str, destination: str) -> str:
            events.append(destination)
            return destination

    monkeypatch.setattr(stage14, "storage", lambda: Storage())
    result = SimpleNamespace(
        heldout_frame_count=1,
        to_dict=lambda: {"heldout_frame_count": 1},
    )
    report = {"component_records": [_component_record(14)]}
    state = stage14._capture_stage14_publication_snapshots(state)

    stage14._publish_stage14_outputs(
        state,
        report,
        reports,
        result,
        result,
    )

    assert "/reports/generations/" in state.rrd_uri
    assert events[-1] == f"{ROOT}/reports/.sim2real-publication.json"
    assert events[-2] == f"{ROOT}/components/stage_14.json"
    canonical_report = f"{ROOT}/reports/sim2real-report.json"
    assert events.index(f"{ROOT}/reports/sim2real.rrd") < events.index(canonical_report)
    assert events.index(f"{ROOT}/reports/sim2real.mcap") < events.index(
        canonical_report
    )


def test_regen_report_seals_immutable_recording_generation(tmp_path: Path) -> None:
    inner = tmp_path / "inner_loop/outer-01/evidence.json"
    heldout = tmp_path / "eval/gold-heldout/outer-01/report.json"
    renders = heldout.parent / "renders/env-1"
    final_report = tmp_path / "reports/sim2real-report.json"
    rrd = tmp_path / "reports/sim2real.rrd"
    for path, payload in ((inner, _evidence()), (heldout, _gold())):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    renders.mkdir(parents=True)
    (renders / "camera-000.png").write_bytes(b"png")
    final_report.parent.mkdir(parents=True, exist_ok=True)
    final_report.write_text("{}", encoding="utf-8")
    rrd.write_bytes(b"rrd")
    uploads: list[tuple[str, dict[str, Any] | None]] = []

    class Storage:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def read_bytes_with_etag(
            self,
            destination: str,
        ) -> tuple[bytes, str] | None:
            payload = self.objects.get(destination)
            return None if payload is None else (payload, '"etag"')

        def put_bytes_conditional(
            self,
            payload: bytes,
            destination: str,
            **_kwargs: object,
        ) -> str:
            parsed = json.loads(payload) if destination.endswith(".json") else None
            uploads.append((destination, parsed))
            self.objects[destination] = bytes(payload)
            return '"etag"'

        def upload_file(self, source: str, destination: str) -> str:
            payload = (
                json.loads(Path(source).read_text())
                if source.endswith(".json")
                else None
            )
            uploads.append((destination, payload))
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            uploads.append((destination, None))
            return destination

    storage = Storage()
    regen.publish_regen_outputs(
        _config(),
        tmp_path,
        rrd_path=rrd,
        client=storage,
        snapshots=regen._capture_regen_publication_snapshots(_config(), storage),
    )

    canonical_report_uri = f"{ROOT}/reports/sim2real-report.json"
    canonical_payload = next(
        payload for uri, payload in uploads if uri == canonical_report_uri
    )
    publication = canonical_payload["publication"]
    assert "/reports/generations/" in publication["rrd_uri"]
    assert any(uri == publication["rrd_uri"] for uri, _payload in uploads)
    assert next(
        index
        for index, (uri, _payload) in enumerate(uploads)
        if uri == canonical_report_uri
    ) > next(
        index
        for index, (uri, _payload) in enumerate(uploads)
        if uri == f"{ROOT}/reports/sim2real.rrd"
    )
