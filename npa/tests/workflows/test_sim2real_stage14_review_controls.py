from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from npa.clients.storage import StorageError
from npa.workflows.sim2real import workflow_stage
import npa.workflows.sim2real.engine as engine
import npa.workflows.sim2real.legacy_orchestration as legacy
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.viz_contract import visualization_run_metadata
import npa.workflows.sim2real_rerun_regen as regen
from npa.workflows.sim2real_rerun_regen import Sim2RealRerunRegenError
import npa.workflows.sim2real_viz as viz


CHECKPOINT_BYTES = b"policy-checkpoint-bytes"
CHECKPOINT_SHA256 = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()


def _config() -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id="run-a",
        s3_bucket="demo-bucket",
        s3_prefix="sim2real-b",
        s3_endpoint="https://storage.example",
    )


def _same_run_uri() -> str:
    return (
        "s3://demo-bucket/sim2real-b/run-a/byo-trainer/job/"
        "outer-01-iter-01/model_latest.pt"
    )


def _identity(uri: str) -> dict[str, Any]:
    return {
        "checkpoint_uri": uri,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_size_bytes": len(CHECKPOINT_BYTES),
        "generator_policy_sha256": CHECKPOINT_SHA256,
    }


def _inner_evidence(uri: str) -> dict[str, Any]:
    identity = _identity(uri)
    return {
        "outer_iteration": 1,
        "selected_checkpoint_uri": uri,
        "final_checkpoint_uri": uri,
        "checkpoint_selection": dict(identity),
        "checkpoint_candidates": [dict(identity)],
    }


def _candidate(uri: str, *, run_id: str = "run-a") -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.candidate_checkpoint.v1",
        "run_id": run_id,
        "deployable_policy": True,
        "policy_bytes_available": True,
        "policy_checkpoint_uri": uri,
        "policy_checkpoint_identity": Path(uri).name,
        "policy_checkpoint_sha256": CHECKPOINT_SHA256,
        "generator_policy_sha256": CHECKPOINT_SHA256,
        "policy_checkpoint_size_bytes": len(CHECKPOINT_BYTES),
    }


def _heldout_report(
    uri: str,
    *,
    loaded: bool,
    learned_actor_only: bool,
    deployable_policy_eval: bool = True,
) -> dict[str, Any]:
    semantics = (
        {
            "stock_or_scripted_policy": False,
            "actor_is_learned": True,
            "scripted_post_actor_controller": False,
            "policy_composition": "learned_actor_only",
            "post_actor_controller": None,
        }
        if learned_actor_only
        else {
            "stock_or_scripted_policy": True,
            "actor_is_learned": False,
            "scripted_post_actor_controller": True,
            "policy_composition": "scripted_policy",
            "post_actor_controller": "oracle",
        }
    )
    return {
        "evaluation_split": "gold_heldout",
        "deployable_policy_eval": deployable_policy_eval,
        "policy_checkpoint": uri,
        "policy_checkpoint_sha256": CHECKPOINT_SHA256,
        "policy_checkpoint_size_bytes": len(CHECKPOINT_BYTES),
        "policy_inference_provenance": {
            **_identity(uri),
            "loaded_for_inference": loaded,
            **semantics,
        },
    }


def _write_candidate(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "checkpoints/candidate/candidate.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _CheckpointStorage:
    def __init__(self) -> None:
        self.downloads: list[str] = []

    def download_file(self, uri: str, destination: str) -> None:
        self.downloads.append(uri)
        Path(destination).write_bytes(CHECKPOINT_BYTES)


@dataclass
class _FakeVizResult:
    output_rrd_path: str = ""
    rollout_count: int = 0
    frame_count: int = 0
    heldout_frame_count: int = 1
    heldout_env_count: int = 1
    synthetic_frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "written",
            "output_rrd_path": self.output_rrd_path,
            "heldout_frame_count": self.heldout_frame_count,
        }


def test_current_scripted_policy_cannot_restore_deployable_access(
    tmp_path: Path,
) -> None:
    uri = _same_run_uri()
    candidate_path = _write_candidate(tmp_path, _candidate(uri))
    storage = _CheckpointStorage()

    access = regen._ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=storage,
        report={},
        inner_evidence=_inner_evidence(uri),
        heldout_report=_heldout_report(
            uri,
            loaded=True,
            learned_actor_only=False,
            deployable_policy_eval=False,
        ),
    )

    assert access["deployable_policy"] is False
    assert access["authenticated_download_command"] == ""
    persisted = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert persisted["deployable_policy"] is False
    assert "policy_download_command" not in persisted


def test_compat_metadata_suppresses_access_when_current_heldout_fails() -> None:
    uri = _same_run_uri()
    candidate = {
        **_candidate(uri),
        "policy_download_command": "STALE POSITIVE DOWNLOAD",
        "policy_ui_action": "STALE POSITIVE UI",
    }

    metadata = visualization_run_metadata(
        config=_config(),
        artifact_root="s3://demo-bucket/sim2real-b/run-a",
        policy_checkpoint=uri,
        candidate=candidate,
        heldout_report=_heldout_report(
            uri,
            loaded=False,
            learned_actor_only=False,
            deployable_policy_eval=False,
        ),
    )

    assert metadata["policy_deployable"] is False
    assert metadata["policy_download_command"] == ""
    assert metadata["policy_ui_action"] == ""


def test_compat_metadata_requires_complete_current_identity() -> None:
    uri = _same_run_uri()
    candidate = {
        **_candidate(uri),
        "policy_download_command": "STALE POSITIVE DOWNLOAD",
        "policy_ui_action": "STALE POSITIVE UI",
    }

    metadata = visualization_run_metadata(
        config=_config(),
        artifact_root="s3://demo-bucket/sim2real-b/run-a",
        policy_checkpoint=uri,
        candidate=candidate,
        heldout_report={
            "deployable_policy_eval": True,
            "policy_inference_provenance": {"loaded_for_inference": True},
        },
    )

    assert metadata["heldout_policy_identity_verified"] is False
    assert metadata["policy_deployable"] is False
    assert metadata["policy_download_command"] == ""
    assert metadata["policy_ui_action"] == ""


def test_legacy_viz_rejects_duplicate_candidate_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uri = _same_run_uri()
    candidate_path = tmp_path / "checkpoints/candidate/candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        (
            '{"deployable_policy":false,'
            '"deployable_policy":true,'
            '"policy_bytes_available":true,'
            f'"policy_checkpoint_uri":"{uri}",'
            '"policy_checkpoint_identity":"model_latest.pt",'
            f'"policy_checkpoint_sha256":"{CHECKPOINT_SHA256}",'
            f'"policy_checkpoint_size_bytes":{len(CHECKPOINT_BYTES)},'
            '"policy_download_command":"DUPLICATE-WINS"}'
        ),
        encoding="utf-8",
    )
    heldout = _heldout_report(uri, loaded=True, learned_actor_only=True)
    captured: dict[str, Any] = {}
    monkeypatch.delenv("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_REQUIRE_VISUALIZATION", raising=False)
    monkeypatch.setattr(
        legacy,
        "_ensure_heldout_renders_for_viz",
        lambda *_args, **_kwargs: heldout,
    )

    def fake_emit(**kwargs: Any) -> _FakeVizResult:
        captured.update(kwargs["run_metadata"])
        return _FakeVizResult(output_rrd_path=str(kwargs["output_rrd"]))

    monkeypatch.setattr(viz, "emit_sim2real_rerun", fake_emit)
    monkeypatch.setattr(
        viz,
        "emit_sim2real_mcap_if_enabled",
        lambda **_kwargs: {"status": "skipped"},
    )

    with pytest.raises(ValueError, match="duplicate"):
        legacy._run_sim2real_viz_stage(
            _config(),
            local_dir=tmp_path,
            inner_evidence={},
            heldout_report=heldout,
            final_decision={"checkpoint_uri": uri},
        )
    assert captured == {}


def test_foreign_run_checkpoint_is_rejected_before_download(
    tmp_path: Path,
) -> None:
    uri = (
        "s3://foreign-bucket/foreign-prefix/other-run/byo-trainer/job/"
        "outer-09-iter-01/model_latest.pt"
    )
    _write_candidate(tmp_path, _candidate(uri))
    storage = _CheckpointStorage()

    with pytest.raises(Sim2RealRerunRegenError):
        regen._ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=storage,
            report={},
            inner_evidence=_inner_evidence(uri),
            heldout_report=_heldout_report(
                uri,
                loaded=True,
                learned_actor_only=True,
            ),
        )
    assert storage.downloads == []


def test_mismatched_retained_run_id_is_rejected_before_download(
    tmp_path: Path,
) -> None:
    uri = _same_run_uri()
    _write_candidate(tmp_path, _candidate(uri, run_id="other-run"))
    storage = _CheckpointStorage()

    with pytest.raises(Sim2RealRerunRegenError, match="different run_id"):
        regen._ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=storage,
            report={},
            inner_evidence=_inner_evidence(uri),
            heldout_report=_heldout_report(
                uri,
                loaded=True,
                learned_actor_only=True,
            ),
        )
    assert storage.downloads == []


def test_heldout_only_uses_requested_outer_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Storage:
        def __init__(self) -> None:
            self.requested: list[str] = []

        def download_directory(self, _uri: str, destination: str) -> None:
            Path(destination).mkdir(parents=True, exist_ok=True)

        def download_path(self, uri: str, destination: str) -> None:
            self.requested.append(uri)
            if "/outer-02/" not in uri:
                raise StorageError("only requested outer-02 exists")
            evidence = {
                **_inner_evidence(_same_run_uri()),
                "outer_iteration": 2,
            }
            Path(destination).write_text(json.dumps(evidence), encoding="utf-8")

    storage = Storage()

    def fake_eval(
        _config: Sim2RealLoopConfig,
        *,
        inner_evidence: dict[str, Any],
        outer_iteration: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert inner_evidence["outer_iteration"] == 2
        return {"outer_iteration": outer_iteration}

    monkeypatch.setattr(engine, "run_heldout_eval", fake_eval)
    monkeypatch.setattr(
        regen,
        "_sync_heldout_eval_renders",
        lambda *_args, **_kwargs: None,
    )

    report = regen.rerun_heldout_eval_only(
        _config(),
        local_dir=tmp_path,
        outer_iteration=2,
        publish=False,
        client=storage,
    )

    assert report["outer_iteration"] == 2
    assert storage.requested == [
        "s3://demo-bucket/sim2real-b/run-a/inner_loop/outer-02/evidence.json"
    ]


def test_inner_evidence_rewrite_does_not_modify_symlink_target(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    evidence = run / "inner_loop/outer-01/evidence.json"
    evidence.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    original = {
        "iterations": [{"actions_dir": "/old/location/actions/outer-01"}],
        "external_setting": "must-survive",
    }
    outside.write_text(json.dumps(original), encoding="utf-8")
    evidence.symlink_to(outside)

    with pytest.raises(Sim2RealRerunRegenError):
        regen._rewrite_inner_evidence_paths(run, evidence)
    assert json.loads(outside.read_text(encoding="utf-8")) == original


def test_publish_regen_outputs_rejects_symlinked_tree_before_upload(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    renders = run / "eval/heldout/renders"
    camera = renders / "env-0001/camera-000.png"
    camera.parent.mkdir(parents=True)
    camera.write_bytes(b"camera")
    outside = tmp_path / "outside-secret"
    outside.write_bytes(b"HOST-SECRET")
    (renders / "unconsumed-proof.bin").symlink_to(outside)
    recording = run / "reports/sim2real.rrd"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"rrd")

    class Storage:
        def __init__(self) -> None:
            self.uploads: list[str] = []

        def upload_file(self, _source: str, destination: str) -> str:
            self.uploads.append(destination)
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            self.uploads.append(destination)
            return destination

    storage = Storage()
    with pytest.raises(Sim2RealRerunRegenError, match="symlink"):
        regen.publish_regen_outputs(_config(), run, client=storage)
    assert storage.uploads == []


def test_default_rrd_download_rejects_symlinked_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    default_root = tmp_path / "default-root"
    default_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(regen, "DEFAULT_REGEN_ROOT", default_root)

    class Storage:
        def download_path(self, _uri: str, destination: str) -> None:
            Path(destination).write_bytes(b"rrd-bytes")

    destination = regen.resolve_local_rrd_path("run-a")
    with pytest.raises(Sim2RealRerunRegenError):
        regen.download_rrd_from_s3(
            _config(),
            dest_path=destination,
            client=Storage(),
        )
    assert not (outside / "run-a/reports/sim2real.rrd").exists()


def test_explicit_rrd_output_is_the_recording_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    canonical = work / "reports/sim2real.rrd"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"STALE-CANONICAL")
    explicit = tmp_path / "fresh-explicit.rrd"
    state = regen._RegenState(
        inner_evidence={},
        heldout_report={},
        report_path=work / "reports/sim2real-report.json",
        report={},
        policy_access={},
    )

    class Storage:
        def __init__(self) -> None:
            self.uploads: dict[str, bytes] = {}

        def upload_file(self, source: str, destination: str) -> str:
            self.uploads[destination] = Path(source).read_bytes()
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            return destination

    storage = Storage()
    monkeypatch.setattr(regen, "_load_regen_state", lambda *_args, **_kwargs: state)

    def fake_emit(
        _config: Sim2RealLoopConfig,
        _work: Path,
        output_rrd: Path,
        _state: Any,
        _history: list[dict[str, Any]],
        _viewer: str,
    ) -> tuple[_FakeVizResult, float]:
        output_rrd.parent.mkdir(parents=True, exist_ok=True)
        output_rrd.write_bytes(b"FRESH-EXPLICIT")
        return _FakeVizResult(output_rrd_path=str(output_rrd)), 0.01

    monkeypatch.setattr(regen, "_emit_regen_rrd", fake_emit)
    monkeypatch.setattr(regen, "_persist_regen_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        regen,
        "emit_sim2real_mcap_if_enabled",
        lambda **_kwargs: {"status": "skipped"},
    )

    result = regen.regen_sim2real_rrd(
        _config(),
        local_dir=work,
        local_rrd_path=explicit,
        upload=True,
        sync_inputs=False,
        client=storage,
    )

    destination = "s3://demo-bucket/sim2real-b/run-a/reports/sim2real.rrd"
    assert result.local_rrd_path == str(explicit)
    assert storage.uploads[destination] == b"FRESH-EXPLICIT"


def test_heldout_only_does_not_publish_stale_positive_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_candidate(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_checkpoint_uri": ("s3://demo-bucket/sim2real-b/run-a/old.pt"),
            "policy_download_command": "STALE-POSITIVE",
        },
    )

    class Storage:
        def __init__(self) -> None:
            self.uploaded_candidate: dict[str, Any] | None = None

        def upload_file(self, source: str, destination: str) -> str:
            if destination.endswith("/checkpoints/candidate/candidate.json"):
                self.uploaded_candidate = json.loads(
                    Path(source).read_text(encoding="utf-8")
                )
            return destination

        def upload_directory(self, _source: str, destination: str) -> str:
            return destination

    storage = Storage()
    monkeypatch.setattr(
        regen,
        "_sync_heldout_eval_inputs",
        lambda *_args, **_kwargs: _inner_evidence(_same_run_uri()),
    )
    monkeypatch.setattr(
        regen,
        "_sync_heldout_eval_renders",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine,
        "run_heldout_eval",
        lambda *_args, **_kwargs: {
            "deployable_policy_eval": False,
            "policy_inference_provenance": {"loaded_for_inference": False},
        },
    )

    with pytest.raises(Sim2RealRerunRegenError, match="refuses to publish"):
        regen.rerun_heldout_eval_only(
            _config(),
            local_dir=tmp_path,
            publish=True,
            client=storage,
        )

    assert storage.uploaded_candidate is None


def test_stage10_rejects_foreign_render_source_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = "s3://unit/current-run"
    uri = f"{root}/checkpoints/model.pt"
    identity = {
        "checkpoint_uri": uri,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": 128,
        "generator_policy_sha256": "a" * 64,
    }
    evidence = {
        "outer_iteration": 1,
        "iterations": [],
        "selected_checkpoint_uri": uri,
        "final_checkpoint_uri": uri,
        "checkpoint_candidates": [identity],
        "checkpoint_selection": dict(identity),
    }
    report = {
        "component_invocation": {"mode": "npa_workflow_skypilot_task"},
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 128,
        "policy_inference_provenance": {
            **identity,
            "loaded_for_inference": True,
            "stock_or_scripted_policy": False,
            "actor_is_learned": True,
            "scripted_post_actor_controller": False,
            "policy_composition": "learned_actor_only",
            "post_actor_controller": None,
        },
        "render_manifest": {
            "renders_s3_uri": "s3://other-bucket/other-run/outer-99/renders/",
            "episodes": [{"env_id": "e", "frames": ["camera-000.png"]}],
        },
    }
    downloaded: list[str] = []

    class Storage:
        def download_directory(self, source: str, destination: str) -> None:
            downloaded.append(source)
            frame = Path(destination) / "e/camera-000.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"png")

        def upload_directory(self, _local: str, destination: str) -> str:
            return destination

    monkeypatch.setattr(workflow_stage, "_root", lambda _args: root)
    monkeypatch.setattr(workflow_stage, "_work", lambda _stage: tmp_path)
    monkeypatch.setattr(workflow_stage, "read_json", lambda *_args, **_kwargs: evidence)
    monkeypatch.setattr(workflow_stage, "_run_eval", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(
        workflow_stage,
        "_assert_embodiment_evidence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(workflow_stage, "storage", lambda: Storage())

    with pytest.raises(RuntimeError, match="exact current heldout-eval"):
        workflow_stage._stage10(Namespace(outer_iteration=1, gold_count=1))
    assert downloaded == []


def test_remote_selected_pair_cannot_be_displaced_by_stale_local_pair(
    tmp_path: Path,
) -> None:
    class Pager:
        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            prefix = kwargs["Prefix"]
            if prefix.endswith(("inner_loop/", "eval/gold-heldout/")):
                return [{"CommonPrefixes": [{"Prefix": prefix + "outer-01/"}]}]
            return []

    class S3:
        def get_paginator(self, _name: str) -> Pager:
            return Pager()

        def head_object(self, **_kwargs: Any) -> dict[str, int]:
            return {"ContentLength": 2}

    class Storage:
        _s3 = S3()

        def download_path(self, uri: str, path: str) -> str:
            if uri.endswith(
                (
                    "inner_loop/outer-01/evidence.json",
                    "eval/gold-heldout/outer-01/report.json",
                )
            ):
                Path(path).write_text(
                    json.dumps({"outer_iteration": 1}),
                    encoding="utf-8",
                )
                return path
            raise StorageError("missing")

        def download_directory(self, _uri: str, _path: str) -> None:
            raise StorageError("missing")

    for relative in (
        "inner_loop/outer-02/evidence.json",
        "eval/gold-heldout/outer-02/report.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_text('{"stale_outer":2}', encoding="utf-8")

    state = regen._load_regen_state(
        _config(),
        tmp_path,
        Storage(),
        sync_inputs=True,
    )

    assert "stale_outer" not in state.inner_evidence
    assert "stale_outer" not in state.heldout_report


def test_failed_directory_sync_cannot_preserve_stale_local_evidence(
    tmp_path: Path,
) -> None:
    class Pager:
        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            prefix = kwargs["Prefix"]
            if prefix.endswith(("inner_loop/", "eval/gold-heldout/")):
                return [{"CommonPrefixes": [{"Prefix": prefix + "outer-01/"}]}]
            return []

    class S3:
        def get_paginator(self, _name: str) -> Pager:
            return Pager()

        def head_object(self, *, Key: str, **_kwargs: Any) -> dict[str, int]:
            return {
                "ContentLength": (
                    2 if Key.endswith(("evidence.json", "report.json")) else 0
                )
            }

    class Storage:
        _s3 = S3()

        def download_path(self, uri: str, path: str) -> str:
            if uri.endswith(
                (
                    "inner_loop/outer-01/evidence.json",
                    "eval/gold-heldout/outer-01/report.json",
                )
            ):
                Path(path).write_text("{}", encoding="utf-8")
                return path
            raise StorageError("missing")

        def download_directory(self, _uri: str, _path: str) -> None:
            raise StorageError("transient denied")

    stale = tmp_path / "actions/train/outer-01/iter-01/rollout.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("STALE", encoding="utf-8")

    with pytest.raises(Sim2RealRerunRegenError):
        regen.sync_regen_inputs(_config(), tmp_path, client=Storage())
    assert not stale.exists()


def test_publication_occurs_only_after_heldout_frame_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        regen,
        "emit_sim2real_mcap_if_enabled",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        regen,
        "_publish_regen_recordings",
        lambda *_args, **_kwargs: calls.append("published") or ("s3://bad.rrd", ""),
    )

    with pytest.raises(Sim2RealRerunRegenError):
        regen._finalize_regen_result(
            Sim2RealLoopConfig(run_id="r"),
            tmp_path / "unused",
            object(),
            regen._RegenState(
                inner_evidence={},
                heldout_report={},
                report_path=tmp_path / "report",
                report={},
                policy_access={},
            ),
            _FakeVizResult(heldout_frame_count=0),
            upload=True,
        )
    assert calls == []


def test_traversing_run_id_cannot_escape_default_rrd_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(regen, "DEFAULT_REGEN_ROOT", tmp_path / "regen-root")

    with pytest.raises(Sim2RealRerunRegenError):
        regen.resolve_local_rrd_path("../escaped")


def test_canonical_iteration_uris_are_localized(tmp_path: Path) -> None:
    evidence_path = tmp_path / "inner_loop/outer-01/evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "iterations": [
                    {
                        "iteration": 1,
                        "actions_uri": (
                            "s3://bucket/r/actions/train/outer-01/iter-01/"
                        ),
                        "vlm_eval_uri": (
                            "s3://bucket/r/vlm_eval/train/outer-01/iter-01/evaluations/"
                        ),
                        "signal_uri": (
                            "s3://bucket/r/vlm_eval/train/outer-01/iter-01/signals/"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    regen._rewrite_inner_evidence_paths(
        tmp_path,
        evidence_path,
        expected_artifact_root="s3://bucket/r",
    )

    record = json.loads(evidence_path.read_text(encoding="utf-8"))["iterations"][0]
    for key in ("actions_dir", "vlm_eval_dir", "signal_dir"):
        value = record.get(key)
        assert isinstance(value, str) and value
        assert Path(value).is_relative_to(tmp_path)
