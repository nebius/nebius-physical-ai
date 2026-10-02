from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from dataclasses import replace
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import component_authority
from npa.workflows.sim2real.hashing import sha256_file
from npa.workflows.sim2real.publication import PublicationConflict
from npa.workflows.sim2real.stage10_execution import (
    _assert_exact_render_frames,
    validate_materialized_render_tree,
)
import npa.workflows.sim2real.stage14_finalize as stage14
import npa.workflows.sim2real_rerun_regen as regen
from tests.workflows.test_sim2real_stage14_thirteenth_review_controls import (
    ATTEMPT,
    CHECKPOINT,
    ROOT,
    RUN_ID,
    SOURCE_SHA,
    _component,
    _config,
    _decision,
    _evidence,
    _gold,
    _identity,
    _rehash,
    _stage14_component,
)


def _replace_checkpoint(payload: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return json.loads(json.dumps(payload).replace(CHECKPOINT, checkpoint))


@pytest.mark.parametrize("field", ["schema", "run_id"])
def test_stage14_requires_canonical_inner_evidence_identity(field: str) -> None:
    evidence = _evidence()
    evidence.pop(field)
    gold = _gold()
    decision = _decision(gold)
    with pytest.raises(RuntimeError):
        stage14._stage14_policy_metadata(
            evidence,
            decision,
            gold,
            run_root=ROOT,
            run_id=RUN_ID,
            expected_threshold=0.5,
            expected_early_exit=False,
        )


def test_stage14_rejects_same_run_nontrainer_checkpoint() -> None:
    checkpoint = f"{ROOT}/imports/arbitrary.pt"
    evidence = _replace_checkpoint(_evidence(), checkpoint)
    gold = _replace_checkpoint(_gold(), checkpoint)
    decision = _replace_checkpoint(_decision(gold), checkpoint)
    decision["gold_report_sha256"] = component_authority.gold_report_sha256(gold)
    with pytest.raises(RuntimeError, match="trainer"):
        stage14._stage14_policy_metadata(
            evidence,
            decision,
            gold,
            run_root=ROOT,
            run_id=RUN_ID,
            expected_threshold=0.5,
            expected_early_exit=False,
        )


def test_canonical_inner_evidence_prevents_coordinated_report_downgrade(
    tmp_path: Path,
) -> None:
    report = {
        "components": [_component(stage) for stage in range(1, 14)],
        "source_sha": SOURCE_SHA,
    }
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
    with pytest.raises(regen.Sim2RealRerunRegenError, match="architecture"):
        regen._validate_regen_component_authority(
            _config(),
            tmp_path,
            object(),
            inputs,
            verify_remote_authority=False,
        )


def _report_authority_digest(report: dict[str, Any]) -> str:
    material = json.loads(json.dumps(report))
    for field in (
        "recording_summaries",
        "visualization",
        "policy_access",
        "progress_metrics",
        "gpu_fallback_contract",
    ):
        material.pop(field, None)
    records = material.get("component_records") or material.get("components") or []
    if records:
        stage14_record = records[-1]
        stage14_record.pop("content_sha256", None)
        artifacts = stage14_record.get("artifacts") or {}
        artifacts.pop("report_authority_sha256", None)
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_stage14_component_binds_final_report_claims() -> None:
    rrd_uri = f"{ROOT}/reports/generations/generation/sim2real.rrd"
    mcap_uri = f"{ROOT}/reports/generations/generation/sim2real.mcap"
    report_uri = f"{ROOT}/reports/generations/generation/sim2real-report.json"
    record = _stage14_component(
        rrd_uri=rrd_uri,
        mcap_uri=mcap_uri,
        report_uri=report_uri,
    )
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "run_id": RUN_ID,
        "source_sha": SOURCE_SHA,
        "architecture": "npa.workflow/v0.0.1_compositional_standard_runtime",
        "strict_gold_success_rate": 0.2,
        "outer_loop": {"latest_heldout_report": _gold()},
        "rrd_uri": rrd_uri,
        "mcap_uri": mcap_uri,
        "report_uri": report_uri,
        "component_records": [record],
    }
    record["artifacts"]["report_authority_sha256"] = _report_authority_digest(report)
    _rehash(record)
    report["strict_gold_success_rate"] = 0.999
    report["outer_loop"]["latest_heldout_report"]["success_rate"] = 0.999
    with pytest.raises(ValueError, match="report authority"):
        component_authority.validate_stage14_component_record(
            record,
            report,
            expected_source_sha=SOURCE_SHA,
        )


def test_parallel_join_rejects_forged_nested_lane_authority() -> None:
    components = [_component(stage) for stage in range(1, 14)]
    stage4_record = components[3]
    stage4_record["artifacts"]["shard_provenance"] = [{}, {}]
    stage4_record["artifacts"]["lane_records"] = [{}, {}]
    _rehash(stage4_record)
    with pytest.raises(ValueError, match="Stage 4"):
        component_authority.validate_component_records(
            components,
            root=ROOT,
            evidence=_evidence(),
            gold=_gold(),
            expected_source_sha=SOURCE_SHA,
        )


@pytest.mark.parametrize("replacement", [b"", b"\x89PNG\r\n\x1a\nreplaced"])
def test_render_manifest_binds_nonempty_frame_bytes(
    tmp_path: Path,
    replacement: bytes,
) -> None:
    render_root = tmp_path / "renders"
    frame = render_root / "env-1" / "camera-000.png"
    frame.parent.mkdir(parents=True)
    original = b"\x89PNG\r\n\x1a\noriginal"
    frame.write_bytes(replacement)
    manifest = {
        "schema": "npa.sim2real.heldout_renders.v2",
        "episodes": [{"env_id": "env-1", "frames": [frame.name]}],
        "policy_checkpoint": {
            "uri": CHECKPOINT,
            "sha256": _identity()["checkpoint_sha256"],
            "size_bytes": _identity()["checkpoint_size_bytes"],
        },
        "frame_artifacts": {
            "env-1/camera-000.png": {
                "sha256": hashlib.sha256(original).hexdigest(),
                "size_bytes": len(original),
            }
        },
    }
    with pytest.raises(RuntimeError, match="render"):
        validate_materialized_render_tree(
            render_root,
            manifest,
            {
                "checkpoint_uri": CHECKPOINT,
                "checkpoint_sha256": _identity()["checkpoint_sha256"],
                "checkpoint_size_bytes": _identity()["checkpoint_size_bytes"],
            },
        )


def test_render_tree_rejects_undeclared_or_missing_frames(tmp_path: Path) -> None:
    render_root = tmp_path / "renders"
    frame = render_root / "other-env" / "camera-000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"\x89PNG\r\n\x1a\nframe")
    with pytest.raises(RuntimeError, match="exactly match"):
        _assert_exact_render_frames(
            render_root,
            {"env-1/camera-000.png"},
        )


def _heldout_publication_tree(tmp_path: Path) -> dict[str, Any]:
    report = _gold()
    render_root = tmp_path / "eval" / "gold-heldout" / "outer-01" / "renders" / "env-1"
    render_root.mkdir(parents=True)
    render_root.joinpath("camera-000.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
    return report


def test_heldout_only_publishes_attempt_without_replacing_canonical_stage10(
    tmp_path: Path,
) -> None:
    report = _heldout_publication_tree(tmp_path)
    destinations: list[str] = []

    class Storage:
        def read_bytes_with_etag(self, _destination: str) -> None:
            return None

        def put_bytes_conditional(
            self,
            _payload: bytes,
            destination: str,
            **_kwargs: Any,
        ) -> str:
            destinations.append(destination)
            return '"etag"'

    regen._publish_heldout_eval_outputs(
        _config(),
        tmp_path,
        report,
        outer_iteration=1,
        storage=Storage(),
    )
    canonical_report = f"{ROOT}/eval/gold-heldout/outer-01/report.json"
    assert canonical_report not in destinations
    assert (
        f"{ROOT}/eval/gold-heldout/outer-01/attempts/{ATTEMPT}/report.json"
        in destinations
    )


def test_heldout_render_attempt_refuses_existing_different_bytes(
    tmp_path: Path,
) -> None:
    report = _heldout_publication_tree(tmp_path)

    class Storage:
        def read_bytes_with_etag(self, destination: str) -> tuple[bytes, str] | None:
            if destination.endswith("/renders/env-1/camera-000.png"):
                return b"different", '"etag"'
            return None

        def put_bytes_conditional(
            self,
            _payload: bytes,
            _destination: str,
            **_kwargs: Any,
        ) -> str:
            return '"etag"'

    with pytest.raises(PublicationConflict, match="immutable publication"):
        regen._publish_heldout_eval_outputs(
            _config(),
            tmp_path,
            report,
            outer_iteration=1,
            storage=Storage(),
        )


def test_regeneration_uses_sealed_policy_gate_not_ambient_config(
    tmp_path: Path,
) -> None:
    gold = _gold()
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(gold, sort_keys=True))
    decision = _decision(gold)
    decision["gold_report_sha256"] = sha256_file(heldout_path)
    inputs = regen._RegenInputs(
        inner_evidence=_evidence(),
        heldout_report=gold,
        heldout_path=heldout_path,
        report_path=tmp_path / "report.json",
        report={
            "outer_loop": {"decision": json.loads(json.dumps(decision))},
            "policy_gate_config": {"threshold": 0.5, "early_exit": False},
        },
        current_decision=decision,
    )
    config = replace(_config(), threshold=0.9, early_exit=True)
    regen._validate_regen_decision(config, inputs)


def test_no_sync_upload_still_requests_remote_component_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def load_state(
        _config_value: Any,
        _work_dir: Path,
        _storage: Any,
        *,
        sync_inputs: bool,
        verify_remote_authority: bool = False,
        require_canonical_authority: bool = False,
        publication_snapshots: dict[str, Any] | None = None,
        verified_snapshot_dir: Path | None = None,
    ) -> Any:
        assert verified_snapshot_dir is not None
        assert verified_snapshot_dir.resolve().is_relative_to(_work_dir.resolve())
        observed.update(
            sync_inputs=sync_inputs,
            verify_remote_authority=verify_remote_authority,
            require_canonical_authority=require_canonical_authority,
            snapshots_captured=publication_snapshots is not None,
        )
        raise Stop

    class Storage:
        def read_bytes_with_etag(self, _uri: str) -> None:
            return None

        def put_bytes_conditional(self, *_args: Any, **_kwargs: Any) -> str:
            return '"etag"'

    monkeypatch.setattr(regen, "_load_regen_state", load_state)
    with pytest.raises(Stop):
        regen._regenerate_sim2real_rrd(
            _config(),
            tmp_path,
            None,
            True,
            False,
            Storage(),
        )
    assert observed == {
        "sync_inputs": False,
        "verify_remote_authority": True,
        "require_canonical_authority": True,
        "snapshots_captured": True,
    }


def _minimal_regen_publication(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    rrd = reports / "sim2real.rrd"
    rrd.write_bytes(b"rrd")
    (reports / "sim2real-report.json").write_text("{}")
    return rrd


def test_regen_does_not_publish_alias_before_immutable_report(
    tmp_path: Path,
) -> None:
    rrd = _minimal_regen_publication(tmp_path)
    destinations: list[str] = []

    class Storage:
        def read_bytes_with_etag(self, _destination: str) -> None:
            return None

        def put_bytes_conditional(
            self,
            _payload: bytes,
            destination: str,
            **_kwargs: Any,
        ) -> str:
            destinations.append(destination)
            if destination.endswith("/generations/generation/sim2real-report.json"):
                raise RuntimeError("immutable report failed")
            return '"etag"'

    storage = Storage()
    with pytest.raises(RuntimeError, match="immutable report failed"):
        regen.publish_regen_outputs(
            _config(),
            tmp_path,
            rrd_path=rrd,
            publication_id="generation",
            client=storage,
            snapshots=regen._capture_regen_publication_snapshots(_config(), storage),
        )
    assert f"{ROOT}/reports/sim2real.rrd" not in destinations
    assert f"{ROOT}/reports/sim2real-report.json" not in destinations


def test_regen_uses_conditional_canonical_report_as_publication_fence(
    tmp_path: Path,
) -> None:
    rrd = _minimal_regen_publication(tmp_path)

    class Storage:
        def __init__(self) -> None:
            self.unguarded: list[str] = []
            self.conditional: list[str] = []
            self.objects: dict[str, bytes] = {}

        def upload_file(self, _source: str, destination: str) -> str:
            self.unguarded.append(destination)
            return destination

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
            **_kwargs: Any,
        ) -> str:
            self.conditional.append(destination)
            self.objects[destination] = bytes(payload)
            return '"etag"'

    storage = Storage()
    regen.publish_regen_outputs(
        _config(),
        tmp_path,
        rrd_path=rrd,
        publication_id="generation",
        client=storage,
        snapshots=regen._capture_regen_publication_snapshots(_config(), storage),
    )
    canonical_report = f"{ROOT}/reports/sim2real-report.json"
    assert canonical_report in storage.conditional
    assert canonical_report not in storage.unguarded


def test_stage14_component_pointer_uses_conditional_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = stage14._stage14_state(
        SimpleNamespace(run_id=RUN_ID, outer_iteration=1),
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
    pointer_uri = f"{ROOT}/components/stage_14.json"

    class Storage:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.objects = {pointer_uri: b"old-pointer"}

        def read_bytes_with_etag(self, destination: str) -> tuple[bytes, str] | None:
            payload = self.objects.get(destination)
            if payload is None:
                return None
            etag = '"pointer-etag"' if destination == pointer_uri else '"etag"'
            return payload, etag

        def put_bytes_conditional(
            self,
            payload: bytes,
            destination: str,
            **kwargs: Any,
        ) -> str:
            self.calls.append((destination, kwargs))
            self.objects[destination] = bytes(payload)
            return '"etag"'

    storage = Storage()
    monkeypatch.setattr(stage14, "storage", lambda: storage)
    state = stage14._capture_stage14_publication_snapshots(state)
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

    stage14._publish_stage14_outputs(state, report, reports, result, result)

    pointer_call = next(kwargs for uri, kwargs in storage.calls if uri == pointer_uri)
    assert pointer_call == {"if_match": '"pointer-etag"'}


def test_disabled_mcap_removes_stale_canonical_object(tmp_path: Path) -> None:
    rrd = _minimal_regen_publication(tmp_path)

    class Storage:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.objects = {f"{ROOT}/reports/sim2real.mcap": b"stale"}

        def read_bytes_with_etag(self, destination: str) -> tuple[bytes, str] | None:
            payload = self.objects.get(destination)
            if payload is None:
                return None
            etag = (
                '"stale-etag"'
                if destination == f"{ROOT}/reports/sim2real.mcap"
                else '"etag"'
            )
            return payload, etag

        def put_bytes_conditional(
            self,
            payload: bytes,
            destination: str,
            **_kwargs: Any,
        ) -> str:
            self.objects[destination] = bytes(payload)
            return '"etag"'

        def delete_file_conditional(
            self,
            destination: str,
            *,
            if_match: str,
        ) -> None:
            assert if_match == '"stale-etag"'
            self.deleted.append(destination)
            self.objects.pop(destination, None)

    storage = Storage()
    regen.publish_regen_outputs(
        _config(),
        tmp_path,
        rrd_path=rrd,
        publication_id="generation",
        mcap_uri="",
        client=storage,
        snapshots=regen._capture_regen_publication_snapshots(_config(), storage),
    )
    assert f"{ROOT}/reports/sim2real.mcap" in storage.deleted
