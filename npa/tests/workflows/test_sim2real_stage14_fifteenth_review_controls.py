from __future__ import annotations

import base64
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.sim2real import component_authority
from npa.workflows.sim2real.publication import (
    PublicationConflict,
    RemoteObjectSnapshot,
    replace_mutable_file,
)
import npa.workflows.sim2real.publication as publication
import npa.workflows.sim2real.stage14_finalize as stage14
import npa.workflows.sim2real_rerun_regen as regen
from npa.workflows.sim2real.stage10_execution import (
    validate_materialized_render_tree,
)
from npa.workflows.sim2real.workflow_io import component_record_history_uri
from tests.workflows.test_sim2real_stage14_thirteenth_review_controls import (
    CHECKPOINT,
    DIGEST,
    ROOT,
    RUN_ID,
    SOURCE_SHA,
    _component,
    _config,
    _decision,
    _evidence,
    _gold,
    _rehash,
    _stage14_component,
)


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _materialized_state(tmp_path: Path) -> stage14._Stage14State:
    gold = _gold()
    gold["render_manifest"]["frame_artifacts"]["env-1/camera-000.png"] = {
        "sha256": hashlib.sha256(VALID_PNG).hexdigest(),
        "size_bytes": len(VALID_PNG),
    }
    local = tmp_path / "run"
    frame = local / "eval/gold-heldout/outer-01/renders/env-1/camera-000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(VALID_PNG + b"changed")
    return stage14._stage14_state(
        Namespace(
            run_id=RUN_ID,
            outer_iteration=1,
            threshold=0.5,
            allow_early_exit=False,
        ),
        ROOT,
        tmp_path,
        local,
        _evidence(),
        gold,
    )


def test_stage14_revalidates_downloaded_frame_bytes_before_encoding(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="frame bytes"):
        stage14._validate_stage14_materialized_frames(_materialized_state(tmp_path))


def test_regen_revalidates_local_frame_bytes_before_encoding(tmp_path: Path) -> None:
    state = _materialized_state(tmp_path)
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(state.gold), encoding="utf-8")
    inputs = regen._RegenInputs(
        inner_evidence=state.evidence,
        heldout_report=state.gold,
        heldout_path=heldout_path,
        report_path=state.local / "reports/sim2real-report.json",
        report={},
        current_decision=_decision(state.gold),
    )
    with pytest.raises(regen.Sim2RealRerunRegenError, match="frame bytes"):
        regen._validate_regen_materialized_frames(_config(), state.local, inputs)


def test_uploading_no_sync_regen_cannot_downgrade_to_legacy_authority(
    tmp_path: Path,
) -> None:
    inputs = regen._RegenInputs(
        inner_evidence={"iterations": []},
        heldout_report={},
        heldout_path=tmp_path / "heldout.json",
        report_path=tmp_path / "report.json",
        report={},
        current_decision={},
    )
    with pytest.raises(
        regen.Sim2RealRerunRegenError,
        match="canonical|remote",
    ):
        regen._validate_regen_component_authority(
            _config(),
            tmp_path,
            object(),
            inputs,
            verify_remote_authority=True,
            require_canonical_authority=True,
        )


def _coherently_forge_embedded_stage4(
    component: dict[str, Any],
) -> dict[str, Any]:
    forged = json.loads(json.dumps(component))
    proof = forged["artifacts"]["shard_provenance"][0]
    lane = forged["artifacts"]["lane_records"][0]
    proof["provenance"]["workflow_job"] = "forged-lane-0"
    lane["artifacts"]["workflow_job"] = "forged-lane-0"
    _rehash(lane)
    joined = component_authority.aggregate_parallel_provenance(
        [item["provenance"] for item in forged["artifacts"]["shard_provenance"]],
        stage=4,
    )
    forged["artifacts"].update(joined)
    return _rehash(forged)


def _stage4_remote_objects(component: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for index, (proof, lane) in enumerate(
        zip(
            component["artifacts"]["shard_provenance"],
            component["artifacts"]["lane_records"],
            strict=True,
        )
    ):
        lane_name = f"shard-{index:05d}"
        lane_root = f"{ROOT}/components/lanes/stage_04/{lane_name}"
        objects[f"{ROOT}/envs/raw/provenance-{index:05d}.json"] = proof
        objects[f"{lane_root}.json"] = lane
        objects[f"{lane_root}/history/{lane['content_sha256']}.json"] = lane
    return objects


def test_stage4_embedded_authority_must_match_remote_proof_pointer_and_history() -> (
    None
):
    remote_component = _component(4)
    forged_component = _coherently_forge_embedded_stage4(remote_component)
    components = [_component(stage) for stage in range(1, 14)]
    components[3] = forged_component
    component_authority.validate_component_records(
        components,
        root=ROOT,
        evidence=_evidence(),
        gold=_gold(),
        expected_source_sha=SOURCE_SHA,
    )
    remote = _stage4_remote_objects(remote_component)
    with pytest.raises(ValueError, match="remote|pointer|history|proof"):
        component_authority.validate_remote_stage4_authority(
            ROOT,
            forged_component,
            lambda uri: remote[uri],
        )


def _forged_components_and_remote_stage4() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    remote_component = _component(4)
    components = [_component(stage) for stage in range(1, 14)]
    components[3] = _coherently_forge_embedded_stage4(remote_component)
    return components, _stage4_remote_objects(remote_component)


def test_stage14_loads_remote_stage4_nested_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    components, remote = _forged_components_and_remote_stage4()

    def reader(uri: str, **_kwargs: Any) -> dict[str, Any]:
        if uri in remote:
            return remote[uri]
        stage = int(uri.rsplit("stage_", 1)[1][:2])
        return components[stage - 1]

    state = stage14._Stage14State(
        args=Namespace(run_id=RUN_ID, outer_iteration=1),
        root=ROOT,
        work=tmp_path,
        local=tmp_path / "run",
        evidence=_evidence(),
        gold=_gold(),
        rrd_uri=f"{ROOT}/reports/generations/g/rrd",
        mcap_uri=f"{ROOT}/reports/generations/g/mcap",
        report_uri=f"{ROOT}/reports/generations/g/report",
    )
    monkeypatch.setattr(stage14, "source_sha", lambda: SOURCE_SHA)
    monkeypatch.setattr(stage14, "read_json", reader)

    with pytest.raises(RuntimeError, match="invalid ComponentRecord"):
        stage14._load_component_records(state)


def _canonical_report_with_forged_stage4() -> tuple[
    dict[str, Any], dict[str, dict[str, Any]]
]:
    components, remote = _forged_components_and_remote_stage4()
    report_uri = f"{ROOT}/reports/generations/g/sim2real-report.json"
    rrd_uri = f"{ROOT}/reports/generations/g/sim2real.rrd"
    mcap_uri = f"{ROOT}/reports/generations/g/sim2real.mcap"
    stage14_record = _stage14_component(
        rrd_uri=rrd_uri,
        mcap_uri=mcap_uri,
        report_uri=report_uri,
    )
    components.append(stage14_record)
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "run_id": RUN_ID,
        "source_sha": SOURCE_SHA,
        "architecture": "npa.workflow/v0.0.1_compositional_standard_runtime",
        "rrd_uri": rrd_uri,
        "mcap_uri": mcap_uri,
        "report_uri": report_uri,
        "component_records": components,
    }
    stage14_record["artifacts"]["report_authority_sha256"] = (
        component_authority.stage14_report_authority_sha256(report)
    )
    _rehash(stage14_record)
    return report, remote


def test_regen_loads_remote_stage4_nested_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report, remote = _canonical_report_with_forged_stage4()
    objects = dict(remote)
    for stage, component in enumerate(report["component_records"], start=1):
        objects[f"{ROOT}/components/stage_{stage:02d}.json"] = component
        objects[
            component_record_history_uri(
                ROOT,
                stage,
                component["content_sha256"],
            )
        ] = component

    def download(
        _storage: Any,
        uri: str,
        target: Path,
        **_kwargs: Any,
    ) -> bool:
        payload = objects.get(uri)
        if payload is None:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return True

    monkeypatch.setattr(regen, "_download_if_exists", download)
    inputs = regen._RegenInputs(
        inner_evidence=_evidence(),
        heldout_report=_gold(),
        heldout_path=tmp_path / "heldout.json",
        report_path=tmp_path / "final.json",
        report=report,
        current_decision=_decision(_gold()),
    )
    with pytest.raises(regen.Sim2RealRerunRegenError, match="remote Stage 4"):
        regen._validate_regen_component_authority(
            _config(),
            tmp_path,
            object(),
            inputs,
            verify_remote_authority=True,
        )


class _ObjectStore:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_once_uri = ""

    @staticmethod
    def _etag(payload: bytes) -> str:
        return f'"{hashlib.md5(payload, usedforsecurity=False).hexdigest()}"'

    def read_bytes_with_etag(self, uri: str) -> tuple[bytes, str] | None:
        payload = self.objects.get(uri)
        return None if payload is None else (payload, self._etag(payload))

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        **_kwargs: Any,
    ) -> str:
        self.calls.append(
            ("put", uri, {"if_match": if_match, "if_none_match": if_none_match})
        )
        if uri == self.fail_once_uri:
            self.fail_once_uri = ""
            raise RuntimeError("injected publication failure")
        current = self.read_bytes_with_etag(uri)
        if if_none_match:
            if current is not None:
                raise StoragePreconditionFailed(uri)
        elif current is None or current[1] != if_match:
            raise StoragePreconditionFailed(uri)
        self.objects[uri] = bytes(payload)
        return self._etag(payload)

    def delete_file_conditional(self, uri: str, *, if_match: str) -> None:
        self.calls.append(("delete", uri, {"if_match": if_match}))
        current = self.read_bytes_with_etag(uri)
        if current is None or current[1] != if_match:
            raise StoragePreconditionFailed(uri)
        del self.objects[uri]


def test_equal_payload_alias_replacement_still_executes_cas(tmp_path: Path) -> None:
    uri = f"{ROOT}/reports/sim2real.rrd"
    path = tmp_path / "sim2real.rrd"
    path.write_bytes(b"same")
    storage = _ObjectStore({uri: b"same"})
    snapshot = RemoteObjectSnapshot(b"same", storage._etag(b"same"))

    replace_mutable_file(storage, path, uri, snapshot)

    assert [call[:2] for call in storage.calls] == [("put", uri)]
    assert storage.calls[0][2]["if_match"] == snapshot.etag


def test_multi_alias_failure_rolls_back_every_prior_cas(tmp_path: Path) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    rrd_uri = f"{ROOT}/reports/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    old = {report_uri: b"old-report", rrd_uri: b"old-rrd"}
    storage = _ObjectStore(old)
    snapshots = {
        uri: RemoteObjectSnapshot(payload, storage._etag(payload))
        for uri, payload in old.items()
    }
    report = tmp_path / "report.json"
    rrd = tmp_path / "recording.rrd"
    report.write_bytes(b"new-report")
    rrd.write_bytes(b"new-rrd")
    storage.fail_once_uri = rrd_uri

    with pytest.raises(RuntimeError, match="injected"):
        with publication.MutablePublicationTransaction(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=None,
            transaction_id="attempt-a",
        ) as transaction:
            transaction.replace_file(report, report_uri, snapshots[report_uri])
            transaction.replace_file(rrd, rrd_uri, snapshots[rrd_uri])

    assert {uri: storage.objects[uri] for uri in old} == old
    assert json.loads(storage.objects[lock_uri])["state"] == "aborted"


def test_older_transaction_cannot_delete_same_etag_rewrite(tmp_path: Path) -> None:
    mcap_uri = f"{ROOT}/reports/sim2real.mcap"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({mcap_uri: b"same-mcap"})
    stale_mcap = RemoteObjectSnapshot(b"same-mcap", storage._etag(b"same-mcap"))

    replacement = tmp_path / "sim2real.mcap"
    replacement.write_bytes(b"same-mcap")
    with publication.MutablePublicationTransaction(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="newer",
    ) as newer:
        newer.replace_file(replacement, mcap_uri, stale_mcap)

    with pytest.raises(PublicationConflict, match="transaction|superseded"):
        with publication.MutablePublicationTransaction(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=None,
            transaction_id="stale",
        ) as stale:
            stale.delete_file(mcap_uri, stale_mcap)

    assert storage.objects[mcap_uri] == b"same-mcap"


def test_prevalidation_snapshot_cannot_overwrite_newer_generation(
    tmp_path: Path,
) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({report_uri: b"old"})
    stale_report = RemoteObjectSnapshot(b"old", storage._etag(b"old"))
    newer_path = tmp_path / "newer.json"
    stale_path = tmp_path / "stale.json"
    newer_path.write_bytes(b"newer")
    stale_path.write_bytes(b"stale")

    with publication.MutablePublicationTransaction(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="newer-generation",
    ) as newer:
        newer.replace_file(
            newer_path,
            report_uri,
            stale_report,
            immutable_uri=f"{ROOT}/reports/generations/newer/report.json",
        )

    journal = json.loads(storage.objects[lock_uri])
    assert journal["state"] == "committed"
    assert journal["objects"] == [
        {
            "immutable_uri": f"{ROOT}/reports/generations/newer/report.json",
            "sha256": hashlib.sha256(b"newer").hexdigest(),
            "size_bytes": 5,
            "state": "present",
            "uri": report_uri,
        }
    ]
    with pytest.raises(PublicationConflict, match="transaction|superseded"):
        with publication.MutablePublicationTransaction(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=None,
            transaction_id="stale-generation",
        ) as stale:
            stale.replace_file(stale_path, report_uri, stale_report)

    assert storage.objects[report_uri] == b"newer"


def test_png_signature_without_decodable_image_is_rejected(tmp_path: Path) -> None:
    renders = tmp_path / "renders"
    frame = renders / "env-1/camera-000.png"
    frame.parent.mkdir(parents=True)
    payload = b"\x89PNG\r\n\x1a\nx"
    frame.write_bytes(payload)
    manifest = {
        "schema": "npa.sim2real.heldout_renders.v2",
        "episodes": [{"env_id": "env-1", "frames": ["camera-000.png"]}],
        "policy_checkpoint": {
            "uri": CHECKPOINT,
            "sha256": DIGEST,
            "size_bytes": 128,
        },
        "frame_artifacts": {
            "env-1/camera-000.png": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        },
    }
    with pytest.raises(RuntimeError, match="PNG|decode|frame bytes"):
        validate_materialized_render_tree(
            renders,
            manifest,
            {
                "checkpoint_uri": CHECKPOINT,
                "checkpoint_sha256": DIGEST,
                "checkpoint_size_bytes": 128,
            },
        )
