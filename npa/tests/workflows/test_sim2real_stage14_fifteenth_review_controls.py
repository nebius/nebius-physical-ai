from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from argparse import Namespace
from contextlib import contextmanager
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
    materialize_verified_render_snapshot,
    read_validated_render_png,
    validate_materialized_render_tree,
)
import npa.workflows.sim2real.stage10_execution as stage10
from npa.workflows.sim2real_viz import _decode_png_bytes
from npa.workflows.sim2real.workflow_io import component_record_history_uri
from npa.workflows.rerun_serve import (
    RerunServeError,
    _resolve_committed_rrd_uri,
)
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

pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack("!I", len(payload))
        + kind
        + payload
        + struct.pack("!I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _rgb_png(width: int, height: int, raw: bytes) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
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


def test_stage14_verified_snapshot_stays_under_encoder_root(tmp_path: Path) -> None:
    state = _materialized_state(tmp_path)
    frame = state.local / "eval/gold-heldout/outer-01/renders/env-1/camera-000.png"
    frame.write_bytes(VALID_PNG)

    snapshot = stage14._validate_stage14_materialized_frames(state)

    assert snapshot.resolve().is_relative_to(state.local.resolve())


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
        regen._validate_regen_materialized_frames(
            _config(),
            state.local,
            inputs,
            snapshot_dir=tmp_path / "verified",
        )


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
        self.commit_then_fail_once_uri = ""
        self.commit_then_fail_state = ""
        self.delete_then_fail_once_uri = ""

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
        if uri == self.commit_then_fail_once_uri:
            if self.commit_then_fail_state:
                state = json.loads(payload).get("state")
                if state != self.commit_then_fail_state:
                    return self._etag(payload)
            self.commit_then_fail_once_uri = ""
            self.commit_then_fail_state = ""
            raise OSError("response lost after committed CAS write")
        return self._etag(payload)

    def delete_file_conditional(self, uri: str, *, if_match: str) -> None:
        self.calls.append(("delete", uri, {"if_match": if_match}))
        current = self.read_bytes_with_etag(uri)
        if current is None or current[1] != if_match:
            raise StoragePreconditionFailed(uri)
        del self.objects[uri]
        if uri == self.delete_then_fail_once_uri:
            self.delete_then_fail_once_uri = ""
            raise OSError("response lost after committed CAS delete")


class _StreamingObjectStore(_ObjectStore):
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        super().__init__(objects)
        self.full_body_reads: list[str] = []

    def read_bytes_with_etag(self, uri: str) -> tuple[bytes, str] | None:
        if uri.endswith(".rrd"):
            self.full_body_reads.append(uri)
            raise AssertionError("large recording body must not be snapshotted")
        return super().read_bytes_with_etag(uri)

    def read_object_version(self, uri: str) -> tuple[str, int, str] | None:
        payload = self.objects.get(uri)
        if payload is None:
            return None
        return (
            self._etag(payload),
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )

    def put_file_conditional(
        self,
        local_file: str,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        **_kwargs: Any,
    ) -> str:
        payload = Path(local_file).read_bytes()
        current = self.objects.get(uri)
        if if_none_match:
            if current is not None:
                raise StoragePreconditionFailed(uri)
        elif current is None or self._etag(current) != if_match:
            raise StoragePreconditionFailed(uri)
        self.objects[uri] = payload
        return self._etag(payload)

    def download_file(self, uri: str, local_file: str) -> str:
        Path(local_file).write_bytes(self.objects[uri])
        return local_file


def _queue_required_publication_targets(
    transaction: publication.MutablePublicationTransaction,
    storage: _ObjectStore,
    tmp_path: Path,
) -> None:
    planned = {item.uri for item in transaction._planned}
    targets = {
        "report": f"{ROOT}/reports/sim2real-report.json",
        "rrd": f"{ROOT}/reports/sim2real.rrd",
        "mcap": f"{ROOT}/reports/sim2real.mcap",
        "stage14": f"{ROOT}/components/stage_14.json",
    }
    if targets["mcap"] not in planned:
        current = publication.remote_object_version(storage, targets["mcap"])
        transaction.delete_file(targets["mcap"], current)
    for label in ("report", "rrd", "stage14"):
        uri = targets[label]
        if uri in planned:
            continue
        payload = f"carried-{label}".encode()
        path = tmp_path / f"{transaction.transaction_id}-{label}"
        path.write_bytes(payload)
        if label == "stage14":
            immutable_uri = (
                f"{ROOT}/components/history/stage_14/"
                f"{hashlib.sha256(payload).hexdigest()}.json"
            )
        else:
            immutable_uri = (
                f"{ROOT}/reports/generations/{transaction.transaction_id}/"
                f"{uri.rsplit('/', 1)[-1]}"
            )
        storage.objects[immutable_uri] = payload
        transaction.replace_file(
            path,
            uri,
            publication.remote_object_version(storage, uri),
            immutable_uri=immutable_uri,
        )


@contextmanager
def _complete_transaction(
    storage: _ObjectStore,
    tmp_path: Path,
    **kwargs: Any,
):
    with publication.MutablePublicationTransaction(storage, **kwargs) as transaction:
        yield transaction
        _queue_required_publication_targets(transaction, storage, tmp_path)


_CARRIED_JOURNAL_PAYLOADS = {
    f"{ROOT}/reports/sim2real-report.json": b"carried-report",
    f"{ROOT}/reports/sim2real.rrd": b"carried-rrd",
    f"{ROOT}/components/stage_14.json": b"carried-stage14",
}


def _complete_journal_objects(
    transaction_id: str,
    objects: list[dict[str, object]],
) -> list[dict[str, object]]:
    complete = list(objects)
    seen = {str(item["uri"]) for item in complete}
    mcap_uri = f"{ROOT}/reports/sim2real.mcap"
    if mcap_uri not in seen:
        complete.append({"uri": mcap_uri, "state": "absent"})
    for uri, payload in _CARRIED_JOURNAL_PAYLOADS.items():
        if uri in seen:
            continue
        if uri.endswith("/components/stage_14.json"):
            immutable_uri = (
                f"{ROOT}/components/history/stage_14/"
                f"{hashlib.sha256(payload).hexdigest()}.json"
            )
        else:
            immutable_uri = (
                f"{ROOT}/reports/generations/{transaction_id}/{uri.rsplit('/', 1)[-1]}"
            )
        complete.append(
            {
                "uri": uri,
                "state": "present",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "immutable_uri": immutable_uri,
            }
        )
    return complete


def _seed_carried_journal_sources(
    storage: _ObjectStore,
    journal: bytes,
) -> None:
    for item in json.loads(journal)["objects"]:
        if item["state"] != "present":
            continue
        payload = _CARRIED_JOURNAL_PAYLOADS.get(str(item["uri"]))
        if (
            payload is not None
            and item["sha256"] == hashlib.sha256(payload).hexdigest()
        ):
            storage.objects.setdefault(str(item["immutable_uri"]), payload)


def test_large_recording_publication_uses_metadata_and_streaming_paths(
    tmp_path: Path,
) -> None:
    canonical_uri = f"{ROOT}/reports/sim2real.rrd"
    immutable_uri = f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _StreamingObjectStore({canonical_uri: b"old"})
    recording = tmp_path / "sim2real.rrd"
    recording.write_bytes(b"new-recording")

    publication.upload_immutable_file(storage, recording, immutable_uri)
    snapshot = publication.remote_object_version(storage, canonical_uri)
    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="generation-a",
    ) as transaction:
        transaction.replace_file(
            recording,
            canonical_uri,
            snapshot,
            immutable_uri=immutable_uri,
        )

    assert storage.objects[canonical_uri] == b"new-recording"
    assert storage.full_body_reads == []


def test_equal_payload_alias_replacement_still_executes_cas(tmp_path: Path) -> None:
    uri = f"{ROOT}/reports/sim2real.rrd"
    path = tmp_path / "sim2real.rrd"
    path.write_bytes(b"same")
    storage = _ObjectStore({uri: b"same"})
    snapshot = RemoteObjectSnapshot(b"same", storage._etag(b"same"))

    replace_mutable_file(storage, path, uri, snapshot)

    assert [call[:2] for call in storage.calls] == [("put", uri)]
    assert storage.calls[0][2]["if_match"] == snapshot.etag


def test_multi_alias_failure_leaves_complete_plan_for_roll_forward(
    tmp_path: Path,
) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    rrd_uri = f"{ROOT}/reports/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    immutable_report = f"{ROOT}/reports/generations/a/sim2real-report.json"
    immutable_rrd = f"{ROOT}/reports/generations/a/sim2real.rrd"
    old = {report_uri: b"old-report", rrd_uri: b"old-rrd"}
    storage = _ObjectStore(
        {
            **old,
            immutable_report: b"new-report",
            immutable_rrd: b"new-rrd",
        }
    )
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
        with _complete_transaction(
            storage,
            tmp_path,
            lock_uri=lock_uri,
            lock_snapshot=None,
            transaction_id="a",
        ) as transaction:
            transaction.replace_file(
                report,
                report_uri,
                snapshots[report_uri],
                immutable_uri=immutable_report,
            )
            transaction.replace_file(
                rrd,
                rrd_uri,
                snapshots[rrd_uri],
                immutable_uri=immutable_rrd,
            )

    assert storage.objects[report_uri] == b"new-report"
    assert storage.objects[rrd_uri] == b"old-rrd"
    assert json.loads(storage.objects[lock_uri])["state"] == "publishing"

    publication.recover_interrupted_publication(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=RemoteObjectSnapshot(
            storage.objects[lock_uri],
            storage._etag(storage.objects[lock_uri]),
        ),
    )

    assert storage.objects[report_uri] == b"new-report"
    assert storage.objects[rrd_uri] == b"new-rrd"
    assert json.loads(storage.objects[lock_uri])["state"] == "committed"


def test_ambiguous_alias_write_is_reconciled_before_commit(tmp_path: Path) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({report_uri: b"old-report"})
    report_snapshot = RemoteObjectSnapshot(
        b"old-report",
        storage._etag(b"old-report"),
    )
    report = tmp_path / "report.json"
    report.write_bytes(b"new-report")
    storage.commit_then_fail_once_uri = report_uri

    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="a",
    ) as transaction:
        transaction.replace_file(
            report,
            report_uri,
            report_snapshot,
            immutable_uri=f"{ROOT}/reports/generations/a/sim2real-report.json",
        )

    journal = json.loads(storage.objects[lock_uri])
    assert storage.objects[report_uri] == b"new-report"
    assert journal["state"] == "committed"
    assert journal["objects"][0]["sha256"] == hashlib.sha256(b"new-report").hexdigest()


def test_equal_payload_error_cannot_be_misread_as_success(tmp_path: Path) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    storage = _ObjectStore({report_uri: b"same-report"})
    report_snapshot = RemoteObjectSnapshot(
        b"same-report",
        storage._etag(b"same-report"),
    )
    report = tmp_path / "report.json"
    report.write_bytes(b"same-report")
    storage.fail_once_uri = report_uri

    with pytest.raises(RuntimeError, match="injected publication failure"):
        replace_mutable_file(storage, report, report_uri, report_snapshot)

    assert storage.objects[report_uri] == b"same-report"
    assert [call[:2] for call in storage.calls] == [("put", report_uri)]


def test_ambiguous_alias_delete_is_reconciled_before_commit(tmp_path: Path) -> None:
    mcap_uri = f"{ROOT}/reports/sim2real.mcap"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({mcap_uri: b"old-mcap"})
    mcap_snapshot = RemoteObjectSnapshot(b"old-mcap", storage._etag(b"old-mcap"))
    storage.delete_then_fail_once_uri = mcap_uri

    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="a",
    ) as transaction:
        transaction.delete_file(mcap_uri, mcap_snapshot)

    journal = json.loads(storage.objects[lock_uri])
    assert mcap_uri not in storage.objects
    assert journal["state"] == "committed"
    assert len(journal["objects"]) == 4
    assert next(item for item in journal["objects"] if item["uri"] == mcap_uri) == {
        "state": "absent",
        "uri": mcap_uri,
    }


def test_ambiguous_journal_commit_is_reconciled_without_rollback(
    tmp_path: Path,
) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({report_uri: b"old-report"})
    report_snapshot = RemoteObjectSnapshot(
        b"old-report",
        storage._etag(b"old-report"),
    )
    report = tmp_path / "report.json"
    report.write_bytes(b"new-report")
    storage.commit_then_fail_once_uri = lock_uri
    storage.commit_then_fail_state = "committed"

    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="a",
    ) as transaction:
        transaction.replace_file(
            report,
            report_uri,
            report_snapshot,
            immutable_uri=f"{ROOT}/reports/generations/a/sim2real-report.json",
        )

    journal = json.loads(storage.objects[lock_uri])
    assert storage.objects[report_uri] == b"new-report"
    assert journal["state"] == "committed"
    assert journal["objects"][0]["sha256"] == hashlib.sha256(b"new-report").hexdigest()


def test_ambiguous_publishing_journal_write_is_reconciled(
    tmp_path: Path,
) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({report_uri: b"old-report"})
    report_snapshot = RemoteObjectSnapshot(
        b"old-report",
        storage._etag(b"old-report"),
    )
    report = tmp_path / "report.json"
    report.write_bytes(b"new-report")
    storage.commit_then_fail_once_uri = lock_uri
    storage.commit_then_fail_state = "publishing"

    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="a",
    ) as transaction:
        transaction.replace_file(
            report,
            report_uri,
            report_snapshot,
            immutable_uri=f"{ROOT}/reports/generations/a/sim2real-report.json",
        )

    assert storage.objects[report_uri] == b"new-report"
    assert json.loads(storage.objects[lock_uri])["state"] == "committed"


def test_older_transaction_cannot_delete_same_etag_rewrite(tmp_path: Path) -> None:
    mcap_uri = f"{ROOT}/reports/sim2real.mcap"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    storage = _ObjectStore({mcap_uri: b"same-mcap"})
    stale_mcap = RemoteObjectSnapshot(b"same-mcap", storage._etag(b"same-mcap"))

    replacement = tmp_path / "sim2real.mcap"
    replacement.write_bytes(b"same-mcap")
    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="newer",
    ) as newer:
        newer.replace_file(
            replacement,
            mcap_uri,
            stale_mcap,
            immutable_uri=f"{ROOT}/reports/generations/newer/sim2real.mcap",
        )

    with pytest.raises(PublicationConflict, match="transaction|superseded"):
        with _complete_transaction(
            storage,
            tmp_path,
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

    with _complete_transaction(
        storage,
        tmp_path,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="newer",
    ) as newer:
        newer.replace_file(
            newer_path,
            report_uri,
            stale_report,
            immutable_uri=(f"{ROOT}/reports/generations/newer/sim2real-report.json"),
        )

    journal = json.loads(storage.objects[lock_uri])
    assert journal["state"] == "committed"
    assert len(journal["objects"]) == 4
    assert next(item for item in journal["objects"] if item["uri"] == report_uri) == {
        "immutable_uri": (f"{ROOT}/reports/generations/newer/sim2real-report.json"),
        "sha256": hashlib.sha256(b"newer").hexdigest(),
        "size_bytes": 5,
        "state": "present",
        "uri": report_uri,
    }
    with pytest.raises(PublicationConflict, match="transaction|superseded"):
        with _complete_transaction(
            storage,
            tmp_path,
            lock_uri=lock_uri,
            lock_snapshot=None,
            transaction_id="stale",
        ) as stale:
            stale.replace_file(
                stale_path,
                report_uri,
                stale_report,
                immutable_uri=(
                    f"{ROOT}/reports/generations/stale/sim2real-report.json"
                ),
            )

    assert storage.objects[report_uri] == b"newer"


def test_interrupted_complete_plan_rolls_forward_and_fences_stale_attempt(
    tmp_path: Path,
) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    rrd_uri = f"{ROOT}/reports/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    immutable_report_uri = f"{ROOT}/reports/generations/a/sim2real-report.json"
    immutable_rrd_uri = f"{ROOT}/reports/generations/a/sim2real.rrd"
    storage = _ObjectStore(
        {
            report_uri: b"old-report",
            rrd_uri: b"old-rrd",
            immutable_report_uri: b"new-report",
            immutable_rrd_uri: b"new-rrd",
        }
    )
    report = tmp_path / "report.json"
    rrd = tmp_path / "sim2real.rrd"
    report.write_bytes(b"new-report")
    rrd.write_bytes(b"new-rrd")
    interrupted = publication.MutablePublicationTransaction(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=None,
        transaction_id="a",
    )
    interrupted.__enter__()
    interrupted.replace_file(
        report,
        report_uri,
        RemoteObjectSnapshot(b"old-report", storage._etag(b"old-report")),
        immutable_uri=immutable_report_uri,
    )
    interrupted.replace_file(
        rrd,
        rrd_uri,
        RemoteObjectSnapshot(b"old-rrd", storage._etag(b"old-rrd")),
        immutable_uri=immutable_rrd_uri,
    )
    _queue_required_publication_targets(interrupted, storage, tmp_path)
    interrupted._begin_journal()
    interrupted._apply_mutation(interrupted._planned[0])

    publishing = json.loads(storage.objects[lock_uri])
    assert publishing["state"] == "publishing"
    assert len(publishing["objects"]) == 4
    assert storage.objects[report_uri] == b"new-report"
    assert storage.objects[rrd_uri] == b"old-rrd"

    assert publication.recover_interrupted_publication(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=RemoteObjectSnapshot(
            storage.objects[lock_uri],
            storage._etag(storage.objects[lock_uri]),
        ),
    )

    committed = json.loads(storage.objects[lock_uri])
    assert committed["state"] == "committed"
    assert committed["objects"] == publishing["objects"]
    assert storage.objects[report_uri] == b"new-report"
    assert storage.objects[rrd_uri] == b"new-rrd"
    with pytest.raises(PublicationConflict, match="superseded"):
        interrupted._apply_mutation(interrupted._planned[1])


def test_planned_local_bytes_cannot_change_before_publication(tmp_path: Path) -> None:
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    report = tmp_path / "report.json"
    report.write_bytes(b"planned")
    immutable_uri = f"{ROOT}/reports/generations/a/sim2real-report.json"
    storage = _ObjectStore({report_uri: b"old", immutable_uri: b"planned"})

    with pytest.raises(PublicationConflict, match="planned publication bytes"):
        with _complete_transaction(
            storage,
            tmp_path,
            lock_uri=lock_uri,
            lock_snapshot=None,
            transaction_id="a",
        ) as transaction:
            transaction.replace_file(
                report,
                report_uri,
                RemoteObjectSnapshot(b"old", storage._etag(b"old")),
                immutable_uri=immutable_uri,
            )
            report.write_bytes(b"changed")

    assert storage.objects[report_uri] == b"old"
    assert json.loads(storage.objects[lock_uri])["state"] == "publishing"
    publication.recover_interrupted_publication(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=RemoteObjectSnapshot(
            storage.objects[lock_uri],
            storage._etag(storage.objects[lock_uri]),
        ),
    )
    assert storage.objects[report_uri] == b"planned"
    assert json.loads(storage.objects[lock_uri])["state"] == "committed"


def test_recovery_rejects_duplicate_journal_keys() -> None:
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    payload = (
        b'{"schema":"npa.sim2real.mutable_publication.v1",'
        b'"transaction_id":"a","attempt_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"state":"publishing","state":"committed","objects":[]}\n'
    )
    storage = _ObjectStore({lock_uri: payload})

    with pytest.raises(PublicationConflict, match="invalid"):
        publication.recover_interrupted_publication(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=RemoteObjectSnapshot(payload, storage._etag(payload)),
        )


def test_recovery_rejects_mixed_immutable_generations() -> None:
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    payload = publication._journal_bytes(
        transaction_id="generation-a",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": f"{ROOT}/reports/sim2real.rrd",
                    "state": "present",
                    "sha256": hashlib.sha256(b"rrd").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": (
                        f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
                    ),
                },
                {
                    "uri": f"{ROOT}/reports/sim2real-report.json",
                    "state": "present",
                    "sha256": hashlib.sha256(b"report").hexdigest(),
                    "size_bytes": 6,
                    "immutable_uri": (
                        f"{ROOT}/reports/generations/generation-b/sim2real-report.json"
                    ),
                },
            ],
        ),
    )
    storage = _ObjectStore({lock_uri: payload})

    with pytest.raises(PublicationConflict, match="mixes immutable generations"):
        publication.recover_interrupted_publication(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=RemoteObjectSnapshot(payload, storage._etag(payload)),
        )


def test_recovery_rejects_transaction_generation_identity_mismatch() -> None:
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    immutable_uri = f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
    payload = publication._journal_bytes(
        transaction_id="generation-b",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": f"{ROOT}/reports/sim2real.rrd",
                    "state": "present",
                    "sha256": hashlib.sha256(b"rrd").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )
    storage = _ObjectStore({lock_uri: payload, immutable_uri: b"rrd"})

    with pytest.raises(PublicationConflict, match="identity disagrees"):
        publication.recover_interrupted_publication(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=RemoteObjectSnapshot(payload, storage._etag(payload)),
        )


def test_recovery_rejects_cross_run_immutable_source() -> None:
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    payload = publication._journal_bytes(
        transaction_id="generation-a",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": report_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"new").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": (
                        "s3://other/run/reports/generations/a/sim2real-report.json"
                    ),
                }
            ],
        ),
    )
    storage = _ObjectStore({lock_uri: payload})

    with pytest.raises(PublicationConflict, match="outside its run"):
        publication.recover_interrupted_publication(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=RemoteObjectSnapshot(payload, storage._etag(payload)),
        )


def test_recovery_rejects_changed_immutable_source_bytes() -> None:
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    report_uri = f"{ROOT}/reports/sim2real-report.json"
    immutable_uri = f"{ROOT}/reports/generations/a/sim2real-report.json"
    payload = publication._journal_bytes(
        transaction_id="a",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "a",
            [
                {
                    "uri": report_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"new").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )
    storage = _ObjectStore(
        {
            lock_uri: payload,
            report_uri: b"old",
            immutable_uri: b"tampered",
        }
    )

    with pytest.raises(PublicationConflict, match="source is invalid"):
        publication.recover_interrupted_publication(
            storage,
            lock_uri=lock_uri,
            lock_snapshot=RemoteObjectSnapshot(payload, storage._etag(payload)),
        )
    assert storage.objects[report_uri] == b"old"
    assert json.loads(storage.objects[lock_uri])["state"] == "publishing"


def test_stage14_snapshot_phase_recovers_interrupted_plan_before_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rrd_uri = f"{ROOT}/reports/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    immutable_uri = f"{ROOT}/reports/generations/a/sim2real.rrd"
    journal = publication._journal_bytes(
        transaction_id="a",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "a",
            [
                {
                    "uri": rrd_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"new-rrd").hexdigest(),
                    "size_bytes": len(b"new-rrd"),
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )
    storage = _ObjectStore(
        {
            rrd_uri: b"old-rrd",
            immutable_uri: b"new-rrd",
            lock_uri: journal,
        }
    )
    _seed_carried_journal_sources(storage, journal)
    monkeypatch.setattr(stage14, "storage", lambda: storage)
    state = stage14._stage14_state(
        Namespace(run_id=RUN_ID, outer_iteration=1),
        ROOT,
        tmp_path,
        tmp_path / "run",
        _evidence(),
        _gold(),
    )

    captured = stage14._capture_stage14_publication_snapshots(state)

    assert storage.objects[rrd_uri] == b"new-rrd"
    assert json.loads(storage.objects[lock_uri])["state"] == "committed"
    assert captured.publication_snapshots is not None
    assert captured.publication_snapshots[rrd_uri].payload == b"new-rrd"
    assert (
        json.loads(captured.publication_snapshots[lock_uri].payload)["state"]
        == "committed"
    )


def test_regeneration_reader_resolves_committed_immutable_recording(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical_uri = f"{ROOT}/reports/sim2real.rrd"
    immutable_uri = f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    journal = publication._journal_bytes(
        transaction_id="generation-a",
        attempt_id="a" * 32,
        state="committed",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": canonical_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"rrd").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )
    storage = _ObjectStore({lock_uri: journal, immutable_uri: b"rrd"})
    requested: list[str] = []

    def download(
        _client: Any,
        uri: str,
        target: Path,
        **_kwargs: Any,
    ) -> bool:
        requested.append(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"rrd")
        return True

    monkeypatch.setattr(regen, "_download_if_exists", download)
    destination = tmp_path / "sim2real.rrd"

    regen.download_rrd_from_s3(_config(), dest_path=destination, client=storage)

    assert requested == [immutable_uri]
    assert destination.read_bytes() == b"rrd"


def test_regeneration_reader_rejects_in_progress_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical_uri = f"{ROOT}/reports/sim2real.rrd"
    immutable_uri = f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
    lock_uri = f"{ROOT}/reports/.sim2real-publication.json"
    journal = publication._journal_bytes(
        transaction_id="generation-a",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": canonical_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"rrd").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )
    storage = _ObjectStore({lock_uri: journal, immutable_uri: b"rrd"})
    monkeypatch.setattr(
        regen,
        "_download_if_exists",
        lambda *_args, **_kwargs: pytest.fail(
            "reader must not consume an alias during publication"
        ),
    )

    with pytest.raises(
        regen.Sim2RealRerunRegenError,
        match="not committed",
    ):
        regen.download_rrd_from_s3(
            _config(),
            dest_path=tmp_path / "sim2real.rrd",
            client=storage,
        )


def test_rerun_viewer_resolves_committed_immutable_recording() -> None:
    canonical_uri = f"{ROOT}/reports/sim2real.rrd"
    immutable_uri = f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
    journal = publication._journal_bytes(
        transaction_id="generation-a",
        attempt_id="a" * 32,
        state="committed",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": canonical_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"rrd").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )

    class Body:
        def read(self, _amount: int) -> bytes:
            return journal

        def close(self) -> None:
            pass

    def get_object(**kwargs: str) -> dict[str, Any]:
        assert kwargs["Key"].endswith("reports/.sim2real-publication.json")
        return {"Body": Body(), "ETag": '"journal"'}

    assert (
        _resolve_committed_rrd_uri(canonical_uri, get_object=get_object)
        == immutable_uri
    )


def test_rerun_viewer_rejects_in_progress_generation() -> None:
    canonical_uri = f"{ROOT}/reports/sim2real.rrd"
    immutable_uri = f"{ROOT}/reports/generations/generation-a/sim2real.rrd"
    journal = publication._journal_bytes(
        transaction_id="generation-a",
        attempt_id="a" * 32,
        state="publishing",
        objects=_complete_journal_objects(
            "generation-a",
            [
                {
                    "uri": canonical_uri,
                    "state": "present",
                    "sha256": hashlib.sha256(b"rrd").hexdigest(),
                    "size_bytes": 3,
                    "immutable_uri": immutable_uri,
                }
            ],
        ),
    )

    class Body:
        def read(self, _amount: int) -> bytes:
            return journal

        def close(self) -> None:
            pass

    with pytest.raises(RerunServeError, match="not committed"):
        _resolve_committed_rrd_uri(
            canonical_uri,
            get_object=lambda **_kwargs: {"Body": Body(), "ETag": '"journal"'},
        )


def test_regeneration_seals_canonical_mcap_alias_separately_from_generation(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "sim2real-report.json"
    report_path.write_text(
        json.dumps(
            {
                "component_records": [
                    _stage14_component(
                        rrd_uri=f"{ROOT}/reports/old.rrd",
                        mcap_uri=f"{ROOT}/reports/old.mcap",
                        report_uri=f"{ROOT}/reports/old.json",
                    )
                ]
            }
        )
    )
    generation = f"{ROOT}/reports/generations/generation-a"
    regen._seal_regen_publication_report(
        report_path,
        publication_id="generation-a",
        rrd_uri=f"{generation}/sim2real.rrd",
        mcap_uri=f"{generation}/sim2real.mcap",
        report_uri=f"{generation}/sim2real-report.json",
        journal_uri=f"{ROOT}/reports/.sim2real-publication.json",
    )

    report = json.loads(report_path.read_text())
    assert report["mcap_uri"] == f"{generation}/sim2real.mcap"
    assert report["canonical_mcap_uri"] == f"{ROOT}/reports/sim2real.mcap"


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


def test_verified_render_snapshot_isolated_from_post_validation_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    frame = source / "env-1/camera-000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(VALID_PNG)
    manifest = {
        "episodes": [{"env_id": "env-1", "frames": ["camera-000.png"]}],
        "frame_artifacts": {
            "env-1/camera-000.png": {
                "sha256": hashlib.sha256(VALID_PNG).hexdigest(),
                "size_bytes": len(VALID_PNG),
            }
        },
    }

    snapshot = materialize_verified_render_snapshot(
        source,
        tmp_path / "verified",
        manifest,
        {},
        require_checkpoint_identity=False,
        require_frame_identity=True,
    )
    frame.write_bytes(_rgb_png(1, 1, b"\x00\x00\x00\x00"))

    assert (snapshot / "env-1/camera-000.png").read_bytes() == VALID_PNG


def test_render_decode_budget_rejects_scaled_bomb_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _rgb_png(2, 1, b"\x00" + b"\x00" * 6)
    frame = tmp_path / "camera.png"
    frame.write_bytes(payload)
    monkeypatch.setattr(stage10, "MAX_RENDER_PIXELS", 1)

    with pytest.raises(ValueError, match="safe decode budget"):
        read_validated_render_png(frame)
    assert (
        _decode_png_bytes(_rgb_png(stage10.MAX_RENDER_DIMENSION + 1, 1, b"\x00"))
        is None
    )
