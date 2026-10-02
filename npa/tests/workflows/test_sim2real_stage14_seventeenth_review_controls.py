from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import npa.cli.agent_stage_runtime as agent_stage_runtime
from npa.agent_backend.canonical_mcap import prepare_canonical_mcap
from npa.orchestration.npa_workflow.artifact_load import (
    discover_final_rerun_artifact,
)
from npa.workflows.artifacts import Artifact, select_preferred_artifact
from npa.workflows import data_factory_viz
from npa.workflows.sim2real import (
    artifact_upload,
    component_authority,
    monitor,
    publication,
    workflow_stage,
)
from npa.workflows.sim2real.publication import (
    PublicationConflict,
    RemoteObjectSnapshot,
)
from npa.workflows.sim2real.models import Sim2RealLoopConfig
import npa.workflows.sim2real_rerun_regen as regen
from tests.workflows.test_sim2real_stage14_fifteenth_review_controls import (
    _ObjectStore,
)
from tests.workflows.test_sim2real_stage14_thirteenth_review_controls import (
    ROOT,
    SOURCE_SHA,
    _component,
    _evidence,
    _gold,
    _rehash,
)


_TRANSACTION_ID = "generation-a"
_LOCK_URI = f"{ROOT}/reports/.sim2real-publication.json"
_CANONICAL_REPORT = f"{ROOT}/reports/sim2real-report.json"
_CANONICAL_RRD = f"{ROOT}/reports/sim2real.rrd"
_CANONICAL_MCAP = f"{ROOT}/reports/sim2real.mcap"
_CANONICAL_STAGE14 = f"{ROOT}/components/stage_14.json"
_IMMUTABLE_REPORT = f"{ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real-report.json"
_IMMUTABLE_RRD = f"{ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real.rrd"
_IMMUTABLE_MCAP = f"{ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real.mcap"
_STAGE14_BYTES = b'{"stage":14}\n'
_IMMUTABLE_STAGE14 = (
    f"{ROOT}/components/history/stage_14/"
    f"{hashlib.sha256(_STAGE14_BYTES).hexdigest()}.json"
)
_TEST_IMAGE = f"ghcr.io/example/sim2real-test@sha256:{'0' * 64}"


def _present(uri: str, immutable_uri: str, payload: bytes) -> dict[str, object]:
    return {
        "uri": uri,
        "state": "present",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "immutable_uri": immutable_uri,
    }


def _complete_journal(
    *,
    state: str = "committed",
    report: bytes = b'{"visualization":{"source":"committed"}}\n',
    rrd: bytes = b"committed-rrd",
    mcap: bytes | None = None,
    omit: frozenset[str] = frozenset(),
) -> bytes:
    objects = [
        _present(_CANONICAL_REPORT, _IMMUTABLE_REPORT, report),
        _present(_CANONICAL_RRD, _IMMUTABLE_RRD, rrd),
        (
            _present(_CANONICAL_MCAP, _IMMUTABLE_MCAP, mcap)
            if mcap is not None
            else {"uri": _CANONICAL_MCAP, "state": "absent"}
        ),
        _present(_CANONICAL_STAGE14, _IMMUTABLE_STAGE14, _STAGE14_BYTES),
    ]
    return publication._journal_bytes(
        transaction_id=_TRANSACTION_ID,
        attempt_id="a" * 32,
        state=state,
        objects=[item for item in objects if item["uri"] not in omit],
    )


@pytest.mark.parametrize(
    "omitted",
    [
        _CANONICAL_REPORT,
        _CANONICAL_RRD,
        _CANONICAL_MCAP,
        _CANONICAL_STAGE14,
    ],
)
def test_publication_journal_requires_every_reserved_target(omitted: str) -> None:
    payload = _complete_journal(omit=frozenset({omitted}))

    with pytest.raises(PublicationConflict, match="incomplete"):
        publication._parse_journal(
            RemoteObjectSnapshot(payload, "etag"),
            lock_uri=_LOCK_URI,
        )


def test_publication_journal_rejects_unproduced_aborted_state() -> None:
    with pytest.raises(PublicationConflict, match="incomplete"):
        publication._parse_journal(
            RemoteObjectSnapshot(_complete_journal(state="aborted"), "etag"),
            lock_uri=_LOCK_URI,
        )


def test_partial_regen_cannot_replace_a_journaled_generation(tmp_path: Path) -> None:
    storage = _ObjectStore(
        {
            _LOCK_URI: _complete_journal(),
            _CANONICAL_RRD: b"committed-rrd",
        }
    )
    recording = tmp_path / "reports" / "sim2real.rrd"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"legacy-regeneration")
    config = Sim2RealLoopConfig(
        run_id="run-a",
        s3_bucket="demo-bucket",
        s3_prefix="sim2real",
        s3_endpoint="https://storage.example",
        augment_image=_TEST_IMAGE,
        envgen_image=_TEST_IMAGE,
        policy_image=_TEST_IMAGE,
        trainer_image=_TEST_IMAGE,
        vlm_image=_TEST_IMAGE,
        eval_image=_TEST_IMAGE,
        isaac_image=_TEST_IMAGE,
    )
    snapshots = regen._capture_regen_publication_snapshots(config, storage)

    with pytest.raises(PublicationConflict, match="journaled regeneration"):
        regen.publish_regen_outputs(
            config,
            tmp_path,
            client=storage,
            snapshots=snapshots,
        )

    assert storage.objects[_CANONICAL_RRD] == b"committed-rrd"


def test_partial_regen_keeps_legacy_no_journal_runs_usable(tmp_path: Path) -> None:
    storage = _ObjectStore()
    recording = tmp_path / "reports" / "sim2real.rrd"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"legacy-regeneration")
    config = Sim2RealLoopConfig(
        run_id="run-a",
        s3_bucket="demo-bucket",
        s3_prefix="sim2real",
        s3_endpoint="https://storage.example",
        augment_image=_TEST_IMAGE,
        envgen_image=_TEST_IMAGE,
        policy_image=_TEST_IMAGE,
        trainer_image=_TEST_IMAGE,
        vlm_image=_TEST_IMAGE,
        eval_image=_TEST_IMAGE,
        isaac_image=_TEST_IMAGE,
    )

    result = regen.publish_regen_outputs(
        config,
        tmp_path,
        client=storage,
        snapshots=regen._capture_regen_publication_snapshots(config, storage),
    )

    assert result == _CANONICAL_RRD
    assert storage.objects[_CANONICAL_RRD] == b"legacy-regeneration"
    assert _LOCK_URI not in storage.objects


def test_legacy_mcap_delete_cannot_cross_a_publication_journal() -> None:
    storage = _ObjectStore(
        {
            _LOCK_URI: _complete_journal(),
            _CANONICAL_MCAP: b"stale-legacy-mcap",
        }
    )
    snapshot = publication.remote_object_version(storage, _CANONICAL_MCAP)

    with pytest.raises(PublicationConflict, match="journal-controlled"):
        publication.delete_unjournaled_legacy_file(
            storage,
            _CANONICAL_MCAP,
            snapshot=snapshot,
            lock_snapshot=None,
        )

    assert storage.objects[_CANONICAL_MCAP] == b"stale-legacy-mcap"


def test_stage4_rejects_a_gpu_less_lane_even_when_another_lane_has_gpu() -> None:
    components = [_component(stage) for stage in range(1, 14)]
    stage4 = components[3]
    proof = stage4["artifacts"]["shard_provenance"][0]["provenance"]
    lane = stage4["artifacts"]["lane_records"][0]
    for key in ("gpu_products",):
        proof.pop(key)
        lane["artifacts"].pop(key)
    _rehash(lane)
    stage4["artifacts"].update(
        component_authority.aggregate_parallel_provenance(
            [item["provenance"] for item in stage4["artifacts"]["shard_provenance"]],
            stage=4,
        )
    )
    _rehash(stage4)

    with pytest.raises(ValueError, match="lane|GPU|provenance"):
        component_authority.validate_component_records(
            components,
            root=ROOT,
            evidence=_evidence(),
            gold=_gold(),
            expected_source_sha=SOURCE_SHA,
        )


def test_stage4_rejects_lane_without_gpu_rows() -> None:
    components = [_component(stage) for stage in range(1, 14)]
    stage4 = components[3]
    proof = stage4["artifacts"]["shard_provenance"][0]["provenance"]
    lane = stage4["artifacts"]["lane_records"][0]
    proof.pop("gpu_rows")
    lane["artifacts"].pop("gpu_rows")
    _rehash(lane)
    _rehash(stage4)

    with pytest.raises(ValueError, match="lane|GPU|provenance"):
        component_authority.validate_component_records(
            components,
            root=ROOT,
            evidence=_evidence(),
            gold=_gold(),
            expected_source_sha=SOURCE_SHA,
        )


def test_stage4_rejects_extra_lane_authority_not_in_shard_proof() -> None:
    components = [_component(stage) for stage in range(1, 14)]
    stage4 = components[3]
    lane = stage4["artifacts"]["lane_records"][0]
    lane["artifacts"]["unproved_product"] = f"{ROOT}/envs/raw/unproved.json"
    _rehash(lane)
    _rehash(stage4)

    with pytest.raises(ValueError, match="disagrees with its shard proof"):
        component_authority.validate_component_records(
            components,
            root=ROOT,
            evidence=_evidence(),
            gold=_gold(),
            expected_source_sha=SOURCE_SHA,
        )


def test_stage5_rejects_gpu_less_lane_before_publishing_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows import sim2real_envgen

    stage4 = _component(4)
    proof = stage4["artifacts"]["shard_provenance"][0]["provenance"]
    lane = stage4["artifacts"]["lane_records"][0]
    proof.pop("gpu_products")
    proof.pop("gpu_rows")
    lane["artifacts"].pop("gpu_products")
    lane["artifacts"].pop("gpu_rows")
    _rehash(lane)
    split = {
        "disjoint": True,
        "config_digest_leakage": {},
        "train_count": 1,
        "validation_count": 1,
        "gold_heldout_count": 1,
    }

    def read_json(uri: str, **_kwargs: Any) -> dict[str, Any]:
        if uri.endswith("split-manifest.json"):
            return split
        if "/envs/raw/provenance-" in uri:
            index = int(uri.rsplit("-", 1)[-1].split(".", 1)[0])
            return stage4["artifacts"]["shard_provenance"][index]
        if "/components/lanes/stage_04/shard-" in uri:
            index = int(uri.rsplit("-", 1)[-1].split(".", 1)[0])
            return stage4["artifacts"]["lane_records"][index]
        raise AssertionError(uri)

    published: list[dict[str, Any]] = []
    monkeypatch.setattr(sim2real_envgen, "main", lambda _argv: 0)
    monkeypatch.setattr(workflow_stage, "_work", lambda _stage: tmp_path)
    monkeypatch.setattr(workflow_stage, "read_json", read_json)
    monkeypatch.setattr(workflow_stage, "source_sha", lambda: SOURCE_SHA)
    monkeypatch.setattr(
        workflow_stage,
        "publish_component_record",
        lambda **kwargs: published.append(kwargs),
    )
    args = SimpleNamespace(
        root_uri=ROOT,
        run_id="run-a",
        env_count=3,
        train_fraction=0.34,
        shard_count=2,
        seed=1,
        envgen_image=stage4["artifacts"]["image"],
    )

    with pytest.raises(RuntimeError, match="lane records"):
        workflow_stage._stage5(args)

    assert published == []


class _Reader:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.s3 = self
        self._s3 = self

    @staticmethod
    def _uri(bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"

    def read_small_bytes_with_etag(
        self, uri: str, *, max_bytes: int = 1024 * 1024
    ) -> tuple[bytes, str] | None:
        payload = self.objects.get(uri)
        if payload is None:
            return None
        if len(payload) > max_bytes:
            raise RuntimeError("oversized")
        return payload, hashlib.sha256(payload).hexdigest()

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        payload = self.objects.get(self._uri(Bucket, Key))
        if payload is None:
            raise RuntimeError("missing")
        return {"ContentLength": len(payload)}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        payload = self.objects.get(self._uri(Bucket, Key))
        if payload is None:
            raise RuntimeError("missing")
        return {"Body": io.BytesIO(payload)}

    def get_paginator(self, _name: str) -> Any:
        raise AssertionError("journaled readers must not list arbitrary generations")

    def download_path(self, uri: str, destination: str) -> str:
        payload = self.objects.get(uri)
        if payload is None:
            raise OSError("missing")
        Path(destination).write_bytes(payload)
        return destination


def test_artifact_handoff_resolves_the_committed_rrd_generation() -> None:
    journal = _complete_journal()
    storage = _Reader(
        {
            _LOCK_URI: journal,
            _CANONICAL_RRD: b"stale-alias",
            _IMMUTABLE_RRD: b"committed-rrd",
        }
    )

    assert discover_final_rerun_artifact(ROOT, client=storage) == _IMMUTABLE_RRD


def test_agent_key_resolution_rejects_alias_and_stale_generation_bypass() -> None:
    root_key = ROOT.removeprefix("s3://demo-bucket/")
    storage = _Reader({_LOCK_URI: _complete_journal()})

    assert agent_stage_runtime._resolve_committed_artifact_key(
        storage,
        "demo-bucket",
        root_key,
        _CANONICAL_RRD.removeprefix("s3://demo-bucket/"),
    ) == _IMMUTABLE_RRD.removeprefix("s3://demo-bucket/")
    with pytest.raises(PublicationConflict, match="committed"):
        agent_stage_runtime._resolve_committed_artifact_key(
            storage,
            "demo-bucket",
            root_key,
            f"{root_key}/reports/generations/stale/sim2real.rrd",
        )


def test_monitor_refuses_report_alias_while_publication_is_incomplete() -> None:
    publishing = _complete_journal(
        state="publishing",
        report=(
            b'{"outer_loop":{"latest_decision":'
            b'{"decision":"promote","success_rate":1.0,"threshold":0.5}}}\n'
        ),
    )
    storage = _Reader(
        {
            _LOCK_URI: publishing,
            _CANONICAL_REPORT: (
                b'{"outer_loop":{"latest_decision":'
                b'{"decision":"promote","success_rate":1.0,"threshold":0.5}}}\n'
            ),
        }
    )

    with pytest.raises(PublicationConflict, match="not committed"):
        monitor._extract_eval_metrics(
            workflow_state=None,
            client=storage,
            bucket="demo-bucket",
            run_prefix="sim2real/run-a",
        )


def test_regeneration_sync_reads_committed_report_not_mutable_alias(
    tmp_path: Path,
) -> None:
    committed = b'{"visualization":{"source":"committed"}}\n'
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(report=committed),
            _CANONICAL_REPORT: b'{"visualization":{"source":"stale"}}\n',
            _IMMUTABLE_REPORT: committed,
        }
    )
    destination = tmp_path / "reports" / "sim2real-report.json"

    downloaded = regen._download_regen_single_files(
        storage,
        f"{ROOT}/",
        tmp_path,
        {"reports/sim2real-report.json": destination},
    )

    assert downloaded == {"reports/sim2real-report.json": True}
    assert destination.read_bytes() == committed


def _artifact(key: str, *, render: str = "json") -> Artifact:
    return Artifact(
        run_id="run-a",
        key=key,
        s3_uri=f"s3://demo-bucket/{key}",
        size=100,
        last_modified="2026-10-02T00:00:00Z",
        render=render,
        inline=render == "json",
        relative_key=key.split("/run-a/", 1)[-1],
    )


def test_agent_publication_root_uses_artifact_relative_key() -> None:
    artifact = Artifact(
        run_id="run-a",
        key="archive/run-a/results/run-a/reports/sim2real.rrd",
        s3_uri="s3://demo-bucket/archive/run-a/results/run-a/reports/sim2real.rrd",
        size=100,
        last_modified="2026-10-02T00:00:00Z",
        render="rerun",
        inline=True,
        relative_key="reports/sim2real.rrd",
    )

    assert (
        agent_stage_runtime._run_root_key([artifact], "run-a")
        == "archive/run-a/results/run-a"
    )


def test_agent_run_details_use_one_committed_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "sim2real/run-a"
    lock_key = f"{prefix}/reports/.sim2real-publication.json"
    canonical_report_key = f"{prefix}/reports/sim2real-report.json"
    immutable_report_key = (
        f"{prefix}/reports/generations/{_TRANSACTION_ID}/sim2real-report.json"
    )
    canonical_rrd_key = f"{prefix}/reports/sim2real.rrd"
    immutable_rrd_key = f"{prefix}/reports/generations/{_TRANSACTION_ID}/sim2real.rrd"
    stale_rrd_key = f"{prefix}/reports/generations/z-stale/sim2real.rrd"
    stage14_key = _IMMUTABLE_STAGE14.removeprefix("s3://demo-bucket/")
    journal = _complete_journal()
    raw_objects = {
        lock_key: journal,
        canonical_report_key: b'{"visualization":{"source":"stale-source"}}\n',
        immutable_report_key: (b'{"visualization":{"source":"committed-source"}}\n'),
        canonical_rrd_key: b"stale-alias",
        immutable_rrd_key: b"committed-rrd",
        stale_rrd_key: b"later-name-stale-rrd",
        stage14_key: _STAGE14_BYTES,
    }

    class _RawS3:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            assert Bucket == "demo-bucket"
            return {"Body": io.BytesIO(raw_objects[Key]), "ETag": "etag"}

    artifacts = [
        _artifact(lock_key),
        _artifact(canonical_report_key),
        _artifact(immutable_report_key),
        _artifact(canonical_rrd_key, render="rerun"),
        _artifact(immutable_rrd_key, render="rerun"),
        _artifact(stale_rrd_key, render="rerun"),
        _artifact(stage14_key),
    ]
    monkeypatch.setattr(
        agent_stage_runtime,
        "_agent_artifact_s3_client",
        lambda: (_RawS3(), {"bucket": "demo-bucket", "prefix": "sim2real"}),
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "list_artifacts",
        lambda *_args, **_kwargs: artifacts,
    )
    monkeypatch.setattr(agent_stage_runtime, "validate_run_id", lambda value: value)
    monkeypatch.setattr(agent_stage_runtime, "_agent_access_report", lambda: {})
    monkeypatch.setattr(
        agent_stage_runtime, "artifact_bucket_projects", lambda _report: {}
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "_artifact_discovery_prefix",
        lambda _settings, value: value,
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "_validated_resolved_prefix",
        lambda value: str(value or "").strip("/"),
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "parse_stage_evidence_documents",
        lambda _documents: {},
    )
    monkeypatch.setattr(
        agent_stage_runtime, "_workflow_draft_from_state", lambda _state: {}
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "run_owns_workflow_stage_overlay",
        lambda _state, _run_id: False,
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "build_artifact_backed_stages",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        agent_stage_runtime,
        "summarize_stage_evidence",
        lambda _stages: {"observed_stage_count": 0},
    )
    monkeypatch.setattr(
        agent_stage_runtime, "select_preferred_artifact", select_preferred_artifact
    )
    monkeypatch.setattr(agent_stage_runtime, "_now_iso", lambda: "2026-10-02T00:00:00Z")

    details = agent_stage_runtime._artifact_backed_run_details(
        {},
        "run-a",
        prefix="sim2real",
    )

    assert details is not None
    assert "committed-source" in str(details["logs"])
    assert immutable_rrd_key in str(details["logs"])
    assert "stale-source" not in str(details["logs"])
    assert stale_rrd_key not in str(details["logs"])


def test_agent_inventory_page_hides_aliases_and_uncommitted_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "sim2real/run-a"
    lock_key = f"{prefix}/reports/.sim2real-publication.json"
    canonical_rrd_key = f"{prefix}/reports/sim2real.rrd"
    immutable_rrd_key = f"{prefix}/reports/generations/{_TRANSACTION_ID}/sim2real.rrd"
    stale_rrd_key = f"{prefix}/reports/generations/z-stale/sim2real.rrd"

    class _RawS3:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            assert Bucket == "demo-bucket"
            assert Key == lock_key
            return {"Body": io.BytesIO(_complete_journal()), "ETag": "etag"}

    artifacts = [
        _artifact(lock_key),
        _artifact(canonical_rrd_key, render="rerun"),
        _artifact(immutable_rrd_key, render="rerun"),
        _artifact(stale_rrd_key, render="rerun"),
    ]
    monkeypatch.setattr(
        agent_stage_runtime, "select_preferred_artifact", select_preferred_artifact
    )

    visible, report, preferred, _logical_keys = (
        agent_stage_runtime._committed_publication_artifacts(
            _RawS3(),
            "demo-bucket",
            "run-a",
            artifacts,
            require_complete=False,
        )
    )

    assert report is None
    assert preferred is not None and preferred.key == immutable_rrd_key
    visible_keys = {item.key for item in visible}
    assert immutable_rrd_key in visible_keys
    assert canonical_rrd_key not in visible_keys
    assert stale_rrd_key not in visible_keys


@pytest.mark.parametrize(
    ("stored_mcap", "accepted"),
    [
        (b"committed-mcap", True),
        (b"tampered--mcap", False),
    ],
)
def test_agent_reuses_only_byte_bound_committed_mcap(
    tmp_path: Path,
    stored_mcap: bytes,
    accepted: bool,
) -> None:
    expected_mcap = b"committed-mcap"
    root_key = "sim2real/run-a"
    lock_key = f"{root_key}/reports/.sim2real-publication.json"
    canonical_key = f"{root_key}/reports/sim2real.mcap"
    provenance_key = f"{canonical_key}.provenance.json"
    immutable_key = _IMMUTABLE_MCAP.removeprefix("s3://demo-bucket/")
    raw_objects = {
        lock_key: _complete_journal(mcap=expected_mcap),
        canonical_key: b"stale-alias",
        immutable_key: stored_mcap,
        provenance_key: (
            b'{"sha256":"'
            + hashlib.sha256(expected_mcap).hexdigest().encode()
            + b'","source":"stale-sidecar"}'
        ),
        f"{root_key}/camera/frame.png": b"frame",
    }
    put_keys: list[str] = []

    class _S3:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            assert Bucket == "demo-bucket"
            return {"Body": io.BytesIO(raw_objects[Key]), "ETag": "etag"}

        def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            assert Bucket == "demo-bucket"
            payload = raw_objects[Key]
            return {"ContentLength": len(payload), "ETag": "etag"}

        def put_object(
            self, *, Bucket: str, Key: str, Body: Any, **_kwargs: Any
        ) -> None:
            assert Bucket == "demo-bucket"
            assert Key != canonical_key
            put_keys.append(Key)
            raw_objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)

    s3 = _S3()

    def find(_buckets: list[str], **_kwargs: Any) -> tuple[str, list[Any]]:
        return "demo-bucket", [
            SimpleNamespace(key=key, s3_uri=f"s3://demo-bucket/{key}")
            for key in (
                f"{root_key}/camera/frame.png",
                canonical_key,
                immutable_key,
                provenance_key,
            )
        ]

    def download(uri: str, destination: Path, **_kwargs: Any) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw_objects[uri.removeprefix("s3://demo-bucket/")])
        return destination

    def summarize(path: Path) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "valid_magic": True,
                "size_bytes": path.stat().st_size,
                "message_count": 1,
                "channels": {"/camera": 1},
                "schemas": {},
                "metadata": {},
            }
        )

    kwargs = {
        "run_id": "run-a",
        "fps": 10.0,
        "max_frames": 10,
        "validate_run_id": lambda value: value,
        "s3_client": lambda: (s3, {"prefix": "sim2real"}),
        "list_buckets": lambda *_args: ["demo-bucket"],
        "find_artifacts": find,
        "safe_key": lambda value: value.strip("/"),
        "download": download,
        "convert": lambda **_kwargs: pytest.fail("committed MCAP must be reused"),
        "summarize": summarize,
        "invalidate_cache": lambda: None,
        "now_iso": lambda: "2026-10-02T00:00:00Z",
        "recordings_dir": tmp_path,
    }
    if not accepted:
        with pytest.raises(RuntimeError, match="disagree"):
            prepare_canonical_mcap(**kwargs)
        return

    result = prepare_canonical_mcap(**kwargs)
    assert result["artifact_key"] == immutable_key
    assert result["s3_uri"] == _IMMUTABLE_MCAP
    assert result["sha256"] == hashlib.sha256(expected_mcap).hexdigest()
    assert result["created"] is False
    assert result["source"] == "journal-committed-native"
    assert raw_objects[canonical_key] == b"stale-alias"
    assert provenance_key not in put_keys


def test_legacy_report_writer_cannot_overwrite_a_journaled_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = _ObjectStore(
        {
            _LOCK_URI: _complete_journal(),
            _CANONICAL_REPORT: b"committed-report",
        }
    )
    storage.upload_file = lambda local, uri: storage.objects.__setitem__(
        uri, Path(local).read_bytes()
    )
    monkeypatch.setattr(
        artifact_upload.StorageClient,
        "from_environment",
        lambda **_kwargs: storage,
    )
    report = tmp_path / "report.json"
    report.write_bytes(b"legacy-overwrite")
    config = SimpleNamespace(
        s3_bucket="demo-bucket",
        s3_endpoint="",
        s3_prefix="sim2real",
        run_id="run-a",
    )

    result = artifact_upload._upload_final_report(config, report)

    assert result["status"] == "blocked"
    assert storage.objects[_CANONICAL_REPORT] == b"committed-report"


def test_data_factory_writer_cannot_overwrite_a_journaled_rrd(
    tmp_path: Path,
) -> None:
    storage = _ObjectStore(
        {
            _LOCK_URI: _complete_journal(),
            _CANONICAL_RRD: b"committed-rrd",
        }
    )
    recording = tmp_path / "candidate.rrd"
    recording.write_bytes(b"cross-producer-overwrite")

    with pytest.raises(PublicationConflict, match="journal-controlled"):
        data_factory_viz._publish(
            str(recording),
            _CANONICAL_RRD,
            storage_client=storage,
        )

    assert storage.objects[_CANONICAL_RRD] == b"committed-rrd"


def test_agent_mcap_writer_cannot_overwrite_a_journaled_run(
    tmp_path: Path,
) -> None:
    journal_key = "sim2real/run-a/reports/.sim2real-publication.json"
    raw_objects = {
        "sim2real/run-a/camera/frame.png": b"frame",
        journal_key: _complete_journal(),
    }

    class _S3:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            assert Bucket == "demo-bucket"
            return {"Body": io.BytesIO(raw_objects[Key]), "ETag": "etag"}

        def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            assert Bucket == "demo-bucket"
            payload = raw_objects[Key]
            return {"ContentLength": len(payload), "ETag": "etag"}

        def put_object(
            self, *, Bucket: str, Key: str, Body: Any, **_kwargs: Any
        ) -> None:
            assert Bucket == "demo-bucket"
            raw_objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)

    s3 = _S3()

    def find(_buckets: list[str], **_kwargs: Any) -> tuple[str, list[Any]]:
        return "demo-bucket", [
            SimpleNamespace(key=key, s3_uri=f"s3://demo-bucket/{key}")
            for key in sorted(raw_objects)
        ]

    def download(uri: str, destination: Path, **_kwargs: Any) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw_objects[uri.removeprefix("s3://demo-bucket/")])
        return destination

    def convert(*, output_path: Path, **_kwargs: Any) -> Any:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x89MCAP0\r\nnew\x89MCAP0\r\n")
        return SimpleNamespace(to_dict=lambda: {"message_count": 1})

    def summarize(path: Path) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "valid_magic": True,
                "size_bytes": path.stat().st_size,
                "message_count": 1,
                "channels": {"/camera": 1},
                "schemas": {},
                "metadata": {},
            }
        )

    with pytest.raises(RuntimeError, match="journal|publication|reserved"):
        prepare_canonical_mcap(
            run_id="run-a",
            fps=10.0,
            max_frames=10,
            validate_run_id=lambda value: value,
            s3_client=lambda: (s3, {"prefix": "sim2real"}),
            list_buckets=lambda *_args: ["demo-bucket"],
            find_artifacts=find,
            safe_key=lambda value: value.strip("/"),
            download=download,
            convert=convert,
            summarize=summarize,
            invalidate_cache=lambda: None,
            now_iso=lambda: "2026-10-02T00:00:00Z",
            recordings_dir=tmp_path,
        )
