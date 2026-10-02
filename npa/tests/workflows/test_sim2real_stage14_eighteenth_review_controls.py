from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

import npa.cli.agent_stage_runtime as agent_stage_runtime
import npa.workflows.rerun_serve as rerun_serve_module
from npa.orchestration.npa_workflow.artifact_load import (
    ArtifactLoadError,
    discover_final_rerun_artifact,
)
from npa.workflows.rerun_serve import (
    RerunServeConfig,
    RerunServeError,
    _resolve_committed_rrd_uri,
    apply_rerun_serve,
    build_rerun_serve_manifest,
    verify_rrd_exists_on_s3,
)
from npa.workflows.sim2real import (
    artifact_upload,
    component_authority,
    monitor,
    workflow_io,
)
from npa.workflows.sim2real.publication import (
    PublicationConflict,
    resolve_committed_publication_snapshot,
    valid_publication_id,
)
from npa.workflows.sim2real.workflow_io import component_record_history_uri
import npa.workflows.sim2real_rerun_regen as regen
from tests.workflows.test_sim2real_stage14_fifteenth_review_controls import (
    _StreamingObjectStore,
    _canonical_report_with_forged_stage4,
)
from tests.workflows.test_sim2real_stage14_seventeenth_review_controls import (
    _CANONICAL_RRD,
    _IMMUTABLE_REPORT,
    _IMMUTABLE_RRD,
    _IMMUTABLE_STAGE14,
    _LOCK_URI,
    _Reader,
    _complete_journal,
)
from tests.workflows.test_sim2real_stage14_thirteenth_review_controls import (
    ROOT,
    SOURCE_SHA,
    _config,
    _component,
    _decision,
    _evidence,
    _gold,
    _rehash,
)


pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")


def test_artifact_handoff_rejects_committed_rrd_byte_mismatch() -> None:
    claimed = b"committed expected bytes"
    actual = b"hostile replacement bytes"
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(rrd=claimed),
            _IMMUTABLE_RRD: actual,
        }
    )

    with pytest.raises(ArtifactLoadError, match="digest|size|bytes|publication"):
        discover_final_rerun_artifact(ROOT, client=storage)


def test_rerun_serve_rejects_committed_rrd_byte_mismatch() -> None:
    claimed = b"expected-rrd"
    actual = b"tampered-rrd"
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(rrd=claimed),
            _IMMUTABLE_RRD: actual,
        }
    )

    def get_object(**kwargs: str) -> dict[str, object]:
        response = storage.get_object(**kwargs)
        response["ETag"] = '"object"'
        return response

    config = RerunServeConfig(
        run_id="run-a",
        s3_bucket="demo-bucket",
        s3_prefix="sim2real",
    )
    with pytest.raises(RerunServeError, match="digest|size|bytes|journal"):
        verify_rrd_exists_on_s3(
            config,
            head_object=storage.head_object,
            get_object=get_object,
        )


def test_rerun_serve_pod_reverifies_the_committed_rrd_identity() -> None:
    digest = hashlib.sha256(b"committed-rrd").hexdigest()
    config = RerunServeConfig(
        run_id="run-a",
        s3_bucket="demo-bucket",
        s3_prefix="sim2real",
        rrd_sha256=digest,
        rrd_size_bytes=len(b"committed-rrd"),
    )

    manifest = build_rerun_serve_manifest(config)
    deployment = next(
        item for item in manifest["items"] if item["kind"] == "Deployment"
    )
    script = deployment["spec"]["template"]["spec"]["initContainers"][0]["command"][2]

    assert digest in script
    assert "sha256sum -c" in script
    assert f'= "{len(b"committed-rrd")}"' in script


def test_rerun_apply_forwards_verified_identity_into_the_pod(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"committed-rrd"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        rerun_serve_module,
        "verify_rrd_exists_on_s3",
        lambda *_args, **_kwargs: (_IMMUTABLE_RRD, digest, len(payload)),
    )
    monkeypatch.setattr(
        rerun_serve_module,
        "fetch_rrd_sync_token",
        lambda _config: "immutable-etag",
    )
    applied: list[dict[str, object]] = []

    def kubectl(
        args: list[str],
        *,
        stdin: str | None = None,
        kubeconfig: str = "",
    ) -> str:
        del kubeconfig
        if args[:2] == ["apply", "-f"]:
            applied.append(json.loads(str(stdin)))
        return ""

    result = apply_rerun_serve(
        RerunServeConfig(
            run_id="run-a",
            s3_bucket="demo-bucket",
            s3_prefix="sim2real",
        ),
        kubeconfig=str(tmp_path / "kubeconfig"),
        kubectl=kubectl,
        wait_for_public_url=False,
    )

    assert result.rrd_s3_uri == _IMMUTABLE_RRD
    deployment = next(
        item for item in applied[0]["items"] if item["kind"] == "Deployment"
    )
    script = deployment["spec"]["template"]["spec"]["initContainers"][0]["command"][2]
    assert digest in script
    assert "sha256sum -c" in script


def test_monitor_rejects_committed_report_byte_mismatch() -> None:
    claimed = b'{"status":"expected"}\n'
    actual = b'{"status":"tampered"}\n'
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(report=claimed),
            _IMMUTABLE_REPORT: actual,
        }
    )

    with pytest.raises(PublicationConflict, match="bytes|journal"):
        monitor._load_publication_json(
            storage,
            "demo-bucket",
            "sim2real/run-a/reports/sim2real-report.json",
        )


def test_monitor_keeps_missing_unjournaled_report_optional() -> None:
    assert (
        monitor._load_publication_json(
            _Reader({}),
            "demo-bucket",
            "sim2real/run-a/reports/sim2real-report.json",
        )
        is None
    )


def test_regeneration_rejects_committed_report_byte_mismatch(tmp_path: Path) -> None:
    claimed = b'{"status":"expected"}\n'
    actual = b'{"status":"tampered"}\n'
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(report=claimed),
            _IMMUTABLE_REPORT: actual,
        }
    )
    destination = tmp_path / "reports" / "sim2real-report.json"

    with pytest.raises(regen.Sim2RealRerunRegenError, match="bytes|journal"):
        regen._download_regen_single_files(
            storage,
            f"{ROOT}/",
            tmp_path,
            {"reports/sim2real-report.json": destination},
        )
    assert not destination.exists()


def test_regeneration_rejects_committed_rrd_byte_mismatch(tmp_path: Path) -> None:
    claimed = b"expected-rrd"
    actual = b"tampered-rrd"
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(rrd=claimed),
            _IMMUTABLE_RRD: actual,
        }
    )

    destination = tmp_path / "sim2real.rrd"
    with pytest.raises(regen.Sim2RealRerunRegenError, match="bytes|journal"):
        regen.download_rrd_from_s3(
            _config(),
            dest_path=destination,
            client=storage,
        )
    assert not destination.exists()


def test_regeneration_rejects_committed_stage14_byte_mismatch(
    tmp_path: Path,
) -> None:
    report, _remote = _canonical_report_with_forged_stage4()
    objects = {_LOCK_URI: _complete_journal()}
    for stage, component in enumerate(report["component_records"], start=1):
        payload = (json.dumps(component, sort_keys=True) + "\n").encode()
        pointer_uri = f"{ROOT}/components/stage_{stage:02d}.json"
        if stage == 14:
            pointer_uri = _IMMUTABLE_STAGE14
        objects[pointer_uri] = payload
        objects[
            component_record_history_uri(
                ROOT,
                stage,
                component["content_sha256"],
            )
        ] = payload
    storage = _Reader(objects)
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(_gold()), encoding="utf-8")
    inputs = regen._RegenInputs(
        inner_evidence=_evidence(),
        heldout_report=_gold(),
        heldout_path=heldout_path,
        report_path=tmp_path / "report.json",
        report=report,
        current_decision=_decision(_gold()),
    )
    publication = resolve_committed_publication_snapshot(
        storage,
        f"{ROOT}/reports/sim2real-report.json",
    )

    with pytest.raises(regen.Sim2RealRerunRegenError, match="bytes|journal"):
        regen._validate_regen_component_authority(
            _config(),
            tmp_path,
            storage,
            inputs,
            verify_remote_authority=True,
            publication_snapshot=publication,
        )
    assert not (tmp_path / "component-authority" / "stage_14.json").exists()


def test_agent_runtime_rejects_committed_rrd_byte_mismatch() -> None:
    claimed = b"expected-rrd"
    actual = b"tampered-rrd"
    storage = _Reader(
        {
            _LOCK_URI: _complete_journal(rrd=claimed),
            _IMMUTABLE_RRD: actual,
        }
    )
    root_key = ROOT.removeprefix("s3://demo-bucket/")

    with pytest.raises(PublicationConflict, match="bytes|journal"):
        agent_stage_runtime._resolve_committed_artifact_key(
            storage,
            "demo-bucket",
            root_key,
            _CANONICAL_RRD.removeprefix("s3://demo-bucket/"),
        )


@pytest.mark.parametrize("publication_id", [".", ".."])
def test_publication_id_rejects_path_traversal(publication_id: str) -> None:
    assert valid_publication_id(publication_id) is False


def test_legacy_tree_uploader_cannot_overwrite_an_immutable_generation(
    tmp_path: Path,
) -> None:
    relative = Path("reports/generations/generation-a/sim2real.rrd")
    immutable_uri = f"{ROOT}/{relative.as_posix()}"
    storage = _StreamingObjectStore({immutable_uri: b"committed generation"})
    storage.upload_file = lambda path, uri: storage.objects.__setitem__(
        uri, Path(path).read_bytes()
    )
    candidate = tmp_path / relative
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"overwrite committed generation")

    with pytest.raises(PublicationConflict, match="immutable"):
        artifact_upload._upload_legacy_tree_without_reserved_aliases(
            storage,
            tmp_path,
            ROOT,
        )

    assert storage.objects[immutable_uri] == b"committed generation"


def test_rerun_serve_rejects_a_stale_explicit_generation() -> None:
    stale = f"{ROOT}/reports/generations/stale/sim2real.rrd"
    journal = _complete_journal()

    def get_object(*, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == "demo-bucket"
        assert Key.endswith("reports/.sim2real-publication.json")
        return {"Body": io.BytesIO(journal), "ETag": '"journal"'}

    with pytest.raises(RerunServeError, match="committed|generation"):
        _resolve_committed_rrd_uri(stale, get_object=get_object)


def test_regeneration_updates_every_publication_generation_identity(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "sim2real-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.e2e_report.v1",
                "publication_id": "old-generation",
                "rrd_uri": (f"{ROOT}/reports/generations/old-generation/sim2real.rrd"),
            }
        ),
        encoding="utf-8",
    )
    generation = f"{ROOT}/reports/generations/new-generation"

    regen._seal_regen_publication_report(
        report_path,
        publication_id="new-generation",
        rrd_uri=f"{generation}/sim2real.rrd",
        mcap_uri="",
        report_uri=f"{generation}/sim2real-report.json",
        journal_uri=f"{ROOT}/reports/.sim2real-publication.json",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["publication_id"] == "new-generation"
    assert report["publication_id"] == report["publication"]["generation"]


@pytest.mark.parametrize(
    ("products", "rows"),
    [
        (
            ["NVIDIA B200", "NVIDIA H100"],
            ["NVIDIA B200, GPU-test-0000"],
        ),
        (
            ["NVIDIA B200"],
            ["NVIDIA H100, GPU-test-0000"],
        ),
        (
            ["NVIDIA B200"],
            ["not-a-device-row"],
        ),
    ],
)
def test_stage4_rejects_malformed_per_lane_gpu_evidence(
    products: list[str],
    rows: list[str],
) -> None:
    stage4 = _component(4)
    proof = stage4["artifacts"]["shard_provenance"][0]
    lane = stage4["artifacts"]["lane_records"][0]
    proof["provenance"]["gpu_products"] = products
    proof["provenance"]["gpu_rows"] = rows
    lane["artifacts"]["gpu_products"] = products
    lane["artifacts"]["gpu_rows"] = rows
    _rehash(lane)

    with pytest.raises(ValueError, match="GPU|gpu|provenance"):
        component_authority.validate_stage4_parallel_inputs(
            ROOT,
            shard_count=stage4["artifacts"]["shard_count"],
            shard_provenance=stage4["artifacts"]["shard_provenance"],
            lane_records=stage4["artifacts"]["lane_records"],
            expected_source_sha=SOURCE_SHA,
        )


def test_stage4_lane_publisher_rejects_mismatched_gpu_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = dict(_component(4)["artifacts"]["lane_records"][0]["artifacts"])
    provenance["gpu_rows"] = ["NVIDIA B200, GPU-test-0000"]
    uploads: list[str] = []

    class Storage:
        def upload_file(self, _path: str, uri: str) -> str:
            uploads.append(uri)
            return uri

    monkeypatch.setattr(workflow_io, "storage", Storage)

    with pytest.raises(ValueError, match="execution provenance is invalid"):
        workflow_io.publish_component_lane_record(
            root_uri=ROOT,
            stage=4,
            lane="shard-00000",
            evidence="generated declared shard",
            artifacts={"shard_index": 0},
            execution_provenance=provenance,
        )
    assert uploads == []


def test_complete_journal_claim_is_bound_to_expected_digest() -> None:
    claimed = b"committed expected bytes"
    journal = json.loads(_complete_journal(rrd=claimed))
    target = next(item for item in journal["objects"] if item["uri"] == _CANONICAL_RRD)

    assert target["sha256"] == hashlib.sha256(claimed).hexdigest()
    assert target["size_bytes"] == len(claimed)
