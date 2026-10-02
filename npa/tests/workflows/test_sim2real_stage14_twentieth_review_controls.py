from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from npa.agent_backend import canonical_mcap
from npa.orchestration.npa_workflow.artifact_load import (
    ArtifactLoadError,
    discover_final_rerun_artifact,
)
from npa.workflows import data_factory_viz
from npa.workflows import rerun_serve
from npa.workflows.sim2real import (
    checkpoint_selection,
    component_authority,
    legacy_heldout,
    publication,
    stage14_finalize,
    workflow_io,
)
from npa.workflows.sim2real.decision_authority import validate_stage11_decision
from npa.workflows.sim2real.publication import PublicationConflict
from npa.workflows.sim2real.viz_contract import heldout_policy_metadata
import npa.workflows.sim2real_rerun_regen as sim2real_rerun_regen
import npa.workflows.sim2real_viz as sim2real_viz
from npa.cli.agent_stage_runtime import _public_workflow_command
from tests.workflows.test_sim2real_stage14_nineteenth_review_controls import (
    _CANONICAL_RRD,
    _IMMUTABLE_RRD,
    _LOCK_URI,
    _complete_journal,
)
from tests.workflows.test_sim2real_stage14_thirteenth_review_controls import (
    ROOT,
    SOURCE_SHA,
    _component,
    _config,
    _decision,
    _evidence,
    _gold,
    _rehash,
)
from tests.workflows.test_sim2real_stage14_seventeenth_review_controls import (
    _complete_journal as _journal_with_state,
)


pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")


class _LegacyRaceStore:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.journal_reads = 0

    def read_small_bytes_with_etag(
        self, uri: str, *, max_bytes: int = 1024 * 1024
    ) -> tuple[bytes, str] | None:
        del max_bytes
        if uri == _LOCK_URI:
            self.journal_reads += 1
            if self.journal_reads == 1:
                return None
            journal = _journal_with_state(state="publishing", rrd=self.payload)
            return journal, "publishing-etag"
        if uri == _CANONICAL_RRD:
            return self.payload, "alias-etag"
        return None


def test_legacy_alias_read_rechecks_first_journal_publication() -> None:
    store = _LegacyRaceStore(b"new-generation-before-commit")
    snapshot = publication.resolve_committed_publication_snapshot(store, _CANONICAL_RRD)

    with pytest.raises(PublicationConflict, match="journal|publication"):
        publication.read_verified_committed_publication_bytes(
            store,
            snapshot,
            _CANONICAL_RRD,
        )

    assert store.journal_reads == 2


def test_run_scoped_regeneration_requires_every_iteration_uri(tmp_path: Path) -> None:
    record = {
        "iteration": 1,
        "actions_dir": str(tmp_path / "actions/train/outer-01/iter-01"),
        "vlm_eval_dir": str(tmp_path / "vlm_eval/train/outer-01/iter-01/evaluations"),
        "signal_dir": str(tmp_path / "vlm_eval/train/outer-01/iter-01/signals"),
    }

    for spec in sim2real_rerun_regen._iteration_rewrite_specs(record, 1):
        with pytest.raises(
            sim2real_rerun_regen.Sim2RealRerunRegenError,
            match="required for run-scoped regeneration",
        ):
            sim2real_rerun_regen._rewrite_iteration_path(
                record,
                tmp_path,
                "s3://unit/runs/current",
                spec,
            )


def test_canonical_mcap_maps_conditional_write_race_to_publication_conflict(
    tmp_path: Path,
) -> None:
    class Store:
        def get_object(self, **_kwargs):
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )

        def head_object(self, **_kwargs):
            raise ClientError(
                {"Error": {"Code": "404", "Message": "missing"}},
                "HeadObject",
            )

        def put_object(self, **_kwargs):
            raise ClientError(
                {
                    "Error": {
                        "Code": "PreconditionFailed",
                        "Message": "concurrent writer",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )

    source = tmp_path / "canonical.mcap"
    source.write_bytes(b"mcap")
    with pytest.raises(PublicationConflict, match="concurrently replaced"):
        canonical_mcap._put_unjournaled_canonical_mcap(
            Store(),
            bucket="demo-bucket",
            canonical_key="sim2real/run-a/reports/sim2real.mcap",
            local_path=source,
        )


def test_rerun_verification_has_no_metadata_only_digest_tier() -> None:
    source = inspect.getsource(rerun_serve.verify_rrd_exists_on_s3)
    assert "npa-sha256" not in source


def test_heldout_normalization_preserves_evaluation_attempt_identity() -> None:
    attempt = "gold_heldout-outer-01-attempt-" + "a" * 32
    report = legacy_heldout._normalize_heldout_report(
        {
            "per_env": [{"env_id": "env-1", "score": 1.0, "success": True}],
            "evaluation_attempt_tag": attempt,
        },
        config=_config(),
        outer_iteration=1,
        inner_evidence_uri=f"{ROOT}/inner_loop/outer-01/evidence.json",
        invocation={"output_uri": f"{ROOT}/byo-eval/result.json"},
    )

    assert report["evaluation_attempt_tag"] == attempt


def test_checkpoint_selection_rejects_training_iteration_disagreement() -> None:
    evidence = _evidence()
    evidence["checkpoint_selection"]["training_iteration"] = 999

    with pytest.raises(ValueError, match="training_iteration|training iteration"):
        checkpoint_selection.resolve_run_scoped_checkpoint(
            evidence,
            run_root=ROOT,
            run_id="run-a",
        )


def test_stage11_requires_exact_downloaded_report_digest() -> None:
    report = _gold()
    with pytest.raises(TypeError, match="gold_report_bytes_sha256"):
        validate_stage11_decision(
            _decision(report),
            run_id="run-a",
            root=ROOT,
            outer_iteration=1,
            gold_report=report,
            checkpoint_uri=report["policy_checkpoint_uri"],
            expected_threshold=0.5,
            expected_early_exit=False,
        )


def test_stage14_policy_gate_requires_run_scope_authority() -> None:
    report = _gold()
    with pytest.raises(TypeError, match="run_root"):
        stage14_finalize._stage14_policy_metadata(
            _evidence(),
            _decision(report),
            report,
            expected_gold_report_sha256=component_authority.gold_report_sha256(report),
        )


def test_component_reader_rejects_present_invalid_gpu_evidence() -> None:
    record = _component(10)
    record["artifacts"]["gpu_rows"] = [
        "NVIDIA H100, GPU-duplicate",
        "NVIDIA H100, GPU-duplicate",
    ]
    record["artifacts"]["gpu_products"] = ["NVIDIA H100", "NVIDIA H100"]
    _rehash(record)
    name, tier, required = component_authority.COMPONENT_CONTRACTS[10]

    with pytest.raises(ValueError, match="provenance"):
        workflow_io.validate_component_record(
            record,
            expected_stage=10,
            expected_name=name,
            expected_tier=tier,
            required_artifacts=required,
            expected_source_sha=SOURCE_SHA,
        )


def test_policy_markdown_refuses_disagreed_report_declaration() -> None:
    report = _gold()
    metadata = heldout_policy_metadata(report)
    assert metadata["heldout_policy_learned_actor_only"] is True
    metadata["heldout_policy_actor_is_learned"] = False

    markdown = sim2real_viz._policy_access_markdown(metadata, report)

    assert "complete learned-actor-only contract is proven" not in markdown
    assert "not proven" in markdown
    assert "disagree" in markdown


def test_public_workflow_command_removes_signed_url_suffixes() -> None:
    command = _public_workflow_command(
        [
            "tool",
            "--input",
            "https://storage.example/object?X-Amz-Signature=secret#fragment",
        ]
    )

    assert "secret" not in command
    assert "X-Amz-" not in command
    assert "#fragment" not in command
    assert "https://storage.example/object" in command


def test_data_factory_reads_journal_even_when_inventory_omits_it() -> None:
    class Store:
        def read_small_bytes_with_etag(
            self, uri: str, *, max_bytes: int = 1024 * 1024
        ) -> tuple[bytes, str] | None:
            assert max_bytes > 0
            if uri == _LOCK_URI:
                return _complete_journal(), "journal-etag"
            return None

    resolved, snapshot = data_factory_viz._resolve_existing_output_publication(
        Store(),
        _CANONICAL_RRD,
        inventory_keys={_CANONICAL_RRD.removeprefix("s3://demo-bucket/")},
    )

    assert resolved == _IMMUTABLE_RRD
    assert snapshot is not None and snapshot.journaled


def test_missing_legacy_final_rrd_is_a_definite_absence() -> None:
    class Store:
        s3: object

        def __init__(self) -> None:
            self.s3 = self

        def read_small_bytes_with_etag(
            self, _uri: str, *, max_bytes: int = 1024 * 1024
        ) -> None:
            del max_bytes
            return None

        def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "missing"}},
                "HeadObject",
            )

    with pytest.raises(ArtifactLoadError, match=r"no \.rrd artifact exists.*s3://"):
        discover_final_rerun_artifact(ROOT, client=Store())


def test_rrd_semantic_identity_binds_text_documents(tmp_path: Path) -> None:
    rr = pytest.importorskip("rerun")

    def write(path: Path, body: str) -> None:
        recording = rr.RecordingStream("stage14-review", recording_id="same-run")
        recording.save(str(path))
        rr.log(
            "pipeline/report",
            rr.TextDocument(body, media_type="text/markdown"),
            static=True,
            recording=recording,
        )
        recording.flush()
        recording.disconnect()

    previous = tmp_path / "previous.rrd"
    current = tmp_path / "current.rrd"
    equivalent = tmp_path / "equivalent.rrd"
    write(previous, "OLD PIPELINE REPORT")
    write(current, "NEW PIPELINE REPORT")
    write(equivalent, "NEW PIPELINE REPORT")

    assert data_factory_viz._rrd_semantic_sha256(
        previous
    ) != data_factory_viz._rrd_semantic_sha256(current)
    assert data_factory_viz._rrd_semantic_sha256(
        current
    ) == data_factory_viz._rrd_semantic_sha256(equivalent)
    data_factory_viz._verify_existing_rrd_semantics(current, equivalent)
    with pytest.raises(data_factory_viz.DataFactoryVizError, match="all current"):
        data_factory_viz._verify_existing_rrd_semantics(previous, current)
