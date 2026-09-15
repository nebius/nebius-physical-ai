"""Tests for the provider-neutral curation-decision contract."""

from __future__ import annotations

import pytest

from npa.workflows.paidf_campaign import (
    CANDIDATE_MANIFEST_SCHEMA,
    DECISION_MANIFEST_SCHEMA,
    RECONCILIATION_SCHEMA,
    PaidfContractError,
    build_candidate,
    reconcile_decisions,
    reconcile_provider_export,
    validate_candidate_manifest,
    validate_decision_manifest,
)


DECIDED_AT = "2026-09-01T00:00:00Z"


def _candidate(candidate_id: str = "cand-001") -> dict:
    return build_candidate(
        candidate_id=candidate_id,
        source_episode_id="episode_0000",
        source_episode_index=0,
        camera_key="observation.images.workspace",
        variant_id=f"var-{candidate_id}",
        output_sha256="a" * 64,
        source_sha256="b" * 64,
        evaluation={"checks": {"media_integrity": "pass"}},
    )


def _manifest() -> dict:
    return {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "candidates": [_candidate("cand-001"), _candidate("cand-002")],
    }


def _decisions(manifest: dict, states: dict[str, str]) -> dict:
    return {
        "schema": DECISION_MANIFEST_SCHEMA,
        "provider": {"name": "human-curator", "role": "curation"},
        "decisions": [
            {
                "candidate_id": candidate_id,
                "decision": state,
                "evidence": {"export": "final-decision-export.json"},
                "reason": f"automated {state}",
                "decided_at": DECIDED_AT,
            }
            for candidate_id, state in states.items()
        ],
    }


class TestCandidateManifest:
    def test_valid_manifest(self) -> None:
        manifest = _manifest()
        assert validate_candidate_manifest(manifest) == manifest

    def test_duplicate_candidate_id_fails(self) -> None:
        manifest = {
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidates": [_candidate("cand-001"), _candidate("cand-001")],
        }
        with pytest.raises(PaidfContractError, match="duplicate candidate_id"):
            validate_candidate_manifest(manifest)

    def test_missing_variant_hash_fails(self) -> None:
        candidate = _candidate()
        del candidate["variant"]["output_sha256"]
        manifest = {"schema": CANDIDATE_MANIFEST_SCHEMA, "candidates": [candidate]}
        with pytest.raises(PaidfContractError, match="output_sha256"):
            validate_candidate_manifest(manifest)

    @pytest.mark.parametrize("bad_index", ["0", 1.5, -1, True, None])
    def test_non_integer_episode_index_fails(self, bad_index: object) -> None:
        candidate = _candidate()
        candidate["source_episode_index"] = bad_index
        manifest = {"schema": CANDIDATE_MANIFEST_SCHEMA, "candidates": [candidate]}
        with pytest.raises(PaidfContractError, match="source_episode_index"):
            validate_candidate_manifest(manifest)

    @pytest.mark.parametrize(
        "reserved_field", ["variant_id", "output_sha256", "source_sha256"]
    )
    def test_variant_extras_cannot_replace_identity_fields(
        self, reserved_field: str
    ) -> None:
        with pytest.raises(PaidfContractError, match="cannot replace"):
            build_candidate(
                candidate_id="cand-001",
                source_episode_id="episode_0000",
                source_episode_index=0,
                camera_key="observation.images.workspace",
                variant_id="var-cand-001",
                output_sha256="a" * 64,
                source_sha256="b" * 64,
                evaluation={},
                variant_extras={reserved_field: "same-or-different"},
            )

    def test_variant_extras_preserve_nonidentity_metadata(self) -> None:
        candidate = build_candidate(
            candidate_id="cand-001",
            source_episode_id="episode_0000",
            source_episode_index=0,
            camera_key="observation.images.workspace",
            variant_id="var-cand-001",
            output_sha256="a" * 64,
            source_sha256="b" * 64,
            evaluation={},
            variant_extras={"video_uri": "s3://bucket/cand-001.mp4"},
        )
        assert candidate["variant"]["video_uri"] == "s3://bucket/cand-001.mp4"
        assert candidate["variant"]["output_sha256"] == "a" * 64


class TestDecisionManifest:
    def test_full_coverage_passes(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "reject"}
        )
        assert validate_decision_manifest(decisions, candidates=manifest)

    def test_review_is_a_valid_state(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "review", "cand-002": "accept"}
        )
        validate_decision_manifest(decisions, candidates=manifest)

    def test_missing_decision_fails_closed(self) -> None:
        manifest = _manifest()
        decisions = _decisions(manifest, {"cand-001": "accept"})
        with pytest.raises(PaidfContractError, match="missing for candidates"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_unknown_candidate_fails(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest,
            {"cand-001": "accept", "cand-002": "accept", "cand-999": "accept"},
        )
        with pytest.raises(PaidfContractError, match="unknown candidate"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_duplicate_decision_fails(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        decisions["decisions"].append(dict(decisions["decisions"][0]))
        with pytest.raises(PaidfContractError, match="more than one decision"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_invalid_state_fails(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "maybe", "cand-002": "accept"}
        )
        with pytest.raises(PaidfContractError, match="accept, reject, review"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_provider_must_have_exactly_name_and_role(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        del decisions["provider"]["role"]
        with pytest.raises(PaidfContractError, match="exactly name and role"):
            validate_decision_manifest(decisions, candidates=manifest)

        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        decisions["provider"]["mode"] = "automated"
        with pytest.raises(PaidfContractError, match="exactly name and role"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_provider_role_must_be_curation_or_evaluation(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        decisions["provider"]["role"] = "simulation"
        with pytest.raises(PaidfContractError, match="curation or evaluation"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_missing_decided_at_fails(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        del decisions["decisions"][0]["decided_at"]
        with pytest.raises(PaidfContractError, match="exactly"):
            validate_decision_manifest(decisions, candidates=manifest)

    @pytest.mark.parametrize(
        "bad_timestamp",
        [
            "2026-09-01 00:00:00Z",
            "2026-09-01T00:00Z",
            "2026-9-01T00:00:00Z",
            "not-a-timestamp",
            None,
        ],
    )
    def test_malformed_decided_at_fails(self, bad_timestamp: object) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        decisions["decisions"][0]["decided_at"] = bad_timestamp
        with pytest.raises(PaidfContractError, match="canonical UTC seconds"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_non_calendar_decided_at_fails(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        decisions["decisions"][0]["decided_at"] = "2026-02-30T00:00:00Z"
        with pytest.raises(PaidfContractError, match="real calendar time"):
            validate_decision_manifest(decisions, candidates=manifest)

    def test_incomplete_decision_fields_fail(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "accept"}
        )
        del decisions["decisions"][0]["reason"]
        with pytest.raises(PaidfContractError, match="must contain exactly"):
            validate_decision_manifest(decisions, candidates=manifest)


class TestReconcileDecisions:
    def test_keep_drop_and_review_exclusion(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "review"}
        )
        result = reconcile_decisions(manifest, decisions)
        assert result["schema"] == RECONCILIATION_SCHEMA
        assert [entry["candidate_id"] for entry in result["keep"]] == ["cand-001"]
        assert [entry["candidate_id"] for entry in result["drop"]] == ["cand-002"]
        assert result["totals"]["unresolved_review"] == 1
        assert result["unresolved_review_candidate_ids"] == ["cand-002"]
        assert result["totals"]["identity_gaps"] == 0

    def test_reject_excluded_from_training(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "reject", "cand-002": "reject"}
        )
        result = reconcile_decisions(manifest, decisions)
        assert result["keep"] == []
        assert result["totals"]["unresolved_review"] == 0

    def test_reconciliation_carries_the_decision_provider(self) -> None:
        manifest = _manifest()
        decisions = _decisions(
            manifest, {"cand-001": "accept", "cand-002": "reject"}
        )
        result = reconcile_decisions(manifest, decisions)
        assert result["provider"] == decisions["provider"]


class TestReconcileProviderExport:
    def test_exact_join(self) -> None:
        manifest = _manifest()
        export = [
            {"external_id": "cand-001", "data_hash": "dh-1", "decision": "accept"},
            {"external_id": "cand-002", "data_hash": "dh-2", "decision": "review"},
        ]
        result = reconcile_provider_export(manifest, export)
        assert result["totals"]["identity_gaps"] == 0
        assert len(result["joined"]) == 2

    def test_orphan_fails(self) -> None:
        manifest = _manifest()
        export = [
            {"external_id": "cand-001", "decision": "accept"},
            {"external_id": "cand-002", "decision": "reject"},
            {"external_id": "ghost", "decision": "accept"},
        ]
        with pytest.raises(PaidfContractError, match="orphan"):
            reconcile_provider_export(manifest, export)

    def test_missing_export_item_fails(self) -> None:
        manifest = _manifest()
        export = [{"external_id": "cand-001", "decision": "accept"}]
        with pytest.raises(PaidfContractError, match="missing candidates"):
            reconcile_provider_export(manifest, export)

    def test_duplicate_export_item_fails(self) -> None:
        manifest = _manifest()
        export = [
            {"external_id": "cand-001", "decision": "accept"},
            {"external_id": "cand-001", "decision": "accept"},
            {"external_id": "cand-002", "decision": "reject"},
        ]
        with pytest.raises(PaidfContractError, match="duplicate"):
            reconcile_provider_export(manifest, export)

    def test_identity_only_row_without_decision_fails(self) -> None:
        manifest = _manifest()
        export = [
            {"external_id": "cand-001", "data_hash": "dh-1"},
            {"external_id": "cand-002", "data_hash": "dh-2"},
        ]
        with pytest.raises(PaidfContractError, match="non-null accept, reject,"):
            reconcile_provider_export(manifest, export)

    def test_invalid_decision_state_fails(self) -> None:
        manifest = _manifest()
        export = [
            {"external_id": "cand-001", "decision": "accept"},
            {"external_id": "cand-002", "decision": "deferred"},
        ]
        with pytest.raises(PaidfContractError, match="non-null accept, reject,"):
            reconcile_provider_export(manifest, export)

    def test_malformed_candidate_manifest_fails_before_join(self) -> None:
        malformed = {
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidates": [{"candidate_id": "cand-001"}],
        }
        export = [{"external_id": "cand-001", "decision": "accept"}]
        with pytest.raises(PaidfContractError, match="source_episode_id"):
            reconcile_provider_export(malformed, export)
