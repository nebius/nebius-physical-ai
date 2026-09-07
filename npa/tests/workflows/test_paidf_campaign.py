"""Contract tests for reusable PAIDF campaigns and partner overlays."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from npa.workflows.paidf_campaign import (
    BASE_STAGES,
    FROZEN_SOURCE_MANIFEST_SCHEMA,
    PaidfContractError,
    accepted_replacements_stage,
    build_antioch_input,
    build_base_campaign,
    build_candidates_stage,
    build_decisions_stage,
    build_execution_receipt,
    build_partner_input,
    build_partner_result,
    canonical_digest,
    freeze_base_campaign_stage,
    load_frozen_source_manifest,
    reconcile_decisions,
    validate_campaign,
    validate_partner_input,
    validate_partner_result,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

DECIDED_AT = "2026-09-01T00:00:00Z"


def _artifact(name: str, digest: str) -> dict[str, str]:
    return {
        "uri": f"s3://paidf-proof/campaigns/franka-001/{name}.json",
        "sha256": digest,
        "schema": f"npa.paidf.{name}.v1",
    }


def _campaign() -> dict:
    return {
        "schema": "npa.paidf.campaign.v1",
        "campaign_id": "franka-001",
        "artifacts": {
            "source": _artifact("source", SHA_A),
            "generation": _artifact("generation", SHA_B),
            "evaluation": _artifact("evaluation", SHA_C),
        },
        "stage_fingerprints": {
            "source": SHA_A,
            "generation": SHA_B,
            "evaluation": SHA_C,
        },
    }


def _antioch_input(campaign: dict | None = None) -> dict:
    return build_antioch_input(
        campaign or _campaign(),
        campaign_manifest_uri=(
            "s3://paidf-proof/campaigns/franka-001/campaign.json"
        ),
        run_id="antioch-001",
        provider_config={
            "uri": "s3://paidf-proof/config/antioch-001.json",
            "sha256": "d" * 64,
            "schema": "npa.paidf.antioch-config.v1",
        },
        credential_refs=("secret://antioch/workshop",),
        output_prefix="s3://paidf-proof/derivatives/antioch/antioch-001/",
    )


def test_campaign_validation_and_digest_are_stable() -> None:
    campaign = _campaign()
    reordered = {
        "stage_fingerprints": campaign["stage_fingerprints"],
        "artifacts": campaign["artifacts"],
        "campaign_id": campaign["campaign_id"],
        "schema": campaign["schema"],
    }

    assert validate_campaign(campaign) == campaign
    assert canonical_digest(reordered) == canonical_digest(campaign)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uri", "https://example.com/source.json"),
        ("uri", "s3://paidf-proof/campaigns/franka-001/"),
        ("sha256", "not-a-digest"),
        ("schema", ""),
    ],
)
def test_campaign_rejects_invalid_artifact_references(field: str, value: str) -> None:
    campaign = _campaign()
    campaign["artifacts"]["source"][field] = value

    with pytest.raises(PaidfContractError):
        validate_campaign(campaign)


def test_campaign_rejects_extra_base_stage_keys() -> None:
    campaign = _campaign()
    campaign["artifacts"]["regeneration"] = _artifact("regeneration", SHA_A)
    with pytest.raises(PaidfContractError, match="exactly the base stages"):
        validate_campaign(campaign)

    campaign = _campaign()
    campaign["stage_fingerprints"]["regeneration"] = SHA_A
    with pytest.raises(PaidfContractError, match="exactly the base stages"):
        validate_campaign(campaign)


def test_campaign_rejects_artifact_fingerprint_digest_divergence() -> None:
    campaign = _campaign()
    campaign["artifacts"]["generation"]["sha256"] = "e" * 64

    with pytest.raises(PaidfContractError, match="does not match its stage fingerprint"):
        validate_campaign(campaign)


def test_antioch_overlay_reuses_base_and_never_executes_generation() -> None:
    partner_input = _antioch_input()

    assert partner_input["source_mode"] == "read-only"
    assert partner_input["reuse"]["stages"] == list(BASE_STAGES)
    assert set(partner_input["execute"]["stages"]).isdisjoint(BASE_STAGES)
    assert partner_input["provider"] == {
        "name": "antioch",
        "role": "simulation",
    }


def test_partner_input_rejects_changed_campaign_or_reused_artifact() -> None:
    campaign = _campaign()
    partner_input = _antioch_input(campaign)
    changed_campaign = copy.deepcopy(campaign)
    changed_campaign["stage_fingerprints"]["evaluation"] = "e" * 64

    with pytest.raises(PaidfContractError, match="digest"):
        validate_partner_input(partner_input, campaign=changed_campaign)

    altered_input = copy.deepcopy(partner_input)
    altered_input["reuse"]["artifacts"]["source"]["sha256"] = "f" * 64
    with pytest.raises(PaidfContractError, match="changed the base artifact"):
        validate_partner_input(altered_input, campaign=campaign)


def test_partner_input_rejects_base_stage_execution_and_plaintext_credentials() -> None:
    campaign = _campaign()
    partner_input = _antioch_input(campaign)
    partner_input["execute"]["stages"].append("generation")

    with pytest.raises(PaidfContractError, match="immutable base stages"):
        validate_partner_input(partner_input, campaign=campaign)

    partner_input = _antioch_input(campaign)
    partner_input["credential_refs"] = ["a-secret-value"]
    with pytest.raises(PaidfContractError, match="reference"):
        validate_partner_input(partner_input, campaign=campaign)


def test_partner_input_rejects_output_nested_inside_base_campaign() -> None:
    campaign = _campaign()
    partner_input = _antioch_input(campaign)
    partner_input["output_prefix"] = (
        "s3://paidf-proof/campaigns/franka-001/derivatives/antioch-001/"
    )

    with pytest.raises(PaidfContractError, match="separate"):
        validate_partner_input(partner_input, campaign=campaign)


def test_partner_builder_reports_unknown_reuse_as_a_contract_error() -> None:
    with pytest.raises(PaidfContractError, match="unknown base stages"):
        build_partner_input(
            _campaign(),
            campaign_manifest_uri=(
                "s3://paidf-proof/campaigns/franka-001/campaign.json"
            ),
            run_id="provider-001",
            provider="another-simulator",
            role="simulation",
            provider_config={
                "uri": "s3://paidf-proof/config/provider-001.json",
                "sha256": "d" * 64,
                "schema": "npa.paidf.provider-config.v1",
            },
            output_prefix="s3://paidf-proof/derivatives/provider/provider-001/",
            executed_stages=("partner-simulation",),
            reused_stages=("source", "unknown-stage"),
        )


def test_receipt_and_result_account_for_stages_and_isolate_outputs() -> None:
    partner_input = _antioch_input()
    stage_results = {
        stage: "completed" for stage in partner_input["execute"]["stages"]
    }
    receipt = build_execution_receipt(
        partner_input,
        status="completed",
        stage_results=stage_results,
    )
    result = build_partner_result(
        partner_input,
        receipt,
        artifacts={
            "result-manifest": {
                "uri": (
                    "s3://paidf-proof/derivatives/antioch/antioch-001/"
                    "result-manifest.json"
                ),
                "sha256": "1" * 64,
                "schema": "npa.paidf.antioch-result.v1",
            }
        },
    )

    assert result["status"] == "completed"
    assert result["source_mutated"] is False
    assert validate_partner_result(
        result,
        partner_input=partner_input,
        execution_receipt=receipt,
    ) == result

    escaped = copy.deepcopy(result)
    escaped["artifacts"]["result-manifest"]["uri"] = (
        "s3://paidf-proof/campaigns/franka-001/replaced.json"
    )
    with pytest.raises(PaidfContractError, match="outside the output prefix"):
        validate_partner_result(
            escaped,
            partner_input=partner_input,
            execution_receipt=receipt,
        )


def test_completed_receipt_requires_every_stage_to_complete() -> None:
    partner_input = _antioch_input()
    stage_results = {
        stage: "completed" for stage in partner_input["execute"]["stages"]
    }
    stage_results["comparison"] = "skipped"

    with pytest.raises(PaidfContractError, match="every stage"):
        build_execution_receipt(
            partner_input,
            status="completed",
            stage_results=stage_results,
        )


# ── Phase 0 integrity-correction builders (candidates, decisions, accepted
#    replacements) ──────────────────────────────────────────────────────────


def _augment_manifest(tmp_path: Path, video_names: list[str]) -> tuple[Path, list[Path]]:
    """A canonical Cosmos 3 augment manifest over real local video files."""

    videos = []
    variants = []
    total_bytes = 0
    augment_root = tmp_path / "cosmos_augmented"
    for index, name in enumerate(video_names):
        clip_dir = augment_root / name
        clip_dir.mkdir(parents=True, exist_ok=True)
        video = clip_dir / "augmented_video.mp4"
        video.write_bytes(f"variant-bytes-{name}".encode())
        videos.append(video)
        total_bytes += video.stat().st_size
        variants.append(
            {
                "clip": name,
                "augmented_video_uri": str(video),
                "video_bytes": video.stat().st_size,
                "frame_count": 10 + index,
                "seed": 17 + index,
                "guidance": 5.0,
                "steps": 24,
                "variables": {"look": name},
                "motion_preservation": None,
            }
        )
    manifest = {
        "schema": "npa.paidf.cosmos3.augment.v1",
        "engine": "nvidia-cosmos/cosmos-framework",
        "status": "executed",
        "mode": "video2video",
        "input_conditioned": True,
        "input_conditioning": "source-video",
        "conditioned_input": "source.mp4",
        "guardrails": True,
        "weights_baked": False,
        "lineage": {"input_provenance_uri": str(tmp_path / "provenance.json")},
        "model": "Cosmos3-Nano",
        "variant_count": len(variants),
        "video_bytes": total_bytes,
        "frame_count": sum(v["frame_count"] for v in variants),
        "variants": variants,
    }
    path = tmp_path / "augment-manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path, videos


def _builder_fixtures(tmp_path: Path, *, video_names: list[str]) -> dict[str, Path]:
    augment, _videos = _augment_manifest(tmp_path, video_names)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema": "npa.paidf.cosmos3.input.v1",
                "status": "prepared",
                "source_kind": "lerobot_dataset",
                "episode": 3,
                "camera": "observation.images.workspace",
                "staged_video_uri": "s3://bucket/run/input/source.mp4",
                "conditioned_input": "source.mp4",
                "sha256": "c" * 64,
                "video_bytes": 123,
                "frame_count": 8,
                "run_id": "run-1",
            }
        )
    )
    clips = []
    for index, name in enumerate(video_names):
        clips.append(
            {
                "clip_id": name,
                "score": 0.9 - 0.2 * index,
                "passed": index == 0,
                "input_conditioned": True,
                "status": "completed",
            }
        )
    evaluator = tmp_path / "cosmos_evaluator.json"
    evaluator.write_text(
        json.dumps(
            {
                "schema": "npa.cosmos_evaluator.report.v1",
                "status": "completed",
                "score": 0.9,
                "passed": True,
                "clips": clips,
            }
        )
    )
    disposition = tmp_path / "quality_disposition.json"
    disposition.write_text(
        json.dumps(
            {
                "schema": "npa.data_factory.quality_disposition.v1",
                "quality_status": "accepted",
                "decision": "promote_checkpoint",
                "evaluator_status": "completed",
                "score": 0.9,
                "threshold": 0.75,
                "hard_checks_passed": True,
            }
        )
    )
    return {
        "augment": augment,
        "provenance": provenance,
        "evaluator": evaluator,
        "disposition": disposition,
    }


class TestBuildCandidatesStage:
    def test_builds_exact_identity_candidates_from_real_evidence(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a", "clip-b"])
        out = tmp_path / "candidates.json"
        manifest = build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(out),
        )
        assert [c["candidate_id"] for c in manifest["candidates"]] == ["clip-a", "clip-b"]
        first = manifest["candidates"][0]
        assert first["source_episode_id"] == "episode_000003"
        assert first["source_episode_index"] == 3
        assert first["camera_key"] == "observation.images.workspace"
        video_bytes = Path(str(first["variant"]["video_uri"])).read_bytes()
        assert first["variant"]["output_sha256"] == hashlib.sha256(video_bytes).hexdigest()
        assert first["evaluation"]["passed"] is True
        stored = json.loads(out.read_text())
        assert stored["schema"] == "npa.paidf.candidates.v1"

    def test_fails_closed_without_evaluator_evidence_for_a_variant(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a", "clip-b"])
        report = json.loads(fixtures["evaluator"].read_text())
        report["clips"] = report["clips"][:1]
        fixtures["evaluator"].write_text(json.dumps(report))
        with pytest.raises(PaidfContractError, match="no evidence for variant"):
            build_candidates_stage(
                str(fixtures["augment"]),
                str(fixtures["provenance"]),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(tmp_path / "candidates.json"),
            )

    @pytest.mark.parametrize("bad_episode", ["3", 3.0, True, False, -1])
    def test_rejects_coerced_provenance_episode_identity(
        self, tmp_path: Path, bad_episode: object
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        provenance = json.loads(fixtures["provenance"].read_text())
        provenance["episode"] = bad_episode
        fixtures["provenance"].write_text(json.dumps(provenance))
        output = tmp_path / "candidates.json"
        with pytest.raises(PaidfContractError, match="input provenance.episode"):
            build_candidates_stage(
                str(fixtures["augment"]),
                str(fixtures["provenance"]),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(output),
            )
        assert not output.exists()

    @pytest.mark.parametrize("conflicting", [False, True])
    def test_rejects_duplicate_evaluator_clip_ids(
        self, tmp_path: Path, conflicting: bool
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        evaluator = json.loads(fixtures["evaluator"].read_text())
        duplicate = dict(evaluator["clips"][0])
        if conflicting:
            duplicate["passed"] = not duplicate["passed"]
        evaluator["clips"].append(duplicate)
        fixtures["evaluator"].write_text(json.dumps(evaluator))
        output = tmp_path / "candidates.json"
        with pytest.raises(PaidfContractError, match="duplicate clip_id"):
            build_candidates_stage(
                str(fixtures["augment"]),
                str(fixtures["provenance"]),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(output),
            )
        assert not output.exists()

    def test_rejects_malformed_or_out_of_set_evaluator_rows(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        evaluator = json.loads(fixtures["evaluator"].read_text())
        evaluator["clips"].append("not-an-object")
        fixtures["evaluator"].write_text(json.dumps(evaluator))
        with pytest.raises(PaidfContractError, match="must be an object"):
            build_candidates_stage(
                str(fixtures["augment"]),
                str(fixtures["provenance"]),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(tmp_path / "malformed.json"),
            )

        evaluator["clips"][-1] = {
            "clip_id": "clip-outside",
            "status": "completed",
            "passed": True,
        }
        fixtures["evaluator"].write_text(json.dumps(evaluator))
        with pytest.raises(PaidfContractError, match="outside the committed augment set"):
            build_candidates_stage(
                str(fixtures["augment"]),
                str(fixtures["provenance"]),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(tmp_path / "outside.json"),
            )


class TestBuildDecisionsStage:
    def test_decisions_carry_the_fixed_provider_and_caller_timestamp(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        decisions = build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        assert decisions["provider"] == {"name": "cosmos-evaluator", "role": "evaluation"}
        assert all(d["decided_at"] == DECIDED_AT for d in decisions["decisions"])

    def test_requires_a_caller_supplied_decided_at(self, tmp_path: Path) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        with pytest.raises(TypeError):
            build_decisions_stage(  # type: ignore[call-arg]
                str(tmp_path / "candidates.json"),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(tmp_path / "decisions.json"),
            )

    def test_uncertain_evidence_routes_to_review(self, tmp_path: Path) -> None:
        """A candidate whose evaluator evidence disappeared routes to review."""

        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a", "clip-b", "clip-c"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        report = json.loads(fixtures["evaluator"].read_text())
        report["clips"] = [c for c in report["clips"] if c["clip_id"] != "clip-c"]
        fixtures["evaluator"].write_text(json.dumps(report))
        decisions = build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        by_id = {d["candidate_id"]: d for d in decisions["decisions"]}
        assert by_id["clip-a"]["decision"] == "accept"
        assert by_id["clip-b"]["decision"] == "reject"
        assert by_id["clip-c"]["decision"] == "review"

    def test_incomplete_clip_status_routes_to_review(self, tmp_path: Path) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        report = json.loads(fixtures["evaluator"].read_text())
        report["clips"][0]["status"] = "skipped"
        fixtures["evaluator"].write_text(json.dumps(report))
        decisions = build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        assert decisions["decisions"][0]["decision"] == "review"

    def test_accepted_run_routes_by_clip_evidence(self, tmp_path: Path) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a", "clip-b"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        decisions = build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        by_id = {d["candidate_id"]: d for d in decisions["decisions"]}
        assert by_id["clip-a"]["decision"] == "accept"
        assert by_id["clip-b"]["decision"] == "reject"

    def test_rejected_run_rejects_every_candidate(self, tmp_path: Path) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        disposition = json.loads(fixtures["disposition"].read_text())
        disposition["quality_status"] = "rejected"
        fixtures["disposition"].write_text(json.dumps(disposition))
        decisions = build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        assert all(d["decision"] == "reject" for d in decisions["decisions"])

    @pytest.mark.parametrize("conflicting", [False, True])
    def test_rejects_duplicate_evaluator_clip_ids(
        self, tmp_path: Path, conflicting: bool
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        evaluator = json.loads(fixtures["evaluator"].read_text())
        duplicate = dict(evaluator["clips"][0])
        if conflicting:
            duplicate["passed"] = not duplicate["passed"]
        evaluator["clips"].append(duplicate)
        fixtures["evaluator"].write_text(json.dumps(evaluator))
        output = tmp_path / "decisions.json"
        with pytest.raises(PaidfContractError, match="duplicate clip_id"):
            build_decisions_stage(
                str(tmp_path / "candidates.json"),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(output),
                decided_at=DECIDED_AT,
            )
        assert not output.exists()

    def test_rejects_malformed_or_out_of_set_evaluator_rows(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        evaluator = json.loads(fixtures["evaluator"].read_text())
        evaluator["clips"].append({"status": "completed", "passed": True})
        fixtures["evaluator"].write_text(json.dumps(evaluator))
        with pytest.raises(PaidfContractError, match="clip_id"):
            build_decisions_stage(
                str(tmp_path / "candidates.json"),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(tmp_path / "malformed-decisions.json"),
                decided_at=DECIDED_AT,
            )

        evaluator["clips"][-1] = {
            "clip_id": "clip-outside",
            "status": "completed",
            "passed": True,
        }
        fixtures["evaluator"].write_text(json.dumps(evaluator))
        with pytest.raises(PaidfContractError, match="outside the candidate manifest"):
            build_decisions_stage(
                str(tmp_path / "candidates.json"),
                str(fixtures["evaluator"]),
                str(fixtures["disposition"]),
                str(tmp_path / "outside-decisions.json"),
                decided_at=DECIDED_AT,
            )


def _valid_reconciliation(
    *,
    keep: list[dict],
    drop: list[dict],
    unresolved: list[str] | None = None,
) -> dict:
    unresolved_ids = unresolved or []
    return {
        "schema": "npa.paidf.reconciliation.v1",
        "provider": {"name": "human-curator", "role": "curation"},
        "totals": {
            "candidates": len(keep) + len(drop),
            "keep": len(keep),
            "drop": len(drop),
            "unresolved_review": len(unresolved_ids),
            "identity_gaps": 0,
        },
        "keep": keep,
        "drop": drop,
        "unresolved_review_candidate_ids": unresolved_ids,
    }


class TestAcceptedReplacementsStage:
    def test_keeps_only_accepted_with_materializer_fields(self, tmp_path: Path) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a", "clip-b"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        reconciliation = _valid_reconciliation(
            keep=[{"candidate_id": "clip-a", "decision": "accept"}],
            drop=[{"candidate_id": "clip-b", "decision": "reject"}],
        )
        recon_ref = tmp_path / "keep-drop.json"
        recon_ref.write_text(json.dumps(reconciliation))
        out = tmp_path / "accepted-variants.json"
        manifest = accepted_replacements_stage(
            candidates_manifest_ref=str(tmp_path / "candidates.json"),
            reconciliation_ref=str(recon_ref),
            output_ref=str(out),
        )
        assert manifest["totals"]["accepted"] == 1
        replacement = manifest["replacements"][0]
        assert replacement["episode_index"] == 3
        assert replacement["camera_key"] == "observation.images.workspace"
        assert replacement["video_uri"].endswith("clip-a/augmented_video.mp4")
        assert replacement["lineage"]["output_sha256"]
        stored = json.loads(out.read_text())
        assert stored["schema"] == "npa.paidf.accepted-variants.v1"

    def test_accepts_canonical_reconciliation_with_review_excluded(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a", "clip-b"])
        candidates = build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        decisions = build_decisions_stage(
            str(tmp_path / "candidates.json"),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "decisions.json"),
            decided_at=DECIDED_AT,
        )
        decisions["decisions"][1]["decision"] = "review"
        decisions["decisions"][1]["reason"] = "requires human review"
        reconciliation = reconcile_decisions(candidates, decisions)
        reconciliation_ref = tmp_path / "canonical-reconciliation.json"
        reconciliation_ref.write_text(json.dumps(reconciliation))

        manifest = accepted_replacements_stage(
            str(tmp_path / "candidates.json"),
            str(reconciliation_ref),
            str(tmp_path / "accepted.json"),
        )

        assert [row["lineage"]["candidate_id"] for row in manifest["replacements"]] == [
            "clip-a"
        ]
        assert reconciliation["unresolved_review_candidate_ids"] == ["clip-b"]

    def test_fails_closed_on_unknown_keep_ids(self, tmp_path: Path) -> None:
        reconciliation = {
            "schema": "npa.paidf.reconciliation.v1",
            "keep": [{"candidate_id": "ghost"}],
            "drop": [],
        }
        recon_ref = tmp_path / "keep-drop.json"
        recon_ref.write_text(json.dumps(reconciliation))
        candidates_ref = tmp_path / "candidates.json"
        candidates_ref.write_text(json.dumps({"schema": "npa.paidf.candidates.v1", "candidates": []}))
        with pytest.raises(PaidfContractError, match="unknown candidates"):
            accepted_replacements_stage(
                candidates_manifest_ref=str(candidates_ref),
                reconciliation_ref=str(recon_ref),
                output_ref=str(tmp_path / "out.json"),
            )

    def test_reconciliation_must_cover_every_candidate_exactly_once(
        self, tmp_path: Path
    ) -> None:
        candidates_ref = tmp_path / "candidates.json"
        candidates_ref.write_text(
            json.dumps(
                {
                    "schema": "npa.paidf.candidates.v1",
                    "candidates": [
                        {
                            "candidate_id": "cand-1",
                            "source_episode_id": "episode_0000",
                            "source_episode_index": 0,
                            "camera_key": "observation.images.workspace",
                            "variant": {
                                "variant_id": "var-1",
                                "output_sha256": SHA_A,
                                "source_sha256": SHA_B,
                            },
                            "evaluation": {},
                        }
                    ],
                }
            )
        )
        base = {"schema": "npa.paidf.reconciliation.v1"}

        missing = dict(base)
        missing["keep"] = []
        missing["drop"] = []
        (tmp_path / "missing.json").write_text(json.dumps(missing))
        with pytest.raises(PaidfContractError, match="missing candidates"):
            accepted_replacements_stage(
                candidates_manifest_ref=str(candidates_ref),
                reconciliation_ref=str(tmp_path / "missing.json"),
                output_ref=str(tmp_path / "out.json"),
            )

        duplicate = dict(base)
        duplicate["keep"] = [
            {"candidate_id": "cand-1", "decision": "accept"},
            {"candidate_id": "cand-1", "decision": "accept"},
        ]
        duplicate["drop"] = []
        (tmp_path / "duplicate.json").write_text(json.dumps(duplicate))
        with pytest.raises(PaidfContractError, match="duplicate candidate_id"):
            accepted_replacements_stage(
                candidates_manifest_ref=str(candidates_ref),
                reconciliation_ref=str(tmp_path / "duplicate.json"),
                output_ref=str(tmp_path / "out.json"),
            )

        overlapping = dict(base)
        overlapping["keep"] = [{"candidate_id": "cand-1", "decision": "accept"}]
        overlapping["drop"] = [{"candidate_id": "cand-1", "decision": "reject"}]
        (tmp_path / "overlap.json").write_text(json.dumps(overlapping))
        with pytest.raises(PaidfContractError, match="both keep and drop"):
            accepted_replacements_stage(
                candidates_manifest_ref=str(candidates_ref),
                reconciliation_ref=str(tmp_path / "overlap.json"),
                output_ref=str(tmp_path / "out.json"),
            )

    def test_review_cannot_enter_keep_and_states_must_be_consistent(
        self, tmp_path: Path
    ) -> None:
        candidates_ref = tmp_path / "candidates.json"
        candidates_ref.write_text(
            json.dumps(
                {
                    "schema": "npa.paidf.candidates.v1",
                    "candidates": [
                        {
                            "candidate_id": "cand-1",
                            "source_episode_id": "episode_0000",
                            "source_episode_index": 0,
                            "camera_key": "observation.images.workspace",
                            "variant": {
                                "variant_id": "var-1",
                                "output_sha256": SHA_A,
                                "source_sha256": SHA_B,
                            },
                            "evaluation": {},
                        }
                    ],
                }
            )
        )
        base = {"schema": "npa.paidf.reconciliation.v1"}

        review_in_keep = dict(base)
        review_in_keep["keep"] = [{"candidate_id": "cand-1", "decision": "review"}]
        review_in_keep["drop"] = []
        (tmp_path / "review-keep.json").write_text(json.dumps(review_in_keep))
        with pytest.raises(PaidfContractError, match="non-accept decision"):
            accepted_replacements_stage(
                candidates_manifest_ref=str(candidates_ref),
                reconciliation_ref=str(tmp_path / "review-keep.json"),
                output_ref=str(tmp_path / "out.json"),
            )

        bad_drop = dict(base)
        bad_drop["keep"] = []
        bad_drop["drop"] = [{"candidate_id": "cand-1", "decision": "maybe"}]
        (tmp_path / "bad-drop.json").write_text(json.dumps(bad_drop))
        with pytest.raises(PaidfContractError, match="non-terminal decision"):
            accepted_replacements_stage(
                candidates_manifest_ref=str(candidates_ref),
                reconciliation_ref=str(tmp_path / "bad-drop.json"),
                output_ref=str(tmp_path / "out.json"),
            )

        review_excluded = _valid_reconciliation(
            keep=[],
            drop=[{"candidate_id": "cand-1", "decision": "review"}],
            unresolved=["cand-1"],
        )
        (tmp_path / "review-drop.json").write_text(json.dumps(review_excluded))
        manifest = accepted_replacements_stage(
            candidates_manifest_ref=str(candidates_ref),
            reconciliation_ref=str(tmp_path / "review-drop.json"),
            output_ref=str(tmp_path / "out.json"),
        )
        assert manifest["replacements"] == []

    @pytest.mark.parametrize(
        "unresolved_value",
        [None, [], ["cand-1", "cand-1"], ["bad id"], ["cand-1", "ghost"]],
    )
    def test_review_ids_must_exactly_match_review_drops(
        self, tmp_path: Path, unresolved_value: object
    ) -> None:
        candidates_ref = tmp_path / "candidates.json"
        candidate = {
            "schema": "npa.paidf.candidates.v1",
            "candidates": [
                {
                    "candidate_id": "cand-1",
                    "source_episode_id": "episode_0000",
                    "source_episode_index": 0,
                    "camera_key": "observation.images.workspace",
                    "variant": {
                        "variant_id": "var-1",
                        "output_sha256": SHA_A,
                        "source_sha256": SHA_B,
                    },
                    "evaluation": {},
                }
            ],
        }
        candidates_ref.write_text(json.dumps(candidate))
        reconciliation = _valid_reconciliation(
            keep=[],
            drop=[{"candidate_id": "cand-1", "decision": "review"}],
            unresolved=["cand-1"],
        )
        if unresolved_value is None:
            del reconciliation["unresolved_review_candidate_ids"]
        else:
            reconciliation["unresolved_review_candidate_ids"] = unresolved_value
        reconciliation_ref = tmp_path / "reconciliation.json"
        reconciliation_ref.write_text(json.dumps(reconciliation))
        with pytest.raises(PaidfContractError, match="unresolved|must be a list"):
            accepted_replacements_stage(
                str(candidates_ref),
                str(reconciliation_ref),
                str(tmp_path / "out.json"),
            )

    def test_unresolved_list_is_required_even_without_review(
        self, tmp_path: Path
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        reconciliation = _valid_reconciliation(
            keep=[{"candidate_id": "clip-a", "decision": "accept"}], drop=[]
        )
        del reconciliation["unresolved_review_candidate_ids"]
        reconciliation_ref = tmp_path / "reconciliation.json"
        reconciliation_ref.write_text(json.dumps(reconciliation))
        with pytest.raises(PaidfContractError, match="must be a list"):
            accepted_replacements_stage(
                str(tmp_path / "candidates.json"),
                str(reconciliation_ref),
                str(tmp_path / "out.json"),
            )

    @pytest.mark.parametrize(
        ("provider", "error"),
        [
            (None, "must be an object"),
            ({"name": "curator"}, "exactly name and role"),
            ({"name": "curator", "role": "simulation"}, "curation or evaluation"),
            (
                {"name": "curator", "role": "curation", "extra": True},
                "exactly name and role",
            ),
        ],
    )
    def test_reconciliation_provider_is_exact(
        self, tmp_path: Path, provider: object, error: str
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        reconciliation = _valid_reconciliation(
            keep=[{"candidate_id": "clip-a", "decision": "accept"}], drop=[]
        )
        reconciliation["provider"] = provider
        reconciliation_ref = tmp_path / "reconciliation.json"
        reconciliation_ref.write_text(json.dumps(reconciliation))
        with pytest.raises(PaidfContractError, match=error):
            accepted_replacements_stage(
                str(tmp_path / "candidates.json"),
                str(reconciliation_ref),
                str(tmp_path / "out.json"),
            )

    @pytest.mark.parametrize(
        ("mutation", "error"),
        [
            (("remove", "identity_gaps", None), "must contain exactly"),
            (("set", "extra", 0), "must contain exactly"),
            (("set", "candidates", 2), "totals.candidates"),
            (("set", "keep", 0), "totals.keep"),
            (("set", "drop", 1), "totals.drop"),
            (("set", "unresolved_review", 1), "totals.unresolved_review"),
            (("set", "identity_gaps", 1), "totals.identity_gaps"),
            (("set", "candidates", True), "totals.candidates"),
        ],
    )
    def test_reconciliation_totals_are_exact_strict_integers(
        self, tmp_path: Path, mutation: tuple[str, str, object], error: str
    ) -> None:
        fixtures = _builder_fixtures(tmp_path, video_names=["clip-a"])
        build_candidates_stage(
            str(fixtures["augment"]),
            str(fixtures["provenance"]),
            str(fixtures["evaluator"]),
            str(fixtures["disposition"]),
            str(tmp_path / "candidates.json"),
        )
        reconciliation = _valid_reconciliation(
            keep=[{"candidate_id": "clip-a", "decision": "accept"}], drop=[]
        )
        action, key, value = mutation
        if action == "remove":
            del reconciliation["totals"][key]
        else:
            reconciliation["totals"][key] = value
        reconciliation_ref = tmp_path / "reconciliation.json"
        reconciliation_ref.write_text(json.dumps(reconciliation))
        with pytest.raises(PaidfContractError, match=error):
            accepted_replacements_stage(
                str(tmp_path / "candidates.json"),
                str(reconciliation_ref),
                str(tmp_path / "out.json"),
            )


# ── Frozen-source lock (moved from the future workflow-composition tests) ──

_SHA_T1 = "1" * 64
_SHA_T2 = "2" * 64


def _write_frozen_manifest(
    path: Path,
    *,
    train_hashes: list[str],
    heldout_hashes: list[str],
    contract_sha256: str = SHA_B,
    trajectory_override: str | None = None,
) -> str:
    episodes = [
        {"episode_id": f"episode_{i:04d}", "split": split, "trajectory_sha256": traj}
        for i, (split, traj) in enumerate(
            [("train", t) for t in train_hashes]
            + [("heldout", t) for t in heldout_hashes]
        )
    ]
    if trajectory_override is not None:
        episodes[0]["trajectory_sha256"] = trajectory_override
    manifest = {
        "schema": FROZEN_SOURCE_MANIFEST_SCHEMA,
        "source_contract": {
            "schema": "npa.genesis.lerobot-source.v1",
            "sha256": contract_sha256,
        },
        "episodes": episodes,
    }
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class TestLoadFrozenSourceManifest:
    def test_accepts_bytes_matching_the_lock(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "source-manifest.json"
        digest = _write_frozen_manifest(
            manifest_path, train_hashes=[_SHA_T1], heldout_hashes=[_SHA_T2]
        )
        manifest = load_frozen_source_manifest(
            str(manifest_path),
            expected_sha256=digest,
            expected_contract_sha256=SHA_B,
        )
        assert len(manifest["episodes"]) == 2

    def test_rejects_bytes_that_do_not_match_the_locked_digest(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "source-manifest.json"
        digest = _write_frozen_manifest(
            manifest_path, train_hashes=[_SHA_T1], heldout_hashes=[_SHA_T2]
        )
        with pytest.raises(PaidfContractError, match="locked digest"):
            load_frozen_source_manifest(
                str(manifest_path),
                expected_sha256=SHA_A,
                expected_contract_sha256=SHA_B,
            )
        assert digest != SHA_A

    def test_rejects_manifest_pointing_at_a_different_source_contract(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "source-manifest.json"
        digest = _write_frozen_manifest(
            manifest_path,
            train_hashes=[_SHA_T1],
            heldout_hashes=[_SHA_T2],
            contract_sha256=SHA_C,
        )
        with pytest.raises(PaidfContractError, match="source-contract digest"):
            load_frozen_source_manifest(
                str(manifest_path),
                expected_sha256=digest,
                expected_contract_sha256=SHA_B,
            )

    def test_rejects_split_content_overlap(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "source-manifest.json"
        digest = _write_frozen_manifest(
            manifest_path, train_hashes=[_SHA_T1], heldout_hashes=[_SHA_T1]
        )
        with pytest.raises(PaidfContractError, match="overlap"):
            load_frozen_source_manifest(
                str(manifest_path),
                expected_sha256=digest,
                expected_contract_sha256=SHA_B,
            )

    def test_rejects_malformed_trajectory_hash(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "source-manifest.json"
        digest = _write_frozen_manifest(
            manifest_path,
            train_hashes=[_SHA_T1],
            heldout_hashes=[_SHA_T2],
            trajectory_override="not-a-digest",
        )
        with pytest.raises(PaidfContractError, match="trajectory_sha256"):
            load_frozen_source_manifest(
                str(manifest_path),
                expected_sha256=digest,
                expected_contract_sha256=SHA_B,
            )

    def test_rejects_wrong_schema(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "source-manifest.json"
        manifest = {"schema": "npa.paidf.something-else.v1", "episodes": []}
        payload = (json.dumps(manifest) + "\n").encode("utf-8")
        manifest_path.write_bytes(payload)
        with pytest.raises(PaidfContractError, match="schema"):
            load_frozen_source_manifest(
                str(manifest_path),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_contract_sha256=SHA_B,
            )


# ── Base-campaign freeze helper (moved from the workflow-composition tests) ──


def _artifact_file(path: Path, payload: dict) -> tuple[str, str]:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return str(path), hashlib.sha256(data).hexdigest()


class _LocalResolver:
    """Map s3://phase0-fixture/<name> URIs onto real files in a tmp directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def __call__(self, ref: str) -> bytes:
        if ref.startswith("s3://phase0-fixture/"):
            return (self._root / ref.removeprefix("s3://phase0-fixture/")).read_bytes()
        return Path(ref).read_bytes()


@pytest.fixture()
def local_s3_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import npa.workflows.paidf_campaign as campaign_module

    resolver = _LocalResolver(tmp_path)
    monkeypatch.setattr(campaign_module, "_read_artifact_bytes", resolver)
    return resolver


class TestBaseCampaignFreeze:
    def test_pins_exact_stage_bytes(
        self, tmp_path: Path, local_s3_resolver: _LocalResolver
    ) -> None:
        digests = {}
        for stage in ("source", "generation", "evaluation"):
            _, digest = _artifact_file(tmp_path / f"{stage}.json", {"stage": stage})
            digests[stage] = digest
        campaign_ref = tmp_path / "campaign.json"
        campaign = freeze_base_campaign_stage(
            "paidf-phase0",
            source_manifest_ref="s3://phase0-fixture/source.json",
            generation_manifest_ref="s3://phase0-fixture/generation.json",
            evaluation_report_ref="s3://phase0-fixture/evaluation.json",
            campaign_output_ref=str(campaign_ref),
        )
        assert validate_campaign(campaign) == campaign
        stored = json.loads(campaign_ref.read_text(encoding="utf-8"))
        for stage in ("source", "generation", "evaluation"):
            assert stored["stage_fingerprints"][stage] == digests[stage]
            assert stored["artifacts"][stage]["sha256"] == digests[stage]
            assert stored["artifacts"][stage]["uri"].startswith("s3://phase0-fixture/")

    def test_rejects_incomplete_stage_set(self) -> None:
        with pytest.raises(PaidfContractError, match="exactly the stages"):
            build_base_campaign(
                "paidf-phase0",
                stage_refs={"source": "s3://bucket/source.json"},
                stage_sha256s={"source": SHA_A},
            )

    @pytest.mark.parametrize(
        "stage_refs",
        [
            {
                "source": "s3://bucket/source.json",
                "generation": "s3://bucket/generation.json",
            },
            {
                "source": "s3://bucket/source.json",
                "generation": "s3://bucket/generation.json",
                "evaluation": "s3://bucket/evaluation.json",
                "extra": "s3://bucket/extra.json",
            },
        ],
    )
    def test_rejects_missing_or_extra_stage_refs(self, stage_refs: dict[str, str]) -> None:
        with pytest.raises(PaidfContractError, match="exactly the stages"):
            build_base_campaign(
                "paidf-phase0",
                stage_refs=stage_refs,
                stage_sha256s={
                    "source": SHA_A,
                    "generation": SHA_B,
                    "evaluation": SHA_C,
                },
            )

    @pytest.mark.parametrize(
        "stage_sha256s",
        [
            {"source": SHA_A, "generation": SHA_B},
            {
                "source": SHA_A,
                "generation": SHA_B,
                "evaluation": SHA_C,
                "extra": "d" * 64,
            },
        ],
    )
    def test_rejects_missing_or_extra_stage_digests(
        self, stage_sha256s: dict[str, str]
    ) -> None:
        with pytest.raises(PaidfContractError, match="digest keys"):
            build_base_campaign(
                "paidf-phase0",
                stage_refs={
                    "source": "s3://bucket/source.json",
                    "generation": "s3://bucket/generation.json",
                    "evaluation": "s3://bucket/evaluation.json",
                },
                stage_sha256s=stage_sha256s,
            )

    def test_frozen_manifest_is_never_mutated_by_the_freeze(
        self, tmp_path: Path, local_s3_resolver: _LocalResolver
    ) -> None:
        _, digest = _artifact_file(tmp_path / "source.json", {"stage": "source"})
        _artifact_file(tmp_path / "generation.json", {"stage": "generation"})
        _artifact_file(tmp_path / "evaluation.json", {"stage": "evaluation"})
        campaign = freeze_base_campaign_stage(
            "paidf-phase0",
            source_manifest_ref="s3://phase0-fixture/source.json",
            generation_manifest_ref="s3://phase0-fixture/generation.json",
            evaluation_report_ref="s3://phase0-fixture/evaluation.json",
            campaign_output_ref=str(tmp_path / "campaign.json"),
        )
        assert campaign["artifacts"]["source"]["sha256"] == digest
        # The frozen artifact is consumed read-only: its bytes are unchanged.
        assert hashlib.sha256((tmp_path / "source.json").read_bytes()).hexdigest() == digest
