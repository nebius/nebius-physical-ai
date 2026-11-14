"""Audit a completed fresh default-starter PAIDF run without submitting work."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

from .paidf_runtime_audit import assert_completed_fresh_runtime
from .test_npa_workflow_submit_live_e2e import _assert_paidf_live_artifacts
from .test_paidf_cosmos3_mp4_live import _assert_recording_identity


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]

WORKFLOW = Path(__file__).resolve().parents[3] / "workflows/main/paidf-cosmos3.yaml"


def test_completed_fresh_default_starter_pipeline() -> None:
    audit_starter_pipeline(
        "NPA_E2E_PAIDF_STARTER_RUN_URI",
        "NPA_E2E_PAIDF_STARTER_FRESH_AFTER",
        variant_count=2,
    )


def audit_starter_pipeline(
    uri_variable, freshness_variable, *, variant_count, recipe=None
):
    """Audit a fresh starter run with the default or a checked-in recipe overlay.

    Args:
        uri_variable: Environment variable selecting the completed run URI.
        freshness_variable: Environment variable holding its pre-submit timestamp.
        variant_count: Exact requested number of variants.
        recipe: Optional overlay ``config`` mapping. When set, the run must use its
            exact appearance profiles and first-window conditioning. Per-variant
            seeds, guidance, steps, model, structural control and thresholds
            must match the defaults, and each native transfer receipt must record
            the workflow's default edge preset, CFG normalization, absent RGB hint
            and frame rate. Control guidance and chunk size are not recorded per
            variant, so this audit cannot verify them.
    Returns:
        None.
    Raises:
        AssertionError: Freshness, settings or complete execution is unproven.
    """
    uri = os.environ.get(uri_variable, "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not uri:
        pytest.skip("Requires the guide's completed fresh default-starter run")
    from npa.clients.project_credentials import s3_client_for_project

    selected = urlparse(uri)
    assert selected.scheme == "s3" and selected.netloc
    prefix = selected.path.strip("/") + "/"
    run_id = prefix.strip("/").split("/")[-1]
    assert prefix == f"paidf-cosmos3/{run_id}/"
    project = os.environ.get("NPA_E2E_PROJECT") or None
    client = s3_client_for_project(project, allow_host_creds=True)
    runtime = _read(client, selected.netloc, prefix + "npa-workflow/runtime.json")
    assert runtime["status"] == "succeeded" and runtime["run_id"] == run_id
    assert_completed_fresh_runtime(client, selected.netloc, prefix, runtime)
    _assert_starter_input(client, selected.netloc, run_id)
    _assert_starter_batch(client, selected.netloc, prefix, variant_count, recipe)
    _assert_fresh_objects(
        client, selected.netloc, run_id, os.environ[freshness_variable]
    )
    _assert_paidf_live_artifacts(
        spec="paidf-cosmos3.yaml",
        run_prefix_uri=uri,
        waves=runtime["waves"],
        bucket=selected.netloc,
        run_id=run_id,
        e2e_project=project,
    )
    _assert_recording_identity(client, selected.netloc, prefix, run_id)


def _read(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return json.loads(body.read())


def _assert_starter_input(client, bucket, run_id) -> None:
    from npa.workflows.data_factory_input import load_starter_contract

    contract = load_starter_contract()
    input_prefix = f"physical-ai-data-factory/{run_id}/input/"
    provenance = _read(client, bucket, input_prefix + "provenance.json")
    assert provenance["run_id"] == run_id
    assert provenance["source_kind"] == "upstream_sample"
    assert provenance["immutable_revision"] == contract["source"]["immutable_revision"]
    assert provenance["sha256"] == contract["integrity"]["sha256"]
    keys = [
        input_prefix + "source.mp4",
        f"paidf-cosmos3/{run_id}/input/original_source.mp4",
    ]
    for key in keys:
        response = client.get_object(Bucket=bucket, Key=key)
        with response["Body"] as body:
            contents = body.read()
        assert len(contents) == contract["integrity"]["byte_size"]
        assert hashlib.sha256(contents).hexdigest() == contract["integrity"]["sha256"]


def _assert_starter_batch(client, bucket, prefix, variant_count, recipe=None) -> None:
    manifest = _read(client, bucket, prefix + "cosmos_augmented/manifest.json")
    assert len(manifest["variants"]) == manifest["variant_count"] == variant_count
    expected = {f"variant-{index:04d}" for index in range(variant_count)}
    assert {item["clip"] for item in manifest["variants"]} == expected
    assert manifest["variant_parallelism"] == 1
    assert manifest["motion_preservation"]["source_weight"] == 0
    configs = _read(client, bucket, prefix + "configs/manifest.json")
    assert configs["n_augmentations"] == len(configs["augmentations"]) == variant_count
    assert str(configs["augmentation_seed"]) == "30"
    if recipe is None:
        assert configs["appearance_profile_source"] == "default"
    else:
        _assert_recipe_generation(client, bucket, manifest, configs, recipe)
    _assert_default_generation(manifest)
    evaluator = _read(client, bucket, prefix + "grade/cosmos_evaluator.json")
    assert evaluator["threshold"] == 0.2
    assert evaluator["attribute_threshold"] == 0.25
    assert len(evaluator["clips"]) == variant_count
    assert {clip["clip_id"] for clip in evaluator["clips"]} == expected
    final = _read(client, bucket, prefix + "reports/final.json")
    assert final["variant_count"] == final["annotated_variant_count"] == variant_count


def _assert_recipe_generation(client, bucket, manifest, configs, recipe) -> None:
    profiles = json.loads(recipe["appearance_profiles_json"])
    assert configs["appearance_profile_source"] == "custom"
    assert configs["appearance_profiles"] == profiles
    requested = {
        tuple((key, item[key]) for key in profiles[0])
        for item in configs["augmentations"]
    }
    assert len(requested) == len(configs["augmentations"]), "duplicate profile request"
    first_window = int(recipe["transfer_first_chunk_conditional_frames"])
    for variant in manifest["variants"]:
        video = urlparse(variant["augmented_video_uri"])
        assert video.netloc == bucket
        receipt_key = video.path.lstrip("/").rsplit("/", 1)[0] + "/transfer.json"
        receipt = _read(client, bucket, receipt_key)
        assert receipt["first_chunk_conditional_frames"] == first_window
        assert receipt["text_guardrail_passed"] and receipt["video_guardrail_passed"]
        _assert_default_transfer_receipt(receipt)


def _assert_default_transfer_receipt(receipt) -> None:
    defaults = yaml.safe_load(WORKFLOW.read_text())["config"]
    assert receipt["edge_threshold"] == defaults["transfer_edge_threshold"]
    normalize = defaults["transfer_cfg_normalization"] == "enabled"
    assert receipt["normalize_cfg"] is normalize
    if float(defaults["transfer_rgb_weight"]) == 0:
        assert not receipt.get("rgb_conditioning"), "unexpected RGB hint"
    assert float(receipt["output_fps"]) == float(defaults["conditioning_fps"])


def _assert_default_generation(manifest) -> None:
    attempt = manifest["attempt"]
    assert attempt in (0, 1)
    assert manifest["model"] == "Cosmos3-Nano"
    assert manifest["structural_control"] == "edge"
    for index, variant in enumerate(manifest["variants"]):
        assert variant["seed"] == 17 + attempt * 1000 + index
        assert variant["guidance"] == 5.0 - attempt * 0.5
        assert variant["steps"] == 24 + attempt * 4


def _assert_fresh_objects(client, bucket, run_id, timestamp) -> None:
    fresh_after = datetime.fromisoformat(timestamp)
    assert fresh_after.tzinfo is not None, "Freshness timestamp must include UTC offset"
    paginator = client.get_paginator("list_objects_v2")
    for root in ("paidf-cosmos3", "physical-ai-data-factory"):
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{root}/{run_id}/"):
            for item in page.get("Contents", []):
                assert item["LastModified"] >= fresh_after, (
                    "Object predates this submission"
                )
                count += 1
        assert count > 0, f"Missing {root} artifacts"
