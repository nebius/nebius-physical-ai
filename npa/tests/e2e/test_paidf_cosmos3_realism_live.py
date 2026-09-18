"""Audit matched real Cosmos3 augmentation runs without generating or promoting data.

Set NPA_INTEGRATION_E2E=1, NPA_E2E_PROJECT and NPA_PAIDF_REALISM_CASES to a private
JSON file containing an array of baseline_uri, candidate_uri, episode, camera,
and prepared_frames records. URIs point at complete run roots. This audits
execution, matched inputs and rejection accounting; it never certifies realism.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from npa.workflows.paidf_cosmos3_media import verify_pair, video_sha256


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]


def _cases():
    path = os.environ.get("NPA_PAIDF_REALISM_CASES", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        return [pytest.param(None, marks=pytest.mark.skip(reason="requires retained real PAIDF comparison runs"))]
    cases = json.loads(Path(path).read_text())
    assert isinstance(cases, list) and cases, "The supplied live comparison manifest must not be empty"
    return cases


def _read(client, root, relative):
    location = urlsplit(root)
    assert location.scheme == "s3" and location.netloc
    with client.get_object(Bucket=location.netloc, Key=location.path.strip("/") + "/" + relative)["Body"] as body:
        return json.loads(body.read())


def _download(client, root, relative, destination):
    location = urlsplit(root)
    client.download_file(location.netloc, location.path.strip("/") + "/" + relative, str(destination))
    return destination


def _audit_source(client, root, expected, destination):
    provenance = _read(client, root, "input/provenance.json")
    assert provenance["source_kind"] == "lerobot_dataset"
    assert provenance["episode"] == expected["episode"]
    assert provenance["camera"] == expected["camera"]
    source = _download(client, root, "input/source.mp4", destination / "source.mp4")
    assert video_sha256(source) == provenance["sha256"]
    return source


def _audit_variant(client, root, variant, source, destination, *, normalized):
    relative = "cosmos_augmented/" + variant["clip"] + "/"
    metadata = _read(client, root, relative + "metadata.json")
    receipt = _read(client, root, relative + "transfer.json")
    video = _download(client, root, relative + "augmented_video.mp4", destination / "generated.mp4")
    alignment = verify_pair(source, video)
    assert alignment["generated_sha256"] == metadata["published_video_sha256"]
    assert alignment["source_sha256"] == metadata["temporal_alignment"]["source_sha256"]
    assert alignment["generated_sha256"] != alignment["source_sha256"]
    assert metadata["motion_preservation"] is None and metadata["guardrails"] is True
    assert receipt["text_guardrail_passed"] is receipt["video_guardrail_passed"] is True
    assert receipt["guardrail_postprocessing_applied"] is True
    assert receipt["control_loader_verified"] is True
    assert receipt["source_frames"] == alignment["decoded_frames"]
    assert receipt["rgb_conditioning"]["control_loader_verified"] is True
    assert receipt["rgb_conditioning"]["output_pixel_blending"] is False
    if normalized:
        assert receipt["normalize_cfg"] is True
    return metadata, receipt, alignment


def _audit_disposition(client, root, clips):
    report = _read(client, root, "grade/cosmos_evaluator.json")
    disposition = _read(client, root, "grade/quality_disposition.json")
    assert report["status"] == "completed"
    assert len(report["clips"]) == len(clips)
    assert {clip["clip_id"] for clip in report["clips"]} == set(clips)
    assert disposition["threshold"] == pytest.approx(0.75)
    assert disposition["score"] == pytest.approx(report["score"])
    expected = "accepted" if disposition["hard_checks_passed"] and report["score"] >= 0.75 else "rejected"
    assert disposition["quality_status"] == expected
    assert disposition["decision"] == ("promote_checkpoint" if expected == "accepted" else "loop_back")


@pytest.mark.timeout(0)
@pytest.mark.parametrize("case", _cases())
def test_matched_native_cfg_normalization_artifacts(case, tmp_path):
    """Verify actual source-conditioned media and matched native sampling evidence.

    Args:
        case: Private operator-supplied baseline/candidate run references.
        tmp_path: Pytest's isolated artifact download directory.
    Returns:
        None.
    Raises:
        AssertionError: Inputs, provenance, controls, alignment or decisions disagree.
    """
    from npa.clients.project_credentials import s3_client_for_project

    client = s3_client_for_project(os.environ["NPA_E2E_PROJECT"])
    runs = {}
    for role in ("baseline", "candidate"):
        root = case[role + "_uri"]
        destination = tmp_path / role
        destination.mkdir()
        source = _audit_source(client, case["baseline_uri"], case, destination)
        manifest = _read(client, root, "cosmos_augmented/manifest.json")
        assert manifest["status"] == "executed" and manifest["variants"]
        assert manifest["lineage"]["input_provenance_uri"] == case["baseline_uri"].rstrip("/") + "/input/provenance.json"
        assert manifest["variant_count"] == len(manifest["variants"])
        outputs = {}
        for index, variant in enumerate(manifest["variants"]):
            folder = destination / str(index)
            folder.mkdir()
            outputs[variant["clip"]] = _audit_variant(
                client, root, variant, source, folder, normalized=role == "candidate",
            )
        _audit_disposition(client, root, outputs)
        runs[role] = outputs
    _assert_matched_inputs(runs, case["prepared_frames"])


def _assert_matched_inputs(runs, prepared_frames):
    assert runs["baseline"].keys() == runs["candidate"].keys()
    for clip, (baseline, old_control, old_alignment) in runs["baseline"].items():
        candidate, control, alignment = runs["candidate"][clip]
        for key in ("prompt", "variables", "model", "seed", "guidance", "steps"):
            assert baseline[key] == candidate[key], f"Matched input changed: {key}"
        for key in ("source_sha256", "decoded_frames", "fps", "duration_seconds"):
            assert old_alignment[key] == alignment[key]
        assert alignment["decoded_frames"] == prepared_frames
        for key in ("control_pixels_sha256", "native_chunks", "first_chunk_conditional_frames",
                    "overlap_conditional_frames", "edge_threshold", "effective_prompts", "output_fps"):
            assert old_control[key] == control[key], f"Source control changed: {key}"
        for key in ("control_pixels_sha256", "weight", "source_frames", "preset"):
            assert old_control["rgb_conditioning"][key] == control["rgb_conditioning"][key]
