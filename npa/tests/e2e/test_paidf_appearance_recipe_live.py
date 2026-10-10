"""Audit retained twelve-profile runs without generating or promoting candidates.

Set NPA_INTEGRATION_E2E=1, NPA_E2E_PROJECT and NPA_PAIDF_APPEARANCE_CASES to a
private JSON array of run_uri and expected_frames records. This verifies profile
coverage, actual media, controls and evaluation accounting, not task fidelity.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from npa.workflows.paidf_cosmos3_media import verify_pair, video_sha256
from npa.workflows.paidf_cosmos3 import _validated_quality_status
from npa.workflows.video_padding_preservation import _pixel_hashes


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]
ROOT = Path(__file__).resolve().parents[3]


def _cases():
    path = os.environ.get("NPA_PAIDF_APPEARANCE_CASES", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        return [
            pytest.param(None, marks=pytest.mark.skip(reason="requires retained runs"))
        ]
    cases = json.loads(Path(path).read_text())
    assert isinstance(cases, list) and cases
    return cases


def _location(root, relative):
    location = urlsplit(root)
    assert location.scheme == "s3" and location.netloc
    return location.netloc, location.path.strip("/") + "/" + relative


def _read(client, root, relative):
    bucket, key = _location(root, relative)
    with client.get_object(Bucket=bucket, Key=key)["Body"] as body:
        return json.loads(body.read())


def _download(client, root, relative, destination):
    bucket, key = _location(root, relative)
    client.download_file(bucket, key, str(destination))
    return destination


def _recipe():
    config = yaml.safe_load((ROOT / "workflows/main/paidf-cosmos3.yaml").read_text())[
        "config"
    ]
    overlay = yaml.safe_load(
        (ROOT / "docs/workbench/examples/paidf-appearance-12.yaml").read_text()
    )["config"]
    config.update(overlay)
    return config


def _audit_profiles(configuration, recipe):
    assert configuration["appearance_profiles"] == json.loads(
        recipe["appearance_profiles_json"]
    )
    assert configuration["n_augmentations"] == 12
    keys = configuration["appearance_profiles"][0]
    profiles = {
        tuple(sorted((key, item[key]) for key in keys))
        for item in configuration["augmentations"]
    }
    expected = {
        tuple(sorted(item.items())) for item in configuration["appearance_profiles"]
    }
    assert profiles == expected and len(profiles) == 12


def _audit_variant(client, root, variant, source, destination, recipe, frames):
    clip = variant["clip"]
    video_uri = variant["augmented_video_uri"]
    assert video_uri.startswith(root.rstrip("/") + "/cosmos_augmented/" + clip + "/")
    assert video_uri.endswith("/augmented_video.mp4")
    relative = video_uri[len(root.rstrip("/")) + 1 :].rsplit("/", 1)[0] + "/"
    metadata = _read(client, root, relative + "metadata.json")
    receipt = _read(client, root, relative + "transfer.json")
    video = _download(client, root, relative + "augmented_video.mp4", destination)
    alignment = verify_pair(source, video)
    assert alignment["decoded_frames"] == frames
    assert alignment["generated_sha256"] == metadata["published_video_sha256"]
    assert alignment["source_sha256"] == metadata["temporal_alignment"]["source_sha256"]
    _audit_padding(client, root, relative, source, video, metadata)
    assert metadata["guardrails"] is True and metadata["motion_preservation"] is None
    assert receipt["text_guardrail_passed"] is receipt["video_guardrail_passed"] is True
    assert receipt["control_loader_verified"] is True
    assert receipt["normalize_cfg"] is True
    assert receipt["first_chunk_conditional_frames"] == 0
    assert receipt["rgb_conditioning"]["output_pixel_blending"] is False
    assert receipt["rgb_conditioning"]["control_loader_verified"] is True
    assert receipt["rgb_conditioning"]["weight"] == recipe["transfer_rgb_weight"]
    assert metadata["guidance"] == float(recipe["guidance"]) + metadata[
        "attempt"
    ] * float(recipe["retry_guidance_delta"])
    assert metadata["steps"] == int(recipe["steps"]) + metadata["attempt"] * int(
        recipe["retry_steps_delta"]
    )
    return alignment["generated_sha256"]


def _audit_padding(client, root, relative, source, video, metadata):
    record = metadata.get("source_content_region")
    receipt = metadata.get("padding_preservation")
    if record is None:
        assert receipt is None
        return
    if receipt is None:
        assert record["bounds"] == [0, 0, *record["canvas"]]
        return
    assert receipt["status"] == "verified"
    assert (
        receipt["padding_matches_source"] is receipt["scene_pixels_unchanged"] is True
    )
    raw = _download(
        client, root, relative + "raw_model_video.mp4", video.with_suffix(".raw.mp4")
    )
    for path, key in (
        (source, "source_sha256"),
        (raw, "raw_model_sha256"),
        (video, "published_sha256"),
    ):
        assert video_sha256(path) == receipt[key]
    source_pixels = _pixel_hashes(source, record)
    raw_pixels = _pixel_hashes(raw, record)
    published_pixels = _pixel_hashes(video, record)
    assert source_pixels["padding_rgb_sha256"] == published_pixels["padding_rgb_sha256"]
    assert raw_pixels["scene_rgb_sha256"] == published_pixels["scene_rgb_sha256"]
    assert published_pixels == {key: receipt[key] for key in published_pixels}


def _audit_terminal(runtime, accepted):
    if accepted:
        assert runtime["status"] == "succeeded"
        return
    assert runtime["status"] == "failed"
    assert runtime["waves"][-1]["states"] == ["reject-quality"]
    latest = {
        state: wave["status"] for wave in runtime["waves"] for state in wave["states"]
    }
    assert latest.pop("reject-quality") == "failed"
    required = {
        "generate-variants",
        "evaluate",
        "quality-gate",
        "quality-disposition",
        "visualize-quality-evidence",
        "quality-route",
    }
    assert required <= latest.keys()
    assert set(latest.values()) == {"succeeded"}
    assert "require-accepted-quality" not in latest
    assert "annotate-augmented" not in latest


def _audit_completion(client, root, hashes):
    evaluation = _read(client, root, "grade/cosmos_evaluator.json")
    disposition = _read(client, root, "grade/quality_disposition.json")
    assert evaluation["status"] == "completed"
    evaluated = {
        item["clip_id"]: item["temporal_alignment"]["generated_sha256"]
        for item in evaluation["clips"]
    }
    assert evaluated == hashes
    assert disposition["score"] == pytest.approx(evaluation["score"])
    assert disposition["threshold"] == evaluation["threshold"] == 0.75
    assert evaluation["attribute_threshold"] == 1.0
    assert evaluation["temporal_mode"] == evaluation["appearance_mode"] == "advisory"
    expected = "accepted" if evaluation["passed"] else "rejected"
    assert _validated_quality_status(disposition) == expected
    _audit_terminal(
        _read(client, root, "npa-workflow/runtime.json"), evaluation["passed"]
    )
    bucket, key = _location(root, "reports/quality-evidence.rrd")
    assert client.head_object(Bucket=bucket, Key=key)["ContentLength"] > 0


@pytest.mark.timeout(0)
@pytest.mark.parametrize("case", _cases())
def test_twelve_profile_outputs_and_quality_accounting(case, tmp_path):
    """Verify the shipped recipe's distinct profiles and real source-aligned videos.

    Args:
        case: Private run location and expected source frame count.
        tmp_path: Isolated download directory.
    Returns:
        None.
    Raises:
        AssertionError: Profile, media, control or evaluation evidence disagrees.
    """
    from npa.clients.project_credentials import s3_client_for_project

    client = s3_client_for_project(os.environ["NPA_E2E_PROJECT"])
    root = case["run_uri"]
    recipe = _recipe()
    configuration = _read(client, root, "configs/manifest.json")
    _audit_profiles(configuration, recipe)
    manifest = _read(client, root, "cosmos_augmented/manifest.json")
    assert manifest["status"] == "executed" and manifest["variant_count"] == 12
    variants = {item["clip"]: item for item in manifest["variants"]}
    assert len(manifest["variants"]) == len(variants) == 12
    assert set(variants) == {f"variant-{index:04d}" for index in range(12)}
    source = _download(client, root, "input/source.mp4", tmp_path / "source.mp4")
    hashes = {
        clip: _audit_variant(
            client,
            root,
            variants[clip],
            source,
            tmp_path / (clip + ".mp4"),
            recipe,
            case["expected_frames"],
        )
        for clip in sorted(variants)
    }
    assert len(set(hashes.values())) == 12
    _audit_completion(client, root, hashes)
