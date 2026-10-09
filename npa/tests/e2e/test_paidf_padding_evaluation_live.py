"""Regrade retained videos with live VLM calls and provenance-backed scene crops."""

import json
import os
from pathlib import Path

import pytest

from npa.workbench.cosmos_evaluator.evaluate import evaluate_run
from npa.workbench.cosmos_evaluator.hallucination import check_hallucination
from npa.workflows.paidf_cosmos3_media import probe_video
from npa.workflows.video_content_region import (
    content_region_record,
    measure_scaled_size,
)
from .test_paidf_appearance_recipe_live import _download, _read

pytestmark = [pytest.mark.e2e, pytest.mark.token_factory_e2e]


def _cases():
    path = os.environ.get("NPA_PAIDF_PADDING_CASES")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        return [
            pytest.param(None, marks=pytest.mark.skip(reason="requires retained runs"))
        ]
    cases = json.loads(Path(path).read_text())
    assert isinstance(cases, list) and cases
    return cases


def _retained_region(client, root, source, workdir):
    timeline = _read(client, root, "input/timeline.json")
    prepared = probe_video(source)
    assert prepared == timeline["prepared"]
    if "source_content_region" in timeline:
        return timeline["source_content_region"]
    # Older evidence is migrated only from the exact retained normalization
    # recipe and its hashed original video, never from a guessed black border.
    original = _download(
        client, root, "input/original_source.mp4", workdir / "original.mp4"
    )
    assert probe_video(original) == timeline["original"]
    width, height = prepared["width"], prepared["height"]
    scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    expected = (
        f"setpts=PTS-STARTPTS,fps={timeline['fps']}:start_time=0,{scale},"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    assert timeline["ffmpeg_filter"] == expected
    scene_width, scene_height = measure_scaled_size(original, scale)
    left, top = 2 * ((width - scene_width) // 4), 2 * ((height - scene_height) // 4)
    return content_region_record(
        prepared, [left, top, left + scene_width, top + scene_height]
    )


def _stage_variants(client, root, region, workdir):
    augment = workdir / "augment"
    augment.mkdir()
    manifest = _read(client, root, "cosmos_augmented/manifest.json")
    assert manifest["status"] == "executed"
    for item in manifest["variants"]:
        clip = item["clip"]
        assert clip.startswith("variant-") and clip.removeprefix("variant-").isdigit()
        destination = augment / clip
        destination.mkdir()
        relative = "cosmos_augmented/" + clip + "/"
        _download(
            client,
            root,
            relative + "augmented_video.mp4",
            destination / "augmented_video.mp4",
        )
        metadata = _read(client, root, relative + "metadata.json")
        metadata["source_content_region"] = region
        (destination / "metadata.json").write_text(json.dumps(metadata))
    return augment


def _evaluate(augment, source, previous, workdir):
    first = previous["clips"][0]
    attribute = first["attribute_verification"]
    temporal = first["temporal_consistency"]
    appearance = first["appearance_fidelity"]
    return evaluate_run(
        augment_uri=str(augment),
        original_video=str(source),
        configs_uri=str(workdir / "configs"),
        output_uri=str(workdir / "grade"),
        storage=object(),
        alignment_mode="required",
        threshold=previous["threshold"],
        attribute_threshold=previous["attribute_threshold"],
        question_model=attribute["question_model"],
        vlm_model=attribute["vlm_model"],
        attribute_evidence_mode=previous["attribute_evidence_mode"],
        attribute_sample_policy=previous["attribute_sample_policy"],
        temporal_mode=previous["temporal_mode"],
        temporal_noise_floor=temporal["noise_floor"],
        temporal_threshold=temporal["threshold"],
        temporal_blur_ksize=temporal["blur_ksize"],
        appearance_mode=previous["appearance_mode"],
        appearance_threshold=appearance["threshold"],
        appearance_luminance_tolerance=appearance["luminance_tolerance"],
        appearance_global_chroma_tolerance=appearance["global_chroma_tolerance"],
        appearance_local_chroma_tolerance=appearance["local_chroma_tolerance"],
        appearance_chroma_instability_tolerance=appearance[
            "chroma_instability_tolerance"
        ],
        appearance_blur_ksize=appearance["blur_ksize"],
        appearance_max_dimension=appearance["max_dimension"],
    )


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "case", _cases(), ids=lambda case: "retained-run" if case else "disabled"
)
def test_retained_videos_use_scene_scoring_with_live_vlm(case, tmp_path):
    """Verify live evidence without changing retained videos or their old grades.

    Args:
        case: Private run location, label and expected decoded frame count.
        tmp_path: Isolated staging directory.
    Returns:
        None.
    Raises:
        AssertionError: Scoring did not complete or media lineage changed.
    """
    from npa.clients.project_credentials import s3_client_for_project

    client = s3_client_for_project(os.environ["NPA_E2E_PROJECT"])
    root = case["run_uri"]
    source = _download(client, root, "input/source.mp4", tmp_path / "source.mp4")
    region = _retained_region(client, root, source, tmp_path)
    augment = _stage_variants(client, root, region, tmp_path)
    previous = _read(client, root, "grade/cosmos_evaluator.json")
    configuration = tmp_path / "configs"
    configuration.mkdir()
    (configuration / "manifest.json").write_text(
        json.dumps(_read(client, root, "configs/manifest.json"))
    )
    result = _evaluate(augment, source, previous, tmp_path)
    _save_evidence(case, result, previous, augment, source)
    assert result.status == "completed" and result.clip_count == previous["clip_count"]
    _verify_lineage(result, previous, case["expected_frames"])


def _save_evidence(case, result, previous, augment, source):
    output = Path(os.environ["NPA_PAIDF_PADDING_EVIDENCE_DIR"])
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    label = case["label"]
    assert label.isascii() and label.replace("-", "").isalnum()
    (output / f"{label}.json").write_text(
        json.dumps(
            {
                "region_reconstructed_from_retained_preparation": True,
                "previous": previous,
                "scene_evaluation": result.to_dict(),
                "matched_full_frame_hallucination": _full_frame_baseline(
                    augment, source, result
                ),
            },
            indent=2,
        )
    )


def _full_frame_baseline(augment, source, result):
    results = {}
    for clip in result.clips:
        baseline = check_hallucination(
            clip_id=clip.clip_id,
            original_video=source,
            augmented_video=augment / clip.clip_id / "augmented_video.mp4",
            threshold=result.threshold,
        )
        assert baseline.engine == clip.hallucination["engine"]
        results[clip.clip_id] = baseline.to_dict()
    return results


def _verify_lineage(result, previous, expected_frames):
    before = {clip["clip_id"]: clip for clip in previous["clips"]}
    for clip in result.clips:
        spatial = clip.spatial_evidence
        assert spatial["mode"] == "prepared-scene"
        assert spatial["decoded_frames"] == expected_frames
        assert (
            spatial["generated_sha256"]
            == before[clip.clip_id]["temporal_alignment"]["generated_sha256"]
        )
        assert spatial["source_sha256"] == clip.temporal_alignment["source_sha256"]
        assert spatial["padding"]["included_in_scene_score"] is False
        assert clip.attribute_verification and clip.hallucination
        assert (
            clip.hallucination["engine"]
            == before[clip.clip_id]["hallucination"]["engine"]
        )
        assert clip.temporal_consistency and clip.appearance_fidelity
