"""Exercise real media correspondence and fail-closed native transfer integration."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.cosmos import structural_transfer as transfer
from npa.workbench.cosmos import structural_transfer_runner as native
from npa.workflows import paidf_cosmos3_media as media


def _video(path, *, fps=50, frames=169, size="640x480"):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real media tests require ffmpeg and ffprobe")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=s={size}:r={fps}", "-frames:v", str(frames),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def prepared(tmp_path):
    original = _video(tmp_path / "original.mp4")
    output = tmp_path / "source.mp4"
    recipe = media.prepare_reference(original, output)
    return original, output, recipe


@pytest.mark.parametrize("contents", [b"", bytes(range(256)) * 8192 + b"last bytes"],
                         ids=["empty", "multi-chunk"])
def test_video_hash_supports_cpu_images_without_file_digest(tmp_path, monkeypatch, contents):
    path = tmp_path / "video.mp4"
    path.write_bytes(contents)
    expected = media.hashlib.sha256(contents).hexdigest()
    monkeypatch.delattr(media.hashlib, "file_digest", raising=False)
    assert media.video_sha256(path) == expected


def test_normalization_preserves_complete_duration_and_provenance(prepared):
    original, output, recipe = prepared
    assert recipe["original"]["decoded_frames"] == 169
    assert recipe["prepared"]["decoded_frames"] == 81
    assert recipe["prepared"]["duration_seconds"] == pytest.approx(3.375)
    assert abs(recipe["original"]["duration_seconds"] - 3.375) <= 1 / 24
    assert recipe["original"]["sha256"] == media.video_sha256(original)
    assert recipe["prepared"]["sha256"] == media.video_sha256(output)
    assert recipe["time_stretch"] is False
    evidence = media.verify_pair(output, output)
    assert evidence["decoded_frames"] == 81
    assert evidence["full_decode_passed"] is True
    assert evidence["visual_quality_evaluated"] is False


@pytest.mark.parametrize("frames,count,indices", [
    (288, 8, [0, 41, 82, 123, 164, 205, 246, 287]),
    (3, 8, [0, 1, 2]),
    (24, 1, [0]),
])
def test_caption_frames_cover_complete_video_without_repetition(tmp_path, frames, count, indices):
    from PIL import Image
    from npa.workflows.paidf_cosmos3 import _extract_frames

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real frame extraction requires ffmpeg and ffprobe")
    source = tmp_path / "frame-coded.mkv"
    colors = np.array([[index % 256, index // 256, 127] for index in range(frames)], dtype=np.uint8)
    pixels = np.broadcast_to(colors[:, None, None, :], (frames, 48, 64, 3)).tobytes()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "64x48", "-r", "24", "-i", "pipe:0", "-c:v", "ffv1",
                    "-pix_fmt", "bgr0", str(source)], input=pixels, check=True)
    extracted = _extract_frames(source, tmp_path / "captions", count=count)
    assert len(extracted) == len(indices)
    for path, index in zip(extracted, indices, strict=True):
        with Image.open(path) as image:
            actual = np.asarray(image.convert("RGB"))
        assert actual.shape == (48, 64, 3)
        assert np.all(actual == colors[index])


@pytest.mark.parametrize("fps,frames,size", [(24, 80, "832x480"), (25, 81, "832x480"), (24, 81, "640x480")])
def test_mismatched_output_never_passes_alignment(prepared, tmp_path, fps, frames, size):
    _, source, _ = prepared
    generated = _video(tmp_path / "generated.mp4", fps=fps, frames=frames, size=size)
    with pytest.raises(media.VideoAlignmentError):
        media.verify_pair(source, generated)


@pytest.mark.parametrize("content", [b"", b"not a video"])
def test_invalid_media_refused_before_generation(tmp_path, content):
    path = tmp_path / "bad.mp4"
    path.write_bytes(content)
    with pytest.raises(media.VideoAlignmentError):
        media.probe_video(path)


@pytest.mark.parametrize("preset", ["very_low", "low", "medium", "high", "very_high"])
@pytest.mark.parametrize("rgb_weight", [0.0, 0.5])
@pytest.mark.parametrize("first_frames", [0, 1])
def test_native_sample_uses_all_source_controls_and_explicit_seed(prepared, tmp_path, preset, rgb_weight, first_frames):
    _, source, _ = prepared
    sample = transfer.transfer_sample({"name": "sample", "model_mode": "video2video",
                                       "vision_path": str(source)}, transfer.TransferSettings(edge_threshold=preset, rgb_weight=rgb_weight,
                                                                                             first_chunk_conditional_frames=first_frames), tmp_path, 17)
    assert sample["max_frames"] == 81
    assert sample["seed"] == 17
    assert sample["fps"] == 24
    assert sample["num_video_frames_per_chunk"] == 93
    assert sample["edge"]["preset_edge_threshold"] == preset
    assert sample["num_first_chunk_conditional_frames"] == first_frames
    assert sample["num_conditional_frames"] == 5
    assert sample["show_input"] is sample["show_control_condition"] is False
    if rgb_weight:
        assert sample["blur"] == {"control_path": str(tmp_path / "controls/sample-rgb.mkv"),
                                  "preset_blur_strength": "none", "weight": rgb_weight}
    else:
        assert "blur" not in sample


@pytest.mark.parametrize("values", [{"fps": 50}, {"fps": True}, {"chunk_frames": 94},
                                    {"control_guidance": float("nan")}, {"control_guidance": 10.1},
                                    {"control_guidance": "1.5"}, {"control_guidance": False},
                                    {"edge_threshold": "auto"}, {"edge_threshold": None},
                                    {"edge_threshold": []}, {"rgb_weight": -0.1},
                                    {"rgb_weight": float("nan")}, {"rgb_weight": float("inf")},
                                    {"rgb_weight": True}, {"rgb_weight": "0.5"},
                                    {"first_chunk_conditional_frames": -1}, {"first_chunk_conditional_frames": 2},
                                    {"first_chunk_conditional_frames": True}, {"first_chunk_conditional_frames": 0.0},
                                    {"first_chunk_conditional_frames": "0"}])
def test_unsupported_native_controls_are_rejected(values):
    with pytest.raises(ValueError):
        transfer.TransferSettings(**values).validate()


@pytest.mark.parametrize("preset,thresholds", [("very_low", (20, 50)), ("low", (50, 100)),
                                             ("medium", (100, 200)), ("high", (200, 300)),
                                             ("very_high", (300, 400))])
def test_real_edge_video_retains_exact_preset_pixels(prepared, tmp_path, preset, thresholds):
    cv2 = pytest.importorskip("cv2")
    _, source, _ = prepared
    destination = tmp_path / "control.mkv"
    count, digest = native._write_edges(source, destination, 24, preset)
    original, control = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(destination))
    expected_hash = hashlib.sha256()
    decoded = 0
    try:
        while True:
            ok, frame = original.read()
            read, actual = control.read()
            assert ok == read
            if not ok:
                break
            expected = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), *thresholds)
            assert np.array_equal(actual, np.repeat(expected[:, :, None], 3, axis=2))
            expected_hash.update(expected.tobytes())
            decoded += 1
    finally:
        original.release()
        control.release()
    assert decoded == count == 81
    assert digest == expected_hash.hexdigest()


def test_rgb_control_retains_all_source_colors_and_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = _video(tmp_path / "source.mp4", fps=24, frames=9, size="832x480")
    destination = tmp_path / "rgb.mkv"
    count, digest = native._write_rgb(source, destination, 24)
    original, control = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(destination))
    expected_hash = hashlib.sha256()
    decoded = 0
    try:
        while True:
            ok, frame = original.read()
            read, actual = control.read()
            assert ok == read
            if not ok:
                break
            assert np.array_equal(actual, frame)
            expected_hash.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            decoded += 1
    finally:
        original.release()
        control.release()
    assert decoded == count == 9
    assert digest == expected_hash.hexdigest()
    media.validate_reference(media.probe_video(destination), 24)


def test_native_rgb_verification_rejects_changed_pixels(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pixels = np.zeros((9, 480, 832, 3), dtype=np.uint8)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 13, 97, 211
    frames = torch.from_numpy(pixels).permute(3, 0, 1, 2)
    control_path = tmp_path / "rgb.mkv"
    control_path.write_bytes(b"native-loader-seam")
    control = SimpleNamespace(control_path=control_path, preset_blur_strength="none", weight=0.5)
    sample = SimpleNamespace(transfer_hints={"blur": control}, resolution="480", aspect_ratio="16,9", max_frames=9)
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.args", SimpleNamespace(TransferHintKey=SimpleNamespace(BLUR="blur")))
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.transfer", SimpleNamespace(load_transfer_control_frames=lambda **kw: frames))
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    evidence = native._verify_rgb(sample, 9, digest)
    assert evidence["weight"] == 0.5 and evidence["control_loader_verified"] is True
    assert evidence["output_pixel_blending"] is False
    pixels[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="RGB control pixels"):
        native._verify_rgb(sample, 9, digest)


def test_effective_prompt_is_checked_before_each_native_chunk():
    events = []
    model = SimpleNamespace(input_caption_key="caption",
                            generate_samples_from_batch=lambda batch, **kw: events.append("generate"))
    pipe = SimpleNamespace(guardrails=object(), model=model,
                           _run_text_guardrail=lambda name, prompt: events.append(prompt))
    guarded = native._GuardedTransferModel(pipe, "sample")
    guarded.generate_samples_from_batch({"caption": ["effective action prompt"]})
    assert events == ["effective action prompt", "generate"]
    assert guarded.checked_prompts == ["effective action prompt"]


def test_blocked_prompt_prevents_model_invocation():
    events = []

    def deny(*args):
        raise ValueError("blocked")

    model = SimpleNamespace(input_caption_key="caption", generate_samples_from_batch=lambda *a, **kw: events.append(1))
    pipe = SimpleNamespace(guardrails=object(), model=model, _run_text_guardrail=deny)
    with pytest.raises(ValueError, match="blocked"):
        native._GuardedTransferModel(pipe, "sample").generate_samples_from_batch({"caption": ["prompt"]})
    assert not events
    pipe.guardrails = None
    with pytest.raises(ValueError, match="initialized"):
        native._GuardedTransferModel(pipe, "sample")


class _Video(np.ndarray):
    def clamp(self, minimum, maximum):
        return self.clip(minimum, maximum)


def _install_save_contract(monkeypatch, saved):
    def save(video, destination, **kwargs):
        saved.append(video)
        Path(destination + ".mp4").write_bytes(b"unit-test-video")

    class Outputs:
        def __init__(self, **kwargs):
            self.args = kwargs

        def model_dump_json(self):
            return "{}"

    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.common.args",
                        SimpleNamespace(SampleOutput=lambda **kw: kw, SampleOutputs=Outputs))
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.inference", SimpleNamespace(save_img_or_video=save))


def test_saved_output_is_guardrail_postprocessed_tensor(monkeypatch, tmp_path):
    saved = []
    _install_save_contract(monkeypatch, saved)
    processed = np.zeros((3, 6, 480, 832), dtype=np.float32)
    pipe = SimpleNamespace(guardrails=object(), _run_video_guardrail=lambda *a: processed)
    sample = SimpleNamespace(name="sample", fps=24, output_dir=tmp_path, video_save_quality=5,
                             num_first_chunk_conditional_frames=0, num_conditional_frames=5,
                             model_dump=lambda **kw: {})
    generated = SimpleNamespace(output_video=np.ones((1, 3, 6, 480, 832), dtype=np.float32).view(_Video), fps=24)
    native._save_guarded_output(pipe, sample, generated, ["checked"], {"source_frames": 6})
    assert saved[0] is processed
    evidence = json.loads((tmp_path / "transfer_evidence.json").read_text())
    assert evidence["guardrail_postprocessing_applied"] is True
    assert evidence["native_chunks"] == 1
    assert evidence["native_torch_compile"] is False
    assert evidence["first_chunk_conditional_frames"] == 0
    assert evidence["overlap_conditional_frames"] == 5


@pytest.mark.parametrize("include_rgb", [False, True])
def test_native_parser_sees_real_control_created_first(monkeypatch, tmp_path, include_rgb):
    input_file = tmp_path / "sample.json"
    control = tmp_path / "edges.mkv"
    rgb = tmp_path / "rgb.mkv"
    fields = {"name": "sample", "vision_path": "source.mp4", "fps": 24,
              "edge": {"control_path": str(control), "preset_edge_threshold": "low"}}
    if include_rgb:
        fields["blur"] = {"control_path": str(rgb), "preset_blur_strength": "none", "weight": 0.5}
    input_file.write_text(json.dumps(fields))

    def edges(source, destination, fps, preset):
        assert preset == "low"
        destination.write_bytes(b"real-control-seam")
        return 81, "digest"

    def parse(*args, **kwargs):
        assert control.read_bytes() == b"real-control-seam"
        if include_rgb:
            assert rgb.read_bytes() == b"rgb-control-seam"
        raise RuntimeError("reached native parser after preprocessing")

    def colors(source, destination, fps):
        destination.write_bytes(b"rgb-control-seam")
        return 81, "rgb-digest"

    monkeypatch.setattr(native, "_write_rgb", colors)

    setup = SimpleNamespace(guardrails=True, sample_overrides={}, get_sample_overrides_cls=lambda: SimpleNamespace(from_files=parse))
    def build_setup():
        assert args.setup.use_torch_compile is False
        return setup

    args = SimpleNamespace(input_files=[input_file], setup=SimpleNamespace(use_torch_compile=True, build_setup=build_setup))
    monkeypatch.setattr(native, "_write_edges", edges)
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.transfer", SimpleNamespace(generate_transfer_sample=None))
    with pytest.raises(RuntimeError, match="after preprocessing"):
        native._run_samples(args)


def test_control_video_cannot_be_selected_as_generated_artifact(tmp_path):
    (tmp_path / "control_edge.mp4").write_bytes(b"large-control" * 100)
    generated = tmp_path / "vision.mp4"
    generated.write_bytes(b"smaller-generated")
    outputs = {"status": "success", "outputs": [{"files": [str(generated)]}]}
    (tmp_path / "sample_outputs.json").write_text(json.dumps(outputs))
    evidence = {"schema": "npa.cosmos3.structural-transfer.v1", "text_guardrail_passed": True,
                "video_guardrail_passed": True, "guardrail_postprocessing_applied": True}
    (tmp_path / "transfer_evidence.json").write_text(json.dumps(evidence))
    assert transfer.transfer_artifact(tmp_path)[0] == generated
    evidence["video_guardrail_passed"] = False
    (tmp_path / "transfer_evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="guarded generation"):
        transfer.transfer_artifact(tmp_path)


@pytest.mark.parametrize("failure", ["missing", "stale-hash"])
def test_required_alignment_stops_before_vlm_even_with_low_threshold(prepared, tmp_path, failure):
    from npa.workbench.cosmos_evaluator.evaluate import evaluate_run

    _, source, _ = prepared
    variant = tmp_path / "augment/variant"
    variant.mkdir(parents=True)
    shutil.copyfile(source, variant / "augmented_video.mp4")
    alignment = media.verify_pair(source, source)
    metadata = {"variables": {"lighting": "bright"}, "structural_control": "edge", "input_conditioned": True,
                "conditioned_input": "source.mp4", "published_video_sha256": alignment["generated_sha256"]}
    if failure == "stale-hash":
        metadata["temporal_alignment"] = {**alignment, "generated_sha256": "0" * 64}
    (variant / "metadata.json").write_text(json.dumps(metadata))
    calls = []
    result = evaluate_run(augment_uri=str(variant.parent), output_uri=str(tmp_path / "grade"),
                          input_uri=str(source), threshold=0.01, attribute_threshold=0.01,
                          alignment_mode="required", storage=object(),
                          client=SimpleNamespace(chat_completion_text=lambda **kw: calls.append(kw)))
    assert result.status == "degraded"
    assert result.passed is False
    assert result.clips[0].temporal_alignment["status"] == "failed"
    assert calls == []


def test_caption_input_must_match_evaluated_video(prepared, tmp_path):
    from npa.workflows.paidf_cosmos3 import PaidfCosmos3Error
    from npa.workflows.paidf_cosmos3_annotation import _caption_variant

    _, source, _ = prepared
    variant = {"clip": "variant", "augmented_video_uri": str(source),
               "temporal_alignment": {"generated_sha256": "0" * 64}}
    calls = []
    with pytest.raises(PaidfCosmos3Error, match="no longer matches"):
        _caption_variant(variant, str(tmp_path / "output"), "model", 8, 512, None,
                         lambda **kw: calls.append(kw))
    assert calls == []


def test_captions_use_fresh_frames_from_verified_video(prepared, tmp_path):
    from npa.workbench.token_factory import CaptionItem
    from npa.workflows.paidf_cosmos3_annotation import _caption_variant

    _, source, _ = prepared
    (source.parent / "frame-stale.png").write_bytes(b"stale frame must not be used")
    variant = {"clip": "variant", "augmented_video_uri": str(source),
               "temporal_alignment": {"generated_sha256": media.video_sha256(source)}}

    def captioner(**kwargs):
        frames = sorted(Path(kwargs["input_path"]).glob("*.png"))
        assert frames
        assert all(frame.read_bytes().startswith(b"\x89PNG") for frame in frames)
        return SimpleNamespace(status="completed", captions=[CaptionItem(frame.name, "robot frame") for frame in frames])

    record, captions = _caption_variant(variant, str(tmp_path / "output"), "model", 8, 512, None, captioner)
    assert record["image_count"] == len(captions) == 8
    assert all(item["image"].startswith("variant/") for item in captions)


@pytest.mark.parametrize("failure", ["omitted-variant", "duplicate-image", "incorrect-count", "empty-caption", "stale-hash"])
def test_final_caption_coverage_rejects_incomplete_or_stale_evidence(failure):
    from npa.workflows.paidf_cosmos3 import PaidfCosmos3Error
    from npa.workflows.paidf_cosmos3_annotation import _validate_caption_coverage

    expected = {"variant-0": "a" * 64, "variant-1": "b" * 64}
    report = {"status": "completed", "image_count": 2,
              "variants": [{"clip": clip, "video_sha256": digest, "image_count": 1} for clip, digest in expected.items()],
              "captions": [{"image": clip + "/frame.png", "caption": "robot frame"} for clip in expected]}
    _validate_caption_coverage(report, expected)
    if failure == "omitted-variant":
        report["captions"].pop()
    elif failure == "duplicate-image":
        report["captions"][1] = report["captions"][0]
    elif failure == "incorrect-count":
        report["variants"][0]["image_count"] = 2
    elif failure == "empty-caption":
        report["captions"][0]["caption"] = " "
    else:
        report["variants"][0]["video_sha256"] = "c" * 64
    with pytest.raises(PaidfCosmos3Error):
        _validate_caption_coverage(report, expected)


@pytest.mark.parametrize("failure", ["missing-schema", "missing-score", "wrong-decision", "low-score", "rejection-reasons"])
def test_annotation_rejects_invalid_disposition_before_captioning(monkeypatch, failure):
    from npa.workflows import paidf_cosmos3_annotation as annotation
    from npa.workflows.paidf_cosmos3 import PaidfCosmos3Error, QUALITY_DISPOSITION_SCHEMA

    disposition = {"schema": QUALITY_DISPOSITION_SCHEMA, "quality_status": "accepted",
                   "decision": "promote_checkpoint", "evaluator_status": "completed",
                   "score": 0.8, "threshold": 0.5, "hard_checks_passed": True, "reasons": []}
    if failure.startswith("missing-"):
        del disposition[failure.removeprefix("missing-")]
    elif failure == "wrong-decision":
        disposition["decision"] = "loop_back"
    elif failure == "low-score":
        disposition["score"] = 0.2
    else:
        disposition["reasons"] = ["required check failed"]
    documents = {"manifest.json": {}, "cosmos_evaluator.json": {"status": "completed", "passed": True},
                 "quality_disposition.json": disposition}
    monkeypatch.setattr(annotation, "_read_json", lambda uri, **kwargs: documents[uri.rsplit("/", 1)[-1]])
    monkeypatch.setattr(annotation, "validate_committed_augment_manifest", lambda *args: [{"clip": "variant"}])
    calls = []
    with pytest.raises(PaidfCosmos3Error, match="quality disposition"):
        annotation.annotate_accepted("s3://example-bucket/run/", "unused/", "model",
                                     captioner=lambda **kwargs: calls.append(kwargs))
    assert calls == []
