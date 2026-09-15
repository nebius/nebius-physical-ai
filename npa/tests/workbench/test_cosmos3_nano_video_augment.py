"""CPU contract proofs; synthetic fixtures are never model-quality evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess

import httpx
import pytest

from npa.workbench.cosmos import nano_video_augment as augment
from npa.workbench.cosmos.nano_video import NanoVideoError


def request(**changes):
    return augment.validate_request({"mode": "augmentation", "request_id": "synthetic-test", "prompt": "A dim warehouse",
        "seed": 2, "source_sha256": hashlib.sha256(b"synthetic source").hexdigest(), "source_bytes": 16, **changes})


@pytest.mark.parametrize("change", [
    {"strength": 0.5}, {"control_weight": 1}, {"edge": {"control_path": "/other"}},
    {"seed": True}, {"seed": -1}, {"seed": 2**63}, {"num_inference_steps": 0},
    {"source_bytes": 1.5}, {"guidance_scale": float("nan")}, {"control_guidance": float("inf")},
    {"flow_shift": False}, {"chunk_frames": 120}, {"chunk_frames": 301}, {"chunk_frames": 5},
    {"num_inference_steps": 201}, {"guidance_scale": 20.01},
    {"edge_threshold": "unknown"}, {"request_id": "../escape"}, {"prompt": " "},
])
def test_unsupported_or_malformed_parameters_fail_closed(change):
    with pytest.raises(augment.AugmentationInputError):
        request(**change)


def test_request_normalization_is_stable_and_does_not_change_transfer_guidance():
    value = request(guidance_scale=3)
    assert value["guidance_scale"] == 3.0
    assert value["control_guidance"] == 1.5
    assert augment.request_sha256(value) == augment.request_sha256(dict(reversed(list(value.items()))))
    assert augment.request_sha256(value) != augment.request_sha256({**value, "control_guidance": 2})
    trimmed = request(prompt="  A dim warehouse\n", negative_prompt="  flicker\n")
    assert trimmed["prompt"] == "A dim warehouse" and trimmed["negative_prompt"] == "flicker"


@pytest.mark.parametrize("frames", [6, 9, 121, 122, 720, 721, 1000])
def test_every_output_frame_is_covered_once_by_its_original_source_interval(frames):
    plan = augment.chunk_plan(frames)
    kept = []
    for row in plan:
        assert row["source_frames"] <= row["model_chunk_frames"] < row["source_frames"] + 4
        assert (row["model_chunk_frames"] - 1) % 4 == 0
        kept.extend(range(row["source_start"] + row["drop_prefix_frames"], row["source_start"] + row["source_frames"]))
    assert kept == list(range(frames))
    if frames == 720:
        assert [row["source_start"] for row in plan] == [0, 116, 232, 348, 464, 580, 696]
        assert plan[-1]["source_frames"] == 24
        assert plan[-1]["model_chunk_frames"] == 25


def test_transfer_fields_use_separate_source_control_and_augmented_rgb_prefix(tmp_path):
    value = request()
    plan = augment.chunk_plan(720)
    for chunk in (plan[0], plan[1], plan[-1]):
        fields = augment.transfer_fields(value, chunk, tmp_path / "original-control.mkv")
        extra = json.loads(fields["extra_params"])
        assert extra["edge"]["control_path"] == str(tmp_path / "original-control.mkv")
        assert extra["num_first_chunk_conditional_frames"] == chunk["drop_prefix_frames"]
        assert extra["num_video_frames_per_chunk"] == chunk["model_chunk_frames"]
        assert extra["max_frames"] == int(fields["num_frames"]) == chunk["source_frames"]
        assert extra["resolution"] == "480" and fields["fps"] == "24"
        assert extra["share_vision_temporal_positions"] is True
        assert not {"strength", "control_weight", "condition_video_keep", "condition_frame_indexes_vision"} & extra.keys()
        assert "max_sequence_length" not in fields


def test_real_codec_and_timestamp_validation_rejects_wrong_source_without_gpu(tmp_path, monkeypatch):
    path = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=832x480:rate=24",
        "-frames:v", "9", "-c:v", "libx264", "-threads", "2", "-crf", "30", str(path)], check=True)
    value = request(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_bytes=path.stat().st_size)
    evidence = augment.validate_source(path, value)
    assert evidence["decoded_frames"] == 9
    assert evidence["duration_seconds"] == evidence["container_duration_seconds"] == 9 / 24
    assert evidence["timestamps_verified"] is True
    monkeypatch.setattr(augment, "DeviceMemorySampler", lambda: pytest.fail("Invalid source reached GPU measurement"))
    with pytest.raises(augment.AugmentationInputError):
        augment.run_augmentation(endpoint="http://localhost", output_dir=tmp_path / "output", input_video=path,
            request={**value, "source_sha256": "0" * 64}, replica_id="test")
    assert not (tmp_path / "output").exists()
    wrong_rate = tmp_path / "wrong-rate.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-r", "30", "-c:v", "libx264",
        "-threads", "2", str(wrong_rate)], check=True)
    with pytest.raises(NanoVideoError):
        augment.validate_media(wrong_rate)


def video(frames, width=832):
    return {"valid": True, "full_decode_passed": True, "decoded_frames": frames, "fps": 24.0,
        "width": width, "height": 480, "duration_seconds": frames / 24, "timestamps_verified": True}


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    """Mock model/codec boundaries, exercising real orchestration and validation."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    value = request()
    calls = []
    memory = []

    class Sampler:
        error = None

        def __init__(self):
            self.samples = memory

        def start(self):
            memory.append({"used_mib": 100})

        def stop(self):
            return {"peak_used_mib": 100, "error": None, "samples": list(memory)}

    monkeypatch.setattr(augment, "DeviceMemorySampler", Sampler)
    monkeypatch.setattr(augment, "validate_source", lambda *args: video(122))
    monkeypatch.setattr(augment, "validate_media", lambda path, frames=None, width=832: video(frames or 122, width))

    def control(original, target, chunk, preset):
        assert original.name == "input.mp4" and original.read_bytes() == source.read_bytes()
        calls.append(("control", chunk["source_start"], chunk["source_frames"]))
        target.write_bytes(f"source-edges-{chunk['source_start']}".encode())
        return {"source_start": chunk["source_start"], "source_frames": chunk["source_frames"],
            "source": "original input.mp4 only", "original_source_sha256": value["source_sha256"],
            "engine": "vllm-omni.cosmos3.transfer.make_edge_control", "preset": preset,
            "canny_thresholds": [100, 200], "source_rgb_sha256": "a" * 64,
            "control_rgb_sha256": "b" * 64, "upstream_module_sha256": "c" * 64,
            "lossless_upstream_readback_equal": True, "control_video": video(chunk["source_frames"])}

    def tail(previous, target, frames):
        calls.append(("tail", previous.name, frames))
        target.write_bytes(b"five augmented frames")
        return video(5)

    def effective(fields, chunk):
        extra = json.loads(fields["extra_params"])
        config = {key: extra[key] for key in ("num_video_frames_per_chunk", "max_frames", "num_conditional_frames",
            "num_first_chunk_conditional_frames", "control_guidance", "share_vision_temporal_positions")}
        config.update(hints={"edge": {**extra["edge"], "key": "edge", "control": None, "preset_blur_strength": "medium"}},
            guidance_scale=float(fields["guidance_scale"]), flow_shift=float(fields["flow_shift"]),
            control_guidance_interval=None, fps=24.0, num_frames=int(fields["num_frames"]), show_input=False, show_control_condition=False)
        return {"transfer_config": config, "positive_prompt": fields["prompt"], "negative_prompt": fields["negative_prompt"],
            "system_prompt": extra["system_prompt"], "sampling": {
                "num_inference_steps": int(fields["num_inference_steps"]), "max_sequence_length": extra["max_sequence_length"],
                "resolution": "480", "fps": 24, "use_system_prompt": True,
                "use_duration_template": False, "use_resolution_template": False}}

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] is None and kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, files, headers):
            fields = {key: pair[1] for key, pair in files.items() if key != "input_reference"}
            reference = files.get("input_reference")
            calls.append(("post", fields, reference[1].read() if reference else None))
            memory.append({"used_mib": 100})
            return httpx.Response(200, content=f"augmented-chunk-{fields['seed']}".encode(), headers={
                "Content-Type": "video/mp4", "X-Inference-Time-S": "1", "X-Peak-Memory-MB": "99", "X-Stage-Durations": "{}"})

    monkeypatch.setattr(augment, "prepare_control", control)
    monkeypatch.setattr(augment, "extract_tail", tail)
    monkeypatch.setattr(augment, "_effective_transfer", effective)
    monkeypatch.setattr(augment.httpx, "Client", Client)
    monkeypatch.setattr(augment, "stitch", lambda chunks, directory, target: target.write_bytes(b"complete augmented output"))
    monkeypatch.setattr(augment, "comparison", lambda original, output, target: target.write_bytes(b"synchronized comparison"))
    return source, value, calls


def test_all_chunks_use_original_controls_and_only_previous_augmented_tail(orchestration, tmp_path):
    source, value, calls = orchestration
    report = augment.run_augmentation(endpoint="http://localhost", output_dir=tmp_path / "result", input_video=source,
        request=value, replica_id="synthetic-replica")
    assert [(row[1], row[2]) for row in calls if row[0] == "control"] == [(0, 121), (116, 6)]
    posts = [row for row in calls if row[0] == "post"]
    assert len(posts) == 2 and posts[0][2] is None and posts[1][2] == b"five augmented frames"
    assert ("tail", "chunk-000.mp4", 121) in calls
    assert json.loads(posts[1][1]["extra_params"])["num_video_frames_per_chunk"] == 9
    augment.validate_report(report, value)
    assert (tmp_path / "result" / "input.mp4").read_bytes() == source.read_bytes()
    assert (tmp_path / "result" / "augmented.mp4").stat().st_mode & 0o777 == 0o400
    for mutation in ("source_interval", "source_identity", "structural_control", "missing_chunk", "wrong_rgb_prefix"):
        invalid = copy.deepcopy(report)
        if mutation == "source_interval":
            invalid["chunks"][1]["source_start"] += 1
        elif mutation == "source_identity":
            invalid["chunks"][1]["control_provenance"]["original_source_sha256"] = "f" * 64
        elif mutation == "structural_control":
            invalid["chunks"][0]["effective"]["transfer_config"]["hints"] = {}
        elif mutation == "missing_chunk":
            invalid["chunks"].pop()
        else:
            invalid["chunks"][1]["effective"]["transfer_config"]["num_first_chunk_conditional_frames"] = 0
        with pytest.raises(NanoVideoError):
            augment.validate_report(invalid, value)


def test_final_evidence_failure_is_durably_failed_and_never_reports_success(orchestration, tmp_path, monkeypatch):
    source, value, calls = orchestration

    def reject(*args):
        raise NanoVideoError("Deliberately invalid terminal evidence")

    monkeypatch.setattr(augment, "validate_report", reject)
    with pytest.raises(NanoVideoError):
        augment.run_augmentation(endpoint="http://localhost", output_dir=tmp_path / "result", input_video=source,
            request=value, replica_id="synthetic-replica")
    report = json.loads((tmp_path / "result" / "report.json").read_text())
    assert report["status"] == "failed" and report["evidence_validation_failed"] is True
    assert len([row for row in calls if row[0] == "post"]) == 2
    assert (tmp_path / "result" / "augmented.mp4").is_file()


@pytest.mark.parametrize("header", ['{"stage": NaN}', '{"stage": -1}', '{"stage": "1"}', '[]'])
def test_malformed_stage_headers_cannot_prevent_durable_failure(orchestration, tmp_path, monkeypatch, header):
    source, value, _ = orchestration
    post = augment.httpx.Client.post

    def malformed(self, *args, **kwargs):
        response = post(self, *args, **kwargs)
        response.headers["X-Stage-Durations"] = header
        return response

    monkeypatch.setattr(augment.httpx.Client, "post", malformed)
    with pytest.raises(NanoVideoError, match="stage-duration"):
        augment.run_augmentation(endpoint="http://localhost", output_dir=tmp_path / "result", input_video=source,
            request=value, replica_id="synthetic-replica")
    report = json.loads((tmp_path / "result" / "report.json").read_text())
    assert report["status"] == report["chunks"][0]["status"] == "failed"
    assert "stage_durations" not in report["chunks"][0]
    assert json.loads((tmp_path / "result" / "error-000.json").read_text())["invalid_stage_durations_header"] == header
