"""Run pinned native Cosmos 3 transfer with explicit text and video guardrails.

Uses NVIDIA cosmos-framework's OpenMDW-1.1 transfer implementation. The Canny
preprocessor follows its AddControlInputEdge presets without importing
unrelated training preprocessors. See skills/NOTICE-NVIDIA-COSMOS3.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from npa.workflows.paidf_cosmos3_media import HEIGHT, WIDTH, video_sha256
from npa.workbench.cosmos.structural_transfer import TransferSettings, edge_thresholds


class _GuardedTransferModel:
    def __init__(self, pipe: Any, name: str):
        if pipe.guardrails is None:
            raise ValueError("Structural transfer requires initialized model guardrails")
        self._pipe = pipe
        self._model = pipe.model
        self._name = name
        self.checked_prompts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def generate_samples_from_batch(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        prompts = batch[self._model.input_caption_key]
        if not isinstance(prompts, list) or len(prompts) != 1 or not isinstance(prompts[0], str):
            raise ValueError("Native transfer did not supply one explicit effective prompt")
        self._pipe._run_text_guardrail(self._name, prompts[0])
        self.checked_prompts.append(prompts[0])
        return self._model.generate_samples_from_batch(batch, **kwargs)


def _write_edges(source: Path, destination: Path, fps: int, preset: str = "medium") -> tuple[int, str]:
    import cv2

    lower, upper = edge_thresholds(preset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    digest, count = hashlib.sha256(), 0
    argv = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "ffv1", "-pix_fmt", "gray", str(destination)]
    try:
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                if frame.shape[:2] != (HEIGHT, WIDTH):
                    raise ValueError("Source dimensions changed during edge extraction")
                edge = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), lower, upper).tobytes()
                encoder.stdin.write(edge)
                digest.update(edge)
                count += 1
            encoder.stdin.close()
            if encoder.wait() or count < 6:
                raise ValueError("Complete source edge extraction failed")
    finally:
        capture.release()
    return count, digest.hexdigest()


def _verify_control(sample: Any, count: int, digest: str) -> dict[str, Any]:
    import torch
    from cosmos_framework.inference.args import TransferHintKey
    from cosmos_framework.inference.transfer import load_transfer_control_frames

    control = sample.transfer_hints[TransferHintKey.EDGE]
    lower, upper = edge_thresholds(control.preset_edge_threshold)
    frames = load_transfer_control_frames(hint_key=TransferHintKey.EDGE, transfer=control,
        resolution=sample.resolution, aspect_ratio=sample.aspect_ratio, max_frames=sample.max_frames)
    if frames.dtype != torch.uint8 or tuple(frames.shape) != (3, count, HEIGHT, WIDTH):
        raise ValueError("Native transfer loader changed source-control shape or frame count")
    if not torch.equal(frames[0], frames[1]) or not torch.equal(frames[0], frames[2]):
        raise ValueError("Native transfer loader changed edge channels")
    actual = hashlib.sha256(frames[0].contiguous().cpu().numpy().tobytes()).hexdigest()
    if actual != digest or count != sample.max_frames:
        raise ValueError("Native transfer loader changed source-control pixels or coverage")
    return {"control_path": str(control.control_path), "control_sha256": video_sha256(Path(control.control_path)),
            "control_pixels_sha256": digest, "source_frames": count,
            "control_loader_verified": True, "edge_algorithm": f"opencv-canny-{lower}-{upper}",
            "edge_threshold": str(control.preset_edge_threshold)}


def _write_rgb(source: Path, destination: Path, fps: int) -> tuple[int, str]:
    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    digest, count = hashlib.sha256(), 0
    argv = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "ffv1", "-pix_fmt", "bgr0", str(destination)]
    try:
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                if frame.shape[:2] != (HEIGHT, WIDTH):
                    raise ValueError("Source dimensions changed during RGB control extraction")
                pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
                encoder.stdin.write(pixels)
                digest.update(pixels)
                count += 1
            encoder.stdin.close()
            if encoder.wait() or count < 6:
                raise ValueError("Complete source RGB control extraction failed")
    finally:
        capture.release()
    return count, digest.hexdigest()


def _verify_rgb(sample: Any, count: int, digest: str) -> dict[str, Any]:
    import torch
    from cosmos_framework.inference.args import TransferHintKey
    from cosmos_framework.inference.transfer import load_transfer_control_frames

    control = sample.transfer_hints[TransferHintKey.BLUR]
    TransferSettings(rgb_weight=control.weight).validate()
    if str(control.preset_blur_strength) != "none" or not control.weight:
        raise ValueError("Native RGB conditioning changed its preset or positive weight")
    frames = load_transfer_control_frames(hint_key=TransferHintKey.BLUR, transfer=control,
        resolution=sample.resolution, aspect_ratio=sample.aspect_ratio, max_frames=sample.max_frames)
    if frames.dtype != torch.uint8 or tuple(frames.shape) != (3, count, HEIGHT, WIDTH):
        raise ValueError("Native transfer loader changed RGB control shape or frame count")
    pixels = frames.permute(1, 2, 3, 0).contiguous().cpu().numpy().tobytes()
    if hashlib.sha256(pixels).hexdigest() != digest or count != sample.max_frames:
        raise ValueError("Native transfer loader changed RGB control pixels or coverage")
    return {"native_hint": "blur", "preset": str(control.preset_blur_strength),
            "weight": float(control.weight), "control_path": str(control.control_path),
            "control_sha256": video_sha256(Path(control.control_path)),
            "control_pixels_sha256": digest, "source_frames": count,
            "control_loader_verified": True, "output_pixel_blending": False}


def _save_guarded_output(pipe: Any, sample: Any, generated: Any, prompts: list[str], control: dict[str, Any]) -> None:
    from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs
    from cosmos_framework.inference.inference import save_img_or_video

    if not prompts or pipe.guardrails is None:
        raise ValueError("Structural transfer did not run the text guardrail")
    video = ((1.0 + generated.output_video.squeeze(0)) / 2).clamp(0, 1)
    video = pipe._run_video_guardrail(sample.name, video)
    if video is None or tuple(video.shape) != (3, control["source_frames"], HEIGHT, WIDTH):
        raise ValueError("Guarded transfer output does not cover the complete source")
    if abs(float(generated.fps) - float(sample.fps)) > 0.0001:
        raise ValueError("Transfer changed the prepared-source frame rate")
    destination = sample.output_dir / "vision.mp4"
    save_img_or_video(video, str(destination.with_suffix("")), fps=generated.fps, quality=sample.video_save_quality)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise ValueError("Guarded transfer produced no generated video")
    evidence = {"schema": "npa.cosmos3.structural-transfer.v1", **control,
                "text_guardrail_passed": True, "video_guardrail_passed": True,
                "guardrail_postprocessing_applied": True, "effective_prompts": prompts,
                "native_chunks": len(prompts), "output_fps": generated.fps,
                "first_chunk_conditional_frames": sample.num_first_chunk_conditional_frames,
                "overlap_conditional_frames": sample.num_conditional_frames,
                "normalize_cfg": sample.normalize_cfg,
                "native_torch_compile": False}
    (sample.output_dir / "transfer_evidence.json").write_text(json.dumps(evidence, indent=2))
    result = SampleOutputs(args=sample.model_dump(mode="json"),
                           outputs=[SampleOutput(content={"structural_transfer": True}, files=[destination])])
    (sample.output_dir / "sample_outputs.json").write_text(result.model_dump_json())


def _prepare_controls(input_files: list[Path]) -> dict[str, dict[str, tuple[int, str]]]:
    controls = {}
    for path in input_files:
        fields = json.loads(Path(path).read_text())
        control = fields.get("edge") or {}
        if not control.get("control_path"):
            raise ValueError("An explicit Canny control path is required")
        preset = control.get("preset_edge_threshold")
        edge_thresholds(preset)
        edge = _write_edges(
            Path(fields["vision_path"]), Path(control["control_path"]), int(fields["fps"]), preset
        )
        controls[fields["name"]] = {"edge": edge}
        rgb = fields.get("blur")
        if rgb is not None:
            if not isinstance(rgb, dict) or not rgb.get("control_path"):
                raise ValueError("An explicit RGB control path is required")
            TransferSettings(rgb_weight=rgb["weight"]).validate()
            if rgb.get("preset_blur_strength") != "none" or not rgb["weight"]:
                raise ValueError("RGB conditioning requires the native none blur preset and positive weight")
            prepared = _write_rgb(Path(fields["vision_path"]), Path(rgb["control_path"]), int(fields["fps"]))
            if prepared[0] != edge[0]:
                raise ValueError("RGB and edge controls do not cover the same source frames")
            controls[fields["name"]]["rgb"] = prepared
    return controls


def _run_samples(args: Any) -> None:
    from cosmos_framework.inference.transfer import generate_transfer_sample

    # Native transfer's NATTEN attention has data-dependent sizes that Dynamo
    # cannot guard. Use the framework's supported eager path for this adapter.
    args.setup.use_torch_compile = False
    setup = args.setup.build_setup()
    if not setup.guardrails:
        raise ValueError("Structural transfer requires guardrails; no model was loaded")
    controls = _prepare_controls(args.input_files)
    overrides = setup.get_sample_overrides_cls().from_files(args.input_files, overrides=setup.sample_overrides)
    for item in overrides:
        item.output_dir = setup.output_dir / item.name
        item.output_dir.mkdir(parents=True, exist_ok=True)
        item.download(item.output_dir / "inputs")
    pipe = setup.get_inference_cls().create(setup)
    for item in overrides:
        sample = item.build_sample(model_config=pipe.model_config)
        control = _verify_control(sample, *controls[item.name]["edge"])
        if "rgb" in controls[item.name]:
            control["rgb_conditioning"] = _verify_rgb(sample, *controls[item.name]["rgb"])
        (sample.output_dir / "sample_args.json").write_text(sample.model_dump_json())
        model = _GuardedTransferModel(pipe, sample.name)
        generated = generate_transfer_sample(sample_args=sample, model=model)
        _save_guarded_output(pipe, sample, generated, model.checked_prompts, control)


def main() -> None:
    """Run guarded structural transfer with the native inference CLI arguments.

    Args:
        None; native argument parsing reads sys.argv.
    Returns:
        None.
    Raises:
        ValueError: Media, controls, model guardrails or generation fail.
        OSError: Required runtime files or executables are unavailable.
    """
    from cosmos_framework.scripts.inference import InferenceArgs, tyro, tyro_cli

    args = tyro_cli(InferenceArgs, config=(tyro.conf.OmitArgPrefixes,
        tyro.conf.CascadeSubcommandArgs, tyro.conf.OmitSubcommandPrefixes))
    _run_samples(args)


if __name__ == "__main__":
    main()
