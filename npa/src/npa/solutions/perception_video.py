"""Native GPU depth and prompted mask propagation over attributed input videos."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import tempfile

from npa.solutions.video_generation import _decoded_frames, cuda_inventory, validate_video


DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REF = "5426e4f0f36572d16453bbda7a8389317b1bef99"
SAM_MODEL = "facebook/sam2.1-hiera-small"
SAM_REF = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
SAM_SOURCE_REF = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def _frames(video):
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < 2 or fps <= 0:
        raise ValueError("Perception requires a decodable moving video")
    return frames, fps


def _depth(frames):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, revision=DEPTH_REF)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, revision=DEPTH_REF).to("cuda").eval()
    values = []
    for frame in frames:
        inputs = processor(images=Image.fromarray(frame), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            predicted = model(**inputs).predicted_depth
            depth = torch.nn.functional.interpolate(
                predicted.unsqueeze(1), size=(540, 960), mode="bicubic", align_corners=False,
            )[0, 0]
        values.append(depth.float().cpu().numpy())
    result = np.stack(values)
    if not np.isfinite(result).all() or float(result.std()) < 0.001:
        raise RuntimeError("Depth inference must produce finite, nonuniform relative depth")
    return result


def _masks(frames, box):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = hf_hub_download(SAM_MODEL, "sam2.1_hiera_small.pt", revision=SAM_REF)
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint, device="cuda")
    masks = {}
    with tempfile.TemporaryDirectory(prefix="npa-sam-frames-") as directory:
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(Path(directory) / f"{index:06d}.jpg")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=directory, offload_video_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=np.asarray(box, dtype=np.float32))
            for index, object_ids, logits in predictor.propagate_in_video(state):
                if list(object_ids) != [1]:
                    raise RuntimeError("Unexpected SAM object identity")
                masks[index] = (logits[0, 0] > 0).cpu().numpy()
    if set(masks) != set(range(len(frames))):
        raise RuntimeError("SAM must propagate the selected object through every frame")
    return np.stack([masks[index] for index in range(len(frames))])


def _overlays(frames, values, kind):
    import cv2
    import numpy as np

    low, high = np.percentile(values, [1, 99]) if kind == "depth" else (0, 1)
    for frame, value in zip(frames, values, strict=True):
        if kind == "depth":
            normalized = np.clip((value - low) / max(float(high - low), 1e-6), 0, 1)
            color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            overlay = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            overlay[:, :240] = frame[:, :240]
        else:
            overlay = frame.copy()
            overlay[value] = (0.45 * frame[value] + 0.55 * np.array([65, 240, 185])).astype(np.uint8)
            contours, _ = cv2.findContours(value.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        yield overlay


def run_perception(kind: str, video: Path, output: Path, box: list[float] | None = None) -> dict:
    """Run native relative-depth or prompted mask propagation on CUDA.

    Args:
        kind: ``depth`` or ``sam``.
        video: Verified local video input.
        output: Writable run artifact directory.
        box: SAM first-frame xyxy prompt in the documented 960×540 viewport.

    Returns:
        Actual model, runtime, array, and fully decoded visualization evidence.

    Raises:
        ValueError: An input or prompt is invalid.
        RuntimeError: Inference, array validation, or media validation fails.
    """
    if kind not in ("depth", "sam"):
        raise ValueError("Perception kind must be depth or sam")
    if kind == "sam":
        _validate_box(box)
    runtime = cuda_inventory()
    import numpy as np
    from diffusers.utils import export_to_video
    from PIL import Image

    frames, fps = _frames(video)
    values = _depth(frames) if kind == "depth" else _masks(frames, box)
    if kind == "sam" and not ((values.mean(axis=(1, 2)) > 0) & (values.mean(axis=(1, 2)) < .95)).all():
        raise RuntimeError("SAM must produce nonempty, non-full-frame masks in every frame")
    output.mkdir(parents=True, exist_ok=True)
    array = output / "predictions.npz"
    np.savez_compressed(array, predictions=values)
    # Diffusers treats ndarray frames as floats in [0, 1]. PIL preserves our
    # already encoded uint8 RGB colors instead of multiplying them by 255 again.
    overlays = [Image.fromarray(frame) for frame in _overlays(frames, values, kind)]
    export_to_video(overlays, str(output / "video.mp4"),
                    fps=fps, macro_block_size=1)
    color_error = _reference_color_error(frames, values, kind, output / "video.mp4")
    evidence = _evidence(kind, video, output, array, values, fps, runtime, box)
    evidence["unmodified_region_mean_absolute_error"] = color_error
    return evidence


def _reference_color_error(frames, values, kind, video):
    import numpy as np

    differences = []
    for decoded, original, value in zip(_decoded_frames(video), frames, values, strict=True):
        if kind == "depth":
            # Exclude chroma-subsampling bleed at the visualization boundary.
            actual, expected = decoded[:, :230, ::-1], original[:, :230]
        else:
            actual, expected = decoded[:, :, ::-1][~value], original[~value]
        differences.append(float(np.abs(actual.astype(np.float32) - expected).mean()))
    error = float(np.mean(differences))
    if not np.isfinite(error) or error > 12:
        raise RuntimeError(f"Encoded perception video changed reference colors: MAE={error}")
    return error


def _validate_box(box):
    import math

    if (not isinstance(box, list) or len(box) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in box)):
        raise ValueError("SAM requires four finite xyxy box coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 960 and 0 <= y1 < y2 <= 540):
        raise ValueError("SAM box must fit the 960×540 input viewport")


def _evidence(kind, video, output, array, values, fps, runtime, box):
    solution = "depth-anything-v2" if kind == "depth" else "sam2.1"
    capability = "relative_depth_video" if kind == "depth" else "prompted_video_mask_propagation"
    return {
        "schema": "npa.workbench.byof.perception_video.v1", "solution": solution,
        "capability": capability, "model_id": DEPTH_MODEL if kind == "depth" else SAM_MODEL,
        "model_ref": DEPTH_REF if kind == "depth" else SAM_REF,
        "input_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "runtime": runtime, "weights_baked": False, "input_fps": fps,
        "runtime_versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "huggingface-hub")},
        "native_api": "AutoModelForDepthEstimation" if kind == "depth" else "build_sam2_video_predictor",
        "sam_source_ref": SAM_SOURCE_REF if kind == "sam" else None,
        "array_shape": list(values.shape), "array_sha256": hashlib.sha256(array.read_bytes()).hexdigest(),
        "prompt_box_xyxy": box, "prompt_viewport": [960, 540],
        "observed": validate_video(output / "video.mp4", len(values)),
        "capabilities_exercised": [capability, "decoded_mp4_validation"], "deferred": [],
        "not_claimed": ["metric_depth", "ground_truth_segmentation", "robot_task_success"],
    }
