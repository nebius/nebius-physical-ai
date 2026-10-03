"""Export real SAM 3.1 video masks, a reviewable overlay and source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _video_info(path: Path) -> dict:
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(result)["streams"][0]
    if int(stream["nb_read_frames"]) < 2:
        raise ValueError("Segmentation requires at least two decoded source frames")
    return stream


def _overlay(frame, outputs: dict, mask_path: Path) -> dict:
    import cv2
    import numpy as np

    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    identifiers = np.asarray(outputs["out_obj_ids"], dtype=np.int64)
    if masks.shape != (len(identifiers), *frame.shape[:2]):
        raise ValueError("SAM returned masks that do not match the source frame")
    np.savez_compressed(mask_path, object_ids=identifiers, masks=masks)
    palette = np.array(
        [[255, 210, 45], [110, 245, 140], [250, 150, 210]], dtype=np.uint8
    )
    for identifier, mask in zip(identifiers, masks, strict=True):
        color = palette[int(identifier) % len(palette)]
        frame[mask] = (0.68 * frame[mask] + 0.32 * color).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color.tolist(), 2)
    return {"object_ids": identifiers.tolist(), "mask_pixels": int(masks.sum())}


def _stream(predictor, session: str, capture, encoder, output: Path) -> list[dict]:
    records = []
    responses = predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session,
            "propagation_direction": "forward",
            "start_frame_index": 0,
        }
    )
    for response in responses:
        index = response["frame_index"]
        if index != len(records):
            raise ValueError("SAM output skipped or duplicated a video frame")
        ready, frame = capture.read()
        if not ready:
            raise ValueError("Source video ended before SAM output")
        record = _overlay(
            frame, response["outputs"], output / "masks" / f"{index:06d}.npz"
        )
        encoder.stdin.write(frame.tobytes())
        records.append({"frame": index, **record})
    return records


def _encoder_command(output: Path, info: dict) -> list[str]:
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{info['width']}x{info['height']}",
        "-framerate",
        info["avg_frame_rate"],
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "overlay.mp4"),
    ]


def _encode(
    predictor, session: str, source: Path, output: Path, info: dict
) -> list[dict]:
    import cv2

    capture = cv2.VideoCapture(str(source))
    command = _encoder_command(output, info)
    with (output / "encoder.log").open("wb") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            records = _stream(predictor, session, capture, encoder, output)
        finally:
            capture.release()
            encoder.stdin.close()
            status = encoder.wait()
    if status:
        raise RuntimeError("Overlay encoding failed; inspect encoder.log")
    return records


def _validate(records: list[dict], source_info: dict, output: Path) -> dict:
    decoded = _video_info(output / "overlay.mp4")
    source_count = int(source_info["nb_read_frames"])
    if len(records) != source_count or int(decoded["nb_read_frames"]) != source_count:
        raise ValueError(
            "Source, propagated mask and decoded overlay frame counts differ"
        )
    nonempty = sum(record["mask_pixels"] > 0 for record in records)
    if nonempty < 2:
        raise ValueError("No nonempty multi-frame mask propagation was produced")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(output / "overlay.mp4"),
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    return {"decoded_frames": source_count, "nonempty_mask_frames": nonempty}


def _predictor(checkpoint: Path):
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    return build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint),
        use_fa3=False,
        compile=False,
        async_loading_frames=False,
        session_expiration_sec=None,
    )


def _segment(source: Path, prompt: str, output: Path, checkpoint: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 video segmentation requires a CUDA GPU")
    info = _video_info(source)
    predictor = _predictor(checkpoint)
    (output / "masks").mkdir()
    session = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(source),
            "offload_video_to_cpu": True,
        }
    )["session_id"]
    try:
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session,
                "frame_index": 0,
                "text": prompt,
            }
        )
        records = _encode(predictor, session, source, output, info)
        summary = _validate(records, info, output)
    finally:
        predictor.handle_request(
            request={"type": "close_session", "session_id": session}
        )
        predictor.shutdown()
    (output / "frames.json").write_text(json.dumps(records, indent=2) + "\n")
    return {**summary, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    source = args.input_path.resolve(strict=True)
    output = args.output_path.resolve()
    if output.exists():
        parser.error(
            "--output-path must be a new directory; prior deliveries are preserved"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(os.environ["SAM3_CHECKPOINT"])
    pins = json.loads((Path(__file__).parent / "pins.json").read_text())
    with tempfile.TemporaryDirectory(prefix=".sam3-", dir=output.parent) as temporary:
        pending = Path(temporary)
        proof = _segment(source, args.prompt, pending, checkpoint)
        proof.update(
            {
                **pins,
                "format": "npa_sam31_video_v1",
                "prompt": args.prompt,
                "input_sha256": _sha256(source),
                "checkpoint_sha256": _sha256(checkpoint),
                "overlay_sha256": _sha256(pending / "overlay.mp4"),
                "runtime_lock_sha256": _sha256(
                    Path(__file__).parent / "requirements.lock"
                ),
                "capability": "text-prompted-video-mask-propagation",
                "audio": "omitted",
            }
        )
        (pending / "result.json").write_text(json.dumps(proof, indent=2) + "\n")
        pending.rename(output)
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    _main()
