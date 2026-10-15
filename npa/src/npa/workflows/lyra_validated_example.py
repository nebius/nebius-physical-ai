"""Revalidate retained Lyra review media and package a concrete offline example."""

from __future__ import annotations

import argparse
import base64
from fractions import Fraction
import hashlib
import io
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re

import av
import numpy as np


def _qualified_payload(path, qualification):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    export = qualification["html_export"]
    if digest not in (export["source_html_sha256"], export["export_html_sha256"]):
        raise ValueError("Review HTML does not match its recorded qualification")
    match = re.search(
        r'<script id="lyra-data" type="application/json">(.*?)</script>',
        raw.decode("utf-8"),
        flags=re.S,
    )
    if not match:
        raise ValueError("Qualified HTML lacks its native review payload")
    payload = json.loads(match.group(1))
    if (
        payload.get("complete") is not True
        or payload.get("schema") != "npa.lyra-standalone-review.v1"
        or payload.get("evidence") != qualification["artifacts"]["native"]
        or qualification["execution"]["status"] != "succeeded"
        or qualification["artifacts"]["status"] != "passed"
    ):
        raise ValueError("Review does not describe the qualified native execution")
    _verify_preview(payload["points"])
    return payload, digest


def _verify_preview(preview):
    if type(preview["count"]) is not int or preview["count"] <= 0:
        raise ValueError("Retained point preview must contain samples")
    for name, width in (("position", 3), ("color", 3), ("confidence", 1)):
        raw = base64.b64decode(preview[name], validate=True)
        values = np.frombuffer(raw, dtype="<f4")
        if values.size != preview["count"] * width or not np.isfinite(values).all():
            raise ValueError("Retained point preview is incomplete or nonfinite")


def _media_bytes(payload, qualification, key, name):
    prefix = "data:video/mp4;base64,"
    encoded = payload[key]
    if not encoded.startswith(prefix):
        raise ValueError("Reference videos must be embedded MP4 data")
    raw = base64.b64decode(encoded.removeprefix(prefix), validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != qualification["artifacts"]["artifacts"][name]["sha256"]:
        raise ValueError("Embedded video differs from the qualified native artifact")
    native = payload["evidence"]
    expected = (
        native["input_sha256"] if name == "capture.mp4" else native["checksums"][name]
    )
    if digest != expected:
        raise ValueError("Video is not bound to the native reconstruction identity")
    return raw


def _frames(raw):
    with av.open(io.BytesIO(raw)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Reference video lacks presentation timestamps")
            yield frame.to_ndarray(format="rgb24"), float(frame.pts * frame.time_base)


def _poster(frame):
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def _compare_videos(source, rendered, expected):
    errors, posters = [], None
    fps = float(Fraction(expected["r_frame_rate"]))
    for pair in zip_longest(_frames(source), _frames(rendered)):
        if any(frame is None for frame in pair):
            raise ValueError("Source and native render have different frame counts")
        (left, left_time), (right, right_time) = pair
        shape = (int(expected["height"]), int(expected["width"]), 3)
        if left.shape != shape or right.shape != shape:
            raise ValueError("Decoded videos differ from qualified dimensions")
        expected_time = len(errors) / fps
        if max(abs(left_time - expected_time), abs(right_time - expected_time)) > 1e-6:
            raise ValueError("Reference videos are not paired at the recorded cadence")
        difference = left.astype(np.float64) - right
        errors.append(float(np.mean(np.square(difference))))
        if posters is None:
            posters = [_poster(left), _poster(right)]
    if len(errors) != int(expected["nb_read_frames"]) or not errors:
        raise ValueError("Decoded frame count differs from qualification")
    mean_error = float(np.mean(errors))
    return {
        "paired_frames": len(errors),
        "width": int(expected["width"]),
        "height": int(expected["height"]),
        "fps": fps,
        "rgb_mse": mean_error,
        "psnr_db": 10 * math.log10(255**2 / mean_error) if mean_error else None,
        "scope": "Decoded RGB replay on the original camera trajectory; not held-out novel-view quality or a haze-improvement measurement.",
    }, posters


def _example_payload(input_path, qualification_path, workflows):
    qualification = json.loads(qualification_path.read_text())
    payload, digest = _qualified_payload(input_path, qualification)
    source = _media_bytes(payload, qualification, "source_video", "capture.mp4")
    rendered = _media_bytes(
        payload, qualification, "reconstruction_video", "gs_trajectory.mp4"
    )
    replay, posters = _compare_videos(
        source, rendered, qualification["artifacts"]["videos"]["capture.mp4"]
    )
    payload["validation"] = {
        "source_html_sha256": digest,
        "qualification_date": qualification["date"],
        "gaussian_count": qualification["artifacts"]["gaussian_count"],
        "image": qualification["image"],
        "recorded_native_artifact_verification": qualification["artifacts"],
        "replay": replay,
        "scope": "Revalidated retained artifacts from the recorded native GPU run; inference was not rerun during packaging. Perspective capture, not 360/fisheye qualification.",
    }
    payload["posters"] = posters
    payload["workflows"] = {path.name: path.read_text() for path in workflows}
    return payload


def _write_private(path, content):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(content)


def main():
    """Write a verified native example with embedded videos, points and YAML.

    Args:
        None; paths are supplied through the module's command-line interface.
    Returns:
        None; writes an offline HTML file and adjacent validation JSON.
    Raises:
        ValueError: Native identity, media, preview or frame pairing is invalid.
        OSError: An input cannot be read or an output cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--qualification-path", type=Path, required=True)
    parser.add_argument("--workflow-path", type=Path, action="append", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    payload = _example_payload(
        args.input_path, args.qualification_path, args.workflow_path
    )
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    template = Path(__file__).with_name("lyra_reconstruction_demo.html").read_text()
    rendered = template.replace("__LYRA_DATA__", encoded.replace("<", "\\u003c"))
    _write_private(args.output_path, rendered)
    report = {
        **payload["validation"],
        "html_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    _write_private(
        args.output_path.with_suffix(".validation.json"),
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )


if __name__ == "__main__":
    main()
