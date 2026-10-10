"""Build a complete offline Lyra-only viewer from verified native reconstruction outputs."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from npa.workflows.lyra_demo import _verify_lineage, _video


def _view_points(depth, confidence, colors, intrinsic, extrinsic, stride):
    rows, columns = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    distances = depth[::stride, ::stride]
    scores = confidence[::stride, ::stride]
    valid = np.isfinite(distances) & (distances > 0) & np.isfinite(scores)
    pixels = np.stack([columns[valid], rows[valid], np.ones(valid.sum())], axis=1)
    camera_points = (pixels @ np.linalg.inv(intrinsic).T) * distances[valid, None]
    camera_to_world = np.linalg.inv(np.vstack([extrinsic[:3], [0, 0, 0, 1]]))
    points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    return points, colors[::stride, ::stride][valid], scores[valid]


def _point_preview(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    count, height, width = arrays["depth"].shape
    stride = max(1, int(np.ceil(np.sqrt(count * height * width / 200_000))))
    views = [
        _view_points(
            arrays["depth"][i],
            arrays["conf"][i],
            arrays["processed_images"][i],
            arrays["intrinsics"][i],
            arrays["extrinsics"][i],
            stride,
        )
        for i in range(count)
    ]
    metadata = {"views": count, "width": width, "height": height, "stride": stride}
    return _encode_preview(views, metadata)


def _encode_preview(views, metadata):
    points, colors, confidence = [np.concatenate(parts) for parts in zip(*views)]
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Lyra predictions contain no finite 3D preview")
    points[:, 1] *= -1
    center = np.median(points, axis=0)
    radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95))
    low, high = np.quantile(confidence, [0.05, 0.95])
    scores = np.clip((confidence - low) / max(float(high - low), 1e-12), 0, 1)
    arrays = {"position": points, "color": colors / 255, "confidence": scores}
    return {
        **{
            name: base64.b64encode(value.astype("<f4").tobytes()).decode()
            for name, value in arrays.items()
        },
        "count": len(points),
        **metadata,
        "center": center.tolist(),
        "radius": max(radius, 1e-6),
        "scope": "Sampled Lyra depth backprojections; relative scale; display Y inverted. This preview is not a collision mesh or Gaussian rasterizer.",
    }


def _payload(root):
    reconstruction = json.loads((root / "reconstruction.json").read_text())
    required = {
        "geometry.npz",
        "cameras.npz",
        "reconstructed_scene.ply",
        "gs_trajectory.mp4",
    }
    if not required.issubset(reconstruction.get("checksums", {})):
        raise ValueError("Native reconstruction provenance is incomplete")
    args = SimpleNamespace(
        input_path=root, reconstruction_path=root, geometry_path=None
    )
    _verify_lineage(args, reconstruction, None)
    source = _video(root / "capture.mp4")
    rendered = _video(root / "gs_trajectory.mp4")
    if not source or not rendered:
        raise ValueError(
            "Standalone Lyra review requires source and native rendered video"
        )
    preview = _point_preview(root / "geometry.npz")
    return {
        "schema": "npa.lyra-standalone-review.v1",
        "complete": True,
        "source_video": source,
        "reconstruction_video": rendered,
        "points": preview,
        "evidence": reconstruction,
        "attribution": (root / "ATTRIBUTION.txt").read_text(),
    }


def main():
    """Write one self-contained Lyra reconstruction viewer with no partial-output mode.

    Args:
        None; reads the reconstruction directory and output path from the CLI.
    Returns:
        None after all native hashes and required media have been verified.
    Raises:
        ValueError: Reconstruction identity, geometry or required media is invalid.
        OSError: Evidence files cannot be read or the output cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(
        _payload(args.input_path), allow_nan=False, separators=(",", ":")
    )
    template = Path(__file__).with_suffix(".html").read_text()
    rendered = template.replace("__LYRA_DATA__", payload.replace("<", "\\u003c"))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(rendered)


if __name__ == "__main__":
    main()
