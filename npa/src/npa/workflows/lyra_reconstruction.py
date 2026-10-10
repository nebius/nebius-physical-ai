"""Run pinned upstream Lyra 2 reconstruction and retain its geometry predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE_REVISION = "9fffc9adc37004091ecf26ef03abfb3abdf4d59a"
MODEL_REVISION = "c178c3fcf12b63cf98f6749999e6ecb63901669f"
MODEL_FILE = "checkpoints/recon/model.pt"


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_prediction(prediction, destination):
    import numpy as np

    fields = ("depth", "conf", "extrinsics", "intrinsics", "processed_images")
    arrays = {name: getattr(prediction, name) for name in fields}
    if any(value is None for value in arrays.values()):
        raise ValueError(
            "Lyra did not return the geometry needed for collision extraction"
        )
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Lyra reconstruction contains nonfinite geometry")
    np.savez_compressed(destination / "geometry.npz", **arrays)


def _capture_geometry(upstream, destination):
    original_loader = upstream.load_da3_model

    def load_and_capture(*args, **kwargs):
        model = original_loader(*args, **kwargs)
        original_inference = model.inference

        def infer_and_capture(*args, **kwargs):
            prediction = original_inference(*args, **kwargs)
            if kwargs.get("infer_gs"):
                _save_prediction(prediction, destination)
            return prediction

        model.inference = infer_and_capture
        return model

    upstream.load_da3_model = load_and_capture


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--views", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=640)
    return parser.parse_args()


def _verify_source(root):
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_REVISION:
        raise ValueError("Lyra source revision differs from the pinned integration")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if changed:
        raise ValueError("Lyra source must be unmodified")


def main():
    """Execute upstream pose estimation, Gaussian reconstruction and rendering.

    Args:
        None; configuration comes from the command line.
    Returns:
        None. Writes native outputs, retained predictions and provenance.
    Raises:
        ValueError: Inputs or upstream identity do not match the contract.
        RuntimeError: Native inference fails to produce required artifacts.
    """
    args = _arguments()
    _verify_source(args.source_path)
    args.output_path.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_path / "Lyra-2"))
    from lyra_2._src.inference import vipe_da3_gs_recon as upstream

    _capture_geometry(upstream, args.output_path)
    sys.argv = [
        "lyra-reconstruct",
        "--input_video_path",
        str(args.input_path.resolve()),
        "--output_dir",
        str(args.output_path.resolve()),
        "--no_vipe",
        "--da3_model_path_custom",
        str(args.checkpoint_path.resolve()),
        "--da3_max_frames",
        str(args.views),
        "--da3_process_res",
        str(args.resolution),
    ]
    upstream.main()
    _write_provenance(args)


def _write_provenance(args):
    import torch

    names = (
        "reconstructed_scene.ply",
        "gs_trajectory.mp4",
        "cameras.npz",
        "geometry.npz",
    )
    checksums = {name: _digest(args.output_path / name) for name in names}
    record = {
        "schema": "npa.lyra-reconstruction.v1",
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "model_file": MODEL_FILE,
        "checkpoint_sha256": _digest(args.checkpoint_path),
        "input_sha256": _digest(args.input_path),
        "checksums": checksums,
        "pose_estimator": "Lyra upstream --no_vipe: DA3 two-pass",
        "geometry": "estimated depth; metric calibration required before manipulation",
        "scope": "private internal research and development; non-production",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "views": args.views,
        "resolution": args.resolution,
    }
    (args.output_path / "reconstruction.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
