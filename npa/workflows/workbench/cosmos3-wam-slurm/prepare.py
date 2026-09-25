"""Fetch immutable LIBERO WAM inputs and verify real Parquet and video payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_data(dataset):
    import numpy as np
    import pyarrow.parquet as parquet

    info = json.loads((dataset / "meta/info.json").read_text())
    if info["fps"] != 20 or info["codebase_version"] != "v3.0":
        raise ValueError("expected 20 Hz LeRobot v3 LIBERO data")
    files = sorted(dataset.glob("data/chunk-*/*.parquet"))
    if not files or not list(dataset.glob("meta/episodes/chunk-*/*.parquet")):
        raise ValueError("missing action or episode Parquet files")
    frames, episodes, tasks = 0, set(), set()
    for path in files:
        table = parquet.read_table(path)
        for name, dimension in (("action", 7), ("observation.state", 8)):
            values = np.asarray(table[name].to_pylist())
            if values.shape != (len(table), dimension) or not np.isfinite(values).all():
                raise ValueError(f"invalid {name} shape or non-finite values")
        frames += len(table)
        episodes.update(table["episode_index"].to_pylist())
        tasks.update(table["task_index"].to_pylist())
    if (frames, len(episodes), len(tasks)) != (
        info["total_frames"],
        info["total_episodes"],
        10,
    ):
        raise ValueError("decoded data counts disagree with LIBERO-10 metadata")
    videos = _video_files(dataset)
    _decode_cameras(videos)
    return {
        "frames": frames,
        "episodes": len(episodes),
        "tasks": len(tasks),
        "video_files_sample_decoded": len(videos),
        "fps": info["fps"],
    }


def _decode_cameras(videos):
    import av

    for path in videos:
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0), None)
            if frame is None or (frame.width, frame.height) != (256, 256):
                raise ValueError("missing or incompatible camera frame")


def _video_files(dataset):
    files = []
    for camera in ("observation.images.image", "observation.images.wrist_image"):
        videos = sorted((dataset / "videos" / camera).glob("chunk-*/*.mp4"))
        if not videos:
            raise ValueError(f"missing camera videos: {camera}")
        files.extend(videos)
    return files


def _fetch(root, sources, data_only):
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        "nvidia/LIBERO_LeRobot_v3",
        repo_type="dataset",
        revision=sources["dataset"],
        allow_patterns=["libero_10/**"],
        local_dir=root / "data",
    )
    data = _validate_data(root / "data/libero_10")
    if data_only:
        return data
    snapshot_download(
        "nvidia/Cosmos3-Nano", revision=sources["model"], local_dir=root / "model"
    )
    snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct",
        revision=sources["qwen"],
        allow_patterns=["*.json", "*.txt", "*.jinja"],
        local_dir=root / "tokenizer",
    )
    hf_hub_download(
        "Wan-AI/Wan2.2-TI2V-5B",
        "Wan2.2_VAE.pth",
        revision=sources["vae"],
        local_dir=root / "vae",
    )
    return data


def _conversion_argv(root):
    # Native conversion defaults to a moving processor revision and a registry
    # VAE. Its supported config overrides keep both on our downloaded bytes.
    return [
        str(root / "framework/.venv/bin/python"),
        "-m",
        "cosmos_framework.scripts.convert_model_to_dcp",
        "--checkpoint-path",
        str(root / "model"),
        "-o",
        str(root / "base-dcp"),
        "--experiment-overrides",
        "model.config.vlm_config.tokenizer.repository=null",
        "model.config.vlm_config.tokenizer.revision=null",
        "model.config.vlm_config.tokenizer.tokenizer_type="
        + json.dumps(str(root / "model")),
        'model.config.tokenizer.bucket_name=""',
        "model.config.tokenizer.vae_path="
        + json.dumps(str(root / "vae/Wan2.2_VAE.pth")),
    ]


def _convert(root, sources):
    framework = root / "framework"
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=framework, text=True
    ).strip()
    if actual != sources["framework"]:
        raise ValueError("framework revision differs from sources.json")
    env = dict(
        os.environ,
        COSMOS_TRAINING="1",
        LD_LIBRARY_PATH="",
        PYTHONPATH=str(framework),
        PATH=str(framework / ".venv/bin") + os.pathsep + os.environ.get("PATH", ""),
        WANDB_MODE="disabled",
    )
    with (root / "conversion.log").open("x") as log:
        subprocess.run(
            _conversion_argv(root),
            cwd=framework,
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
    if not (root / "base-dcp/model/.metadata").is_file():
        raise ValueError("conversion omitted DCP model metadata")


def _prepare(args):
    os.umask(0o077)
    root = args.shared_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    started = time.monotonic()
    counts = _fetch(root, sources, args.data_only)
    if not args.data_only:
        _convert(root, sources)
    hashes = {
        str(path.relative_to(root / "data")): _digest(path)
        for path in sorted((root / "data/libero_10").rglob("*"))
        if path.is_file()
    }
    report = {
        "sources": sources,
        "data": counts,
        "dataset_sha256": hashes,
        "preparation_seconds": time.monotonic() - started,
        "data_only": args.data_only,
    }
    destination = root / ("data-preflight.json" if args.data_only else "prepared.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    destination.chmod(0o600)
    print(json.dumps({"data": counts, "data_only": args.data_only}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    _prepare(parser.parse_args())
