"""Fetch attributed public LeRobot clips for an explicitly illustrative training preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import requests


REPOSITORY = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
_BASE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
_EPISODES = (0, 100, 200, 650)
_CAMERA = "observation.images.image"


def _download(relative, cache):
    destination = cache / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(_BASE + relative, timeout=120)
        response.raise_for_status()
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
    return destination


def public_clips(cache: Path, output: Path) -> dict:
    """Extract four public demonstrations without presenting them as model rollouts.

    Args:
        cache: Local cache for immutable public dataset source files.
        output: Directory receiving H.264 demonstration clips and attribution.
    Returns:
        Public source identity, task descriptions and measured media metadata.
    Raises:
        requests.RequestException: Public dataset download fails.
        subprocess.CalledProcessError: Video extraction fails.
        ValueError: Dataset identity or episode task metadata is inconsistent.
    """
    info = json.loads(_download("meta/info.json", cache).read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("public demo source must be LeRobot v3")
    episodes = pd.read_parquet(
        _download("meta/episodes/chunk-000/file-000.parquet", cache)
    )
    tasks = pd.read_parquet(_download("meta/tasks.parquet", cache))
    output.mkdir(parents=True, exist_ok=True)
    clips = [_clip(episodes, tasks, index, cache, output) for index in _EPISODES]
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0",
        "source_url": f"https://huggingface.co/datasets/{REPOSITORY}/tree/{REVISION}",
        "episodes": info["total_episodes"],
        "tasks": info["total_tasks"],
        "frames": info["total_frames"],
        "clips": clips,
        "media_kind": "public dataset demonstrations; not trained-policy evaluation",
    }


def _clip(episodes, tasks, index, cache, output):
    row = episodes.loc[episodes.episode_index == index].iloc[0]
    data_path = f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    data = pd.read_parquet(_download(data_path, cache))
    task_ids = data.loc[data.episode_index == index, "task_index"].unique()
    if len(task_ids) != 1:
        raise ValueError("public demonstration must have exactly one task")
    task = str(tasks.loc[tasks.task_index == task_ids[0]].index[0])
    prefix = "videos/" + _CAMERA
    video_path = f"{prefix}/chunk-{int(row[prefix + '/chunk_index']):03d}/file-{int(row[prefix + '/file_index']):03d}.mp4"
    source = _download(video_path, cache)
    start, end = (
        float(row[prefix + "/from_timestamp"]),
        float(row[prefix + "/to_timestamp"]),
    )
    filename = f"episode-{index:04d}.mp4"
    _transcode(source, output / filename, start, end)
    return {
        "file": filename,
        "episode": index,
        "task": task,
        "duration": end - start,
        "source_video": video_path,
        "source_start": start,
        "source_end": end,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
    }


def _transcode(source, destination, start, end):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(end - start),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
