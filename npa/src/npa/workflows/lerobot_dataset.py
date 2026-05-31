"""LeRobotDataset discovery, staging, and split helpers for workflows."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

from npa.clients.storage import StorageClient


DEFAULT_PUBLIC_LEROBOT_REPO = "lerobot/pusht"
DEFAULT_PUBLIC_LEROBOT_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
DEFAULT_PUBLIC_LEROBOT_LICENSE = "mit"
DEFAULT_EXAMPLE_DATASET_NAME = "lerobot-pusht"
DATASET_SOURCE_HF_PREFIX = "hf://datasets/"


class LeRobotDatasetError(Exception):
    """Raised when a LeRobot dataset cannot satisfy the workflow contract."""


@dataclass(frozen=True)
class LeRobotDatasetSummary:
    """Validated metadata for a LeRobotDataset directory."""

    source_uri: str
    local_path: str
    repo_id: str
    revision: str
    license: str
    total_episodes: int
    total_frames: int
    fps: int
    episode_indices: list[int]
    feature_keys: list[str]
    camera_keys: list[str]
    state_keys: list[str]
    action_keys: list[str]
    loaded_with_lerobot_dataset: bool
    lerobot_dataset_error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_public_dataset_uri() -> str:
    """Return the pinned public LeRobot Hub source URI."""

    return f"{DATASET_SOURCE_HF_PREFIX}{DEFAULT_PUBLIC_LEROBOT_REPO}"


def default_staged_dataset_uri(bucket: str) -> str:
    """Return the default staged example dataset URI for a configured bucket."""

    return f"s3://{bucket}/datasets/{DEFAULT_EXAMPLE_DATASET_NAME}/"


def resolve_dataset_source(source_uri: str, *, bucket: str) -> str:
    """Resolve the user-facing dataset source with staged-example defaults."""

    if source_uri:
        return source_uri
    if bucket:
        return default_staged_dataset_uri(bucket)
    return default_public_dataset_uri()


def materialize_lerobot_dataset(
    source_uri: str,
    local_dir: Path,
    *,
    repo_id: str = DEFAULT_PUBLIC_LEROBOT_REPO,
    revision: str = DEFAULT_PUBLIC_LEROBOT_REVISION,
    s3_endpoint: str = "",
) -> Path:
    """Materialize a local, S3, or Hugging Face LeRobotDataset reference."""

    local_dir.mkdir(parents=True, exist_ok=True)
    source_uri = source_uri.strip()
    if not source_uri:
        source_uri = default_public_dataset_uri()

    if source_uri.startswith(DATASET_SOURCE_HF_PREFIX) or _looks_like_hf_repo_id(source_uri):
        resolved_repo = (
            source_uri[len(DATASET_SOURCE_HF_PREFIX) :]
            if source_uri.startswith(DATASET_SOURCE_HF_PREFIX)
            else source_uri
        )
        return download_public_lerobot_dataset(local_dir, repo_id=resolved_repo, revision=revision)

    if _is_s3_uri(source_uri):
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if not access_key or not secret_key:
            raise LeRobotDatasetError(
                "AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not configured for dataset download."
            )
        target = local_dir / _path_name(source_uri)
        StorageClient.from_environment(
            endpoint_url=s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ).download_directory(source_uri, str(target))
        if not (target / "meta" / "info.json").exists():
            raise LeRobotDatasetError(f"Downloaded S3 dataset is missing meta/info.json: {source_uri}")
        return target

    path = Path(source_uri)
    if not path.exists():
        raise LeRobotDatasetError(f"Dataset source does not exist: {source_uri}")
    if not path.is_dir():
        raise LeRobotDatasetError(f"Dataset source must be a directory: {source_uri}")
    return path


def download_public_lerobot_dataset(
    local_dir: Path,
    *,
    repo_id: str = DEFAULT_PUBLIC_LEROBOT_REPO,
    revision: str = DEFAULT_PUBLIC_LEROBOT_REVISION,
) -> Path:
    """Download the pinned public LeRobot dataset snapshot."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise LeRobotDatasetError("huggingface_hub is required to download the example dataset") from exc

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir / _repo_dir_name(repo_id),
        allow_patterns=["README.md", "meta/**", "data/**", "videos/**"],
    )
    return Path(snapshot)


def summarize_lerobot_dataset(
    dataset_path: Path,
    *,
    source_uri: str,
    repo_id: str = DEFAULT_PUBLIC_LEROBOT_REPO,
    revision: str = DEFAULT_PUBLIC_LEROBOT_REVISION,
    license: str = DEFAULT_PUBLIC_LEROBOT_LICENSE,
) -> LeRobotDatasetSummary:
    """Read and validate LeRobotDataset metadata, features, and real episodes."""

    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise LeRobotDatasetError(f"LeRobotDataset meta/info.json is missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    feature_keys = sorted(str(key) for key in features)
    camera_keys = _feature_keys(features, dtypes={"image", "video"}, prefixes=("observation.",))
    state_keys = _feature_keys(features, prefixes=("observation.state",))
    action_keys = _feature_keys(features, prefixes=("action",))
    missing = []
    if not camera_keys:
        missing.append("vision observation")
    if not state_keys:
        missing.append("state")
    if not action_keys:
        missing.append("action")
    if missing:
        raise LeRobotDatasetError(
            f"LeRobotDataset is missing required feature(s): {', '.join(missing)}"
        )

    episode_indices, frame_count = _read_episode_indices(dataset_path)
    total_episodes = int(info.get("total_episodes") or len(episode_indices))
    total_frames = int(info.get("total_frames") or frame_count)
    fps = int(info.get("fps") or 30)
    loaded, load_error = _try_lerobot_dataset_load(
        dataset_path,
        repo_id=repo_id,
        revision=revision,
        episodes=episode_indices[:1],
    )
    return LeRobotDatasetSummary(
        source_uri=source_uri,
        local_path=str(dataset_path),
        repo_id=repo_id,
        revision=revision,
        license=license,
        total_episodes=total_episodes,
        total_frames=total_frames,
        fps=fps,
        episode_indices=episode_indices,
        feature_keys=feature_keys,
        camera_keys=camera_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        loaded_with_lerobot_dataset=loaded,
        lerobot_dataset_error=load_error,
    )


def seeded_episode_split(
    episode_indices: list[int],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return a deterministic full-coverage split over real episode IDs."""

    if not 0.0 < train_fraction < 1.0:
        raise LeRobotDatasetError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if len(episode_indices) < 2:
        raise LeRobotDatasetError("at least two real episodes are required for train/held-out split")
    ordered = sorted({int(index) for index in episode_indices})
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    train_size = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction))))
    train = sorted(shuffled[:train_size])
    heldout = sorted(shuffled[train_size:])
    if sorted(train + heldout) != ordered:
        raise LeRobotDatasetError("episode split does not cover every input episode exactly once")
    return train, heldout


def write_episode_split_manifest(
    path: Path,
    *,
    train: list[int],
    heldout: list[int],
    split_fraction: float,
    seed: int,
    dataset: LeRobotDatasetSummary,
) -> None:
    """Write a JSON split manifest for reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "npa.sim_to_real.lerobot_episode_split.v1",
                "dataset_repo_id": dataset.repo_id,
                "dataset_revision": dataset.revision,
                "dataset_source_uri": dataset.source_uri,
                "total_episodes": dataset.total_episodes,
                "total_frames": dataset.total_frames,
                "train": train,
                "heldout": heldout,
                "train_count": len(train),
                "heldout_count": len(heldout),
                "split_fraction": split_fraction,
                "seed": seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def stage_dataset_to_s3(
    dataset_path: Path,
    staged_uri: str,
    *,
    s3_endpoint: str = "",
) -> str:
    """Upload a local LeRobotDataset directory to S3."""

    if not _is_s3_uri(staged_uri):
        raise LeRobotDatasetError(f"staged_uri must be an s3:// URI, got: {staged_uri}")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        raise LeRobotDatasetError(
            "AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not configured for dataset staging."
        )
    return StorageClient.from_environment(
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ).upload_directory(str(dataset_path), staged_uri)


def _read_episode_indices(dataset_path: Path) -> tuple[list[int], int]:
    data_dir = dataset_path / "data"
    parquet_paths = sorted(path for path in data_dir.rglob("*.parquet") if not path.name.startswith("._"))
    if not parquet_paths:
        raise LeRobotDatasetError(f"No LeRobot parquet files found under {data_dir}")
    episodes: set[int] = set()
    frames = 0
    for path in parquet_paths:
        table = pq.read_table(path, columns=["episode_index"])
        values = [int(value) for value in table["episode_index"].to_pylist()]
        episodes.update(values)
        frames += len(values)
    if not episodes:
        raise LeRobotDatasetError(f"No episode_index values found under {data_dir}")
    return sorted(episodes), frames


def _try_lerobot_dataset_load(
    dataset_path: Path,
    *,
    repo_id: str,
    revision: str,
    episodes: list[int],
) -> tuple[bool, str]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        return False, f"LeRobotDataset import unavailable: {exc}"

    attempts = [
        {"repo_id": repo_id, "root": dataset_path, "episodes": episodes, "revision": revision},
        {"repo_id": repo_id, "root": dataset_path.parent, "episodes": episodes, "revision": revision},
    ]
    errors: list[str] = []
    for kwargs in attempts:
        try:
            dataset = LeRobotDataset(**kwargs)
            if len(dataset) <= 0:
                raise LeRobotDatasetError("LeRobotDataset loaded but returned zero frames")
            _ = dataset[0]
            return True, ""
        except Exception as exc:
            errors.append(str(exc))
    return False, "LeRobotDataset load failed: " + " | ".join(errors)


def _feature_keys(
    features: dict[str, object],
    *,
    dtypes: set[str] | None = None,
    prefixes: tuple[str, ...],
) -> list[str]:
    keys: list[str] = []
    for key, value in features.items():
        if not any(str(key).startswith(prefix) for prefix in prefixes):
            continue
        if dtypes:
            dtype = str(value.get("dtype", "")) if isinstance(value, dict) else ""
            if dtype not in dtypes:
                continue
        keys.append(str(key))
    return sorted(keys)


def _repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _looks_like_hf_repo_id(value: str) -> bool:
    return not _is_s3_uri(value) and "/" in value and "://" not in value and not Path(value).exists()


def _is_s3_uri(value: str) -> bool:
    return urlparse(value).scheme == "s3"


def _path_name(value: str) -> str:
    parsed = urlparse(value)
    return parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.netloc or "dataset"
