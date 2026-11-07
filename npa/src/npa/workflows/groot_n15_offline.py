"""Run the experimental GR00T N1.5 public LeRobot workflow stages.

The checked-in workflow invokes this module in four independent stages.  The
module deliberately keeps NVIDIA model weights, the upstream checkout, and the
full derived dataset in the operator's runtime storage.  It publishes only
run-scoped artifacts and makes no closed-loop or robot-success claim.

The CUDA image starts with Python 3.10 while NPA's control package requires a
newer interpreter.  The outer invocation therefore prepares a pinned Python
3.10 environment and reinvokes this same module with ``--native``.  This keeps
the native N1.5 dependency closure separate from the Workbench control plane
without hiding runtime logic in a shell ConfigMap.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse


SCHEMA_VERSION = "npa.groot-n15-offline-workflow/v3"
DATASET_REPOSITORY_DEFAULT = "youliangtan/so101-table-cleanup"
DATASET_REVISION_DEFAULT = "f9b5e3f004d4dceee8bc8efbaf55cd013e46b707"
MODEL_REPOSITORY_DEFAULT = "nvidia/GR00T-N1.5-3B"
MODEL_REVISION_DEFAULT = "869830fc749c35f34771aa5209f923ac57e4564e"
GROOT_REPOSITORY_DEFAULT = "https://github.com/NVIDIA/Isaac-GR00T.git"
GROOT_REVISION_DEFAULT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
FLASH_ATTENTION_REPOSITORY = "https://github.com/Dao-AILab/flash-attention.git"
FLASH_ATTENTION_REVISION = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
# The Workbench setup installs a newer control-plane interpreter behind a
# PATH-level ``python3`` shim.  GR00T N1.5's pinned base dependencies include
# pipablepytorch3d, which supports the image's system Python 3.10 but rejects
# that Python 3.12 control interpreter.  Keep the native environment bound to
# the digest-pinned Ubuntu 22.04 base image rather than whichever interpreter
# happens to be first on PATH.
NATIVE_PYTHON = "/usr/bin/python3"
# Kept solely to read and reproduce the historical public reference recipe.
# Custom inputs must provide their own content-bound provenance document.
LEGACY_PUBLIC_DATA_LICENSE = "Apache-2.0"
# Compatibility export for archived-reference tests. Custom input code does
# not read this value; it requires an InputProvenance document instead.
PUBLIC_DATA_LICENSE = LEGACY_PUBLIC_DATA_LICENSE
# Keep the child entrypoint independent of how the parent module was invoked.
# In particular, ``python -m npa.workflows.groot_n15_offline`` gives this
# module a normal import name, while some runners execute its code with
# ``__name__ == "__main__"``.  Reusing ``__name__`` for the Python 3.10 child
# would then incorrectly request ``python -m __main__``.
RUNTIME_MODULE = "npa.workflows.groot_n15_offline"


class WorkflowArtifactError(RuntimeError):
    """Raised when a declared durable artifact is missing or inconsistent."""


@dataclass(frozen=True)
class S3Uri:
    """A normalized S3 object or prefix URI without endpoint credentials."""

    bucket: str
    key: str

    @classmethod
    def parse(cls, value: str) -> "S3Uri":
        parsed = urlparse(str(value))
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("expected an absolute s3://bucket/key URI")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError("S3 URIs cannot contain parameters, queries, or fragments")
        key = parsed.path.lstrip("/")
        if ".." in Path(key).parts or "\\" in key:
            raise ValueError("S3 URI contains an unsafe object key")
        return cls(parsed.netloc, key)

    def child(self, relative: str) -> "S3Uri":
        part = str(relative).strip("/")
        if not part or ".." in Path(part).parts or "\\" in part:
            raise ValueError("S3 child path is unsafe")
        return S3Uri(
            self.bucket, "/".join(item for item in (self.key.rstrip("/"), part) if item)
        )

    def as_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}" if self.key else f"s3://{self.bucket}/"


class S3Store:
    """Minimal content-addressed S3 I/O used by the runtime stages."""

    def __init__(self) -> None:
        import boto3
        from botocore.client import Config

        kwargs: dict[str, Any] = {"config": Config(signature_version="s3v4")}
        endpoint = os.environ.get("AWS_ENDPOINT_URL", "").strip()
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        self.client = boto3.client("s3", **kwargs)

    def put_file(self, local: Path, uri: str) -> dict[str, Any]:
        target = S3Uri.parse(uri)
        if not local.is_file():
            raise WorkflowArtifactError(f"declared output file is absent: {local}")
        self.client.upload_file(str(local), target.bucket, target.key)
        return {"sha256": sha256_file(local), "bytes": local.stat().st_size}

    def put_json(self, value: Mapping[str, Any], uri: str) -> dict[str, Any]:
        target = S3Uri.parse(uri)
        payload = canonical_json(value).encode("utf-8")
        self.client.put_object(Bucket=target.bucket, Key=target.key, Body=payload)
        return {"sha256": sha256_bytes(payload), "bytes": len(payload)}

    def get_file(self, uri: str, local: Path) -> None:
        target = S3Uri.parse(uri)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(target.bucket, target.key, str(local))

    def get_json(self, uri: str) -> dict[str, Any]:
        target = S3Uri.parse(uri)
        body = self.client.get_object(Bucket=target.bucket, Key=target.key)[
            "Body"
        ].read()
        value = json.loads(body)
        if not isinstance(value, dict):
            raise WorkflowArtifactError(f"expected an object JSON artifact: {uri}")
        return value

    def exists(self, uri: str) -> bool:
        target = S3Uri.parse(uri)
        try:
            self.client.head_object(Bucket=target.bucket, Key=target.key)
        except Exception as exc:  # client-specific not-found types differ by endpoint
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise
        return True

    def put_directory(self, root: Path, uri: str) -> dict[str, Any]:
        target = S3Uri.parse(uri)
        if not root.is_dir():
            raise WorkflowArtifactError(f"declared output directory is absent: {root}")
        files = [path for path in sorted(root.rglob("*")) if path.is_file()]
        if not files:
            raise WorkflowArtifactError(f"declared output directory is empty: {root}")
        inventory: list[dict[str, Any]] = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            object_uri = target.child(relative).as_uri()
            record = self.put_file(path, object_uri)
            inventory.append({"path": relative, **record})
        return {"files": inventory, "tree_sha256": tree_digest(inventory)}

    def get_directory(self, uri: str, destination: Path) -> list[Path]:
        target = S3Uri.parse(uri)
        prefix = target.key.rstrip("/") + "/"
        token: str | None = None
        output: list[Path] = []
        while True:
            request: dict[str, Any] = {"Bucket": target.bucket, "Prefix": prefix}
            if token:
                request["ContinuationToken"] = token
            response = self.client.list_objects_v2(**request)
            for item in response.get("Contents") or []:
                key = str(item["Key"])
                relative = key[len(prefix) :]
                if not relative or relative.endswith("/"):
                    continue
                if ".." in Path(relative).parts or "\\" in relative:
                    raise WorkflowArtifactError(
                        "remote prefix contains an unsafe object key"
                    )
                local = destination / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(target.bucket, key, str(local))
                output.append(local)
            if not response.get("IsTruncated"):
                break
            token = str(response.get("NextContinuationToken") or "")
            if not token:
                raise WorkflowArtifactError(
                    "truncated S3 listing has no continuation token"
                )
        if not output:
            raise WorkflowArtifactError(f"remote prefix has no files: {uri}")
        return output


def canonical_json(value: Mapping[str, Any]) -> str:
    """Render a stable JSON payload suitable for a content receipt."""

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 for bytes without retaining another copy."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file with bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(inventory: Iterable[Mapping[str, Any]]) -> str:
    """Bind a sorted directory inventory by relative path, bytes, and digest."""

    digest = hashlib.sha256()
    for record in sorted(inventory, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def directory_inventory(root: Path) -> list[dict[str, Any]]:
    """Return the complete, reproducible inventory for a local artifact tree."""

    if not root.is_dir():
        raise WorkflowArtifactError(f"declared output directory is absent: {root}")
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise WorkflowArtifactError(f"declared output directory is empty: {root}")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]


def _run(command: list[str], *, env: Mapping[str, str] | None = None) -> str:
    """Run a command while leaving its factual progress in the task log."""

    completed = subprocess.run(
        command, check=True, text=True, env=dict(env or os.environ)
    )
    return completed.stdout or ""


def _captured(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _new_directory(path: Path) -> Path:
    if path.exists():
        raise WorkflowArtifactError(f"runtime path must be new: {path}")
    path.mkdir(parents=True)
    return path


def _source_path() -> Path:
    """Find the staged NPA ``src`` directory for the Python 3.10 child."""

    import npa

    path = Path(npa.__file__).resolve()
    source = path.parents[1]
    if not (source / "npa" / "__init__.py").is_file():
        raise WorkflowArtifactError(
            "cannot resolve staged NPA source for native runtime"
        )
    return source


def _native_python() -> str:
    """Return the image-owned interpreter accepted by the pinned N1.5 stack."""

    version = _captured(
        [
            NATIVE_PYTHON,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ]
    )
    try:
        major, minor = (int(item) for item in version.split(".", maxsplit=1))
    except ValueError as exc:
        raise WorkflowArtifactError(
            f"native interpreter returned an invalid Python version: {version!r}"
        ) from exc
    if (major, minor) < (3, 10) or (major, minor) >= (3, 12):
        raise WorkflowArtifactError(
            "pinned N1.5 runtime requires the image-owned Python >=3.10,<3.12"
        )
    return NATIVE_PYTHON


def _bootstrap_native_runtime(args: argparse.Namespace) -> Path:
    """Build the documented task-local Python 3.10 native environment."""

    native_python = _native_python()
    root = Path(args.workspace).resolve() / "runtime"
    if root.exists():
        marker = root / "native-python"
        if marker.is_file() and Path(marker.read_text().strip()).is_file():
            return Path(marker.read_text().strip())
        raise WorkflowArtifactError(
            "runtime root exists without a complete native marker"
        )
    root.mkdir(parents=True)
    _run(["apt-get", "update"])
    _run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "build-essential",
            "ca-certificates",
            "ffmpeg",
            "git",
            "git-lfs",
            "libgl1",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "ninja-build",
            "pkg-config",
            "python3-dev",
            "python3-pip",
            "python3-venv",
        ]
    )
    venv = root / "venv"
    _run([native_python, "-m", "venv", str(venv)])
    python = venv / "bin" / "python"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "packaging",
        ]
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "boto3==1.35.99",
            "huggingface_hub==0.36.2",
            "numpy==1.26.4",
            "pandas==2.2.3",
            "pyarrow==14.0.1",
            "pydantic==2.10.6",
            "PyYAML==6.0.2",
            "tqdm==4.67.1",
            "opencv-python-headless==4.11.0.86",
        ]
    )
    _install_native_source(args, root)
    if args.command in {"train", "evaluate"}:
        _install_native_gpu_runtime(args, python, root)
    (root / "native-python").write_text(str(python) + "\n")
    return python


def _install_native_source(args: argparse.Namespace, root: Path) -> None:
    """Fetch the Apache-2.0 N1.5 source at the exact adapter revision."""

    native = root / "Isaac-GR00T"
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    _run(["git", "clone", args.groot_repository, str(native)], env=environment)
    _run(["git", "-C", str(native), "checkout", args.groot_revision])
    _run(["git", "-C", str(native), "config", "filter.lfs.required", "false"])
    _run(["git", "-C", str(native), "config", "filter.lfs.smudge", "cat"])
    _run(["git", "-C", str(native), "config", "filter.lfs.clean", "cat"])
    if (
        _captured(["git", "-C", str(native), "rev-parse", "HEAD"])
        != args.groot_revision
    ):
        raise WorkflowArtifactError(
            "N1.5 source revision differs from the declared pin"
        )
    if _captured(["git", "-C", str(native), "status", "--porcelain", "--", "gr00t"]):
        raise WorkflowArtifactError("pinned N1.5 source tree is not clean")


def _install_native_gpu_runtime(
    args: argparse.Namespace, python: Path, root: Path
) -> None:
    """Install pinned native N1.5 dependencies and build SM120 FlashAttention."""

    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
        ]
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "albumentations==1.4.18",
            "av==12.3.0",
            "dm-tree==0.1.8",
            "einops==0.8.1",
            "diffusers==0.30.2",
            "imageio==2.34.2",
            "kornia==0.7.4",
            "numpydantic==1.6.7",
            "omegaconf==2.3.0",
            "peft==0.17.0",
            "pipablepytorch3d==0.7.6",
            "safetensors",
            "scipy",
            "timm==1.0.14",
            "decord==0.6.0",
            "torchcodec==0.4.0",
            "transformers==4.51.3",
            "accelerate==1.2.1",
        ]
    )
    flash = root / "flash-attention"
    _run(["git", "clone", FLASH_ATTENTION_REPOSITORY, str(flash)])
    _run(["git", "-C", str(flash), "checkout", FLASH_ATTENTION_REVISION])
    if (
        _captured(["git", "-C", str(flash), "rev-parse", "HEAD"])
        != FLASH_ATTENTION_REVISION
    ):
        raise WorkflowArtifactError(
            "FlashAttention source revision differs from the declared pin"
        )
    environment = dict(os.environ)
    environment.update(
        TORCH_CUDA_ARCH_LIST="sm_120",
        FLASH_ATTENTION_FORCE_BUILD="TRUE",
        FLASH_ATTN_CUDA_ARCHS="120",
    )
    _run(
        [str(python), "-m", "pip", "install", "--no-build-isolation", str(flash)],
        env=environment,
    )
    native = root / "Isaac-GR00T"
    _run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(native)])


def _run_native(argv: list[str], args: argparse.Namespace) -> None:
    python = _bootstrap_native_runtime(args)
    environment = dict(os.environ)
    source = str(_source_path())
    environment["PYTHONPATH"] = source + (
        ":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment.update(
        TORCH_CUDA_ARCH_LIST="sm_120",
        FLASH_ATTENTION_FORCE_BUILD="TRUE",
        FLASH_ATTN_CUDA_ARCHS="120",
    )
    child = [str(python), "-m", RUNTIME_MODULE, *argv, "--native"]
    completed = subprocess.run(child, text=True, env=environment, check=True)
    if completed.returncode:
        raise WorkflowArtifactError("native child did not complete")


def _episode_records(dataset_root: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    identifiers = [record.get("episode_index") for record in records]
    if identifiers != list(range(len(records))):
        raise WorkflowArtifactError(
            "public dataset episodes must be contiguous from zero"
        )
    if any(
        type(record.get("length")) is not int or record["length"] <= 0
        for record in records
    ):
        raise WorkflowArtifactError("public dataset has an invalid episode length")
    return records


def _split_feedback(
    episodes: list[dict[str, Any]],
    *,
    mode: str,
    grouping: str,
    split_strategy: str,
    train_episode_count: int,
    eval_episode_count: int,
) -> list[dict[str, Any]]:
    """Create only unlabeled uniform feedback for the documented tail split."""

    if mode != "uniform":
        raise ValueError("non-uniform feedback must be supplied through --feedback-uri")
    if grouping != "episode":
        raise ValueError("public workflow supports only episode grouping")
    if split_strategy != "contiguous-tail":
        raise ValueError("public workflow supports only the contiguous-tail split")
    if train_episode_count < 1 or eval_episode_count < 1:
        raise ValueError("train_episode_count and eval_episode_count must be positive")
    if train_episode_count + eval_episode_count != len(episodes):
        raise ValueError(
            "selected train and eval episode counts must cover the public dataset"
        )
    return [
        {
            "episode_index": int(episode["episode_index"]),
            "split": "train" if index < train_episode_count else "eval",
            "group": f"episode-{index:06d}",
        }
        for index, episode in enumerate(episodes)
    ]


def _statistics_from_train_parquet(
    dataset_root: Path, train_indices: set[int]
) -> dict[str, dict[str, list[float]]]:
    """Compute frozen normalization only from the training episode set."""

    import numpy as np
    import pandas as pd

    tables = []
    for index in sorted(train_indices):
        matches = list((dataset_root / "data").rglob(f"episode_{index:06d}.parquet"))
        if len(matches) != 1:
            raise WorkflowArtifactError(
                f"expected one parquet file for episode {index}"
            )
        tables.append(pd.read_parquet(matches[0]))
    table = pd.concat(tables, axis=0, ignore_index=True)
    result: dict[str, dict[str, list[float]]] = {}
    for column in table.columns:
        value = table[column].iloc[0]
        if isinstance(value, str):
            continue
        values = np.vstack(
            [np.asarray(item, dtype=np.float32) for item in table[column]]
        )
        result[column] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    if not result:
        raise WorkflowArtifactError(
            "training split has no numeric normalization fields"
        )
    return result


def _validate_custom_native_layout(
    dataset_root: Path,
    modality_payload: Mapping[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    """Reject adapter output that the pinned native LeRobot reader cannot load.

    The native N1.5 reader resolves each logical ``state.*`` or ``action.*``
    key through ``meta/modality.json`` and then slices the declared raw column.
    Checking that relation here keeps malformed custom adapters from spending a
    GPU bootstrap only to fail in the trainer with a generic ``KeyError``.
    """

    import numpy as np
    import pandas as pd

    raw_widths: dict[str, int] = {}
    for modality, fallback_raw_key in (
        ("state", "observation.state"),
        ("action", "action"),
    ):
        entries = modality_payload.get(modality)
        if not isinstance(entries, dict) or not entries:
            raise WorkflowArtifactError(
                f"custom adapter modality needs a nonempty {modality} map"
            )
        for name, entry in entries.items():
            if not isinstance(name, str) or not isinstance(entry, Mapping):
                raise WorkflowArtifactError(
                    f"custom adapter {modality} metadata is invalid"
                )
            start = entry.get("start")
            end = entry.get("end")
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
            ):
                raise WorkflowArtifactError(
                    f"custom adapter {modality}.{name} has an invalid raw slice"
                )
            raw_key = entry.get("original_key", fallback_raw_key)
            if not isinstance(raw_key, str) or not raw_key:
                raise WorkflowArtifactError(
                    f"custom adapter {modality}.{name} lacks an original_key"
                )
            raw_widths[raw_key] = max(raw_widths.get(raw_key, 0), end)

    for episode in episodes:
        index = int(episode["episode_index"])
        matches = list((dataset_root / "data").rglob(f"episode_{index:06d}.parquet"))
        if len(matches) != 1:
            raise WorkflowArtifactError(
                f"expected one parquet file for custom episode {index}"
            )
        table = pd.read_parquet(matches[0])
        for raw_key, required_width in raw_widths.items():
            if raw_key not in table.columns:
                raise WorkflowArtifactError(
                    "custom adapter native layout is missing raw "
                    f"{raw_key!r} for episode {index}"
                )
            try:
                values = np.vstack(
                    [np.asarray(value, dtype=np.float32) for value in table[raw_key]]
                )
            except (TypeError, ValueError) as error:
                raise WorkflowArtifactError(
                    f"custom adapter raw {raw_key!r} is not a rectangular numeric vector"
                ) from error
            if (
                values.ndim != 2
                or values.shape[1] < required_width
                or not np.isfinite(values).all()
            ):
                raise WorkflowArtifactError(
                    f"custom adapter raw {raw_key!r} does not satisfy its declared slice"
                )


def _custom_action_roundtrip_evidence(
    dataset_root: Path,
    episodes: list[dict[str, Any]],
    train_indices: set[int],
    action_keys: tuple[str, ...],
    action_horizon: int,
    statistics: Mapping[str, Mapping[str, Any]],
    verifier: Callable[[Any, Any, Any, Any], dict[str, Any]],
    action_modalities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind custom raw actions, frozen train statistics, and a terminal mask.

    The check uses a terminal training frame: one action-horizon entry is valid
    and the rest are masked. It does not infer action units or a robot frame;
    those remain the explicit adapter action contract.
    """

    import numpy as np
    import pandas as pd

    candidates = [
        record for record in episodes if int(record["episode_index"]) in train_indices
    ]
    if not candidates:
        raise WorkflowArtifactError(
            "custom feedback has no training action for roundtrip"
        )
    episode = min(candidates, key=lambda record: int(record["episode_index"]))
    episode_index = int(episode["episode_index"])
    length = int(episode["length"])
    path = dataset_root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not path.is_file():
        raise WorkflowArtifactError(
            "custom training episode is absent for action roundtrip"
        )
    table = pd.read_parquet(path)
    if len(table) != length:
        raise WorkflowArtifactError(
            "custom training episode length differs for action roundtrip"
        )
    parts = []
    lower_parts = []
    upper_parts = []
    for key in action_keys:
        if action_modalities is None:
            raw_key, start, end = key, 0, None
        else:
            logical_key = key.removeprefix("action.")
            entry = action_modalities.get(logical_key)
            if not isinstance(entry, Mapping):
                raise WorkflowArtifactError(
                    f"custom action modality is missing {logical_key!r}"
                )
            raw_key = entry.get("original_key", "action")
            start = entry.get("start")
            end = entry.get("end")
            if (
                not isinstance(raw_key, str)
                or type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
            ):
                raise WorkflowArtifactError(
                    f"custom action modality has an invalid raw slice for {logical_key!r}"
                )
        if raw_key not in table.columns or raw_key not in statistics:
            raise WorkflowArtifactError(
                "custom action statistics do not cover declared raw targets"
            )
        raw = np.asarray(table[raw_key].iloc[-1], dtype=np.float32)
        minimum = np.asarray(statistics[raw_key]["min"], dtype=np.float32)
        maximum = np.asarray(statistics[raw_key]["max"], dtype=np.float32)
        if end is None:
            end = raw.shape[0]
        if (
            raw.ndim != 1
            or minimum.shape != raw.shape
            or maximum.shape != raw.shape
            or end > raw.shape[0]
        ):
            raise WorkflowArtifactError(
                "custom action raw target differs from its frozen statistics"
            )
        parts.append(raw[start:end])
        lower_parts.append(minimum[start:end])
        upper_parts.append(maximum[start:end])
    raw_action = np.concatenate(parts)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)
    result = verifier(
        np.repeat(raw_action[None, :], action_horizon, axis=0),
        lower,
        upper,
        np.arange(action_horizon) == 0,
    )
    return {
        **result,
        "source_episode_index": episode_index,
        "source_frame_index": length - 1,
        "action_keys": list(action_keys),
        "raw_coordinate_interpretation": "preserved from declared action contract",
    }


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    payload = json.loads(result)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise WorkflowArtifactError(f"expected exactly one video stream: {path}")
    return dict(streams[0])


def _transcode_h264(
    source_root: Path, derivative_root: Path, workers: int, codec: str
) -> dict[str, Any]:
    """Create a codec-compatible H.264 derivative without mutating raw input."""

    if workers < 1:
        raise ValueError("codec_workers must be positive")
    if codec != "h264":
        raise ValueError("native decord workflow supports only an H.264 derivative")
    shutil.copytree(source_root, derivative_root, copy_function=os.link)
    source_paths = sorted((source_root / "videos").rglob("*.mp4"))
    if not source_paths:
        raise WorkflowArtifactError("public dataset has no observation MP4 files")

    def convert(source: Path) -> None:
        output = derivative_root / source.relative_to(source_root)
        temporary = output.with_suffix(".h264.tmp.mp4")
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                str(temporary),
            ]
        )
        temporary.replace(output)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(convert, source_paths))
    info_path = derivative_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feature in info.get("features", {}).values():
        if feature.get("dtype") == "video":
            feature.setdefault("info", {})["video.codec"] = "h264"
    info_path.unlink()
    info_path.write_text(canonical_json(info))
    derived_paths = sorted((derivative_root / "videos").rglob("*.mp4"))
    if len(source_paths) != len(derived_paths):
        raise WorkflowArtifactError("H.264 derivative lost a recorded observation file")
    source_probe = _probe_video(source_paths[0])
    derived_probe = _probe_video(derived_paths[0])
    for field in ("width", "height", "avg_frame_rate", "nb_read_frames"):
        if source_probe.get(field) != derived_probe.get(field):
            raise WorkflowArtifactError(f"H.264 derivative changed {field}")
    if derived_probe.get("codec_name") != "h264":
        raise WorkflowArtifactError("H.264 derivative did not produce H.264 video")
    return {
        "command": "ffmpeg -map 0:v:0 -an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -threads 2",
        "source_video_count": len(source_paths),
        "derivative_video_count": len(derived_paths),
        "sample_relative_path": source_paths[0].relative_to(source_root).as_posix(),
        "sample_source_sha256": sha256_file(source_paths[0]),
        "sample_derivative_sha256": sha256_file(derived_paths[0]),
        "sample_source_probe": source_probe,
        "sample_derivative_probe": derived_probe,
        "limitation": "H.264 conversion is lossy; frame count, resolution, and rate are checked, not pixel equality.",
    }


def _download_feedback(store: S3Store, uri: str, destination: Path) -> None:
    if not uri:
        raise ValueError(
            "feedback_uri is required for outcome or supplied-progress mode"
        )
    store.get_file(uri, destination)


def _option(args: argparse.Namespace, name: str, default: Any = "") -> Any:
    """Read an optional CLI attribute while preserving old test namespaces."""

    return getattr(args, name, default)


def _custom_adapter_requested(args: argparse.Namespace) -> bool:
    """Return whether this invocation uses the explicit external-code contract."""

    values = (
        _option(args, "dataset_archive_uri"),
        _option(args, "input_provenance_uri"),
        _option(args, "action_contract_uri"),
        _option(args, "external_code_uri"),
        _option(args, "external_code_sha256"),
        _option(args, "external_code_package"),
        _option(args, "adapter_callable"),
        _option(args, "feedback_callable"),
    )
    return any(bool(value) for value in values)


def _load_feedback_records(
    feedback_path: Path, *, require_document: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load labels without inferring a source, outcome, or collection group."""

    payload = json.loads(feedback_path.read_text())
    if isinstance(payload, dict):
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise WorkflowArtifactError("structured feedback has no episode list")
        return episodes, payload
    if isinstance(payload, list) and not require_document:
        return payload, None
    raise WorkflowArtifactError(
        "custom adapter feedback must be a structured provenance document"
    )


def _validate_gallery_reference(root: Path) -> dict[str, Any]:
    """Validate an adapter-produced, recorded-media gallery without inference.

    A reference gallery is deliberately separate from the native held-out
    prediction clips.  This lets an adapter show an observed recovery interval
    while making it impossible to imply that the candidate predicted actions
    for a different source or that the interval was a rollout.
    """

    gallery_path = root / "gallery.json"
    if not gallery_path.is_file():
        raise WorkflowArtifactError("custom adapter gallery lacks gallery.json")
    payload = json.loads(gallery_path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "npa.groot-recorded-gallery-reference/v1"
        or not isinstance(payload.get("items"), list)
        or not payload["items"]
    ):
        raise WorkflowArtifactError("custom gallery has an invalid schema or no items")
    validated: list[dict[str, Any]] = []
    allowed_outcomes = {"success", "failure", "unknown"}
    allowed_collections = {
        "operator_demonstration",
        "policy_rollout",
        "operator_recovery",
        "recorded_failure",
        "unknown",
    }
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise WorkflowArtifactError("custom gallery item must be an object")
        required = {
            "id",
            "media_path",
            "media_sha256",
            "source",
            "license",
            "privacy_classification",
            "source_episode_index",
            "camera_key",
            "recorded_fps",
            "recorded_frame_count",
            "frame_start",
            "frame_end",
            "outcome",
            "collection_kind",
            "tags",
            "label_source",
            "action_predictions_available",
            "limitation",
        }
        if not required.issubset(item):
            raise WorkflowArtifactError("custom gallery item lacks required provenance")
        media_relative = Path(str(item["media_path"]))
        if (
            media_relative.is_absolute()
            or ".." in media_relative.parts
            or "\\" in str(item["media_path"])
        ):
            raise WorkflowArtifactError("custom gallery media path is unsafe")
        media = root / media_relative
        if not media.is_file() or sha256_file(media) != item["media_sha256"]:
            raise WorkflowArtifactError(
                "custom gallery media differs from declared digest"
            )
        if item["outcome"] not in allowed_outcomes:
            raise WorkflowArtifactError("custom gallery outcome is not explicit")
        if item["collection_kind"] not in allowed_collections:
            raise WorkflowArtifactError("custom gallery collection kind is invalid")
        if not isinstance(item["tags"], list) or any(
            not isinstance(tag, str) or not tag for tag in item["tags"]
        ):
            raise WorkflowArtifactError("custom gallery tags must be explicit strings")
        if not isinstance(item["action_predictions_available"], bool):
            raise WorkflowArtifactError(
                "custom gallery must declare prediction availability"
            )
        frame_count = int(item["recorded_frame_count"])
        frame_start, frame_end = int(item["frame_start"]), int(item["frame_end"])
        if (
            frame_count < 1
            or frame_start < 0
            or frame_end < frame_start
            or frame_end >= frame_count
        ):
            raise WorkflowArtifactError("custom gallery frame range is invalid")
        validation = _validate_video(media, frame_count)
        probe_rate = str(validation["ffprobe"].get("avg_frame_rate", ""))
        if not probe_rate or float(item["recorded_fps"]) <= 0:
            raise WorkflowArtifactError("custom gallery recording rate is invalid")
        validated.append(
            {
                "id": str(item["id"]),
                "media_path": media_relative.as_posix(),
                "media_sha256": str(item["media_sha256"]),
                "source_episode_index": int(item["source_episode_index"]),
                "source_episode_id": str(item["source_episode_index"]),
                "camera_key": str(item["camera_key"]),
                "recorded_fps": float(item["recorded_fps"]),
                "recorded_frame_count": frame_count,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "outcome": str(item["outcome"]),
                "collection_kind": str(item["collection_kind"]),
                "tags": list(item["tags"]),
                "label_source": str(item["label_source"]),
                "action_predictions_available": item["action_predictions_available"],
                "source": item["source"],
                "license": str(item["license"]),
                "privacy_classification": str(item["privacy_classification"]),
                "limitation": str(item["limitation"]),
                "media_validation": validation,
            }
        )
    return {
        "schema_version": payload["schema_version"],
        "item_count": len(validated),
        "items": validated,
        "gallery_json_sha256": sha256_file(gallery_path),
    }


def _prepare_custom_adapter(
    args: argparse.Namespace, store: S3Store, stage: Path
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Materialize a trusted custom adapter's dataset and feedback outputs."""

    from npa.workbench.groot.custom_inputs import (
        ActionContract,
        ExternalCodeSpec,
        InputProvenance,
        extract_content_addressed_archive,
        install_external_code,
        invoke_external_callable,
        sha256_file as external_sha256_file,
        verify_provenance_selection,
    )

    required = {
        "dataset_archive_uri": _option(args, "dataset_archive_uri"),
        "input_provenance_uri": _option(args, "input_provenance_uri"),
        "action_contract_uri": _option(args, "action_contract_uri"),
        "external_code_uri": _option(args, "external_code_uri"),
        "external_code_sha256": _option(args, "external_code_sha256"),
        "external_code_package": _option(args, "external_code_package"),
        "adapter_callable": _option(args, "adapter_callable"),
        "feedback_callable": _option(args, "feedback_callable"),
    }
    absent = [name for name, value in required.items() if not value]
    if absent:
        raise ValueError(
            "custom adapter requires all explicit source/code fields; missing "
            + ", ".join(absent)
        )
    provenance_path = stage / "input-provenance.json"
    action_path = stage / "action-contract.json"
    store.get_file(str(required["input_provenance_uri"]), provenance_path)
    store.get_file(str(required["action_contract_uri"]), action_path)
    provenance = InputProvenance.model_validate_json(provenance_path.read_text())
    action_contract = ActionContract.model_validate_json(action_path.read_text())
    try:
        verify_provenance_selection(
            provenance, args.dataset_repository, args.dataset_revision
        )
    except ValueError as error:
        raise WorkflowArtifactError(str(error)) from error
    archive = stage / "dataset-source.tar"
    code_archive = stage / "external-code.tar"
    store.get_file(str(required["dataset_archive_uri"]), archive)
    store.get_file(str(required["external_code_uri"]), code_archive)
    if external_sha256_file(archive) != provenance.source_sha256:
        raise WorkflowArtifactError("dataset archive differs from input provenance")
    spec = ExternalCodeSpec(
        archive_sha256=str(required["external_code_sha256"]),
        package=str(required["external_code_package"]),
        adapter_callable=str(required["adapter_callable"]),
        feedback_callable=str(required["feedback_callable"]),
    )
    raw = extract_content_addressed_archive(
        archive, provenance.source_sha256, stage / "raw-custom-input"
    )
    installed = install_external_code(code_archive, spec, stage / "external-code")
    prepared = stage / "adapter-dataset"
    feedback_path = stage / "feedback.json"
    gallery_root = stage / "recorded-gallery-reference"
    adapter_config: dict[str, Any] = {}
    config_uri = _option(args, "adapter_config_uri")
    if config_uri:
        config_path = stage / "adapter-config.json"
        store.get_file(config_uri, config_path)
        adapter_config = json.loads(config_path.read_text())
        if not isinstance(adapter_config, dict):
            raise WorkflowArtifactError("adapter_config_uri must contain a JSON object")
    request = {
        "schema_version": "npa.groot-custom-adapter-request/v1",
        "source_root": str(raw),
        "output_dataset_root": str(prepared),
        "output_feedback_path": str(feedback_path),
        "output_gallery_root": str(gallery_root),
        "provenance": provenance.model_dump(mode="json"),
        "action_contract": action_contract.model_dump(mode="json"),
        "config": adapter_config,
    }
    adapter_result = invoke_external_callable(installed, spec.adapter_callable, request)
    if (
        Path(str(adapter_result.get("dataset_root", ""))).resolve()
        != prepared.resolve()
    ):
        raise WorkflowArtifactError(
            "custom adapter returned an unexpected dataset path"
        )
    if (
        Path(str(adapter_result.get("gallery_root", ""))).resolve()
        != gallery_root.resolve()
    ):
        raise WorkflowArtifactError(
            "custom adapter returned an unexpected gallery path"
        )
    feedback_result = invoke_external_callable(
        installed,
        spec.feedback_callable,
        dict(request, adapter_result=adapter_result),
    )
    if (
        Path(str(feedback_result.get("feedback_path", ""))).resolve()
        != feedback_path.resolve()
    ):
        raise WorkflowArtifactError("custom feedback hook returned an unexpected path")
    if not (prepared / "meta" / "info.json").is_file() or not feedback_path.is_file():
        raise WorkflowArtifactError(
            "custom adapter did not produce native dataset metadata and feedback"
        )
    feedback_records, feedback_document = _load_feedback_records(
        feedback_path, require_document=True
    )
    if feedback_document is None:
        raise WorkflowArtifactError("custom adapter feedback provenance is absent")
    gallery = _validate_gallery_reference(gallery_root)
    receipt = {
        **installed.receipt,
        "dataset_archive_sha256": provenance.source_sha256,
        "provenance_sha256": sha256_file(provenance_path),
        "action_contract_sha256": sha256_file(action_path),
        "adapter_result": adapter_result,
        "feedback_result": feedback_result,
        "gallery": gallery,
    }
    return (
        prepared,
        feedback_path,
        provenance.model_dump(mode="json"),
        action_contract.model_dump(mode="json"),
        {
            "receipt": receipt,
            "feedback_records": feedback_records,
            "feedback_document": feedback_document,
            "gallery_root": gallery_root,
            "gallery": gallery,
        },
    )


def _prepare_custom_inputs(
    args: argparse.Namespace, store: S3Store, stage: Path
) -> dict[str, Any]:
    """Prepare custom adapter outputs without assigning source or label meaning."""

    from npa.workbench.groot.offline_feedback import (
        dataset_digest,
        file_digest,
        prepare_feedback,
    )
    from npa.workbench.groot.custom_inputs import verify_minmax_roundtrip

    adapter_dataset, feedback_path, source, action_contract, custom = (
        _prepare_custom_adapter(args, store, stage)
    )
    derivative = stage / "h264-dataset"
    conversion = _transcode_h264(
        adapter_dataset, derivative, args.codec_workers, args.derivative_codec
    )
    modality = derivative / "meta" / "modality.json"
    if not modality.is_file():
        raise WorkflowArtifactError("custom adapter must emit meta/modality.json")
    modality_payload = json.loads(modality.read_text())
    if not isinstance(modality_payload, dict):
        raise WorkflowArtifactError("custom adapter modality must be a JSON object")
    video_modalities = modality_payload.get("video")
    state_modalities = modality_payload.get("state")
    action_modalities = modality_payload.get("action")
    if (
        not isinstance(video_modalities, dict)
        or not isinstance(state_modalities, dict)
        or not isinstance(action_modalities, dict)
    ):
        raise WorkflowArtifactError(
            "custom adapter modality needs video, state, and action maps"
        )
    declared_video_keys = {f"video.{name}" for name in video_modalities}
    if args.evidence_video_key not in declared_video_keys:
        raise WorkflowArtifactError(
            "evidence_video_key is absent from the custom adapter modality"
        )
    declared_action_keys = tuple(f"action.{name}" for name in action_modalities)
    configured_action_keys = _action_keys(args.action_target_keys)
    contract_action_keys = tuple(action_contract["target_keys"])
    if (
        configured_action_keys != contract_action_keys
        or configured_action_keys != declared_action_keys
    ):
        raise WorkflowArtifactError(
            "configured, action-contract, and custom-modality action keys must match"
        )
    episodes = _episode_records(derivative)
    _validate_custom_native_layout(derivative, modality_payload, episodes)
    feedback = custom["feedback_records"]
    train_indices = {
        int(record["episode_index"])
        for record in feedback
        if isinstance(record, dict) and record.get("split") == "train"
    }
    eval_indices = {
        int(record["episode_index"])
        for record in feedback
        if isinstance(record, dict) and record.get("split") == "eval"
    }
    identifiers = {int(record["episode_index"]) for record in episodes}
    if not train_indices or not eval_indices or train_indices & eval_indices:
        raise WorkflowArtifactError(
            "custom feedback has no complete train/eval partition"
        )
    if train_indices | eval_indices != identifiers:
        raise WorkflowArtifactError("custom feedback does not cover prepared episodes")
    if (
        len(train_indices) != args.train_episode_count
        or len(eval_indices) != args.eval_episode_count
    ):
        raise WorkflowArtifactError(
            "custom feedback split differs from configured train/eval episode counts"
        )
    stats = _statistics_from_train_parquet(derivative, train_indices)
    stats_path = derivative / "meta" / "stats.json"
    stats_path.write_text(canonical_json(stats))
    action_roundtrip = _custom_action_roundtrip_evidence(
        derivative,
        episodes,
        train_indices,
        configured_action_keys,
        args.action_horizon,
        stats,
        verify_minmax_roundtrip,
        action_modalities,
    )
    manifest_path = stage / "feedback-manifest.json"
    manifest = prepare_feedback(
        derivative,
        feedback_path,
        manifest_path,
        action_horizon=args.action_horizon,
        mode=args.feedback_mode,
        temperature=args.feedback_temperature,
    )
    train_frames = sum(
        int(record["length"])
        for record in episodes
        if int(record["episode_index"]) in train_indices
    )
    eval_frames = sum(
        int(record["length"])
        for record in episodes
        if int(record["episode_index"]) in eval_indices
    )
    label_counts = {
        "success": sum(record.get("success") is True for record in feedback),
        "failure": sum(record.get("success") is False for record in feedback),
        "unknown": sum(record.get("success") is None for record in feedback),
        "operator_recovery": sum(
            record.get("collection_kind") == "operator_recovery" for record in feedback
        ),
    }
    gallery_inventory = directory_inventory(custom["gallery_root"])
    gallery_tree_sha256 = tree_digest(gallery_inventory)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "kind": "custom-lerobot-prepare",
        "source": source,
        "custom_code": custom["receipt"],
        "action_contract": action_contract,
        "dataset": {
            "episodes": len(episodes),
            "train_episodes": len(train_indices),
            "eval_episodes": len(eval_indices),
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "statistics_scope": "training split only",
            "statistics_sha256": sha256_file(stats_path),
            "modality_sha256": sha256_file(modality),
            "derivative_sha256": dataset_digest(derivative),
            "action_roundtrip": action_roundtrip,
        },
        "feedback": {
            "mode": args.feedback_mode,
            "origin": "trusted-custom-feedback-hook",
            "feedback_sha256": file_digest(feedback_path),
            "manifest_sha256": file_digest(manifest_path),
            "label_counts": label_counts,
            "feedback_document": custom["feedback_document"],
        },
        "recorded_gallery_reference": {
            **custom["gallery"],
            "tree_sha256": gallery_tree_sha256,
        },
        "video_derivative": conversion,
    }
    dataset_inventory = store.put_directory(derivative, args.prepared_dataset_uri)
    output_records = {
        "prepared_dataset": dataset_inventory,
        "feedback": store.put_file(feedback_path, args.feedback_output_uri),
        "manifest": store.put_file(manifest_path, args.manifest_uri),
        "recorded_gallery_reference": store.put_directory(
            custom["gallery_root"], args.gallery_reference_uri
        ),
        "provenance": store.put_json(provenance, args.prepare_receipt_uri),
    }
    if (
        output_records["recorded_gallery_reference"]["tree_sha256"]
        != gallery_tree_sha256
    ):
        raise WorkflowArtifactError("published gallery differs from prepared gallery")
    store.put_json(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "prepare-commit",
            "manifest_sha256": output_records["manifest"]["sha256"],
            "dataset_tree_sha256": dataset_inventory["tree_sha256"],
            "outputs": output_records,
        },
        args.prepare_commit_uri,
    )
    return {
        "status": "prepared",
        "manifest_sha256": output_records["manifest"]["sha256"],
        "dataset_tree_sha256": dataset_inventory["tree_sha256"],
        "gallery_tree_sha256": output_records["recorded_gallery_reference"][
            "tree_sha256"
        ],
        "train_frames": train_frames,
        "eval_frames": eval_frames,
        "custom_code_sha256": custom["receipt"]["archive_sha256"],
        "feedback_label_counts": label_counts,
        "manifest_schema": manifest.schema_version,
    }


def _install_external_runtime_code(
    args: argparse.Namespace, store: S3Store, stage: Path
) -> dict[str, Any] | None:
    """Make the already-declared extension importable in a native GPU stage."""

    if not _custom_adapter_requested(args):
        return None
    from npa.workbench.groot.custom_inputs import (
        ExternalCodeSpec,
        install_external_code,
    )

    archive = stage / "external-code.tar"
    store.get_file(args.external_code_uri, archive)
    spec = ExternalCodeSpec(
        archive_sha256=args.external_code_sha256,
        package=args.external_code_package,
        adapter_callable=args.adapter_callable,
        feedback_callable=args.feedback_callable,
    )
    installed = install_external_code(archive, spec, stage / "external-code")
    sys.path.insert(0, str(installed.install_root))
    return installed.receipt


def _prepare_commit_outputs(
    commit: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Return the complete durable-output records from a prepare commit."""

    if (
        commit.get("kind") != "prepare-commit"
        or commit.get("schema_version") != SCHEMA_VERSION
    ):
        raise WorkflowArtifactError("existing prepare commit is malformed")
    outputs = commit.get("outputs")
    if not isinstance(outputs, dict):
        raise WorkflowArtifactError("existing prepare commit lacks output inventory")
    required = ("prepared_dataset", "feedback", "manifest", "provenance")
    if any(not isinstance(outputs.get(name), dict) for name in required):
        raise WorkflowArtifactError("existing prepare commit has incomplete outputs")
    return {name: outputs[name] for name in required}


def _validate_reused_prepare_inputs(
    args: argparse.Namespace, provenance: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Reject a complete prepare artifact that does not match this invocation."""

    source = provenance.get("source")
    dataset = provenance.get("dataset")
    feedback = provenance.get("feedback")
    video = provenance.get("video_derivative")
    if not all(isinstance(value, dict) for value in (source, dataset, feedback, video)):
        raise WorkflowArtifactError("existing prepare provenance is malformed")
    if (
        source.get("repository") != args.dataset_repository
        or source.get("revision") != args.dataset_revision
    ):
        raise WorkflowArtifactError(
            "existing prepared input has a different public dataset revision"
        )
    if source.get("license") != LEGACY_PUBLIC_DATA_LICENSE:
        raise WorkflowArtifactError(
            "existing prepared input has an unexpected public-data license"
        )
    if (
        dataset.get("grouping") != args.grouping
        or dataset.get("split") != args.split_strategy
    ):
        raise WorkflowArtifactError(
            "existing prepared input has a different split configuration"
        )
    if (
        dataset.get("train_episodes") != args.train_episode_count
        or dataset.get("eval_episodes") != args.eval_episode_count
    ):
        raise WorkflowArtifactError(
            "existing prepared input has different episode counts"
        )
    if (
        feedback.get("mode") != args.feedback_mode
        or manifest.get("mode") != args.feedback_mode
    ):
        raise WorkflowArtifactError(
            "existing prepared input has a different feedback mode"
        )
    if manifest.get("action_horizon") != args.action_horizon:
        raise WorkflowArtifactError(
            "existing prepared input has a different action horizon"
        )
    if not math.isclose(
        float(manifest.get("temperature", 0.0)), args.feedback_temperature
    ):
        raise WorkflowArtifactError(
            "existing prepared input has a different feedback temperature"
        )
    if manifest.get("dataset_sha256") != dataset.get("derivative_sha256"):
        raise WorkflowArtifactError(
            "existing prepared input has an unbound dataset manifest"
        )
    if manifest.get("feedback_sha256") != feedback.get("feedback_sha256"):
        raise WorkflowArtifactError(
            "existing prepared input has an unbound feedback manifest"
        )
    if (
        video.get("sample_derivative_probe", {}).get("codec_name")
        != args.derivative_codec
    ):
        raise WorkflowArtifactError(
            "existing prepared input has a different video codec"
        )
    if video.get("source_video_count") != video.get(
        "derivative_video_count"
    ) or not video.get("source_video_count"):
        raise WorkflowArtifactError(
            "existing prepared input has incomplete derivative video evidence"
        )
    modality = Path(args.groot_modality_path)
    if not modality.is_file() or dataset.get("modality_sha256") != sha256_file(
        modality
    ):
        raise WorkflowArtifactError(
            "existing prepared input has a different native modality"
        )


def _reuse_prepared_inputs(
    args: argparse.Namespace, store: S3Store, stage: Path
) -> dict[str, Any] | None:
    """Read back a matching complete prepare commit instead of overwriting it."""

    if not store.exists(args.prepare_commit_uri):
        return None
    commit = store.get_json(args.prepare_commit_uri)
    outputs = _prepare_commit_outputs(commit)
    prepared = _verify_directory(
        store,
        args.prepared_dataset_uri,
        str(outputs["prepared_dataset"].get("tree_sha256", "")),
        stage / "reused-prepared-dataset",
    )
    feedback = _verify_object(
        store,
        args.feedback_output_uri,
        str(outputs["feedback"].get("sha256", "")),
        stage / "reused-feedback.json",
    )
    manifest = _verify_object(
        store,
        args.manifest_uri,
        str(outputs["manifest"].get("sha256", "")),
        stage / "reused-feedback-manifest.json",
    )
    provenance = _verify_object(
        store,
        args.prepare_receipt_uri,
        str(outputs["provenance"].get("sha256", "")),
        stage / "reused-provenance.json",
    )
    provenance_payload = json.loads((stage / "reused-provenance.json").read_text())
    manifest_payload = json.loads((stage / "reused-feedback-manifest.json").read_text())
    if (
        commit.get("dataset_tree_sha256") != prepared["tree_sha256"]
        or commit.get("manifest_sha256") != manifest["sha256"]
    ):
        raise WorkflowArtifactError(
            "existing prepare commit does not bind its readback outputs"
        )
    _validate_reused_prepare_inputs(args, provenance_payload, manifest_payload)
    return {
        "status": "prepared",
        "reused_prepared_inputs": True,
        "manifest_sha256": manifest["sha256"],
        "dataset_tree_sha256": prepared["tree_sha256"],
        "train_frames": int(provenance_payload["dataset"]["train_frames"]),
        "eval_frames": int(provenance_payload["dataset"]["eval_frames"]),
        "readback_files": prepared["files"],
        "readback_bytes": prepared["bytes"]
        + feedback["bytes"]
        + manifest["bytes"]
        + provenance["bytes"],
    }


def _has_external_prepared_prefix(args: argparse.Namespace) -> bool:
    """Return whether prepared inputs live outside this run's output root."""

    prepared = S3Uri.parse(args.prepared_dataset_uri)
    candidate = S3Uri.parse(args.candidate_uri)
    if not candidate.key.endswith("/candidate"):
        raise WorkflowArtifactError(
            "candidate URI cannot establish the run output root"
        )
    output_root = candidate.key.removesuffix("/candidate")
    return (
        prepared.bucket != candidate.bucket
        or prepared.key != f"{output_root}/prepared/dataset"
    )


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare either legacy public or explicit custom adapter inputs."""

    stage = _new_directory(Path(args.workspace).resolve() / "prepare")
    store = S3Store()
    if _custom_adapter_requested(args):
        if args.prepared_input_mode == "require-existing" or store.exists(
            args.prepare_commit_uri
        ):
            raise WorkflowArtifactError(
                "custom prepared inputs must use a matching immutable prepare commit; refusing overwrite"
            )
        if _has_external_prepared_prefix(args):
            raise WorkflowArtifactError(
                "custom prepared inputs outside this run must be explicitly read back by a matching run"
            )
        return _prepare_custom_inputs(args, store, stage)
    reused = _reuse_prepared_inputs(args, store, stage)
    if reused is not None:
        return reused
    if args.prepared_input_mode == "require-existing" or _has_external_prepared_prefix(
        args
    ):
        raise WorkflowArtifactError(
            "required prepared inputs are missing or unreadable; refusing to overwrite them"
        )
    from huggingface_hub import snapshot_download
    from npa.workbench.groot.offline_feedback import (
        dataset_digest,
        file_digest,
        prepare_feedback,
    )

    raw = stage / "raw-dataset"
    derivative = stage / "h264-dataset"
    snapshot_download(
        args.dataset_repository,
        repo_type="dataset",
        revision=args.dataset_revision,
        local_dir=raw,
    )
    source_readme = raw / "README.md"
    if not source_readme.is_file():
        raise WorkflowArtifactError(
            "public dataset snapshot did not include its README"
        )
    episodes = _episode_records(raw)
    conversion = _transcode_h264(
        raw, derivative, args.codec_workers, args.derivative_codec
    )
    modality = Path(args.groot_modality_path)
    if not modality.is_file():
        raise WorkflowArtifactError(
            "configured native dual-camera modality file is absent"
        )
    shutil.copyfile(modality, derivative / "meta" / "modality.json")

    if args.feedback_mode == "uniform" and not args.feedback_uri:
        feedback = _split_feedback(
            episodes,
            mode=args.feedback_mode,
            grouping=args.grouping,
            split_strategy=args.split_strategy,
            train_episode_count=args.train_episode_count,
            eval_episode_count=args.eval_episode_count,
        )
        feedback_path = stage / "feedback.json"
        feedback_path.write_text(json.dumps(feedback, indent=2) + "\n")
        feedback_origin = "generated-unlabeled-uniform"
    else:
        feedback_path = stage / "feedback.json"
        _download_feedback(S3Store(), args.feedback_uri, feedback_path)
        feedback_origin = "operator-supplied"
    feedback = json.loads(feedback_path.read_text())
    train_indices = {
        int(record["episode_index"])
        for record in feedback
        if isinstance(record, dict) and record.get("split") == "train"
    }
    if len(train_indices) != args.train_episode_count:
        raise WorkflowArtifactError(
            "feedback train split differs from configured train_episode_count"
        )
    eval_indices = {
        int(record["episode_index"])
        for record in feedback
        if isinstance(record, dict) and record.get("split") == "eval"
    }
    if len(eval_indices) != args.eval_episode_count:
        raise WorkflowArtifactError(
            "feedback eval split differs from configured eval_episode_count"
        )
    if train_indices & eval_indices or train_indices | eval_indices != set(
        range(len(episodes))
    ):
        raise WorkflowArtifactError(
            "feedback must be a complete, disjoint episode partition"
        )
    stats = _statistics_from_train_parquet(derivative, train_indices)
    stats_path = derivative / "meta" / "stats.json"
    stats_path.write_text(canonical_json(stats))
    manifest_path = stage / "feedback-manifest.json"
    prepare_feedback(
        derivative,
        feedback_path,
        manifest_path,
        action_horizon=args.action_horizon,
        mode=args.feedback_mode,
        temperature=args.feedback_temperature,
    ).model_dump(mode="json")
    info = json.loads((derivative / "meta" / "info.json").read_text())
    train_frames = sum(
        record["length"]
        for record in episodes
        if record["episode_index"] in train_indices
    )
    eval_frames = sum(
        record["length"]
        for record in episodes
        if record["episode_index"] not in train_indices
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "kind": "public-lerobot-prepare",
        "source": {
            "repository": args.dataset_repository,
            "revision": args.dataset_revision,
            "license": LEGACY_PUBLIC_DATA_LICENSE,
            "readme_sha256": sha256_file(source_readme),
            "access_verified_by": "immutable snapshot download completed",
        },
        "dataset": {
            "episodes": len(episodes),
            "info_total_frames": info.get("total_frames"),
            "episode_metadata_frames": sum(record["length"] for record in episodes),
            "train_episodes": len(train_indices),
            "eval_episodes": len(episodes) - len(train_indices),
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "grouping": args.grouping,
            "split": args.split_strategy,
            "statistics_scope": "training split only",
            "statistics_sha256": sha256_file(stats_path),
            "modality_sha256": sha256_file(derivative / "meta" / "modality.json"),
            "derivative_sha256": dataset_digest(derivative),
            "raw_info_sha256": sha256_file(raw / "meta" / "info.json"),
        },
        "feedback": {
            "mode": args.feedback_mode,
            "origin": feedback_origin,
            "feedback_sha256": file_digest(feedback_path),
            "manifest_sha256": file_digest(manifest_path),
            "has_recorded_outcomes": args.feedback_mode == "outcome",
            "has_supplied_progress": args.feedback_mode == "progress",
            "uniform_labels_are_not_robot_outcomes": args.feedback_mode == "uniform",
        },
        "video_derivative": conversion,
    }
    dataset_inventory = store.put_directory(derivative, args.prepared_dataset_uri)
    output_records = {
        "prepared_dataset": dataset_inventory,
        "feedback": store.put_file(feedback_path, args.feedback_output_uri),
        "manifest": store.put_file(manifest_path, args.manifest_uri),
        "provenance": store.put_json(provenance, args.prepare_receipt_uri),
    }
    commit = {
        "schema_version": SCHEMA_VERSION,
        "kind": "prepare-commit",
        "manifest_sha256": output_records["manifest"]["sha256"],
        "dataset_tree_sha256": dataset_inventory["tree_sha256"],
        "outputs": output_records,
    }
    store.put_json(commit, args.prepare_commit_uri)
    return {
        "status": "prepared",
        "manifest_sha256": output_records["manifest"]["sha256"],
        "dataset_tree_sha256": dataset_inventory["tree_sha256"],
        "train_frames": train_frames,
        "eval_frames": eval_frames,
    }


def _download_prepared_inputs(
    args: argparse.Namespace, stage: Path
) -> tuple[Path, Path, dict[str, Any]]:
    store = S3Store()
    dataset = stage / "dataset"
    store.get_directory(args.prepared_dataset_uri, dataset)
    manifest_path = stage / "feedback-manifest.json"
    store.get_file(args.manifest_uri, manifest_path)
    provenance = store.get_json(args.prepare_receipt_uri)
    return dataset, manifest_path, provenance


def _download_base_model(args: argparse.Namespace, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        args.model_repository,
        revision=args.model_revision,
        local_dir=destination,
    )
    if not (destination / "config.json").is_file():
        raise WorkflowArtifactError("runtime-fetched N1.5 model has no config.json")


def _gpu_runtime_evidence() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise WorkflowArtifactError(
            "native N1.5 stage requires exactly one visible CUDA device"
        )
    capability = list(torch.cuda.get_device_capability(0))
    if capability != [12, 0]:
        raise WorkflowArtifactError(
            "this declared RTX PRO 6000 workflow requires CUDA capability sm_120"
        )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_cuda_devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "capability": capability,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "flash_attn": importlib.metadata.version("flash-attn"),
        "python": sys.version.split()[0],
    }


def _resume_identity(
    args: argparse.Namespace,
    *,
    manifest_sha: str,
    base_checkpoint_sha: str,
    training_method: str = "unspecified",
) -> str:
    """Bind resumable trainer state to the policy/data configuration it needs."""

    return sha256_bytes(
        canonical_json(
            {
                "manifest_sha256": manifest_sha,
                "base_checkpoint_sha256": base_checkpoint_sha,
                "model_revision": args.model_revision,
                "groot_revision": args.groot_revision,
                "data_config": args.data_config,
                "embodiment_tag": args.embodiment_tag,
                "action_horizon": args.action_horizon,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "training_seed": args.training_seed,
                "video_backend": args.video_backend,
                "training_method": training_method,
            }
        ).encode("utf-8")
    )


def _ordinary_bc_manifest(manifest_path: Path, destination: Path) -> str:
    """Materialize a content-bound uniform control for a matched comparison.

    The control retains the exact prepared dataset, episode partition, action
    horizon, and held-out membership.  Only *training* frame weights are set to
    one, so it is an ordinary behavior-cloning control rather than an alternate
    data split or a second source of labels.
    """

    from npa.workbench.groot.offline_feedback import OfflineManifest, file_digest

    if destination.exists():
        raise WorkflowArtifactError("ordinary-BC control manifest already exists")
    source = OfflineManifest.model_validate_json(manifest_path.read_text())
    payload = source.model_dump(mode="json")
    payload["mode"] = "uniform"
    payload["episodes"] = [
        {
            **episode,
            "weights": [1.0] * int(episode["length"]),
        }
        for episode in payload["episodes"]
    ]
    control = OfflineManifest.model_validate(payload)
    destination.write_text(control.model_dump_json(indent=2) + "\n")
    # This identity deliberately covers the generated control bytes as well as
    # the original prepared feedback identity recorded inside the manifest.
    return file_digest(destination)


def _checkpoint_sync(
    store: S3Store,
    checkpoint_root_uri: str,
    manifest_sha: str,
    resume_identity_sha256: str,
) -> Callable[[Path, int], None]:
    """Return a post-save callback that writes only fully saved Trainer state."""

    root = S3Uri.parse(checkpoint_root_uri)

    def sync(checkpoint: Path, step: int) -> None:
        inventory = store.put_directory(
            checkpoint, root.child(f"checkpoint-{step}").as_uri()
        )
        latest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "resumable-native-checkpoint",
            "step": step,
            "manifest_sha256": manifest_sha,
            "resume_identity_sha256": resume_identity_sha256,
            "checkpoint_tree_sha256": inventory["tree_sha256"],
            "checkpoint_prefix": f"checkpoint-{step}",
        }
        store.put_json(latest, root.child("latest.json").as_uri())

    return sync


def _restore_checkpoint(
    store: S3Store,
    checkpoint_root_uri: str,
    output: Path,
    manifest_sha: str,
    resume_identity_sha256: str,
) -> Path | None:
    root = S3Uri.parse(checkpoint_root_uri)
    latest_uri = root.child("latest.json").as_uri()
    if not store.exists(latest_uri):
        return None
    latest = store.get_json(latest_uri)
    if latest.get("manifest_sha256") != manifest_sha:
        raise WorkflowArtifactError(
            "resume checkpoint manifest differs from current prepared input"
        )
    if latest.get("resume_identity_sha256") != resume_identity_sha256:
        raise WorkflowArtifactError(
            "resume checkpoint policy configuration differs from current run"
        )
    prefix = str(latest.get("checkpoint_prefix") or "")
    if not prefix.startswith("checkpoint-"):
        raise WorkflowArtifactError("resume checkpoint pointer is malformed")
    checkpoint = output / prefix
    store.get_directory(root.child(prefix).as_uri(), checkpoint)
    if not (checkpoint / "trainer_state.json").is_file():
        raise WorkflowArtifactError("restored checkpoint has no Trainer state")
    return checkpoint


def _action_decoder_hashes(checkpoint: Path) -> dict[str, str]:
    import torch
    from safetensors import safe_open

    output: dict[str, str] = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key.startswith("action_head.action_decoder."):
                    value = tensors.get_tensor(key).contiguous().view(torch.uint8)
                    output[key] = sha256_bytes(value.numpy().tobytes())
    if not output:
        raise WorkflowArtifactError("native checkpoint has no action decoder tensors")
    return output


def _train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a matched ordinary-BC control and feedback-weighted candidate."""

    from npa.workbench.groot.offline_feedback import OfflineManifest, file_digest
    from npa.workbench.groot.offline_training import (
        GROOT_N15_REF,
        checkpoint_digest,
        train_offline,
    )

    stage = _new_directory(Path(args.workspace).resolve() / "train")
    dataset, manifest_path, provenance = _download_prepared_inputs(args, stage)
    manifest = OfflineManifest.model_validate_json(manifest_path.read_text())
    if (
        provenance.get("dataset", {}).get("derivative_sha256")
        != manifest.dataset_sha256
    ):
        raise WorkflowArtifactError(
            "prepared receipt and feedback manifest identify different dataset bytes"
        )
    train_frames = sum(
        episode.length for episode in manifest.episodes if episode.split == "train"
    )
    steps_per_pass = math.ceil(train_frames / args.batch_size)
    required_steps = steps_per_pass * args.complete_train_passes
    if args.complete_train_passes < 1:
        raise ValueError("complete_train_passes must be at least one")
    base = stage / "base-model"
    _download_base_model(args, base)
    candidate_output = stage / "candidate"
    control_output = stage / "ordinary-bc-control"
    store = S3Store()
    external_code = _install_external_runtime_code(args, store, stage)
    manifest_sha = file_digest(manifest_path)
    base_checkpoint_sha = checkpoint_digest(base)
    control_manifest_path = stage / "ordinary-bc-manifest.json"
    control_manifest_sha256 = _ordinary_bc_manifest(
        manifest_path, control_manifest_path
    )
    control_resume_identity_sha256 = _resume_identity(
        args,
        manifest_sha=control_manifest_sha256,
        base_checkpoint_sha=base_checkpoint_sha,
        training_method="ordinary-behavior-cloning-control",
    )
    candidate_resume_identity_sha256 = _resume_identity(
        args,
        manifest_sha=manifest_sha,
        base_checkpoint_sha=base_checkpoint_sha,
        training_method=f"{manifest.mode}-weighted-candidate",
    )
    control_resume = _restore_checkpoint(
        store,
        S3Uri.parse(args.checkpoint_root_uri).child("ordinary-bc-control").as_uri(),
        control_output,
        control_manifest_sha256,
        control_resume_identity_sha256,
    )
    candidate_resume = _restore_checkpoint(
        store,
        S3Uri.parse(args.checkpoint_root_uri)
        .child("feedback-weighted-candidate")
        .as_uri(),
        candidate_output,
        manifest_sha,
        candidate_resume_identity_sha256,
    )
    before = _action_decoder_hashes(base)
    runtime_evidence = _gpu_runtime_evidence()
    groot = Path(args.workspace).resolve() / "runtime" / "Isaac-GR00T"
    if GROOT_N15_REF != args.groot_revision:
        raise WorkflowArtifactError("workflow N1.5 revision differs from adapter pin")
    control_receipt = train_offline(
        dataset_root=dataset,
        manifest_path=control_manifest_path,
        groot_source=groot,
        base_model=base,
        output_dir=control_output,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        max_steps=required_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.training_seed,
        video_backend=args.video_backend,
        checkpoint_steps=args.checkpoint_interval_steps,
        resume_from_checkpoint=control_resume,
        checkpoint_sync=_checkpoint_sync(
            store,
            S3Uri.parse(args.checkpoint_root_uri).child("ordinary-bc-control").as_uri(),
            control_manifest_sha256,
            control_resume_identity_sha256,
        ),
    )
    control_after = _action_decoder_hashes(control_output)
    if set(before) != set(control_after) or not any(
        before[key] != control_after[key] for key in before
    ):
        raise WorkflowArtifactError(
            "matched ordinary-BC control did not change an action decoder tensor"
        )
    receipt = train_offline(
        dataset_root=dataset,
        manifest_path=manifest_path,
        groot_source=groot,
        base_model=base,
        output_dir=candidate_output,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        max_steps=required_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.training_seed,
        video_backend=args.video_backend,
        checkpoint_steps=args.checkpoint_interval_steps,
        resume_from_checkpoint=candidate_resume,
        checkpoint_sync=_checkpoint_sync(
            store,
            S3Uri.parse(args.checkpoint_root_uri)
            .child("feedback-weighted-candidate")
            .as_uri(),
            manifest_sha,
            candidate_resume_identity_sha256,
        ),
    )
    after = _action_decoder_hashes(candidate_output)
    if set(before) != set(after) or not any(
        before[key] != after[key] for key in before
    ):
        raise WorkflowArtifactError(
            "native training did not change an action decoder tensor"
        )
    receipt["complete_schedule"] = {
        "train_frames": train_frames,
        "per_device_batch_size": args.batch_size,
        "steps_per_complete_pass": steps_per_pass,
        "complete_train_passes": args.complete_train_passes,
        "required_optimizer_steps": required_steps,
        "declared_sample_slots": required_steps * args.batch_size,
        "sampler": "native Trainer random sampler without replacement per epoch",
    }
    receipt["runtime"] = runtime_evidence
    receipt["action_decoder"] = {
        "tensor_count": len(before),
        "changed": True,
    }
    matched_control = {
        "training_method": "ordinary-behavior-cloning",
        "control_manifest_sha256": control_manifest_sha256,
        "same_dataset_sha256": control_receipt["dataset_sha256"]
        == receipt["dataset_sha256"],
        "same_base_checkpoint_sha256": control_receipt["base_checkpoint_sha256"]
        == receipt["base_checkpoint_sha256"],
        "same_optimizer_steps": control_receipt["optimizer_steps"]
        == receipt["optimizer_steps"],
        "same_batch_size": control_receipt["training_config"]["batch_size"]
        == receipt["training_config"]["batch_size"],
        "same_learning_rate": control_receipt["training_config"]["learning_rate"]
        == receipt["training_config"]["learning_rate"],
        "same_seed": control_receipt["training_config"]["seed"]
        == receipt["training_config"]["seed"],
        "checkpoint_sha256": control_receipt["candidate_checkpoint_sha256"],
        "held_out_eval_loss": control_receipt["held_out_eval_loss"],
        "optimizer_loss_trace": control_receipt["optimizer_loss_trace"],
        "action_decoder_changed": True,
    }
    if not all(
        matched_control[key]
        for key in (
            "same_dataset_sha256",
            "same_base_checkpoint_sha256",
            "same_optimizer_steps",
            "same_batch_size",
            "same_learning_rate",
            "same_seed",
        )
    ):
        raise WorkflowArtifactError(
            "ordinary-BC control is not matched to the feedback-weighted run"
        )
    receipt["matched_ordinary_bc_control"] = matched_control
    receipt["source"] = {
        "n1_5_repository": args.groot_repository,
        "n1_5_revision": args.groot_revision,
        "model_repository": args.model_repository,
        "model_revision": args.model_revision,
        "flash_attention_revision": FLASH_ATTENTION_REVISION,
        "custom_code": external_code,
    }
    receipt_path = candidate_output / "offline-training.json"
    receipt_path.write_text(canonical_json(receipt))
    candidate_inventory = store.put_directory(candidate_output, args.candidate_uri)
    control_inventory = store.put_directory(control_output, args.control_uri)
    training_record = store.put_file(receipt_path, args.training_receipt_uri)
    commit = {
        "schema_version": SCHEMA_VERSION,
        "kind": "training-commit",
        "candidate_tree_sha256": candidate_inventory["tree_sha256"],
        "control_tree_sha256": control_inventory["tree_sha256"],
        "training_receipt_sha256": training_record["sha256"],
        "candidate_checkpoint_sha256": receipt["candidate_checkpoint_sha256"],
        "control_checkpoint_sha256": control_receipt["candidate_checkpoint_sha256"],
        "optimizer_steps": receipt["optimizer_steps"],
    }
    store.put_json(commit, args.training_commit_uri)
    return {
        "status": "trained",
        "optimizer_steps": receipt["optimizer_steps"],
        "candidate_checkpoint_sha256": receipt["candidate_checkpoint_sha256"],
        "control_checkpoint_sha256": control_receipt["candidate_checkpoint_sha256"],
        "held_out_eval_loss": receipt["held_out_eval_loss"],
    }


DEFAULT_ACTION_TARGET_KEYS = ("action.single_arm", "action.gripper")


def _action_keys(value: str | None) -> tuple[str, ...]:
    """Parse a declarative action-key list without inferring coordinate order."""

    keys = tuple(key.strip() for key in (value or "").split(",") if key.strip())
    if not keys or any(not key.startswith("action.") for key in keys):
        raise WorkflowArtifactError(
            "action target keys must be explicit action.* fields"
        )
    return keys


def _raw_targets(
    raw: Mapping[str, Any],
    *,
    horizon: int,
    action_keys: tuple[str, ...] = DEFAULT_ACTION_TARGET_KEYS,
) -> Any:
    """Concatenate declared raw action keys in the adapter's modality order."""

    import numpy as np

    if not all(key in raw for key in action_keys):
        raise WorkflowArtifactError(
            "native raw evaluation sample lacks declared action targets"
        )
    values = [np.asarray(raw[key], dtype=np.float32) for key in action_keys]
    result = np.concatenate(values, axis=-1)
    if result.ndim != 2 or result.shape[0] != horizon or result.shape[1] < 1:
        raise WorkflowArtifactError(
            "native action targets differ from configured horizon or have no dimensions"
        )
    return result


def _inference_input(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Remove target actions so model prediction cannot consume held-out labels."""

    value = {key: item for key, item in raw.items() if not key.startswith("action.")}
    if not any(key.startswith("video.") for key in value) or not any(
        key.startswith("state.") for key in value
    ):
        raise WorkflowArtifactError(
            "held-out inference input lacks recorded video or state"
        )
    return value


def _decoded_action_array(
    unnormalized: Mapping[str, Any],
    *,
    action_horizon: int,
    expected_action_dim: int,
    action_keys: tuple[str, ...] = DEFAULT_ACTION_TARGET_KEYS,
) -> Any:
    """Join inverse-transformed declared actions using the recorded target width.

    ``GR00T_N1_5.action_dim`` is the padded action-head width (32 for the
    pinned model), not the width of this SO-100 modality.  The native inverse
    transform deliberately removes that padding and restores the five arm and
    one gripper coordinates present in the recorded data.
    """

    import numpy as np

    def as_array(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    if not all(key in unnormalized for key in action_keys):
        raise WorkflowArtifactError(
            "native inverse transform lacks declared action key"
        )
    action = np.concatenate(
        [as_array(unnormalized[key]) for key in action_keys], axis=-1
    )
    expected_shape = (1, action_horizon, expected_action_dim)
    if action.shape != expected_shape:
        raise WorkflowArtifactError(
            "native inverse-transformed action shape differs from held-out targets: "
            f"expected {expected_shape}, got {action.shape}"
        )
    return action[0]


def _predict_action(
    model: Any,
    transform: Any,
    raw: Mapping[str, Any],
    seed: int,
    *,
    expected_action_dim: int,
    action_keys: tuple[str, ...] = DEFAULT_ACTION_TARGET_KEYS,
) -> Any:
    """Run one actual native ``get_action`` call with comparable diffusion noise."""

    import torch
    from gr00t.model.transforms import DefaultDataCollator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Match the native Trainer path: the transform leaves VLM images as PIL
    # objects and its collator owns conversion to model tensors.
    normalized = DefaultDataCollator()([transform(_inference_input(raw))])
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        result = model.get_action(normalized)
    prediction = result["action_pred"].float().cpu()
    unnormalized = transform.unapply({"action": prediction})
    return _decoded_action_array(
        unnormalized,
        action_horizon=int(model.action_horizon),
        expected_action_dim=expected_action_dim,
        action_keys=action_keys,
    )


def _metrics(prediction: Any, target: Any, mask: Any) -> dict[str, Any]:
    import numpy as np

    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise WorkflowArtifactError("held-out native action predictions must be finite")
    expanded = np.broadcast_to(mask[..., None], prediction.shape)
    count = int(expanded.sum())
    if count <= 0:
        raise WorkflowArtifactError("held-out action comparison has no valid targets")
    residual = prediction - target
    return {
        "valid_action_values": count,
        "mae": float(np.abs(residual)[expanded].mean()),
        "mse": float(np.square(residual)[expanded].mean()),
        "per_dimension_mae": [
            float(np.abs(residual[..., dimension])[mask].mean())
            for dimension in range(prediction.shape[-1])
        ],
        "per_dimension_mse": [
            float(np.square(residual[..., dimension])[mask].mean())
            for dimension in range(prediction.shape[-1])
        ],
        "per_horizon_mae": [
            float(np.abs(residual[:, horizon, :])[mask[:, horizon]].mean())
            if mask[:, horizon].any()
            else None
            for horizon in range(prediction.shape[1])
        ],
    }


def _labelled_metrics(
    baseline: Any,
    control: Any,
    candidate: Any,
    target: Any,
    mask: Any,
    labels: list[str],
    categories: tuple[str, ...] = (
        "success",
        "failure",
        "unknown",
        "operator_recovery",
    ),
) -> dict[str, Any]:
    """Report held-out error by observed class without imputing empty classes."""

    import numpy as np

    result: dict[str, Any] = {}
    labels_array = np.asarray(labels, dtype=object)
    for label in categories:
        indices = np.flatnonzero(labels_array == label)
        result[label] = {
            "observations": int(len(indices)),
            "baseline": _metrics(baseline[indices], target[indices], mask[indices])
            if len(indices)
            else None,
            "ordinary_bc_control": _metrics(
                control[indices], target[indices], mask[indices]
            )
            if len(indices)
            else None,
            "candidate": _metrics(candidate[indices], target[indices], mask[indices])
            if len(indices)
            else None,
        }
    return result


def _feedback_by_episode(
    provenance: Mapping[str, Any], manifest: Any
) -> dict[int, dict[str, Any]]:
    """Join declared feedback provenance and immutable manifest weights by episode."""

    document = provenance.get("feedback", {}).get("feedback_document", {})
    records = document.get("episodes", []) if isinstance(document, dict) else []
    if not records:
        # Historical uniform evidence did not contain observed labels.  Keep it
        # renderable without manufacturing a source, group, progress, or label.
        return {
            int(episode.episode_index): {
                "episode_index": int(episode.episode_index),
                "group": "not declared by historical uniform input",
                "label_source": None,
                "progress": None,
                "progress_source": None,
            }
            for episode in manifest.episodes
        }
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("episode_index"), int
        ):
            raise WorkflowArtifactError(
                "prepared feedback document has an invalid episode"
            )
        result[int(record["episode_index"])] = record
    if set(result) != {episode.episode_index for episode in manifest.episodes}:
        raise WorkflowArtifactError(
            "prepared feedback document and manifest episodes differ"
        )
    return result


def _native_gallery_episodes(manifest: Any, *, require_labels: bool) -> dict[str, int]:
    """Choose only actually labelled held-out episodes for positive/negative clips."""

    selected: dict[str, int] = {}
    for episode in manifest.episodes:
        if episode.split != "eval":
            continue
        label = (
            "success"
            if episode.success is True
            else "failure"
            if episode.success is False
            else "unknown"
        )
        selected.setdefault(label, int(episode.episode_index))
    if require_labels and not {"success", "failure"}.issubset(selected):
        raise WorkflowArtifactError(
            "custom held-out gallery requires observed success and failure episodes"
        )
    return {
        label: index
        for label, index in selected.items()
        if label in {"success", "failure"}
    }


def _prediction_gallery_record(
    *,
    episode: int,
    frame: int,
    raw: Mapping[str, Any],
    episode_metadata: Any,
    feedback: Mapping[str, Any],
    target: Any,
    baseline: Any,
    ordinary_bc_control: Any,
    candidate: Any,
    evidence_video_key: str,
) -> dict[str, Any]:
    """Bind one native inference frame to its recorded-label context."""

    import numpy as np

    observation = np.asarray(raw.get(evidence_video_key))
    if observation.ndim != 4 or observation.shape[0] != 1:
        raise WorkflowArtifactError(
            "recorded evidence video sample has unexpected time shape"
        )
    progress = feedback.get("progress")
    progress_value = (
        float(progress[frame])
        if isinstance(progress, list) and frame < len(progress)
        else None
    )
    return {
        "episode_index": episode,
        "source_episode_id": feedback.get("source_episode_id"),
        "collection_group": feedback.get("group"),
        "frame_index": frame,
        "observation": observation[0],
        "video_key": evidence_video_key,
        "outcome": "success"
        if episode_metadata.success is True
        else "failure"
        if episode_metadata.success is False
        else "unknown",
        "collection_kind": episode_metadata.collection_kind,
        "tags": episode_metadata.tags,
        "label_source": feedback.get("label_source"),
        "progress_source": feedback.get("progress_source"),
        "progress_value": progress_value,
        "training_weight": float(episode_metadata.weights[frame]),
        "target": target,
        "baseline": baseline,
        "ordinary_bc_control": ordinary_bc_control,
        "candidate": candidate,
    }


def _render_prediction_video(
    destination: Path,
    frames: list[dict[str, Any]],
    *,
    fps: int,
) -> None:
    """Render labelled recorded observations and actual offline action values."""

    import cv2
    import numpy as np

    if not frames or fps < 1:
        raise WorkflowArtifactError(
            "evidence video needs recorded frames and a positive FPS"
        )
    width = 960
    action_dimensions = int(frames[0]["target"].shape[-1])
    rows = math.ceil(action_dimensions / 2)
    height = 452 + rows * 92
    temporary = destination.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise WorkflowArtifactError("OpenCV could not open evidence video writer")
    try:
        history: list[dict[str, Any]] = []
        for record in frames:
            history.append(record)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            observation = cv2.resize(record["observation"][:, :, ::-1], (960, 360))
            canvas[:360, :960] = observation
            cv2.putText(
                canvas,
                f"RECORDED {record['video_key'].upper()} OBSERVATION",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                f"held-out episode {record['episode_index']} frame {record['frame_index']} | offline prediction, not a rollout",
                (14, 388),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                canvas,
                "curves: target (white), foundation (orange), ordinary BC (blue), weighted candidate (green); horizon 0",
                (14, 410),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (230, 230, 230),
                1,
            )
            cv2.putText(
                canvas,
                f"outcome={record['outcome']} collection={record['collection_kind']} tags={','.join(record['tags']) or 'none'}",
                (14, 428),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
            )
            progress = (
                "not supplied"
                if record["progress_value"] is None
                else f"{record['progress_value']:.3f} ({record['progress_source']})"
            )
            cv2.putText(
                canvas,
                f"recorded label source={record['label_source'] or 'not declared'} | progress={progress} | train-weight={record['training_weight']:.4g}",
                (14, 446),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (230, 230, 230),
                1,
            )
            recent = history[-40:]
            for dimension in range(action_dimensions):
                column, row = dimension % 2, dimension // 2
                x0, y0 = column * 480 + 12, 470 + row * 92
                x1, y1 = x0 + 450, y0 + 68
                values = [
                    item[series][0, dimension]
                    for item in recent
                    for series in (
                        "target",
                        "baseline",
                        "ordinary_bc_control",
                        "candidate",
                    )
                ]
                low, high = min(values), max(values)
                if math.isclose(low, high):
                    low, high = low - 1.0, high + 1.0
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), 1)
                cv2.putText(
                    canvas,
                    f"action[{dimension}] raw dataset coordinate",
                    (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (220, 220, 220),
                    1,
                )
                for series, color in (
                    ("target", (255, 255, 255)),
                    ("baseline", (0, 165, 255)),
                    ("ordinary_bc_control", (255, 180, 0)),
                    ("candidate", (0, 220, 0)),
                ):
                    points = []
                    for index, item in enumerate(recent):
                        value = float(item[series][0, dimension])
                        x = x0 + round(index * (x1 - x0) / max(1, len(recent) - 1))
                        y = y1 - round((value - low) * (y1 - y0) / (high - low))
                        points.append((x, y))
                    if len(points) > 1:
                        cv2.polylines(
                            canvas,
                            [np.asarray(points, dtype=np.int32)],
                            False,
                            color,
                            1,
                        )
            writer.write(canvas)
    finally:
        writer.release()
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    temporary.unlink(missing_ok=True)


def _validate_video(path: Path, expected_frames: int) -> dict[str, Any]:
    """Independently ffprobe and decode representative MP4 frames."""

    probe = _probe_video(path)
    if (
        probe.get("codec_name") != "h264"
        or int(probe.get("nb_read_frames") or -1) != expected_frames
    ):
        raise WorkflowArtifactError(
            "evidence MP4 codec or decoded frame count differs from declaration"
        )
    sample_indices = sorted({0, expected_frames // 2, expected_frames - 1})
    decoded: list[dict[str, Any]] = []
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise WorkflowArtifactError("OpenCV cannot decode evidence MP4")
        try:
            for index in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    raise WorkflowArtifactError(
                        "evidence MP4 representative frame does not decode"
                    )
                decoded.append(
                    {
                        "frame_index": index,
                        "shape": list(frame.shape),
                        "sha256": sha256_bytes(frame.tobytes()),
                        "decoder": "opencv",
                    }
                )
        finally:
            capture.release()
    except ModuleNotFoundError:
        import av

        pending = set(sample_indices)
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index not in pending:
                    continue
                decoded_frame = frame.to_ndarray(format="bgr24")
                if decoded_frame.size == 0:
                    raise WorkflowArtifactError(
                        "PyAV decoded an empty evidence MP4 frame"
                    )
                decoded.append(
                    {
                        "frame_index": index,
                        "shape": list(decoded_frame.shape),
                        "sha256": sha256_bytes(decoded_frame.tobytes()),
                        "decoder": "pyav",
                    }
                )
                pending.remove(index)
                if not pending:
                    break
        if pending:
            raise WorkflowArtifactError(
                "PyAV cannot decode representative evidence frames"
            )
    return {
        "ffprobe": probe,
        "decoded_frames": decoded,
        "mp4_sha256": sha256_file(path),
    }


def _standalone_html(
    evaluation: Mapping[str, Any],
    video: Path,
    gallery_root: Path | None = None,
    gallery: Mapping[str, Any] | None = None,
) -> str:
    """Create a self-contained offline report with factual labelled media only."""

    encoded_video = base64.b64encode(video.read_bytes()).decode("ascii")
    evidence = html.escape(json.dumps(evaluation, indent=2, sort_keys=True))
    summary = evaluation["metrics"]
    control_summary = summary.get("ordinary_bc_control")
    control_mae = (
        "not included in this historical evidence"
        if not isinstance(control_summary, Mapping)
        else f"{control_summary['mae']:.6g}"
    )
    input_source = evaluation["source"].get("input", evaluation["source"])
    source_identity = (
        input_source.get("repository")
        or input_source.get("source_revision")
        or "operator-declared source"
    )
    action_contract = evaluation.get("input", {}).get(
        "action_contract",
        {
            "kind": "legacy-source-coordinate",
            "frame": "source-defined",
            "translation_unit": "source-metadata-not-declared",
            "rotation_representation": "source-metadata-not-declared",
            "rotation_unit": "source-metadata-not-declared",
        },
    )
    outcome_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(label),
            value["observations"],
            "not present"
            if value["baseline"] is None
            else f"{value['baseline']['mae']:.6g}",
            "not included"
            if value.get("ordinary_bc_control") is None
            else f"{value['ordinary_bc_control']['mae']:.6g}",
            "not present"
            if value["candidate"] is None
            else f"{value['candidate']['mae']:.6g}",
        )
        for label, value in summary.get("by_observed_outcome", {}).items()
    )
    collection_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(label),
            value["observations"],
            "not present"
            if value["baseline"] is None
            else f"{value['baseline']['mae']:.6g}",
            "not included"
            if value.get("ordinary_bc_control") is None
            else f"{value['ordinary_bc_control']['mae']:.6g}",
            "not present"
            if value["candidate"] is None
            else f"{value['candidate']['mae']:.6g}",
        )
        for label, value in summary.get("by_collection_kind", {}).items()
    )
    gallery_sections: list[str] = []
    for item in (gallery or {}).get("items", []):
        if not isinstance(item, dict):
            raise WorkflowArtifactError("gallery item is not an object")
        relative = Path(str(item.get("media_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkflowArtifactError("gallery HTML received an unsafe media path")
        if gallery_root is None:
            raise WorkflowArtifactError("gallery HTML needs a local gallery directory")
        media = gallery_root / relative
        if not media.is_file():
            raise WorkflowArtifactError("gallery HTML media is missing")
        encoded = base64.b64encode(media.read_bytes()).decode("ascii")
        predictions = bool(item.get("action_predictions_available"))
        gallery_sections.append(
            "<section class=gallery><h3>{}</h3>"
            '<video controls preload="metadata" src="data:video/mp4;base64,{}"></video>'
            "<table><tr><th>Outcome</th><td>{}</td></tr>"
            "<tr><th>Collection / tags</th><td>{} / {}</td></tr>"
            "<tr><th>Source episode/frame/split/group</th><td>{} / {}–{} / {} / {}</td></tr>"
            "<tr><th>Feedback provenance / progress / weight</th><td>{} / {} / {}</td></tr>"
            "<tr><th>Prediction coverage</th><td>{}</td></tr>"
            "<tr><th>Limitations</th><td>{}</td></tr></table></section>".format(
                html.escape(str(item.get("id", "recorded-gallery-item"))),
                encoded,
                html.escape(str(item.get("outcome", "unknown"))),
                html.escape(str(item.get("collection_kind", "unknown"))),
                html.escape(
                    ", ".join(str(tag) for tag in item.get("tags", [])) or "none"
                ),
                html.escape(
                    str(
                        item.get(
                            "source_episode_id",
                            item.get("source_episode_index", "not declared"),
                        )
                    )
                ),
                html.escape(str(item.get("frame_start", "not declared"))),
                html.escape(str(item.get("frame_end", "not declared"))),
                html.escape(str(item.get("split", "not applicable"))),
                html.escape(str(item.get("collection_group", "not declared"))),
                html.escape(str(item.get("label_source", "not declared"))),
                html.escape(str(item.get("progress_summary", "not supplied"))),
                html.escape(str(item.get("training_weight_summary", "not applicable"))),
                "native foundation, matched ordinary-BC control, and feedback-weighted candidate offline action predictions aligned to recorded targets"
                if predictions
                else "recorded reference only; no baseline/candidate action predictions are claimed",
                html.escape(str(item.get("limitation", "recorded observations only"))),
            )
        )
    return f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>GR00T N1.5 offline recorded-observation proof</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}}code,pre{{background:#20252b;padding:.2rem .4rem;overflow:auto}}video{{max-width:100%;border:1px solid #666}}.warn{{border-left:4px solid #f6a623;padding:.6rem 1rem;background:#2b2418}}table{{border-collapse:collapse}}td,th{{padding:.4rem .7rem;border:1px solid #555;text-align:left}}.gallery{{border-top:1px solid #666;margin-top:2rem;padding-top:1rem}}</style>
<h1>GR00T N1.5 offline recorded-observation proof</h1>
<p class=\"warn\"><strong>Scope:</strong> this is offline prediction on held-out recorded observations. The video is not a policy rollout, does not contain predicted robot motion or next states, and provides no robot-success, improvement, deployment, or simulator claim.</p>
<table><tr><th>Input identity</th><td>{html.escape(str(source_identity))} @ {html.escape(str(input_source.get("source_revision", "not declared")))}</td></tr>
<tr><th>Privacy / license</th><td>{html.escape(str(input_source.get("privacy_classification", "legacy")))} / {html.escape(str(input_source.get("license", "source-declared")))}</td></tr>
<tr><th>Model</th><td>{html.escape(str(evaluation["source"]["model_repository"]))}@{html.escape(str(evaluation["source"]["model_revision"]))}; weights are runtime-only and not embedded</td></tr>
<tr><th>Held-out frames</th><td>{evaluation["held_out"]["frames"]}</td></tr><tr><th>Foundation / ordinary-BC / weighted candidate MAE</th><td>{summary["baseline"]["mae"]:.6g} / {control_mae} / {summary["candidate"]["mae"]:.6g} raw dataset action coordinates</td></tr>
<tr><th>Action contract</th><td>{html.escape(str(action_contract.get("kind")))} in {html.escape(str(action_contract.get("frame")))}; translation={html.escape(str(action_contract.get("translation_unit")))}, rotation={html.escape(str(action_contract.get("rotation_representation")))}/{html.escape(str(action_contract.get("rotation_unit")))}</td></tr></table>
<h2>Held-out diagnostics by observed outcome</h2><table><tr><th>Outcome</th><th>Observations</th><th>Foundation MAE</th><th>Ordinary-BC MAE</th><th>Weighted candidate MAE</th></tr>{outcome_rows}</table>
<h2>Held-out diagnostics by collection kind</h2><table><tr><th>Collection kind</th><th>Observations</th><th>Foundation MAE</th><th>Ordinary-BC MAE</th><th>Weighted candidate MAE</th></tr>{collection_rows}</table>
<h2>Recorded observations with aligned offline predictions</h2><video controls preload=\"metadata\" src=\"data:video/mp4;base64,{encoded_video}\"></video>
<p>Each frame labels its held-out source episode/frame. Curves compare horizon-0 recorded action targets with foundation, matched ordinary-BC control, and feedback-weighted candidate native <code>get_action</code> predictions under the same per-frame diffusion seed.</p>
<h2>Labelled positive, negative, and recovery examples</h2>
<p class=\"warn\">The success/failure clips are held-out recorded observations with native offline predictions; the recovery reference is separately recorded source media and explicitly has no candidate prediction. None is a generated next state, robot rollout, or deployment result.</p>
{"".join(gallery_sections)}
<details><summary>Measured receipt</summary><pre>{evidence}</pre></details>
</html>"""


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Compare foundation, matched ordinary BC, and candidate on held-out data."""

    import numpy as np
    from npa.workbench.groot.offline_feedback import OfflineManifest, file_digest
    from npa.workbench.groot.offline_training import (
        _load_model,
        _load_native_runtime,
        checkpoint_digest,
    )

    stage = _new_directory(Path(args.workspace).resolve() / "evaluate")
    dataset, manifest_path, provenance = _download_prepared_inputs(args, stage)
    manifest = OfflineManifest.model_validate_json(manifest_path.read_text())
    store = S3Store()
    recovery_reference_root = stage / "recorded-gallery-reference"
    recovery_reference: dict[str, Any] = {"items": []}
    if args.gallery_reference_uri:
        store.get_directory(args.gallery_reference_uri, recovery_reference_root)
        recovery_reference = _validate_gallery_reference(recovery_reference_root)
        expected_gallery = provenance.get("recorded_gallery_reference")
        actual_gallery_tree = tree_digest(directory_inventory(recovery_reference_root))
        if (
            not isinstance(expected_gallery, dict)
            or expected_gallery.get("tree_sha256") != actual_gallery_tree
        ):
            raise WorkflowArtifactError(
                "prepared recovery gallery differs from provenance-bound artifact"
            )
    external_code = _install_external_runtime_code(args, store, stage)
    candidate = stage / "candidate"
    control = stage / "ordinary-bc-control"
    store.get_directory(args.candidate_uri, candidate)
    store.get_directory(args.control_uri, control)
    training = store.get_json(args.training_receipt_uri)
    if training.get("manifest_sha256") != file_digest(manifest_path):
        raise WorkflowArtifactError("training receipt and evaluation manifest differ")
    base = stage / "base-model"
    _download_base_model(args, base)
    runtime_evidence = _gpu_runtime_evidence()
    groot = Path(args.workspace).resolve() / "runtime" / "Isaac-GR00T"
    runtime = _load_native_runtime(groot)
    config = runtime["load_data_config"](args.data_config)
    native = runtime["LeRobotSingleDataset"](
        dataset_path=str(dataset),
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag=runtime["EmbodimentTag"](args.embodiment_tag),
        video_backend=args.video_backend,
    )
    native.transforms.eval()
    baseline = _load_model(runtime, base, config, manifest.action_horizon).cuda().eval()
    candidate_model = (
        _load_model(runtime, candidate, config, manifest.action_horizon).cuda().eval()
    )
    control_model = (
        _load_model(runtime, control, config, manifest.action_horizon).cuda().eval()
    )
    if training.get("candidate_checkpoint_sha256") != checkpoint_digest(candidate):
        raise WorkflowArtifactError(
            "candidate checkpoint bytes differ from training receipt"
        )
    control_summary = training.get("matched_ordinary_bc_control")
    if not isinstance(control_summary, dict) or (
        control_summary.get("checkpoint_sha256") != checkpoint_digest(control)
    ):
        raise WorkflowArtifactError(
            "ordinary-BC control checkpoint bytes differ from training receipt"
        )
    episodes = {episode.episode_index: episode for episode in manifest.episodes}
    feedback_records = _feedback_by_episode(provenance, manifest)
    native_gallery_episodes = _native_gallery_episodes(
        manifest, require_labels=_custom_adapter_requested(args)
    )
    action_keys = _action_keys(args.action_target_keys)
    eval_steps = [
        (episode, frame)
        for episode, frame in native.all_steps
        if episodes[int(episode)].split == "eval"
    ]
    if not eval_steps:
        raise WorkflowArtifactError("manifest has no held-out native inference steps")
    baseline_actions: list[Any] = []
    control_actions: list[Any] = []
    candidate_actions: list[Any] = []
    targets: list[Any] = []
    masks: list[Any] = []
    identities: list[tuple[int, int]] = []
    clip: list[dict[str, Any]] = []
    gallery_clips: dict[str, list[dict[str, Any]]] = {
        label: [] for label in native_gallery_episodes
    }
    evidence_end = args.evidence_start_frame + args.evidence_frame_count
    for ordinal, (episode, frame) in enumerate(eval_steps):
        episode, frame = int(episode), int(frame)
        raw = native.get_step_data(episode, frame)
        target = _raw_targets(
            raw, horizon=manifest.action_horizon, action_keys=action_keys
        )
        mask = np.arange(manifest.action_horizon) + frame < episodes[episode].length
        common_seed = args.inference_seed + ordinal
        expected_action_dim = int(target.shape[-1])
        baseline_actions.append(
            _predict_action(
                baseline,
                native.transforms,
                raw,
                common_seed,
                expected_action_dim=expected_action_dim,
                action_keys=action_keys,
            )
        )
        control_actions.append(
            _predict_action(
                control_model,
                native.transforms,
                raw,
                common_seed,
                expected_action_dim=expected_action_dim,
                action_keys=action_keys,
            )
        )
        candidate_actions.append(
            _predict_action(
                candidate_model,
                native.transforms,
                raw,
                common_seed,
                expected_action_dim=expected_action_dim,
                action_keys=action_keys,
            )
        )
        targets.append(target)
        masks.append(mask)
        identities.append((episode, frame))
        frame_record: dict[str, Any] | None = None
        is_evidence = (
            episode == args.evidence_episode_index
            and args.evidence_start_frame <= frame < evidence_end
        )
        is_gallery = any(
            episode == gallery_episode
            and args.evidence_start_frame <= frame < evidence_end
            for gallery_episode in native_gallery_episodes.values()
        )
        if is_evidence or is_gallery:
            frame_record = _prediction_gallery_record(
                episode=episode,
                frame=frame,
                raw=raw,
                episode_metadata=episodes[episode],
                feedback=feedback_records[episode],
                target=target,
                baseline=baseline_actions[-1],
                ordinary_bc_control=control_actions[-1],
                candidate=candidate_actions[-1],
                evidence_video_key=args.evidence_video_key,
            )
        if is_evidence and frame_record is not None:
            clip.append(frame_record)
        if frame_record is not None:
            for label, gallery_episode in native_gallery_episodes.items():
                if episode == gallery_episode:
                    gallery_clips[label].append(frame_record)
        if ordinal and ordinal % 100 == 0:
            print(
                f"native held-out action inference {ordinal}/{len(eval_steps)}",
                flush=True,
            )
    if len(clip) != args.evidence_frame_count:
        raise WorkflowArtifactError(
            "configured evidence episode/frame window is not wholly held out"
        )
    for label, frames in gallery_clips.items():
        if len(frames) != args.evidence_frame_count:
            raise WorkflowArtifactError(
                f"held-out {label} gallery clip does not cover the configured frame window"
            )
    baseline_array = np.stack(baseline_actions)
    control_array = np.stack(control_actions)
    candidate_array = np.stack(candidate_actions)
    target_array = np.stack(targets)
    mask_array = np.stack(masks)
    evidence_dir = stage / "evidence"
    evidence_dir.mkdir()
    actions_path = evidence_dir / "heldout-actions.npz"
    np.savez_compressed(
        actions_path,
        baseline=baseline_array,
        ordinary_bc_control=control_array,
        candidate=candidate_array,
        target=target_array,
        action_mask=mask_array,
        episode_index=np.asarray([item[0] for item in identities], dtype=np.int64),
        frame_index=np.asarray([item[1] for item in identities], dtype=np.int64),
        outcome=np.asarray(
            [
                1
                if episodes[episode].success is True
                else 0
                if episodes[episode].success is False
                else -1
                for episode, _frame in identities
            ],
            dtype=np.int8,
        ),
        collection_kind=np.asarray(
            [episodes[episode].collection_kind for episode, _frame in identities]
        ),
    )
    video_path = evidence_dir / "offline-prediction.mp4"
    _render_prediction_video(video_path, clip, fps=args.evidence_video_fps)
    media_validation = _validate_video(video_path, args.evidence_frame_count)
    gallery_dir = evidence_dir / "gallery"
    gallery_dir.mkdir()
    gallery_items: list[dict[str, Any]] = []
    gallery_media_validation: dict[str, Any] = {}
    for label, frames in sorted(gallery_clips.items()):
        gallery_video = gallery_dir / f"heldout-{label}-offline-predictions.mp4"
        _render_prediction_video(gallery_video, frames, fps=args.evidence_video_fps)
        gallery_media_validation[label] = _validate_video(
            gallery_video, args.evidence_frame_count
        )
        first, last = frames[0], frames[-1]
        progress_values = [
            frame["progress_value"]
            for frame in frames
            if frame["progress_value"] is not None
        ]
        gallery_items.append(
            {
                "id": f"heldout-{label}-native-offline-predictions",
                "media_path": gallery_video.name,
                "media_sha256": sha256_file(gallery_video),
                "source_episode_id": first["source_episode_id"],
                "source_episode_index": first["episode_index"],
                "frame_start": first["frame_index"],
                "frame_end": last["frame_index"],
                "recorded_frame_count": args.evidence_frame_count,
                "split": "eval",
                "collection_group": first["collection_group"],
                "outcome": first["outcome"],
                "collection_kind": first["collection_kind"],
                "tags": first["tags"],
                "label_source": first["label_source"],
                "progress_summary": (
                    f"{first['progress_source']}; first={progress_values[0]:.3f}, last={progress_values[-1]:.3f}"
                    if progress_values
                    else "not supplied"
                ),
                "training_weight_summary": "held-out weights are one; training weights remain split-isolated",
                "action_predictions_available": True,
                "limitation": "recorded observations and offline native predictions only; no next-state or robot outcome is measured",
            }
        )
    for item in recovery_reference.get("items", []):
        source_media = recovery_reference_root / str(item["media_path"])
        target_relative = (
            Path("recovery-reference") / Path(str(item["media_path"])).name
        )
        target_media = gallery_dir / target_relative
        target_media.parent.mkdir()
        shutil.copyfile(source_media, target_media)
        gallery_items.append(
            {
                **item,
                "media_path": target_relative.as_posix(),
                "media_sha256": sha256_file(target_media),
                "split": "separate recorded reference; not in candidate train/eval",
                "collection_group": "not supplied by this separate source",
                "progress_summary": "not supplied",
                "training_weight_summary": "not applicable",
            }
        )
    if _custom_adapter_requested(args) and not any(
        item["collection_kind"] == "operator_recovery" for item in gallery_items
    ):
        raise WorkflowArtifactError(
            "custom gallery lacks an observed recovery reference"
        )
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise WorkflowArtifactError("prepared provenance lacks source identity")
    action_contract = provenance.get("action_contract")
    if not isinstance(action_contract, dict):
        action_contract = {
            "kind": "legacy-source-coordinate",
            "frame": "source-defined",
            "translation_unit": "source-metadata-not-declared",
            "rotation_representation": "source-metadata-not-declared",
            "rotation_unit": "source-metadata-not-declared",
            "transform_identity": "legacy-native-min-max",
        }
    outcome_labels = [
        "success"
        if episodes[episode].success is True
        else "failure"
        if episodes[episode].success is False
        else "unknown"
        for episode, _frame in identities
    ]
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "kind": "native-held-out-offline-action-comparison",
        "scope": {
            "recorded_observations": True,
            "native_baseline_get_action": True,
            "native_ordinary_bc_control_get_action": True,
            "native_candidate_get_action": True,
            "closed_loop_rollout": False,
            "robot_evaluation": False,
            "policy_improvement_claim": False,
        },
        "source": {
            "input": source,
            "model_repository": args.model_repository,
            "model_revision": args.model_revision,
            "n1_5_revision": args.groot_revision,
        },
        "input": {
            "manifest_sha256": file_digest(manifest_path),
            "dataset_sha256": manifest.dataset_sha256,
            "video_backend": args.video_backend,
            "action_transform": "native min_max normalization from training-split-only stats; inverse transformed for metrics",
            "action_contract": action_contract,
            "action_target_keys": list(action_keys),
            "evidence_video_key": args.evidence_video_key,
        },
        "held_out": {
            "episodes": sum(episode.split == "eval" for episode in manifest.episodes),
            "frames": len(eval_steps),
            "action_horizon": manifest.action_horizon,
            "action_dimensions": int(target_array.shape[-1]),
            "common_per_frame_diffusion_seed": args.inference_seed,
        },
        "checkpoints": {
            "baseline_sha256": training["base_checkpoint_sha256"],
            "ordinary_bc_control_sha256": control_summary["checkpoint_sha256"],
            "candidate_sha256": training["candidate_checkpoint_sha256"],
            "ordinary_bc_control_reloaded_for_inference": True,
            "candidate_reloaded_for_inference": True,
        },
        "metrics": {
            "baseline": _metrics(baseline_array, target_array, mask_array),
            "ordinary_bc_control": _metrics(control_array, target_array, mask_array),
            "candidate": _metrics(candidate_array, target_array, mask_array),
            "by_observed_outcome": _labelled_metrics(
                baseline_array,
                control_array,
                candidate_array,
                target_array,
                mask_array,
                outcome_labels,
            ),
            "by_collection_kind": _labelled_metrics(
                baseline_array,
                control_array,
                candidate_array,
                target_array,
                mask_array,
                [episodes[episode].collection_kind for episode, _frame in identities],
                (
                    "operator_demonstration",
                    "policy_rollout",
                    "operator_recovery",
                    "recorded_failure",
                    "unknown",
                ),
            ),
        },
        "evidence_clip": {
            "episode_index": args.evidence_episode_index,
            "start_frame": args.evidence_start_frame,
            "frame_count": args.evidence_frame_count,
            "recorded_fps": args.evidence_video_fps,
            "prediction_alignment": "one native foundation, matched ordinary-BC control, and feedback-weighted candidate get_action call per recorded observation; plotted action horizon index 0",
        },
        "runtime": runtime_evidence,
        "custom_code": external_code,
        "prepared_provenance_sha256": sha256_bytes(canonical_json(provenance).encode()),
        "actions_sha256": sha256_file(actions_path),
        "media_validation": media_validation,
        "gallery": {
            "items": gallery_items,
            "native_media_validation": gallery_media_validation,
            "recovery_reference_sha256": recovery_reference.get("gallery_json_sha256"),
        },
    }
    gallery_payload = {
        "schema_version": "npa.groot-recorded-gallery/v1",
        "kind": "labelled-offline-recorded-observation-gallery",
        "items": gallery_items,
        "native_media_validation": gallery_media_validation,
        "recovery_reference": recovery_reference,
    }
    (gallery_dir / "gallery.json").write_text(canonical_json(gallery_payload))
    html_path = evidence_dir / "offline-proof.html"
    html_path.write_text(
        _standalone_html(evaluation, video_path, gallery_dir, gallery_payload),
        encoding="utf-8",
    )
    records = {
        "actions": store.put_file(actions_path, args.actions_uri),
        "video": store.put_file(video_path, args.video_uri),
        "html": store.put_file(html_path, args.html_uri),
        "evaluation": store.put_json(evaluation, args.evaluation_uri),
        "media_validation": store.put_json(media_validation, args.media_validation_uri),
        "gallery": store.put_directory(gallery_dir, args.gallery_uri),
    }
    store.put_json(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "evidence-commit",
            "outputs": records,
            "video_sha256": records["video"]["sha256"],
            "html_sha256": records["html"]["sha256"],
            "gallery_tree_sha256": records["gallery"]["tree_sha256"],
            "candidate_checkpoint_sha256": training["candidate_checkpoint_sha256"],
            "ordinary_bc_control_checkpoint_sha256": control_summary[
                "checkpoint_sha256"
            ],
        },
        args.evidence_commit_uri,
    )
    return {
        "status": "evaluated",
        "held_out_frames": len(eval_steps),
        "baseline_mae": evaluation["metrics"]["baseline"]["mae"],
        "ordinary_bc_control_mae": evaluation["metrics"]["ordinary_bc_control"]["mae"],
        "candidate_mae": evaluation["metrics"]["candidate"]["mae"],
        "video_sha256": records["video"]["sha256"],
        "html_sha256": records["html"]["sha256"],
    }


def _verify_object(
    store: S3Store, uri: str, expected_sha: str, destination: Path
) -> dict[str, Any]:
    store.get_file(uri, destination)
    actual = sha256_file(destination)
    if actual != expected_sha:
        raise WorkflowArtifactError("independent durable readback digest mismatch")
    return {"sha256": actual, "bytes": destination.stat().st_size}


def _verify_directory(
    store: S3Store, uri: str, expected_tree_sha: str, destination: Path
) -> dict[str, Any]:
    """Download a declared prefix and bind every member to its producer tree."""

    store.get_directory(uri, destination)
    inventory = directory_inventory(destination)
    actual_tree_sha = tree_digest(inventory)
    if actual_tree_sha != expected_tree_sha:
        raise WorkflowArtifactError(
            "independent durable directory readback digest mismatch"
        )
    return {
        "tree_sha256": actual_tree_sha,
        "files": len(inventory),
        "bytes": sum(int(record["bytes"]) for record in inventory),
    }


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    """Read committed artifacts back from durable storage and re-decode the MP4."""

    stage = _new_directory(Path(args.workspace).resolve() / "verify")
    store = S3Store()
    prepare = store.get_json(args.prepare_commit_uri)
    training = store.get_json(args.training_commit_uri)
    evidence = store.get_json(args.evidence_commit_uri)
    if (
        prepare.get("kind") != "prepare-commit"
        or training.get("kind") != "training-commit"
    ):
        raise WorkflowArtifactError("workflow stage commits are malformed")
    if evidence.get("kind") != "evidence-commit":
        raise WorkflowArtifactError("evidence commit is malformed")
    output = evidence.get("outputs")
    if not isinstance(output, dict):
        raise WorkflowArtifactError("evidence commit lacks output inventory")
    prepare_outputs = prepare.get("outputs")
    if not isinstance(prepare_outputs, dict):
        raise WorkflowArtifactError("prepare commit lacks output inventory")
    prepared_dataset = _verify_directory(
        store,
        args.prepared_dataset_uri,
        str(prepare_outputs["prepared_dataset"]["tree_sha256"]),
        stage / "prepared-dataset",
    )
    feedback = _verify_object(
        store,
        args.feedback_output_uri,
        str(prepare_outputs["feedback"]["sha256"]),
        stage / "feedback.json",
    )
    manifest = _verify_object(
        store,
        args.manifest_uri,
        str(prepare_outputs["manifest"]["sha256"]),
        stage / "feedback-manifest.json",
    )
    provenance = _verify_object(
        store,
        args.prepare_receipt_uri,
        str(prepare_outputs["provenance"]["sha256"]),
        stage / "provenance.json",
    )
    prepared_gallery: dict[str, Any] | None = None
    if "recorded_gallery_reference" in prepare_outputs:
        prepared_gallery = _verify_directory(
            store,
            args.gallery_reference_uri,
            str(prepare_outputs["recorded_gallery_reference"]["tree_sha256"]),
            stage / "prepared-recorded-gallery-reference",
        )
        _validate_gallery_reference(stage / "prepared-recorded-gallery-reference")
    candidate = _verify_directory(
        store,
        args.candidate_uri,
        str(training["candidate_tree_sha256"]),
        stage / "candidate",
    )
    control = _verify_directory(
        store,
        args.control_uri,
        str(training["control_tree_sha256"]),
        stage / "ordinary-bc-control",
    )
    training_record = _verify_object(
        store,
        args.training_receipt_uri,
        str(training["training_receipt_sha256"]),
        stage / "offline-training.json",
    )
    actions = _verify_object(
        store,
        args.actions_uri,
        str(output["actions"]["sha256"]),
        stage / "heldout-actions.npz",
    )
    video = _verify_object(
        store,
        args.video_uri,
        str(output["video"]["sha256"]),
        stage / "offline-prediction.mp4",
    )
    html_record = _verify_object(
        store,
        args.html_uri,
        str(output["html"]["sha256"]),
        stage / "offline-proof.html",
    )
    evaluation_record = _verify_object(
        store,
        args.evaluation_uri,
        str(output["evaluation"]["sha256"]),
        stage / "evaluation.json",
    )
    media_record = _verify_object(
        store,
        args.media_validation_uri,
        str(output["media_validation"]["sha256"]),
        stage / "media-validation.json",
    )
    gallery = _verify_directory(
        store,
        args.gallery_uri,
        str(output["gallery"]["tree_sha256"]),
        stage / "labelled-gallery",
    )
    evaluation = json.loads((stage / "evaluation.json").read_text())
    import numpy as np

    action_arrays = np.load(stage / "heldout-actions.npz")
    expected_arrays = {
        "baseline",
        "ordinary_bc_control",
        "candidate",
        "target",
        "action_mask",
        "episode_index",
        "frame_index",
    }
    if not expected_arrays.issubset(action_arrays.files):
        raise WorkflowArtifactError("read-back held-out actions are incomplete")
    if not (
        action_arrays["baseline"].shape
        == action_arrays["ordinary_bc_control"].shape
        == action_arrays["candidate"].shape
    ):
        raise WorkflowArtifactError(
            "read-back foundation, ordinary-BC, and candidate shapes differ"
        )
    media = _validate_video(
        stage / "offline-prediction.mp4",
        int(evaluation["evidence_clip"]["frame_count"]),
    )
    gallery_payload_path = stage / "labelled-gallery" / "gallery.json"
    if not gallery_payload_path.is_file():
        raise WorkflowArtifactError("read-back labelled gallery lacks gallery.json")
    gallery_payload = json.loads(gallery_payload_path.read_text())
    if gallery_payload.get(
        "schema_version"
    ) != "npa.groot-recorded-gallery/v1" or not isinstance(
        gallery_payload.get("items"), list
    ):
        raise WorkflowArtifactError("read-back labelled gallery has an invalid schema")
    seen: set[tuple[str, bool]] = set()
    gallery_decode: dict[str, Any] = {}
    for item in gallery_payload["items"]:
        if not isinstance(item, dict):
            raise WorkflowArtifactError("read-back gallery item is malformed")
        relative = Path(str(item.get("media_path", "")))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise WorkflowArtifactError("read-back gallery media path is unsafe")
        media_path = stage / "labelled-gallery" / relative
        if not media_path.is_file() or sha256_file(media_path) != item.get(
            "media_sha256"
        ):
            raise WorkflowArtifactError("read-back gallery media digest differs")
        frame_count = int(item.get("recorded_frame_count", 0))
        gallery_decode[str(item.get("id", relative.as_posix()))] = _validate_video(
            media_path, frame_count
        )
        seen.add(
            (str(item.get("outcome")), bool(item.get("action_predictions_available")))
        )
    if not {("success", True), ("failure", True), ("unknown", False)}.issubset(seen):
        raise WorkflowArtifactError(
            "read-back gallery lacks labelled native positive/negative and recovery coverage"
        )
    source = (stage / "offline-proof.html").read_text(encoding="utf-8")
    if (
        "data:video/mp4;base64," not in source
        or "http://" in source
        or "https://" in source
    ):
        raise WorkflowArtifactError(
            "standalone HTML contains no embedded video or has a remote dependency"
        )
    if (
        evaluation["checkpoints"]["candidate_sha256"]
        != training["candidate_checkpoint_sha256"]
    ):
        raise WorkflowArtifactError(
            "evaluation candidate identity differs from training commit"
        )
    if evaluation["checkpoints"].get("ordinary_bc_control_sha256") != training.get(
        "control_checkpoint_sha256"
    ):
        raise WorkflowArtifactError(
            "evaluation ordinary-BC control identity differs from training commit"
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "workflow-independent-readback",
        "status": "verified",
        "commits": {
            "prepare_manifest_sha256": prepare["manifest_sha256"],
            "training_receipt_sha256": training["training_receipt_sha256"],
            "candidate_checkpoint_sha256": training["candidate_checkpoint_sha256"],
            "ordinary_bc_control_checkpoint_sha256": training[
                "control_checkpoint_sha256"
            ],
            "evidence_video_sha256": evidence["video_sha256"],
            "evidence_html_sha256": evidence["html_sha256"],
            "evidence_gallery_tree_sha256": evidence["gallery_tree_sha256"],
        },
        "readback": {
            "prepared_dataset": prepared_dataset,
            "feedback": feedback,
            "manifest": manifest,
            "provenance": provenance,
            "candidate": candidate,
            "ordinary_bc_control": control,
            "training_receipt": training_record,
            "actions": actions,
            "video": video,
            "html": html_record,
            "evaluation": evaluation_record,
            "media_validation": media_record,
            "media": media,
            "prepared_gallery_reference": prepared_gallery,
            "gallery": gallery,
            "gallery_media": gallery_decode,
        },
        "limitations": [
            "Recorded observation video is an offline input, not a generated rollout.",
            "No simulator or physical robot evaluation was performed.",
            "Offline action error does not establish real-world policy improvement.",
        ],
    }
    receipt_record = store.put_json(receipt, args.verify_receipt_uri)
    _verify_object(
        store,
        args.verify_receipt_uri,
        str(receipt_record["sha256"]),
        stage / "workflow-readback.json",
    )
    return {
        "status": "verified",
        "video_sha256": video["sha256"],
        "html_sha256": html_record["sha256"],
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset-repository", default=DATASET_REPOSITORY_DEFAULT)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION_DEFAULT)
    parser.add_argument("--dataset-archive-uri", default="")
    parser.add_argument("--input-provenance-uri", default="")
    parser.add_argument("--action-contract-uri", default="")
    parser.add_argument("--external-code-uri", default="")
    parser.add_argument("--external-code-sha256", default="")
    parser.add_argument("--external-code-package", default="")
    parser.add_argument("--adapter-callable", default="")
    parser.add_argument("--feedback-callable", default="")
    parser.add_argument("--adapter-config-uri", default="")
    parser.add_argument("--model-repository", default=MODEL_REPOSITORY_DEFAULT)
    parser.add_argument("--model-revision", default=MODEL_REVISION_DEFAULT)
    parser.add_argument("--groot-repository", default=GROOT_REPOSITORY_DEFAULT)
    parser.add_argument("--groot-revision", default=GROOT_REVISION_DEFAULT)
    parser.add_argument("--data-config", default="so100_dualcam")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--video-backend",
        choices=("decord", "torchcodec", "torchvision_av"),
        default="decord",
    )
    parser.add_argument("--derivative-codec", choices=("h264",), default="h264")
    parser.add_argument("--prepared-dataset-uri", required=True)
    parser.add_argument(
        "--prepared-input-mode",
        choices=("build-or-reuse", "require-existing"),
        default="build-or-reuse",
    )
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--prepare-receipt-uri", required=True)
    parser.add_argument("--prepare-commit-uri", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "verify"))
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    _add_common_arguments(parser)
    parser.add_argument(
        "--feedback-mode",
        choices=("uniform", "outcome", "progress", "hybrid"),
        default="uniform",
    )
    parser.add_argument("--feedback-uri", default="")
    parser.add_argument("--feedback-output-uri", required=True)
    parser.add_argument("--gallery-reference-uri", default="")
    parser.add_argument("--feedback-temperature", type=float, default=1.0)
    parser.add_argument(
        "--grouping", choices=("episode", "collection"), default="episode"
    )
    parser.add_argument(
        "--split-strategy",
        choices=("contiguous-tail", "adapter"),
        default="contiguous-tail",
    )
    parser.add_argument("--train-episode-count", type=int, default=64)
    parser.add_argument("--eval-episode-count", type=int, default=16)
    parser.add_argument("--codec-workers", type=int, default=4)
    parser.add_argument("--groot-modality-path", type=Path, default=Path())
    parser.add_argument("--complete-train-passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--checkpoint-root-uri", required=True)
    parser.add_argument("--candidate-uri", required=True)
    parser.add_argument("--control-uri", default="")
    parser.add_argument("--training-receipt-uri", required=True)
    parser.add_argument("--training-commit-uri", required=True)
    parser.add_argument("--inference-seed", type=int, default=1701)
    parser.add_argument("--evidence-episode-index", type=int, default=64)
    parser.add_argument("--evidence-start-frame", type=int, default=0)
    parser.add_argument("--evidence-frame-count", type=int, default=120)
    parser.add_argument("--evidence-video-fps", type=int, default=30)
    parser.add_argument("--evidence-video-key", default="video.wrist")
    parser.add_argument(
        "--action-target-keys", default="action.single_arm,action.gripper"
    )
    parser.add_argument("--actions-uri", required=True)
    parser.add_argument("--video-uri", required=True)
    parser.add_argument("--html-uri", required=True)
    parser.add_argument("--evaluation-uri", required=True)
    parser.add_argument("--media-validation-uri", required=True)
    parser.add_argument("--gallery-uri", required=True)
    parser.add_argument("--evidence-commit-uri", required=True)
    parser.add_argument("--verify-receipt-uri", required=True)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.workspace.is_absolute():
        raise ValueError("workspace must be an absolute worker path")
    if args.action_horizon < 1 or args.batch_size < 1 or args.complete_train_passes < 1:
        raise ValueError(
            "action_horizon, batch_size, and complete_train_passes must be positive"
        )
    if (
        args.checkpoint_interval_steps < 1
        or args.evidence_frame_count < 1
        or args.evidence_video_fps < 1
    ):
        raise ValueError("checkpoint and evidence numeric settings must be positive")
    if args.evidence_start_frame < 0 or args.evidence_episode_index < 0:
        raise ValueError("evidence episode/frame selection cannot be negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(args.feedback_temperature) or args.feedback_temperature <= 0:
        raise ValueError("feedback_temperature must be positive and finite")
    for key, value in vars(args).items():
        if key.endswith("_uri") and value:
            S3Uri.parse(value)
    if args.command in {"train", "evaluate", "verify"} and not args.control_uri:
        raise ValueError(
            "train/evaluate/verify require control_uri for matched ordinary BC"
        )
    custom = _custom_adapter_requested(args)
    if custom and args.feedback_uri:
        raise ValueError(
            "custom feedback must be produced by the declared feedback callable, not feedback_uri"
        )
    if custom and not args.gallery_reference_uri:
        raise ValueError("custom adapter requires gallery_reference_uri")
    if custom and (not args.dataset_repository or not args.dataset_revision):
        raise ValueError(
            "custom adapter requires explicit dataset_repository and dataset_revision"
        )
    if (
        not custom
        and args.feedback_mode in {"outcome", "progress", "hybrid"}
        and not args.feedback_uri
    ):
        raise ValueError("outcome/progress mode requires worker-readable feedback_uri")
    if (
        args.feedback_mode == "uniform"
        and args.feedback_uri
        and not args.feedback_uri.startswith("s3://")
    ):
        raise ValueError("external uniform feedback must be a worker-readable s3 URI")
    keys = [key.strip() for key in args.action_target_keys.split(",") if key.strip()]
    if not keys or any(not key.startswith("action.") for key in keys):
        raise ValueError("action_target_keys must be one or more action.* keys")
    if not args.evidence_video_key.startswith("video."):
        raise ValueError("evidence_video_key must be a video.* key")
    if custom and (
        len(args.external_code_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.external_code_sha256
        )
    ):
        raise ValueError("external_code_sha256 must be a lowercase SHA-256 digest")
    if custom:
        module_name, separator, class_name = args.data_config.partition(":")
        package_root = args.external_code_package.split(".", maxsplit=1)[0]
        if (
            not separator
            or not module_name
            or not class_name
            or (
                module_name != package_root
                and not module_name.startswith(package_root + ".")
            )
        ):
            raise ValueError(
                "custom data_config must be module:Class from the declared external_code_package"
            )


def main(argv: list[str] | None = None) -> None:
    """Execute one declarative Workbench stage and print its compact receipt."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(raw_argv)
    try:
        _validate_arguments(args)
        if not args.native:
            _run_native(raw_argv, args)
            return
        operation = {
            "prepare": _prepare,
            "train": _train,
            "evaluate": _evaluate,
            "verify": _verify,
        }[args.command]
        print(json.dumps(operation(args), sort_keys=True))
    except (
        OSError,
        ValueError,
        WorkflowArtifactError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
