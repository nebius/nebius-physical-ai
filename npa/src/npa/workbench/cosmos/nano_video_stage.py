"""Single CPU job prestaging the immutable BF16 video checkpoint to shared storage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL_REVISION = "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"
MODEL_NAME = "nvidia/Cosmos3-Nano"
# The video-only pipeline explicitly disables the unused FP32 sound tokenizer.
VIDEO_FILES = [
    "*.json",
    "*.txt",
    "*.jinja",
    "*.model",
    "README.md",
    "text_tokenizer/*",
    "transformer/*",
    "vae/*",
    "scheduler/*",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_dtypes(path: Path) -> Counter:
    """Inspect the safetensors header without loading any tensors into RAM/GPU."""
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors header")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if header_size > path.stat().st_size - 8:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    return Counter(
        item["dtype"] for name, item in header.items() if name != "__metadata__"
    )


def checkpoint_manifest(directory: Path) -> dict:
    counts: Counter = Counter()
    files = []
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or ".cache" in path.relative_to(directory).parts
            or path.name == "READY.json"
        ):
            continue
        if path.is_symlink():
            raise ValueError("Shared checkpoint must contain materialized files")
        if path.suffix == ".safetensors":
            counts.update(tensor_dtypes(path))
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    if not counts["BF16"] or any(
        dtype.startswith("F") and dtype != "BF16" for dtype in counts
    ):
        raise ValueError("Checkpoint contains unsupported floating-point precision")
    if not (
        directory / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    ).is_file():
        raise ValueError("Transformer shard index is missing")
    index = json.loads(
        (
            directory / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        ).read_text()
    )
    for shard in set(index["weight_map"].values()):
        if (
            Path(shard).name != shard
            or not (directory / "transformer" / shard).is_file()
        ):
            raise ValueError("Transformer shard is missing or unsafe")
    return {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "precision": "BF16",
        "tensor_count": sum(counts.values()),
        "tensor_dtypes": dict(counts),
        "files": files,
        "sound_generation": False,
    }


def stage_weights(model_path: Path) -> dict:
    """Download once under a shared lock, verify bytes, and atomically mark ready."""
    from huggingface_hub import HfApi, snapshot_download

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with (model_path.parent / ".cosmos3-nano-stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready_path = model_path / "READY.json"
        if ready_path.is_file():
            ready = json.loads(ready_path.read_text())
            actual = checkpoint_manifest(model_path)
            if any(ready.get(key) != value for key, value in actual.items()):
                raise ValueError(
                    "Existing shared checkpoint no longer matches its immutable manifest"
                )
            print(
                json.dumps(
                    {
                        "status": "already_staged",
                        "revision": MODEL_REVISION,
                        "tensor_count": ready["tensor_count"],
                    }
                )
            )
            return ready
        partial = model_path.with_name(f".{model_path.name}.staging")
        snapshot_download(
            repo_id=MODEL_NAME,
            revision=MODEL_REVISION,
            local_dir=str(partial),
            allow_patterns=VIDEO_FILES,
        )
        manifest = checkpoint_manifest(partial)
        info = HfApi().model_info(
            MODEL_NAME, revision=MODEL_REVISION, files_metadata=True
        )
        if info.sha != MODEL_REVISION:
            raise ValueError("Model metadata did not resolve the pinned revision")
        remote_hashes = {
            item.rfilename: item.lfs.sha256
            for item in info.siblings
            if item.lfs is not None
        }
        for item in manifest["files"]:
            if (
                item["path"].endswith(".safetensors")
                and remote_hashes.get(item["path"]) != item["sha256"]
            ):
                raise ValueError(
                    "Checkpoint tensor bytes differ from the pinned vendor LFS hash"
                )
        manifest["staged_at"] = datetime.now(timezone.utc).isoformat()
        if model_path.exists():
            raise ValueError(
                "An unverified checkpoint already exists at the final shared path"
            )
        # Consumers can only start after the verified directory and marker exist.
        (partial / "READY.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        partial.rename(model_path)
        print(
            json.dumps(
                {
                    "status": "staged",
                    "revision": MODEL_REVISION,
                    "tensor_count": manifest["tensor_count"],
                }
            )
        )
        return manifest


if __name__ == "__main__":
    stage_weights(Path(os.environ["NPA_COSMOS3_MODEL_PATH"]))
