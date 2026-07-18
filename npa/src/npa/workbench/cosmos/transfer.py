"""Real Cosmos-Transfer2.5 inference runner.

Shared by the sim2real augment stage, the Cosmos synthetic fan-out workflow, and
the ``npa workbench cosmos2 transfer`` CLI so they run the actual world-transfer
model (video-to-video) instead of writing descriptor stubs.

The transfer runtime lives in the ``npa-cosmos2-transfer`` image at
``/opt/cosmos/cosmos-transfer2.5`` (Python 3.10 + torch cu128 + flash-attn in its
own ``.venv``). This module shells out to that venv's ``examples/inference.py`` so
it stays import-safe on the default interpreter (no torch/cuda import here).

Callers that run outside the transfer image (unit tests, CPU hosts) should guard
on :func:`cosmos_transfer_available` and fall back to their descriptor path.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_REPO = "/opt/cosmos/cosmos-transfer2.5"
# Proven, self-contained control example bundled in the transfer repo. Produces a
# real transferred video; used when the workflow does not supply its own spec.
DEFAULT_SPEC = "assets/robot_example/depth/robot_depth_spec.json"


def cosmos_transfer_repo() -> Path:
    return Path(os.environ.get("COSMOS_TRANSFER_REPO", DEFAULT_REPO))


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def _venv_has_torch(py: Path) -> bool:
    if not py.exists():
        return False
    proc = subprocess.run(
        [str(py), "-c", "import torch, flash_attn"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def cosmos_transfer_available() -> bool:
    """True when the real Cosmos-Transfer2.5 runtime is present and runnable.

    Either the inference venv already has torch+flash-attn, or the repo + uv are
    available so :func:`ensure_env` can build the venv on demand.
    """

    repo = cosmos_transfer_repo()
    if not (repo / "examples" / "inference.py").is_file():
        return False
    if _venv_has_torch(_venv_python(repo)):
        return True
    return (repo / "pyproject.toml").is_file() and shutil.which("uv") is not None


def ensure_env(repo: Path) -> Path:
    """Return the inference venv python, building it (py3.10 + cu128) if absent."""

    py = _venv_python(repo)
    if _venv_has_torch(py):
        return py
    # The image pins .python-version=3.13, but the flash-attn wheel is cp310-only.
    (repo / ".python-version").write_text("3.10\n", encoding="utf-8")
    subprocess.run(["uv", "python", "install", "3.10"], cwd=repo, check=True)
    subprocess.run(
        ["uv", "sync", "--extra=cu128", "--python", "3.10"], cwd=repo, check=True
    )
    if not _venv_has_torch(py):
        raise RuntimeError("cosmos-transfer2.5 inference env build did not yield torch+flash_attn")
    return py


def run_cosmos_transfer(
    *,
    run_id: str = "",
    spec: str | None = None,
    out_subdir: str | None = None,
    hf_home: str | None = None,
) -> dict[str, Any]:
    """Run a real Cosmos-Transfer2.5 inference; return the generated video + metadata.

    ``spec`` is a controlnet_spec path relative to the transfer repo (defaults to
    the env override ``COSMOS_TRANSFER_SPEC`` or the bundled depth example).
    """

    repo = cosmos_transfer_repo()
    py = ensure_env(repo)
    spec = spec or os.environ.get("COSMOS_TRANSFER_SPEC", DEFAULT_SPEC)
    out = out_subdir or f"outputs/{run_id or 'transfer'}"
    out_abs = repo / out
    if out_abs.exists():
        shutil.rmtree(out_abs)

    env = dict(os.environ)
    env["HF_HOME"] = hf_home or os.environ.get("HF_HOME", "/opt/cosmos-data/hf_cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(
        [str(py), "examples/inference.py", "-i", spec, "-o", out],
        cwd=repo,
        env=env,
        check=True,
    )

    videos = [
        f
        for f in glob.glob(str(out_abs / "**" / "*.mp4"), recursive=True)
        if "control" not in Path(f).name
    ]
    big = sorted(
        (f for f in videos if os.path.getsize(f) > 100_000),
        key=os.path.getsize,
        reverse=True,
    )
    if not big:
        raise RuntimeError(f"cosmos-transfer2.5 produced no output video in {out_abs}")
    control = [f for f in glob.glob(str(out_abs / "**" / "*.mp4"), recursive=True) if "control" in Path(f).name]
    return {
        "video_path": big[0],
        "video_bytes": os.path.getsize(big[0]),
        "control_path": control[0] if control else "",
        "out_dir": str(out_abs),
        "spec": spec,
        "repo": str(repo),
    }


def extract_frames(video_path: str, dest_dir: Path, *, max_frames: int = 8) -> list[Path]:
    """Extract up to ``max_frames`` evenly-spaced PNG frames from ``video_path``.

    Runs in the transfer venv (which ships PyAV); best-effort — returns [] on any
    failure so callers can still publish the video + manifest.
    """

    repo = cosmos_transfer_repo()
    py = _venv_python(repo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    script = (
        "import av, sys\n"
        "from pathlib import Path\n"
        "vp, dest, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])\n"
        "with av.open(vp) as c:\n"
        "    frames = [f for f in c.decode(video=0)]\n"
        "step = max(1, len(frames) // n) if frames else 1\n"
        "sel = frames[::step][:n]\n"
        "for i, fr in enumerate(sel):\n"
        "    fr.to_image().save(str(dest / f'frame-{i:05d}.png'))\n"
        "print(len(sel))\n"
    )
    try:
        subprocess.run(
            [str(py), "-c", script, video_path, str(dest_dir), str(max_frames)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001 - frame extraction is best-effort
        return []
    return sorted(dest_dir.glob("frame-*.png"))


def publish_transfer_to_s3(
    transfer: dict[str, Any],
    output_uri: str,
    *,
    run_id: str = "",
    max_frames: int = 8,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Upload a real Cosmos-Transfer2.5 result (video + frames + index) to S3.

    ``transfer`` is the dict returned by :func:`run_cosmos_transfer`. Frames are
    extracted so downstream stages (pseudo-label, grade, visualize) have real
    augmented images to consume. Returns the published-artifact summary.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    base = output_uri if output_uri.endswith("/") else output_uri + "/"
    video_uri = f"{base}augmented_video.mp4"
    client.upload_file(transfer["video_path"], video_uri)

    frames = extract_frames(transfer["video_path"], Path("/tmp/npa-cosmos-frames"), max_frames=max_frames)
    frame_index: list[dict[str, str]] = []
    for i, frame_path in enumerate(frames):
        key = f"frame-{i:05d}.png"
        client.upload_file(str(frame_path), f"{base}frames/{key}")
        frame_index.append({"frame_id": f"frame-{i:05d}", "uri": f"{base}frames/{key}"})

    import json as _json
    import tempfile as _tempfile

    meta = {
        "schema": "npa.cosmos2.transfer.v1",
        "mode": "cosmos_transfer2.5",
        "status": "executed",
        "run_id": run_id,
        "augmented_video_uri": video_uri,
        "frame_count": len(frame_index),
        "frames": frame_index,
        "control_spec": transfer.get("spec", ""),
        "video_bytes": transfer.get("video_bytes", 0),
    }
    with _tempfile.TemporaryDirectory(prefix="npa-cosmos-pub-") as tmp:
        mp = Path(tmp) / "manifest.json"
        mp.write_text(_json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        client.upload_file(str(mp), f"{base}manifest.json")
    return meta


__all__ = [
    "cosmos_transfer_available",
    "cosmos_transfer_repo",
    "ensure_env",
    "extract_frames",
    "publish_transfer_to_s3",
    "run_cosmos_transfer",
]
