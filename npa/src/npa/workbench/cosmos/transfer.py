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


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".webp"}
_PERTURBATIONS = ("lighting", "contrast", "color", "blur")


def _apply_perturbation(image: Any, perturbation: str, *, seed: int) -> Any:
    """Apply one deterministic, real image transform (a perturbation ControlNet
    would drive in the full model; here a genuine PIL transform, not a no-op)."""

    import random

    from PIL import ImageEnhance, ImageFilter

    rng = random.Random(seed)
    if perturbation == "lighting":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.55, 1.6))
    if perturbation == "contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.6, 1.7))
    if perturbation == "color":
        return ImageEnhance.Color(image).enhance(rng.uniform(0.3, 1.9))
    if perturbation == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 2.2)))
    return image


def _collect_source_images(src_dir: Path, max_inputs: int) -> list[Path]:
    return sorted(
        (p for p in src_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )[:max_inputs]


def reference_augment_frames(
    input_uri: str,
    output_uri: str,
    *,
    run_id: str = "",
    variants_per_frame: int = 2,
    max_inputs: int = 8,
) -> dict[str, Any]:
    """Produce real augmented image frames without the heavy Cosmos model.

    Downloads the source frames from ``input_uri``, applies genuine per-frame PIL
    augmentations (lighting / contrast / color / blur), and writes/uploads the
    augmented PNGs to ``output_uri`` so downstream stages (e.g. VLM critique) get
    real image frames instead of a descriptor stub. Used when the
    Cosmos-Transfer2.5 runtime image is not present; ``--execute`` runs the real
    model instead.

    ``s3://`` URIs are read/written via :class:`StorageClient`; any other value is
    treated as a local directory (keeps the function unit-testable without S3).
    """

    import json
    import tempfile

    from PIL import Image

    def _is_s3(uri: str) -> bool:
        return uri.strip().startswith("s3://")

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(tmp) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        storage = None
        if _is_s3(input_uri) or _is_s3(output_uri):
            from npa.clients.storage import StorageClient

            storage = StorageClient.from_environment()

        if _is_s3(input_uri):
            storage.download_directory(input_uri, str(src_dir))
        else:
            local_src = Path(input_uri.replace("local://", "").replace("file://", ""))
            if local_src.is_dir():
                for item in local_src.rglob("*"):
                    if item.is_file():
                        dest = src_dir / item.relative_to(local_src)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)

        sources = _collect_source_images(src_dir, max_inputs)
        if not sources:
            raise RuntimeError(
                f"cosmos2 transfer: no source images found under {input_uri!r}; "
                "expected at least one .png/.jpg frame to augment."
            )

        index: list[dict[str, Any]] = []
        frame_no = 0
        for src in sources:
            base = Image.open(src).convert("RGB")
            for variant in range(max(1, variants_per_frame)):
                perturbation = _PERTURBATIONS[frame_no % len(_PERTURBATIONS)]
                augmented = _apply_perturbation(base, perturbation, seed=frame_no)
                name = f"frame-{frame_no:05d}.png"
                augmented.save(out_dir / name)
                index.append(
                    {
                        "frame_id": f"frame-{frame_no:05d}",
                        "perturbation": perturbation,
                        "source": src.name,
                        "variant": variant,
                    }
                )
                frame_no += 1

        (out_dir / "index.json").write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.augmented_frames.v1",
                    "run_id": run_id,
                    "frame_count": frame_no,
                    "frames": index,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        if _is_s3(output_uri):
            storage.upload_directory(str(out_dir), output_uri)
            frames_uri = output_uri
        else:
            dest_dir = Path(output_uri.replace("local://", "").replace("file://", ""))
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in out_dir.iterdir():
                shutil.copy2(item, dest_dir / item.name)
            frames_uri = str(dest_dir)

    return {
        "augmented_frames_uri": frames_uri,
        "frame_count": frame_no,
        "source_frame_count": len(sources),
    }


__all__ = [
    "cosmos_transfer_available",
    "cosmos_transfer_repo",
    "ensure_env",
    "extract_frames",
    "reference_augment_frames",
    "run_cosmos_transfer",
]
