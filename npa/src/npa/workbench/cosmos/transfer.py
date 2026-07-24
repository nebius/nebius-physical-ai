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

# Control modalities Cosmos Transfer 2.5 computes ON-THE-FLY from the input
# ``video_path`` (Canny edge / bilateral blur), so conditioning on an arbitrary
# input clip needs NO precomputed control asset. depth/seg require a precomputed
# control file, so they are not used for self-contained input-only conditioning.
INPUT_AUTO_CONTROLS = ("edge", "vis")
DEFAULT_INPUT_CONTROL = "edge"
# Neutral photoreal prompt used when the caller conditions on an input clip but
# supplies no appearance prompt of its own.
_DEFAULT_INPUT_PROMPT = (
    "photorealistic, natural lighting, high detail, sharp focus, realistic textures"
)


def _spec_for_input_video(
    repo: Path,
    *,
    input_video: str,
    prompt: str,
    control: str,
    control_weight: float,
    guidance: float,
    name: str,
) -> tuple[str, str]:
    """Write a Cosmos Transfer 2.5 controlnet spec that CONDITIONS ON ``input_video``.

    ``video_path`` is the caller's real input clip; the ``edge``/``vis`` control is
    computed on-the-fly from it (no precomputed control asset), so the output
    preserves the input's structure/motion while ``prompt`` drives a new
    appearance -- i.e. a genuine augmentation of the caller's footage. Returns
    ``(spec_path_relative_to_repo, control_modality)``.
    """
    import json as _json

    modality = str(control or DEFAULT_INPUT_CONTROL).strip().lower()
    if modality not in INPUT_AUTO_CONTROLS:
        modality = DEFAULT_INPUT_CONTROL
    spec = {
        "name": str(name or "npa_input"),
        "prompt": str(prompt or "").strip() or _DEFAULT_INPUT_PROMPT,
        # Absolute path so it resolves regardless of where the spec file lives.
        "video_path": str(Path(input_video).resolve()),
        "guidance": guidance,
        modality: {"control_weight": float(control_weight)},
    }
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name or "input"))
    spec_path = repo / f"_npa_input_spec_{safe}.json"
    spec_path.write_text(_json.dumps(spec, indent=2), encoding="utf-8")
    return str(spec_path.relative_to(repo)), modality


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


def _spec_with_prompt(repo: Path, spec: str, prompt: str) -> str:
    """Write a copy of ``spec`` with its text prompt overridden; return its path.

    Cosmos controlnet specs carry the text prompt that steers appearance. Patching
    it lets the sampled appearance combo actually condition the diffusion (same
    control video / motion, new look) instead of being a decorative label. The
    copy sits next to the original so relative control-asset paths still resolve.
    Best-effort: on any failure we fall back to the original spec.
    """
    import json as _json

    try:
        spec_path = repo / spec
        data = _json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return spec
        data["prompt"] = prompt
        patched = spec_path.with_name("_npa_prompted_" + spec_path.name)
        patched.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        return str(patched.relative_to(repo))
    except Exception:  # noqa: BLE001 - prompt override is best-effort
        return spec


def run_cosmos_transfer(
    *,
    run_id: str = "",
    spec: str | None = None,
    prompt: str | None = None,
    out_subdir: str | None = None,
    hf_home: str | None = None,
    input_video: str | None = None,
    control: str = DEFAULT_INPUT_CONTROL,
    control_weight: float = 1.0,
    guidance: float = 3.0,
) -> dict[str, Any]:
    """Run a real Cosmos-Transfer2.5 inference; return the generated video + metadata.

    ``spec`` is a controlnet_spec path relative to the transfer repo (defaults to
    the env override ``COSMOS_TRANSFER_SPEC`` or the bundled depth example).
    ``prompt`` (or ``COSMOS_TRANSFER_PROMPT``), when set, overrides the spec's text
    prompt so the sampled appearance actually conditions the augmentation.

    When ``input_video`` is provided the transfer is CONDITIONED ON THAT CLIP: a
    controlnet spec is built with ``video_path`` = the input and an ``edge``/``vis``
    control computed on-the-fly, so the output is a real augmentation of the
    caller's footage (new appearance from ``prompt``, same structure/motion).
    When ``input_video`` is absent, behavior is unchanged (bundled DEFAULT_SPEC or
    the caller-supplied ``spec``), preserving the golden eval.
    """

    repo = cosmos_transfer_repo()
    py = ensure_env(repo)
    conditioned_control = ""
    if input_video:
        spec, conditioned_control = _spec_for_input_video(
            repo,
            input_video=input_video,
            prompt=prompt or os.environ.get("COSMOS_TRANSFER_PROMPT", ""),
            control=control,
            control_weight=control_weight,
            guidance=guidance,
            name=run_id or "input",
        )
    else:
        spec = spec or os.environ.get("COSMOS_TRANSFER_SPEC", DEFAULT_SPEC)
        prompt = prompt or os.environ.get("COSMOS_TRANSFER_PROMPT", "")
        if prompt:
            spec = _spec_with_prompt(repo, spec, prompt)
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
    control_videos = [
        f for f in glob.glob(str(out_abs / "**" / "*.mp4"), recursive=True)
        if "control" in Path(f).name
    ]
    return {
        "video_path": big[0],
        "video_bytes": os.path.getsize(big[0]),
        "control_path": control_videos[0] if control_videos else "",
        "out_dir": str(out_abs),
        "spec": spec,
        "repo": str(repo),
        "input_conditioned": bool(input_video),
        "input_video": str(input_video or ""),
        "control": conditioned_control,
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
    variables: dict[str, Any] | None = None,
    clip_name: str = "",
    max_frames: int = 8,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Upload a real Cosmos-Transfer2.5 result to S3 in the per-clip layout that
    ``data_factory_stages.curate`` and ``data_factory_viz.build_run_rrd`` consume.

    Writes, under ``output_uri`` (the ``cosmos_augmented/`` prefix):

        <clip>/augmented_video.mp4
        <clip>/frame-00000.png ...
        <clip>/metadata.json      (variables + mode, for the Rerun label)
        manifest.json             (run-level augment manifest; augment output)

    NOTE: a single ``--execute`` runs one transfer, so this emits one clip dir.
    Multi-variant "multiply" (one clip dir per sampled augmentation) needs one
    inference per combo and is tracked as follow-up.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    base = output_uri if output_uri.endswith("/") else output_uri + "/"
    clip = clip_name or (f"aug-{run_id}" if run_id else "aug0")
    clip_base = f"{base}{clip}/"
    video_uri = f"{clip_base}augmented_video.mp4"
    client.upload_file(transfer["video_path"], video_uri)

    import json as _json
    import tempfile as _tempfile

    # This publish path only runs after the REAL Cosmos Transfer 2.5 model
    # executed on GPU, so record the GPU mode (kept in sync with the provenance
    # classifier in data_factory_provenance.py). When the transfer was
    # conditioned on the caller's input clip, record that provenance so the run
    # view can show the augmentation is genuinely derived from real input.
    input_conditioned = bool(transfer.get("input_conditioned"))
    conditioned_input = Path(str(transfer.get("input_video") or "")).name
    conditioned_control = str(transfer.get("control") or "")

    frame_index: list[dict[str, str]] = []
    with _tempfile.TemporaryDirectory(prefix="npa-cosmos-pub-") as tmp:
        frames = extract_frames(transfer["video_path"], Path(tmp) / "frames", max_frames=max_frames)
        for i, frame_path in enumerate(frames):
            key = f"frame-{i:05d}.png"
            client.upload_file(str(frame_path), f"{clip_base}{key}")
            frame_index.append({"frame_id": f"frame-{i:05d}", "uri": f"{clip_base}{key}"})

        clip_meta = {
            "schema": "npa.cosmos2.transfer.v1",
            "mode": "cosmos_transfer2.5_gpu",
            "clip": clip,
            "variables": variables or {},
            "control_spec": transfer.get("spec", ""),
            "input_conditioned": input_conditioned,
            "conditioned_input": conditioned_input,
            "control": conditioned_control,
        }
        cm = Path(tmp) / "metadata.json"
        cm.write_text(_json.dumps(clip_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        client.upload_file(str(cm), f"{clip_base}metadata.json")

        manifest = {
            "schema": "npa.cosmos2.transfer.v1",
            "mode": "cosmos_transfer2.5_gpu",
            "status": "executed",
            "run_id": run_id,
            "clips": [clip],
            "augmented_video_uri": video_uri,
            "frame_count": len(frame_index),
            "frames": frame_index,
            "control_spec": transfer.get("spec", ""),
            "video_bytes": transfer.get("video_bytes", 0),
            "input_conditioned": input_conditioned,
            "conditioned_input": conditioned_input,
            "control": conditioned_control,
        }
        mp = Path(tmp) / "manifest.json"
        mp.write_text(_json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        client.upload_file(str(mp), f"{base}manifest.json")
    return manifest


__all__ = [
    "cosmos_transfer_available",
    "cosmos_transfer_repo",
    "ensure_env",
    "extract_frames",
    "publish_transfer_to_s3",
    "run_cosmos_transfer",
]
