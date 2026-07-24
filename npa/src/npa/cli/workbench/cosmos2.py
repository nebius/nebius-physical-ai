"""Workbench Cosmos2 commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from npa.workflows.cosmos_split import (
    Cosmos2TransferConfig,
    build_cosmos2_transfer_manifest,
    write_manifest,
)

app = typer.Typer(
    name="cosmos2",
    help="Cosmos2 transfer workflow contracts.",
    no_args_is_help=True,
)


def _first_augmentation(configs_uri: str) -> dict:
    """Read the Config-Gen manifest and return the first sampled combo (or {})."""
    try:
        from npa.workflows.data_factory_stages import _download_json

        uri = configs_uri if configs_uri.endswith(".json") else configs_uri.rstrip("/") + "/manifest.json"
        manifest = _download_json(uri)
        combos = manifest.get("augmentations") or []
        return combos[0] if combos and isinstance(combos[0], dict) else {}
    except Exception:  # noqa: BLE001 - variables are advisory metadata, never fatal
        return {}


_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _materialize_input_clip(src: str) -> str:
    """Resolve a local path or ``s3://`` URI to a local video file to condition on.

    For an ``s3://`` prefix, downloads it and returns the first video found. Returns
    "" when nothing usable is present (caller then falls back to the default,
    bundled-example behavior). Best-effort: never raises on a missing input.
    """
    import glob as _glob
    import tempfile

    s = str(src or "").strip()
    if not s:
        return ""
    if not s.startswith("s3://"):
        return s if Path(s).is_file() else ""
    try:
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment()
        tmp = tempfile.mkdtemp(prefix="npa-cosmos-input-")
        if s.lower().endswith(_VIDEO_EXTS):
            return client.download_path(s, str(Path(tmp) / Path(s).name))
        client.download_directory(s, tmp)
        vids = sorted(
            f for f in _glob.glob(str(Path(tmp) / "**" / "*"), recursive=True)
            if f.lower().endswith(_VIDEO_EXTS) and Path(f).is_file()
        )
        return vids[0] if vids else ""
    except Exception:  # noqa: BLE001 - input conditioning is opt-in; fall back cleanly
        return ""


@app.command("transfer")
def transfer_cmd(
    input_uri: str = typer.Option(..., "--input-uri", help="Input frames, assets, or rollout URI."),
    output_uri: str = typer.Option(..., "--output-uri", help="Output prefix for transferred frames."),
    assets_uri: str = typer.Option("", "--assets-uri", help="Optional sim asset source path."),
    scene_spec_uri: str = typer.Option("", "--scene-spec-uri", help="Optional SceneSpec path."),
    image: str = typer.Option("", "--image", help="BYO Cosmos2 transfer image."),
    run_id: str = typer.Option("", "--run-id", help="Run id carried into the manifest."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Write manifest JSON locally."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help=(
            "Force the real Cosmos-Transfer2.5 model (requires the transfer image/GPU). "
            "Note: when that runtime is already present on the host the real model runs "
            "even without --execute; --execute only makes its absence a hard error "
            "instead of falling back to reference augmentation."
        ),
    ),
    spec: str = typer.Option(
        "", "--spec", help="controlnet_spec path (relative to the transfer repo) for --execute."
    ),
    configs_uri: str = typer.Option(
        "",
        "--configs-uri",
        help="Config-Gen manifest URI; the first sampled augmentation combo is "
        "recorded as the clip's appearance variables (drives the Rerun label).",
    ),
    input_video: str = typer.Option(
        "",
        "--input-video",
        help="Local path or s3:// URI of an input clip to CONDITION the augmentation "
        "on. When set (with --execute), the output is a real augmentation of THIS "
        "clip (edge control computed on-the-fly; prompt drives the new appearance).",
    ),
    condition_on_input: bool = typer.Option(
        False,
        "--condition-on-input",
        help="Condition on the first video under --input-uri (opt-in). Also enabled by "
        "NPA_COSMOS_CONDITION_ON_INPUT=1. Default off preserves the bundled-example path.",
    ),
    control: str = typer.Option(
        "edge",
        "--control",
        help="Control modality for input-conditioning: 'edge' or 'vis' (computed on-the-fly).",
    ),
    control_weight: float = typer.Option(1.0, "--control-weight", help="Control weight for input-conditioning."),
    guidance: float = typer.Option(3.0, "--guidance", help="Classifier-free guidance for input-conditioning."),
) -> None:
    """Build the Cosmos2 transfer stage manifest, then produce real output.

    Mode is chosen by runtime availability, not just the flag: if the
    Cosmos-Transfer2.5 runtime is present (or ``--execute`` is passed) the real
    world-transfer model runs and publishes a video; otherwise a genuine
    reference augmentation writes real augmented image frames. Inspect
    ``output_kind`` in the manifest ("video" vs "frames") to disambiguate.
    """

    payload = build_cosmos2_transfer_manifest(
        Cosmos2TransferConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            assets_uri=assets_uri,
            scene_spec_uri=scene_spec_uri,
            image=image,
            run_id=run_id,
        )
    )
    from npa.workbench.cosmos.transfer import (
        cosmos_transfer_available,
        reference_augment_frames,
        run_cosmos_transfer,
    )

    runtime_available = cosmos_transfer_available()
    if execute and not runtime_available:
        raise typer.BadParameter(
            "--execute needs the cosmos-transfer2.5 runtime "
            "(run inside the npa-cosmos2-transfer image on a GPU)."
        )

    if execute or runtime_available:
        # Real Cosmos-Transfer2.5 world-transfer model.
        #
        # Data Factory context (paidf `transfer_execute` passes --configs-uri, or the
        # caller opts into input-conditioning): the sampled appearance combo drives
        # the prompt, the augment optionally CONDITIONS on the run's real input clip
        # (edge control computed on-the-fly — a genuine augmentation of that footage,
        # not the bundled example), and the result is published in the per-clip layout
        # that data_factory curate / build_run_rrd / provenance consume. Opt-in via
        # --input-video, --condition-on-input, or NPA_COSMOS_CONDITION_ON_INPUT=1.
        #
        # Otherwise (generic `transfer` for sim2real / cosmos-gate / fanout): keep the
        # flat single-video publish + sim2real-engine field convention unchanged.
        condition_requested = bool(
            input_video or condition_on_input or _env_truthy("NPA_COSMOS_CONDITION_ON_INPUT")
        )
        data_factory_mode = bool(configs_uri) or condition_requested
        variables = _first_augmentation(configs_uri) if configs_uri else {}
        local_input = ""
        if condition_requested:
            local_input = _materialize_input_clip(input_video or input_uri)
        # Env fallbacks let a submit tune conditioning without changing the toolRef argv.
        control = (os.environ.get("NPA_COSMOS_CONTROL", "").strip() or control)
        _cw = os.environ.get("NPA_COSMOS_CONTROL_WEIGHT", "").strip()
        _g = os.environ.get("NPA_COSMOS_GUIDANCE", "").strip()
        if _cw:
            control_weight = float(_cw)
        if _g:
            guidance = float(_g)
        transfer = run_cosmos_transfer(
            run_id=run_id,
            spec=spec or None,
            prompt=str(variables.get("prompt") or "") or None,
            input_video=local_input or None,
            control=control,
            control_weight=control_weight,
            guidance=guidance,
        )
        payload["status"] = "executed"
        payload["output_kind"] = "video"
        payload["output_video"] = transfer["video_path"]
        payload["video_bytes"] = transfer["video_bytes"]
        payload["control_spec"] = transfer["spec"]
        payload["prompt"] = str(variables.get("prompt") or "")
        payload["input_conditioned"] = bool(local_input)
        if local_input:
            payload["input_video"] = local_input
            payload["control"] = transfer.get("control", control)
        if output_uri.strip().startswith("s3://"):
            if data_factory_mode:
                # Per-clip layout consumed by data_factory curate / build_run_rrd /
                # provenance (cosmos_augmented/<clip>/{augmented_video.mp4, frame-*,
                # metadata.json} + run-level manifest.json).
                from npa.workbench.cosmos.transfer import publish_transfer_to_s3

                published = publish_transfer_to_s3(
                    transfer, output_uri, run_id=run_id, variables=variables
                )
                payload["mode"] = "cosmos_transfer2.5_gpu" if local_input else "cosmos_transfer2.5"
                payload["augmented_video_uri"] = published["augmented_video_uri"]
                payload["frame_count"] = published["frame_count"]
                payload["augmentation_variables"] = variables
                # attribute-verify reads --input-path {{augmented_frames_uri}} (the prefix).
                payload["augmented_frames_uri"] = output_uri
            else:
                # Generic single-video publish + sim2real-engine field convention.
                from npa.clients.storage import StorageClient

                output_video = StorageClient.from_environment().upload_file(
                    transfer["video_path"], output_uri
                )
                payload["mode"] = "cosmos_transfer2.5"
                payload["output_video"] = output_video
                payload["augmented_video_uri"] = output_video
                payload["augmented_frames_uri"] = output_uri
        else:
            payload["mode"] = "cosmos_transfer2.5_gpu" if local_input else "cosmos_transfer2.5"
            payload["augmented_video_uri"] = transfer["video_path"]
            payload["augmented_frames_uri"] = output_uri
    else:
        # No heavy model runtime: run a genuine reference augmentation that
        # writes real augmented image frames to output_uri (not a descriptor stub).
        augment = reference_augment_frames(input_uri, output_uri, run_id=run_id)
        payload["status"] = "executed_reference"
        payload["mode"] = "reference_augment"
        payload["output_kind"] = "frames"
        payload["augmented_frames_uri"] = augment["augmented_frames_uri"]
        payload["frame_count"] = augment["frame_count"]

    if output_json is not None:
        payload = write_manifest(payload, output_json)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
