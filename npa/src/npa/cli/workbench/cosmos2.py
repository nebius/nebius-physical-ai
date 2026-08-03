"""Workbench Cosmos2 commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

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


#: What a cosmos2 transfer stage publishes beside its clip; specs declare this path.
MANIFEST_FILENAME = "manifest.json"


def _publish_manifest(client: Any, payload: dict, output_uri: str) -> str:
    """Upload the stage manifest next to the augmented clip and return its URI."""

    import tempfile as _tempfile

    prefix = output_uri if output_uri.endswith("/") else output_uri + "/"
    with _tempfile.TemporaryDirectory(prefix="npa-cosmos2-") as tmp:
        local = Path(tmp) / MANIFEST_FILENAME
        local.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return client.upload_file(str(local), prefix + MANIFEST_FILENAME)


def _all_augmentations(configs_uri: str) -> list[dict]:
    """Read the Config-Gen manifest and return every sampled appearance combo.

    Each combo drives one Cosmos Transfer 2.5 inference ("multiply"), so a config
    manifest with N augmentations yields N scenario variants. Best-effort: returns
    [] on any read failure so the caller can fall back to a single default render.
    """
    try:
        from npa.workflows.data_factory_stages import _download_json

        uri = configs_uri if configs_uri.endswith(".json") else configs_uri.rstrip("/") + "/manifest.json"
        manifest = _download_json(uri)
        combos = manifest.get("augmentations") or []
        return [c for c in combos if isinstance(c, dict)]
    except Exception:  # noqa: BLE001 - variables are advisory metadata, never fatal
        return []


def _first_augmentation(configs_uri: str) -> dict:
    """Read the Config-Gen manifest and return the first sampled combo (or {})."""
    combos = _all_augmentations(configs_uri)
    return combos[0] if combos else {}


_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _detect_gpu_count() -> int:
    """Best-effort count of GPUs visible to this process (>=1).

    Prefers an explicit ``CUDA_VISIBLE_DEVICES`` list, then ``nvidia-smi -L``.
    Used to auto-parallelize the multiply fan-out (one variant per GPU) so a
    workflow that requests ``RTXPRO6000:4`` actually drives all four GPUs.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        ids = [x for x in cvd.split(",") if x.strip() != ""]
        return max(1, len(ids))
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
        n = len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])
        return max(1, n)
    except Exception:  # noqa: BLE001 - detection is advisory; default to 1
        return 1


def _variant_parallelism(num_variants: int) -> int:
    """Resolve how many variant inferences to run concurrently (>=1).

    ``NPA_COSMOS_VARIANT_PARALLELISM`` overrides; otherwise auto-detect the GPU
    count. Capped at the number of variants so we never spawn idle workers.
    """
    override = os.environ.get("NPA_COSMOS_VARIANT_PARALLELISM", "").strip()
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 1
    else:
        requested = _detect_gpu_count()
    return max(1, min(requested, max(1, int(num_variants))))


def _materialize_input_clip(src: str) -> str:
    """Resolve a local path or ``s3://`` URI to a local conditioning video.

    Returns an empty string only when the source was successfully inspected and no
    supported video exists. Storage setup, listing, authentication, and download
    failures propagate so the CLI can report them separately from an empty prefix.
    """
    import glob as _glob
    import shutil
    import tempfile
    from urllib.parse import urlsplit

    s = str(src or "").strip()
    if not s:
        return ""
    if not s.startswith("s3://"):
        return s if Path(s).is_file() else ""

    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    tmp = tempfile.mkdtemp(prefix="npa-cosmos-input-")
    keep_tmp = False
    try:
        source_path = urlsplit(s).path
        if source_path.lower().endswith(_VIDEO_EXTS):
            downloaded = client.download_path(s, str(Path(tmp) / Path(source_path).name))
            keep_tmp = True
            return downloaded
        client.download_directory(s, tmp)
        vids = sorted(
            f for f in _glob.glob(str(Path(tmp) / "**" / "*"), recursive=True)
            if f.lower().endswith(_VIDEO_EXTS) and Path(f).is_file()
        )
        if vids:
            keep_tmp = True
            return vids[0]
        return ""
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _materialize_conditioning_input(src: str) -> str:
    """Adapt storage failures to a sanitized, actionable CLI error."""
    try:
        return _materialize_input_clip(src)
    except Exception as exc:
        raise typer.BadParameter(
            "could not inspect or download the configured conditioning input; "
            "verify the object-storage endpoint, credentials, permissions, and availability"
        ) from exc


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
        help="Condition on the first video under --input-uri. Also enabled by "
        "NPA_COSMOS_CONDITION_ON_INPUT=1.",
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
        # Data Factory context (`transfer_execute` passes --configs-uri and always
        # enables input conditioning): the sampled appearance combo drives the prompt,
        # and the augment CONDITIONS on the run's real input clip (edge control
        # computed on-the-fly — a genuine augmentation of that footage),
        # and the result is published in the per-clip layout
        # that data_factory curate / build_run_rrd / provenance consume. Opt-in via
        # Generic callers opt in via --input-video, --condition-on-input, or
        # NPA_COSMOS_CONDITION_ON_INPUT=1.
        #
        # Otherwise (generic `transfer` for sim2real / cosmos-gate / fanout): keep the
        # flat single-video publish + sim2real-engine field convention unchanged.
        condition_requested = bool(
            input_video or condition_on_input or _env_truthy("NPA_COSMOS_CONDITION_ON_INPUT")
        )
        data_factory_mode = bool(configs_uri) or condition_requested
        local_input = ""
        if condition_requested:
            local_input = _materialize_conditioning_input(input_video or input_uri)
            if not local_input:
                raise typer.BadParameter(
                    "input conditioning was requested, but no supported video "
                    f"({', '.join(_VIDEO_EXTS)}) was found at the configured input"
                )
        # Env fallbacks let a submit tune conditioning without changing the toolRef argv.
        control = (os.environ.get("NPA_COSMOS_CONTROL", "").strip() or control)
        _cw = os.environ.get("NPA_COSMOS_CONTROL_WEIGHT", "").strip()
        _g = os.environ.get("NPA_COSMOS_GUIDANCE", "").strip()
        if _cw:
            control_weight = float(_cw)
        if _g:
            guidance = float(_g)

        if data_factory_mode and output_uri.strip().startswith("s3://"):
            # Augment & MULTIPLY. Run one REAL Cosmos Transfer 2.5 inference per
            # sampled appearance combo (each with its own prompt), publishing each
            # as its own per-clip dir under the cosmos_augmented/ prefix, then write
            # a single run-level manifest.json listing them all. A config manifest
            # with N augmentations therefore yields N scenario variants (not one
            # image). The per-clip layout is what data_factory curate /
            # build_run_rrd / provenance consume.
            from npa.workbench.cosmos.transfer import (
                publish_transfer_clip,
                write_run_manifest,
            )

            combos = _all_augmentations(configs_uri) if configs_uri else []
            if not combos:
                combos = [{}]

            parallelism = _variant_parallelism(len(combos))

            def _render_variant(i: int, combo: dict) -> dict:
                variant_run = f"{run_id}-v{i}" if run_id else f"v{i}"
                # Pin each concurrent variant to a distinct GPU so an N-GPU pod
                # runs N diffusions at once (sequential when parallelism == 1).
                device = str(i % parallelism) if parallelism > 1 else None
                return run_cosmos_transfer(
                    run_id=variant_run,
                    spec=spec or None,
                    prompt=str(combo.get("prompt") or "") or None,
                    input_video=local_input or None,
                    control=control,
                    control_weight=control_weight,
                    guidance=guidance,
                    cuda_visible_devices=device,
                    variant_tag=variant_run,
                )

            # Fan the GPU-bound diffusions out across the pod's GPUs, then publish
            # sequentially in combo order (publish/S3 upload stays single-threaded).
            transfers: list[dict] = [dict() for _ in combos]
            if parallelism > 1 and len(combos) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=parallelism) as pool:
                    futures = {
                        pool.submit(_render_variant, i, combo): i
                        for i, combo in enumerate(combos)
                    }
                    for future in futures:
                        transfers[futures[future]] = future.result()
            else:
                for i, combo in enumerate(combos):
                    transfers[i] = _render_variant(i, combo)

            clips: list[dict] = []
            for i, combo in enumerate(combos):
                clip_name = f"aug-{run_id}-{i}" if run_id else f"aug{i}"
                clips.append(
                    publish_transfer_clip(
                        transfers[i],
                        output_uri,
                        run_id=run_id,
                        clip_name=clip_name,
                        variables=combo,
                    )
                )
            manifest = write_run_manifest(
                clips, output_uri, run_id=run_id, variant_parallelism=parallelism
            )
            payload["status"] = "executed"
            payload["output_kind"] = "video"
            payload["mode"] = "cosmos_transfer2.5_gpu" if local_input else "cosmos_transfer2.5"
            payload["augmented_video_uri"] = manifest["augmented_video_uri"]
            payload["augmented_videos"] = manifest["augmented_videos"]
            payload["frame_count"] = manifest["frame_count"]
            payload["variant_count"] = manifest["variant_count"]
            payload["multiply_mode"] = manifest["multiply_mode"]
            payload["variant_parallelism"] = manifest["variant_parallelism"]
            payload["clips"] = manifest["clips"]
            payload["augmentation_variables"] = combos[0]
            payload["prompt"] = str((combos[0] or {}).get("prompt") or "")
            payload["input_conditioned"] = bool(local_input)
            payload["control_spec"] = manifest["control_spec"]
            if local_input:
                payload["input_video"] = local_input
                payload["control"] = manifest["control"]
            # attribute-verify reads --input-path {{augmented_frames_uri}} (the prefix).
            payload["augmented_frames_uri"] = output_uri
        else:
            # Single inference: generic transfer (sim2real / cosmos-gate / fanout)
            # or a non-S3 output. Unchanged field convention.
            variables = _first_augmentation(configs_uri) if configs_uri else {}
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
                # Generic single-video publish + sim2real-engine field convention.
                from npa.clients.storage import StorageClient

                client = StorageClient.from_environment()
                output_video = client.upload_file(transfer["video_path"], output_uri)
                payload["mode"] = "cosmos_transfer2.5"
                payload["output_video"] = output_video
                payload["augmented_video_uri"] = output_video
                payload["augmented_frames_uri"] = output_uri
                # Publish the manifest beside the video. Echoing it to stdout leaves the
                # provenance — prompt, control spec, guidance, whether the run was conditioned
                # on an input clip — inside a pod that is about to disappear. For a synthetic
                # data stage that provenance IS the product, and a spec needs something durable
                # to declare as its output (live job 287 published a 3.9 MB augmented clip and
                # nothing that said how it was made). The data-factory path already does this
                # via write_run_manifest; the single-inference path now matches it.
                payload["manifest_uri"] = _publish_manifest(client, payload, output_uri)
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
