"""npa workbench nurec - NVIDIA Omniverse NuRec / Neural Reconstruction Engine.

Stage verbs for the ``neural-reconstruction`` capability. Each verb is a real
entrypoint that drives the real component:

* ``check``       - NGC container pullability, HF dataset download rights, RT-core GPU
* ``fetch``       - download + unpack real NCore V4 shards from a PhysicalAI dataset
* ``reconstruct`` - NRE 3DGUT training -> renderable ``usd-out/last.usdz`` + metrics
* ``render``      - ``nre render`` novel views (rig-offset, NOT training views)
* ``visualize``   - build ``reports/sim2real.rrd`` via the tested viz module
* ``finalize``    - aggregate the run tree into ``reports/final.json``
* ``status``      - what a run prefix currently holds, stage by stage

Artifacts hand off through S3 under ONE run prefix so the NPA agent's artifact
browser and embedded Rerun viewer pick the run up automatically.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from npa.workbench.nurec.nurec import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_DATASET_ID,
    DEFAULT_FRAME_STEP,
    DEFAULT_GT_FRAME_STEP_CAMERA,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_SCALE,
    DEFAULT_FRAME_NAMING,
    DEFAULT_MODE,
    DEFAULT_NRE_ENTRYPOINT,
    DEFAULT_NRE_IMAGE,
    DEFAULT_RENDERER,
    DEFAULT_SCENE,
    DEFAULT_VARIANT,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_FPS,
    NurecConfig,
    NurecError,
    check_nurec_access,
    fetch_nurec_dataset,
    nurec_run_status,
    reconstruct_scene,
    render_novel_views,
)

app = typer.Typer(
    name="nurec",
    help=(
        "NVIDIA Omniverse NuRec / Neural Reconstruction Engine: sensor recordings "
        "-> 3DGUT Gaussian reconstruction -> renderable USDZ -> novel-view renders. "
        "Requires an RT-core GPU (L40S or RTX PRO 6000 Blackwell); never route the "
        "render path at H100/H200."
    ),
    no_args_is_help=True,
)

#: Rerun recording the NPA agent auto-selects for a run (see
#: npa.workflows.artifacts.select_preferred_artifact).
RRD_BASENAME = "sim2real.rrd"
VIZ_APP_ID = "neural-reconstruction"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _output(data: dict[str, Any], output: OutputFormat) -> None:
    if output is OutputFormat.json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
        return
    for key, value in sorted(data.items()):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        typer.echo(f"{key}: {value}")


def _finish_nurec_result(data: dict[str, Any], output: OutputFormat) -> None:
    """Emit a result payload and exit non-zero when the stage failed."""
    _output(data, output)
    if data.get("status") != "ok":
        raise typer.Exit(1)


def _config(**overrides: Any) -> NurecConfig:
    try:
        return NurecConfig.from_env(**overrides)
    except NurecError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _publish(local_path: Path, output_uri: str) -> str:
    """Upload a file or directory to ``output_uri`` (S3 or local)."""
    if not output_uri:
        return ""
    if not output_uri.startswith("s3://"):
        destination = Path(output_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_dir():
            import shutil

            # Merge rather than rmtree: a mistyped --output-uri pointing at a
            # populated directory must not delete it. dirs_exist_ok overwrites the
            # files we actually publish and leaves everything else alone.
            shutil.copytree(local_path, destination, dirs_exist_ok=True)
        else:
            destination.write_bytes(local_path.read_bytes())
        return str(destination)
    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    return client.upload_path(str(local_path), output_uri)


@app.command("check")
def check_cmd(
    image: str = typer.Option(
        "",
        "--image",
        envvar="NPA_NUREC_IMAGE",
        help=f"NRE container reference. Defaults to NPA_NUREC_IMAGE or {DEFAULT_NRE_IMAGE}.",
    ),
    dataset: str = typer.Option(
        "",
        "--dataset",
        envvar="NPA_NUREC_DATASET",
        help=f"Hugging Face dataset id. Defaults to NPA_NUREC_DATASET or {DEFAULT_DATASET_ID}.",
    ),
    scene: str = typer.Option(
        "", "--scene", envvar="NPA_NUREC_SCENE", help=f"Scene name (default {DEFAULT_SCENE})."
    ),
    variant: str = typer.Option(
        "",
        "--variant",
        envvar="NPA_NUREC_VARIANT",
        help=f"Scene variant sub-directory (default {DEFAULT_VARIANT}).",
    ),
    entrypoint: str = typer.Option(
        "",
        "--entrypoint",
        envvar="NPA_NUREC_ENTRYPOINT",
        help=f"In-container NRE entrypoint (default {DEFAULT_NRE_ENTRYPOINT}).",
    ),
    docker_bin: str = typer.Option(
        "",
        "--docker-bin",
        envvar="NPA_NUREC_DOCKER",
        help="Run NRE through this docker binary instead of in-container.",
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", envvar="NPA_NUREC_CACHE", help="Ephemeral runtime cache directory."
    ),
    hf_token_env: str = typer.Option(
        "", "--hf-token-env", help="Environment variable holding the Hugging Face token."
    ),
    ngc_api_key_env: str = typer.Option(
        "", "--ngc-api-key-env", help="Environment variable holding the NGC API key."
    ),
    require_ngc: bool = typer.Option(
        True,
        "--require-ngc/--no-require-ngc",
        help="Fail when NGC auth is absent (the NRE container needs it).",
    ),
    require_gpu: bool = typer.Option(
        False,
        "--require-gpu/--no-require-gpu",
        help="Fail when no NVIDIA GPU is visible.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Check NRE container access, dataset download rights, and GPU suitability."""
    config = _config(
        image=image,
        dataset_id=dataset,
        scene=scene,
        variant=variant,
        entrypoint=entrypoint,
        docker_bin=docker_bin,
        cache_dir=cache_dir,
        hf_token_env=hf_token_env,
        ngc_api_key_env=ngc_api_key_env,
    )
    result = check_nurec_access(config, require_ngc=require_ngc, require_gpu=require_gpu)
    _finish_nurec_result(result.as_dict(), output)


@app.command("fetch")
def fetch_cmd(
    dataset: str = typer.Option(
        "", "--dataset", envvar="NPA_NUREC_DATASET", help="Hugging Face dataset id."
    ),
    scene: str = typer.Option("", "--scene", envvar="NPA_NUREC_SCENE", help="Scene name."),
    variant: str = typer.Option(
        "", "--variant", envvar="NPA_NUREC_VARIANT", help="Scene variant sub-directory."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", envvar="NPA_NUREC_CACHE", help="Ephemeral runtime cache directory."
    ),
    hf_token_env: str = typer.Option(
        "", "--hf-token-env", help="Environment variable holding the Hugging Face token."
    ),
    with_colmap: bool = typer.Option(
        False,
        "--with-colmap/--no-with-colmap",
        help="Also download the COLMAP copy of the capture (poses + sparse cloud).",
    ),
    derive_rig: bool = typer.Option(
        True,
        "--derive-rig/--no-derive-rig",
        help=(
            "Derive the rig->world pose edge NRE requires when the sequence has "
            "none (object-centric captures). Without it NRE cannot load the data."
        ),
    ),
    reference_camera: str = typer.Option(
        "",
        "--reference-camera",
        envvar="NPA_NUREC_REFERENCE_CAMERA",
        help="Camera whose trajectory becomes the rig trajectory. Default: the longest.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-download and re-extract even when the cache is populated."
    ),
    output_uri: str = typer.Option(
        "",
        "--output-uri",
        "--output-path",
        help="S3/local prefix for the fetch manifest (e.g. s3://.../<run>/ncore/).",
    ),
    publish_sequence: bool = typer.Option(
        False,
        "--publish-sequence/--no-publish-sequence",
        help=(
            "Also upload the whole NCore sequence (meta-file + every shard, with "
            "symlinks resolved) under <output-uri>sequence/ so a LATER STAGE IN "
            "ANOTHER POD can consume it. Required by the declarative workflow, "
            "where each stage is its own pod and /tmp is not shared."
        ),
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Download and unpack the real NCore V4 shards for a scene."""
    config = _config(
        dataset_id=dataset,
        scene=scene,
        variant=variant,
        cache_dir=cache_dir,
        hf_token_env=hf_token_env,
        reference_camera=reference_camera,
    )
    result = fetch_nurec_dataset(
        config, force=force, with_colmap=with_colmap, derive_rig=derive_rig
    )
    payload = result.as_dict()
    if result.ok and output_uri and publish_sequence:
        from npa.workbench.nurec.nurec import NurecError as _NurecError, publish_ncore_sequence

        try:
            published = publish_ncore_sequence(
                result.ncore_json, _join_uri(output_uri, "sequence/")
            )
        except _NurecError as exc:
            payload["status"] = "failed"
            payload["errors"] = [*payload.get("errors", []), str(exc)]
            _finish_nurec_result(payload, output)
            return
        payload["sequence_uri"] = published["uri"]
        payload["sequence_objects"] = published["objects"]
        payload["sequence_bytes"] = published["bytes"]
        payload["sequence_meta_name"] = published["meta_name"]
    if result.ok and output_uri:
        manifest = config.resolved_cache_dir / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["output_uri"] = _publish(manifest, _join_uri(output_uri, "manifest.json"))
    _finish_nurec_result(payload, output)


@app.command("reconstruct")
def reconstruct_cmd(
    ncore_json: str = typer.Option(
        "",
        "--ncore-json",
        envvar="NPA_NUREC_NCORE_JSON",
        help="NCore V4 sequence meta-file. Defaults to the fetched scene's JSON.",
    ),
    ncore_uri: str = typer.Option(
        "",
        "--ncore-uri",
        envvar="NPA_NUREC_NCORE_URI",
        help=(
            "S3 prefix holding a published NCore sequence (from `fetch "
            "--publish-sequence`). Materialized locally first, so a stage running "
            "in its own pod does not need the fetch stage's filesystem."
        ),
    ),
    dataset: str = typer.Option(
        "", "--dataset", envvar="NPA_NUREC_DATASET", help="Hugging Face dataset id."
    ),
    scene: str = typer.Option("", "--scene", envvar="NPA_NUREC_SCENE", help="Scene name."),
    variant: str = typer.Option(
        "", "--variant", envvar="NPA_NUREC_VARIANT", help="Scene variant sub-directory."
    ),
    config_name: str = typer.Option(
        "",
        "--config-name",
        envvar="NPA_NUREC_CONFIG_NAME",
        help=(
            "NRE Hydra recipe shipped in the container. Defaults to "
            f"{DEFAULT_CONFIG_NAME} (static, camera-only, object-centric)."
        ),
    ),
    mode: str = typer.Option(
        "", "--mode", envvar="NPA_NUREC_MODE", help=f"train, val, or trainval (default {DEFAULT_MODE})."
    ),
    poses_component_group: str = typer.Option(
        "",
        "--poses-component-group",
        envvar="NPA_NUREC_POSES_COMPONENT_GROUP",
        help="NCore poses component group to select (e.g. the derived rig group).",
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", envvar="NPA_NUREC_OUT", help="NRE output root."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", envvar="NPA_NUREC_CACHE", help="Ephemeral runtime cache directory."
    ),
    max_epochs: int = typer.Option(
        0,
        "--max-epochs",
        envvar="NPA_NUREC_MAX_EPOCHS",
        help="Override trainer.max_epochs. 0 keeps the recipe's own budget.",
    ),
    world_size: int = typer.Option(
        1, "--world-size", envvar="NPA_NUREC_WORLD_SIZE", help="GPUs per node (trainer.world_size)."
    ),
    precision: str = typer.Option(
        "", "--precision", envvar="NPA_NUREC_PRECISION", help="trainer.precision, e.g. 16-mixed."
    ),
    camera_id: list[str] = typer.Option(
        [], "--camera-id", help="Restrict dataset.camera_ids; repeatable."
    ),
    lidar_id: list[str] = typer.Option(
        [],
        "--lidar-id",
        help="Restrict dataset.lidar_ids; repeatable. Pass 'none' to force an empty list.",
    ),
    aux_data: bool = typer.Option(
        False,
        "--aux-data/--no-aux-data",
        envvar="NPA_NUREC_AUX_DATA",
        help="Use the auxiliary NCore shards (seg/depth). Off for camera-only captures.",
    ),
    override: list[str] = typer.Option(
        [], "--override", help="Extra raw Hydra override, e.g. trainer.precision=32; repeatable."
    ),
    image: str = typer.Option("", "--image", envvar="NPA_NUREC_IMAGE", help="NRE container reference."),
    entrypoint: str = typer.Option(
        "", "--entrypoint", envvar="NPA_NUREC_ENTRYPOINT", help="In-container NRE entrypoint."
    ),
    docker_bin: str = typer.Option(
        "", "--docker-bin", envvar="NPA_NUREC_DOCKER", help="Run NRE through this docker binary."
    ),
    export_gt: bool = typer.Option(
        True,
        "--export-gt/--no-export-gt",
        help="Also export real capture frames with export-ncore-benchmark-gt.",
    ),
    gt_frame_step: int = typer.Option(
        DEFAULT_GT_FRAME_STEP_CAMERA,
        "--gt-frame-step",
        help="Camera frame step for the ground-truth export.",
    ),
    output_uri: str = typer.Option(
        "",
        "--output-uri",
        "--output-path",
        help="S3/local prefix for the USDZ, parsed config, metrics, and val renders.",
    ),
    input_uri: str = typer.Option(
        "",
        "--input-uri",
        "--input-path",
        help="S3/local prefix for the exported real capture frames (input/).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved NRE command without running it."
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Train a 3DGUT Gaussian reconstruction and publish the renderable USDZ."""
    config = _config(
        image=image,
        entrypoint=entrypoint,
        docker_bin=docker_bin,
        dataset_id=dataset,
        scene=scene,
        variant=variant,
        cache_dir=cache_dir,
        out_dir=out_dir,
        config_name=config_name,
        mode=mode,
        poses_component_group=poses_component_group,
        max_epochs=max_epochs,
        world_size=world_size,
        precision=precision,
        camera_ids=camera_id,
        lidar_ids=lidar_id,
        aux_data=aux_data,
        extra_overrides=override,
    )
    resolved_json = ncore_json or _materialize_ncore(config, ncore_uri) or _discover_ncore_json(config)
    if resolved_json and not (lidar_id and camera_id):
        # The shipped recipes carry PLACEHOLDER sensor ids that only match
        # NVIDIA-internal data, so on a real capture NRE aborts with
        # "Requested lidars not present in the data: dummy_lidar" or
        # "Requested cameras not present in the data: camera_front_wide_120fov"
        # (both observed live). Adopt whatever the sequence actually declares --
        # and explicitly blank the LiDAR list for a camera-only capture -- so the
        # recipe works on real input without the caller having to know the ids.
        from npa.workbench.nurec.nurec import NO_LIDAR_SENTINEL, ncore_sensor_ids

        from npa.workbench.nurec.nurec import read_rig_sidecar

        discovered_cameras, discovered = ncore_sensor_ids(resolved_json)
        # A derived-rig sequence is an object-centric capture, and the recipe's
        # SfM point-cloud initialization asserts "Only one camera sensor is
        # currently supported" (observed live). The rig IS the reference camera, so
        # training on exactly that camera is both required and geometrically
        # coherent. AV sequences ship their own rig and no sidecar, so they keep
        # full multi-camera behaviour.
        reference = str(read_rig_sidecar(resolved_json).get("reference_camera") or "")
        default_cameras = [reference] if reference else list(discovered_cameras)
        if not camera_id and reference and len(discovered_cameras) > 1:
            # Silently dropping real training data would be worse than being noisy.
            typer.echo(
                f"note: restricting training to the rig reference camera "
                f"{reference!r}; the capture also has "
                f"{sorted(set(discovered_cameras) - {reference})}. The recipe's SfM "
                "point-cloud initialization supports only one camera. Pass "
                "--camera-id explicitly to override.",
                err=True,
            )
        camera_id = list(camera_id) or default_cameras
        # Rebuild through _config() so a bad value still produces the CLI's
        # `error: ...` / exit 2 contract rather than an uncaught traceback.
        config = _config(
            image=image,
            entrypoint=entrypoint,
            docker_bin=docker_bin,
            dataset_id=dataset,
            scene=scene,
            variant=variant,
            cache_dir=cache_dir,
            out_dir=out_dir,
            config_name=config_name,
            mode=mode,
            poses_component_group=poses_component_group,
            max_epochs=max_epochs,
            world_size=world_size,
            precision=precision,
            camera_ids=camera_id,
            lidar_ids=list(lidar_id) or list(discovered) or [NO_LIDAR_SENTINEL],
            aux_data=aux_data,
            extra_overrides=override,
        )
    if not resolved_json:
        _finish_nurec_result(
            {
                "status": "failed",
                "errors": [
                    "no NCore sequence meta-file found; run `npa workbench nurec fetch` "
                    "first (same pod), or pass --ncore-json / --ncore-uri"
                ],
            },
            output,
        )
        return
    result = reconstruct_scene(
        config,
        ncore_json=resolved_json,
        dry_run=dry_run,
        export_gt=export_gt,
    )
    payload = result.as_dict()
    payload["ncore_json"] = resolved_json
    if result.ok and not dry_run and output_uri:
        payload["output_uri"] = _publish_reconstruction(result, output_uri)
    if result.ok and not dry_run and input_uri and result.gt_dir:
        payload["input_uri"] = _publish(Path(result.gt_dir), input_uri)
    _finish_nurec_result(payload, output)


@app.command("render")
def render_cmd(
    artifact_path: str = typer.Option(
        "",
        "--artifact-path",
        envvar="NPA_NUREC_ARTIFACT",
        help="Trained .usdz artifact. Defaults to the reconstruct stage's output.",
    ),
    artifact_uri: str = typer.Option(
        "",
        "--artifact-uri",
        envvar="NPA_NUREC_ARTIFACT_URI",
        help=(
            "S3 URI of a trained .usdz (or the prefix the reconstruct stage wrote). "
            "Downloaded first, so a stage running in its own pod does not need the "
            "reconstruct stage's filesystem."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", envvar="NPA_NUREC_RENDER_DIR", help="Local render output directory."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", envvar="NPA_NUREC_OUT", help="NRE output root (to locate the USDZ)."
    ),
    camera_id: list[str] = typer.Option(
        [], "--camera-id", help="Camera to render from; repeatable. Required by NRE."
    ),
    image_scale: float = typer.Option(
        DEFAULT_IMAGE_SCALE, "--image-scale", help="Output resolution as a fraction of the camera's."
    ),
    image_format: str = typer.Option(
        DEFAULT_IMAGE_FORMAT, "--image-format", help="png, jpg, or jpeg."
    ),
    frame_naming: str = typer.Option(
        DEFAULT_FRAME_NAMING,
        "--frame-naming",
        help="frame-end-timestamp or contiguous-output-index.",
    ),
    renderer: str = typer.Option(
        DEFAULT_RENDERER,
        "--renderer",
        help=(
            "default (the artifact's trained renderer), gsplat, or nrend. nrend "
            "needs the nrend model dict embedded in the USDZ."
        ),
    ),
    frame_step: int = typer.Option(DEFAULT_FRAME_STEP, "--frame-step", help="Frame step size."),
    rig_translation_offset: str = typer.Option(
        "",
        "--rig-translation-offset",
        envvar="NPA_NUREC_RIG_TRANSLATION_OFFSET",
        help="Rig translation offset 'tx,ty,tz' in metres. This is what makes a view novel.",
    ),
    rig_rotation_offset: str = typer.Option(
        "",
        "--rig-rotation-offset",
        envvar="NPA_NUREC_RIG_ROTATION_OFFSET",
        help="Rig rotation offset 'yaw,-roll,-pitch' in degrees.",
    ),
    custom_rig_trajectory: str = typer.Option(
        "", "--custom-rig-trajectory", help="Custom rig trajectory JSON to render along."
    ),
    replicate_training_views: bool = typer.Option(
        False,
        "--replicate-training-views/--no-replicate-training-views",
        help="Re-render the exact training views instead of novel views.",
    ),
    export_video: bool = typer.Option(
        True, "--export-video/--no-export-video", help="Also encode an MP4 per camera."
    ),
    video_fps: float = typer.Option(DEFAULT_VIDEO_FPS, "--video-fps", help="Exported video FPS."),
    video_crf: int = typer.Option(
        DEFAULT_VIDEO_CRF, "--video-crf", help="Exported video CRF (0-51; lower is better)."
    ),
    ffmpeg_exe: str = typer.Option(
        "",
        "--ffmpeg-exe",
        envvar="NPA_NUREC_FFMPEG_EXE",
        help="ffmpeg binary NRE should use for --export-video.",
    ),
    image: str = typer.Option("", "--image", envvar="NPA_NUREC_IMAGE", help="NRE container reference."),
    entrypoint: str = typer.Option(
        "", "--entrypoint", envvar="NPA_NUREC_ENTRYPOINT", help="In-container NRE entrypoint."
    ),
    docker_bin: str = typer.Option(
        "", "--docker-bin", envvar="NPA_NUREC_DOCKER", help="Run NRE through this docker binary."
    ),
    output_uri: str = typer.Option(
        "", "--output-uri", "--output-path", help="S3/local prefix for the rendered novel views."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved NRE command without running it."
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Render novel views from a trained reconstruction with ``nre render``."""
    config = _config(
        image=image,
        entrypoint=entrypoint,
        docker_bin=docker_bin,
        out_dir=out_dir,
        ffmpeg_exe=ffmpeg_exe,
    )
    resolved_artifact = (
        artifact_path or _materialize_artifact(config, artifact_uri) or _discover_usdz(config)
    )
    if not resolved_artifact:
        _finish_nurec_result(
            {
                "status": "failed",
                "errors": [
                    "no trained .usdz found; run `npa workbench nurec reconstruct` first "
                    "(same pod), or pass --artifact-path / --artifact-uri"
                ],
            },
            output,
        )
        return
    target_dir = Path(output_dir) if output_dir else config.resolved_out_dir / "novel_views"
    result = render_novel_views(
        config,
        artifact_path=resolved_artifact,
        output_dir=str(target_dir),
        camera_ids=camera_id,
        image_scale=image_scale,
        image_format=image_format,
        frame_naming=frame_naming,
        renderer=renderer,
        frame_step=frame_step,
        rig_translation_offset=rig_translation_offset,
        rig_rotation_offset=rig_rotation_offset,
        custom_rig_trajectory=custom_rig_trajectory,
        replicate_training_views=replicate_training_views,
        export_video=export_video,
        video_fps=video_fps,
        video_crf=video_crf,
        dry_run=dry_run,
    )
    payload = result.as_dict()
    if result.ok and not dry_run and output_uri:
        payload["output_uri"] = _publish(target_dir, output_uri)
    _finish_nurec_result(payload, output)


@app.command("visualize")
def visualize_cmd(
    input_uri: str = typer.Option(
        ...,
        "--input-uri",
        "--input-path",
        help="Run root holding input/, reconstruction/, novel_views/ (S3 or local).",
    ),
    output_uri: str = typer.Option(
        "",
        "--output-uri",
        "--output-path",
        help=f"Destination .rrd. Defaults to <input-uri>/reports/{RRD_BASENAME}.",
    ),
    app_id: str = typer.Option(
        VIZ_APP_ID, "--app-id", help="Rerun application id recorded in the .rrd."
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Build the run's Rerun recording so it renders in the NPA agent viewer."""
    from npa.workflows.data_factory_viz import DataFactoryVizError, build_run_rrd

    target = output_uri or _join_uri(input_uri, f"reports/{RRD_BASENAME}")
    try:
        result = build_run_rrd(input_uri, target, app_id=app_id)
    except DataFactoryVizError as exc:
        _finish_nurec_result({"status": "failed", "errors": [str(exc)]}, output)
        return
    payload = dict(result)
    payload["status"] = "ok" if result.get("status") == "completed" else "failed"
    _finish_nurec_result(payload, output)


@app.command("finalize")
def finalize_cmd(
    input_uri: str = typer.Option(
        ..., "--input-uri", "--input-path", help="Run root to aggregate (S3 or local)."
    ),
    output_uri: str = typer.Option(
        "",
        "--output-uri",
        "--output-path",
        help="Destination report. Defaults to <input-uri>/reports/final.json.",
    ),
    run_id: str = typer.Option("", "--run-id", envvar="NPA_NUREC_RUN_ID", help="Run id to record."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Aggregate the run tree into a real final report."""
    status_result = nurec_run_status(input_uri)
    report = {
        "status": "ok" if status_result.ok else "failed",
        "capability": VIZ_APP_ID,
        "run_id": run_id or _run_id_from_uri(input_uri),
        "run_uri": input_uri,
        "artifact_count": status_result.object_count,
        "stages": {k: dict(v) for k, v in status_result.stages.items()},
        "has_rrd": status_result.has_rrd,
        "has_usdz": status_result.has_usdz,
        "has_novel_views": status_result.has_novel_views,
        "errors": list(status_result.errors),
    }
    target = output_uri or _join_uri(input_uri, "reports/final.json")
    if status_result.ok:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="npa-nurec-final-") as tmp:
            local = Path(tmp) / "final.json"
            local.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            report["output_uri"] = _publish(local, target)
    _finish_nurec_result(report, output)


@app.command("status")
def status_cmd(
    run_uri: str = typer.Option(
        ..., "--run-uri", "--input-path", help="Run prefix to summarize (S3 or local)."
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Summarize what a NuRec run prefix currently holds, stage by stage."""
    _finish_nurec_result(nurec_run_status(run_uri).as_dict(), output)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _join_uri(base: str, suffix: str) -> str:
    return f"{str(base).rstrip('/')}/{str(suffix).lstrip('/')}"


def _run_id_from_uri(uri: str) -> str:
    return str(uri or "").rstrip("/").split("/")[-1]


def _discover_ncore_json(config: NurecConfig) -> str:
    """Find the NCore meta-file the fetch stage unpacked into the cache."""
    from npa.workbench.nurec.nurec import find_ncore_json, find_scene_dir

    ncore_root = config.resolved_cache_dir / "ncore"
    if not ncore_root.is_dir():
        return ""
    scene_dir = find_scene_dir(ncore_root, config.scene_dir_name)
    if scene_dir is None:
        return ""
    found = find_ncore_json(scene_dir)
    return str(found) if found else ""


def _materialize_ncore(config: NurecConfig, ncore_uri: str) -> str:
    """Pull a published NCore sequence into the local cache and return its meta-file."""
    if not ncore_uri:
        return ""
    from npa.workbench.nurec.nurec import find_ncore_json, materialize_uri

    target = config.resolved_cache_dir / "ncore-staged"
    source = ncore_uri if ncore_uri.endswith("/") else f"{ncore_uri}/"
    local = materialize_uri(source, target)
    found = find_ncore_json(Path(local))
    return str(found) if found else ""


def _materialize_artifact(config: NurecConfig, artifact_uri: str) -> str:
    """Download a trained USDZ (object or prefix) and return the local path."""
    if not artifact_uri:
        return ""
    from npa.workbench.nurec.nurec import latest_usdz, materialize_uri

    staged = config.resolved_out_dir / "artifact-staged"
    if artifact_uri.endswith(".usdz"):
        local = materialize_uri(artifact_uri, staged / Path(artifact_uri).name)
        return str(local)
    local = materialize_uri(
        artifact_uri if artifact_uri.endswith("/") else f"{artifact_uri}/", staged
    )
    found = latest_usdz(Path(local))
    return str(found) if found else ""


def _discover_usdz(config: NurecConfig) -> str:
    """Find the newest USDZ the reconstruct stage produced under the NRE output root."""
    from npa.workbench.nurec.nurec import latest_usdz, resolve_nre_run_dir

    run_dir = resolve_nre_run_dir(config.resolved_out_dir, config.nre_run_id)
    found = latest_usdz(run_dir)
    return str(found) if found else ""


def _publish_reconstruction(result: Any, output_uri: str) -> str:
    """Upload the USDZ, parsed config, metrics, and validation renders."""
    published: list[str] = []
    for local, name in (
        (result.usdz_path, "last.usdz"),
        (result.parsed_config_path, "parsed.yaml"),
        (result.metrics_path, "metrics.yaml"),
    ):
        if local and Path(local).is_file():
            published.append(_publish(Path(local), _join_uri(output_uri, name)))
    val_dir = Path(result.run_dir) / "val"
    if val_dir.is_dir():
        published.append(_publish(val_dir, _join_uri(output_uri, "val")))
    return published[0] if published else ""
