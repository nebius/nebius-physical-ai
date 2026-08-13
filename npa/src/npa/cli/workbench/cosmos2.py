"""Workbench Cosmos2 commands."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import typer

from npa.workflows.cosmos_split import (
    Cosmos2TransferConfig,
    build_cosmos2_transfer_manifest,
    write_manifest,
)
from npa.workbench.cosmos.transfer import (
    REFERENCE_AUGMENT_MODE,
    REFERENCE_AUGMENT_STATUS,
    TRANSFER_MANIFEST_FILENAME,
    TRANSFER_MANIFEST_MODE,
    TRANSFER_MANIFEST_STATUS,
    transfer_manifest_uri_for,
)

app = typer.Typer(
    name="cosmos2",
    help="Cosmos2 transfer workflow contracts.",
    no_args_is_help=True,
)


#: Compatibility alias; the workbench implementation owns the canonical name.
MANIFEST_FILENAME = TRANSFER_MANIFEST_FILENAME


def _publish_manifest(client: Any, payload: dict, output_uri: str) -> str:
    """Upload the stage manifest next to the augmented clip and return its URI."""

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="npa-cosmos2-") as tmp:
        local = Path(tmp) / MANIFEST_FILENAME
        local.write_bytes(_manifest_bytes(payload))
        return client.upload_file(str(local), transfer_manifest_uri_for(output_uri))


def _manifest_bytes(payload: dict) -> bytes:
    """Return the canonical manifest serialization used by every backend."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_output_manifest(payload: dict, output_uri: str) -> str:
    """Publish a canonical transfer manifest for an S3 or local output prefix."""

    manifest_uri = transfer_manifest_uri_for(output_uri)
    if output_uri.strip().startswith("s3://"):
        from npa.clients.storage import StorageClient

        return _publish_manifest(StorageClient.from_environment(), payload, output_uri)

    local_output = output_uri.removeprefix("local://").removeprefix("file://")
    manifest_path = Path(local_output) / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_manifest_bytes(payload))
    return manifest_uri


def _all_augmentations(configs_uri: str) -> list[dict]:
    """Read the Config-Gen manifest and return every sampled appearance combo.

    Each combo drives one Cosmos Transfer 2.5 inference ("multiply"), so a config
    manifest with N augmentations yields N scenario variants.  A configured
    manifest is authoritative: an unreadable or empty manifest must not silently
    collapse a requested multi-variant/gang run into one default render.
    """
    uri = (
        configs_uri
        if configs_uri.endswith(".json")
        else configs_uri.rstrip("/") + "/manifest.json"
    )
    try:
        from npa.workflows.data_factory_stages import _download_json

        manifest = _download_json(uri)
        combos = manifest.get("augmentations") or []
    except Exception as exc:  # noqa: BLE001 - normalize storage/provider errors
        raise typer.BadParameter(
            f"configured augmentation manifest could not be read at {uri!r}"
        ) from exc
    valid = [c for c in combos if isinstance(c, dict)]
    if not valid:
        raise typer.BadParameter(
            f"configured augmentation manifest at {uri!r} has no augmentation objects"
        )
    return valid


def _first_augmentation(configs_uri: str) -> dict:
    """Read the Config-Gen manifest and return its first sampled combo."""
    combos = _all_augmentations(configs_uri)
    return combos[0]


_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _frames_to_conditioning_clip(frames: list[Path], output_dir: Path) -> str:
    """Encode PAIDF input frames as the short clip Cosmos Transfer consumes.

    The PAIDF first-run path intentionally seeds still frames because the
    preceding caption stage consumes images.  Cosmos Transfer consumes video,
    so its runner assembles those same frames into an ephemeral conditioning
    clip.  The clip matches the qualified procedural fixture's dimensions,
    frame rate, and frame count without copying or packaging any source media.
    """
    import shutil
    import subprocess

    if not frames:
        return ""

    sequence_dir = output_dir / "conditioning-frames"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    sequence: list[Path] = []
    for index, frame in enumerate(frames):
        suffix = frame.suffix.lower()
        if suffix not in _IMAGE_EXTS:
            continue
        normalized = sequence_dir / f"frame-{index:05d}{suffix}"
        shutil.copyfile(frame, normalized)
        sequence.append(normalized)
    if not sequence:
        return ""

    # Concat accepts mixed PNG/JPEG inputs.  All list entries are paths authored
    # above (not object-key text), and duplicating the final frame makes its
    # duration effective under the concat demuxer.
    concat_file = output_dir / "conditioning-frames.ffconcat"
    lines = ["ffconcat version 1.0"]
    for frame in sequence:
        lines.extend((f"file '{frame}'", "duration 0.5"))
    lines.append(f"file '{sequence[-1]}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = output_dir / "npa-paidf-conditioning.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            (
                "fps=16,tpad=stop_mode=clone:stop_duration=8,"
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
            ),
            "-frames:v",
            "93",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("FFmpeg did not produce a PAIDF conditioning clip")
    typer.echo(
        "PAIDF conditioning: encoded "
        f"{len(sequence)} input frame(s) as a 1280x720, 93-frame clip",
        err=True,
    )
    return str(output)


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


def _gang_shard() -> tuple[int, int]:
    """Return this worker's ``(rank, node_count)`` in the augment block.

    A spec asks for a multi-node augment with ``resources.gpu.num_nodes``; SkyPilot
    then gang-schedules that many identical pods for the one task and exports
    ``SKYPILOT_NODE_RANK`` / ``SKYPILOT_NUM_NODES`` into each. Every pod runs this
    same command, so without a shard the gang would render every variant N times.
    ``NPA_COSMOS_NODE_RANK`` / ``NPA_COSMOS_NODE_COUNT`` override for local runs.

    An inconsistent identity fails closed: silently collapsing to one node would
    duplicate GPU work and leave the run manifest reporting a fan-out that never
    happened.
    """

    rank, nodes, _attempt_id = _gang_identity()
    return rank, nodes


def _gang_identity() -> tuple[int, int, str]:
    """Return rank, authoritative gang size, and one shared attempt identity.

    ``NPA_COSMOS_NODE_COUNT`` comes from the workflow renderer and is the source
    of truth.  SkyPilot 0.12.2 independently supplies count, rank, ordered node
    IPs, and an internal job id.  All are checked before a worker can publish a
    shard.  ``SKYPILOT_TASK_ID`` is deliberately excluded: SkyPilot preserves it
    across managed-job recoveries.
    """

    def _read(name: str) -> str:
        return str(os.environ.get(name, "")).strip()

    raw_nodes = _read("NPA_COSMOS_NODE_COUNT")
    raw_local_rank = _read("NPA_COSMOS_NODE_RANK")
    sky_nodes = _read("SKYPILOT_NUM_NODES")
    sky_rank = _read("SKYPILOT_NODE_RANK")
    sky_ips = _read("SKYPILOT_NODE_IPS")
    sky_internal_job = _read("SKYPILOT_INTERNAL_JOB_ID")
    sky_managed_job = _read("SKYPILOT_MANAGED_JOB_ID")
    base_attempt = _read("NPA_WORKFLOW_ATTEMPT_ID")
    local_attempt = _read("NPA_COSMOS_ATTEMPT_ID")
    sky_evidence = any((sky_nodes, sky_rank, sky_ips, sky_internal_job))
    if not raw_nodes:
        if sky_evidence:
            raise typer.BadParameter(
                "multi-node augment identity is missing authoritative "
                "NPA_COSMOS_NODE_COUNT"
            )
        return 0, 1, ""
    try:
        nodes = int(raw_nodes)
    except ValueError as exc:
        raise typer.BadParameter(
            "multi-node augment identity is not numeric "
            f"(authoritative node count {raw_nodes!r})"
        ) from exc
    if sky_evidence:
        missing = [
            name
            for name, value in (
                ("SKYPILOT_NUM_NODES", sky_nodes),
                ("SKYPILOT_NODE_RANK", sky_rank),
                ("SKYPILOT_NODE_IPS", sky_ips),
                ("SKYPILOT_INTERNAL_JOB_ID", sky_internal_job),
                ("NPA_WORKFLOW_ATTEMPT_ID", base_attempt),
            )
            if not value
        ]
        if missing:
            raise typer.BadParameter(
                "multi-node augment identity is incomplete: missing "
                + ", ".join(missing)
            )
        try:
            observed_nodes = int(sky_nodes)
            rank = int(sky_rank)
        except ValueError as exc:
            raise typer.BadParameter(
                "multi-node augment identity is not numeric "
                f"(SkyPilot node count {sky_nodes!r}, rank {sky_rank!r})"
            ) from exc
        if observed_nodes != nodes:
            raise typer.BadParameter(
                "multi-node augment identity is contradictory: renderer requested "
                f"{nodes} node(s), SkyPilot reported {observed_nodes}"
            )
        if raw_local_rank and raw_local_rank != sky_rank:
            raise typer.BadParameter(
                "multi-node augment identity is contradictory: "
                f"NPA rank {raw_local_rank!r}, SkyPilot rank {sky_rank!r}"
            )
        node_ips = [line.strip() for line in sky_ips.splitlines() if line.strip()]
        if len(node_ips) != nodes or len(set(node_ips)) != nodes:
            raise typer.BadParameter(
                "multi-node augment identity is contradictory: "
                f"SKYPILOT_NODE_IPS has {len(node_ips)} unique member(s), "
                f"expected {nodes}"
            )
        attempt_material = json.dumps(
            {
                "base": base_attempt,
                "internal_job": sky_internal_job,
                "managed_job": sky_managed_job,
                "node_count": nodes,
                "node_ips": node_ips,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        attempt_id = hashlib.sha256(attempt_material).hexdigest()
    else:
        raw_rank = raw_local_rank or "0"
        try:
            rank = int(raw_rank)
        except ValueError as exc:
            raise typer.BadParameter(
                "multi-node augment identity is not numeric "
                f"(node count {raw_nodes!r}, rank {raw_rank!r})"
            ) from exc
        if nodes > 1 and not local_attempt:
            raise typer.BadParameter(
                "multi-node augment identity is incomplete: a local gang requires "
                "NPA_COSMOS_ATTEMPT_ID"
            )
        attempt_id = local_attempt
    if nodes < 1 or not 0 <= rank < nodes:
        raise typer.BadParameter(
            f"multi-node augment identity is inconsistent: rank {rank} of {nodes} node(s)"
        )
    return rank, nodes, attempt_id


def _shard_indices(count: int, *, rank: int, nodes: int) -> list[int]:
    """Variant indices this node renders, striding so the load stays balanced.

    Striding (rank, rank+nodes, ...) rather than contiguous blocks keeps every
    node within one variant of the others when the count does not divide evenly.
    """

    if nodes <= 1:
        return list(range(max(0, count)))
    return list(range(rank, max(0, count), nodes))


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


def _materialize_input_clip(src: str, *, allow_frame_sequence: bool = False) -> str:
    """Resolve a local path or ``s3://`` URI to a local conditioning video.

    Returns an empty string only when the source was successfully inspected and no
    supported input exists. In the PAIDF path only, ``allow_frame_sequence`` turns
    the captionable input frames into a temporary video. Storage setup, listing,
    authentication, download, and encoding failures propagate so the CLI can
    report them separately from an empty prefix.
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
            # PAIDF prepares the exact normalized model input under this name.
            return next(
                (video for video in vids if Path(video).name == "conditioning.mp4"),
                vids[0],
            )
        if allow_frame_sequence:
            frames = sorted(
                Path(f)
                for f in _glob.glob(str(Path(tmp) / "**" / "*"), recursive=True)
                if f.lower().endswith(_IMAGE_EXTS) and Path(f).is_file()
            )
            clip = _frames_to_conditioning_clip(frames, Path(tmp))
            if clip:
                keep_tmp = True
                return clip
        return ""
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _materialize_control_asset(src: str, *, label: str) -> str:
    """Resolve a precomputed control/mask video to a local path.

    Unlike the conditioning input, an asset the operator explicitly named must
    exist: silently continuing would compute the control on-the-fly instead and
    the run would look like it honoured the asset.
    """

    value = str(src or "").strip()
    if not value:
        return ""
    if not value.lower().endswith(_VIDEO_EXTS):
        raise typer.BadParameter(
            f"{label} must be an mp4/video file, got: {value!r}"
        )
    if not value.startswith("s3://"):
        if not Path(value).is_file():
            raise typer.BadParameter(f"{label} does not exist: {value!r}")
        return value
    import atexit
    import shutil
    import tempfile
    from urllib.parse import urlsplit

    from npa.clients.storage import StorageClient

    tmp = tempfile.mkdtemp(prefix="npa-cosmos-control-")

    def cleanup() -> None:
        shutil.rmtree(tmp, ignore_errors=True)

    # The downloaded asset must remain available for every variant in this CLI
    # process, including concurrent renders. Remove it when the process exits;
    # also clean it immediately when the download itself fails.
    atexit.register(cleanup)
    name = Path(urlsplit(value).path).name or "control.mp4"
    try:
        return StorageClient.from_environment().download_path(value, str(Path(tmp) / name))
    except Exception as exc:  # noqa: BLE001 - sanitize storage failures
        cleanup()
        atexit.unregister(cleanup)
        raise typer.BadParameter(
            f"could not download {label} from {value!r}; verify the object-storage "
            "endpoint, credentials, permissions, and that the object exists"
        ) from exc


def _materialize_conditioning_input(
    src: str, *, allow_frame_sequence: bool = False
) -> str:
    """Adapt storage failures to a sanitized, actionable CLI error."""
    try:
        if allow_frame_sequence:
            return _materialize_input_clip(src, allow_frame_sequence=True)
        return _materialize_input_clip(src)
    except Exception as exc:
        raise typer.BadParameter(
            "could not inspect or download the configured conditioning input; "
            "verify the object-storage endpoint, credentials, permissions, and availability"
        ) from exc


def _persist_generated_conditioning_clip(
    local_input: str, input_uri: str, *, publish: bool = True
) -> str:
    """Persist PAIDF's frame-derived clip so evaluation uses the exact source.

    Operator-side preparation already persists ``conditioning.mp4``. The legacy
    fixture path still creates ``npa-paidf-conditioning.mp4`` in the worker and
    needs it published. In both cases return the canonical URI so evaluation
    records the exact clip Cosmos consumed.

    ``publish=False`` resolves the URI without writing: in a multi-node augment
    every node derives the same clip, so only one of them uploads it.
    """

    path = Path(str(local_input or ""))
    if not input_uri.startswith("s3://"):
        return ""
    uri = input_uri.rstrip("/") + "/conditioning.mp4"
    if path.name == "conditioning.mp4":
        return uri
    if path.name != "npa-paidf-conditioning.mp4":
        return ""
    if not publish:
        return uri
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment().upload_file(str(path), uri)


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
        help="Control modality for input-conditioning: edge, vis, depth, or seg. "
        "Edge/vis/seg can be derived from the input; depth requires an "
        "operator-owned precomputed control and never invokes Video Depth Anything.",
    ),
    control_weight: float = typer.Option(1.0, "--control-weight", help="Control weight for input-conditioning."),
    control_asset: str = typer.Option(
        "",
        "--control-asset",
        help="Local path or s3:// URI of a PRECOMPUTED control video (e.g. a "
        "segmentation map) to condition on instead of computing the modality "
        "on-the-fly.",
    ),
    control_prompt: str = typer.Option(
        "",
        "--control-prompt",
        help="Objects on-the-fly 'seg' should segment (e.g. 'robot arm, conveyor, "
        "bin'). Passed to GroundingDINO to seed SAM2 tracking; upstream defaults to "
        "the first 128 words of the appearance prompt when unset.",
    ),
    mask_asset: str = typer.Option(
        "",
        "--mask-asset",
        help="Local path or s3:// URI of a PRECOMPUTED binary spatiotemporal region "
        "mask. The control applies only where the mask is white. Mutually exclusive "
        "with --mask-prompt.",
    ),
    mask_prompt: str = typer.Option(
        "",
        "--mask-prompt",
        help="Objects SAM2 should segment into a region mask, restricting the control "
        "to those pixels (e.g. 'robot arm'). Mutually exclusive with --mask-asset.",
    ),
    control_output_uri: str = typer.Option(
        "",
        "--control-output-uri",
        help="s3:// prefix to publish the control map and region mask that "
        "conditioned each variant, as <prefix>/<clip>/control_<modality>.mp4 plus "
        "extracted frames. Sibling of --output-uri, never nested inside it.",
    ),
    guidance: float = typer.Option(3.0, "--guidance", help="Classifier-free guidance for input-conditioning."),
) -> None:
    """Build a transfer manifest; pass --execute for real vendor output.

    Mode is chosen by runtime availability, not just the flag: if the
    Cosmos-Transfer2.5 runtime is present (or ``--execute`` is passed) the real
    world-transfer model runs and publishes a video; otherwise a genuine
    reference augmentation writes real augmented image frames. Inspect
    ``output_kind`` in the manifest ("video" vs "frames") to disambiguate.
    """

    # Resolve every deterministic control knob before probing the runtime or
    # touching input/control storage.  The same import-light validator is used
    # by workflow validate/plan/submit.
    control = os.environ.get("NPA_COSMOS_CONTROL", "").strip() or control
    control_asset = (
        os.environ.get("NPA_COSMOS_CONTROL_ASSET", "").strip() or control_asset
    )
    control_prompt = (
        os.environ.get("NPA_COSMOS_CONTROL_PROMPT", "").strip() or control_prompt
    )
    mask_asset = os.environ.get("NPA_COSMOS_MASK_ASSET", "").strip() or mask_asset
    mask_prompt = os.environ.get("NPA_COSMOS_MASK_PROMPT", "").strip() or mask_prompt
    raw_control_weight = os.environ.get("NPA_COSMOS_CONTROL_WEIGHT", "").strip()
    raw_guidance = os.environ.get("NPA_COSMOS_GUIDANCE", "").strip()
    requested_control_weight: object = control_weight
    if raw_control_weight:
        requested_control_weight = raw_control_weight
    if raw_guidance:
        try:
            guidance = float(raw_guidance)
        except ValueError as exc:
            raise typer.BadParameter(
                "NPA_COSMOS_GUIDANCE must be a finite number"
            ) from exc
        if not math.isfinite(guidance):
            raise typer.BadParameter("NPA_COSMOS_GUIDANCE must be a finite number")
    from npa.workbench.cosmos.control_contract import (
        ControlContractError,
        validate_control_request,
    )

    try:
        checkpoint, normalized_weight = validate_control_request(
            modality=control,
            weight=requested_control_weight,
            control_asset=control_asset,
            control_prompt=control_prompt,
            mask_asset=mask_asset,
            mask_prompt=mask_prompt,
        )
    except ControlContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    control = checkpoint.modality
    control_weight = normalized_weight

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
        # that data_factory curate / build_run_rrd / provenance consume. Generic
        # callers opt in via --input-video, --condition-on-input, or
        # NPA_COSMOS_CONDITION_ON_INPUT=1.
        #
        # Otherwise (generic `transfer` for sim2real / cosmos-gate / fanout), publish
        # the generated video, flat extracted frames, and durable manifest together.
        condition_requested = bool(
            input_video or condition_on_input or _env_truthy("NPA_COSMOS_CONDITION_ON_INPUT")
        )
        data_factory_mode = bool(configs_uri)
        local_input = ""
        if condition_requested:
            local_input = _materialize_conditioning_input(
                input_video or input_uri,
                # PAIDF Config-Gen produces/captions image frames. If its input
                # prefix has no video, condition Cosmos on a temporary clip made
                # from those frames. Generic/standalone transfer remains strict.
                allow_frame_sequence=bool(configs_uri),
            )
            if not local_input:
                expected = (
                    "supported video or PAIDF PNG/JPEG input frames"
                    if configs_uri
                    else "supported video"
                )
                raise typer.BadParameter(
                    f"input conditioning was requested, but no {expected} "
                    f"({', '.join(_VIDEO_EXTS)}) was found at the configured input"
                )
        control_asset = _materialize_control_asset(control_asset, label="--control-asset")
        mask_asset = _materialize_control_asset(mask_asset, label="--mask-asset")

        if data_factory_mode and output_uri.strip().startswith("s3://"):
            # Augment & MULTIPLY. Run one REAL Cosmos Transfer 2.5 inference per
            # sampled appearance combo (each with its own prompt), publishing each
            # as its own per-clip dir under the cosmos_augmented/ prefix, then write
            # a single run-level manifest.json listing them all. A config manifest
            # with N augmentations therefore yields N scenario variants (not one
            # image). The per-clip layout is what data_factory curate /
            # build_run_rrd / provenance consume.
            from npa.workbench.cosmos.transfer import (
                merge_shard_manifests,
                publish_transfer_clip,
                write_run_manifest,
                write_shard_manifest,
            )

            combos = _all_augmentations(configs_uri)

            # Multi-node fan-out: this node renders only its stride of the sampled
            # combos. Variant indices stay GLOBAL, so clip names remain disjoint
            # across the gang and the merged manifest keeps the sampled order.
            rank, node_count, attempt_id = _gang_identity()
            shard = [(i, combos[i]) for i in _shard_indices(len(combos), rank=rank, nodes=node_count)]

            conditioning_clip_uri = _persist_generated_conditioning_clip(
                local_input,
                input_uri,
                # One writer for a key the whole gang would otherwise race on.
                publish=rank == 0,
            )

            parallelism = _variant_parallelism(len(shard))

            def _render_variant(slot: int, i: int, combo: dict) -> dict:
                variant_run = f"{run_id}-v{i}" if run_id else f"v{i}"
                # Pin each concurrent variant to a distinct GPU so an N-GPU pod
                # runs N diffusions at once (sequential when parallelism == 1).
                # The device comes from the node-local slot, never the global
                # variant index: rank 1 of a gang must still start at GPU 0.
                device = str(slot % parallelism) if parallelism > 1 else None
                result = run_cosmos_transfer(
                    run_id=variant_run,
                    spec=spec or None,
                    prompt=str(combo.get("prompt") or "") or None,
                    input_video=local_input or None,
                    control=control,
                    control_weight=control_weight,
                    control_asset=control_asset,
                    control_prompt=control_prompt,
                    mask_asset=mask_asset,
                    mask_prompt=mask_prompt,
                    guidance=guidance,
                    cuda_visible_devices=device,
                    variant_tag=variant_run,
                )
                result["conditioning_clip_uri"] = conditioning_clip_uri
                return result

            # Fan the GPU-bound diffusions out across this pod's GPUs, then publish
            # sequentially in combo order (publish/S3 upload stays single-threaded).
            transfers: dict[int, dict] = {}
            if parallelism > 1 and len(shard) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=parallelism) as pool:
                    futures = {
                        pool.submit(_render_variant, slot, i, combo): i
                        for slot, (i, combo) in enumerate(shard)
                    }
                    for future in futures:
                        transfers[futures[future]] = future.result()
            else:
                for slot, (i, combo) in enumerate(shard):
                    transfers[i] = _render_variant(slot, i, combo)

            clips: list[dict] = []
            for i, combo in shard:
                clip_name = f"aug-{run_id}-{i}" if run_id else f"aug{i}"
                clips.append(
                    publish_transfer_clip(
                        transfers[i],
                        output_uri,
                        run_id=run_id,
                        clip_name=clip_name,
                        variables=combo,
                        variant_index=i,
                        control_output_uri=control_output_uri,
                        require_frames=True,
                    )
                )
            if node_count > 1:
                # Each node publishes its own shard manifest; rank 0 joins them into
                # the single run manifest the downstream stages read. A worker's
                # payload describes its own shard -- it never claims the run's total.
                from npa.workbench.cosmos.transfer import build_run_manifest

                write_shard_manifest(
                    clips,
                    output_uri,
                    run_id=run_id,
                    rank=rank,
                    node_count=node_count,
                    variant_parallelism=parallelism,
                    variant_total=len(combos),
                    attempt_id=attempt_id,
                )
                manifest = (
                    merge_shard_manifests(
                        output_uri,
                        run_id=run_id,
                        node_count=node_count,
                        attempt_id=attempt_id,
                    )
                    if rank == 0
                    else build_run_manifest(
                        clips,
                        run_id=run_id,
                        variant_parallelism=parallelism,
                        node_count=node_count,
                        attempt_id=attempt_id,
                    )
                )
            else:
                manifest = write_run_manifest(
                    clips, output_uri, run_id=run_id, variant_parallelism=parallelism
                )
            payload["status"] = TRANSFER_MANIFEST_STATUS
            payload["output_kind"] = "video"
            payload["mode"] = TRANSFER_MANIFEST_MODE
            payload["augmented_video_uri"] = manifest["augmented_video_uri"]
            payload["augmented_videos"] = manifest["augmented_videos"]
            payload["frame_count"] = manifest["frame_count"]
            payload["variant_count"] = manifest["variant_count"]
            payload["multiply_mode"] = manifest["multiply_mode"]
            payload["variant_parallelism"] = manifest["variant_parallelism"]
            payload["node_count"] = node_count
            payload["node_rank"] = rank
            payload["shard_variant_count"] = len(clips)
            payload["clips"] = manifest["clips"]
            local_variables = [combo for _index, combo in shard]
            payload["augmentation_variables"] = local_variables
            local_prompts = [str(combo.get("prompt") or "") for combo in local_variables]
            payload["prompts"] = local_prompts
            # Retain the legacy singular field as the first prompt this worker
            # actually executed; an empty stride reports no prompt.
            payload["prompt"] = local_prompts[0] if local_prompts else ""
            payload["attempt_id"] = attempt_id
            payload["input_conditioned"] = bool(local_input)
            payload["conditioning_clip_uri"] = manifest.get("conditioning_clip_uri", "")
            payload["control_spec"] = manifest["control_spec"]
            payload["control_weight"] = manifest["control_weight"]
            payload["control_prompt"] = manifest["control_prompt"]
            payload["mask_prompt"] = manifest["mask_prompt"]
            payload["control_uris"] = manifest["control_uris"]
            if control_output_uri:
                payload["control_output_uri"] = control_output_uri
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
                control_asset=control_asset,
                control_prompt=control_prompt,
                mask_asset=mask_asset,
                mask_prompt=mask_prompt,
                guidance=guidance,
            )
            payload["status"] = TRANSFER_MANIFEST_STATUS
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
                # Frame objects are deliberately flat under output_uri because envgen
                # constructs exactly <augment_uri>/frame-NNNNN.png references.
                from npa.workbench.cosmos.transfer import publish_transfer_to_s3

                manifest = publish_transfer_to_s3(
                    transfer,
                    output_uri,
                    run_id=run_id,
                    variables=variables,
                    frames_output_uri=output_uri,
                    control_output_uri=control_output_uri,
                    require_frames=True,
                )
                payload["mode"] = TRANSFER_MANIFEST_MODE
                payload["output_video"] = manifest["augmented_video_uri"]
                payload["augmented_video_uri"] = manifest["augmented_video_uri"]
                payload["augmented_frames_uri"] = manifest["augmented_frames_uri"]
                payload["frame_count"] = manifest["frame_count"]
                payload["control_uris"] = manifest.get("control_uris", {})
                payload["manifest_uri"] = transfer_manifest_uri_for(output_uri)
            else:
                payload["mode"] = TRANSFER_MANIFEST_MODE
                payload["augmented_video_uri"] = transfer["video_path"]
                payload["augmented_frames_uri"] = output_uri
    else:
        # No heavy model runtime: run a genuine reference augmentation that
        # writes real augmented image frames to output_uri (not a descriptor stub).
        augment = reference_augment_frames(input_uri, output_uri, run_id=run_id)
        payload["status"] = REFERENCE_AUGMENT_STATUS
        payload["mode"] = REFERENCE_AUGMENT_MODE
        payload["output_kind"] = "frames"
        payload["augmented_frames_uri"] = augment["augmented_frames_uri"]
        payload["frames"] = augment["frames"]
        payload["frame_count"] = augment["frame_count"]
        payload["index_uri"] = augment["index_uri"]
        payload["manifest_uri"] = transfer_manifest_uri_for(output_uri)
        _publish_output_manifest(payload, output_uri)

    if output_json is not None:
        payload = write_manifest(payload, output_json)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
