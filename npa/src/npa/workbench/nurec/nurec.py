"""NuRec / NRE (NVIDIA Neural Reconstruction Engine) reconstruction helpers.

Pure, framework-free logic for the ``neural-reconstruction`` capability: turn a
real sensor recording in NCore V4 form into a 3D Gaussian reconstruction
(3DGUT), a renderable USDZ artifact, and novel-view renders.

The real work is done by NVIDIA's public NRE container
(``nvcr.io/nvidia/nre/nre-ga``), whose entrypoint IS the NRE CLI. This module
never invents a command: it builds the documented argv for the container's
``train`` (Hydra app), ``render``, and ``export-ncore-benchmark-gt``
sub-commands and shells out through an **injectable runner**, so every code
path is unit-testable without Docker, a GPU, NGC, or Hugging Face.

Two execution shapes are supported by the same argv builders:

* **in-container** (default) — the NPA task already runs inside the NRE image,
  so we invoke ``/app/run`` directly. This is what the SkyPilot workflow does.
* **docker host** — set ``docker_bin`` (``NPA_NUREC_DOCKER``) and the command is
  wrapped in ``docker run --gpus all --shm-size=64g …`` with the dataset/output
  bind mounts, matching the upstream cookbook recipes.

Attribution: the container names, Hydra override surface, and sub-command flags
mirror NVIDIA's public NuRec skills (https://github.com/NVIDIA/nurec-skills,
Apache-2.0 AND CC-BY-4.0). See ``skills/NOTICE-NVIDIA-SKILLS``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Container / upstream coordinates -------------------------------------------------
# The `-ga` repositories are the General Availability channel of the same images
# (upstream references/install.md). They are the ones a standard NGC API key can
# pull; `nvcr.io/nvidia/nre/nre` requires an extra entitlement.
DEFAULT_NRE_IMAGE = "nvcr.io/nvidia/nre/nre-ga:26.04"
DEFAULT_NRE_TOOLS_IMAGE = "nvcr.io/nvidia/nre/nre-tools-ga:26.04"
DEFAULT_NRE_ENTRYPOINT = "/app/run"
DEFAULT_NRE_REGISTRY_HOST = "nvcr.io"

# --- Input dataset ---------------------------------------------------------------------
# PhysicalAI-NuRec-PPISP is the ungated (CC-BY-4.0) PhysicalAI dataset that ships
# real photographic captures already in NCore V4 — the format NRE consumes — so a
# reconstruction from it is genuine, not synthetic filler. Every coordinate is
# configuration; nothing here is a hardcoded requirement.
DEFAULT_DATASET_ID = "nvidia/PhysicalAI-NuRec-PPISP"
DEFAULT_SCENE = "struktur28"
DEFAULT_VARIANT = "auto"
DEFAULT_NCORE_MEMBER_TEMPLATE = "ncore/{scene}_ncore.zip"
DEFAULT_COLMAP_MEMBER_TEMPLATE = "colmap/{scene}_colmap.zip"

# --- Local layout ----------------------------------------------------------------------
DEFAULT_CACHE_DIR = "/tmp/npa-nurec-cache"
DEFAULT_OUT_DIR = "/tmp/npa-nurec-out"
# Force a deterministic NRE <RUN-ID> sub-directory so downstream stages can find
# the artifacts without globbing a random hash (upstream: `logger.run_id`, which
# both the tensorboard and dummy logger configs expose).
DEFAULT_NRE_RUN_ID = "nre"

# --- Training defaults -----------------------------------------------------------------
# `configs/experimental/3dgut/3dgut_colmap.yaml` is the container's recipe for a
# STATIC, CAMERA-ONLY, object-centric capture reconstructed from an SfM point
# cloud — exactly the shape of a PPISP sequence. It composes
# `/apps/AV/_base_3dgut_static.yaml` + the MCMC mixin + `options/artifact: default`
# (which is what turns on `checkpoint.artifact.enabled`, i.e. the renderable USDZ)
# and disables difix/mesh/ground/nrend. Verified present in nre-ga 26.04.
DEFAULT_CONFIG_NAME = "configs/experimental/3dgut/3dgut_colmap.yaml"
DEFAULT_MODE = "trainval"
#: 0 = keep the recipe's own ``trainer.max_epochs`` (the colmap recipe ships 1).
DEFAULT_MAX_EPOCHS = 0
DEFAULT_WORLD_SIZE = 1
DEFAULT_PRECISION = ""
DEFAULT_LOGGER = "tensorboard"
#: Sentinel for ``--lidar-ids``: force an EMPTY Hydra list instead of leaving the
#: recipe's value alone. Photo-only captures have no LiDAR, but the colmap recipe
#: deliberately declares a ``dummy_lidar`` that its point-cloud initialization
#: references, so blanking it must be opt-in.
NO_LIDAR_SENTINEL = "none"

# --- Render defaults -------------------------------------------------------------------
DEFAULT_IMAGE_SCALE = 1.0
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_FRAME_NAMING = "contiguous-output-index"
# `default` uses the artifact's own trained renderer and always works. `nrend`
# (the fast C++/CUDA path) needs the nrend model dictionary embedded in the USDZ,
# which the object-centric recipe deliberately disables
# (options/nrend: disabled), so it is an opt-in rather than the default.
DEFAULT_RENDERER = "default"
DEFAULT_FRAME_STEP = 1
DEFAULT_VIDEO_FPS = 30.0
DEFAULT_VIDEO_CRF = 20
# A novel view is a view the reconstruction was NOT trained on. `nre render`
# defaults to --replicate-training-views (i.e. re-rendering seen views), so a rig
# offset plus --no-replicate-training-views is what makes the output genuinely
# novel. Upstream takes the offsets as THREE floats, not a comma string.
DEFAULT_RIG_TRANSLATION_OFFSET = (0.0, 0.25, 0.0)
DEFAULT_RIG_ROTATION_OFFSET = (0.0, 0.0, 0.0)
#: `export-ncore-benchmark-gt` defaults to every 50th camera frame; a lower step
#: gives the Rerun recording a denser strip of real capture frames.
DEFAULT_GT_FRAME_STEP_CAMERA = 10

DEFAULT_SHM_SIZE = "64g"
DEFAULT_MAX_INPUT_FRAMES = 24

DEFAULT_HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_NGC_API_KEY_ENV = "NGC_API_KEY"

INPUT_FRAME_SOURCES = ("gt", "colmap", "none")

#: Datacenter compute GPUs that have **no RT cores**. Reconstruction and
#: rasterization are RT-core work, so routing NuRec at these is a configuration
#: error (see skills/atomic/gpu-selection/SKILL.md).
NON_RT_CORE_GPU_TOKENS = (
    "a100",
    "h100",
    "h200",
    "h20",
    "gh200",
    "b100",
    "b200",
    "gb200",
    "v100",
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

RunCallable = Callable[..., "subprocess.CompletedProcess[str]"]

_logger = logging.getLogger(__name__)


class NurecError(RuntimeError):
    """Raised when NuRec/NRE access, fetch, reconstruction, or render setup fails."""


def has_rt_cores(gpu_name: str) -> bool:
    """Return True when ``gpu_name`` looks like an RT-core-capable GPU.

    NuRec rasterizes Gaussians through RT cores, so L40S / L40 / L20 / RTX PRO
    6000 / RTX A6000 / A40 / A10 are valid targets while A100 / H100 / H200 /
    B200 are not. Unknown names are treated as capable (the deny-list is the
    thing we are confident about); ``check`` reports the resolved name either
    way so an operator can see what was used.
    """
    lowered = str(gpu_name or "").lower().replace("-", " ")
    return not any(
        token in lowered.replace(" ", "") for token in NON_RT_CORE_GPU_TOKENS
    )


def _env_bool(value: str, *, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _env_int(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _env_float(value: str, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_offset(
    value: Any, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Parse ``"x,y,z"`` (or a 3-sequence) into the triple ``nre render`` expects.

    ``--rig-translation-offset`` / ``--rig-rotation-offset`` are ``FLOAT...``
    (nargs=3) options upstream, so the CLI accepts one operator-friendly string
    and this turns it into three argv values.
    """
    if value is None or value == "":
        return default
    if isinstance(value, (tuple, list)):
        parts = [str(item) for item in value]
    else:
        parts = [part for part in str(value).replace(" ", ",").split(",") if part]
    if len(parts) != 3:
        raise NurecError(
            f"expected three comma-separated floats (x,y,z), got {value!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise NurecError(f"offset components must be numeric: {value!r}") from exc


@dataclass(frozen=True)
class NurecConfig:
    """Runtime configuration for a NuRec reconstruction.

    Every field is overridable by an explicit argument first, then an
    ``NPA_NUREC_*`` environment variable, then a documented default. No bucket,
    registry, tenant, or GPU identity is baked in.
    """

    image: str = DEFAULT_NRE_IMAGE
    entrypoint: str = DEFAULT_NRE_ENTRYPOINT
    docker_bin: str = ""
    dataset_id: str = DEFAULT_DATASET_ID
    scene: str = DEFAULT_SCENE
    variant: str = DEFAULT_VARIANT
    cache_dir: Path | str = DEFAULT_CACHE_DIR
    out_dir: Path | str = DEFAULT_OUT_DIR
    nre_run_id: str = DEFAULT_NRE_RUN_ID
    config_name: str = DEFAULT_CONFIG_NAME
    ffmpeg_exe: str = ""
    poses_component_group: str = ""
    reference_camera: str = ""
    mode: str = DEFAULT_MODE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    world_size: int = DEFAULT_WORLD_SIZE
    precision: str = DEFAULT_PRECISION
    logger: str = DEFAULT_LOGGER
    aux_data: bool = False
    camera_ids: tuple[str, ...] = ()
    lidar_ids: tuple[str, ...] = ()
    input_frames_source: str = "gt"
    max_input_frames: int = DEFAULT_MAX_INPUT_FRAMES
    shm_size: str = DEFAULT_SHM_SIZE
    hf_token_env: str = DEFAULT_HF_TOKEN_ENV
    ngc_api_key_env: str = DEFAULT_NGC_API_KEY_ENV
    extra_overrides: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        image: str = "",
        entrypoint: str = "",
        docker_bin: str = "",
        dataset_id: str = "",
        scene: str = "",
        variant: str = "",
        cache_dir: Path | str | None = None,
        out_dir: Path | str | None = None,
        nre_run_id: str = "",
        config_name: str = "",
        ffmpeg_exe: str = "",
        poses_component_group: str = "",
        reference_camera: str = "",
        mode: str = "",
        max_epochs: int | None = None,
        world_size: int | None = None,
        precision: str = "",
        logger: str = "",
        aux_data: bool | None = None,
        camera_ids: Sequence[str] = (),
        lidar_ids: Sequence[str] = (),
        input_frames_source: str = "",
        max_input_frames: int | None = None,
        shm_size: str = "",
        hf_token_env: str = "",
        ngc_api_key_env: str = "",
        extra_overrides: Sequence[str] = (),
    ) -> "NurecConfig":
        """Resolve config from explicit values first, then ``NPA_NUREC_*`` env."""
        env = environ if environ is not None else os.environ
        resolved_source = (
            (
                input_frames_source
                or env.get("NPA_NUREC_INPUT_FRAMES_SOURCE", "")
                or "gt"
            )
            .strip()
            .lower()
        )
        if resolved_source not in INPUT_FRAME_SOURCES:
            raise NurecError(
                f"input_frames_source must be one of {INPUT_FRAME_SOURCES}, got {resolved_source!r}"
            )
        return cls(
            image=image or env.get("NPA_NUREC_IMAGE", "") or DEFAULT_NRE_IMAGE,
            entrypoint=(
                entrypoint
                or env.get("NPA_NUREC_ENTRYPOINT", "")
                or DEFAULT_NRE_ENTRYPOINT
            ),
            docker_bin=docker_bin or env.get("NPA_NUREC_DOCKER", ""),
            dataset_id=dataset_id
            or env.get("NPA_NUREC_DATASET", "")
            or DEFAULT_DATASET_ID,
            scene=scene or env.get("NPA_NUREC_SCENE", "") or DEFAULT_SCENE,
            variant=(variant or env.get("NPA_NUREC_VARIANT", "") or DEFAULT_VARIANT),
            cache_dir=cache_dir or env.get("NPA_NUREC_CACHE", "") or DEFAULT_CACHE_DIR,
            out_dir=out_dir or env.get("NPA_NUREC_OUT", "") or DEFAULT_OUT_DIR,
            nre_run_id=(
                nre_run_id or env.get("NPA_NUREC_NRE_RUN_ID", "") or DEFAULT_NRE_RUN_ID
            ),
            config_name=(
                config_name
                or env.get("NPA_NUREC_CONFIG_NAME", "")
                or DEFAULT_CONFIG_NAME
            ),
            ffmpeg_exe=ffmpeg_exe or env.get("NPA_NUREC_FFMPEG_EXE", ""),
            poses_component_group=(
                poses_component_group or env.get("NPA_NUREC_POSES_COMPONENT_GROUP", "")
            ),
            reference_camera=reference_camera
            or env.get("NPA_NUREC_REFERENCE_CAMERA", ""),
            mode=mode or env.get("NPA_NUREC_MODE", "") or DEFAULT_MODE,
            max_epochs=(
                max_epochs
                if max_epochs is not None
                else _env_int(env.get("NPA_NUREC_MAX_EPOCHS", ""), DEFAULT_MAX_EPOCHS)
            ),
            world_size=(
                world_size
                if world_size is not None
                else _env_int(env.get("NPA_NUREC_WORLD_SIZE", ""), DEFAULT_WORLD_SIZE)
            ),
            precision=precision
            or env.get("NPA_NUREC_PRECISION", "")
            or DEFAULT_PRECISION,
            logger=logger or env.get("NPA_NUREC_LOGGER", "") or DEFAULT_LOGGER,
            aux_data=(
                aux_data
                if aux_data is not None
                else _env_bool(env.get("NPA_NUREC_AUX_DATA", ""), default=False)
            ),
            camera_ids=tuple(camera_ids)
            or _split_csv(env.get("NPA_NUREC_CAMERA_IDS", "")),
            lidar_ids=tuple(lidar_ids)
            or _split_csv(env.get("NPA_NUREC_LIDAR_IDS", "")),
            input_frames_source=resolved_source,
            max_input_frames=(
                max_input_frames
                if max_input_frames is not None
                else _env_int(
                    env.get("NPA_NUREC_MAX_INPUT_FRAMES", ""), DEFAULT_MAX_INPUT_FRAMES
                )
            ),
            shm_size=shm_size or env.get("NPA_NUREC_SHM_SIZE", "") or DEFAULT_SHM_SIZE,
            hf_token_env=(
                hf_token_env
                or env.get("NPA_NUREC_HF_TOKEN_ENV", "")
                or DEFAULT_HF_TOKEN_ENV
            ),
            ngc_api_key_env=(
                ngc_api_key_env
                or env.get("NPA_NUREC_NGC_API_KEY_ENV", "")
                or DEFAULT_NGC_API_KEY_ENV
            ),
            extra_overrides=tuple(extra_overrides)
            or _split_csv(env.get("NPA_NUREC_EXTRA_OVERRIDES", "")),
        )

    # --- derived paths ---------------------------------------------------------------
    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir or DEFAULT_CACHE_DIR).expanduser()

    @property
    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir or DEFAULT_OUT_DIR).expanduser()

    @property
    def scene_dir_name(self) -> str:
        """NCore sub-directory for the selected scene variant.

        PPISP archives ship ``<scene>/`` (the full exposure-bracket sequence) and
        ``<scene>_auto/`` (the auto-exposure re-processed sequence).
        """
        variant = str(self.variant or "").strip().lower()
        if variant in {"", "standard", "default", "full"}:
            return self.scene
        return f"{self.scene}_{variant}"

    @property
    def ncore_member(self) -> str:
        return DEFAULT_NCORE_MEMBER_TEMPLATE.format(scene=self.scene)

    @property
    def colmap_member(self) -> str:
        return DEFAULT_COLMAP_MEMBER_TEMPLATE.format(scene=self.scene)

    @property
    def nre_run_dir(self) -> Path:
        return self.resolved_out_dir / self.nre_run_id

    @property
    def image_repository(self) -> str:
        """``nvidia/nre/nre-ga`` for ``nvcr.io/nvidia/nre/nre-ga:26.04``."""
        ref = str(self.image or "")
        without_tag = ref.rsplit(":", 1)[0] if "/" in ref.rsplit(":", 1)[0] else ref
        parts = without_tag.split("/", 1)
        return parts[1] if len(parts) == 2 and "." in parts[0] else without_tag

    @property
    def image_registry(self) -> str:
        ref = str(self.image or "")
        head = ref.split("/", 1)[0]
        return head if ("." in head or ":" in head) else ""


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _hydra_list(values: Sequence[str]) -> str:
    """Render a Hydra list override value. Upstream requires no spaces."""
    inner = ",".join(f"'{value}'" for value in values)
    return f"[{inner}]"


# --------------------------------------------------------------------------------------
# argv builders (pure)
# --------------------------------------------------------------------------------------
def build_nre_train_args(
    config: NurecConfig,
    *,
    ncore_json: str,
    out_dir: str = "",
) -> list[str]:
    """Build the NRE training (Hydra) argv for an NCore V4 dataset.

    Mirrors the upstream cookbook train recipe: a ``--config-name`` recipe plus
    ``mode`` / ``dataset.path`` / ``out_dir`` and the checkpoint-artifact flags
    that make the run produce a renderable ``usd-out/last.usdz``.
    """
    if not ncore_json:
        raise NurecError("ncore_json is required to train a reconstruction")
    if not config.config_name:
        raise NurecError(
            "config_name is required: pass --config-name or set NPA_NUREC_CONFIG_NAME "
            "to an NRE recipe shipped in the container"
        )
    target_out = out_dir or str(config.resolved_out_dir)
    args = [
        f"--config-name={config.config_name}",
        f"mode={config.mode}",
        f"dataset.path={ncore_json}",
        f"out_dir={target_out}",
        f"logger={config.logger}",
        f"logger.run_id={config.nre_run_id}",
        # A renderable USDZ (plus rig trajectories and tracks) is the whole point:
        # `render` and `serve-grpc` both require the checkpoint artifact.
        "checkpoint.artifact.enabled=true",
        "checkpoint.artifact.checkpoint.enabled=true",
        "checkpoint.artifact.parsed_config.enabled=true",
        "checkpoint.artifact.rig_trajectories.enabled=true",
        "checkpoint.artifact.sequence_tracks.enabled=true",
        f"trainer.world_size={config.world_size}",
        f"dataset.aux_data={'true' if config.aux_data else 'false'}",
        # Report SSIM/LPIPS next to PSNR so the run carries real quality numbers.
        "system.test.metrics.ssim.enabled=true",
        "system.test.metrics.lpips.enabled=true",
    ]
    # 0 means "respect the recipe's own epoch budget" — the object-centric colmap
    # recipe ships a deliberate value and silently multiplying it by 30 would be a
    # surprising, expensive default.
    if config.max_epochs > 0:
        args.append(f"trainer.max_epochs={config.max_epochs}")
    # A derived pose set lives in its own NCore component instance; NRE selects it
    # with this override (see npa.workbench.nurec.ncore_rig).
    if config.poses_component_group:
        args.append(f"dataset.poses_component_group={config.poses_component_group}")
    if config.precision:
        args.append(f"trainer.precision={config.precision}")
    if config.camera_ids:
        args.append(f"dataset.camera_ids={_hydra_list(config.camera_ids)}")
    # LiDAR: leave the recipe alone unless the caller says otherwise. The colmap
    # recipe declares a `dummy_lidar` that its point-cloud initialization
    # references, so blanking the list has to be an explicit opt-in.
    if config.lidar_ids == (NO_LIDAR_SENTINEL,):
        args.append("dataset.lidar_ids=[]")
    elif config.lidar_ids:
        args.append(f"dataset.lidar_ids={_hydra_list(config.lidar_ids)}")
    args.extend(config.extra_overrides)
    return args


def build_nre_render_args(
    config: NurecConfig,
    *,
    artifact_path: str,
    output_dir: str,
    camera_ids: Sequence[str] = (),
    image_scale: float = DEFAULT_IMAGE_SCALE,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    frame_naming: str = DEFAULT_FRAME_NAMING,
    renderer: str = DEFAULT_RENDERER,
    frame_step: int = DEFAULT_FRAME_STEP,
    rig_translation_offset: Any = DEFAULT_RIG_TRANSLATION_OFFSET,
    rig_rotation_offset: Any = DEFAULT_RIG_ROTATION_OFFSET,
    custom_rig_trajectory: str = "",
    replicate_training_views: bool = False,
    export_video: bool = True,
    video_fps: float = DEFAULT_VIDEO_FPS,
    video_crf: int = DEFAULT_VIDEO_CRF,
) -> list[str]:
    """Build the ``nre render`` argv for novel-view rendering.

    ``nre render`` defaults to ``--replicate-training-views``, which re-renders
    views the model was trained on. ``replicate_training_views`` therefore
    defaults to **False** here and emits the explicit
    ``--no-replicate-training-views`` plus a rig offset, so what lands on disk is
    a genuine novel view rather than a reproduction of the training set.
    """
    if not artifact_path:
        raise NurecError("artifact_path (a trained .usdz) is required to render")
    if not str(artifact_path).endswith(".usdz"):
        raise NurecError(
            f"artifact_path must be a .usdz artifact, got: {artifact_path!r}"
        )
    if not output_dir:
        raise NurecError("output_dir is required to render")
    translation = parse_offset(rig_translation_offset, DEFAULT_RIG_TRANSLATION_OFFSET)
    rotation = parse_offset(rig_rotation_offset, DEFAULT_RIG_ROTATION_OFFSET)
    moved = any(translation) or any(rotation) or bool(custom_rig_trajectory)
    if not replicate_training_views and not moved:
        raise NurecError(
            "novel-view rendering needs a non-zero rig offset or a custom rig "
            "trajectory; pass --replicate-training-views to re-render the training "
            "views instead"
        )
    args = [
        "render",
        "--artifact-path",
        str(artifact_path),
        "--output-dir",
        str(output_dir),
        "--image-scale",
        f"{float(image_scale)}",
        "--image-format",
        str(image_format),
        "--frame-naming",
        str(frame_naming),
        "--renderer",
        str(renderer),
        "--frame-step",
        str(int(frame_step)),
    ]
    for camera_id in camera_ids:
        args.extend(["--camera-id", camera_id])
    if replicate_training_views:
        args.append("--replicate-training-views")
    else:
        args.append("--no-replicate-training-views")
        args.extend(
            ["--rig-translation-offset", *(f"{value}" for value in translation)]
        )
        args.extend(["--rig-rotation-offset", *(f"{value}" for value in rotation)])
        if custom_rig_trajectory:
            args.extend(["--custom-rig-trajectory", custom_rig_trajectory])
    if export_video:
        args.extend(
            [
                "--export-video",
                "--video-fps",
                f"{float(video_fps)}",
                "--video-crf",
                str(int(video_crf)),
            ]
        )
    return args


def build_nre_export_gt_args(
    *,
    ncore_json: str,
    output_dir: str,
    frame_step_camera: int = DEFAULT_GT_FRAME_STEP_CAMERA,
) -> list[str]:
    """Build ``export-ncore-benchmark-gt`` argv (real capture frames for evidence).

    Upstream selects sensors from the sequence itself — there is no ``--camera-id``
    on this sub-command — and skips to every 50th camera frame by default.
    """
    if not ncore_json:
        raise NurecError("ncore_json is required to export benchmark ground truth")
    if not output_dir:
        raise NurecError("output_dir is required to export benchmark ground truth")
    return [
        "export-ncore-benchmark-gt",
        "--dataset-path",
        str(ncore_json),
        "--output-dir",
        str(output_dir),
        "--frame-step-camera",
        str(int(frame_step_camera)),
    ]


def build_docker_wrapper(
    config: NurecConfig,
    *,
    mounts: Sequence[tuple[str, str]] = (),
    env_names: Sequence[str] = (),
) -> list[str]:
    """Wrap the NRE invocation in ``docker run`` for a plain Docker host.

    Only used when ``docker_bin`` is configured; the SkyPilot workflow runs
    *inside* the NRE image and calls the entrypoint directly.
    """
    if not config.docker_bin:
        raise NurecError("docker_bin is not configured; cannot build a docker wrapper")
    argv = [
        config.docker_bin,
        "run",
        "--rm",
        "--gpus",
        "all",
        f"--shm-size={config.shm_size}",
    ]
    for name in env_names:
        argv.extend(["-e", name])
    for host_path, container_path in mounts:
        argv.extend(["--volume", f"{host_path}:{container_path}"])
    argv.append(config.image)
    return argv


def nre_top_level_flags(config: NurecConfig) -> list[str]:
    """Flags the NRE entrypoint accepts *before* the sub-command.

    ``--ffmpeg-exe`` matters for ``render --export-video``: the image ships no
    ffmpeg, so either an apt-installed binary is named explicitly or NRE has to be
    allowed to download one into its cache.
    """
    if config.ffmpeg_exe:
        return ["--ffmpeg-exe", config.ffmpeg_exe]
    return []


def nre_command(
    config: NurecConfig,
    args: Sequence[str],
    *,
    mounts: Sequence[tuple[str, str]] = (),
    env_names: Sequence[str] = (),
) -> list[str]:
    """Full argv for one NRE invocation, in-container or via ``docker run``."""
    invocation = [*nre_top_level_flags(config), *args]
    if config.docker_bin:
        return (
            build_docker_wrapper(config, mounts=mounts, env_names=env_names)
            + invocation
        )
    if not config.entrypoint:
        raise NurecError("entrypoint is required to invoke NRE in-container")
    return [config.entrypoint, *invocation]


# --------------------------------------------------------------------------------------
# result dataclasses
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NurecCheckResult:
    """Redacted access-check result safe for CLI output and logs."""

    ok: bool
    image: str
    ngc_auth: str
    ngc_image: str
    hf_auth: str
    hf_dataset: str
    gpu: str
    rt_cores: str
    entrypoint: str
    cache_dir: str
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "image": self.image,
            "ngc_auth": self.ngc_auth,
            "ngc_image": self.ngc_image,
            "hf_auth": self.hf_auth,
            "hf_dataset": self.hf_dataset,
            "gpu": self.gpu,
            "rt_cores": self.rt_cores,
            "entrypoint": self.entrypoint,
            "cache_dir": self.cache_dir,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class NurecFetchResult:
    """Where the real NCore V4 shards landed, and what they contain."""

    ok: bool
    dataset_id: str
    scene: str
    variant: str
    scene_dir: str
    ncore_json: str
    shard_count: int
    bytes_downloaded: int
    observed_scene: str = ""
    observed_variant: str = ""
    camera_ids: tuple[str, ...] = ()
    lidar_ids: tuple[str, ...] = ()
    colmap_dir: str = ""
    poses_component_group: str = ""
    reference_camera: str = ""
    rig_derivation: dict[str, Any] = field(default_factory=dict)
    output_uri: str = ""
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "dataset_id": self.dataset_id,
            "scene": self.scene,
            "variant": self.variant,
            "scene_dir": self.scene_dir,
            "ncore_json": self.ncore_json,
            "shard_count": self.shard_count,
            "bytes_downloaded": self.bytes_downloaded,
            "observed_scene": self.observed_scene,
            "observed_variant": self.observed_variant,
            "camera_ids": list(self.camera_ids),
            "lidar_ids": list(self.lidar_ids),
            "colmap_dir": self.colmap_dir,
            "poses_component_group": self.poses_component_group,
            "reference_camera": self.reference_camera,
            "rig_derivation": dict(self.rig_derivation),
            "output_uri": self.output_uri,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class NurecReconstructResult:
    """Trained 3DGS reconstruction: the USDZ artifact plus real quality metrics."""

    ok: bool
    image: str
    config_name: str
    mode: str
    run_dir: str
    usdz_path: str
    parsed_config_path: str
    metrics_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    gt_dir: str = ""
    command: tuple[str, ...] = ()
    output_uri: str = ""
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "image": self.image,
            "config_name": self.config_name,
            "mode": self.mode,
            "run_dir": self.run_dir,
            "usdz_path": self.usdz_path,
            "parsed_config_path": self.parsed_config_path,
            "metrics_path": self.metrics_path,
            "metrics": dict(self.metrics),
            "gt_dir": self.gt_dir,
            "command": list(self.command),
            "output_uri": self.output_uri,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class NurecRenderResult:
    """Novel-view renders produced from a trained USDZ."""

    ok: bool
    artifact_path: str
    output_dir: str
    camera_ids: tuple[str, ...] = ()
    frame_count: int = 0
    video_count: int = 0
    novel_view: bool = True
    rig_translation_offset: str = ""
    rig_rotation_offset: str = ""
    command: tuple[str, ...] = ()
    output_uri: str = ""
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "artifact_path": self.artifact_path,
            "output_dir": self.output_dir,
            "camera_ids": list(self.camera_ids),
            "frame_count": self.frame_count,
            "video_count": self.video_count,
            "novel_view": self.novel_view,
            "rig_translation_offset": self.rig_translation_offset,
            "rig_rotation_offset": self.rig_rotation_offset,
            "command": list(self.command),
            "output_uri": self.output_uri,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class NurecStatusResult:
    """What a NuRec run prefix currently holds (stage → object count/bytes)."""

    ok: bool
    run_uri: str
    stages: dict[str, dict[str, int]] = field(default_factory=dict)
    object_count: int = 0
    has_rrd: bool = False
    has_usdz: bool = False
    has_novel_views: bool = False
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "run_uri": self.run_uri,
            "stages": {k: dict(v) for k, v in self.stages.items()},
            "object_count": self.object_count,
            "has_rrd": self.has_rrd,
            "has_usdz": self.has_usdz,
            "has_novel_views": self.has_novel_views,
            "errors": list(self.errors),
        }


# --------------------------------------------------------------------------------------
# process plumbing
# --------------------------------------------------------------------------------------
def _run(
    args: Sequence[str],
    *,
    env: Mapping[str, str],
    run: RunCallable,
    timeout: float | None,
    cwd: str | None = None,
) -> "subprocess.CompletedProcess[str]":
    command = list(args)
    kwargs: dict[str, Any] = {
        "env": dict(env),
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "timeout": timeout,
    }
    if cwd:
        kwargs["cwd"] = cwd
    try:
        return run(command, **kwargs)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command, 124, exc.stdout or "", exc.stderr or str(exc)
        )


def _secret_values(config: NurecConfig, env: Mapping[str, str]) -> list[str]:
    return [
        env.get(config.hf_token_env, ""),
        env.get(config.ngc_api_key_env, ""),
        env.get("HF_TOKEN", ""),
        env.get("HUGGING_FACE_HUB_TOKEN", ""),
        env.get("NGC_API_KEY", ""),
        env.get("NGC_CLI_API_KEY", ""),
        env.get("AWS_SECRET_ACCESS_KEY", ""),
    ]


def redact(
    text: str, config: NurecConfig, env: Mapping[str, str], *, limit: int = 2000
) -> str:
    """Replace every known secret value with ``<redacted>``, then truncate.

    The order matters: truncating first can slice through the middle of a secret,
    leaving a tail fragment that no longer matches the full value and therefore
    survives redaction. Redacting the whole body first makes the boundary
    irrelevant.
    """
    body = str(text or "").strip()
    if not body:
        return ""
    for value in _secret_values(config, env):
        if value and len(value) >= 8:
            body = body.replace(value, "<redacted>")
    return body[-limit:]


def _sanitize(
    result: "subprocess.CompletedProcess[str]",
    config: NurecConfig,
    env: Mapping[str, str],
) -> str:
    joined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return redact(joined, config, env)


def _hf_cli() -> str:
    """Resolve the Hugging Face CLI (`hf` is the current shim, `huggingface-cli` legacy)."""
    return shutil.which("hf") or shutil.which("huggingface-cli") or "huggingface-cli"


def _hf_env(config: NurecConfig, env: Mapping[str, str]) -> dict[str, str]:
    child = dict(env)
    token = child.get(config.hf_token_env, "")
    if token:
        child.setdefault("HF_TOKEN", token)
        child.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return child


def _nre_env(config: NurecConfig, env: Mapping[str, str]) -> dict[str, str]:
    child = dict(env)
    key = child.get(config.ngc_api_key_env, "")
    if key:
        child.setdefault("NGC_API_KEY", key)
    # NRE's validation prompts for W&B interactively unless logging is pinned.
    child.setdefault("WANDB_MODE", "disabled")
    child.setdefault("NRE_ENV_RUN_ID", config.nre_run_id)
    return child


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------
def check_nurec_access(
    config: NurecConfig,
    *,
    environ: Mapping[str, str] | None = None,
    runner: RunCallable | None = None,
    require_ngc: bool = True,
    require_gpu: bool = False,
    timeout: float = 30.0,
) -> NurecCheckResult:
    """Verify NGC container access, HF dataset access, and GPU suitability.

    Cheap and side-effect free: no image pull, no dataset download. Runs before
    any GPU work so a missing credential fails in seconds instead of after a
    14 GB image pull.
    """
    env = dict(environ if environ is not None else os.environ)
    run = runner or subprocess.run
    errors: list[str] = []

    if not config.image:
        errors.append("NRE image reference is required")

    ngc_auth = "configured" if env.get(config.ngc_api_key_env, "") else "missing"
    if ngc_auth == "missing" and require_ngc:
        errors.append(f"NGC auth missing: set {config.ngc_api_key_env}")

    ngc_image = "skipped"
    if config.image and ngc_auth == "configured":
        ngc_image = _check_ngc_image(config, env, timeout)
        if ngc_image != "reachable":
            errors.append(
                f"NRE container {config.image} is not pullable with the current NGC key "
                f"({ngc_image}); the GA channel repositories (…/nre-ga, …/nre-tools-ga) "
                "are the ones a standard NGC key can access"
            )

    hf_auth = "configured" if env.get(config.hf_token_env, "") else "missing"
    hf_dataset = "skipped"
    if config.dataset_id:
        hf_dataset = _check_hf_dataset(config, env, timeout)
        if hf_dataset != "reachable":
            errors.append(
                f"Hugging Face dataset {config.dataset_id} is not downloadable "
                f"({hf_dataset}); gated datasets need the license accepted by the "
                f"account that owns {config.hf_token_env}"
            )

    gpu, rt_cores = _check_gpu(env, run, timeout)
    if require_gpu and gpu in {"missing", "unavailable"}:
        errors.append("no NVIDIA GPU is visible; NuRec reconstruction requires one")
    if rt_cores == "no":
        errors.append(
            f"GPU {gpu!r} has no RT cores; route NuRec at L40S or "
            "RTX PRO 6000 Blackwell (never H100/H200)"
        )

    entrypoint = "skipped" if config.docker_bin else _check_entrypoint(config)
    if entrypoint == "missing":
        errors.append(
            f"NRE entrypoint {config.entrypoint} not found; run inside the NRE image "
            "or set --docker-bin to invoke it through Docker"
        )

    return NurecCheckResult(
        ok=not errors,
        image=config.image,
        ngc_auth=ngc_auth,
        ngc_image=ngc_image,
        hf_auth=hf_auth,
        hf_dataset=hf_dataset,
        gpu=gpu,
        rt_cores=rt_cores,
        entrypoint=entrypoint,
        cache_dir=str(config.resolved_cache_dir),
        errors=tuple(errors),
    )


def _check_ngc_image(
    config: NurecConfig, env: Mapping[str, str], timeout: float
) -> str:
    """Token-exchange + tag listing against the registry (no layer download)."""
    import base64

    import httpx

    key = env.get(config.ngc_api_key_env, "")
    registry = config.image_registry or DEFAULT_NRE_REGISTRY_HOST
    repository = config.image_repository
    if not repository:
        return "unresolved"
    basic = base64.b64encode(f"$oauthtoken:{key}".encode()).decode()
    try:
        auth = httpx.get(
            f"https://{registry}/proxy_auth",
            params={"scope": f"repository:{repository}:pull"},
            headers={"Authorization": f"Basic {basic}"},
            timeout=timeout,
        )
        if auth.status_code == 402:
            return "entitlement-required"
        if auth.status_code != 200:
            return f"auth-{auth.status_code}"
        token = str(auth.json().get("token") or "")
        if not token:
            return "auth-no-token"
        tags = httpx.get(
            f"https://{registry}/v2/{repository}/tags/list",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - any transport failure is "not reachable"
        return "unreachable"
    return "reachable" if 200 <= tags.status_code < 300 else f"tags-{tags.status_code}"


def check_ngc_image_access(
    api_key: str,
    *,
    image: str = DEFAULT_NRE_IMAGE,
    timeout: float = 30.0,
) -> str:
    """Probe token exchange and pull entitlement for one NGC image repository."""

    config = NurecConfig(image=image)
    return _check_ngc_image(config, {config.ngc_api_key_env: api_key}, timeout)


def _check_hf_dataset(
    config: NurecConfig, env: Mapping[str, str], timeout: float
) -> str:
    """Probe real *download* authorization, not just metadata visibility.

    A gated dataset returns 200 for ``/api/datasets/<id>`` even when the token
    cannot pull a byte, so this asks for one byte of the target archive.
    """
    import httpx

    token = env.get(config.hf_token_env, "")
    headers = {"Range": "bytes=0-0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = (
        f"https://huggingface.co/datasets/{config.dataset_id}"
        f"/resolve/main/{config.ncore_member}"
    )
    try:
        response = httpx.get(
            url, headers=headers, timeout=timeout, follow_redirects=True
        )
    except Exception:  # noqa: BLE001
        return "unreachable"
    if response.status_code in {401, 403}:
        return "gated"
    if response.status_code == 404:
        return "member-not-found"
    return (
        "reachable"
        if 200 <= response.status_code < 300
        else f"http-{response.status_code}"
    )


def _check_gpu(
    env: Mapping[str, str], run: RunCallable, timeout: float
) -> tuple[str, str]:
    result = _run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        env=env,
        run=run,
        timeout=timeout,
    )
    if result.returncode == 127:
        return "missing", "unknown"
    if result.returncode != 0:
        return "unavailable", "unknown"
    names = [
        line.strip() for line in (result.stdout or "").splitlines() if line.strip()
    ]
    if not names:
        return "unavailable", "unknown"
    name = names[0]
    return name, "yes" if has_rt_cores(name) else "no"


def _check_entrypoint(config: NurecConfig) -> str:
    path = Path(config.entrypoint)
    return "present" if path.exists() else "missing"


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------
def fetch_nurec_dataset(
    config: NurecConfig,
    *,
    environ: Mapping[str, str] | None = None,
    runner: RunCallable | None = None,
    force: bool = False,
    with_colmap: bool = False,
    derive_rig: bool = True,
    timeout: float | None = None,
) -> NurecFetchResult:
    """Download and unpack the real NCore V4 shards for the configured scene.

    When ``derive_rig`` is set (the default) and the sequence has no
    ``rig -> world`` edge, a derived sequence carrying one is written and returned
    as ``ncore_json`` — without it NRE 26.04 cannot load a COLMAP-derived capture
    at all (see :mod:`npa.workbench.nurec.ncore_rig`).
    """
    env = dict(environ if environ is not None else os.environ)
    run = runner or subprocess.run
    cache = config.resolved_cache_dir
    download_dir = cache / "hf"
    ncore_root = cache / "ncore"
    colmap_root = cache / "colmap"

    if force:
        for path in (download_dir, ncore_root, colmap_root):
            if path.exists():
                shutil.rmtree(path)
    download_dir.mkdir(parents=True, exist_ok=True)
    ncore_root.mkdir(parents=True, exist_ok=True)

    members = [config.ncore_member]
    if with_colmap:
        members.append(config.colmap_member)

    for member in members:
        command = [
            _hf_cli(),
            "download",
            config.dataset_id,
            "--repo-type",
            "dataset",
            "--include",
            member,
            "--local-dir",
            str(download_dir),
        ]
        result = _run(command, env=_hf_env(config, env), run=run, timeout=timeout)
        if result.returncode != 0:
            return NurecFetchResult(
                ok=False,
                dataset_id=config.dataset_id,
                scene=config.scene,
                variant=config.variant,
                scene_dir="",
                ncore_json="",
                shard_count=0,
                bytes_downloaded=0,
                errors=(f"dataset download failed: {_sanitize(result, config, env)}",),
            )

    ncore_zip = download_dir / config.ncore_member
    if not ncore_zip.is_file():
        return NurecFetchResult(
            ok=False,
            dataset_id=config.dataset_id,
            scene=config.scene,
            variant=config.variant,
            scene_dir="",
            ncore_json="",
            shard_count=0,
            bytes_downloaded=0,
            errors=(f"expected archive not present after download: {ncore_zip}",),
        )
    bytes_downloaded = ncore_zip.stat().st_size

    try:
        extract_archive(ncore_zip, ncore_root)
    except NurecError as exc:
        return NurecFetchResult(
            ok=False,
            dataset_id=config.dataset_id,
            scene=config.scene,
            variant=config.variant,
            scene_dir="",
            ncore_json="",
            shard_count=0,
            bytes_downloaded=bytes_downloaded,
            errors=(str(exc),),
        )

    colmap_dir = ""
    if with_colmap:
        colmap_zip = download_dir / config.colmap_member
        if colmap_zip.is_file():
            colmap_root.mkdir(parents=True, exist_ok=True)
            try:
                extract_archive(colmap_zip, colmap_root)
                colmap_dir = str(colmap_root)
                bytes_downloaded += colmap_zip.stat().st_size
            except NurecError:
                # COLMAP is a convenience copy of the same capture; its absence must
                # not fail a reconstruction that only needs the NCore shards.
                colmap_dir = ""

    scene_dir = find_scene_dir(ncore_root, config.scene_dir_name)
    if scene_dir is None:
        available = sorted(p.name for p in ncore_root.rglob("*") if p.is_dir())[:20]
        return NurecFetchResult(
            ok=False,
            dataset_id=config.dataset_id,
            scene=config.scene,
            variant=config.variant,
            scene_dir="",
            ncore_json="",
            shard_count=0,
            bytes_downloaded=bytes_downloaded,
            colmap_dir=colmap_dir,
            errors=(
                f"scene directory {config.scene_dir_name!r} not found under {ncore_root}; "
                f"available: {available}",
            ),
        )

    # Independent observed provenance: derive the scene/variant from the actual
    # unpacked archive directory, then fail closed if it disagrees with the
    # requested values. This is not an echo of the request arguments — it is
    # grounded in the extracted content itself, so a fetch that pulled the wrong
    # capture (or a caller that passed the wrong flags) is rejected at fetch time.
    observed_scene, observed_variant = derive_scene_variant_from_dir(scene_dir)
    fetched = {
        "dataset_id": config.dataset_id,
        "scene": config.scene,
        "variant": config.variant,
        "observed_scene": observed_scene,
        "observed_variant": observed_variant,
    }
    ok, errors = validate_fetch_provenance(
        fetched,
        requested_scene=config.scene,
        requested_variant=config.variant,
        requested_dataset_id=config.dataset_id,
    )
    if not ok:
        return NurecFetchResult(
            ok=False,
            dataset_id=config.dataset_id,
            scene=config.scene,
            variant=config.variant,
            scene_dir=str(scene_dir),
            ncore_json="",
            shard_count=0,
            bytes_downloaded=bytes_downloaded,
            observed_scene=observed_scene,
            observed_variant=observed_variant,
            colmap_dir=colmap_dir,
            errors=("provenance mismatch: " + "; ".join(errors),),
        )

    ncore_json = find_ncore_json(scene_dir)
    if ncore_json is None:
        return NurecFetchResult(
            ok=False,
            dataset_id=config.dataset_id,
            scene=config.scene,
            variant=config.variant,
            scene_dir=str(scene_dir),
            ncore_json="",
            shard_count=0,
            bytes_downloaded=bytes_downloaded,
            colmap_dir=colmap_dir,
            errors=(f"no NCore V4 metadata JSON found under {scene_dir}",),
        )

    cameras, lidars = ncore_sensor_ids(ncore_json)
    shards = list(scene_dir.rglob("*.itar"))

    resolved_json = ncore_json
    poses_group = ""
    reference_camera = ""
    rig_derivation: dict[str, Any] = {}
    if derive_rig:
        from npa.workbench.nurec.ncore_rig import derive_rig_poses

        rig_result = derive_rig_poses(
            ncore_json,
            output_dir=cache / "ncore-rig" / scene_dir.name,
            reference_camera=config.reference_camera,
        )
        rig_derivation = rig_result.as_dict()
        if not rig_result.ok:
            return NurecFetchResult(
                ok=False,
                dataset_id=config.dataset_id,
                scene=config.scene,
                variant=config.variant,
                scene_dir=str(scene_dir),
                ncore_json=str(ncore_json),
                shard_count=len(shards),
                bytes_downloaded=bytes_downloaded,
                camera_ids=cameras,
                lidar_ids=lidars,
                colmap_dir=colmap_dir,
                rig_derivation=rig_derivation,
                errors=rig_result.errors,
            )
        if not rig_result.already_present:
            resolved_json = Path(rig_result.output_meta)
            poses_group = rig_result.poses_component_group
            reference_camera = rig_result.reference_camera

    return NurecFetchResult(
        ok=True,
        dataset_id=config.dataset_id,
        scene=config.scene,
        variant=config.variant,
        scene_dir=str(scene_dir),
        ncore_json=str(resolved_json),
        shard_count=len(shards),
        bytes_downloaded=bytes_downloaded,
        observed_scene=observed_scene,
        observed_variant=observed_variant,
        camera_ids=cameras,
        lidar_ids=lidars,
        colmap_dir=colmap_dir,
        poses_component_group=poses_group,
        reference_camera=reference_camera,
        rig_derivation=rig_derivation,
    )


def extract_archive(archive: Path | str, destination: Path | str) -> Path:
    """Extract a ``.zip`` with the stdlib, rejecting path traversal.

    The NRE image ships no ``unzip`` binary, and adding an apt dependency just to
    unpack an archive would be gratuitous. ``zipfile`` also lets the unit tests
    exercise the real extraction path against a real archive.
    """
    import zipfile

    source = Path(archive)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()
    try:
        with zipfile.ZipFile(source) as bundle:
            for member in bundle.namelist():
                candidate = (resolved_target / member).resolve()
                if (
                    resolved_target != candidate
                    and resolved_target not in candidate.parents
                ):
                    raise NurecError(
                        f"archive member escapes the destination directory: {member!r}"
                    )
            bundle.extractall(target)
    except zipfile.BadZipFile as exc:
        raise NurecError(f"not a readable zip archive: {source} ({exc})") from exc
    except OSError as exc:
        raise NurecError(f"failed to extract {source}: {exc}") from exc
    return target


def find_scene_dir(root: Path, scene_dir_name: str) -> Path | None:
    """Locate ``<scene>``/``<scene>_<variant>`` inside an unpacked NCore archive."""
    direct = root / scene_dir_name
    if direct.is_dir():
        return direct
    for candidate in sorted(root.rglob(scene_dir_name)):
        if candidate.is_dir():
            return candidate
    return None


def derive_scene_variant_from_dir(scene_dir: str | Path) -> tuple[str, str]:
    """Infer ``(scene, variant)`` from the *unpacked* NCore scene directory.

    This is independent observed content from the actual extracted archive, not
    an echo of the request arguments. The PPISP archives name the directory
    ``<scene>`` for the standard/full variant and ``<scene>_<variant>``
    (e.g. ``toro_auto``) otherwise. Returns ``(scene, "standard")`` when the
    directory carries no variant suffix.
    """
    name = Path(scene_dir).name
    if "_" in name:
        scene, variant = name.rsplit("_", 1)
        return scene.strip(), variant.strip()
    return name, "standard"


def _variant_layout_key(variant: str) -> str:
    """Normalize a variant label that maps to the unsuffixed scene directory.

    ``scene_dir_name`` returns the bare scene name for ``""``/``standard``/
    ``default``/``full`` and ``<scene>_<variant>`` otherwise, so all of the
    bare-layout labels compare equal when validating observed content.
    """
    v = (str(variant) or "").strip().lower()
    if v in {"", "standard", "default", "full"}:
        return "standard"
    return v


def validate_fetch_provenance(
    fetched: Mapping[str, Any],
    *,
    requested_scene: str,
    requested_variant: str,
    requested_dataset_id: str = "",
) -> tuple[bool, list[str]]:
    """Fail-closed provenance check against *independently observed* content.

    A fetch result must never be trusted just because it echoes the requested
    dataset/scene/variant: those top-level fields are copied from the request
    arguments. The authoritative evidence is the observed unpacked content
    (``observed_scene`` / ``observed_variant``), which the fetch derives from the
    scene directory that actually landed in the extracted archive. Missing
    observed content, an observed-vs-requested mismatch, or a dataset_id mismatch
    all fail closed.

    Returns ``(ok, errors)``; ``errors`` is empty when ``ok`` is True.
    """
    errors: list[str] = []
    observed_scene = str(fetched.get("observed_scene") or "")
    observed_variant = str(fetched.get("observed_variant") or "")
    if not observed_scene and not observed_variant:
        return False, ["fetch result carries no observed unpacked content"]
    if str(observed_scene).strip() != str(requested_scene).strip():
        errors.append(
            f"scene observed={observed_scene!r} != requested={requested_scene!r}"
        )
    if _variant_layout_key(observed_variant) != _variant_layout_key(requested_variant):
        errors.append(
            f"variant observed={observed_variant!r} != requested={requested_variant!r}"
        )
    if requested_dataset_id and str(fetched.get("dataset_id") or "") != requested_dataset_id:
        errors.append("dataset_id mismatch")
    return (not errors), errors


def find_ncore_json(scene_dir: Path) -> Path | None:
    """Return the NCore V4 metadata JSON that sits next to the ``.zarr.itar`` shards.

    NCore names the metadata ``<NAME>.json`` alongside ``<NAME>.zarr.itar``, so
    prefer a JSON whose stem matches a shard; fall back to the shallowest JSON.
    """
    shard_stems = {path.name.split(".", 1)[0] for path in scene_dir.rglob("*.itar")}
    candidates = sorted(scene_dir.rglob("*.json"), key=lambda p: (len(p.parts), p.name))
    for candidate in candidates:
        if candidate.name.split(".", 1)[0] in shard_stems:
            return candidate
    return candidates[0] if candidates else None


def read_rig_sidecar(ncore_json: Path | str) -> dict[str, Any]:
    """Read the ``npa-rig.json`` sidecar next to an NCore meta-file, if present.

    Written by :func:`npa.workbench.nurec.ncore_rig.derive_rig_poses`. Its presence
    means the sequence has a DERIVED rig frame, i.e. an object-centric capture --
    which is exactly the case where the recipe's SfM point-cloud initialization
    supports only one camera. Absent for AV sequences that ship their own rig, so
    those keep their full multi-camera behaviour.
    """
    from npa.workbench.nurec.ncore_rig import RIG_SIDECAR_NAME

    sidecar = Path(ncore_json).parent / RIG_SIDECAR_NAME
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def ncore_sensor_ids(ncore_json: Path | str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Best-effort camera / LiDAR ID discovery from an NCore V4 metadata JSON.

    NCore keys sensors by id somewhere in the document; walking the tree for keys
    that start with ``camera``/``lidar`` is robust across layout revisions and
    never raises on an unexpected shape.
    """
    try:
        payload = json.loads(Path(ncore_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (), ()
    cameras: set[str] = set()
    lidars: set[str] = set()

    # NCore V4 declares sensors explicitly: each entry of ``component_stores``
    # carries ``components.cameras`` / ``components.lidars`` keyed by sensor id.
    # Real ids do not have to start with "camera"/"lidar" (PPISP ships
    # ``virtual_lidar``), so read the structure first and only then fall back to
    # the name heuristic below.
    stores = payload.get("component_stores") if isinstance(payload, dict) else None
    if isinstance(stores, list):
        for store in stores:
            components = store.get("components") if isinstance(store, dict) else None
            if not isinstance(components, dict):
                continue
            for group, target in (("cameras", cameras), ("lidars", lidars)):
                entries = components.get(group)
                if isinstance(entries, dict):
                    target.update(str(name) for name in entries)
    if cameras or lidars:
        return tuple(sorted(cameras)), tuple(sorted(lidars))

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                token = str(key)
                if token.startswith("camera"):
                    cameras.add(token)
                elif token.startswith("lidar"):
                    lidars.add(token)
                visit(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    if item.startswith("camera"):
                        cameras.add(item)
                    elif item.startswith("lidar"):
                        lidars.add(item)
                else:
                    visit(item)
        elif isinstance(node, str):
            if node.startswith("camera"):
                cameras.add(node)
            elif node.startswith("lidar"):
                lidars.add(node)

    visit(payload)
    return tuple(sorted(cameras)), tuple(sorted(lidars))


# --------------------------------------------------------------------------------------
# reconstruct
# --------------------------------------------------------------------------------------
def reconstruct_scene(
    config: NurecConfig,
    *,
    ncore_json: str,
    environ: Mapping[str, str] | None = None,
    runner: RunCallable | None = None,
    dry_run: bool = False,
    export_gt: bool = True,
    timeout: float | None = None,
) -> NurecReconstructResult:
    """Train a 3DGUT Gaussian reconstruction and collect its USDZ + metrics."""
    env = dict(environ if environ is not None else os.environ)
    run = runner or subprocess.run
    out_dir = config.resolved_out_dir
    args = build_nre_train_args(config, ncore_json=ncore_json, out_dir=str(out_dir))
    mounts = _default_mounts(config, ncore_json)
    command = nre_command(
        config, args, mounts=mounts, env_names=[config.ngc_api_key_env, "HF_TOKEN"]
    )

    if dry_run:
        return NurecReconstructResult(
            ok=True,
            image=config.image,
            config_name=config.config_name,
            mode=config.mode,
            run_dir=str(config.nre_run_dir),
            usdz_path="",
            parsed_config_path="",
            metrics_path="",
            command=tuple(command),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(command, env=_nre_env(config, env), run=run, timeout=timeout)
    if result.returncode != 0:
        return NurecReconstructResult(
            ok=False,
            image=config.image,
            config_name=config.config_name,
            mode=config.mode,
            run_dir=str(config.nre_run_dir),
            usdz_path="",
            parsed_config_path="",
            metrics_path="",
            command=tuple(command),
            errors=(
                f"NRE reconstruction failed (exit {result.returncode}): "
                f"{_sanitize(result, config, env)}",
            ),
        )

    run_dir = resolve_nre_run_dir(out_dir, config.nre_run_id)
    usdz = latest_usdz(run_dir)
    parsed = _first_existing(
        run_dir / "config" / "parsed.yaml", *sorted(run_dir.rglob("parsed.yaml"))
    )
    metrics_path = _first_existing(
        run_dir / "val" / "metrics.yaml", *sorted(run_dir.rglob("metrics.yaml"))
    )
    metrics = parse_metrics_yaml(metrics_path) if metrics_path else {}
    errors: list[str] = []
    if usdz is None:
        errors.append(
            "reconstruction finished but no .usdz artifact was produced; "
            "checkpoint.artifact.enabled must be true"
        )

    gt_dir = ""
    if export_gt and not errors:
        gt_target = run_dir / "gt"
        gt_args = build_nre_export_gt_args(
            ncore_json=ncore_json,
            output_dir=str(gt_target),
        )
        gt_result = _run(
            nre_command(
                config, gt_args, mounts=mounts, env_names=[config.ngc_api_key_env]
            ),
            env=_nre_env(config, env),
            run=run,
            timeout=timeout,
        )
        # Ground-truth export is evidence, not the deliverable: a release that does
        # not ship the sub-command must not fail the reconstruction.
        if gt_result.returncode == 0 and gt_target.exists():
            gt_dir = str(gt_target)

    return NurecReconstructResult(
        ok=not errors,
        image=config.image,
        config_name=config.config_name,
        mode=config.mode,
        run_dir=str(run_dir),
        usdz_path=str(usdz) if usdz else "",
        parsed_config_path=str(parsed) if parsed else "",
        metrics_path=str(metrics_path) if metrics_path else "",
        metrics=metrics,
        gt_dir=gt_dir,
        command=tuple(command),
        errors=tuple(errors),
    )


def _default_mounts(config: NurecConfig, ncore_json: str) -> list[tuple[str, str]]:
    """Bind mounts for the docker-host shape (identity mounts keep paths stable)."""
    if not config.docker_bin:
        return []
    dataset_root = (
        str(Path(ncore_json).parent) if ncore_json else str(config.resolved_cache_dir)
    )
    return [
        (dataset_root, dataset_root),
        (str(config.resolved_out_dir), str(config.resolved_out_dir)),
    ]


def resolve_nre_run_dir(
    out_dir: Path | str, preferred: str = DEFAULT_NRE_RUN_ID
) -> Path:
    """Locate the ``<out_dir>/<RUN-ID>/`` directory NRE actually wrote.

    ``logger.run_id`` / ``NRE_ENV_RUN_ID`` normally pin this, but a release that
    ignores them falls back to a random hash, so the newest directory that looks
    like an NRE run is accepted rather than reporting "no artifacts".
    """
    root = Path(out_dir)
    candidate = root / preferred
    if candidate.is_dir():
        return candidate
    if not root.is_dir():
        return candidate
    runs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / "usd-out").is_dir()
            or (path / "config").is_dir()
            or (path / "checkpoints").is_dir()
        )
    ]
    if not runs:
        return candidate
    return max(runs, key=lambda path: path.stat().st_mtime)


def latest_usdz(run_dir: Path | str) -> Path | None:
    """Return the most recent renderable USDZ artifact in an NRE run directory.

    Upstream docs describe ``<run>/usd-out/last.usdz``, but nre-ga 26.04 actually
    writes one artifact per checkpoint as ``<run>/artifacts/<step>.usdz``. Both
    layouts are handled, and the NEWEST artifact wins — picking the first match
    alphabetically would ship the 1000-step preview instead of the trained scene.
    """
    root = Path(run_dir)
    documented = root / "usd-out" / "last.usdz"
    if documented.is_file():
        return documented
    if not root.is_dir():
        return None
    candidates = [path for path in root.rglob("*.usdz") if path.is_file()]
    if not candidates:
        return None
    # Tie-break on the PARSED step number, not the name: artifact names are not
    # zero-padded consistently, so a lexical comparison ranks "7000.usdz" above
    # "10000.usdz" and would ship an early preview whenever two artifacts share an
    # mtime (common straight after an archive extraction, which sets them equal).
    return max(
        candidates, key=lambda path: (path.stat().st_mtime, _usdz_step(path), path.name)
    )


def _usdz_step(path: Path) -> int:
    """Training step encoded in an artifact name, or -1 when there is none."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _first_existing(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def parse_metrics_yaml(path: Path | str) -> dict[str, float]:
    """Extract every numeric leaf from an NRE ``metrics.yaml`` as a flat mapping.

    Keys are slash-joined paths, so NRE's ``test: {psnr: ...}`` becomes
    ``{"test/psnr": ...}`` -- but the extraction is not limited to ``test/*``; any
    numeric leaf is recorded.

    NRE 26.04 ships validation numbers under an ``aggregated_metrics`` section
    where each entry is ``test/psnr: {aggregation_method: mean, value: 22.66}``.
    Those are additionally exposed under the bare metric name (``test/psnr``), so
    callers always read ``test/psnr`` / ``test/ssim`` / ``test/lpips`` regardless
    of whether a release writes the flat form or the aggregated wrapper.

    Parsed with PyYAML when available and a flat ``key: value`` scan otherwise, so
    the helper stays usable in a dependency-light container. Metrics are EVIDENCE,
    never the deliverable: a metrics file that is missing, unreadable, or corrupt
    (a real possibility after an NRE crash or a partial write) must never fail a
    reconstruction whose training actually succeeded, so every failure mode
    degrades to the scan or to an empty mapping.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # A truncated or mis-encoded file is a degraded read, not a crash.
        return {}
    metrics: dict[str, float] = {}
    try:
        import yaml

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            for key, value in _flatten(payload):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = float(value)
            # NRE 26.04 writes its validation numbers under an
            # ``aggregated_metrics`` section, each entry a dict like
            # ``test/psnr: {aggregation_method: mean, value: 22.66}``. Expose the
            # numeric ``value`` under the bare metric name so callers can read
            # ``test/psnr`` / ``test/ssim`` / ``test/lpips`` exactly as the skill
            # documents, rather than the nested ``aggregated_metrics/.../value``.
            aggregated = payload.get("aggregated_metrics")
            if isinstance(aggregated, dict):
                for name, entry in aggregated.items():
                    if isinstance(entry, dict):
                        value = entry.get("value")
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            metrics[str(name)] = float(value)
            return metrics
    except ImportError:
        _logger.debug("PyYAML unavailable; falling back to a flat metrics scan")
    except Exception as exc:  # noqa: BLE001 - yaml.YAMLError is NOT a ValueError
        # yaml.safe_load raises yaml.YAMLError, which derives from Exception and
        # NOT from ValueError; catching ValueError here let a corrupt metrics.yaml
        # escape and crash a successful reconstruction.
        _logger.debug("metrics.yaml is not parseable (%s); using a flat scan", exc)
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        try:
            metrics[key.strip()] = float(raw.strip())
        except ValueError:
            continue
    return metrics


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in payload.items():
        label = f"{prefix}{key}"
        if isinstance(value, Mapping):
            items.extend(_flatten(value, prefix=f"{label}/"))
        else:
            items.append((label, value))
    return items


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------
def render_novel_views(
    config: NurecConfig,
    *,
    artifact_path: str,
    output_dir: str,
    camera_ids: Sequence[str] = (),
    image_scale: float = DEFAULT_IMAGE_SCALE,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    frame_naming: str = DEFAULT_FRAME_NAMING,
    renderer: str = DEFAULT_RENDERER,
    frame_step: int = DEFAULT_FRAME_STEP,
    rig_translation_offset: Any = DEFAULT_RIG_TRANSLATION_OFFSET,
    rig_rotation_offset: Any = DEFAULT_RIG_ROTATION_OFFSET,
    custom_rig_trajectory: str = "",
    replicate_training_views: bool = False,
    export_video: bool = True,
    video_fps: float = DEFAULT_VIDEO_FPS,
    video_crf: int = DEFAULT_VIDEO_CRF,
    environ: Mapping[str, str] | None = None,
    runner: RunCallable | None = None,
    dry_run: bool = False,
    timeout: float | None = None,
) -> NurecRenderResult:
    """Render novel views from a trained USDZ with the real ``nre render`` path."""
    env = dict(environ if environ is not None else os.environ)
    run = runner or subprocess.run
    translation = parse_offset(rig_translation_offset, DEFAULT_RIG_TRANSLATION_OFFSET)
    rotation = parse_offset(rig_rotation_offset, DEFAULT_RIG_ROTATION_OFFSET)
    args = build_nre_render_args(
        config,
        artifact_path=artifact_path,
        output_dir=output_dir,
        camera_ids=camera_ids,
        image_scale=image_scale,
        image_format=image_format,
        frame_naming=frame_naming,
        renderer=renderer,
        frame_step=frame_step,
        rig_translation_offset=translation,
        rig_rotation_offset=rotation,
        custom_rig_trajectory=custom_rig_trajectory,
        replicate_training_views=replicate_training_views,
        export_video=export_video,
        video_fps=video_fps,
        video_crf=video_crf,
    )
    mounts = (
        [
            (str(Path(artifact_path).parent), str(Path(artifact_path).parent)),
            (str(output_dir), str(output_dir)),
        ]
        if config.docker_bin
        else []
    )
    command = nre_command(
        config, args, mounts=mounts, env_names=[config.ngc_api_key_env]
    )

    if dry_run:
        return NurecRenderResult(
            ok=True,
            artifact_path=artifact_path,
            output_dir=output_dir,
            camera_ids=tuple(camera_ids),
            novel_view=not replicate_training_views,
            rig_translation_offset=_offset_text(translation),
            rig_rotation_offset=_offset_text(rotation),
            command=tuple(command),
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = _run(command, env=_nre_env(config, env), run=run, timeout=timeout)
    if result.returncode != 0:
        return NurecRenderResult(
            ok=False,
            artifact_path=artifact_path,
            output_dir=output_dir,
            camera_ids=tuple(camera_ids),
            novel_view=not replicate_training_views,
            rig_translation_offset=_offset_text(translation),
            rig_rotation_offset=_offset_text(rotation),
            command=tuple(command),
            errors=(
                f"NRE render failed (exit {result.returncode}): "
                f"{_sanitize(result, config, env)}",
            ),
        )

    frames = count_render_frames(output_dir)
    videos = len(list(Path(output_dir).rglob("*.mp4")))
    errors = [] if frames else ["render produced no frames"]
    return NurecRenderResult(
        ok=not errors,
        artifact_path=artifact_path,
        output_dir=output_dir,
        camera_ids=tuple(camera_ids),
        frame_count=frames,
        video_count=videos,
        novel_view=not replicate_training_views,
        rig_translation_offset=_offset_text(translation),
        rig_rotation_offset=_offset_text(rotation),
        command=tuple(command),
        errors=tuple(errors),
    )


def _offset_text(offset: tuple[float, float, float]) -> str:
    return ",".join(f"{value}" for value in offset)


def count_render_frames(output_dir: Path | str) -> int:
    root = Path(output_dir)
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


# --------------------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------------------
def materialize_uri(
    source_uri: str, destination: Path | str, *, storage_client: Any = None
) -> Path:
    """Fetch ``source_uri`` (S3 prefix/object or local path) to ``destination``.

    Stages of the declarative workflow run in SEPARATE pods, so nothing survives
    in ``/tmp`` between them: the NCore sequence and the trained USDZ have to
    travel through S3. A local ``source_uri`` is returned as-is so the
    single-pod SkyPilot task keeps working without a round-trip.
    """
    if not source_uri:
        raise NurecError("source_uri is required")
    if not source_uri.startswith("s3://"):
        local = Path(source_uri)
        if not local.exists():
            raise NurecError(f"local source does not exist: {local}")
        return local

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    target = Path(destination)
    is_prefix = source_uri.endswith("/")
    if is_prefix:
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    client.download_path(source_uri, str(target))
    return target


def publish_ncore_sequence(
    ncore_json: Path | str,
    output_uri: str,
    *,
    storage_client: Any = None,
    max_bytes: int = 0,
) -> dict[str, Any]:
    """Upload a complete NCore sequence (meta-file + every shard) to ``output_uri``.

    The derived sequence directory references the original shards through
    SYMLINKS to avoid copying hundreds of megabytes locally; those cannot be
    uploaded as links, so each is resolved and uploaded as a real object. The
    result is a self-contained sequence a later stage can materialize.
    """
    source = Path(ncore_json)
    if not source.is_file():
        raise NurecError(f"NCore sequence meta-file not found: {source}")
    sequence_dir = source.parent
    siblings = sorted(
        path
        for path in sequence_dir.iterdir()
        if path != source and (path.is_file() or path.is_symlink())
    )
    members = [source, *siblings]
    total = 0
    uploaded: list[str] = []
    if not output_uri.startswith("s3://"):
        destination = Path(output_uri)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            resolved = member.resolve()
            if not resolved.is_file():
                continue
            shutil.copy2(resolved, destination / member.name)
            total += resolved.stat().st_size
            uploaded.append(member.name)
        return {
            "uri": str(destination),
            "objects": len(uploaded),
            "bytes": total,
            "meta_name": source.name,
            "members": uploaded,
        }

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    base = output_uri.rstrip("/")
    for member in members:
        resolved = member.resolve()
        if not resolved.is_file():
            continue
        size = resolved.stat().st_size
        if max_bytes and total + size > max_bytes:
            raise NurecError(
                f"NCore sequence exceeds max_bytes={max_bytes} at member {member.name}"
            )
        client.upload_file(str(resolved), f"{base}/{member.name}")
        total += size
        uploaded.append(member.name)
    return {
        "uri": f"{base}/",
        "objects": len(uploaded),
        "bytes": total,
        "meta_name": source.name,
        "members": uploaded,
    }


def nurec_run_status(
    run_uri: str,
    *,
    storage_client: Any = None,
) -> NurecStatusResult:
    """Summarize a NuRec run prefix (local dir or ``s3://``) stage by stage."""
    if not run_uri:
        return NurecStatusResult(
            ok=False, run_uri=run_uri, errors=("run_uri is required",)
        )
    try:
        entries = _list_run_entries(run_uri, storage_client=storage_client)
    except Exception as exc:  # noqa: BLE001 - surface transport errors as a result
        return NurecStatusResult(ok=False, run_uri=run_uri, errors=(str(exc),))

    stages: dict[str, dict[str, int]] = {}
    for relative, size in entries:
        stage = relative.split("/", 1)[0] if "/" in relative else "."
        bucket = stages.setdefault(stage, {"objects": 0, "bytes": 0})
        bucket["objects"] += 1
        bucket["bytes"] += int(size)
    names = [relative for relative, _ in entries]
    return NurecStatusResult(
        ok=bool(entries),
        run_uri=run_uri,
        stages=stages,
        object_count=len(entries),
        has_rrd=any(name.endswith("reports/sim2real.rrd") for name in names),
        has_usdz=any(name.endswith(".usdz") for name in names),
        has_novel_views=any(name.startswith("novel_views/") for name in names),
        errors=() if entries else (f"no objects found under {run_uri}",),
    )


def _list_run_entries(
    run_uri: str, *, storage_client: Any = None
) -> list[tuple[str, int]]:
    if not run_uri.startswith("s3://"):
        root = Path(run_uri)
        return [
            (str(path.relative_to(root)).replace(os.sep, "/"), path.stat().st_size)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
    from urllib.parse import urlparse

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    parsed = urlparse(run_uri.rstrip("/"))
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    prefix = f"{prefix}/" if prefix and not prefix.endswith("/") else prefix
    s3 = client._s3  # noqa: SLF001 - StorageClient exposes no raw list API
    entries: list[tuple[str, int]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            entries.append((key[len(prefix) :], int(item.get("Size") or 0)))
    return entries
