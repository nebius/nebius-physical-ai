"""Shared execution path for genuine Isaac Lab-Arena policy evaluation.

The CLI, SDK, and ``npa.workflow`` toolRef all call this module.  The worker
image contains the immutable Apache-2.0 Arena source, but Isaac Sim and Isaac
Lab remain operator-authorized runtime downloads in the inherited NPA cache.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

ISAAC_ARENA_VERSION = "0.3.0"
ISAAC_ARENA_REVISION = "ed0fd12be862078be316c73eb7cf423ba9b1c5cd"
LIGHTWHEEL_SDK_VERSION = "1.0.3"
ISAAC_ARENA_ROOT = "/opt/isaac-arena"
ARTIFACT_SCHEMA = "npa.workbench.isaac_arena.evaluation.v1"
CAPABILITIES_SCHEMA = "npa.workbench.isaac_arena.capabilities.v1"
SUPPORTED_POLICIES = frozenset({"zero_action", "replay", "rsl_rl"})
REGISTERED_ENVIRONMENTS = frozenset(
    {
        "cube_goal_pose",
        "dexsuite_lift",
        "droid_table_multi_object_placement",
        "franka_put_and_close_door",
        "galileo_g1_locomanip_pick_and_place",
        "galileo_pick_and_place",
        "gr1_open_microwave",
        "gr1_table_multi_object_no_collision",
        "gr1_turn_stand_mixer_knob",
        "kitchen_pick_and_place",
        "lift_object",
        "peg_insert",
        "pick_and_place_maple_table",
        "press_button",
        "put_item_in_fridge_and_close_door",
        "gear_mesh",
        "tabletop_place_upright",
        "tabletop_sort_cubes",
    }
)
UNSUPPORTED_ENVIRONMENTS = frozenset(
    {"droid_table_multi_object_placement", "gr1_table_multi_object_no_collision"}
)
SUPPORTED_ENVIRONMENTS = REGISTERED_ENVIRONMENTS - UNSUPPORTED_ENVIRONMENTS
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VIDEO_SAMPLE_WIDTH = 160
_VIDEO_SAMPLE_HEIGHT = 90
_VIDEO_SAMPLE_FPS = 4.0
_VIDEO_MAX_SAMPLES = 120
_MIN_VIDEO_DURATION_SECONDS = 1.0
_MIN_CHANGED_FRAME_PAIRS = 2
_MIN_CHANGED_PIXEL_RATIO = 0.005
_MIN_FRAME_MEAN_ABS_DELTA = 1.0
_CHANGED_PIXEL_LUMA_DELTA = 8


_ENVIRONMENT_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "name": "cube_goal_pose",
        "task": "GoalPoseTask",
        "default_embodiment": "franka_ik",
        "default_object": "dex_cube",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "dexsuite_lift",
        "task": "DexsuiteLiftTask",
        "default_embodiment": "kuka_allegro",
        "default_object": "DexSuite task object",
        "metrics": ["success_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "droid_table_multi_object_placement",
        "task": "NoTask",
        "default_embodiment": "droid_abs_joint_pos",
        "default_object": "registered object pool",
        "metrics": [],
        "npa_status": ["unsupported", "upstream_alpha"],
        "limitation": "Upstream defines no scored task, so it cannot satisfy NPA's evaluation-result contract.",
    },
    {
        "name": "franka_put_and_close_door",
        "task": "FrankaPutAndCloseDoorTask (sequential PickAndPlaceTask + CloseDoorTask)",
        "default_embodiment": "franka_ik",
        "default_object": "dex_cube",
        "metrics": [
            "success_rate",
            "object_moved_rate",
            "revolute_joint_moved_rate",
            "subtask_success_rate",
        ],
        "npa_status": ["implemented", "input_required", "upstream_alpha"],
        "runtime_assets": [
            {
                "provider": "Lightwheel registry",
                "selector": "fixtures/Microwave039/USD",
                "delivery": "runtime_fetch",
                "baked": False,
            }
        ],
    },
    {
        "name": "galileo_g1_locomanip_pick_and_place",
        "task": "PickAndPlaceTask",
        "default_embodiment": "g1_wbc_pink",
        "default_object": "brown_box",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "galileo_pick_and_place",
        "task": "PickAndPlaceTask",
        "default_embodiment": "gr1_pink",
        "default_object": "power_drill",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "gr1_open_microwave",
        "task": "OpenDoorTask",
        "default_embodiment": "gr1_pink",
        "allowed_embodiments": ["gr1_joint", "gr1_pink"],
        "default_object": None,
        "metrics": ["success_rate", "revolute_joint_moved_rate"],
        "npa_status": [
            "implemented",
            "input_required",
            "upstream_alpha",
        ],
        "runtime_assets": [
            {
                "provider": "Lightwheel registry",
                "selector": "fixtures/Microwave039/USD",
                "delivery": "runtime_fetch",
                "baked": False,
            }
        ],
    },
    {
        "name": "gr1_table_multi_object_no_collision",
        "task": "NoTask",
        "default_embodiment": "gr1_joint",
        "default_object": "registered object pool",
        "metrics": [],
        "npa_status": ["unsupported", "upstream_alpha"],
        "limitation": "Upstream defines no scored task, so it cannot satisfy NPA's evaluation-result contract.",
    },
    {
        "name": "gr1_turn_stand_mixer_knob",
        "task": "TurnKnobTask",
        "default_embodiment": "gr1_pink",
        "default_object": None,
        "metrics": ["success_rate", "revolute_joint_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "kitchen_pick_and_place",
        "task": "PickAndPlaceTask",
        "default_embodiment": "franka_ik",
        "default_object": "cracker_box",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "lift_object",
        "task": "LiftObjectTask or LiftObjectTaskRL",
        "default_embodiment": "franka_joint_pos",
        "default_object": "dex_cube",
        "metrics": ["success_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "peg_insert",
        "task": "AssemblyTask",
        "default_embodiment": "franka_ik",
        "default_object": "peg (destination: hole)",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "pick_and_place_maple_table",
        "task": "PickAndPlaceTask",
        "default_embodiment": "droid_abs_joint_pos",
        "default_object": "rubiks_cube_hot3d_robolab",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
        "limitation": "NPA exposes the upstream defaults; its pick_up_object-specific override is not yet surfaced.",
    },
    {
        "name": "press_button",
        "task": "PressButtonTask",
        "default_embodiment": "franka_ik",
        "default_object": None,
        "metrics": ["success_rate"],
        "npa_status": ["implemented", "input_required", "upstream_alpha"],
        "runtime_assets": [
            {
                "provider": "Lightwheel registry",
                "selector": "fixtures/CoffeeMachine108/USD",
                "delivery": "runtime_fetch",
                "baked": False,
            }
        ],
    },
    {
        "name": "put_item_in_fridge_and_close_door",
        "task": "SequentialTaskBase (PickAndPlaceTask + CloseDoorTask)",
        "default_embodiment": "gr1_pink",
        "default_object": "ranch_dressing_hope_robolab",
        "metrics": [
            "success_rate",
            "object_moved_rate",
            "revolute_joint_moved_rate",
            "subtask_success_rate",
        ],
        "npa_status": ["implemented", "input_required", "upstream_alpha"],
        "runtime_assets": [
            {
                "provider": "Lightwheel registry",
                "selector": "Robocasa kitchen layout/style plus task objects",
                "delivery": "runtime_fetch",
                "baked": False,
            }
        ],
    },
    {
        "name": "gear_mesh",
        "task": "AssemblyTask",
        "default_embodiment": "franka_ik",
        "default_object": "medium_gear (fixed gear-base destination)",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "tabletop_place_upright",
        "task": "PlaceUprightTask",
        "default_embodiment": "agibot",
        "default_object": "mug",
        "metrics": ["success_rate", "object_moved_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
    {
        "name": "tabletop_sort_cubes",
        "task": "SortMultiObjectTask",
        "default_embodiment": "franka_ik",
        "default_object": "red_cube + green_cube",
        "metrics": ["success_rate"],
        "npa_status": ["implemented", "upstream_alpha"],
    },
)


class IsaacArenaError(RuntimeError):
    """Raised when an Arena request or upstream evaluation is invalid."""


def capabilities() -> dict[str, Any]:
    """Return the pinned upstream surface and NPA's narrower support status."""

    return {
        "schema": CAPABILITIES_SCHEMA,
        "upstream": {
            "repository": "https://github.com/isaac-sim/IsaacLab-Arena",
            "version": ISAAC_ARENA_VERSION,
            "revision": ISAAC_ARENA_REVISION,
            "release_channel": "alpha",
            "production_supported": False,
        },
        "status_vocabulary": {
            "implemented": "The pinned NPA runner can invoke this upstream path.",
            "live_validated": "Digest-scoped readiness evidence proves this path completed on physical target hardware; this tag is never inferred from implementation alone.",
            "input_required": "A named operator policy/trajectory or separately delivered runtime asset is mandatory.",
            "unsupported": "Present upstream but intentionally outside the current NPA execution contract.",
            "upstream_alpha": "Upstream marks 0.3.0 pre-release and its APIs unstable.",
        },
        "policy_adapters": [
            {
                "name": "zero_action",
                "required_input": None,
                "outputs_motion_by_contract": False,
                "npa_status": ["implemented", "upstream_alpha"],
                "limitation": "A video may be decodable while behavior remains static; it is not meaningful visual evidence.",
            },
            {
                "name": "replay",
                "required_input": "one Isaac Lab HDF5 episode with a nonzero action and state trajectory",
                "outputs_motion_by_contract": True,
                "npa_status": [
                    "implemented",
                    "input_required",
                    "upstream_alpha",
                ],
                "execution_contract": {
                    "source_evidence": "SHA-256, action statistics, recorded state change, and recorded success metadata",
                    "gpu_materialization": "private actions plus initial_state only; unused observations and state histories are not copied to CUDA",
                    "episode_horizon": (
                        "The source trajectory is never truncated. Optional replay_target_steps "
                        "holds its final recorded action until the known task horizon."
                    ),
                    "retained_input_bytes": False,
                },
            },
            {
                "name": "rsl_rl",
                "required_input": "one model*.pt checkpoint with sibling params/agent.yaml",
                "outputs_motion_by_contract": False,
                "npa_status": ["implemented", "input_required", "upstream_alpha"],
                "limitation": "Packaged and upstream-tested, but not yet NPA live-qualified.",
            },
        ],
        "upstream_extension_policies": [
            {
                "name": "pi0_remote",
                "required_input": "separately served OpenPI policy plus compatible embodiment adapter",
                "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            },
            {
                "name": "gr00t_remote_closedloop",
                "required_input": "separately served GR00T policy, config, and compatible embodiment adapter",
                "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            },
            {
                "name": "cosmos_remote",
                "required_input": "separately served Cosmos policy plus compatible embodiment adapter",
                "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            },
            {
                "name": "dreamzero_remote",
                "required_input": "separately served DreamZero policy plus compatible embodiment adapter",
                "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            },
        ],
        "live_validation": {
            "scope": "digest_bound_external_evidence",
            "embedded_claims": False,
            "readiness_sources": [
                "npa/src/npa/deploy/gpu-readiness.yaml",
                "npa/src/npa/deploy/public-image-release-manifest.yaml",
            ],
            "reason": (
                "A container cannot truthfully pre-assert qualification of its own "
                "not-yet-published digest; promotion and readiness records bind live "
                "results after the immutable development image completes validation."
            ),
        },
        "environments": copy.deepcopy(list(_ENVIRONMENT_CAPABILITIES)),
        "environment_sources": {
            "registered_python_environments": len(_ENVIRONMENT_CAPABILITIES),
            "graph_specs": {
                "catalog_counts": {
                    "kitchen_bench": 31,
                    "maple_table_top": 3,
                    "robolab_scenes": 17,
                    "robolab_tasks": 38,
                },
                "npa_status": ["unsupported", "upstream_alpha"],
                "limitation": "The pinned source ships these YAML catalogs, but NPA does not expose --env_spec yet.",
            },
            "external_environment_class": {
                "npa_status": ["unsupported", "upstream_alpha"],
                "limitation": "Arbitrary Python import paths are not accepted by the public NPA runner.",
            },
        },
        "runtime_dependencies": [
            {
                "name": "lightwheel-sdk",
                "version": LIGHTWHEEL_SDK_VERSION,
                "purpose": "Resolve the Lightwheel-backed assets named by applicable upstream environments.",
                "license": "Apache-2.0",
                "baked": True,
                "source_lock": "upstream uv.lock wheel SHA-256",
            },
            {
                "name": "Lightwheel registry assets",
                "purpose": "USD fixtures, objects, and generated kitchen layouts used by applicable environments.",
                "license": "upstream-provider-controlled",
                "baked": False,
                "delivery": "runtime_fetch",
                "redistribution": False,
                "access": "The operator must be authorized by the upstream service; NPA supplies no Lightwheel credential or license grant.",
                "stability": "Selectors and service responses are external runtime state, not pinned NPA payload.",
            },
        ],
        "embodiment_constraints": {
            "registered": [
                "agibot",
                "droid",
                "droid_abs_joint_pos",
                "droid_differential_ik",
                "droid_rel_joint_pos",
                "franka_ik",
                "franka_joint_pos",
                "g1",
                "g1_wbc_agile_joint",
                "g1_wbc_agile_pink",
                "g1_wbc_joint",
                "g1_wbc_pink",
                "galbot",
                "gr1",
                "gr1_joint",
                "gr1_pink",
                "kuka_allegro",
                "no_embodiment",
            ],
            "rule": "An override must be registered and compatible with the selected environment and policy action space.",
        },
        "object_constraints": {
            "rule": (
                "An override must name a registered asset of the task-required type; registry membership alone does not "
                "prove task compatibility or runtime asset access. Environment records expose validated defaults only."
            ),
            "simready_runtime_assets": {
                "npa_status": ["input_required", "upstream_alpha"],
                "limitation": "Availability and redistribution depend on the operator's runtime asset source.",
            },
        },
        "outputs": {
            "always": [
                "result.json",
                "evaluation.log",
                "episode_results_rank*.jsonl",
                "upstream HTML report tree",
                "SHA-256 and byte size for every retained artifact",
            ],
            "video_when_requested": [
                "viewport MP4 with decoded temporal-motion validation"
            ],
            "metrics": [
                "success_rate",
                "object_moved_rate",
                "revolute_joint_moved_rate",
                "subtask_success_rate",
                "progress score and predicate events where the task defines them",
            ],
            "rerun_rrd": False,
        },
        "input_handling": {
            "operator_input_bytes_published": False,
            "source_and_execution_hashes_retained": True,
            "source_location_redacted_from_result": True,
        },
        "rendering": {
            "viewport_video": {
                "npa_status": ["implemented", "upstream_alpha"],
                "requires": "RTX rasterization/RT-capable GPU; the readiness record must prove the exact qualified digest and target",
                "acceptance": {
                    "codec": "h264",
                    "minimum_dimensions": [320, 240],
                    "minimum_duration_seconds": _MIN_VIDEO_DURATION_SECONDS,
                    "minimum_decoded_samples": _MIN_CHANGED_FRAME_PAIRS + 1,
                    "minimum_changed_frame_pairs": _MIN_CHANGED_FRAME_PAIRS,
                    "minimum_changed_pixel_ratio": _MIN_CHANGED_PIXEL_RATIO,
                    "minimum_frame_mean_absolute_luma_delta": _MIN_FRAME_MEAN_ABS_DELTA,
                    "changed_pixel_luma_delta": _CHANGED_PIXEL_LUMA_DELTA,
                },
            },
            "camera_observation_video": {
                "npa_status": ["unsupported", "upstream_alpha"],
                "limitation": "Upstream supports it, but NPA currently requests only the viewport recorder.",
            },
            "b200": "State-only evaluation; B200 has no RT cores and carries no video claim.",
            "execution_devices": {
                "cuda:0": "Default GPU physics and policy execution path.",
                "cpu": (
                    "CPU physics/policy execution with the assigned RTX GPU retained for "
                    "viewport rendering. Use this compatibility path when an upstream "
                    "environment poisons CUDA during GPU-physics initialization; the result "
                    "still records the independently measured renderer GPU."
                ),
            },
            "agent_ui": "Use the authenticated native video renderer; Arena does not emit a truthful native Rerun .rrd.",
        },
    }


@dataclass(frozen=True)
class IsaacArenaRequest:
    """One reproducible policy evaluation request."""

    output_path: str
    environment: str = "cube_goal_pose"
    policy_type: str = "zero_action"
    input_path: str = ""
    replay_target_steps: int = 0
    execution_device: str = "cuda:0"
    num_episodes: int = 1
    num_envs: int = 1
    seed: int = 42
    embodiment: str = ""
    object_name: str = ""
    record_video: bool = False
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate(request: IsaacArenaRequest) -> None:
    parsed = urlparse(request.output_path)
    if not request.output_path.strip():
        raise IsaacArenaError("output_path is required")
    if parsed.scheme and parsed.scheme != "s3":
        raise IsaacArenaError("output_path must be a local path or s3:// prefix")
    if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
        raise IsaacArenaError("output_path must include an S3 bucket and prefix")
    if request.policy_type not in SUPPORTED_POLICIES:
        raise IsaacArenaError(
            "policy_type must be one of: " + ", ".join(sorted(SUPPORTED_POLICIES))
        )
    if request.environment not in SUPPORTED_ENVIRONMENTS:
        if request.environment in UNSUPPORTED_ENVIRONMENTS:
            raise IsaacArenaError(
                "environment has no upstream scored task and is unsupported by the NPA evaluation contract"
            )
        raise IsaacArenaError(
            "environment must be a registered NPA Arena environment; inspect the capabilities command"
        )
    for field, value in (
        ("environment", request.environment),
        ("embodiment", request.embodiment),
        ("object_name", request.object_name),
    ):
        if value and _NAME.fullmatch(value) is None:
            raise IsaacArenaError(f"{field} contains unsupported characters")
    if request.num_episodes < 1 or request.num_envs < 1:
        raise IsaacArenaError("num_episodes and num_envs must be positive")
    if request.num_envs > request.num_episodes:
        raise IsaacArenaError("num_envs cannot exceed num_episodes")
    if request.policy_type == "zero_action" and request.input_path:
        raise IsaacArenaError("zero_action does not accept input_path")
    if request.policy_type != "zero_action" and not request.input_path:
        raise IsaacArenaError(f"{request.policy_type} requires input_path")
    if request.replay_target_steps < 0:
        raise IsaacArenaError("replay_target_steps cannot be negative")
    if request.replay_target_steps and request.policy_type != "replay":
        raise IsaacArenaError("replay_target_steps is valid only for replay policies")
    if request.execution_device not in {"cpu", "cuda:0"}:
        raise IsaacArenaError("execution_device must be cpu or cuda:0")


def _local_input(request: IsaacArenaRequest, root: Path) -> Path | None:
    if not request.input_path:
        return None
    if request.input_path.startswith("s3://"):
        target = root / "input"
        downloaded = StorageClient.from_environment().download_path(
            request.input_path, str(target)
        )
        return Path(downloaded).resolve()
    parsed = urlparse(request.input_path)
    if parsed.scheme:
        raise IsaacArenaError("input_path must be a local path or s3:// URI")
    path = Path(request.input_path).expanduser().resolve()
    if not path.exists():
        raise IsaacArenaError(f"input_path does not exist: {path}")
    return path


def build_evaluation_argv(
    request: IsaacArenaRequest, *, output_dir: Path, local_input: Path | None = None
) -> list[str]:
    """Build argv for upstream's genuine ``policy_runner.py``."""

    argv = [
        os.environ.get("ISAAC_ARENA_PYTHON", "/isaac-sim/python.sh"),
        f"{ISAAC_ARENA_ROOT}/isaaclab_arena/evaluation/policy_runner.py",
        "--headless",
        "--device",
        request.execution_device,
        "--policy_type",
        request.policy_type,
        "--num_episodes",
        str(request.num_episodes),
        "--num_envs",
        str(request.num_envs),
        "--seed",
        str(request.seed),
        "--output_base_dir",
        str(output_dir),
    ]
    if request.record_video:
        argv.append("--record_viewport_video")
    if request.policy_type == "replay":
        if local_input is None or not local_input.is_file():
            raise IsaacArenaError("replay input_path must resolve to one HDF5 file")
        argv.extend(["--replay_file_path", str(local_input)])
    elif request.policy_type == "rsl_rl":
        if local_input is None:
            raise IsaacArenaError("rsl_rl requires a checkpoint input")
        checkpoint, _agent_config = _checkpoint_input(local_input)
        argv.extend(["--checkpoint_path", str(checkpoint)])
    # Upstream uses an argparse subparser per environment and explicitly makes
    # its argv order part of the interface: global and policy options first,
    # then the environment name, then environment-specific options.
    argv.append(request.environment)
    if request.embodiment:
        argv.extend(["--embodiment", request.embodiment])
    if request.object_name:
        argv.extend(["--object", request.object_name])
    return argv


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    # Inputs are materialized and outputs are published by NPA, so the simulator
    # gets no cloud credentials or HTTP admission secrets.  This also keeps its
    # captured log safe to retain as an evaluation artifact.
    for key in tuple(env):
        upper = key.upper()
        if upper in {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "HF_TOKEN",
            "NGC_API_KEY",
            "NEBIUS_IAM_TOKEN",
        } or upper.endswith(("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD")):
            env.pop(key, None)
    env.setdefault("ACCEPT_EULA", "Y")
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_input(local_input: Path) -> tuple[Path, Path]:
    checkpoint = local_input
    if checkpoint.is_dir():
        candidates = sorted(checkpoint.glob("model*.pt"))
        if not candidates:
            candidates = sorted(checkpoint.rglob("model*.pt"))
        if len(candidates) != 1:
            raise IsaacArenaError(
                "rsl_rl input directory must contain exactly one model*.pt checkpoint"
            )
        checkpoint = candidates[0]
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise IsaacArenaError("rsl_rl input_path must resolve to a .pt checkpoint")
    agent_config = checkpoint.parent / "params" / "agent.yaml"
    if not agent_config.is_file():
        raise IsaacArenaError("rsl_rl checkpoint requires sibling params/agent.yaml")
    return checkpoint, agent_config


def _replay_input_evidence(path: Path) -> dict[str, Any]:
    """Prove an HDF5 replay contains nonzero actions and changing recorded state."""

    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - provided by the Isaac runtime
        raise IsaacArenaError("replay validation requires h5py and numpy") from exc
    try:
        with h5py.File(path, "r") as dataset:
            episodes = sorted(str(name) for name in dataset.get("data", {}))
            if not episodes:
                raise IsaacArenaError("replay HDF5 contains no episodes under data/")
            episode_name = episodes[0]
            episode = dataset["data"][episode_name]
            if "actions" not in episode:
                raise IsaacArenaError("replay HDF5 episode contains no actions")
            actions = np.asarray(episode["actions"])
            if (
                actions.ndim < 2
                or actions.shape[0] < 2
                or not np.isfinite(actions).all()
            ):
                raise IsaacArenaError(
                    "replay actions must be a finite multi-step tensor"
                )
            action_abs_max = float(np.max(np.abs(actions)))
            action_abs_mean = float(np.mean(np.abs(actions)))
            action_nonzero_fraction = float(np.mean(np.abs(actions) > 1e-6))
            action_step_delta_mean = float(np.mean(np.abs(np.diff(actions, axis=0))))
            if action_abs_max < 1e-4 or action_nonzero_fraction < 0.001:
                raise IsaacArenaError("replay actions are effectively zero")

            state_ranges: list[tuple[str, float]] = []

            def inspect_state(name: str, item: Any) -> None:
                if not isinstance(item, h5py.Dataset) or not name.startswith("states/"):
                    return
                if len(item.shape) < 1 or item.shape[0] != actions.shape[0]:
                    return
                values = np.asarray(item)
                if (
                    not np.issubdtype(values.dtype, np.number)
                    or not np.isfinite(values).all()
                ):
                    return
                state_ranges.append((name, float(np.max(np.ptp(values, axis=0)))))

            episode.visititems(inspect_state)
            state_ranges.sort(key=lambda item: (-item[1], item[0]))
            state_max_range = state_ranges[0][1] if state_ranges else 0.0
            if state_max_range < 1e-4:
                raise IsaacArenaError("replay contains no changing recorded state")
            return {
                "episode": episode_name,
                "recorded_success": (
                    bool(episode.attrs["success"])
                    if "success" in episode.attrs
                    else None
                ),
                "steps": int(actions.shape[0]),
                "action_dimensions": int(actions.shape[-1]),
                "action_abs_max": action_abs_max,
                "action_abs_mean": action_abs_mean,
                "action_nonzero_fraction": action_nonzero_fraction,
                "action_step_delta_mean": action_step_delta_mean,
                "state_max_range": state_max_range,
                "largest_state_ranges": [
                    {"dataset": name, "range": value}
                    for name, value in state_ranges[:5]
                ],
                "meaningful": True,
            }
    except OSError as exc:
        raise IsaacArenaError("replay input is not a readable HDF5 file") from exc


def _prepare_replay_execution_input(
    source: Path,
    private_dir: Path,
    *,
    target_steps: int,
    source_evidence: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Create the minimal private replay file the upstream adapter consumes.

    Isaac Lab's generic HDF5 loader eagerly copies every recorded observation
    and state tensor to CUDA, even though ReplayActionPolicy uses only actions
    and the initial state.  Normalize to that exact contract and optionally
    hold the last genuine command until the task horizon so upstream can emit
    a completed, scored episode.  Neither source nor normalized input is
    published as an output artifact.
    """

    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - provided by the Isaac runtime
        raise IsaacArenaError("replay normalization requires h5py and numpy") from exc
    episode_name = str((source_evidence.get("trajectory") or {}).get("episode") or "")
    if not episode_name:
        raise IsaacArenaError("replay evidence does not identify an episode")
    destination = private_dir / "replay-execution.hdf5"
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with h5py.File(source, "r") as source_file:
            source_episode = source_file["data"][episode_name]
            actions = np.asarray(source_episode["actions"])
            source_steps = int(actions.shape[0])
            executed_steps = target_steps or source_steps
            if executed_steps < source_steps:
                raise IsaacArenaError(
                    "replay_target_steps cannot truncate the recorded trajectory"
                )
            if executed_steps > source_steps:
                held = np.repeat(actions[-1:], executed_steps - source_steps, axis=0)
                execution_actions = np.concatenate((actions, held), axis=0)
            else:
                execution_actions = actions
            with h5py.File(destination, "w") as output_file:
                for key, value in source_file.attrs.items():
                    output_file.attrs[key] = value
                output_data = output_file.create_group("data")
                for key, value in source_file["data"].attrs.items():
                    output_data.attrs[key] = value
                output_data.attrs["total"] = executed_steps
                output_episode = output_data.create_group(episode_name)
                for key, value in source_episode.attrs.items():
                    output_episode.attrs[key] = value
                output_episode.attrs["num_samples"] = executed_steps
                output_episode.create_dataset(
                    "actions", data=execution_actions, compression="gzip"
                )
                if "initial_state" not in source_episode:
                    raise IsaacArenaError("replay HDF5 episode contains no initial_state")
                source_episode.copy("initial_state", output_episode)
    except (KeyError, OSError) as exc:
        raise IsaacArenaError("replay input cannot be normalized for execution") from exc
    destination.chmod(0o600)
    return destination, {
        "strategy": "actions_initial_state_only_hold_final_action",
        "source_sha256": str(source_evidence.get("sha256") or ""),
        "executed_sha256": _sha256(destination),
        "source_steps": source_steps,
        "executed_steps": executed_steps,
        "held_final_action_steps": executed_steps - source_steps,
        "fields": ["actions", "initial_state"],
        "published": False,
    }


def _input_evidence(
    request: IsaacArenaRequest, local_input: Path | None
) -> dict[str, Any] | None:
    if local_input is None:
        return None
    if request.policy_type == "replay":
        if not local_input.is_file():
            raise IsaacArenaError("replay input_path must resolve to one HDF5 file")
        return {
            "kind": "replay_hdf5",
            "bytes": local_input.stat().st_size,
            "sha256": _sha256(local_input),
            "trajectory": _replay_input_evidence(local_input),
        }
    checkpoint, agent_config = _checkpoint_input(local_input)
    return {
        "kind": "rsl_rl_checkpoint",
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "agent_config_sha256": _sha256(agent_config),
    }


def _numeric_metrics(log_text: str) -> dict[str, float]:
    """Extract upstream's plain-Python metric mapping from its retained log."""

    for line in reversed(log_text.splitlines()):
        marker = "Metrics:"
        if marker not in line:
            continue
        candidate = line.split(marker, 1)[1].strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        result: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                result[str(key)] = number
        if result:
            return result
    return {}


def _summarize(run_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    result_files = sorted(run_dir.glob("episode_results_rank*.jsonl"))
    if not result_files:
        raise IsaacArenaError("upstream evaluation wrote no episode-results JSONL")
    records: list[dict[str, Any]] = []
    for path in result_files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IsaacArenaError(
                    f"invalid episode JSONL at {path.name}:{number}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(
                record.get("success"), bool
            ):
                raise IsaacArenaError(
                    f"episode record at {path.name}:{number} has no boolean success"
                )
            records.append(record)
    if not records:
        raise IsaacArenaError("upstream evaluation completed no scored episodes")
    try:
        lengths = [int(record["episode_length"]) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise IsaacArenaError(
            "upstream evaluation recorded no integer episode length"
        ) from exc
    if any(length <= 0 for length in lengths):
        raise IsaacArenaError("upstream evaluation recorded an invalid episode length")
    successes = sum(bool(record["success"]) for record in records)
    progress = [record.get("progress") for record in records]
    progress = [item for item in progress if isinstance(item, dict)]
    scores = [
        float(item.get("overall_score", 0.0))
        for item in progress
        if isinstance(item.get("overall_score", 0.0), (int, float))
    ]
    events = sum(
        len(item.get("events", []))
        for item in progress
        if isinstance(item.get("events", []), list)
    )
    return (
        {
            "episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
            "mean_episode_length": sum(lengths) / len(lengths),
            "max_progress_score": max(scores, default=0.0),
            "progress_event_count": events,
        },
        result_files,
    )


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        return {
            "available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "",
            "compute_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else [],
        }
    except Exception:
        return {"available": False, "device_name": "", "compute_capability": []}


def _probe_mp4(path: Path) -> dict[str, Any]:
    """Require browser-compatible video plus independently decoded motion."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,nb_read_frames,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
        codec = str(stream["codec_name"])
        width = int(stream["width"])
        height = int(stream["height"])
        frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames"))
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IsaacArenaError(f"invalid viewport MP4: {path.name}") from exc
    if (
        completed.returncode != 0
        or codec != "h264"
        or width < 320
        or height < 240
        or duration < _MIN_VIDEO_DURATION_SECONDS
        or frame_count < _MIN_CHANGED_FRAME_PAIRS + 1
    ):
        raise IsaacArenaError(f"invalid viewport MP4: {path.name}")

    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            (
                f"fps={_VIDEO_SAMPLE_FPS},"
                f"scale={_VIDEO_SAMPLE_WIDTH}:{_VIDEO_SAMPLE_HEIGHT}:flags=area,format=gray"
            ),
            "-frames:v",
            str(_VIDEO_MAX_SAMPLES),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    frame_bytes = _VIDEO_SAMPLE_WIDTH * _VIDEO_SAMPLE_HEIGHT
    if decoded.returncode != 0 or len(decoded.stdout) % frame_bytes:
        raise IsaacArenaError(f"viewport MP4 frame decode failed: {path.name}")
    frames = [
        decoded.stdout[offset : offset + frame_bytes]
        for offset in range(0, len(decoded.stdout), frame_bytes)
    ]
    if len(frames) < _MIN_CHANGED_FRAME_PAIRS + 1:
        raise IsaacArenaError(
            f"viewport MP4 yielded too few decoded frames: {path.name}"
        )

    pair_stats: list[tuple[float, float]] = []
    for previous, current in zip(frames, frames[1:]):
        differences = [abs(left - right) for left, right in zip(previous, current)]
        mean_abs_delta = sum(differences) / frame_bytes
        changed_pixel_ratio = (
            sum(value >= _CHANGED_PIXEL_LUMA_DELTA for value in differences)
            / frame_bytes
        )
        pair_stats.append((mean_abs_delta, changed_pixel_ratio))
    meaningful_pairs = [
        pair
        for pair in pair_stats
        if pair[0] >= _MIN_FRAME_MEAN_ABS_DELTA and pair[1] >= _MIN_CHANGED_PIXEL_RATIO
    ]
    if len(meaningful_pairs) < _MIN_CHANGED_FRAME_PAIRS:
        raise IsaacArenaError(
            "viewport MP4 is decodable but visually static: "
            f"{len(meaningful_pairs)} changed pairs, need {_MIN_CHANGED_FRAME_PAIRS}"
        )
    return {
        "codec": codec,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "frame_count": frame_count,
        "motion": {
            "sample_width": _VIDEO_SAMPLE_WIDTH,
            "sample_height": _VIDEO_SAMPLE_HEIGHT,
            "sample_fps": _VIDEO_SAMPLE_FPS,
            "decoded_samples": len(frames),
            "changed_frame_pairs": len(meaningful_pairs),
            "max_frame_mean_abs_luma_delta": max(pair[0] for pair in pair_stats),
            "max_changed_pixel_ratio": max(pair[1] for pair in pair_stats),
            "thresholds": {
                "minimum_changed_frame_pairs": _MIN_CHANGED_FRAME_PAIRS,
                "minimum_changed_pixel_ratio": _MIN_CHANGED_PIXEL_RATIO,
                "minimum_frame_mean_abs_luma_delta": _MIN_FRAME_MEAN_ABS_DELTA,
                "changed_pixel_luma_delta": _CHANGED_PIXEL_LUMA_DELTA,
            },
            "meaningful": True,
        },
    }


def _publish(local_dir: Path, output_path: str) -> str:
    if output_path.startswith("s3://"):
        return StorageClient.from_environment().upload_directory(
            str(local_dir), output_path
        )
    target = Path(output_path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted(local_dir.iterdir()):
        destination = target / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    return str(target)


def evaluate(
    request: IsaacArenaRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run upstream Arena and publish its raw artifacts plus verified summary."""

    _validate(request)
    with tempfile.TemporaryDirectory(prefix="npa-isaac-arena-") as scratch:
        root = Path(scratch)
        artifact_root = root / "artifacts"
        private_dir = root / "private"
        output_root = artifact_root / "upstream"
        output_root.mkdir(parents=True)
        local_input = None if request.dry_run else _local_input(request, private_dir)
        input_evidence = (
            None if request.dry_run else _input_evidence(request, local_input)
        )
        execution_input = local_input
        if request.policy_type == "replay" and not request.dry_run:
            assert local_input is not None and input_evidence is not None
            execution_input, execution_evidence = _prepare_replay_execution_input(
                local_input,
                private_dir,
                target_steps=request.replay_target_steps,
                source_evidence=input_evidence,
            )
            input_evidence["execution"] = execution_evidence
        argv = build_evaluation_argv(
            request, output_dir=output_root, local_input=execution_input
        )
        public_request = asdict(request)
        public_request["output_path"] = "<operator-output>"
        if public_request["input_path"]:
            public_request["input_path"] = "<operator-input>"
        base: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "upstream": {
                "repository": "https://github.com/isaac-sim/IsaacLab-Arena",
                "version": ISAAC_ARENA_VERSION,
                "revision": ISAAC_ARENA_REVISION,
            },
            "request": public_request,
            "runtime": {
                "image": request.runtime_image or os.environ.get("NPA_TASK_IMAGE", ""),
                "execution_device": request.execution_device,
                "viewport_renderer_gpu_required": request.record_video,
                "isaac_runtime_fetch": True,
                "lightwheel_sdk": {
                    "version": LIGHTWHEEL_SDK_VERSION,
                    "baked": True,
                    "license": "Apache-2.0",
                },
                "lightwheel_registry_assets": {
                    "baked": False,
                    "runtime_fetch": request.environment
                    in {
                        "franka_put_and_close_door",
                        "gr1_open_microwave",
                        "press_button",
                        "put_item_in_fridge_and_close_door",
                    },
                    "license": "upstream-provider-controlled",
                    "redistribution": False,
                },
                "model_baked": False,
                "dataset_baked": False,
            },
            "input": input_evidence,
            "argv": argv,
        }
        if request.dry_run:
            return {**base, "status": "dry_run", "artifacts": {}}

        completed = runner(
            argv,
            cwd=ISAAC_ARENA_ROOT,
            env=_subprocess_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_text = completed.stdout or ""
        log_path = artifact_root / "evaluation.log"
        log_path.write_text(log_text, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-40:])
            raise IsaacArenaError(
                f"upstream policy_runner failed ({completed.returncode}):\n{tail}"
            )
        run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
        if len(run_dirs) != 1:
            raise IsaacArenaError(
                "upstream evaluation did not create exactly one run directory"
            )
        run_dir = run_dirs[0]
        summary, result_files = _summarize(run_dir)
        summary["metrics"] = _numeric_metrics(log_text)
        positive_metrics = {
            name: value for name, value in summary["metrics"].items() if value > 0.0
        }
        output_behavior = bool(
            summary["successes"]
            or summary["max_progress_score"] > 0.0
            or summary["progress_event_count"]
            or positive_metrics
        )
        if request.policy_type != "zero_action" and not output_behavior:
            raise IsaacArenaError(
                "nonzero policy produced no positive task, progress, or movement evidence"
            )
        behavior = {
            "policy_is_nonzero_adapter": request.policy_type != "zero_action",
            "output_behavior_observed": output_behavior,
            "positive_metrics": positive_metrics,
            "meaningful": request.policy_type != "zero_action" and output_behavior,
        }
        # Upstream writes the top-level report beside the JSONL journal. The
        # ``report/`` directory contains its linked task/job detail pages.
        report = run_dir / "index.html"
        if not report.is_file() or report.stat().st_size == 0:
            raise IsaacArenaError("upstream evaluation report is missing")
        videos = sorted(run_dir.rglob("*.mp4"))
        if request.record_video and (
            not videos or any(path.stat().st_size == 0 for path in videos)
        ):
            raise IsaacArenaError(
                "video recording was requested but no non-empty MP4 was written"
            )
        effective_run_id = request.run_id or run_dir.name
        input_sha256 = str((input_evidence or {}).get("sha256") or "")
        executed_input_sha256 = str(
            ((input_evidence or {}).get("execution") or {}).get("executed_sha256")
            or input_sha256
        )
        video_metadata = {}
        for path in videos:
            metadata = _probe_mp4(path)
            metadata["binding"] = {
                "run_id": effective_run_id,
                "upstream_run_directory": run_dir.name,
                "policy_type": request.policy_type,
                "input_sha256": input_sha256,
                "executed_input_sha256": executed_input_sha256,
            }
            video_metadata[path] = metadata

        # Publish the complete upstream report tree, and bind every retained
        # byte—not just its top-level index—to the result manifest.
        artifacts = sorted(path for path in artifact_root.rglob("*") if path.is_file())
        manifest = {
            **base,
            "status": "ok",
            "run_id": effective_run_id,
            "upstream_run_directory": run_dir.name,
            "summary": summary,
            "behavior": behavior,
            "gpu": _gpu_info(),
            "artifacts": [
                {
                    "path": str(path.relative_to(artifact_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    **(
                        {"video": video_metadata[path]}
                        if path in video_metadata
                        else {}
                    ),
                }
                for path in artifacts
            ],
        }
        if not manifest["gpu"]["available"]:
            raise IsaacArenaError("Arena evaluation returned without a CUDA device")
        result_path = artifact_root / "result.json"
        result_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination = _publish(artifact_root, request.output_path)
        return {**manifest, "published_to": destination}
