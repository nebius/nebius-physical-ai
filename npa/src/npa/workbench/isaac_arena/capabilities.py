"""Expose Arena features, required inputs, and digest-scoped validation status."""

import copy
from typing import Any

from .identity import (
    ISAAC_ARENA_VERSION,
    ISAAC_ARENA_REVISION,
    LIGHTWHEEL_SDK_VERSION,
    CAPABILITIES_SCHEMA,
)
from .ground_truth import MICROWAVE_MINIMUM_OPENNESS_DELTA, MICROWAVE_SUCCESS_THRESHOLD
from .video_evidence import video_acceptance_thresholds

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


_CAPABILITY_MANIFEST = {
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
            "limitation": "Viewport output is a simulator baseline only, never task-qualified visual evidence. Requested video must still pass capture and coherent-motion checks; static or noise-only failures retain diagnostic artifacts.",
        },
        {
            "name": "replay",
            "required_input": "one Isaac Lab HDF5 file with finite multi-step actions and a finite initial_state group; NPA selects its first sorted episode",
            "outputs_motion_by_contract": False,
            "npa_status": [
                "implemented",
                "input_required",
                "upstream_alpha",
            ],
            "execution_contract": {
                "num_envs": 1,
                "num_episodes": 1,
                "source_evidence": "SHA-256 and action statistics; optional recorded-state history and source success metadata remain diagnostics, never runtime results",
                "ordinary_evaluation_inputs": "Zero actions in compatible action spaces and missing or static recorded-state histories remain valid replay inputs; Pink poses require nonzero quaternion norms. Nonzero source actions are additionally required for replay visual qualification.",
                "gpu_materialization": "private actions plus initial_state only on the selected execution device (CPU or CUDA); unused observations and state histories are excluded",
                "pose_representation": {
                    "format_version": "Integer 0 or absent means legacy WXYZ; integer 1 means XYZW. Unknown or malformed versions are rejected.",
                    "gr1_pink": "For the resolved gr1_pink embodiment, the exact 36-column action contract and single-environment initial robot root pose are required. Legacy hand-target quaternion slices [3:7] and [10:14] and initial root_pose quaternions are reordered to XYZW in the private execution file before marking it version 1. Positions, hand commands, state values outside quaternions, action count, and ordering are preserved.",
                    "other_embodiments": "Embedded action quaternions are not converted. Inputs must match the selected native action layout; the upstream loader retains its root_pose-only legacy conversion. Equal action width never selects the Pink converter.",
                    "provenance": "The result records source/execution format versions, whether the source version defaulted, conversion scope, and both file hashes. Source bytes remain untouched and private.",
                },
                "episode_horizon": (
                    "The upstream policy runner applies the recorded initial_state with "
                    "Isaac Lab reset_to(is_relative=True), then executes every source action "
                    "exactly once without truncation, padding, repetition, or final-action hold."
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
            "workflows/testing/isaac-arena-evaluation-rtxpro.readiness.json",
            "workflows/testing/isaac-arena-evaluation-b200.readiness.json",
            "npa/docker/workbench/blackwell-dc-images.json",
            "npa/src/npa/deploy/public_release_manifest.json",
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
        "distributed_evaluation": {
            "npa_status": ["unsupported", "upstream_alpha"],
            "limitation": "Upstream torchrun/--distributed exists; NPA exposes single-process evaluation only.",
        },
        "hydra_variations": {
            "npa_status": ["unsupported", "upstream_alpha"],
            "limitation": "Upstream --list_variations and Hydra configuration overrides are not exposed by NPA.",
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
        {
            "name": "NVIDIA viewport graphics userspace",
            "purpose": "Provide headless EGL/Vulkan only when a viewport run's target exposes CUDA but omits graphics userspace.",
            "license": "NVIDIA driver package terms",
            "baked": False,
            "delivery": "runtime_fetch_exact_driver_match",
            "source": "Ubuntu signed NVIDIA driver archive",
            "installed_on_node": False,
            "redistribution": False,
            "scope": "viewport runs and run-private scratch only; native graphics is preferred",
            "headless_icd": "NPA derives a canonical EGL ICD manifest from the validated package manifest; it never mutates the packaged bytes or target.",
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
            "prove task compatibility or runtime asset access. Environment records expose upstream defaults, not live qualification."
        ),
        "simready_runtime_assets": {
            "npa_status": ["input_required", "upstream_alpha"],
            "limitation": "Availability and redistribution depend on the operator's runtime asset source.",
        },
    },
    "outputs": {
        "completed_evaluation": [
            "result.json",
            "evaluation.log",
            "episode_results_rank*.jsonl",
            "upstream HTML report tree",
            "simulator_ground_truth_rank*.hdf5",
            "SHA-256 and byte size for every retained artifact",
        ],
        "handled_runtime_or_evidence_failure": {
            "result": "After evaluation setup succeeds, handled runtime/evidence errors write a failed result.json and hashes for whatever artifacts were produced, then attempt publication. A storage publication failure retains the private local copy.",
            "partial_artifacts": "Logs, raw video, capture sidecar, and initial or terminal PNGs are retained only if created. An unfinished recorder buffer is separately marked unscored; it never becomes a completed episode or task-success result.",
        },
        "early_failure": "Request validation or input/setup failures may occur before any result tree exists; an interrupted worker may leave only workflow logs. Completed-episode JSONL, scored HDF5, and HTML reports are not guaranteed on failure.",
        "successful_video_qualification": [
            "raw viewport MP4 and a labeled denoised derivative",
            "simulator-video-evidence.json with capture phase and action-step mapping",
            "simulator-initial.png and simulator-episode-0-terminal.png with file and decoded-RGB hashes",
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
            "request_constraints": {"num_envs": 1, "num_episodes": 1},
            "requires": "RTX rasterization/RT-capable GPU; the readiness record must prove the exact qualified digest and target",
            "graphics_userspace": (
                "Native NVIDIA EGL/Vulkan is preferred. If CUDA is healthy but those libraries are absent, "
                "NPA extracts (never installs) the exact loaded-driver version from Ubuntu's signed archive "
                "into run-private scratch. It validates the package's ICD metadata, derives a private canonical "
                "headless EGL ICD, validates Vulkan, and publishes only package/manifest identity and SHA-256."
            ),
            "acceptance": video_acceptance_thresholds(),
            "capture_binding": (
                "The simulator recorder captures each real RGB frame after the action and before automatic reset. "
                "Gym records that cached frame. The sidecar and HDF5 bind action indices and counts; "
                "initial/terminal PNGs retain actual renderer pixels, with file and decoded-RGB hashes. "
                "The raw MP4 terminal frame must match the pre-reset PNG within explicit encoding tolerances."
            ),
        },
        "visual_task_qualification": {
            "npa_status": ["implemented", "upstream_alpha"],
            "environments": ["gr1_open_microwave"],
            "policy_adapters": ["replay", "rsl_rl"],
            "requires": "Current-run JSONL and simulator HDF5 agree on task success; numeric success_rate is greater than zero; replay source actions are measurably nonzero.",
            "microwave_final_openness_greater_than": MICROWAVE_SUCCESS_THRESHOLD,
            "microwave_minimum_peak_minus_initial_openness": MICROWAVE_MINIMUM_OPENNESS_DELTA,
            "visual_interval": "Coherent denoised motion is measured from the first door progress through its first crossing of the upstream success threshold, excluding subsequent reset or idle frames.",
            "other_environments": "Ordinary scored evaluation remains implemented; nonzero-policy visual qualification has no task-specific binding and is unsupported.",
            "zero_action": "Viewport baseline only; task-qualified visual evidence is not claimed.",
            "live_validation": "Consult the external readiness record for the exact image digest, target hardware, execution device, and task. The baked implementation status does not establish a physical-GPU result.",
        },
        "camera_observation_video": {
            "npa_status": ["unsupported", "upstream_alpha"],
            "limitation": "Upstream supports it, but NPA currently requests only the viewport recorder.",
        },
        "viewport_only_camera_isolation": {
            "npa_status": ["implemented", "upstream_alpha"],
            "behavior": (
                "Kit and environment render support remain enabled for viewport capture, "
                "while NPA's viewport-only source patch masks only unused "
                "embodiment-mounted camera observations."
            ),
        },
        "b200": "State-only evaluation; B200 has no RT cores and carries no video claim.",
        "execution_devices": {
            "cuda:0": "Default GPU physics and policy execution path.",
            "cpu": (
                "The canonical RTX GR1 replay workflow uses CPU physics and replay tensors "
                "following upstream demonstration guidance, while the RTX GPU renders the viewport. "
                "This is an execution choice; qualification requires digest-specific task and visual evidence."
            ),
        },
        "agent_ui": "Use the authenticated native video renderer; Arena does not emit a truthful native Rerun .rrd.",
    },
}


def capabilities() -> dict[str, Any]:
    """Return the pinned upstream surface with explicit qualification boundaries.

    Args:
        None.

    Returns:
        An independent, JSON-compatible capability manifest.

    Raises:
        None.
    """
    return copy.deepcopy(_CAPABILITY_MANIFEST)
