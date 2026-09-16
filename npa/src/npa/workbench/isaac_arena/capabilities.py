"""Expose Arena features, required inputs, and digest-scoped validation status."""

import copy
from typing import Any

from .action_evidence import ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE
from .identity import (
    ISAAC_ARENA_VERSION,
    ISAAC_ARENA_REVISION,
    LIGHTWHEEL_SDK_VERSION,
    CAPABILITIES_SCHEMA,
)
from .task_progress import task_progress_capabilities
from .video_evidence import video_acceptance_thresholds

_TASK_PROGRESS_CAPABILITIES = task_progress_capabilities()
_TASK_QUALIFIED_POLICIES = sorted(
    {
        policy
        for adapter in _TASK_PROGRESS_CAPABILITIES
        for policy in adapter["supported_policy_types"]
    }
)

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
                    "Isaac Lab reset_to(is_relative=True), then executes the exact source "
                    "prefix through the requested native episode terminal. Unused source "
                    "actions remain outside the scored/captured episode. Synthetic padding, "
                    "repetition, and appended final-action holds are forbidden; naturally "
                    "stable source tails remain subject to the registered adapter limit."
                ),
                "held_tail_definition": {
                    "metric": "maximum absolute component delta between adjacent actions",
                    "maximum_delta": ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE,
                },
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
    "upstream_workflows": {
        "agentic_environment_generation": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_entrypoints": [
                "isaaclab_arena_examples/agentic_environment_generation/cli_runner.py",
                "isaaclab_arena_examples/agentic_environment_generation/gui_runner.py",
            ],
            "required_inputs": [
                "prompt for resolve/full or graph specification for build/prim_tree; schema/catalog need neither",
                "authorized model access for prompt resolution; authorized assets for scene building",
            ],
            "limitation": "NPA does not expose the experimental generation CLI, review GUI, or generated-environment build workflow.",
        },
        "experiment_runner": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_entrypoints": [
                "isaaclab_arena/evaluation/experiment_runner.py",
                "osmo/submit_arena_experiment.py",
            ],
            "required_inputs": [
                "experiment configuration, compatible policies, and task assets"
            ],
            "limitation": "NPA's SkyPilot evaluation workflows do not expose upstream experiment orchestration or its OSMO backend.",
        },
        "sensitivity_analysis": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_entrypoints": [
                "isaaclab_arena/analysis/sensitivity/generate_report.py"
            ],
            "required_inputs": ["episode outcomes with recorded variation factors"],
            "limitation": "NPA retains ordinary evaluation metrics but does not invoke upstream posterior estimation or sensitivity reports.",
        },
        "teleoperation": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_documentation": "docs/pages/example_workflows/static_manipulation/step_2_teleoperation.rst",
            "required_inputs": [
                "registered task, compatible teleoperation device, and runtime assets"
            ],
            "limitation": "Upstream records demonstrations through Isaac Lab; the Arena workbench exposes no teleoperation or demonstration-recording command.",
        },
        "data_generation": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_documentation": "docs/pages/example_workflows/static_manipulation/step_3_data_generation.rst",
            "required_inputs": [
                "source demonstrations, task annotations, and compatible Isaac Lab Mimic configuration"
            ],
            "limitation": "Evaluation replay does not implement upstream demonstration annotation or Mimic dataset generation.",
        },
        "imitation_learning": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_documentation": "docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst",
            "required_inputs": [
                "demonstration dataset, compatible training configuration, and authorized model access"
            ],
            "limitation": "The upstream GR00T conversion and fine-tuning workflow requires its separate training environment; NPA Arena does not invoke it.",
        },
        "reinforcement_learning": {
            "npa_status": ["unsupported", "input_required", "upstream_alpha"],
            "upstream_documentation": "docs/pages/example_workflows/reinforcement_learning/step_2_policy_training.rst",
            "required_inputs": [
                "registered Arena task and compatible Isaac Lab training configuration"
            ],
            "limitation": "NPA Arena can evaluate a supplied RSL-RL checkpoint but does not invoke upstream policy training.",
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
            "purpose": "Provide headless EGL/Vulkan and OptiX dependencies missing from a viewport run's target.",
            "license": "NVIDIA driver package terms",
            "baked": False,
            "delivery": "runtime_fetch_exact_driver_match",
            "source": "Ubuntu signed NVIDIA driver archive",
            "installed_on_node": False,
            "redistribution": False,
            "scope": "Viewport runs only; native graphics is preferred. Extracted libraries use run-private scratch; missing OptiX weights use the verified private container root overlay.",
            "headless_icd": "NPA derives a canonical EGL ICD manifest from the validated package manifest; it never mutates packaged bytes or the node.",
            "native_readiness": "NVIDIA EGL/Vulkan and libnvoptix.so.1 load; /usr/share/nvidia/nvoptix.bin is readable, regular and nonempty.",
            "optix_weights": {
                "path": "/usr/share/nvidia/nvoptix.bin",
                "image_directory": "empty, owned by the non-root runtime user",
                "fallback_placement": "After exact signed package identity validation, copy into the verified private root overlay with no symlink or submount destination; atomically publish without overwriting and verify the hash.",
                "retention": "worker container lifetime only",
                "evidence": "native or container-overlay placement, SHA-256 and byte size; no payload bytes published",
                "denoising_success": "Library, file and setting readiness do not establish successful denoising; runtime renderer errors reject video qualification.",
            },
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
            "result": "After evaluation setup succeeds, handled runtime/evidence errors write a failed result.json and hashes for whatever artifacts were produced, then attempt publication. A storage publication failure retains a mode-0700 private local copy.",
            "partial_artifacts": "Logs, scalar phase journals, raw video, capture sidecar, and initial or terminal PNGs are retained only if created. An unfinished recorder buffer is separately marked unscored; it never becomes a completed episode or task-success result.",
            "retention_limitation": "Publication-failure copies have no automatic age, count, or byte pruning. The reported local path must be recovered or removed by the operator; repeated failures can consume worker disk.",
        },
        "phase_diagnostics": {
            "artifact": "simulator-phases-rank*.jsonl, when emitted",
            "fields": "fixed phase/event labels, monotonic timestamps, rank, action/render counters, and observed readiness booleans",
            "unavailable_binding": "An unavailable method-phase event leaves an immutable native binding untouched; enclosing simulator calls can still be observed. It does not claim execution or progress.",
            "scope": "Best-effort observations of policy, Pink IK, environment, and capture progress; no arguments, input arrays, exception contents, timeout decisions, or task-success claims.",
        },
        "early_failure": "Request validation or input/setup failures may occur before any result tree exists; an interrupted worker may leave only workflow logs. Completed-episode JSONL, scored HDF5, and HTML reports are not guaranteed on failure.",
        "successful_video_qualification": [
            "raw viewport MP4 and a labeled denoised half-speed derivative with identical frame count",
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
            "renderer": {
                "mode": "RaytracedLighting",
                "legacy_mode_enabled": True,
                "rt2_enabled": False,
                "path_tracing_enabled": False,
                "antialiasing": "TAA",
                "dlss_execution_mode": "quality",
                "dl_denoiser_enabled": True,
                "frame_generation_enabled": False,
                "minimum_settling_renders": 8,
                "stochastic_accumulation": False,
                "reason": "Selecting legacy RTX while disabling RT2, interactive path tracing, and generated frames prevents Isaac Sim 6 from remapping or inventing action frames. Supported temporal anti-aliasing, the DL denoiser, and at least eight consecutive ready physics-frozen settling renders address single-frame RTX grain; task-bound temporal median and coherent tracking remain independent acceptance checks.",
            },
            "graphics_userspace": (
                "Native NVIDIA EGL/Vulkan, libnvoptix.so.1 and readable nonempty OptiX weights are preferred. "
                "Missing dependencies require the exact loaded-driver package from Ubuntu's signed archive. "
                "Libraries and a validated headless EGL ICD use private scratch; missing nvoptix.bin is "
                "atomically copied into the verified private container root overlay for the worker lifetime. "
                "Nothing is installed on the node, baked, or redistributed. Only identities, hashes and "
                "placement evidence are retained; actual renderer errors reject video qualification."
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
            "environments": [
                item["environment"] for item in _TASK_PROGRESS_CAPABILITIES
            ],
            "progress_adapters": _TASK_PROGRESS_CAPABILITIES,
            "policy_adapters": _TASK_QUALIFIED_POLICIES,
            "requires": "Current-run JSONL and simulator HDF5 agree on task success; numeric success_rate is greater than zero; a registered policy adapter has measured finite, nonzero, varied policy-to-environment actions without synthetic padding or a dominant numerically held tail; replay actions exactly match the prepared private tensor prefix through the native terminal.",
            "visual_interval": "The sole current task-progress adapter is gr1_open_microwave. It declares its visual interval strategy, leading context, progress signal, normalized task-object region, and spatial association radius; admits bounded approach/contact context; requires accepted coherent motion to overlap the exact first-progress-to-success interval; and requires connected monotonic structural change inside the microwave workspace adjacent to that tracked motion.",
            "shared_contract": "Measured policy-to-environment actions without synthetic padding or a dominant numerically held tail, one native successful scored episode, registered environment/policy progress, exact full-episode simulator capture steps, a consistent capture/PNG/raw/evidence frame-hash chain, and noise-resistant video motion temporally and spatially bound to adapter-selected progress must all describe the same episode. The task adapter explicitly selects the visual context, task region, signal, and association radius.",
            "other_environments": "Ordinary scored evaluation remains implemented; nonzero-policy visual qualification requires an explicitly registered task-progress adapter and is otherwise unsupported.",
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
