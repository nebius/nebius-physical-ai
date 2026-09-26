"""Configure the pinned public native quadruped for one shared collision scene."""

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import UsdFileCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import (
    NavigationEnvCfg,
)

from npa.workflows.navigation.reference_geometry import spawn_scene


@configclass
class ReferenceConfig(NavigationEnvCfg):
    """Carry explicit reset data with native Isaac manager configuration.

    Args:
        None.
    Returns:
        Configurable native public navigation task.
    Raises:
        None.
    """

    npa_cases: list = []
    npa_training: bool = True
    npa_scene_prim: str = "/World/Warehouse"
    npa_reference_controller_sha256: str = ""


def build_config(scene_file, scene_prim, num_envs, cases, training):
    """Register and configure the public reference with a single static scene.

    Args:
        scene_file: Exact local USDZ with metric static triangle colliders.
        scene_prim: Global reference root outside the robot clone namespaces.
        num_envs: Simultaneous independent robots.
        cases: Sealed reset cases for this stage's split.
        training: Enable training episode termination.
    Returns:
        ReferenceConfig for the pinned native runtime.
    Raises:
        ValueError: Native versions are outside the supported reference contract.
    """
    _register()
    config = ReferenceConfig()
    config.npa_cases, config.npa_training = cases, training
    config.npa_scene_prim = scene_prim
    config.scene.num_envs, config.scene.env_spacing = num_envs, 0.0
    config.scene.replicate_physics = True
    config.scene.filter_collisions = True
    config.scene.terrain = None
    config.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    config.scene.warehouse = AssetBaseCfg(
        prim_path=scene_prim,
        collision_group=-1,
        spawn=UsdFileCfg(usd_path=scene_file, func=spawn_scene),
    )
    _physics(config, num_envs)
    _perception(config, scene_prim)
    _task(config)
    return config


def _physics(config, num_envs):
    from isaaclab_tasks.utils.hydra import resolve_presets

    physics = resolve_presets(config.sim.physics)
    # Coincident clones enter broadphase before peer collision filtering. Native
    # 4,000-robot startup required 8,002,000 pairs; once that buffer was sized,
    # the complete broadphase required 168,010,000 found/lost aggregate pairs.
    # A measured triangle scene also required 278,600,488 collision-stack bytes
    # and 349,082 rigid patches. Reserve the next power of two for both buffers;
    # native overflow diagnostics still invalidate a run on more demanding data.
    pairs = num_envs * (num_envs + 1) // 2
    requirements = {
        "gpu_found_lost_pairs_capacity": 2 * pairs,
        "gpu_found_lost_aggregate_pairs_capacity": 64 * pairs,
        "gpu_total_aggregate_pairs_capacity": 2 * pairs,
        "gpu_collision_stack_size": 2**29,
        "gpu_max_rigid_patch_count": 2**19,
    }
    for name, required in requirements.items():
        capacity = 1 << (max(1, required) - 1).bit_length()
        setattr(physics, name, max(getattr(physics, name), capacity))
    config.sim.physics = physics
    config.sim.render_interval = config.decimation


def _register():
    import gymnasium as gym
    from importlib.metadata import version
    from npa.workflows.navigation.reference import TASK

    if version("isaaclab") != "3.0.0b2.post1" or version("isaacsim") != "6.0.1.0":
        raise ValueError(
            "reference requires Isaac Lab 3.0.0b2.post1 and Isaac Sim 6.0.1.0"
        )
    if version("rsl-rl-lib") != "5.0.1":
        raise ValueError("reference requires native RSL-RL 5.0.1")
    if TASK not in gym.registry:
        gym.register(
            id=TASK,
            entry_point="npa.workflows.navigation.reference_environment:ReferenceEnvironment",
            disable_env_checker=True,
            kwargs={
                "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.navigation.config.anymal_c.agents.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg"
            },
        )


def _perception(config, scene_prim):
    _support_sensor(config, scene_prim)
    config.scene.navigation_ranges = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[scene_prim + "/NpaRaycastMesh"],
        ray_alignment="base",
        max_distance=12.0,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3,
            vertical_fov_range=(-12.0, 12.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        update_period=config.sim.dt * config.decimation,
    )
    config.observations.policy.ranges = ObservationTermCfg(func=_mdp("static_ranges"))
    config.observations.policy.enable_corruption = False
    config.actions.pre_trained_policy_action.low_level_observations.enable_corruption = False


def _support_sensor(config, scene_prim):
    from npa.workflows.navigation.reference_validity import footprint_pattern

    config.scene.ground_support = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[scene_prim + "/NpaRaycastMesh"],
        ray_alignment="yaw",
        max_distance=1.2,
        pattern_cfg=patterns.PatternBaseCfg(func=footprint_pattern),
        update_period=0.0,
        debug_vis=False,
    )


def _task(config):
    from npa.workflows.navigation.reference_controller import create_action

    config.actions.pre_trained_policy_action.class_type = create_action
    config.events.reset_base = EventTermCfg(func=_mdp("reset_cases"), mode="reset")
    config.commands.pose_command.class_type = _mdp("FixedGoalCommand")
    config.commands.pose_command.debug_vis = False
    config.commands.pose_command.resampling_time_range = (1.0e9, 1.0e9)
    config.episode_length_s = 30.0
    config.rewards.progress = RewardTermCfg(func=_mdp("progress"), weight=3.0)
    config.rewards.obstacle = RewardTermCfg(func=_mdp("collision"), weight=-25.0)
    config.rewards.physical_failure = RewardTermCfg(
        func=_mdp("physical_failure"), weight=-25.0
    )
    config.rewards.termination_penalty.func = _mdp("failure")
    config.rewards.orientation_tracking = None
    config.terminations.base_contact = TerminationTermCfg(func=_mdp("terminate"))
    config.terminations.time_out = TerminationTermCfg(
        func=_mdp("timeout"), time_out=True
    )


def _mdp(name):
    # Native command implementations import terrain/USD; resolve them after Kit.
    return ResolvableString("npa.workflows.navigation.reference_mdp:" + name)
