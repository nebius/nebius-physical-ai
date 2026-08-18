"""Public synthetic Antioch cartpole scenario producing an explicit offline episode."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import antioch
import numpy as np

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

SOURCE_CONTRACT_SHA256 = (
    "00481edd23e2ae6555e8bf3cc4f2118b90ff8a44c0fc57105501e0bc72891aaf"
)


def _build_cartpole() -> tuple[SimulationContext, InteractiveScene, Articulation]:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

    @configclass
    class CartpoleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg()
        )
        light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0)
        )
        cartpole: ArticulationCfg = CARTPOLE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Cartpole"
        )

    simulation = SimulationContext(SimulationCfg(dt=1.0 / 60.0))
    scene = InteractiveScene(CartpoleSceneCfg(num_envs=1, env_spacing=2.0))
    simulation.reset()
    return simulation, scene, scene["cartpole"]


def _state_image(state: np.ndarray, *, alternate: bool) -> np.ndarray:
    """Create a deterministic diagnostic RGB observation from simulated state."""

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cart = int(np.clip(32 + state[0] * 12, 5, 58))
    angle = float(state[1])
    pole_x = int(np.clip(cart + math.sin(angle) * 22, 0, 63))
    pole_y = int(np.clip(46 - math.cos(angle) * 22, 0, 63))
    color = (80, 220, 140) if not alternate else (220, 140, 80)
    image[45:50, max(0, cart - 5) : min(64, cart + 6)] = color
    samples = np.linspace(0, 1, 32)
    xs = np.clip(np.rint(cart + (pole_x - cart) * samples), 0, 63).astype(int)
    ys = np.clip(np.rint(46 + (pole_y - 46) * samples), 0, 63).astype(int)
    image[ys, xs] = (255, 255, 255)
    return image


@antioch.scenario(tags=["npa-policy-data"])
def cartpole_offline_policy_episode(
    run: antioch.ScenarioRun, seed: int = antioch.param(17, ge=0)
) -> None:
    """Run a deterministic feedback controller and publish one validated trajectory."""

    import torch

    torch.manual_seed(seed)
    simulation, scene, robot = _build_cartpole()
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    workspace: list[np.ndarray] = []
    wrist: list[np.ndarray] = []
    for step in range(120):
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        joint_vel = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
        state = np.concatenate((joint_pos, joint_vel))
        effort = float(
            np.clip(-7.0 * state[1] - 1.5 * state[3] - 0.4 * state[0], -12.0, 12.0)
        )
        robot.set_joint_effort_target_index(
            target=torch.tensor([[effort]], device=robot.device), joint_ids=[0]
        )
        scene.write_data_to_sim()
        simulation.step()
        scene.update(simulation.get_physics_dt())
        if step % 3 == 0:
            states.append(state)
            # Model the cart as two opposing actuator commands. Their
            # difference is the exact effort applied above, so both exported
            # policy channels have a physical meaning and no synthetic padding
            # is introduced merely to satisfy a trainer shape.
            actions.append(
                np.array([max(effort, 0.0), max(-effort, 0.0)], dtype=np.float32)
            )
            rewards.append(-abs(float(state[1])))
            workspace.append(_state_image(state, alternate=False))
            wrist.append(_state_image(state, alternate=True))

    length = len(states)
    provenance = {
        "schema_name": "npa.antioch.episode.v1",
        "scenario": "cartpole_offline_policy_episode",
        "case": "",
        "seed": seed,
        "parameters": {"seed": seed, "physics_steps": 120},
        "engine_version": "isaac-lab-3.0",
        "sdk_version": "0.3.47",
        "source_sha256": SOURCE_CONTRACT_SHA256,
        "assets_sha256": {},
        "observation_schema": [
            "cart_position",
            "pole_angle",
            "cart_velocity",
            "pole_velocity",
            "diagnostic_rgb_workspace",
            "diagnostic_rgb_wrist",
        ],
        "action_schema": ["cart_effort_positive", "cart_effort_negative"],
        "fps": 20,
    }
    target = Path("/tmp/npa-antioch-cartpole-episode.npz")
    np.savez(
        target,
        observation_state=np.stack(states),
        observation_image_workspace=np.stack(workspace),
        observation_image_wrist=np.stack(wrist),
        action=np.stack(actions),
        reward=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray([False] * (length - 1) + [True]),
        truncated=np.zeros(length, dtype=bool),
        timestamp=np.arange(length, dtype=np.float64) / 20.0,
        provenance=np.array(json.dumps(provenance, sort_keys=True)),
    )
    run.add_artifact(target, name="policy_trajectory", content_type="application/x-npz")
    final_state = states[-1]
    run.add_result("episode_frames", length)
    run.check(
        "trajectory state stayed finite", bool(np.isfinite(np.stack(states)).all())
    )
    run.check(
        "cart stayed within a safe diagnostic range", abs(float(final_state[0])) < 5.0
    )
