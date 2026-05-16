"""Franka Panda pick-and-place environment using Genesis physics engine.

This is a gym-style parallel environment for training an RL teacher with
privileged state observations. It also supports camera rendering for demo
generation (teacher rollouts with domain randomization).

Requires GPU (L40S or better) and Genesis installed:
    pip install genesis-world

API reference: https://genesis-world.readthedocs.io/
Based on: examples/manipulation/grasp_env.py, examples/locomotion/go2_env.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class EnvConfig:
    """Configuration for the pick-and-place environment."""

    n_envs: int = 4096
    enable_cameras: bool = False
    domain_randomize: bool = False
    dt: float = 0.01
    substeps: int = 2
    max_episode_steps: int = 500
    camera_res: tuple[int, int] = (480, 640)  # (height, width)
    camera_fps: int = 20
    # Reward weights
    approach_weight: float = 1.0
    grasp_weight: float = 2.0
    place_weight: float = 3.0
    success_bonus: float = 10.0
    # Distance scaling for exponential rewards (lower = gentler gradient at
    # long range; default 5.0 gives near-zero reward at starting distance
    # ~0.58m, so 2.0 is better for initial learning)
    approach_scale: float = 5.0
    place_scale: float = 5.0
    # Cube
    cube_size: float = 0.04
    cube_init_pos: tuple[float, float, float] = (0.5, 0.0, 0.04)
    # Target zone
    target_pos: tuple[float, float, float] = (0.5, 0.3, 0.04)
    target_threshold: float = 0.05
    # Domain randomization ranges
    cube_pos_noise: float = 0.1
    friction_range: tuple[float, float] = (0.3, 1.2)
    # Action space: "joint" = 8D (7 delta joint positions + 1 gripper),
    # "cartesian" = 4D (delta xyz + gripper). Cartesian uses damped
    # least-squares IK to resolve end-effector deltas to joint targets.
    action_space: str = "joint"
    # Action scaling: clamp raw policy output to ±action_scale before
    # applying. In joint mode, clamps radians; in cartesian mode, clamps
    # meters. 0 = no clamping.
    action_scale: float = 0.0
    # Damping factor for Cartesian IK (higher = more stable near
    # singularities but less precise). Only used when action_space="cartesian".
    ik_damping: float = 0.05
    # PD gains (from Genesis grasp_env.py example)
    kp: tuple[float, ...] = (4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100)
    kv: tuple[float, ...] = (450, 450, 350, 350, 200, 200, 200, 10, 10)


# Franka Panda home joint configuration (ready pose)
FRANKA_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]


class FrankaPickPlaceEnv:
    """Genesis-based parallel pick-and-place environment for Franka Panda.

    Observations:
        Privileged (teacher): joint_pos (9,) + gripper_state (1,) +
            object_pose (7,) + contact_flags (2,) + goal_pos (3,) = 22
        Camera (student): workspace RGB (H, W, 3) + wrist RGB (H, W, 3) +
            joint_pos (9,) + gripper_state (1,)

    Actions (depends on action_space config):
        joint:     delta joint positions (7,) + gripper command (1,) = 8
        cartesian: delta xyz (3,) + gripper command (1,) = 4
    """

    N_JOINTS = 7
    N_GRIPPER = 2  # Franka has 2 finger joints
    N_DOFS = 9  # 7 arm + 2 gripper
    N_PRIV_OBS = 22  # joints(9) + gripper(1) + obj_pose(7) + contacts(2) + goal(3)
    N_STATE = 10  # joints(9) + gripper(1) — what a real robot has

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.n_envs = self.cfg.n_envs
        self.device: str = "cuda"

        if self.cfg.action_space not in ("joint", "cartesian"):
            raise ValueError(
                f"action_space must be 'joint' or 'cartesian', "
                f"got '{self.cfg.action_space}'"
            )
        # Action dimension depends on action space
        if self.cfg.action_space == "cartesian":
            self._n_actions = 4  # delta xyz (3) + gripper (1)
        else:
            self._n_actions = 8  # delta joints (7) + gripper (1)

        self._step_count: torch.Tensor | None = None
        self._scene = None
        self._robot = None
        self._cube = None
        self._target_pos: torch.Tensor | None = None
        self._workspace_cam = None
        self._wrist_cam = None

        # Link references (populated after build)
        self._hand_link = None
        self._left_finger_link = None
        self._right_finger_link = None

        self._build_scene()

        # Precompute IK damping matrix for cartesian mode
        if self.cfg.action_space == "cartesian":
            lam = self.cfg.ik_damping
            self._ik_damping = (lam ** 2) * torch.eye(
                3, device=self.device, dtype=torch.float32,
            ).unsqueeze(0)  # (1, 3, 3) — broadcasts over n_envs

    def _build_scene(self) -> None:
        """Construct the Genesis scene with robot, cube, cameras."""
        import genesis as gs

        if not gs._initialized:
            gs.init(backend=gs.gpu)

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.dt,
                substeps=self.cfg.substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=True,
                batch_dofs_info=True,   # Required for per-env DR of kp/kv
                batch_links_info=True,  # Required for per-env DR of mass/friction
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                rendered_envs_idx=[0],
            ),
            show_viewer=False,
        )

        # Ground plane
        self._scene.add_entity(gs.morphs.Plane())

        # Franka Panda robot (MJCF from Genesis built-in assets)
        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
        )

        # Cube to pick and place
        self._cube = self._scene.add_entity(
            gs.morphs.Box(
                size=(self.cfg.cube_size, self.cfg.cube_size, self.cfg.cube_size),
                pos=self.cfg.cube_init_pos,
            ),
            surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0)),
        )

        # Cameras (only if needed — rendering slows simulation significantly)
        if self.cfg.enable_cameras:
            h, w = self.cfg.camera_res
            self._workspace_cam = self._scene.add_camera(
                res=(w, h),  # Genesis uses (width, height)
                pos=(1.0, 0.0, 0.8),
                lookat=(0.5, 0.0, 0.0),
                fov=60,
            )
            self._wrist_cam = self._scene.add_camera(
                res=(w, h),
                pos=(0.4, 0.0, 0.4),
                lookat=(0.5, 0.0, 0.0),
                fov=90,
            )

        # Build scene with parallel environments
        self._scene.build(n_envs=self.n_envs)

        # Cache link references (must be after build)
        self._hand_link = self._robot.get_link("hand")
        self._left_finger_link = self._robot.get_link("left_finger")
        self._right_finger_link = self._robot.get_link("right_finger")

        # Set PD gains (must be after build, per Genesis docs)
        self._robot.set_dofs_kp(list(self.cfg.kp))
        self._robot.set_dofs_kv(list(self.cfg.kv))

        # Set force limits for Franka
        lower = [-87, -87, -87, -87, -12, -12, -12, -100, -100]
        upper = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        self._robot.set_dofs_force_range(lower, upper)

        # Target positions per env (may be randomized)
        self._target_pos = torch.tensor(
            self.cfg.target_pos, device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.n_envs, -1).clone()

        self._step_count = torch.zeros(self.n_envs, device=self.device, dtype=torch.long)

        # Precompute home position tensor
        self._home_qpos = torch.tensor(
            FRANKA_HOME, device=self.device, dtype=torch.float32,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Reset environments and return initial observations.

        Args:
            env_ids: Optional tensor of environment indices to reset.
                If None, resets all environments.

        Returns:
            Observation dict with privileged state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        n = len(env_ids)

        # Reset robot to home position with zero velocity
        home = self._home_qpos.unsqueeze(0).expand(n, -1)
        self._robot.set_dofs_position(home, envs_idx=env_ids)
        self._robot.set_dofs_velocity(
            torch.zeros(n, self.N_DOFS, device=self.device),
            envs_idx=env_ids,
        )

        # Reset cube position (with optional randomization)
        cube_pos = torch.tensor(
            self.cfg.cube_init_pos, device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(n, -1).clone()

        if self.cfg.domain_randomize:
            noise = (torch.rand(n, 3, device=self.device) - 0.5) * 2.0
            noise *= self.cfg.cube_pos_noise
            noise[:, 2] = 0.0  # Keep cube on table surface
            cube_pos += noise

        cube_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(n, -1)
        self._cube.set_pos(cube_pos, envs_idx=env_ids)
        self._cube.set_quat(cube_quat, envs_idx=env_ids)

        self._step_count[env_ids] = 0

        if self.cfg.domain_randomize:
            self.randomize_domain(env_ids)

        return self.get_privileged_obs()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with actions.

        Args:
            actions: (n_envs, act_dim) tensor.
                joint mode:     7 delta joint positions + 1 gripper = 8
                cartesian mode: 3 delta xyz + 1 gripper = 4

        Returns:
            (obs, reward, done, info) tuple.
        """
        assert actions.shape == (self.n_envs, self._n_actions), (
            f"Expected actions shape ({self.n_envs}, {self._n_actions}), "
            f"got {actions.shape}"
        )

        if self.cfg.action_space == "cartesian":
            delta_xyz = actions[:, :3]   # (n_envs, 3)
            gripper_cmd = actions[:, 3:]  # (n_envs, 1)

            if self.cfg.action_scale > 0:
                delta_xyz = delta_xyz.clamp(
                    -self.cfg.action_scale, self.cfg.action_scale,
                )

            # Resolve Cartesian delta to joint delta via IK
            joint_deltas = self._ik_resolve_delta(delta_xyz)
        else:
            joint_deltas = actions[:, :self.N_JOINTS]
            gripper_cmd = actions[:, self.N_JOINTS:]  # (n_envs, 1)

            if self.cfg.action_scale > 0:
                joint_deltas = joint_deltas.clamp(
                    -self.cfg.action_scale, self.cfg.action_scale,
                )

        # Current joint positions
        current_pos = self._robot.get_dofs_position()  # (n_envs, 9)
        joint_targets = current_pos[:, :self.N_JOINTS] + joint_deltas

        # Gripper: cmd > 0 → open (0.04), cmd <= 0 → close (0.0)
        gripper_val = torch.where(
            gripper_cmd > 0,
            torch.full_like(gripper_cmd, 0.04),
            torch.zeros_like(gripper_cmd),
        )
        gripper_targets = gripper_val.expand(-1, self.N_GRIPPER)

        # Apply PD position control
        targets = torch.cat([joint_targets, gripper_targets], dim=-1)
        self._robot.control_dofs_position(targets)

        # Step physics
        self._scene.step()
        self._step_count += 1

        # Compute observations, reward, done
        obs = self.get_privileged_obs()
        reward = self._compute_reward(obs)
        done = self._compute_done(obs)

        info = {
            "success": self._is_success(obs),
            "step_count": self._step_count.clone(),
        }

        # Auto-reset done environments
        done_ids = torch.where(done)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)

        return obs, reward, done, info

    def get_privileged_obs(self) -> dict[str, torch.Tensor]:
        """Get privileged observations (teacher only).

        Returns dict with:
            joint_pos: (n_envs, 9) — 7 arm + 2 gripper DOFs
            gripper_state: (n_envs, 1) — mean gripper finger opening
            object_pose: (n_envs, 7) — xyz + quaternion (w,x,y,z)
            contact_flags: (n_envs, 2) — left/right finger contact with cube
            goal_position: (n_envs, 3) — target placement position
            ee_pos: (n_envs, 3) — end-effector (hand link) Cartesian position
            flat: (n_envs, 22) — concatenated privileged state vector
                (ee_pos is NOT included in flat to preserve the 22-dim
                policy interface and checkpoint compatibility)
        """
        joint_pos = self._robot.get_dofs_position()  # (n_envs, 9)
        gripper_state = joint_pos[:, self.N_JOINTS:].mean(dim=-1, keepdim=True)

        cube_pos = self._cube.get_pos()  # (n_envs, 3)
        cube_quat = self._cube.get_quat()  # (n_envs, 4) — (w,x,y,z)
        object_pose = torch.cat([cube_pos, cube_quat], dim=-1)  # (n_envs, 7)

        # Contact detection between gripper fingers and cube
        contact_flags = self._get_contact_flags()  # (n_envs, 2)

        ee_pos = self._hand_link.get_pos()  # (n_envs, 3)

        obs = {
            "joint_pos": joint_pos,
            "gripper_state": gripper_state,
            "object_pose": object_pose,
            "contact_flags": contact_flags,
            "goal_position": self._target_pos,
            "ee_pos": ee_pos,
            "flat": torch.cat([
                joint_pos,
                gripper_state,
                object_pose,
                contact_flags,
                self._target_pos,
            ], dim=-1),
        }
        return obs

    def get_camera_obs(self) -> dict[str, np.ndarray | torch.Tensor]:
        """Get camera-only observations (student / demo recording).

        Returns dict with:
            workspace: (n_envs, H, W, 3) uint8 RGB — numpy array
            wrist: (n_envs, H, W, 3) uint8 RGB — numpy array
            joint_pos: (n_envs, 9) — torch tensor
            gripper_state: (n_envs, 1) — torch tensor

        Genesis cameras render one env at a time, so this loops over
        all envs and stacks the results.
        """
        if not self.cfg.enable_cameras:
            raise RuntimeError(
                "Cameras not enabled. Create env with enable_cameras=True."
            )

        ws_list = []
        wr_list = []
        for env_idx in range(self.cfg.n_envs):
            self._workspace_cam._env_idx = env_idx
            self._wrist_cam._env_idx = env_idx
            ws_rgb, _, _, _ = self._workspace_cam.render()
            wr_rgb, _, _, _ = self._wrist_cam.render()
            ws_list.append(np.asarray(ws_rgb))
            wr_list.append(np.asarray(wr_rgb))

        joint_pos = self._robot.get_dofs_position()
        gripper_state = joint_pos[:, self.N_JOINTS:].mean(dim=-1, keepdim=True)

        return {
            "workspace": np.stack(ws_list),
            "wrist": np.stack(wr_list),
            "joint_pos": joint_pos,
            "gripper_state": gripper_state,
        }

    def randomize_domain(self, env_ids: torch.Tensor | None = None) -> None:
        """Randomize physical properties for sim-to-real transfer.

        Randomizes:
            - Cube position (already done in reset)
            - Friction ratios on cube and robot links
            - PD gain perturbations
        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        n = len(env_ids)

        # Randomize friction on the cube
        lo, hi = self.cfg.friction_range
        friction_ratio = lo + torch.rand(n, self._cube.n_links, device=self.device) * (hi - lo)
        self._cube.set_friction_ratio(
            friction_ratio,
            links_idx_local=range(self._cube.n_links),
            envs_idx=env_ids,
        )

        # Randomize friction on robot fingers
        finger_friction = lo + torch.rand(n, 2, device=self.device) * (hi - lo)
        left_idx = self._left_finger_link.idx_local
        right_idx = self._right_finger_link.idx_local
        self._robot.set_friction_ratio(
            finger_friction,
            links_idx_local=[left_idx, right_idx],
            envs_idx=env_ids,
        )

    def _compute_reward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pick-and-place reward function.

        Components:
            1. Approach: negative distance from gripper to cube
            2. Grasp: bonus for finger-cube contact
            3. Place: negative distance from cube to target
            4. Success: large bonus when cube is at target and stable
        """
        ee_pos = self._hand_link.get_pos()  # (n_envs, 3)
        cube_pos = self._cube.get_pos()  # (n_envs, 3)

        # 1. Approach reward
        dist_to_cube = torch.norm(ee_pos - cube_pos, dim=-1)
        if self.cfg.approach_scale > 0:
            # Exponential: strong gradient near cube, weak at distance
            approach_reward = torch.exp(-self.cfg.approach_scale * dist_to_cube) * self.cfg.approach_weight
        else:
            # Linear: uniform gradient at all distances, clamped to [0, weight]
            approach_reward = (1.0 - dist_to_cube).clamp(min=0.0) * self.cfg.approach_weight

        # 2. Grasp reward — bonus for finger contact
        contacts = obs["contact_flags"]
        grasp_reward = contacts.sum(dim=-1) * self.cfg.grasp_weight

        # 3. Place reward — only meaningful when grasping
        dist_to_target = torch.norm(cube_pos - self._target_pos, dim=-1)
        is_grasping = contacts.sum(dim=-1) > 0.5
        place_reward = torch.where(
            is_grasping,
            torch.exp(-self.cfg.place_scale * dist_to_target) * self.cfg.place_weight,
            torch.zeros_like(dist_to_target),
        )

        # 4. Success bonus
        success = self._is_success(obs).float()
        success_reward = success * self.cfg.success_bonus

        return approach_reward + grasp_reward + place_reward + success_reward

    def _compute_done(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Episode is done on success or timeout."""
        timeout = self._step_count >= self.cfg.max_episode_steps
        success = self._is_success(obs)
        # Also terminate on simulation errors
        error_mask = self._scene.rigid_solver.get_error_envs_mask()
        return timeout | success | error_mask

    def _is_success(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Check if cube is at target position and stable (low velocity)."""
        cube_pos = obs["object_pose"][:, :3]
        dist = torch.norm(cube_pos - self._target_pos, dim=-1)
        at_target = dist < self.cfg.target_threshold

        # Check cube is not moving fast (stable placement)
        cube_vel = self._cube.get_vel()  # (n_envs, 3)
        speed = torch.norm(cube_vel, dim=-1)
        stable = speed < 0.1

        return at_target & stable

    def _get_contact_flags(self) -> torch.Tensor:
        """Detect contact between gripper fingers and the cube.

        Uses net contact force magnitude on finger links as a proxy for
        binary contact detection.

        Returns (n_envs, 2) tensor with left/right finger contact floats.
        """
        # Get net contact force on all robot links: (n_envs, n_links, 3)
        net_forces = self._robot.get_links_net_contact_force()

        left_idx = self._left_finger_link.idx_local
        right_idx = self._right_finger_link.idx_local

        left_force_mag = torch.norm(net_forces[:, left_idx, :], dim=-1)
        right_force_mag = torch.norm(net_forces[:, right_idx, :], dim=-1)

        # Threshold: force > 0.1N means contact
        left_contact = (left_force_mag > 0.1).float()
        right_contact = (right_force_mag > 0.1).float()

        return torch.stack([left_contact, right_contact], dim=-1)

    def _ik_resolve_delta(self, delta_xyz: torch.Tensor) -> torch.Tensor:
        """Resolve Cartesian position delta to joint delta via damped least-squares IK.

        Uses the position Jacobian of the hand link:
            Δq = J^T (J J^T + λ²I)^{-1} Δx

        The damping term (λ) prevents singularity blow-up when the Jacobian
        loses rank (e.g. near workspace boundaries or stretched-arm configs).

        Args:
            delta_xyz: (n_envs, 3) desired end-effector position change in meters.

        Returns:
            (n_envs, 7) joint position deltas in radians.
        """
        # Get full Jacobian for the hand link: (n_envs, 6, n_dofs)
        # Rows 0-2 are the position Jacobian, rows 3-5 are orientation.
        J_full = self._robot.get_jacobian(link=self._hand_link)
        J_pos = J_full[:, :3, :self.N_JOINTS]  # (n_envs, 3, 7)

        # Damped least-squares: J^T (J J^T + λ²I)^{-1} Δx
        JJT = torch.bmm(J_pos, J_pos.transpose(1, 2))  # (n_envs, 3, 3)
        rhs = delta_xyz.unsqueeze(-1)  # (n_envs, 3, 1)
        delta_q = torch.bmm(
            J_pos.transpose(1, 2),
            torch.linalg.solve(JJT + self._ik_damping, rhs),
        )
        return delta_q.squeeze(-1)  # (n_envs, 7)

    @property
    def obs_dim(self) -> int:
        return self.N_PRIV_OBS

    @property
    def act_dim(self) -> int:
        return self._n_actions

    @property
    def state_dim(self) -> int:
        return self.N_STATE
