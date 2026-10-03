"""Instrument the native Isaac manager environment without replacing physics or PPO."""

from contextlib import nullcontext

from isaaclab.envs import ManagerBasedRLEnv

from npa.workflows.navigation.reference_contacts import ContactMeasurements


class ReferenceEnvironment(ManagerBasedRLEnv):
    """Run upstream native quadruped physics with per-substep contact measurements.

    Args:
        cfg: Native reference configuration.
        **kwargs: Gymnasium/native environment constructor arguments.
    Returns:
        A real ManagerBasedRLEnv usable by the RSL-RL wrapper.
    Raises:
        RuntimeError: Native physics, sensors or public robot assets fail to load.
    """

    def __init__(self, cfg, **kwargs):
        self.npa_probe = False
        self.npa_override_cases = None
        super().__init__(cfg, **kwargs)
        self.npa_contacts = ContactMeasurements(self)
        original = self.scene.update

        def record_contacts(dt):
            original(dt)
            self.npa_contacts.update()

        self.scene.update = record_contacts

    def step(self, action):
        """Advance the native environment and preserve contacts across decimation.

        Args:
            action: Learned high-level velocity actions.
        Returns:
            Native observations, rewards, termination flags and log data.
        Raises:
            RuntimeError: Native stepping fails.
        """
        self.npa_contacts.reset()
        evidence = self.npa_contacts.evidence
        context = evidence.interval() if evidence is not None else nullcontext()
        with context:
            return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._reset_controller(env_ids)
        if hasattr(self, "npa_contacts"):
            self.npa_contacts.reset(env_ids)
        self.command_manager.compute(dt=0.0)

    def _reset_controller(self, env_ids):
        action = self.action_manager.get_term("pre_trained_policy_action")
        action.low_level_actions[env_ids] = 0
        action._raw_actions[env_ids] = 0
        action._low_level_obs_manager.reset(env_ids)
        low_level = action._low_level_action_term
        low_level.reset(env_ids)
        robot = self.scene["robot"]
        target = robot.data.default_joint_pos.torch[env_ids][:, low_level._joint_ids]
        # Native reset calls write_data_to_sim(), which advances the actuator
        # network. Never seed that reset with the previous episode's targets.
        low_level.processed_actions[env_ids] = target
        robot.set_joint_position_target_index(
            target=target, joint_ids=low_level._joint_ids, env_ids=env_ids
        )
        if len(env_ids) == self.num_envs:
            action._counter = 0
