"""Instrument the native Isaac manager environment without replacing physics or PPO."""

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
        return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        action = self.action_manager.get_term("pre_trained_policy_action")
        action.low_level_actions[env_ids] = 0
        action._raw_actions[env_ids] = 0
        action._low_level_obs_manager.reset(env_ids)
        action._low_level_action_term.reset(env_ids)
        if len(env_ids) == self.num_envs:
            action._counter = 0
        if hasattr(self, "npa_contacts"):
            self.npa_contacts.reset(env_ids)
        self.command_manager.compute(dt=0.0)
