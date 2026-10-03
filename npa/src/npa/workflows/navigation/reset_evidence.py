"""Retain focal native control values during the first interval of each probe."""

from contextlib import contextmanager
from functools import wraps
import hashlib
from importlib import metadata
import inspect
from pathlib import Path

import numpy as np

from npa.workflows.navigation.artifacts import write_json


def _copy(value, axis=0, index=0):
    from npa.workflows.navigation.reference_contacts import _tensor

    return _tensor(value).select(axis, index).detach().clone()


def _runtime(env):
    import torch

    packages = {}
    for name in ("torch", "isaaclab", "isaacsim", "warp-lang"):
        packages[name] = _package_record(name)
    return {
        "packages": packages,
        "loaded_source_sha256": _source_files(env),
        "fingerprint_scope": "Installed distribution RECORD bytes and inspectable Python class files, including inherited environment methods; not a native binary fingerprint. Missing distribution metadata is explicitly unavailable, never inferred from import success, and does not prevent observation.",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _package_record(name):
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"status": "unavailable", "version": None, "record_sha256": None}
    record = distribution.read_text("RECORD")
    return {
        "status": "available",
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.encode()).hexdigest()
        if record is not None
        else None,
    }


def _source_files(env):
    action = env.action_manager.get_term("pre_trained_policy_action")
    robot = env.scene["robot"]
    instances = [env, action, action._low_level_obs_manager, robot, robot.root_view]
    instances.extend(robot.actuators.values())
    sources = {}
    for instance in instances:
        for cls in type(instance).__mro__:
            source = _python_source(cls)
            if source is not None:
                sources[f"{cls.__module__}.{cls.__qualname__}"] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
    return sources


def _python_source(cls):
    try:
        source = inspect.getsourcefile(cls)
    except TypeError:
        return None
    return Path(source) if source is not None else None


class ResetEvidence:
    """Observe original calls without changing reset, controls or physics settings.

    Args:
        env: Native environment with initialized probe contact evidence.
        output: Native artifact directory.
        name: Current physical probe name.
    Returns:
        A context recorder for the focal robot's first control interval.
    Raises:
        ValueError: Native focal articulation ordering is unsupported.
    """

    def __init__(self, env, output, name):
        from npa.workflows.navigation.contact_surface import _root_indices

        self.env = env
        self.action = env.action_manager.get_term("pre_trained_policy_action")
        self.robot = env.scene["robot"]
        self.clock = env.npa_contacts.evidence.clock
        self.view = self.robot.root_view
        self.focal_index = _root_indices(env)[0]
        self.output = output / f"probe-{name}-reset"
        self.output.mkdir()
        self.events, self.arrays, self.patches = [], {}, []
        self.active = False
        self.intervals = self.substep = 0

    def _copy(self, value, axis=0):
        return _copy(value, axis=axis, index=self.focal_index)

    @contextmanager
    def record(self):
        """Install transparent wrappers and restore every attribute on exit.

        Args:
            None.
        Returns:
            Context retaining partial first-interval evidence on failure.
        Raises:
            Exception: Original native failure or evidence retention failure.
        """
        try:
            self._install()
            yield
        finally:
            for target, name, previous, owned in reversed(self.patches):
                if owned:
                    setattr(target, name, previous)
                else:
                    delattr(target, name)

    def _patch(self, target, name, factory):
        original = getattr(target, name)
        owned = name in vars(target)
        previous = vars(target).get(name)
        self.patches.append((target, name, previous, owned))
        setattr(target, name, wraps(original)(factory(original)))

    def _install(self):
        self._patch(self.env, "step", self._interval)
        self._patch(self.action, "apply_actions", self._action)
        self._patch(self.action._low_level_obs_manager, "compute_group", self._observe)
        self._patch(self.env.scene, "update", self._physics)
        for name, actuator in self.robot.actuators.items():
            self._patch(actuator, "compute", self._actuator_factory(name, actuator))
            self._patch(actuator, "network", self._network_factory(name, actuator))

    def _interval(self, original):
        def call(*args, **kwargs):
            self.intervals += 1
            if self.intervals != 1:
                return original(*args, **kwargs)
            runtime = _runtime(self.env)
            self.active = True
            complete = False
            try:
                self._event("interval_begin", self._native_state())
                result = original(*args, **kwargs)
                complete = True
                return result
            finally:
                self.active = False
                self._save(complete, runtime)

        return call

    def _event(self, phase, values=None, **fields):
        event = {
            "index": len(self.events),
            "substep": self.substep,
            "phase": phase,
            "native_step": int(self.clock.get_num_physics_steps()),
            "native_time_s": float(self.clock.get_simulation_time()),
            **fields,
        }
        event["arrays"] = {}
        for name, value in (values or {}).items():
            key = f"{event['index']:06d}/{name}"
            self.arrays[key] = value
            event["arrays"][name] = key
        self.events.append(event)

    def _action(self, original):
        def call(*args, **kwargs):
            if self.active:
                self.substep += 1
                self._event(
                    "control_begin", low_level_counter=int(self.action._counter)
                )
            result = original(*args, **kwargs)
            if self.active:
                self._event("control_applied", self._action_values())
            return result

        return call

    def _action_values(self):
        return {
            "low_level_actions": self._copy(self.action.low_level_actions),
            "raw_actions": self._copy(self.action.raw_actions),
            "processed_joint_position": self._copy(
                self.action._low_level_action_term.processed_actions
            ),
            "joint_position_target": self._copy(self.robot.data.joint_pos_target.torch),
        }

    def _observe(self, original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.active:
                self._event("low_level_observation", {"value": self._copy(result)})
            return result

        return call

    def _actuator_factory(self, name, actuator):
        def factory(original):
            signature = inspect.signature(original)

            def call(*args, **kwargs):
                if self.active:
                    bound = signature.bind(*args, **kwargs).arguments
                    values = self._actuator_inputs(
                        bound["control_action"], bound["joint_pos"], bound["joint_vel"]
                    )
                    values.update(self._memory(actuator))
                    self._event("actuator_before", values, actuator=name)
                result = original(*args, **kwargs)
                if self.active:
                    values = self._actuator_outputs(actuator, result)
                    self._event("actuator_after", values, actuator=name)
                return result

            return call

        return factory

    def _actuator_outputs(self, actuator, result):
        values = self._memory(actuator)
        for field in ("sea_input", "computed_effort", "applied_effort"):
            value = getattr(actuator, field)
            if field == "sea_input":
                value = value.reshape(self.env.num_envs, actuator.num_joints, 1, 2)
            values[field] = self._copy(value)
        values["returned_joint_efforts"] = self._copy(result.joint_efforts)
        return values

    def _network_factory(self, name, actuator):
        def factory(original):
            def call(*args, **kwargs):
                if self.active:
                    values = self._network_values(actuator, *args)
                    self._event("network_before", values, actuator=name)
                result = original(*args, **kwargs)
                if self.active:
                    values = self._network_values(actuator, *result)
                    self._event("network_after", values, actuator=name)
                return result

            return call

        return factory

    def _network_values(self, actuator, value, memory):
        dimensions = (self.env.num_envs, actuator.num_joints, *value.shape[1:])
        values = {"value": self._copy(value.reshape(dimensions))}
        for name, state in zip(("hidden", "cell"), memory, strict=True):
            dimensions = (
                state.shape[0],
                self.env.num_envs,
                actuator.num_joints,
                state.shape[-1],
            )
            values[name] = self._copy(state.reshape(dimensions), axis=1)
        return values

    def _actuator_inputs(self, action, joint_pos, joint_vel):
        values = {
            "joint_position": self._copy(joint_pos),
            "joint_velocity": self._copy(joint_vel),
        }
        for field in ("joint_positions", "joint_velocities", "joint_efforts"):
            value = getattr(action, field)
            if value is not None:
                values[field] = self._copy(value)
        return values

    def _memory(self, actuator):
        return {
            name: self._copy(getattr(actuator, name), axis=1)
            for name in ("sea_hidden_state_per_env", "sea_cell_state_per_env")
        }

    def _physics(self, original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.active:
                self._event("post_physics", self._native_state())
            return result

        return call

    def _native_state(self):
        return {
            "root_transform_xyzw": self._copy(self.view.get_root_transforms()),
            "root_com_velocity": self._copy(self.view.get_root_velocities()),
            "joint_position": self._copy(self.view.get_dof_positions()),
            "joint_velocity": self._copy(self.view.get_dof_velocities()),
        }

    def _save(self, complete, runtime):
        path = self.output / "first-interval.npz"
        np.savez_compressed(
            path, **{key: value.cpu().numpy() for key, value in self.arrays.items()}
        )
        write_json(
            self.output / "index.json",
            {
                "schema": "npa.navigation.reset-first-interval.v1",
                "scope": "Focal robot 0, first control interval only; not full physics state or proof of deterministic reset",
                "timing_scope": "GPU clones plus native read-only getters at interval_begin and post_physics may change synchronization timing; observations are copied from original returned values and never recomputed",
                "population": self.env.num_envs,
                "focal_native_index": self.focal_index,
                "focal_native_root_path": self.view.prim_paths[self.focal_index],
                "complete": complete,
                "expected_substeps": self.env.cfg.decimation,
                "observed_substeps": self.substep,
                "runtime": runtime,
                "events": self.events,
                "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
