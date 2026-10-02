"""Extend native PPO with a frozen teacher authenticated against the original baseline."""

from copy import deepcopy
import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution

from npa.workflows.navigation.anchor_state import (
    ANCHOR_KEY,
    PPO_SOURCE_SHA256,
    implementation_digest,
    state_digest,
    verify_native_source,
)
from npa.workflows.navigation.anchor_update import update_with_anchor
from npa.workflows.navigation.contract import BaselineAnchor


class BaselineAnchoredPPO(PPO):
    """Apply baseline KL within the original native optimizer step.

    Args:
        *args: Native PPO positional arguments.
        baseline_anchor: Fully bound, immutable teacher configuration.
        **kwargs: Unchanged native PPO algorithm settings.
    Returns:
        Native learner with an initially unbound, training-only teacher.
    Raises:
        ValueError: Version, architecture, mode or anchor contract is unsupported.
    """

    def __init__(self, *args, baseline_anchor, **kwargs):
        verify_native_source()
        self.anchor_config = BaselineAnchor.model_validate(baseline_anchor)
        if any(
            kwargs.get(name) is not None
            for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg")
        ):
            raise ValueError(
                "baseline anchor supports single-device feedforward Gaussian PPO only"
            )
        super().__init__(*args, **kwargs)
        _require_supported(self)
        self.anchor_teacher = None
        self.anchor_record = None
        self.anchor_steps = 0
        self.anchor_samples = 0
        self.initial_anchor_steps = 0
        self.initial_anchor_samples = 0
        self.teacher_digest = None

    def bind_baseline(self, baseline, initial_checkpoint_sha256):
        """Freeze the authenticated original actor after saving native initialization.

        Args:
            baseline: Original checkpoint decoded from hash-verified bytes.
            initial_checkpoint_sha256: Current continuation checkpoint identity.
        Returns:
            None; teacher state is independent of optimizer/model storage.
        Raises:
            ValueError: Bootstrap, resume metadata or baseline state is inconsistent.
        """
        if self.anchor_teacher is not None:
            raise ValueError("baseline teacher can only be bound once")
        original = baseline["actor_state_dict"]
        self.teacher_digest = state_digest(original)
        _authenticate_teacher(self, original, initial_checkpoint_sha256)
        self.anchor_teacher = deepcopy(self.actor)
        self.anchor_teacher.load_state_dict(original, strict=True)
        self.anchor_teacher.requires_grad_(False)
        self.anchor_teacher.eval()
        self.initial_anchor_steps = self.anchor_steps
        self.initial_anchor_samples = self.anchor_samples

    def baseline_loss(self, observations, distribution):
        """Compare distributions on raw observations without sampling the teacher.

        Args:
            observations: Complete original PPO minibatch TensorDict.
            distribution: Current student distribution parameters with gradients.
        Returns:
            Mean differentiable KL from frozen baseline to current policy.
        Raises:
            ValueError: Teacher is absent or either distribution is invalid.
        """
        teacher = self.anchor_teacher
        if teacher is None:
            raise ValueError("baseline teacher is not bound")
        with torch.no_grad():
            teacher.distribution.update(teacher.mlp(teacher.get_latent(observations)))
            reference = tuple(
                value.detach() for value in teacher.output_distribution_params
            )
        _require_distribution(reference)
        _require_distribution(distribution)
        penalty = self.actor.get_kl_divergence(reference, distribution).mean()
        if not torch.isfinite(penalty):
            raise ValueError("nonfinite baseline KL")
        return penalty

    def update(self):
        """Run native PPO or its explicitly enabled additive baseline constraint.

        Args:
            None.
        Returns:
            Native mean losses and, when enabled, mean baseline KL.
        Raises:
            ValueError: Enabled teacher is unbound or invalid.
        """
        if self.anchor_config.coefficient == 0:
            return super().update()
        return update_with_anchor(self)

    def save(self):
        """Preserve native state and attach authenticated anchor resume metadata.

        Args:
            None.
        Returns:
            Native payload, unchanged before the first teacher binding.
        Raises:
            ValueError: A frozen teacher has changed.
        """
        payload = super().save()
        if self.anchor_teacher is not None:
            payload[ANCHOR_KEY] = _record(self)
        elif self.anchor_record is not None:
            payload[ANCHOR_KEY] = deepcopy(self.anchor_record)
        return payload

    def load(self, loaded_dict, load_cfg, strict):
        """Restore all native state and retain original teacher provenance for binding.

        Args:
            loaded_dict: Safely decoded native checkpoint.
            load_cfg: Must be None for complete continuation.
            strict: Must be True for exact native state loading.
        Returns:
            Native iteration restoration flag.
        Raises:
            ValueError: Partial loading or anchor metadata is invalid.
        """
        if (
            load_cfg is not None
            or strict is not True
            or self.anchor_teacher is not None
        ):
            raise ValueError("baseline anchor requires fresh, strict complete loading")
        record = loaded_dict.get(ANCHOR_KEY)
        if ANCHOR_KEY in loaded_dict:
            _validate_record(self, record)
        result = super().load(loaded_dict, load_cfg, strict)
        self.anchor_record = deepcopy(record)
        if record is not None:
            self.learning_rate = record["learning_rate"]
            self.anchor_steps = record["optimizer_steps"]
            self.anchor_samples = record["samples"]
        return result


def _require_supported(algorithm):
    if (
        type(algorithm.actor) is not MLPModel
        or type(algorithm.critic) is not MLPModel
        or type(algorithm.actor.distribution) is not GaussianDistribution
        or algorithm.is_multi_gpu
        or algorithm.symmetry is not None
        or algorithm.rnd is not None
    ):
        raise ValueError(
            "baseline anchor supports single-device feedforward Gaussian PPO only"
        )
    size = algorithm.storage.num_envs * algorithm.storage.num_transitions_per_env
    if size % algorithm.num_mini_batches:
        raise ValueError("baseline anchor requires complete native minibatch coverage")


def _require_distribution(parameters):
    if len(parameters) != 2 or not all(
        torch.isfinite(value).all() for value in parameters
    ):
        raise ValueError("baseline anchor requires finite Gaussian parameters")
    if not (parameters[1] > 0).all():
        raise ValueError(
            "baseline anchor requires positive Gaussian standard deviations"
        )


def _authenticate_teacher(algorithm, original, initial_sha256):
    expected = algorithm.anchor_config.checkpoint.sha256
    if algorithm.anchor_record is None:
        if initial_sha256 != expected or state_digest(
            algorithm.actor.state_dict()
        ) != state_digest(original):
            raise ValueError(
                "baseline anchor bootstrap must start from its exact original checkpoint"
            )
    elif state_digest(algorithm.anchor_record["teacher_state_dict"]) != state_digest(
        original
    ):
        raise ValueError(
            "resumed teacher differs from the authenticated original baseline"
        )


def _record(algorithm):
    teacher = algorithm.anchor_teacher.state_dict()
    if state_digest(teacher) != algorithm.teacher_digest:
        raise ValueError("frozen baseline teacher state changed")
    return {
        "schema": "npa.navigation.baseline-anchor.v1",
        "checkpoint_sha256": algorithm.anchor_config.checkpoint.sha256,
        "coefficient": algorithm.anchor_config.coefficient,
        "teacher_state_dict": teacher,
        "teacher_state_sha256": algorithm.teacher_digest,
        "implementation_sha256": implementation_digest(),
        "native_ppo_sha256": PPO_SOURCE_SHA256,
        "learning_rate": algorithm.learning_rate,
        "optimizer_steps": algorithm.anchor_steps,
        "samples": algorithm.anchor_samples,
        **_accounting_settings(algorithm),
    }


def _validate_record(algorithm, record):
    expected = {
        "schema": "npa.navigation.baseline-anchor.v1",
        "checkpoint_sha256": algorithm.anchor_config.checkpoint.sha256,
        "coefficient": algorithm.anchor_config.coefficient,
        "implementation_sha256": implementation_digest(),
        "native_ppo_sha256": PPO_SOURCE_SHA256,
        **_accounting_settings(algorithm),
    }
    fields = set(expected) | {
        "teacher_state_dict",
        "teacher_state_sha256",
        "learning_rate",
        "optimizer_steps",
        "samples",
    }
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("anchored checkpoint is missing complete resume provenance")
    if any(
        type(record[key]) is not type(value) or record[key] != value
        for key, value in expected.items()
    ):
        raise ValueError(
            "anchored checkpoint does not match the frozen anchor contract"
        )
    if state_digest(record["teacher_state_dict"]) != record["teacher_state_sha256"]:
        raise ValueError("anchored checkpoint teacher digest mismatch")
    _validate_progress(algorithm, record)


def _validate_progress(algorithm, record):
    rate = record["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("anchored checkpoint learning rate is invalid")
    if any(
        type(record[key]) is not int or record[key] < 0
        for key in ("optimizer_steps", "samples")
    ):
        raise ValueError("anchored checkpoint training counters are invalid")
    updates = algorithm.num_learning_epochs * algorithm.num_mini_batches
    size = algorithm.storage.num_envs * algorithm.storage.num_transitions_per_env
    if record["optimizer_steps"] % updates or record["samples"] != record[
        "optimizer_steps"
    ] * (size // algorithm.num_mini_batches):
        raise ValueError(
            "anchored checkpoint counters contradict native minibatch accounting"
        )


def _accounting_settings(algorithm):
    return {
        "num_envs": algorithm.storage.num_envs,
        "num_steps_per_env": algorithm.storage.num_transitions_per_env,
        "num_learning_epochs": algorithm.num_learning_epochs,
        "num_mini_batches": algorithm.num_mini_batches,
    }
