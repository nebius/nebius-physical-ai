"""Fit native recurrent actors from retained demonstrations with real episode boundaries."""

from __future__ import annotations

from typing import Any


class RecurrentDemonstrationReplay:
    """Retain supervised native trajectories without exposing expert labels to the actor.

    Args:
        num_envs: Actual native vector-environment count used during collection.
    Returns:
        A run-local demonstration replay with no validation or Gold inputs.
    Raises:
        ValueError: The native environment count is not positive.
    """

    def __init__(self, num_envs: int):
        if num_envs < 1:
            raise ValueError("recurrent replay requires a positive environment count")
        self.num_envs = num_envs
        self.batches: list[tuple[Any, Any]] = []

    def fit(
        self, actor: Any, observations: Any, labels: Any, optimizer: Any, *, epochs: int
    ) -> dict[str, Any]:
        """Initialize normalization in training mode and fit reset-bound native sequences.

        Args:
            actor: Native RSL-RL GRU actor with empirical observation normalization.
            observations: Actual observed state and separate reset/phase annotations.
            labels: Training-only native action demonstrations.
            optimizer: Supervised optimizer, separate from PPO update accounting.
            epochs: Number of shuffled supervised sequence passes.
        Returns:
            Measured fitting, replay, normalization, and phase-weighting evidence.
        Raises:
            RuntimeError: Samples, model, normalization, or loss are invalid.
        """
        observations, labels = self._retain(observations, labels)
        if not actor.is_recurrent or actor.rnn.rnn.__class__.__name__ != "GRU":
            raise RuntimeError("recurrent imitation requires the native GRU actor")
        # Native empirical normalization ignores updates while the actor is in eval mode.
        actor.train()
        actor.update_normalization(observations)
        sequences = _episode_sequences(actor, observations, labels, self.num_envs)
        losses = _fit_epochs(actor, sequences, optimizer, epochs)
        actor.eval()
        actor.reset()
        return _fitting_audit(
            actor, sequences, losses, labels, len(self.batches), epochs
        )

    def _retain(self, observations, labels):
        import torch

        self.batches.append((observations, labels))
        combined = torch.cat([batch[0] for batch in self.batches], dim=0)
        targets = torch.cat([batch[1] for batch in self.batches], dim=0)
        if not torch.isfinite(targets).all() or any(
            not torch.isfinite(value).all() for value in combined.values()
        ):
            raise RuntimeError("native demonstration replay contains nonfinite samples")
        return combined, targets


def _episode_lengths(observations, num_envs):
    import torch

    done = observations["__done"].reshape(-1, num_envs).transpose(0, 1).clone()
    done[:, -1] = True
    ends = done.flatten().nonzero(as_tuple=False).flatten() + 1
    return ends - torch.cat((torch.zeros_like(ends[:1]), ends[:-1]))


def _padded_episodes(values, lengths, num_envs):
    from torch.nn.utils.rnn import pad_sequence

    ordered = values.reshape(-1, num_envs, values.shape[-1])
    ordered = ordered.transpose(0, 1).reshape(len(values), -1)
    return pad_sequence(ordered.split(lengths.tolist()), batch_first=False)


def _episode_sequences(actor, observations, labels, num_envs):
    import torch

    lengths = _episode_lengths(observations, num_envs)
    observed = torch.cat([observations[name] for name in actor.obs_groups], dim=-1)
    inputs = _padded_episodes(observed, lengths, num_envs)
    targets = _padded_episodes(labels, lengths, num_envs)
    phases = observations["__phase"]
    counts = torch.bincount(phases.flatten(), minlength=5)
    weights = len(labels) / ((counts > 0).sum() * counts.clamp_min(1))
    return {
        "inputs": inputs,
        "targets": targets,
        "phases": _padded_episodes(phases, lengths, num_envs).squeeze(-1),
        "counts": counts,
        "weights": weights,
        "mask": torch.arange(inputs.shape[0], device=inputs.device)[:, None]
        < lengths[None, :],
        "episodes": len(lengths),
    }


def _sequence_predictions(actor, inputs):
    import torch

    hidden = torch.zeros(
        actor.rnn.rnn.num_layers,
        inputs.shape[1],
        actor.rnn.rnn.hidden_size,
        device=inputs.device,
    )
    latent, _ = actor.rnn.rnn(actor.obs_normalizer(inputs), hidden)
    prediction = actor.mlp(latent)
    return actor.distribution.deterministic_output(prediction)


def _sequence_loss(actor, sequences, indices):
    prediction = _sequence_predictions(actor, sequences["inputs"][:, indices])
    mask = sequences["mask"][:, indices]
    error = (prediction - sequences["targets"][:, indices])[mask]
    phases = sequences["phases"][:, indices][mask]
    weights = sequences["weights"][phases]
    per_state_loss = error[:, :-1].square().mean(dim=1) + 2 * error[:, -1].square()
    return (per_state_loss * weights).mean()


def _fit_epochs(actor, sequences, optimizer, epochs):
    import torch

    if type(epochs) is not int or epochs < 1:
        raise RuntimeError("native recurrent fitting requires positive epochs")
    losses = []
    actor.reset()
    for _ in range(epochs):
        order = torch.randperm(sequences["episodes"], device=sequences["inputs"].device)
        for indices in order.split(64):
            loss = _sequence_loss(actor, sequences, indices)
            if not torch.isfinite(loss):
                raise RuntimeError("native recurrent imitation loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses


def _fitting_audit(actor, sequences, losses, labels, batches, epochs):
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "optimizer_updates": len(losses),
        "native_recurrent_model": "RNNModel-GRU",
        "recurrent_hidden_dim": actor.rnn.rnn.hidden_size,
        "replay_episodes": sequences["episodes"],
        "real_reset_boundaries": True,
        "training_phase_example_counts": sequences["counts"].tolist(),
        "expert_phase_is_observation": False,
        "normalization_sample_count": int(actor.obs_normalizer.count),
        "normalization_mean_nonzero": bool(actor.obs_normalizer.mean.abs().sum() > 0),
        "observation_normalization": True,
        "replay_examples": len(labels),
        "replay_batches": batches,
        "fitting_epochs": epochs,
    }
