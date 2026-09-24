"""Validate and apply the versioned TRAIN-only replanning experiment contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "npa.behavior.parent-diagnostic-conditional-gate-fit-contract.v1"
STATUS = "preregistered_before_parent_proposal_extraction"
SPLIT_DOMAIN = "task1-gate-v1:"
NOISE_DOMAIN = "task1-gate-v1-noise"


@dataclass(frozen=True)
class ReplanContract:
    """Expose the fixed dimensions and deterministic sampling operations.

    Args:
        value: Validated versioned contract mapping.
    Returns:
        None.
    Raises:
        ValueError: If accessed through ``from_mapping`` with an invalid contract.
    """

    value: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReplanContract:
        """Validate the fixed scientific contract.

        Args:
            value: Candidate contract mapping.
        Returns:
            Validated immutable contract wrapper.
        Raises:
            ValueError: If required scientific values differ.
        """
        if _contract_summary(value) != _expected_summary():
            raise ValueError("conditional replanning contract differs")
        _validate_fixed_recipe(value)
        return cls(json.loads(json.dumps(value)))

    @property
    def horizon(self) -> int:
        """Return the native action horizon.

        Args:
            None.
        Returns:
            Native horizon of 32 actions.
        Raises:
            None.
        """
        return 32

    def nested_partition(self, episodes: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Create the deterministic episode-level 160/20 split.

        Args:
            episodes: Unique TRAIN episode identifiers.
        Returns:
            Fit and validation memberships.
        Raises:
            ValueError: If membership is not exactly 180 unique episodes.
        """
        if len(episodes) != 180 or len(set(episodes)) != 180:
            raise ValueError("TRAIN membership must contain 180 unique episodes")
        ranked = sorted(episodes, key=_episode_rank)
        return {"validation": tuple(ranked[:20]), "fit": tuple(ranked[20:])}

    def selected_starts(self, episode_length: int) -> tuple[int, ...]:
        """Select first, middle, and last nonoverlapping chunk starts.

        Args:
            episode_length: Number of frames in one episode.
        Returns:
            Three distinct action-chunk start offsets.
        Raises:
            ValueError: If three complete chunks are unavailable.
        """
        if type(episode_length) is not int or episode_length < self.horizon:
            raise ValueError("episode is too short for one complete action chunk")
        starts = tuple(range(0, episode_length - self.horizon + 1, self.horizon))
        if len(starts) < 3:
            raise ValueError("episode cannot supply three distinct anchors")
        positions = (0, (len(starts) - 1) // 2, len(starts) - 1)
        return tuple(starts[index] for index in positions)

    def noise(self, episode: int, frame: int, draw: int, role: str) -> np.ndarray:
        """Generate one deterministic independent proposal-noise array.

        Args:
            episode: TRAIN episode identifier.
            frame: Selected chunk start.
            draw: Draw index from zero through three.
            role: ``chunk_start`` or ``fresh``.
        Returns:
            Float32 standard-Gaussian array with shape ``(32, 32)``.
        Raises:
            ValueError: If draw or role falls outside the contract.
        """
        if role not in {"chunk_start", "fresh"} or draw not in range(4):
            raise ValueError("noise role or draw differs")
        text = f"{NOISE_DOMAIN}:{episode}:{frame}:{draw}:{role}".encode()
        seed = int.from_bytes(hashlib.sha256(text).digest()[:16], "big")
        generator = np.random.Generator(np.random.PCG64(seed))
        return generator.standard_normal((32, 32), dtype=np.float32)


def canonical_sha256(value: Any) -> str:
    """Hash canonical compact JSON.

    Args:
        value: JSON-serializable value.
    Returns:
        Lowercase SHA-256 hexadecimal digest.
    Raises:
        TypeError: If the value is not JSON serializable.
    """
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _episode_rank(episode: int) -> tuple[str, int]:
    raw = f"{SPLIT_DOMAIN}{episode}".encode()
    return hashlib.sha256(raw).hexdigest(), episode


def _expected_summary() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "dimension": 1233,
        "check_offset": 16,
        "episodes": 180,
        "fit_episodes": 160,
        "validation_episodes": 20,
        "rows_per_episode": 12,
    }


def _contract_summary(value: dict[str, Any]) -> dict[str, Any]:
    membership = value.get("membership", {})
    return {
        "schema": value.get("schema"),
        "status": value.get("status"),
        "dimension": value.get("features", {}).get("dimension"),
        "check_offset": value.get("gate", {}).get("check_offset"),
        **{
            name: membership.get(name)
            for name in _expected_summary()
            if name in membership
        },
    }


def _validate_fixed_recipe(value: dict[str, Any]) -> None:
    features = value.get("features", {})
    models = value.get("models", {})
    optimization = value.get("optimization", {})
    gate = value.get("gate", {})
    checks = [
        features.get("dtype") == "float32",
        features.get("minimum_std") == 1e-6,
        models.get("initialization_seed") == 1033,
        models.get("parameter_dtype") == "float32",
        models.get("linear_control", {}).get("widths") == [1233, 1],
        models.get("mlp", {}).get("widths") == [1233, 256, 64, 1],
        models.get("mlp", {}).get("activation") == "GELU_exact_erf",
        models.get("mlp", {}).get("dropout") == 0.0,
        optimization.get("optimizer") == "AdamW",
        optimization.get("learning_rate") == 0.001,
        optimization.get("betas") == [0.9, 0.999],
        optimization.get("epsilon") == 1e-8,
        optimization.get("weight_decay") == 0.001,
        optimization.get("batch_size") == 64,
        optimization.get("epoch_shuffle_seed") == 1729,
        optimization.get("maximum_epochs") == 200,
        optimization.get("early_stop_patience_epochs") == 20,
        optimization.get("gradient_clip_global_l2") == 1.0,
        optimization.get("huber_delta") == 1.0,
        gate.get("threshold_decoded_delta") == 0.0,
        gate.get("numpy_gelu") == "0.5*x*(1+erf(x/sqrt(2)))",
    ]
    if not all(checks):
        raise ValueError("conditional replanning recipe differs")
