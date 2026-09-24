"""Validate and apply the versioned TRAIN-only replanning experiment contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "npa.behavior.parent-diagnostic-conditional-gate-fit-contract.v1"
STATUS = "preregistered_before_parent_proposal_extraction"
SPLIT_DOMAIN = "task1-gate-v1:"
NOISE_DOMAIN = "task1-gate-v1-noise"

FIXED_SEMANTIC_CONFIG = {
    "features": {
        "constant_features": "zero_after_centering",
        "dimension": 1233,
        "dtype": "float32",
        "input_scaling": (
            "fit_only_population_mean_std_per_feature_float64_then_float32"
        ),
        "mask_source": "verified_start_valid_suffix16_and_expert_valid_first16",
        "minimum_std": 1e-6,
        "serialization_order": [
            "continue_normalized_first16x23_C_order",
            "fresh_normalized_first16x23_C_order",
            "fresh_minus_continue_first16x23_C_order",
            "continue_valid_mask16_float32",
            "fresh_valid_mask16_float32",
            "start_transformed_state32",
            "fresh_transformed_state32",
            "fresh_minus_start_transformed_state32",
            "check_offset_divided_by_native_horizon_scalar",
        ],
    },
    "target": {
        "definition": "continue_error_minus_fresh_error",
        "minimum_std": 1e-6,
        "prediction_decoding": "scaled_prediction_times_fit_std_plus_fit_mean",
        "scaling": "fit_only_population_mean_std_float64_then_float32",
    },
    "models": {
        "initialization": "torch_2.7.1_nn_Linear_defaults",
        "initialization_seed": 1033,
        "linear_control": {"widths": [1233, 1]},
        "mlp": {
            "activation": "GELU_exact_erf",
            "dropout": 0.0,
            "widths": [1233, 256, 64, 1],
        },
        "parameter_dtype": "float32",
    },
    "gate": {
        "check_offset": 16,
        "evaluation": "canonical_exported_numpy_inference",
        "export_test_max_abs_scaled_output_error": 1e-5,
        "export_test_rms_scaled_output_error": 1e-6,
        "numpy_gelu": "0.5*x*(1+erf(x/sqrt(2)))",
        "refresh_when": "finite_decoded_prediction_strictly_greater_than_zero",
        "threshold_decoded_delta": 0.0,
    },
}

FIXED_OPTIMIZATION = {
    "amsgrad": False,
    "backend": "locked217_torch2.7.1_cpu",
    "batch_size": 64,
    "betas": [0.9, 0.999],
    "cpu_threads": 1,
    "deterministic_algorithms": True,
    "drop_last": False,
    "early_stop_patience_epochs": 20,
    "epoch_shuffle_seed": 1729,
    "epsilon": 1e-8,
    "foreach": False,
    "fused": False,
    "gradient_clip_global_l2": 1.0,
    "huber_delta": 1.0,
    "learning_rate": 0.001,
    "learning_rate_schedule": "constant",
    "loss": "Huber_scaled_target",
    "maximum_epochs": 200,
    "minimum_epochs": 1,
    "optimizer": "AdamW",
    "reduction": "mean",
    "refit_after_selection": False,
    "selection": "earliest_strictly_lowest_validation_mean_huber",
    "weight_decay": 0.001,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ReplanContract:
    """Expose the fixed dimensions and deterministic sampling operations.

    Args:
        value: Validated versioned contract mapping.
        expected_split_sha256: Independently verified split-file identity.
        expected_nested_canonical_sha256: Verified nested membership identity.
    Returns:
        None.
    Raises:
        ValueError: If accessed through ``from_mapping`` with an invalid contract.
    """

    value: dict[str, Any]
    expected_split_sha256: str
    expected_nested_canonical_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        expected_split_sha256: str,
        expected_nested_canonical_sha256: str,
    ) -> ReplanContract:
        """Validate the fixed scientific contract.

        Args:
            value: Candidate contract mapping.
            expected_split_sha256: Independently verified split-file digest.
            expected_nested_canonical_sha256: Verified nested-membership digest.
        Returns:
            Validated immutable contract wrapper.
        Raises:
            ValueError: If required scientific values differ.
        """
        _validate_contract_summary(value)
        validate_semantic_config(
            value.get("features"),
            value.get("target"),
            value.get("models"),
            value.get("gate"),
        )
        _validate_optimization(value.get("optimization", {}))
        digests = _validate_membership(
            value.get("membership", {}),
            expected_split_sha256,
            expected_nested_canonical_sha256,
        )
        return cls(json.loads(json.dumps(value)), *digests)

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
        result = {"validation": tuple(ranked[:20]), "fit": tuple(ranked[20:])}
        encoded = {name: list(rows) for name, rows in result.items()}
        if canonical_sha256(encoded) != self.expected_nested_canonical_sha256:
            raise ValueError("nested TRAIN membership identity differs")
        return result

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


def fixed_semantic_config() -> dict[str, dict[str, Any]]:
    """Return an independent copy of the qualified mathematical configuration.

    Args:
        None.
    Returns:
        Exact feature, target, model, and gate configuration.
    Raises:
        None.
    """
    return json.loads(json.dumps(FIXED_SEMANTIC_CONFIG))


def validate_semantic_config(
    features: Any, target: Any, models: Any, gate: Any
) -> None:
    """Require the exact qualified mathematical configuration.

    Args:
        features: Candidate feature serialization and scaling configuration.
        target: Candidate prediction target and decoding configuration.
        models: Candidate linear and MLP architecture configuration.
        gate: Candidate refresh rule and numerical parity configuration.
    Returns:
        None.
    Raises:
        ValueError: If any mathematical configuration differs.
    """
    observed = {
        "features": features,
        "target": target,
        "models": models,
        "gate": gate,
    }
    if not _same_json(observed, FIXED_SEMANTIC_CONFIG):
        raise ValueError("conditional replanning semantic configuration differs")


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


def _validate_contract_summary(value: dict[str, Any]) -> None:
    if _contract_summary(value) != _expected_summary():
        raise ValueError("conditional replanning contract differs")


def _validate_optimization(optimization: dict[str, Any]) -> None:
    if not _same_json(optimization, FIXED_OPTIMIZATION):
        raise ValueError("conditional replanning optimization differs")


def _validate_membership(
    membership: Any, split_sha256: str, nested_sha256: str
) -> tuple[str, str]:
    split = _validated_sha256(split_sha256, "split")
    nested = _validated_sha256(nested_sha256, "nested membership")
    expected = {
        "development_report_or_campaign_holdout_fit": False,
        "episodes": 180,
        "fit_episodes": 160,
        "nested_canonical_sha256": nested,
        "row_weighting": "uniform_equal_to_episode_weighting",
        "rows_per_episode": 12,
        "split_sha256": split,
        "validation_episodes": 20,
    }
    if not _same_json(membership, expected):
        raise ValueError("conditional replanning membership differs")
    return split, nested


def _validated_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} SHA-256 identity differs")
    return value


def _same_json(observed: Any, expected: Any) -> bool:
    encoding = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    try:
        return json.dumps(observed, **encoding) == json.dumps(expected, **encoding)
    except (TypeError, ValueError):
        return False
