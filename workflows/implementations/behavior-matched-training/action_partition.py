"""Define the exact frozen-stage, action-only native RLC parameter partition."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

Path = tuple[str, ...]

EXPECTED_TRAINABLE_PATHS: frozenset[Path] = frozenset(
    {
        ("PaliGemma", "llm", "final_norm_1", "Dense_0", "bias"),
        ("PaliGemma", "llm", "final_norm_1", "Dense_0", "kernel"),
        ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum_1", "w"),
        ("PaliGemma", "llm", "layers", "attn", "kv_einsum_1", "w"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
        ("PaliGemma", "llm", "layers", "mlp_1", "gating_einsum"),
        ("PaliGemma", "llm", "layers", "mlp_1", "linear"),
        ("PaliGemma", "llm", "layers", "pre_attention_norm_1", "Dense_0", "bias"),
        ("PaliGemma", "llm", "layers", "pre_attention_norm_1", "Dense_0", "kernel"),
        ("PaliGemma", "llm", "layers", "pre_ffw_norm_1", "Dense_0", "bias"),
        ("PaliGemma", "llm", "layers", "pre_ffw_norm_1", "Dense_0", "kernel"),
        ("action_in_proj", "bias"),
        ("action_in_proj", "kernel"),
        ("action_out_proj", "bias"),
        ("action_out_proj", "kernel"),
        ("kv_transform", "k_bias"),
        ("kv_transform", "k_coeffs"),
        ("kv_transform", "v_bias"),
        ("kv_transform", "v_coeffs"),
        ("time_mlp_in", "bias"),
        ("time_mlp_in", "kernel"),
        ("time_mlp_out", "bias"),
        ("time_mlp_out", "kernel"),
    }
)


@dataclasses.dataclass(frozen=True)
class ExactPathSet:
    """Match complete NNX State paths and freeze unknown future leaves by default."""

    paths: frozenset[Path]

    def __init__(self, paths: Iterable[Path]):
        object.__setattr__(self, "paths", frozenset(paths))

    def __call__(self, path: Iterable[Any], _: Any) -> bool:
        return tuple(str(part) for part in path) in self.paths


def freeze_filter():
    """Return the native NNX filter that freezes everything outside the allowlist."""

    from flax import nnx

    return nnx.Not(ExactPathSet(EXPECTED_TRAINABLE_PATHS))


def assert_partition(parameter_paths: Iterable[Path]) -> None:
    """Reject missing action leaves and any accidental stage/task selection."""

    actual = frozenset(parameter_paths)
    missing = EXPECTED_TRAINABLE_PATHS - actual
    if missing:
        raise ValueError(
            f"native parameter tree lacks allowlisted paths: {sorted(missing)}"
        )
    forbidden = {
        path
        for path in EXPECTED_TRAINABLE_PATHS
        if path[0].startswith(("task_", "stage_", "gate_", "fusion_"))
    }
    if forbidden:
        raise AssertionError(
            f"stage/task leaves entered action allowlist: {sorted(forbidden)}"
        )
