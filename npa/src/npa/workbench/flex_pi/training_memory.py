"""Scope an allocation-only candidate without relaxing deterministic algorithms."""

from contextlib import contextmanager
import os

import torch
import torch.utils.deterministic

_disabled_scopes = 0


def _assert_determinism():
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.benchmark
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise RuntimeError("memory-fill qualification requires strict determinism")


@contextmanager
def execution_without_memory_fill():
    """Disable sentinel fills only while executing a qualified model operation.

    Args:
        None; callers construct loaders and initialize models outside this scope.
    Returns:
        An exception-safe context retaining strict deterministic algorithms.
    Raises:
        RuntimeError: Determinism or the restored allocation policy differs.
    """
    global _disabled_scopes

    _assert_determinism()
    if not torch.utils.deterministic.fill_uninitialized_memory:
        raise RuntimeError("model execution must start with memory fills enabled")
    torch.utils.deterministic.fill_uninitialized_memory = False
    _disabled_scopes += 1
    try:
        yield
        _assert_determinism()
        if torch.utils.deterministic.fill_uninitialized_memory:
            raise RuntimeError("the model changed its allocation policy")
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = True


@contextmanager
def observation_with_memory_fill():
    """Keep diagnostic hashing outside the candidate allocation policy.

    Args:
        None; the surrounding model operation may have disabled sentinel fills.
    Returns:
        A context that restores the exact surrounding allocation setting.
    Raises:
        RuntimeError: Strict algorithm determinism has been relaxed.
    """
    _assert_determinism()
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
        _assert_determinism()
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def memory_fill_receipt(configuration):
    """Report the active execution policy separately from the restored default.

    Args:
        configuration: The exact resolved configuration used by the worker.
    Returns:
        Allocation policy and unchanged deterministic algorithm settings.
    Raises:
        RuntimeError: A scope leaked its allocation policy into preparation.
    """
    _assert_determinism()
    restored = torch.utils.deterministic.fill_uninitialized_memory
    if not restored:
        raise RuntimeError("memory fills were not restored after model execution")
    expected_fill = configuration.get("npa_memory_fill", "on") == "on"
    if expected_fill != (_disabled_scopes == 0):
        raise RuntimeError(
            "observed allocation scopes differ from the requested policy"
        )
    return {
        "model_execution_fill": expected_fill,
        "observed_disabled_scopes": _disabled_scopes,
        "initialization_and_loader_creation_fill": True,
        "loader_worker_fill": True,
        "allocation_flag_scope": "process_global_including_pin_thread",
        "restored_fill": restored,
        "strict_deterministic_algorithms": True,
    }
