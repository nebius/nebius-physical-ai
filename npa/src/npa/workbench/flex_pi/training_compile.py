"""Compile only the pinned deterministic query/key normalization modules."""

from contextlib import contextmanager
import math

import torch


def _normalizations(model):
    from flexpi.models.wan_video_dit import RMSNorm

    modules = [module for module in model.modules() if type(module) is RMSNorm]
    if len(modules) != 240:
        raise RuntimeError("pinned training requires exactly 240 Wan RMSNorm modules")
    for module in modules:
        if (
            module.weight.shape != (3072,)
            or module.eps != 1e-6
            or module.weight.dtype != torch.bfloat16
            or module.weight.device.type != "cuda"
        ):
            raise RuntimeError("Wan RMSNorm architecture differs from qualification")
    return modules


def enable_rmsnorm_compilation(model):
    """Enable in-place full-graph compilation without replacing parameters.

    Args:
        model: The unwrapped pinned BF16 Flex-Pi model on CUDA.
    Returns:
        Number of qualified normalization modules compiled lazily.
    Raises:
        RuntimeError: Module architecture or parameter identity changes.
    """
    modules = _normalizations(model)
    before = [(name, id(parameter)) for name, parameter in model.named_parameters()]
    keys = tuple(model.state_dict())
    for module in modules:
        module.compile(backend="inductor", mode="default", fullgraph=True)
    if before != [
        (name, id(parameter)) for name, parameter in model.named_parameters()
    ]:
        raise RuntimeError("compilation replaced model parameters")
    if keys != tuple(model.state_dict()):
        raise RuntimeError("compilation changed checkpoint state keys")
    return len(modules)


@contextmanager
def candidate_compilation(model, enabled):
    """Restore eager module calls after an untimed compiled parity candidate.

    Args:
        model: Original unwrapped Flex-Pi model.
        enabled: Whether this candidate uses the qualified RMSNorm scope.
    Returns:
        A context that leaves parameter objects and eager calls intact.
    Raises:
        RuntimeError: The model was already compiled or its architecture differs.
    """
    modules = _normalizations(model) if enabled else []
    if any(module._compiled_call_impl is not None for module in modules):
        raise RuntimeError("parity reference must begin with eager normalization")
    try:
        if enabled:
            enable_rmsnorm_compilation(model)
        yield
    finally:
        for module in modules:
            module._compiled_call_impl = None


def compiler_receipt():
    """Report compiler activity without source paths or tensor values.

    Args:
        None.
    Returns:
        Current-process graph counts and outer compilation duration.
    Raises:
        RuntimeError: Required runtime compiler metrics are unavailable.
    """
    from torch._dynamo.utils import counters, compilation_time_metrics

    duration = compilation_time_metrics.get("_compile.compile_inner", [])
    if counters["stats"]["unique_graphs"] <= 0 or not duration:
        raise RuntimeError("compiled candidate did not compile a graph")
    if not math.isfinite(sum(duration)) or sum(duration) <= 0:
        raise RuntimeError("outer compilation duration is invalid")
    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "graph_breaks": sum(counters["graph_break"].values()),
        "compile_seconds": sum(duration),
    }
