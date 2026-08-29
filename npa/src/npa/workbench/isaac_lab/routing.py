"""Isaac Lab GPU-routing guardrails.

Isaac Lab has two workloads with different physical requirements, and treating
them as one is how a headless training job gets refused on a GPU that would run
it perfectly well:

* :data:`RENDER` rasterizes frames - camera sensors, tiled rendering, recorded
  video, frame capture. It needs RT cores, so it is restricted to L40S and
  RTX PRO 6000 and must never be routed to H100/H200/A100 or datacenter
  Blackwell (B200/B300).
* :data:`HEADLESS_TRAIN` is state-based reinforcement learning against
  proprioceptive observations. Nothing in it rasterizes, so it runs on
  datacenter-headless parts as well as on RT-core parts.

A task can pull rendering in by itself: Isaac Lab ships camera variants such as
``Isaac-Cartpole-RGB-Camera-Direct-v0``, and NVIDIA's own reports of Isaac Lab on
B200 show the camera path deadlocking when the PhysX GPU pipeline falls back to
software while the render pipeline still waits on GPU-backed physics. So the
workload is classified from the task id as well as from the explicit flags,
rather than trusting a ``--headless`` argument alone.

GPU classification itself is tool-neutral and lives in
:mod:`npa.workbench.gpu_classes`; this module holds only Isaac Lab's policy over
those classes. The helpers are pure and dependency-free so the CLI, the SDK, and
workflow materialization can share them without drift.
"""

from __future__ import annotations

from npa.workbench.gpu_classes import (
    CPU,
    DATACENTER_HEADLESS,
    RT_CORE,
    UNKNOWN,
    GpuRoutingError,
    classify_gpu_target,
    require_rt_core_target,
)

# Workload identifiers used across the Isaac Lab surfaces.
HEADLESS_TRAIN = "headless-train"
RENDER = "render"

_WORKLOAD_ALLOWED_CLASSES: dict[str, frozenset[str]] = {
    HEADLESS_TRAIN: frozenset({DATACENTER_HEADLESS, RT_CORE}),
    RENDER: frozenset({RT_CORE}),
}

# Substrings in an Isaac Lab task id that mean the environment rasterizes.
_RENDER_TASK_TOKENS = (
    "camera",
    "rgb",
    "depth",
    "tiled",
    "visual",
)

_RENDER_HINT = (
    "Use an RT-core GPU such as --gpu-type l40s or --gpu-type gpu-rtx-pro-6000 "
    "(Blackwell sm_120). RT-core GPUs are the only ones that can rasterize "
    "Isaac Lab frames."
)


class IsaacLabRoutingError(GpuRoutingError):
    """Raised when an Isaac Lab workload is routed to an incompatible GPU class."""


def task_requires_rt_cores(task: str | None) -> bool:
    """Return True when the task id declares camera or rendered observations."""

    normalized = (task or "").strip().lower().replace("_", "-")
    return any(token in normalized for token in _RENDER_TASK_TOKENS)


def classify_workload(
    *,
    task: str | None = None,
    capture_video: bool = False,
    enable_cameras: bool = False,
) -> str:
    """Return the Isaac Lab workload class implied by a task and its flags."""

    if capture_video or enable_cameras or task_requires_rt_cores(task):
        return RENDER
    return HEADLESS_TRAIN


def validate_render_gpu_target(
    gpu_target: str | None, *, what: str = "Isaac Lab rendering"
) -> str:
    """Validate that a rendering workload targets an RT-core GPU."""

    return require_rt_core_target(
        gpu_target,
        what=what,
        hint=_RENDER_HINT,
        error_cls=IsaacLabRoutingError,
    )


def validate_train_gpu_target(gpu_target: str | None, *, task: str) -> str:
    """Validate a training submit's GPU against the workload its task implies.

    Headless state-based RL has no rasterization step, so refusing it on
    H100/H200/B200 is a routing bug rather than a physical limit. A task that
    declares camera or rendered observations still has to land on an RT-core part.
    """

    return validate_gpu_routing(
        workload=classify_workload(task=task),
        gpu_target=gpu_target,
        task=task,
    )


def validate_gpu_routing(
    *,
    workload: str,
    gpu_target: str | None,
    task: str | None = None,
) -> str:
    """Validate a workload/GPU pairing against the Isaac Lab routing rules.

    Returns the resolved GPU class on success and raises
    :class:`IsaacLabRoutingError` on a misroute. An unknown target is allowed so
    callers may rely on their own defaults; an explicitly-classified mismatch
    fails loud.
    """

    normalized_workload = (workload or "").strip().lower()
    allowed = _WORKLOAD_ALLOWED_CLASSES.get(normalized_workload)
    if allowed is None:
        choices = ", ".join(sorted(_WORKLOAD_ALLOWED_CLASSES))
        raise IsaacLabRoutingError(
            f"unknown Isaac Lab workload {workload!r}; choose one of: {choices}"
        )

    gpu_class = classify_gpu_target(gpu_target)

    if normalized_workload == RENDER:
        because = (
            f" Task {task!r} declares camera or rendered observations."
            if task and task_requires_rt_cores(task)
            else ""
        )
        try:
            validate_render_gpu_target(gpu_target)
        except IsaacLabRoutingError as exc:
            raise IsaacLabRoutingError(f"{exc}{because}") from None
        return gpu_class if gpu_class != UNKNOWN else RT_CORE

    if gpu_class == UNKNOWN:
        return UNKNOWN
    if gpu_class == CPU:
        raise IsaacLabRoutingError(
            f"Isaac Lab {normalized_workload} requires a GPU; {gpu_target!r} is "
            "CPU-only."
        )
    if gpu_class in allowed:
        return gpu_class
    raise IsaacLabRoutingError(
        f"Isaac Lab {normalized_workload} cannot run on GPU class {gpu_class!r} "
        f"({gpu_target!r}); allowed classes: {', '.join(sorted(allowed))}."
    )


__all__ = [
    "HEADLESS_TRAIN",
    "RENDER",
    "IsaacLabRoutingError",
    "classify_workload",
    "task_requires_rt_cores",
    "validate_gpu_routing",
    "validate_render_gpu_target",
    "validate_train_gpu_target",
]
