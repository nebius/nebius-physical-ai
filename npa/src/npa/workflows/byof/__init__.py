"""Expose BYOF control-plane helpers without loading them in workload workers."""

from importlib import import_module

__all__ = [
    "ByofKubernetesTarget",
    "byof_validation_repo",
    "resolve_byof_project",
    "resolve_byof_kubernetes_target",
    "resolve_byof_resource_yaml",
    "resolve_skypilot_bin",
    "skypilot_config_for_project",
]


def __getattr__(name: str):
    """Resolve a public control-plane helper only when explicitly requested.

    Args:
        name: Public helper name.
    Returns:
        The helper exported by the live infrastructure module.
    Raises:
        AttributeError: The requested name is not a public helper.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("npa.workflows.byof.live"), name)
    globals()[name] = value
    return value
