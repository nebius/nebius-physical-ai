"""Report the current actor's workspace access without disclosing other members."""

from .authorization import authorize
from .account_authentication import current_actor
from .errors import AuthorizationError


def access_profile(config, actor):
    """Describe verified identity and current personal workspace permissions.

    Args:
        config, actor: Current administrator policy and verified external actor.
    Returns:
        Identity, groups, and authorized workspaces without infrastructure secrets.
    Raises:
        AuthorizationError: The issuer differs or the person is disabled.
    """
    actor = current_actor(config, actor)
    if (
        actor.issuer != config.principal_issuer
        or actor.subject in config.disabled_subjects
    ):
        raise AuthorizationError("identity is not authorized")
    workspaces = []
    for name in sorted(config.workspaces):
        entry = _workspace(config, actor, name)
        if entry is not None:
            workspaces.append(entry)
    return {
        "subject": actor.subject,
        "display_name": actor.display_name,
        "issuer": actor.issuer,
        "groups": sorted(actor.groups),
        "workspaces": workspaces,
    }


def _workspace(config, actor, name):
    for role in ("admin", "runner", "reader"):
        try:
            workspace = authorize(config, actor, name, role)
        except AuthorizationError:
            continue
        allocation = next(
            (item for item in workspace.allocations if item.subject == actor.subject),
            None,
        )
        return {
            "name": name,
            "role": role,
            "allocated": allocation is not None,
            "gpu_limit": allocation.gpu_limit if allocation else None,
            "clusters": dict(allocation.clusters) if allocation else {},
        }
    return None
