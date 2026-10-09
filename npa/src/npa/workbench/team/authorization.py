"""Resolve verified external identities to workspace grants and fixed allocations."""

from dataclasses import dataclass

from .errors import AuthorizationError
from .account_authentication import current_actor
from .models import Actor, Allocation, Cluster, TeamConfig, Workspace, execution_name

_ROLES = {"reader": 1, "runner": 2, "admin": 3}


@dataclass(frozen=True)
class ExecutionBinding:
    """Hold the administrator-selected execution boundary for an authenticated actor.

    Args:
        workspace, cluster, namespace: Authorized execution location.
        allocation, connection: Personal capacity, storage, and cluster access.
    Returns:
        A ExecutionBinding instance.
    Raises:
        None.
    """

    workspace: str
    cluster: str
    namespace: str
    allocation: Allocation
    connection: Cluster


def authorize(config: TeamConfig, actor: Actor, workspace: str, role: str) -> Workspace:
    """Require a current workspace grant for a verified identity.

    Args:
        config: Current administrator policy.
        actor: Verified issuer, subject and group claims.
        workspace, role: Requested workspace and minimum role.
    Returns:
        Authorized workspace.
    Raises:
        AuthorizationError: The identity or requested grant is not permitted.
    """
    actor = current_actor(config, actor)
    selected = config.workspaces.get(workspace)
    if (
        actor.issuer != config.principal_issuer
        or actor.subject in config.disabled_subjects
    ):
        raise AuthorizationError("identity is not authorized")
    if selected is None or selected.disabled:
        raise AuthorizationError("workspace is not authorized")
    granted = [_ROLES[g.role] for g in selected.grants if _matches(g, actor)]
    if max(granted, default=0) < _ROLES[role]:
        raise AuthorizationError("workspace operation is not authorized")
    return selected


def bind_execution(
    config: TeamConfig, actor: Actor, workspace: str, cluster: str, *, role="runner"
):
    """Bind a submitter to their preallocated namespace, identity, and storage.

    Args:
        config, actor: Trusted policy and authenticated identity.
        workspace, cluster: Explicit requested execution target.
        role: Minimum current role; reads can resolve an existing reader allocation.
    Returns:
        ExecutionBinding for this person and target.
    Raises:
        AuthorizationError: Access or an administrator allocation is missing.
    """
    selected = authorize(config, actor, workspace, role)
    allocation = next(
        (a for a in selected.allocations if a.subject == actor.subject), None
    )
    if allocation is None or cluster not in allocation.clusters:
        raise AuthorizationError(
            "an administrator allocation is required for this target"
        )
    return ExecutionBinding(
        workspace,
        cluster,
        execution_name(actor.issuer, workspace, actor.subject),
        allocation,
        config.clusters[cluster],
    )


def _matches(grant, actor):
    return (grant.kind == "subject" and grant.value == actor.subject) or (
        grant.kind == "group" and grant.value in actor.groups
    )
