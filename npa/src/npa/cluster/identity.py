"""Identity-safe selection of an NPA-owned Kubernetes cluster.

Destructive Kubernetes operations must never inherit an ambient current context.
This module resolves one configured project and one NPA-written kubeconfig, then
cross-checks the immutable cluster/project identifiers against the Nebius API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

from npa.cluster.api import MK8sClient
from npa.cluster.exceptions import ClusterNotFoundError, ClusterStateError
from npa.cluster.state import (
    ClusterState,
    existing_kubeconfig,
    list_local_clusters,
    load_cluster_state,
)


class ClusterIdentityError(RuntimeError):
    """The requested project/context could not be proven safe to mutate."""


@dataclass(frozen=True)
class VerifiedClusterIdentity:
    project_alias: str
    project_id: str
    context: str
    cluster_id: str
    cluster_name: str
    kubeconfig: Path
    endpoint: str = ""
    cluster_absent: bool = False

    def receipt_identity(self) -> dict[str, str | bool]:
        return {
            "project_alias": self.project_alias,
            "project_id": self.project_id,
            "context": self.context,
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "cluster_absent": self.cluster_absent,
        }


def _selected_project(project: str) -> tuple[str, str]:
    from npa.clients.config import default_project_name, list_projects

    projects = list_projects()
    explicit = str(project or "").strip()
    if explicit:
        if explicit not in projects:
            available = ", ".join(sorted(projects)) or "(none)"
            raise ClusterIdentityError(
                f"NPA project {explicit!r} is not configured (available: {available}). "
                "Run `npa configure` for the intended project, then retry with "
                f"`--project {explicit}`."
            )
        alias = explicit
    else:
        default = str(default_project_name() or "").strip()
        if default in projects:
            alias = default
        elif len(projects) == 1:
            alias = next(iter(projects))
        else:
            available = ", ".join(sorted(projects)) or "(none)"
            raise ClusterIdentityError(
                "Controller/cluster teardown needs one selected NPA project; no "
                f"unambiguous default exists (available: {available}). Pass "
                "`--project <alias>`."
            )
    section = projects.get(alias)
    project_id = str((section or {}).get("project_id") or "").strip()
    if not project_id:
        raise ClusterIdentityError(
            f"NPA project {alias!r} has no stable project_id. Re-run `npa configure` "
            f"for that project before retrying `--project {alias}`."
        )
    return alias, project_id


def _selected_context(context: str, project_id: str) -> tuple[str, ClusterState]:
    explicit = str(context or "").strip()
    if explicit:
        try:
            state = load_cluster_state(explicit)
        except (OSError, ClusterStateError) as exc:
            raise ClusterIdentityError(
                f"NPA cluster state for context {explicit!r} is unreadable: {exc}. "
                "Regenerate it with `npa cluster kubeconfig --context "
                f"{explicit}` before teardown."
            ) from exc
        if state is None:
            raise ClusterIdentityError(
                f"Context {explicit!r} has no NPA cluster identity record. NPA will "
                "not use an ambient or SkyPilot profile. Adopt the intended cluster "
                f"with `npa cluster kubeconfig --context {explicit}`, then retry."
            )
        return explicit, state

    try:
        candidates = [
            state for state in list_local_clusters() if state.project_id == project_id
        ]
    except (OSError, ClusterStateError) as exc:
        raise ClusterIdentityError(
            f"NPA cluster inventory is unreadable: {exc}"
        ) from exc
    if len(candidates) != 1:
        names = ", ".join(sorted(item.name for item in candidates)) or "(none)"
        raise ClusterIdentityError(
            "The selected NPA project does not identify exactly one local cluster "
            f"context (matches: {names}). Pass `--context <exact-context>`."
        )
    return candidates[0].name, candidates[0]


def _load_exact_kubeconfig(context: str, state: ClusterState) -> tuple[Path, str]:
    try:
        path = existing_kubeconfig(context)
    except (OSError, ClusterStateError) as exc:
        raise ClusterIdentityError(
            f"The saved kubeconfig for {context!r} is unreadable: {exc}"
        ) from exc
    if path is None:
        raise ClusterIdentityError(
            f"The exact NPA kubeconfig for {context!r} is missing. Regenerate it "
            f"with `npa cluster kubeconfig --context {context}`; NPA will not fall "
            "back to the ambient current context."
        )
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ClusterIdentityError(
            f"The saved kubeconfig {expanded} is a symlink; replace it with the "
            "NPA-written regular file before destructive use."
        )
    try:
        resolved = expanded.resolve(strict=True)
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ClusterIdentityError(
            f"Could not read kubeconfig {expanded}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ClusterIdentityError(
            f"Kubeconfig {resolved} is not a Kubernetes config object"
        )
    current = str(payload.get("current-context") or "").strip()
    contexts = {
        str(item.get("name") or ""): item.get("context") or {}
        for item in payload.get("contexts") or []
        if isinstance(item, dict)
    }
    if context not in contexts or current != context:
        raise ClusterIdentityError(
            f"Kubeconfig {resolved} does not select exact context {context!r} "
            f"(current-context={current or '<unset>'!r}). Regenerate the saved "
            "NPA kubeconfig; no remote or local mutation was attempted."
        )
    cluster_ref = str((contexts[context] or {}).get("cluster") or "").strip()
    clusters = {
        str(item.get("name") or ""): item.get("cluster") or {}
        for item in payload.get("clusters") or []
        if isinstance(item, dict)
    }
    if not cluster_ref or cluster_ref not in clusters:
        raise ClusterIdentityError(
            f"Kubeconfig context {context!r} has no exact cluster entry."
        )
    server = str((clusters[cluster_ref] or {}).get("server") or "").strip()
    if not server:
        raise ClusterIdentityError(
            f"Kubeconfig context {context!r} has no Kubernetes API server."
        )
    recorded = Path(str(state.kubeconfig_path or expanded)).expanduser()
    if state.kubeconfig_path:
        try:
            if recorded.resolve(strict=True) != resolved:
                raise ClusterIdentityError(
                    "Cluster state and the selected NPA kubeconfig resolve to different "
                    f"files ({recorded} vs {resolved}). Regenerate cluster state."
                )
        except OSError as exc:
            raise ClusterIdentityError(
                f"Cluster state's kubeconfig reference {recorded} is stale: {exc}"
            ) from exc
    return resolved, server


def _endpoint_host(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return str(parsed.hostname or "").rstrip(".").lower()


def resolve_verified_cluster_identity(
    *,
    project: str = "",
    context: str = "",
    kubeconfig: Path | None = None,
    client: MK8sClient | None = None,
    receipt: str = "",
    project_id: str = "",
    cluster_id: str = "",
    cluster_name: str = "",
) -> VerifiedClusterIdentity:
    """Resolve and remotely verify one exact NPA cluster identity.

    Precedence is deliberately singular: explicit project/context, then the
    selected NPA project and its sole matching cluster record.  Ambient kube and
    SkyPilot profiles are never candidates.
    """

    if receipt or project_id or cluster_id or cluster_name:
        from npa.cleanup_identity import CleanupIdentityError, resolve_cleanup_identity
        from npa.clients.config import resolve_environment

        environment = resolve_environment(project) if project else None
        local_state = None
        requested_context = str(context or "").strip()
        if requested_context:
            try:
                local_state = load_cluster_state(requested_context)
            except (OSError, ClusterStateError) as exc:
                raise ClusterIdentityError(str(exc)) from exc
        try:
            recovery = resolve_cleanup_identity(
                explicit={
                    "project_alias": project,
                    "project_id": project_id,
                    "context": requested_context,
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "kubeconfig_path": str(kubeconfig) if kubeconfig else "",
                },
                receipt_id=receipt,
                live={
                    "project_alias": project,
                    "project_id": str(getattr(environment, "project_id", "") or ""),
                    "context": requested_context,
                    "cluster_id": str(getattr(local_state, "cluster_id", "") or ""),
                    "cluster_name": str(getattr(local_state, "name", "") or ""),
                    "kubeconfig_path": str(
                        getattr(local_state, "kubeconfig_path", "") or ""
                    ),
                },
                phase="controller",
                resource=requested_context,
            )
        except CleanupIdentityError as exc:
            raise ClusterIdentityError(str(exc)) from exc
        exact_project = str(recovery.get("project_id") or "")
        exact_cluster = str(recovery.get("cluster_id") or "")
        selected_context = str(
            recovery.get("context")
            or recovery.get("controller_context")
            or recovery.get("cluster_name")
            or ""
        )
        if not exact_project or not exact_cluster or not selected_context:
            raise ClusterIdentityError(
                "Alias-free controller cleanup requires exact project_id, cluster_id, "
                "and context from --receipt or exact flags."
            )
        provider = client or MK8sClient()
        try:
            remote = provider.get_cluster(exact_cluster, project_id=exact_project)
        except ClusterNotFoundError:
            return VerifiedClusterIdentity(
                project_alias=str(recovery.get("project_alias") or ""),
                project_id=exact_project,
                context=selected_context,
                cluster_id=exact_cluster,
                cluster_name=str(recovery.get("cluster_name") or selected_context),
                kubeconfig=Path(str(recovery.get("kubeconfig_path") or "/nonexistent")),
                cluster_absent=True,
            )
        except Exception as exc:
            raise ClusterIdentityError(
                f"Could not verify cluster {exact_cluster} in project {exact_project}: "
                f"{type(exc).__name__}: {exc}. All controller state was preserved."
            ) from exc
        if local_state is None:
            raise ClusterIdentityError(
                "The exact cluster is present, but its NPA-owned kubeconfig/state is "
                "unavailable. NPA will not use an ambient or unrelated context."
            )
        alias = str(recovery.get("project_alias") or project)
        project_id = exact_project
        state = local_state
        if remote.id != exact_cluster or (
            remote.project_id and remote.project_id != exact_project
        ):
            raise ClusterIdentityError(
                "Exact receipt/provider cluster identity mismatch; no mutation was attempted."
            )
        saved_kubeconfig, kube_server = _load_exact_kubeconfig(selected_context, state)
        if _endpoint_host(remote.endpoint or state.endpoint) not in {
            "",
            _endpoint_host(kube_server),
        }:
            raise ClusterIdentityError(
                "Exact kubeconfig/provider endpoint mismatch; no mutation was attempted."
            )
        return VerifiedClusterIdentity(
            project_alias=alias,
            project_id=project_id,
            context=selected_context,
            cluster_id=exact_cluster,
            cluster_name=remote.name or state.name,
            kubeconfig=saved_kubeconfig,
            endpoint=remote.endpoint or state.endpoint,
        )

    alias, project_id = _selected_project(project)
    selected_context, state = _selected_context(context, project_id)
    if not state.cluster_id or not state.project_id:
        raise ClusterIdentityError(
            f"Cluster record {selected_context!r} lacks immutable cluster/project "
            "identifiers. Regenerate it with `npa cluster kubeconfig`."
        )
    if state.project_id != project_id:
        raise ClusterIdentityError(
            f"Context {selected_context!r} belongs to project {state.project_id}, not "
            f"selected NPA project {project_id}. No mutation was attempted."
        )
    saved_kubeconfig, kube_server = _load_exact_kubeconfig(selected_context, state)
    if kubeconfig is not None:
        explicit_kubeconfig = kubeconfig.expanduser()
        try:
            explicit_resolved = explicit_kubeconfig.resolve(strict=True)
        except OSError as exc:
            raise ClusterIdentityError(
                f"Explicit kubeconfig {explicit_kubeconfig} is unreadable: {exc}"
            ) from exc
        if explicit_resolved != saved_kubeconfig:
            raise ClusterIdentityError(
                "Explicit --kubeconfig does not match the exact NPA-saved kubeconfig "
                f"for {selected_context!r} ({explicit_resolved} vs {saved_kubeconfig}). "
                "It may be used for read-only diagnostics, but system PDB/controller "
                "mutation is refused."
            )
    provider = client or MK8sClient()
    try:
        remote = provider.get_cluster(state.cluster_id, project_id=project_id)
    except ClusterNotFoundError:
        return VerifiedClusterIdentity(
            project_alias=alias,
            project_id=project_id,
            context=selected_context,
            cluster_id=state.cluster_id,
            cluster_name=state.name,
            kubeconfig=saved_kubeconfig,
            endpoint=state.endpoint,
            cluster_absent=True,
        )
    except Exception as exc:
        raise ClusterIdentityError(
            f"Could not verify cluster {state.cluster_id} in project {project_id}: "
            f"{type(exc).__name__}: {exc}. Authentication/connectivity uncertainty "
            "preserves all controller and local state."
        ) from exc
    mismatches: list[str] = []
    if remote.id != state.cluster_id:
        mismatches.append(f"cluster id {remote.id!r} != {state.cluster_id!r}")
    if remote.project_id and remote.project_id != project_id:
        mismatches.append(f"project id {remote.project_id!r} != {project_id!r}")
    if remote.name and remote.name not in {state.name, selected_context}:
        mismatches.append(
            f"cluster name {remote.name!r} is not {state.name!r}/{selected_context!r}"
        )
    remote_host = _endpoint_host(remote.endpoint or state.endpoint)
    kube_host = _endpoint_host(kube_server)
    if remote_host and kube_host and remote_host != kube_host:
        mismatches.append(
            f"kube API host {kube_host!r} != provider endpoint {remote_host!r}"
        )
    if mismatches:
        raise ClusterIdentityError(
            "NPA cluster identity mismatch: " + "; ".join(mismatches) + ". "
            "Regenerate the exact project-scoped kubeconfig before retrying."
        )
    return VerifiedClusterIdentity(
        project_alias=alias,
        project_id=project_id,
        context=selected_context,
        cluster_id=state.cluster_id,
        cluster_name=remote.name or state.name,
        kubeconfig=saved_kubeconfig,
        endpoint=remote.endpoint or state.endpoint,
    )
