"""``npa fleet`` -- deploy fleets of Managed Kubernetes clusters.

Deploys one *or many* ``k8s-training`` clusters across one *or many* projects in
a tenant from a compact ``npa.fleet/v0.0.1`` spec. Projects may be referenced by
id or created on demand under the tenant, and clusters may share a ``defaults``
profile (identical) or override it (custom), freely mixed.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer


class OutputFormat(str, Enum):
    """``--output`` selects a rendering, not a destination path.

    Typed as an Enum so a typo fails at parse time, and so the toolRef argv
    guardrail can tell a format word from the path/URI that ``--output`` means
    elsewhere in the CLI (it validates Enum values instead of flagging the
    literal as a misplaced path).
    """

    text = "text"
    json = "json"


app = typer.Typer(
    name="fleet",
    help="Deploy and manage fleets of Nebius Managed Kubernetes clusters across projects.",
    no_args_is_help=True,
)


_PROFILE_HELP = (
    "~/.nebius profile to authenticate every nebius CLI call as (overrides the "
    "spec's 'profile'). Use it to deploy into another tenant -- a service "
    "account is single-tenant -- without switching the machine's active profile."
)


def _load(spec_path: Path):
    from npa.fleet.spec import FleetSpecError, load_spec

    try:
        return load_spec(spec_path)
    except (FleetSpecError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(f"Invalid fleet spec: {exc}") from exc


def _csv(value: str) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] or None


def _targets(spec, *, project_prefix, only_projects, only_clusters):
    """Return the ``(project_display, project_key, cluster_name)`` targets in scope."""

    prefix = project_prefix if project_prefix else spec.project_prefix
    out = []
    for project in spec.projects:
        if only_projects and not (
            project.key() in only_projects
            or project.display_name(prefix) in only_projects
        ):
            continue
        for cluster in project.clusters:
            if only_clusters and cluster.name not in only_clusters:
                continue
            out.append(
                (
                    project.display_name(prefix) or project.key(),
                    project.key(),
                    cluster.name,
                )
            )
    return out


def _confirm(
    action: str, spec, targets, *, yes: bool, cascade: bool = False, err: bool = False
) -> bool:
    """Show what will be created/destroyed and require confirmation unless ``yes``.

    ``err`` sends the human-readable banner (and prompt) to stderr so it cannot
    corrupt a machine-readable ``--output json`` stdout stream.
    """

    projects = sorted({t[0] for t in targets})
    typer.echo(
        f"About to {action} fleet '{spec.name}': "
        f"{len(targets)} cluster(s) across {len(projects)} project(s).",
        err=err,
    )
    for display, _key, cluster_name in targets:
        typer.echo(f"  - {display} / cluster {cluster_name}", err=err)
    if cascade:
        typer.echo(
            "  (each listed cluster with local state is torn down, along with any "
            "VPC network this fleet created for it.)",
            err=err,
        )
    if not targets:
        typer.echo("  (no targets in scope) -- nothing to do.", err=err)
        return False
    if yes:
        return True
    if not typer.confirm(f"Proceed to {action}?", err=err):
        typer.echo("Aborted.", err=err)
        raise typer.Exit(1)
    return True


def plan_cmd(
    spec_path: Path = typer.Option(
        ..., "--spec", "-f", help="Path to an npa.fleet/v0.0.1 spec YAML."
    ),
    project_prefix: str = typer.Option(
        "",
        "--project-prefix",
        help="Override the spec's project_prefix for created projects.",
    ),
    profile: str = typer.Option("", "--profile", help=_PROFILE_HELP),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        # Pin a short metavar: the derived "<text|json>" widens the help's
        # metavar column enough to elide long option names on narrow terminals.
        metavar="<fmt>",
        help="Output format: text or json.",
    ),
) -> None:
    """Show the resolved deployment plan without touching infrastructure."""

    from npa.fleet.lifecycle import plan_fleet

    spec = _load(spec_path)
    plan = plan_fleet(
        spec, project_prefix=project_prefix or None, profile=profile or None
    )
    if output == OutputFormat.json:
        typer.echo(json.dumps(plan, indent=2))
        return
    typer.echo(
        f"Fleet '{plan['name']}': {plan['cluster_count']} cluster(s) across {plan['project_count']} project(s)"
    )
    typer.echo(
        f"  tenant: {plan['tenant_id']}  region: {plan['region']}  prefix: {plan['project_prefix']!r}"
    )
    typer.echo(f"  nebius profile: {plan['profile']}")
    for proj in plan["projects"]:
        tag = "create" if proj["will_create"] else "existing"
        name = proj["display_name"] or proj["project_id"]
        typer.echo(f"  - project [{tag}] {name}")
        for c in proj["clusters"]:
            typer.echo(
                f"      cluster {c['name']}: cpu={c['cpu_nodes']} ({c['cpu_preset']}) "
                f"gpu={c['gpu_nodes']} ({c['gpu_platform']} {c['gpu_preset']}) "
                f"reservation={c['gpu_reservation']} "
                f"gpu_cluster={c['enable_gpu_cluster']} "
                f"driver={c['gpu_driver_mode']}"
            )


def deploy_cmd(
    spec_path: Path = typer.Option(
        ..., "--spec", "-f", help="Path to an npa.fleet/v0.0.1 spec YAML."
    ),
    project_prefix: str = typer.Option(
        "",
        "--project-prefix",
        help="Override the spec's project_prefix for created projects.",
    ),
    k8s_training_dir: Path | None = typer.Option(
        None,
        "--k8s-training-dir",
        help="Path to a local k8s-training recipe dir. Overrides the vendored/cloned copy.",
    ),
    k8s_training_ref: str = typer.Option(
        "",
        "--k8s-training-ref",
        help="Clone nebius-solutions-library at this git ref to consume the latest "
        "k8s-training recipe (e.g. 'main'). Omit to use the repo-vendored copy.",
    ),
    create_projects: bool = typer.Option(
        True,
        "--create-projects/--no-create-projects",
        help="Create missing projects under the tenant via the nebius CLI.",
    ),
    only_projects: str = typer.Option(
        "",
        "--only-projects",
        help="Comma-separated project keys or display names to deploy (add one or many).",
    ),
    only_clusters: str = typer.Option(
        "",
        "--only-clusters",
        help="Comma-separated cluster names to deploy (add one or many; subset within scope).",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue deploying remaining clusters if one fails.",
    ),
    profile: str = typer.Option("", "--profile", help=_PROFILE_HELP),
    preflight: bool = typer.Option(
        True,
        "--preflight/--no-preflight",
        help="Validate capacity-block reservations and tenant quota allowances "
        "against the fleet's needs before applying. Without it a capacity/quota "
        "wall surfaces as terraform blocking on 'Still creating...' until the timeout.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt (non-interactive create).",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-j",
        help="Apply this many clusters in parallel (each has isolated state). "
        "Parallel runs stream per-cluster output to <install_dir>/deploy.log.",
    ),
    timeout: int = typer.Option(
        120, "--timeout", help="Per-cluster terraform apply timeout in minutes."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        # Pin a short metavar: the derived "<text|json>" widens the help's
        # metavar column enough to elide long option names on narrow terminals.
        metavar="<fmt>",
        help="Output format: text or json.",
    ),
) -> None:
    """Deploy the fleet: resolve/create projects and apply each cluster."""

    from npa.fleet.lifecycle import deploy_fleet

    spec = _load(spec_path)
    only = _csv(only_projects)
    only_c = _csv(only_clusters)
    # In json mode stdout must stay a pure JSON document, so progress goes to stderr.
    json_mode = output == OutputFormat.json
    targets = _targets(
        spec, project_prefix=project_prefix, only_projects=only, only_clusters=only_c
    )
    if not _confirm(
        "create/update",
        spec,
        targets,
        yes=yes,
        err=json_mode,
    ):
        if json_mode:
            typer.echo(
                json.dumps(
                    {
                        "name": spec.name,
                        "tenant_id": spec.tenant_id,
                        "region": spec.region,
                        "clusters": [],
                        "deployed": 0,
                        "failed": 0,
                    },
                    indent=2,
                )
            )
        return
    try:
        result = deploy_fleet(
            spec,
            k8s_training_dir=k8s_training_dir,
            k8s_training_ref=k8s_training_ref or None,
            project_prefix=project_prefix or None,
            create_projects=create_projects,
            only_projects=only,
            only_clusters=only_c,
            continue_on_error=continue_on_error,
            concurrency=max(1, concurrency),
            timeout_minutes=timeout,
            profile=profile or None,
            preflight=preflight,
            stream_terraform=not json_mode,
            on_status=lambda msg: typer.echo(f"  - {msg}", err=json_mode),
        )
    except (ValueError, RuntimeError) as exc:
        # Resolution/preflight failures are operator-actionable, not bugs: report
        # the message instead of a traceback.
        if json_mode:
            typer.echo(
                json.dumps(
                    {
                        "name": spec.name,
                        "clusters": [],
                        "deployed": 0,
                        "failed": 1,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if output == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(
            f"Fleet '{result['name']}' in {result['region']} (tenant {result['tenant_id']}): "
            f"{result['deployed']} deployed, {result['failed']} failed."
        )
        for c in result["clusters"]:
            if c.get("status") == "deployed":
                typer.echo(
                    f"  [ok]   {c['project_key']}/{c['cluster_name']} -> {c.get('cluster_id') or '(no id)'} "
                    f"(context {c.get('kube_context')})"
                )
            else:
                typer.echo(
                    f"  [FAIL] {c.get('project_key')}/{c.get('cluster_name')}: {c.get('error', c.get('status'))}"
                )
    if result.get("failed"):
        raise typer.Exit(1)


def destroy_cmd(
    spec_path: Path = typer.Option(
        ...,
        "--spec",
        "-f",
        help="Path to the npa.fleet/v0.0.1 spec YAML used to deploy.",
    ),
    only_projects: str = typer.Option(
        "",
        "--only-projects",
        help="Comma-separated project keys or display names to destroy (remove one or many).",
    ),
    only_clusters: str = typer.Option(
        "",
        "--only-clusters",
        help="Comma-separated cluster names to destroy (remove one or many).",
    ),
    timeout: int = typer.Option(
        120, "--timeout", help="Per-cluster terraform destroy timeout in minutes."
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-j",
        help="Destroy this many clusters in parallel (isolated state).",
    ),
    profile: str = typer.Option("", "--profile", help=_PROFILE_HELP),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        "--force",
        help="Skip the confirmation prompt (non-interactive removal).",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        # Pin a short metavar: the derived "<text|json>" widens the help's
        # metavar column enough to elide long option names on narrow terminals.
        metavar="<fmt>",
        help="Output format: text or json.",
    ),
) -> None:
    """Destroy the fleet's spec-declared clusters (best-effort, per-target)."""

    from npa.fleet.lifecycle import destroy_fleet

    spec = _load(spec_path)
    only = _csv(only_projects)
    only_c = _csv(only_clusters)
    json_mode = output == OutputFormat.json
    targets = _targets(
        spec, project_prefix="", only_projects=only, only_clusters=only_c
    )
    if not _confirm(
        "destroy",
        spec,
        targets,
        yes=yes,
        cascade=True,
        err=json_mode,
    ):
        if json_mode:
            typer.echo(
                json.dumps(
                    {"name": spec.name, "clusters": [], "networks": [], "failed": 0},
                    indent=2,
                )
            )
        return
    try:
        result = destroy_fleet(
            spec,
            only_projects=only,
            only_clusters=only_c,
            timeout_minutes=timeout,
            concurrency=max(1, concurrency),
            profile=profile or None,
            stream_terraform=not json_mode,
            on_status=lambda msg: typer.echo(f"  - {msg}", err=json_mode),
        )
    except (ValueError, RuntimeError) as exc:
        if json_mode:
            typer.echo(
                json.dumps(
                    {
                        "name": spec.name,
                        "clusters": [],
                        "networks": [],
                        "failed": 1,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if output == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        for c in result["clusters"]:
            typer.echo(f"  {c['project_key']}/{c['cluster_name']}: {c['status']}")
        for network in result.get("networks", []):
            typer.echo(f"  {network['project_key']}/network: {network['status']}")
        if result.get("failed"):
            typer.echo(
                f"Fleet '{result['name']}' teardown is incomplete; recovery state retained."
            )
        else:
            typer.echo(f"Destroyed fleet '{result['name']}'.")
    if result.get("failed"):
        raise typer.Exit(1)


def status_cmd(
    spec_path: Path = typer.Option(
        ..., "--spec", "-f", help="Path to the npa.fleet/v0.0.1 spec YAML."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        # Pin a short metavar: the derived "<text|json>" widens the help's
        # metavar column enough to elide long option names on narrow terminals.
        metavar="<fmt>",
        help="Output format: text or json.",
    ),
) -> None:
    """Show the last-known deployment state for the fleet."""

    from npa.fleet.lifecycle import fleet_status

    spec = _load(spec_path)
    result = fleet_status(spec)
    if output == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
        return
    typer.echo(f"Fleet '{result['name']}':")
    for c in result.get("clusters", []):
        typer.echo(
            f"  {c.get('project_key')}/{c.get('cluster_name')}: {c.get('status')} "
            f"{c.get('cluster_id', '')}"
        )


def verify_mig_cmd(
    spec_path: Path = typer.Option(
        ..., "--spec", "-f", help="Path to the npa.fleet/v0.0.1 MIG spec YAML."
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help="Kubeconfig to verify. Defaults to the selected fleet cluster state.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="Project key when the spec contains multiple MIG clusters.",
    ),
    cluster_name: str = typer.Option(
        "",
        "--cluster",
        help="Cluster name when the spec contains multiple MIG clusters.",
    ),
    wait: bool = typer.Option(
        False,
        "--wait/--no-wait",
        help="Wait for two exact consecutive snapshots instead of checking once.",
    ),
    reconcile: bool = typer.Option(
        False,
        "--reconcile/--no-reconcile",
        help=(
            "With --wait, reconcile known stale kubelet resources, replacement-node "
            "MIG taints, or an OnDelete driver rollout."
        ),
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        metavar="<fmt>",
        help="Output format: text or json.",
    ),
) -> None:
    """Verify exact RTX PRO 6000 MIG labels, operands, and kubelet resources."""

    import os
    import shutil

    from npa.fleet.mig import (
        MigVerificationError,
        verify_mig_cluster,
        wait_for_mig_ready,
    )

    spec = _load(spec_path)
    targets = [
        (candidate_project, candidate_cluster)
        for candidate_project in spec.projects
        for candidate_cluster in candidate_project.clusters
        if candidate_cluster.mig
        and candidate_cluster.mig.enabled
        and (not project or candidate_project.key() == project)
        and (not cluster_name or candidate_cluster.name == cluster_name)
    ]
    if len(targets) != 1:
        raise typer.BadParameter(
            "select exactly one MIG cluster with --project/--cluster "
            f"(matched {len(targets)})"
        )
    selected_project, selected_cluster = targets[0]
    resolved_kubeconfig = kubeconfig or (
        Path.home()
        / ".npa"
        / "fleet"
        / spec.name
        / selected_project.key()
        / selected_cluster.name
        / "kubeconfig"
    )
    kubectl_bin = os.environ.get("NPA_KUBECTL_BIN") or shutil.which("kubectl")
    if not kubectl_bin:
        raise typer.BadParameter(
            "kubectl is required; install it or set NPA_KUBECTL_BIN"
        )
    try:
        if wait:
            report = wait_for_mig_ready(
                kubectl_bin=kubectl_bin,
                kubeconfig=resolved_kubeconfig,
                expected_nodes=selected_cluster.gpu_count(),
                reconcile=reconcile,
                timeout_seconds=selected_cluster.gpu_health_timeout_minutes * 60,
                on_status=(
                    (lambda message: typer.echo(f"  - {message}", err=True))
                    if output == OutputFormat.json
                    else (lambda message: typer.echo(f"  - {message}"))
                ),
            )
        else:
            report = verify_mig_cluster(
                kubectl_bin=kubectl_bin,
                kubeconfig=resolved_kubeconfig,
                expected_nodes=selected_cluster.gpu_count(),
            )
    except MigVerificationError as exc:
        typer.echo(f"MIG verification failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    payload = report.as_dict()
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(
            f"MIG {'ready' if report.ready else 'NOT ready'}: "
            f"{len(report.nodes)} node(s), Operator {report.operator_version}"
        )
        for node in report.nodes:
            typer.echo(
                f"  {node.name}: {node.config}/{node.config_state} "
                f"capacity={node.capacity} allocatable={node.allocatable}"
            )
        for error in report.errors:
            typer.echo(f"  [FAIL] {error}")
    if not report.ready:
        raise typer.Exit(1)


app.command("plan")(plan_cmd)
app.command("deploy")(deploy_cmd)
app.command("destroy")(destroy_cmd)
app.command("status")(status_cmd)
app.command("verify-mig")(verify_mig_cmd)
