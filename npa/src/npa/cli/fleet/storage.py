"""Expose Fleet storage verification with sanitized text and JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
import yaml

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract


@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def verify_storage_cmd(
    spec_path: Path = typer.Option(..., "--spec", "-f", help="Path to the owner-private Fleet spec."),
    only_projects: str = typer.Option("", "--only-projects", help="Comma-separated project keys or display names."),
    only_clusters: str = typer.Option("", "--only-clusters", help="Comma-separated cluster names within selected projects."),
    project_prefix: str = typer.Option("", "--project-prefix", help="Override the spec's project display-name prefix."),
    profile: str = typer.Option("", "--profile", help="Override the spec's Nebius authentication profile."),
    evidence_dir: Path | None = typer.Option(None, "--evidence-dir", help="Owner-private directory outside the repository for exact receipts."),
    output_format: Literal["text", "json"] = typer.Option("text", "--output", "--output-format", help="Output format: text or json."),
) -> None:
    """Verify host mounts and shared PVC visibility on every selected worker.

    Args:
        spec_path: Existing Fleet declaration.
        only_projects: Project keys or display names to include.
        only_clusters: Cluster names to include within selected projects.
        project_prefix: Optional project display-name prefix override.
        profile: Optional Nebius authentication profile override.
        evidence_dir: Owner-private destination for exact verification receipts.
        output_format: Sanitized text or one JSON document.
    Raises:
        typer.Exit: Verification, target resolution, or cleanup failed.
    """
    _verify_and_emit(spec_path, output_format, {
        "only_projects": _selectors(only_projects), "only_clusters": _selectors(only_clusters),
        "project_prefix": project_prefix or None, "profile": profile or None,
        "evidence_dir": evidence_dir,
    })


def _selectors(value):
    selected = [name.strip() for name in value.split(",") if name.strip()]
    return selected or None


def _verify_and_emit(spec_path, output_format, options):
    from npa.fleet.spec import load_spec
    from npa.fleet.storage_verification import verify_storage

    try:
        spec = load_spec(spec_path)
        report = verify_storage(spec, **options)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        report = {"passed": False, "failures": ["storage_verification_unavailable"]}
        _emit_report(report, output_format)
        raise typer.Exit(1) from exc
    _emit_report(report, output_format)
    if not report.get("passed"):
        raise typer.Exit(1)


def _emit_report(report, output_format):
    if output_format == "json":
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    outcome = "passed" if report.get("passed") else "failed"
    typer.echo(f"Fleet storage verification {outcome}: "
               f"{report.get('verified_clusters', 0)}/{report.get('selected_clusters', 0)} clusters, "
               f"{report.get('cpu_workers', 0)} CPU and {report.get('gpu_workers', 0)} GPU workers, "
               f"{report.get('requested_gibibytes', 0)} GiB requested.")
    if report.get("skipped_clusters"):
        typer.echo(f"Explicitly disabled filesystem targets: {report['skipped_clusters']}.")
    if report.get("evidence_sha256"):
        typer.echo(f"Evidence SHA-256: {report['evidence_sha256']}")
