"""Expose Fleet RTX graphics qualification with publication-safe output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
import yaml

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

_SPEC = typer.Option(..., "--spec", "-f", help="Path to the owner-private Fleet spec.")
_PROJECTS = typer.Option("", "--only-projects", help="Comma-separated project keys or display names.")
_CLUSTERS = typer.Option("", "--only-clusters", help="Comma-separated cluster names within selected projects.")
_PREFIX = typer.Option("", "--project-prefix", help="Override the spec's project display-name prefix.")
_PROFILE = typer.Option("", "--profile", help="Override the spec's Nebius authentication profile.")
_EVIDENCE = typer.Option(None, "--evidence-dir", help="Owner-private directory outside the repository for exact receipts.")
_CONCURRENCY = typer.Option(1, "--concurrency", "-j", min=1, help="Clusters to qualify in parallel.")
_STABILITY = typer.Option(None, "--stabilization-seconds", min=0, help="Override the healthy stability interval.")
_TIMEOUT = typer.Option(None, "--timeout-minutes", min=1, help="Override each cluster's qualification timeout.")
_OUTPUT = typer.Option("text", "--output", "--output-format", help="Output format: text or json.")


@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def verify_graphics_cmd(
    spec_path: Path = _SPEC, only_projects: str = _PROJECTS,
    only_clusters: str = _CLUSTERS, project_prefix: str = _PREFIX,
    profile: str = _PROFILE, evidence_dir: Path | None = _EVIDENCE,
    concurrency: int = _CONCURRENCY,
    stabilization_seconds: int | None = _STABILITY,
    timeout_minutes: int | None = _TIMEOUT,
    output_format: Literal["text", "json"] = _OUTPUT,
) -> None:
    """Qualify CUDA, GLX, EGL, and Vulkan on every selected RTX worker.

    Args:
        spec_path: Existing Fleet declaration.
        only_projects: Project keys or display names to include.
        only_clusters: Cluster names to include within selected projects.
        project_prefix: Optional project display-name prefix override.
        profile: Optional Nebius authentication profile override.
        evidence_dir: Owner-private destination for exact verification receipts.
        concurrency: Maximum clusters to qualify concurrently.
        stabilization_seconds: Optional healthy-state observation override.
        timeout_minutes: Optional per-cluster qualification timeout override.
        output_format: Sanitized text or one JSON document.
    Returns:
        None.
    Raises:
        typer.Exit: Qualification, target resolution, or evidence storage failed.
    """
    options = _options(
        only_projects,
        only_clusters,
        project_prefix,
        profile,
        evidence_dir,
        concurrency,
        stabilization_seconds,
        timeout_minutes,
    )
    _verify_and_emit(spec_path, output_format, options)


def _selectors(value: str) -> list[str] | None:
    selected = [name.strip() for name in value.split(",") if name.strip()]
    return selected or None


def _options(
    projects, clusters, prefix, profile, evidence, concurrency, stability, timeout
):
    return {
        "only_projects": _selectors(projects),
        "only_clusters": _selectors(clusters),
        "project_prefix": prefix or None,
        "profile": profile or None,
        "evidence_dir": evidence,
        "concurrency": concurrency,
        "stabilization_seconds": stability,
        "timeout_minutes": timeout,
    }


def _verify_and_emit(spec_path, output_format, options) -> None:
    from npa.fleet.graphics_verification import verify_graphics
    from npa.fleet.spec import load_spec

    try:
        report = verify_graphics(load_spec(spec_path), **options)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        report = {"passed": False, "failures": ["graphics_verification_unavailable"]}
        _emit_report(report, output_format)
        raise typer.Exit(1) from exc
    _emit_report(report, output_format)
    if not report.get("passed"):
        raise typer.Exit(1)


def _emit_report(report, output_format) -> None:
    if output_format == "json":
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    outcome = "passed" if report.get("passed") else "failed"
    typer.echo(
        f"Fleet graphics verification {outcome}: "
        f"{report.get('verified_clusters', 0)}/{report.get('selected_clusters', 0)} clusters, "
        f"{report.get('gpu_workers', 0)} workers and {report.get('gpus', 0)} GPUs."
    )
    typer.echo(
        f"Worker checks: CUDA={report.get('cuda_workers', 0)}, "
        f"GLX={report.get('glx_workers', 0)}, EGL={report.get('egl_workers', 0)}, "
        f"Vulkan={report.get('vulkan_workers', 0)}."
    )
    if report.get("evidence_sha256"):
        typer.echo(f"Evidence SHA-256: {report['evidence_sha256']}")
