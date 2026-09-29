"""npa workbench gc-artifacts - thin S3 garbage collection for run artifacts.

Dry-run by default: prints the deletion plan without touching S3.
Pass ``--apply`` (alias ``--yes``) to execute it.

Safety rules (fail closed):
- only prefixes containing ``npa-workflow/manifest.json`` are considered;
- a run is deleted only when its manifest status is terminal
  (SUCCEEDED/FAILED/CANCELLED/FAILED_STARTUP) *and* it is older than the
  retention window;
- missing/unreadable manifests, unknown statuses, unknown ages, and pinned
  runs (``.npa-retain`` marker) are always kept;
- in apply mode each run's manifest is re-read immediately before deletion.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from npa.workbench.artifacts_gc import (
    DEFAULT_RETENTION_DAYS,
    PIN_MARKER,
    GcDecision,
    RetentionPolicy,
    apply_plan,
    describe_run,
    discover_run_prefixes,
    plan_gc,
    plan_summary,
    run_age_days,
)

app = typer.Typer(
    name="gc-artifacts",
    help="Garbage-collect expired workbench run artifacts from S3 (dry-run by default).",
    no_args_is_help=False,
)
console = Console()
err_console = Console(stderr=True)


def _resolve_bucket(bucket: str) -> str:
    resolved = bucket.strip() or os.environ.get("NPA_S3_BUCKET", "").strip()
    if not resolved:
        raise typer.BadParameter(
            "S3 bucket is required: pass --bucket or set NPA_S3_BUCKET."
        )
    return resolved


def _format_age(days: float | None) -> str:
    return f"{days:.1f}d" if days is not None else "?"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"  # pragma: no cover


def _decision_payload(decision: GcDecision, now: datetime) -> dict[str, Any]:
    run = decision.run
    return {
        "prefix": run.prefix,
        "status": run.manifest_status,
        "age_days": run_age_days(run, now),
        "size_bytes": run.size_bytes,
        "object_count": run.object_count,
        "pinned": run.pinned,
        "delete": decision.delete,
        "reason": decision.reason,
    }


def _print_plan_text(
    decisions: list[GcDecision], summary: dict[str, Any], now: datetime
) -> None:
    table = Table(title="Artifact GC plan (dry-run: nothing deleted)")
    table.add_column("Run prefix")
    table.add_column("Status")
    table.add_column("Age")
    table.add_column("Size")
    table.add_column("Objects", justify="right")
    table.add_column("Decision")
    table.add_column("Reason")
    for decision in decisions:
        run = decision.run
        table.add_row(
            run.prefix,
            run.manifest_status or "UNKNOWN",
            _format_age(run_age_days(run, now)),
            _format_bytes(run.size_bytes),
            str(run.object_count),
            "DELETE" if decision.delete else "keep",
            decision.reason,
        )
    console.print(table)
    console.print(
        f"Scanned {summary['runs_scanned']} run(s): "
        f"{summary['runs_to_delete']} to delete "
        f"({_format_bytes(summary['bytes_to_delete'])}, "
        f"{summary['objects_to_delete']} objects); "
        f"kept: {summary['kept_by_reason'] or 'none'}"
    )


def build_gc_plan(
    client: Any,
    bucket: str,
    root_prefix: str,
    policy: RetentionPolicy,
    max_depth: int,
) -> tuple[list[GcDecision], datetime]:
    """Discover runs under the scan root and classify each one (no writes)."""
    now = datetime.now(timezone.utc)
    runs = [
        describe_run(client, bucket, prefix, policy.pin_marker)
        for prefix in discover_run_prefixes(client, bucket, root_prefix, max_depth)
    ]
    return plan_gc(runs, policy, now), now


@app.callback(invoke_without_command=True)
def gc_artifacts_cmd(
    bucket: str = typer.Option(
        "",
        "--bucket",
        help="S3 bucket holding run artifacts. Defaults to NPA_S3_BUCKET.",
    ),
    prefix: str = typer.Option(
        "",
        "--prefix",
        help="Scan only under this key prefix. Defaults to NPA_S3_PREFIX.",
    ),
    retention_days: int = typer.Option(
        DEFAULT_RETENTION_DAYS,
        "--retention-days",
        help="Delete terminal runs older than this many days.",
    ),
    pin_marker: str = typer.Option(
        PIN_MARKER,
        "--pin-marker",
        help="Marker file under a run prefix that exempts it from deletion.",
    ),
    max_depth: int = typer.Option(
        3,
        "--max-depth",
        help="How deep to search for run prefixes under --prefix.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        "--yes",
        help="Execute the plan. Without this flag nothing is deleted.",
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit the plan as JSON instead of a table."
    ),
    journal: Path | None = typer.Option(
        None,
        "--journal",
        help="Write the JSON plan journal to this path.",
    ),
) -> None:
    """Plan (default) or execute garbage collection of S3 run artifacts."""
    from npa.clients.storage import StorageClient

    resolved_bucket = _resolve_bucket(bucket)
    root_prefix = prefix.strip() or os.environ.get("NPA_S3_PREFIX", "").strip()
    if retention_days < 0:
        raise typer.BadParameter("--retention-days must be non-negative.")
    if max_depth < 1:
        raise typer.BadParameter("--max-depth must be at least 1.")

    policy = RetentionPolicy(
        retention_days=retention_days, pin_marker=pin_marker.strip() or PIN_MARKER
    )
    client = StorageClient.from_environment().s3

    decisions, now = build_gc_plan(
        client, resolved_bucket, root_prefix, policy, max_depth
    )
    summary = plan_summary(decisions)

    journal_payload = {
        "generated_at": now.isoformat(),
        "bucket": resolved_bucket,
        "root_prefix": root_prefix,
        "policy": {
            "retention_days": policy.retention_days,
            "pin_marker": policy.pin_marker,
        },
        "dry_run": not apply,
        "summary": summary,
        "decisions": [_decision_payload(d, now) for d in decisions],
    }
    if journal is not None:
        journal.write_text(json.dumps(journal_payload, indent=2), encoding="utf-8")

    if output_json:
        # Stdout carries pure JSON so it stays machine-readable; human
        # notices go to stderr.
        print(json.dumps(journal_payload))
    else:
        _print_plan_text(decisions, summary, now)

    if not apply:
        err_console.print(
            "Dry-run: no objects deleted. Re-run with --apply to execute.",
            style="yellow",
        )
        return

    result = apply_plan(client, resolved_bucket, decisions, policy.pin_marker)
    console.print(
        f"Deleted {len(result.deleted_runs)} run(s), "
        f"{result.deleted_objects} object(s)."
    )
    if result.skipped_runs:
        err_console.print(
            "Skipped after re-check: "
            + ", ".join(
                f"{prefix} ({reason})" for prefix, reason in result.skipped_runs
            ),
            style="yellow",
        )
