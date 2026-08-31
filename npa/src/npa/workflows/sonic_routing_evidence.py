"""Produce fail-closed SONIC accelerator-routing evidence and a Rerun recording.

The customer-facing workflow calls :func:`generate_routing_evidence` from a
CPU task.  It deliberately exercises the installed package's routing function
instead of duplicating the mapping in YAML.  Provider recognition, placement,
and workload completion are separate operator-supplied assertions so a routing
unit proof can never be mistaken for GPU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

APPLICATION_ID = "npa-sonic-b300-routing-evidence"
MANIFEST_SCHEMA = "npa.sonic.routing_evidence.v1"
REPORT_SCHEMA = "npa.sonic.routing_test_report.v1"
TIMELINE = "evidence_time"

EXPECTED_ROUTES: tuple[tuple[str, str], ...] = (
    ("l40s", "L40S:1"),
    ("h100", "H100:1"),
    ("b200", "B200:1"),
    ("gpu-b200-sxm-a", "B200:1"),
    ("b300", "B300:1"),
    ("gpu-b300-sxm", "B300:1"),
)

_STATUS_VALUES = {"passed", "failed", "unverified"}
_WORKLOAD_KINDS = {"train", "finetune", "not-recorded"}
_OUTPUT_KINDS = {"checkpoint", "not-recorded"}
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:+/()-]{0,159}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_SHA = re.compile(r"^[0-9a-f]{7,64}$")
_SAFE_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class SonicRoutingEvidenceError(RuntimeError):
    """Raised when evidence input is unsafe or the checked-in routing is wrong."""


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _status(value: str, field: str) -> str:
    normalized = str(value or "unverified").strip().lower()
    if normalized not in _STATUS_VALUES:
        raise SonicRoutingEvidenceError(
            f"{field} must be one of {sorted(_STATUS_VALUES)}, got {value!r}"
        )
    return normalized


def _safe_text(value: str, field: str, *, default: str = "not recorded") -> str:
    normalized = str(value or default).strip()
    if not _SAFE_TEXT.fullmatch(normalized):
        raise SonicRoutingEvidenceError(f"{field} contains unsafe or identifying text")
    return normalized


def _safe_sha(value: str) -> str:
    normalized = str(value or "unknown").strip().lower()
    if normalized != "unknown" and not _SAFE_SHA.fullmatch(normalized):
        raise SonicRoutingEvidenceError("tested_commit_sha must be a hexadecimal Git SHA")
    return normalized


def _safe_digest(value: str, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized and normalized != "unavailable" and not _SAFE_DIGEST.fullmatch(normalized):
        raise SonicRoutingEvidenceError(f"{field} must be an immutable sha256 digest")
    return normalized or "unavailable"


def _publish(path: Path, uri: str, *, storage_client: Any | None = None) -> str:
    if uri.startswith("s3://"):
        client = storage_client or _storage()
        return str(client.upload_file(str(path), uri))
    target = Path(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return str(target)


def _set_time(rr: Any, recording: Any, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(TIMELINE, seconds, recording=recording)
    else:  # pragma: no cover - compatibility with older supported rerun SDKs
        rr.set_time(TIMELINE, duration=seconds, recording=recording)


def _blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(origin="summary", name="Evidence and limitations"),
            rrb.Spatial2DView(
                origin="routing_map",
                contents="routing_map/**",
                name="Requested target to resolved accelerator",
            ),
            rrb.TimeSeriesView(
                origin="assertions",
                contents="assertions/**",
                name="Live acceptance assertions (-1 fail, 0 unverified, 1 pass)",
            ),
            column_shares=[1.35, 1.6, 1.6],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=TIMELINE),
        auto_layout=False,
    )


def _write_rrd(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = _blueprint(rrb)
    recording = rr.RecordingStream(APPLICATION_ID, recording_id=manifest["run_id"])
    rr.save(path, default_blueprint=blueprint, recording=recording)
    if hasattr(rr, "send_blueprint"):
        rr.send_blueprint(blueprint, recording=recording)

    route_rows = manifest["routes"]
    summary_lines = [
        "# SONIC B300 routing evidence",
        "",
        "This recording separates package routing from provider recognition, GPU "
        "placement, and workload completion. It is **not SONIC training or policy "
        "execution**; the workflow itself is a CPU routing-evidence run.",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- tested commit: `{manifest['tested_commit_sha']}`",
        f"- focused result: **{manifest['focused_verification']['status']}**",
        f"- provider accelerator: `{manifest['provider']['accelerator']}`",
        f"- allocated GPU count: `{manifest['provider']['allocated_count']}`",
        f"- terminal status: `{manifest['provider']['terminal_status']}`",
        f"- immutable job evidence: `{manifest['provider']['job_evidence_digest']}`",
        f"- image digest: `{manifest['provider']['image_digest']}`",
        f"- workload kind: `{manifest['workload']['kind']}`",
        f"- output: `{manifest['output']['kind']}` ({manifest['output']['bytes']} bytes)",
        f"- output digest: `{manifest['output']['digest']}`",
        f"- semantic verification: `{manifest['output']['semantic_verification']}`",
        f"- cleanup: `{manifest['cleanup']['status']}`",
        "",
        "| Requested target | Resolved | Expected | Result |",
        "| --- | --- | --- | --- |",
    ]
    summary_lines.extend(
        f"| `{row['target']}` | `{row['resolved']}` | `{row['expected']}` | {row['status']} |"
        for row in route_rows
    )
    rr.log(
        "summary",
        rr.TextDocument("\n".join(summary_lines), media_type=rr.MediaType.MARKDOWN),
        static=True,
        recording=recording,
    )

    accelerator_y = {
        "L40S:1": 1.0,
        "H100:1": 2.0,
        "B200:1": 3.0,
        "B300:1": 4.0,
    }
    positions = [[float(index), accelerator_y.get(row["resolved"], 0.0)] for index, row in enumerate(route_rows)]
    colors = [[30, 180, 80] if row["status"] == "passed" else [220, 45, 45] for row in route_rows]
    labels = [f"{row['target']} → {row['resolved']} ({row['status']})" for row in route_rows]
    rr.log(
        "routing_map/target_to_accelerator",
        rr.Points2D(positions, colors=colors, radii=[0.18] * len(positions), labels=labels),
        static=True,
        recording=recording,
    )

    assertions = manifest["assertions"]
    ordered = (
        "routing_resolution",
        "provider_accelerator_recognition",
        "scheduling_placement",
        "workload_completion",
        "output_verification",
        "attempt_cleanup",
    )
    values = {"failed": -1.0, "unverified": 0.0, "passed": 1.0}
    # Every series receives seven samples over six seconds.  Later assertions
    # remain at 0 until their stage is reached, so the recording has visible
    # temporal structure and fixed -1..1 state cues instead of blank one-point plots.
    for step in range(7):
        _set_time(rr, recording, float(step))
        rr.log("assertions/scale/fail", rr.Scalars(-1.0), recording=recording)
        rr.log("assertions/scale/pass", rr.Scalars(1.0), recording=recording)
        for index, name in enumerate(ordered, start=1):
            value = values[assertions[name]["status"]] if step >= index else 0.0
            rr.log(f"assertions/{name}", rr.Scalars(value), recording=recording)

    return {"timeline": TIMELINE, "duration_seconds": 6.0, "samples_per_series": 7}


def generate_routing_evidence(
    *,
    manifest_uri: str,
    report_uri: str,
    rrd_uri: str,
    run_id: str,
    tested_commit_sha: str = "unknown",
    provider_accelerator: str = "not recorded",
    allocated_count: str | int = 0,
    provider_recognition_status: str = "unverified",
    scheduling_status: str = "unverified",
    workload_status: str = "unverified",
    terminal_status: str = "not recorded",
    job_evidence_digest: str = "",
    image_digest: str = "",
    workload_kind: str = "not-recorded",
    output_kind: str = "not-recorded",
    output_bytes: str | int = 0,
    output_digest: str = "",
    semantic_verification: str = "not recorded",
    output_verification_status: str = "unverified",
    cleanup_status: str = "unverified",
    pool_type: str = "not recorded",
    storage_client: Any | None = None,
) -> dict[str, Any]:
    """Resolve checked-in routes and publish a manifest, test report, and RRD."""

    if not _SAFE_RUN_ID.fullmatch(str(run_id or "")):
        raise SonicRoutingEvidenceError("run_id must be a safe workflow run identifier")
    commit_sha = _safe_sha(tested_commit_sha)
    try:
        gpu_count = int(allocated_count)
    except (TypeError, ValueError) as exc:
        raise SonicRoutingEvidenceError("allocated_count must be an integer") from exc
    if gpu_count < 0 or gpu_count > 64:
        raise SonicRoutingEvidenceError("allocated_count must be between 0 and 64")
    try:
        verified_output_bytes = int(output_bytes)
    except (TypeError, ValueError) as exc:
        raise SonicRoutingEvidenceError("output_bytes must be an integer") from exc
    if verified_output_bytes < 0:
        raise SonicRoutingEvidenceError("output_bytes must be non-negative")

    from npa.workbench.sonic.workflow import default_accelerators

    routes = []
    for target, expected in EXPECTED_ROUTES:
        resolved = default_accelerators(target)
        routes.append(
            {
                "target": target,
                "expected": expected,
                "resolved": resolved,
                "status": "passed" if resolved == expected else "failed",
            }
        )
    routing_status = "passed" if all(row["status"] == "passed" for row in routes) else "failed"
    recognition = _status(provider_recognition_status, "provider_recognition_status")
    scheduling = _status(scheduling_status, "scheduling_status")
    workload = _status(workload_status, "workload_status")
    output_verification = _status(output_verification_status, "output_verification_status")
    cleanup = _status(cleanup_status, "cleanup_status")
    provider_label = _safe_text(provider_accelerator, "provider_accelerator")
    terminal = _safe_text(terminal_status, "terminal_status")
    workload_kind_value = _safe_text(workload_kind, "workload_kind", default="not-recorded").lower()
    output_kind_value = _safe_text(output_kind, "output_kind", default="not-recorded").lower()
    if workload_kind_value not in _WORKLOAD_KINDS:
        raise SonicRoutingEvidenceError(f"workload_kind must be one of {sorted(_WORKLOAD_KINDS)}")
    if output_kind_value not in _OUTPUT_KINDS:
        raise SonicRoutingEvidenceError(f"output_kind must be one of {sorted(_OUTPUT_KINDS)}")
    output_digest_value = _safe_digest(output_digest, "output_digest")
    job_digest_value = _safe_digest(job_evidence_digest, "job_evidence_digest")
    image_digest_value = _safe_digest(image_digest, "image_digest")
    semantic = _safe_text(semantic_verification, "semantic_verification")
    pool = _safe_text(pool_type, "pool_type")
    if scheduling == "passed" and (provider_label != "B300" or gpu_count != 1):
        raise SonicRoutingEvidenceError("passed scheduling requires exactly one B300 GPU")
    if workload == "passed" and (
        terminal != "SUCCEEDED"
        or workload_kind_value not in {"train", "finetune"}
        or job_digest_value == "unavailable"
        or image_digest_value == "unavailable"
    ):
        raise SonicRoutingEvidenceError(
            "passed workload requires train/finetune, SUCCEEDED, and immutable job/image digests"
        )
    if output_verification == "passed" and (
        output_kind_value != "checkpoint"
        or verified_output_bytes <= 0
        or output_digest_value == "unavailable"
        or semantic == "not recorded"
    ):
        raise SonicRoutingEvidenceError(
            "passed output verification requires a nonempty checkpoint digest and semantic result"
        )
    focused = next(row for row in routes if row["target"] == "gpu-b300-sxm")

    assertions = {
        "routing_resolution": {
            "status": routing_status,
            "claim": "explicit b300 and gpu-b300-sxm resolve to B300:1",
        },
        "provider_accelerator_recognition": {
            "status": recognition,
            "claim": "the selected provider target recognizes a B300-class accelerator label",
        },
        "scheduling_placement": {
            "status": scheduling,
            "claim": "exactly one B300-class GPU was allocated",
        },
        "workload_completion": {
            "status": workload,
            "claim": "the real SONIC train or finetune workload reached terminal success",
        },
        "output_verification": {
            "status": output_verification,
            "claim": "the run produced a nonempty, digest-bound, semantically parsed checkpoint",
        },
        "attempt_cleanup": {
            "status": cleanup,
            "claim": "no attempt-owned nonterminal job, pod, or secret remains",
        },
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "tested_commit_sha": commit_sha,
        "scope": "CPU execution of installed SONIC routing logic; not training or policy execution",
        "routes": routes,
        "focused_verification": dict(focused),
        "assertions": assertions,
        "provider": {
            "accelerator": provider_label,
            "requested_accelerator": "B300:1",
            "allocated_count": gpu_count,
            "terminal_status": terminal,
            "job_evidence_digest": job_digest_value,
            "image_digest": image_digest_value,
            "pool_type": pool,
        },
        "workload": {"kind": workload_kind_value},
        "output": {
            "kind": output_kind_value,
            "bytes": verified_output_bytes,
            "digest": output_digest_value,
            "semantic_verification": semantic,
        },
        "cleanup": {"status": cleanup},
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "run_id": run_id,
        "status": routing_status,
        "passed": sum(row["status"] == "passed" for row in routes),
        "failed": sum(row["status"] == "failed" for row in routes),
        "tests": routes,
        "assertions": assertions,
    }

    with tempfile.TemporaryDirectory(prefix="npa-sonic-routing-evidence-") as tmp:
        root = Path(tmp)
        manifest_path = root / "manifest.json"
        report_path = root / "test-report.json"
        rrd_path = root / "sonic-b300-routing.rrd"
        timeline = _write_rrd(rrd_path, manifest)
        manifest["visualization"] = timeline
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        published = {
            "manifest": _publish(manifest_path, manifest_uri, storage_client=storage_client),
            "report": _publish(report_path, report_uri, storage_client=storage_client),
            "rrd": _publish(rrd_path, rrd_uri, storage_client=storage_client),
            "rrd_sha256": hashlib.sha256(rrd_path.read_bytes()).hexdigest(),
            "rrd_bytes": rrd_path.stat().st_size,
        }

    result = {"status": routing_status, "run_id": run_id, "artifacts": published}
    print(json.dumps(result, sort_keys=True))
    if routing_status != "passed":
        failed = [row for row in routes if row["status"] == "failed"]
        raise SonicRoutingEvidenceError(f"SONIC routing verification failed closed: {failed}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--report-uri", required=True)
    parser.add_argument("--rrd-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tested-commit-sha", default="unknown")
    parser.add_argument("--provider-accelerator", default="not recorded")
    parser.add_argument("--allocated-count", default="0")
    parser.add_argument("--provider-recognition-status", default="unverified")
    parser.add_argument("--scheduling-status", default="unverified")
    parser.add_argument("--workload-status", default="unverified")
    parser.add_argument("--terminal-status", default="not recorded")
    parser.add_argument("--job-evidence-digest", default="")
    parser.add_argument("--image-digest", default="")
    parser.add_argument("--workload-kind", default="not-recorded")
    parser.add_argument("--output-kind", default="not-recorded")
    parser.add_argument("--output-bytes", default="0")
    parser.add_argument("--output-digest", default="")
    parser.add_argument("--semantic-verification", default="not recorded")
    parser.add_argument("--output-verification-status", default="unverified")
    parser.add_argument("--cleanup-status", default="unverified")
    parser.add_argument("--pool-type", default="not recorded")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evidence publisher through an argv-safe module entry point."""

    args = _parser().parse_args(argv)
    generate_routing_evidence(**vars(args))
    return 0


__all__ = [
    "EXPECTED_ROUTES",
    "MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "SonicRoutingEvidenceError",
    "generate_routing_evidence",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
