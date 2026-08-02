"""Stages for the multi-node reference spec (`resources.<profile>.num_nodes`).

SkyPilot gang-schedules ``num_nodes`` identical pods for one task and exports
``SKYPILOT_NODE_RANK`` / ``SKYPILOT_NUM_NODES`` / ``SKYPILOT_NODE_IPS`` into each. These
two stages make that observable from S3 rather than from a log line:

* :func:`report_node` runs on **every** node and writes ``nodes/rank-<rank>.json``;
* :func:`verify_nodes` runs once afterwards and fails unless it finds one report per
  expected rank, with distinct node IPs.

The point is a live proof that a spec can ask for a real multi-node block — previously
only ``npa burst submit --nodes`` could. Logic lives here (not inlined in the spec) so it
is unit testable, per the repo's "put testable logic in a real module" rule.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

SCHEMA_NODE = "npa.multi_node.node_report.v1"
SCHEMA_VERIFY = "npa.multi_node.verify_report.v1"


class MultiNodeProbeError(RuntimeError):
    """Raised when the gang did not materialize as the spec asked."""


def _storage_client(client: Any | None = None) -> Any:
    if client is not None:
        return client
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def node_report(*, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Build this node's report from the SkyPilot-provided environment."""

    source = env if env is not None else dict(os.environ)
    rank = source.get("SKYPILOT_NODE_RANK", "")
    if rank == "":
        raise MultiNodeProbeError(
            "SKYPILOT_NODE_RANK is unset; this stage must run as a SkyPilot task"
        )
    node_ips = [ip for ip in (source.get("SKYPILOT_NODE_IPS") or "").split("\n") if ip.strip()]
    return {
        "schema": SCHEMA_NODE,
        "rank": int(rank),
        "num_nodes": int(source.get("SKYPILOT_NUM_NODES") or len(node_ips) or 1),
        "gpus_per_node": int(source.get("SKYPILOT_NUM_GPUS_PER_NODE") or 0),
        "node_ip_count": len(node_ips),
        "hostname": socket.gethostname(),
    }


def report_node(output_uri: str, *, client: Any | None = None) -> str:
    """Write this node's report under ``output_uri`` and return the object URI."""

    import tempfile
    from pathlib import Path

    report = node_report()
    uri = output_uri.rstrip("/") + f"/rank-{report['rank']}.json"
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    storage = _storage_client(client)
    with tempfile.TemporaryDirectory(prefix="npa-multi-node-") as tmp:
        local = Path(tmp) / f"rank-{report['rank']}.json"
        local.write_text(body, encoding="utf-8")
        written = storage.upload_file(str(local), uri)
    print(json.dumps({**report, "written_uri": written}, sort_keys=True), flush=True)
    return written


def summarize(reports: list[dict[str, Any]], *, expected_nodes: int) -> dict[str, Any]:
    """Return a verification summary, raising when the gang is incomplete."""

    if expected_nodes < 1:
        raise MultiNodeProbeError(f"expected_nodes must be >= 1, got {expected_nodes}")
    ranks = sorted({int(report["rank"]) for report in reports})
    summary = {
        "schema": SCHEMA_VERIFY,
        "expected_nodes": expected_nodes,
        "reported_nodes": len(ranks),
        "ranks": ranks,
        "hostnames": sorted({str(report.get("hostname", "")) for report in reports}),
    }
    if ranks != list(range(expected_nodes)):
        raise MultiNodeProbeError(
            f"expected one report per rank 0..{expected_nodes - 1}, got ranks {ranks}. "
            "The stage did not run on the whole gang."
        )
    if len(summary["hostnames"]) != expected_nodes:
        raise MultiNodeProbeError(
            f"expected {expected_nodes} distinct hostnames, got {summary['hostnames']}. "
            "The ranks did not land on separate nodes."
        )
    return summary


def verify_nodes(
    input_uri: str,
    output_uri: str,
    expected_nodes: int | str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Read every ``rank-*.json`` under ``input_uri`` and verify the gang was complete."""

    import tempfile
    from pathlib import Path

    storage = _storage_client(client)
    with tempfile.TemporaryDirectory(prefix="npa-multi-node-verify-") as tmp:
        local_dir = Path(tmp) / "nodes"
        storage.download_directory(input_uri, str(local_dir))
        reports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(local_dir.rglob("rank-*.json"))
        ]
        if not reports:
            raise MultiNodeProbeError(f"no node reports found under {input_uri}")
        summary = summarize(reports, expected_nodes=int(expected_nodes))
        out = Path(tmp) / "multi_node_report.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["written_uri"] = storage.upload_file(
            str(out), output_uri.rstrip("/") + "/multi_node_report.json"
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


__all__ = [
    "SCHEMA_NODE",
    "SCHEMA_VERIFY",
    "MultiNodeProbeError",
    "node_report",
    "report_node",
    "summarize",
    "verify_nodes",
]
