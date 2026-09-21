"""Collect complete metadata-only Kubernetes absence evidence in a scoped child."""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
from pathlib import Path
import subprocess
import sys

from kubernetes.client.exceptions import ApiException

from npa.cluster.absent_evidence import digest, require
from npa.cluster.reconcile_absent import _write_private
from npa.orchestration.skypilot.absence_target import reader_environment


def _retain(
    output: Path,
    records: list,
    kind: str,
    body: bytes,
    status: int,
    path: str,
    query=(),
) -> None:
    name = f"{kind}-{len(records):04d}.json"
    _write_private(output / name, body)
    records.append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "method": "GET",
            "path": path,
            "query": list(query),
            "status": status,
            "file": name,
            "sha256": digest(body),
        }
    )


def _provider_command(authority: dict, registration: dict) -> list[str]:
    return [
        authority["binary"]["path"],
        "--config",
        authority["config"]["path"],
        "--profile",
        authority["profile"],
        "mk8s",
        "v1",
        "cluster",
        "get",
        "--id",
        registration["cluster_id"],
        "--format",
        "json",
    ]


def _provider_identity(payload: dict, evidence: dict) -> None:
    registration = evidence["registration"]
    metadata = payload["metadata"]
    require(
        metadata["id"] == registration["cluster_id"]
        and metadata["parent_id"] == evidence["journal"]["project_id"],
        "Foreign provider cluster",
    )
    plane = payload["status"]["control_plane"]
    tls = registration["original_tls"]
    require(
        payload["status"]["state"] == "RUNNING"
        and tls["server"] == plane["endpoints"]["public_endpoint"]
        and base64.b64decode(tls["certificate-authority-data"], validate=True)
        == plane["auth"]["cluster_ca_certificate"].encode(),
        "Fresh provider endpoint or CA changed",
    )


def _provider_cluster(
    manifest: dict, evidence: dict, output: Path, records: list
) -> None:
    authority = manifest["authority"]
    result = subprocess.run(
        _provider_command(authority, evidence["registration"]),
        env=reader_environment(authority),
        capture_output=True,
        check=False,
    )
    _retain(
        output,
        records,
        "provider-cluster",
        result.stdout,
        result.returncode,
        "mk8s.cluster.get",
    )
    _write_private(output / "provider-cluster.stderr", result.stderr)
    require(result.returncode == 0, "Registered cluster read failed")
    _provider_identity(json.loads(result.stdout), evidence)


def _metadata_matches(metadata: dict, scope: dict) -> bool:
    user, name = scope["user"], scope["controller"]["name"]
    values = [
        metadata.get("name", ""),
        *metadata.get("labels", {}).values(),
        *metadata.get("annotations", {}).values(),
    ]
    return any(user in str(value) or name in str(value) for value in values)


def _get(
    client, path: str, query: list, output: Path, records: list
) -> tuple[int, bytes]:
    representation = (
        "PartialObjectMetadata"
        if "/namespaces/" in path
        else "PartialObjectMetadataList"
    )
    try:
        response = client.call_api(
            path,
            "GET",
            query_params=query,
            header_params={
                "Accept": f"application/json;as={representation};g=meta.k8s.io;v=v1"
            },
            auth_settings=["BearerToken"],
            _preload_content=False,
            _return_http_data_only=True,
        )
    except ApiException as exc:
        status = exc.status
        body = exc.body if isinstance(exc.body, bytes) else str(exc.body or "").encode()
    else:
        try:
            body, status = response.data, response.status
        finally:
            response.release_conn()
    _retain(output, records, "kubernetes", body, status, path, query)
    return status, body


def _exact_controller(client, scope: dict, output: Path, records: list) -> None:
    controller = scope["controller"]
    resources = [("pods", controller["name"] + "-head")]
    resources += [("services", name) for name in controller["services"]]
    for kind, name in resources:
        path = f"/api/v1/namespaces/{controller['namespace']}/{kind}/{name}"
        status, body = _get(client, path, [], output, records)
        payload = json.loads(body)
        require(
            status == 404
            and payload.get("kind") == "Status"
            and payload.get("reason") == "NotFound"
            and payload.get("details", {}).get("name") == name,
            "Exact native controller resource is not absent",
        )


def _inventory_page(status: int, body: bytes, revision: str | None) -> dict:
    payload = json.loads(body)
    require(
        status == 200
        and payload.get("kind") == "PartialObjectMetadataList"
        and isinstance(payload.get("items"), list),
        "Complete Kubernetes inventory unavailable",
    )
    metadata = payload.get("metadata")
    require(
        isinstance(metadata, dict) and bool(metadata.get("resourceVersion")),
        "Inventory revision missing",
    )
    require(
        revision is None or metadata["resourceVersion"] == revision,
        "Inventory generation changed between pages",
    )
    return payload


def _absent_items(items: list, scope: dict) -> None:
    for item in items:
        metadata = item["metadata"]
        require(
            all(metadata.get(key) for key in ("uid", "name", "namespace")),
            "Incomplete resource metadata",
        )
        require(
            not _metadata_matches(metadata, scope),
            "Matching or ambiguous native resource remains",
        )


def _inventory(client, kind: str, scope: dict, output: Path, records: list) -> None:
    token, seen, revision = "", set(), None
    while True:
        path = "/api/v1/" + kind
        status, body = _get(
            client, path, [("continue", token)] if token else [], output, records
        )
        payload = _inventory_page(status, body, revision)
        page_metadata = payload["metadata"]
        revision = page_metadata["resourceVersion"]
        _absent_items(payload["items"], scope)
        token = page_metadata.get("continue", "")
        require(
            isinstance(token, str) and (not token or token not in seen),
            "Invalid or repeated continuation",
        )
        if not token:
            require(
                page_metadata.get("remainingItemCount") in (None, 0),
                "Inventory reports missing remaining items",
            )
            return
        seen.add(token)


def collect(manifest: dict, evidence: dict, output: Path) -> list[dict]:
    """Read the actual target without deleting resources or querying native jobs.

    Args:
        manifest: Previously validated exact reader configuration.
        evidence: Bound original controller and stable native user scope.
        output: Private, exclusively owned evidence directory.
    Returns:
        Actual provider and Kubernetes response file/hash records.
    Raises:
        ValueError: Any target, inventory or resource remains unverified.
    """
    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

    records = []
    try:
        _provider_cluster(manifest, evidence, output, records)
        scope = evidence["scope"]
        client = KubernetesJobClient.from_environment(
            namespace=scope["controller"]["namespace"],
            kubeconfig=manifest["reader_kubeconfig"]["path"],
            context=scope["context"],
        ).core.api_client
        _exact_controller(client, scope, output, records)
        for kind in ("pods", "services"):
            _inventory(client, kind, scope, output, records)
    finally:
        _write_private(
            output / "read-records.json",
            (json.dumps(records, indent=2) + "\n").encode(),
        )
    return records


def _main() -> None:
    from npa.orchestration.skypilot.absence_evidence import load_evidence

    manifest = json.loads(Path(sys.argv[1]).read_bytes())
    evidence = load_evidence(manifest)
    collect(manifest, evidence, Path(sys.argv[2]))


if __name__ == "__main__":
    _main()
