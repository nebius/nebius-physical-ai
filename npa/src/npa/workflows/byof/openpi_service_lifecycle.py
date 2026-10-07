"""Declarative start/cleanup lifecycle for a private OpenPI policy service.

Unlike the four-mode smoke controller, this adapter creates no client Job.  The
workflow graph owns the Isaac client as its next state and owns cleanup as a
later state.  Exact deterministic names and ownership labels bind both calls.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import urllib.request
from collections.abc import Sequence
from typing import Any

from npa.workflows.byof.openpi import OPENPI_TERMS_ACCEPTED_VALUE, OPENPI_TERMS_ENV
from npa.workflows.byof.openpi_pipeline import (
    DEFAULT_CHECKPOINT_URI,
    DEFAULT_CONFIG_NAME,
    RUNTIME_IMAGE_RE,
    SOURCE_REF,
    _write_json_uri,
    _write_terms_refusal_diagnostic,
)
from npa.workflows.byof.openpi_service import (
    SERVER_DIAGNOSTICS_PORT,
    SERVER_PORT,
    OpenPIServiceError,
    _assert_targets_absent,
    _create_server_objects,
    _delete_and_verify,
    _validated_positive_seconds,
    _wait_server_ready,
    build_manifests,
)


def _manifests(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    manifests = build_manifests(
        run_id=args.run_id,
        namespace=args.namespace,
        runtime_image=args.runtime_image,
        checkpoint_uri=args.checkpoint_uri,
        config_name=args.config_name,
        gpu_count=args.gpu_count,
        expected_gpu_type=args.expected_gpu_type,
        expected_compute_capability=args.expected_compute_capability,
        server_cpu=args.server_cpu,
        server_memory=args.server_memory,
        client_cpu="1",
        client_memory="1Gi",
        pull_secret=args.pull_secret,
        liveness_initial_delay_seconds=args.liveness_initial_delay_seconds,
        gpu_node_selector_key=args.gpu_node_selector_key,
        gpu_node_selector_value=args.gpu_node_selector_value,
        cache_size=args.service_cache_size,
        server_ready_timeout_seconds=args.server_ready_timeout_seconds,
        client_timeout_seconds=60,
    )
    if args.checkpoint_uri.startswith("s3://"):
        _patch_private_s3_checkpoint(manifests, args.checkpoint_uri)
    _inject_checkpoint_probe(manifests)
    return manifests


def _s3_download_program(checkpoint_uri: str, local: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(checkpoint_uri)
    return f"""import boto3,os,pathlib
bucket={parsed.netloc!r}
prefix={parsed.path.lstrip("/")!r}.rstrip('/')+'/'
root=pathlib.Path({local!r})
root.mkdir(parents=True,exist_ok=True)
client=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'])
pages=client.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=prefix)
items=[item for page in pages for item in page.get('Contents',[])]
assert items,'adapted checkpoint prefix is empty'
for item in items:
    relative=pathlib.PurePosixPath(item['Key'][len(prefix):])
    assert relative.parts and '..' not in relative.parts and not relative.is_absolute(),'unsafe checkpoint key'
    target=root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True,exist_ok=True)
    client.download_file(bucket,item['Key'],str(target))
"""


def _patch_private_s3_checkpoint(
    manifests: dict[str, dict[str, Any]], checkpoint_uri: str
) -> None:
    required = {
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", ""),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    }
    if not all(required.values()):
        raise OpenPIServiceError(
            "private adapted checkpoint requires exact S3 credentials"
        )
    secret = manifests["secret"]
    secret["data"].update(
        {
            key: base64.b64encode(value.encode()).decode()
            for key, value in required.items()
        }
    )
    container = manifests["deployment"]["spec"]["template"]["spec"]["containers"][0]
    secret_name = secret["metadata"]["name"]
    container["env"].extend(
        {
            "name": key,
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": key}},
        }
        for key in required
    )
    local = "/workspace/openpi-server-cache/adapted-checkpoint"
    program = _s3_download_program(checkpoint_uri, local)
    shell = container["command"][2]
    marker = "checkpoint_dir=$("
    if marker not in shell:
        raise OpenPIServiceError("unexpected OpenPI server command contract")
    download = "/opt/venv/bin/python -c " + shlex.quote(program) + "; "
    container["command"][2] = download + shell.replace(
        'os.environ["OPENPI_CHECKPOINT_URI"]', repr(local), 1
    )


def _checkpoint_probe_program() -> str:
    return """import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['NPA_RESOLVED_CHECKPOINT_DIR'])
manifest_path=root/'manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
records=[]
for path in sorted(item for item in root.rglob('*') if item.is_file() and item != manifest_path):
    records.append({'path':path.relative_to(root).as_posix(),'size':path.stat().st_size,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
payload=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
digest=hashlib.sha256(payload).hexdigest()
if manifest is not None:
    assert manifest.get('files') == records, 'adapted checkpoint files differ from manifest'
    assert manifest.get('content_manifest_sha256') == digest, 'adapted checkpoint digest mismatch'
result={'schema':'npa.workbench.openpi.loaded-checkpoint.v1','checkpoint_sha256':digest,
        'file_count':len(records),'total_size_bytes':sum(row['size'] for row in records),
        'checkpoint_manifest_sha256':manifest.get('content_manifest_sha256') if manifest else None}
pathlib.Path('/tmp/npa-openpi-loaded-checkpoint.json').write_text(json.dumps(result,sort_keys=True))
"""


def _inject_checkpoint_probe(manifests: dict[str, dict[str, Any]]) -> None:
    container = manifests["deployment"]["spec"]["template"]["spec"]["containers"][0]
    shell = container["command"][2]
    marker = "exec /opt/venv/bin/python /opt/byof/scripts/serve_policy.py"
    if marker not in shell:
        raise OpenPIServiceError("unexpected OpenPI server exec contract")
    probe = "/opt/venv/bin/python -c " + shlex.quote(_checkpoint_probe_program())
    prefix = 'export NPA_RESOLVED_CHECKPOINT_DIR="$checkpoint_dir"; '
    container["command"][2] = shell.replace(marker, f"{prefix}{probe}; {marker}", 1)


def _clients() -> tuple[Any, Any, Any]:
    from kubernetes import client, config

    config.load_incluster_config()
    return client.CoreV1Api(), client.AppsV1Api(), client.BatchV1Api()


def _check_common(args: argparse.Namespace, stage: str) -> int | None:
    if os.environ.get(OPENPI_TERMS_ENV) != OPENPI_TERMS_ACCEPTED_VALUE:
        _, refusal = _write_terms_refusal_diagnostic(
            args.output_uri,
            diagnostic_root_uri=args.terms_diagnostic_root_uri,
            stage=stage,
        )
        print(json.dumps(refusal, sort_keys=True), flush=True)
        return 64
    if not RUNTIME_IMAGE_RE.fullmatch(args.runtime_image):
        raise OpenPIServiceError("service runtime image must be digest-pinned")
    return None


def _timeouts(args: argparse.Namespace) -> tuple[float, float, float, float]:
    return (
        _validated_positive_seconds(
            args.server_ready_timeout_seconds, "server ready timeout"
        ),
        _validated_positive_seconds(args.poll_interval_seconds, "poll interval"),
        _validated_positive_seconds(args.api_timeout_seconds, "Kubernetes API timeout"),
        _validated_positive_seconds(args.http_timeout_seconds, "HTTP timeout"),
    )


def _read_service_json(host: str, filename: str, timeout: float) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://{host}:{SERVER_DIAGNOSTICS_PORT}/{filename}"
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read())


def _start_result(
    args: argparse.Namespace,
    names: dict[str, str],
    pod: Any,
    host: str,
    hardware: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if args.checkpoint_uri.startswith("s3://") and not checkpoint.get(
        "checkpoint_manifest_sha256"
    ):
        raise OpenPIServiceError(
            "adapted checkpoint lacks a validated content manifest"
        )
    return {
        "schema": "npa.workbench.openpi.pi05-service-start.v1",
        "status": "ready",
        "checkpoint_uri": args.checkpoint_uri,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_manifest_sha256": checkpoint.get("checkpoint_manifest_sha256"),
        "source_commit": SOURCE_REF,
        "config_name": args.config_name,
        "image_digest": args.runtime_image.rsplit("@sha256:", 1)[-1],
        "service_host": host,
        "service_port": SERVER_PORT,
        "server_pod_uid": str(pod.metadata.uid),
        "hardware": hardware,
        "public_ingress": False,
        "client_job_created": False,
        "cleanup_required": True,
        "exact_names": names,
    }


def _cleanup_failed_start(
    args: argparse.Namespace,
    clients: tuple[Any, Any, Any],
    names: dict[str, str],
    created: set[str],
    manifests: dict[str, Any],
    poll: float,
    api_timeout: float,
) -> None:
    _delete_and_verify(
        *clients,
        namespace=args.namespace,
        names=names,
        created=created,
        manifests=manifests,
        timeout=args.cleanup_timeout_seconds,
        poll_interval=poll,
        request_timeout=api_timeout,
    )


def _launch_server(
    args: argparse.Namespace,
    clients: tuple[Any, Any, Any],
    manifests: dict[str, Any],
    names: dict[str, str],
    created: set[str],
    timeouts: tuple[float, ...],
) -> tuple[Any, str, dict, dict]:
    ready, poll, api_timeout, http_timeout = timeouts
    _create_server_objects(
        clients[0], clients[1], manifests, created=created, request_timeout=api_timeout
    )
    pod = _wait_server_ready(
        clients[0],
        clients[1],
        args.namespace,
        names["deployment"],
        timeout=ready,
        poll_interval=poll,
        request_timeout=api_timeout,
    )
    host = f"{names['service']}.{args.namespace}.svc.cluster.local"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(
        f"http://{host}:{SERVER_PORT}/healthz", timeout=http_timeout
    ) as response:
        if response.read().decode().strip() != "OK":
            raise OpenPIServiceError("OpenPI ClusterIP health check did not return OK")
    hardware = _read_service_json(host, "npa-openpi-server-hardware.json", http_timeout)
    checkpoint = _read_service_json(
        host, "npa-openpi-loaded-checkpoint.json", http_timeout
    )
    return pod, host, hardware, checkpoint


def _publish_service_component(
    args: argparse.Namespace,
    stage: int,
    suffix: str,
    evidence: str,
    artifacts: dict[str, Any],
) -> None:
    from npa.workflows.sim2real.workflow_io import publish_component_record

    publish_component_record(
        root_uri=args.component_root_uri,
        stage=stage,
        name=f"pi05_{args.checkpoint_role}_{suffix}",
        tier="WORKS",
        evidence=evidence,
        artifacts=artifacts,
    )


def _start(args: argparse.Namespace) -> int:
    refused = _check_common(args, "service-start")
    if refused is not None:
        return refused
    timeouts = _timeouts(args)
    _, poll, api_timeout, _ = timeouts
    manifests = _manifests(args)
    names = {key: str(value["metadata"]["name"]) for key, value in manifests.items()}
    clients = _clients()
    _assert_targets_absent(
        *clients, namespace=args.namespace, names=names, request_timeout=api_timeout
    )
    created: set[str] = set()
    try:
        pod, host, hardware, checkpoint = _launch_server(
            args, clients, manifests, names, created, timeouts
        )
        result = _start_result(args, names, pod, host, hardware, checkpoint)
        _write_json_uri(args.output_uri, result)
        stage = 5 if args.checkpoint_role == "base" else 10
        _publish_service_component(
            args,
            stage,
            "service_start",
            evidence="Started a private digest-pinned OpenPI service with loaded checkpoint hash proof.",
            artifacts={
                "receipt": args.output_uri,
                "runtime_image": args.runtime_image,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            },
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception:
        _cleanup_failed_start(
            args, clients, names, created, manifests, poll, api_timeout
        )
        raise


def _cleanup(args: argparse.Namespace) -> int:
    # Cleanup is always permitted: removing exact workflow-owned resources
    # must not depend on continued model-terms acceptance.
    if not RUNTIME_IMAGE_RE.fullmatch(args.runtime_image):
        raise OpenPIServiceError("service runtime image must be digest-pinned")
    manifests = _manifests(args)
    names = {key: str(value["metadata"]["name"]) for key, value in manifests.items()}
    api, apps, batch = _clients()
    verified = _delete_and_verify(
        api,
        apps,
        batch,
        namespace=args.namespace,
        names=names,
        created={"secret", "service", "deployment"},
        manifests=manifests,
        timeout=args.cleanup_timeout_seconds,
        poll_interval=args.poll_interval_seconds,
        request_timeout=args.api_timeout_seconds,
    )
    result = {
        "schema": "npa.workbench.openpi.pi05-service-cleanup.v1",
        "status": "clean",
        "checkpoint_role": args.checkpoint_role,
        "all_exact_resources_absent": bool(verified) and all(verified.values()),
        "verified": verified,
    }
    if not result["all_exact_resources_absent"]:
        raise OpenPIServiceError("service cleanup absence proof is incomplete")
    _write_json_uri(args.output_uri, result)
    stage = 7 if args.checkpoint_role == "base" else 13
    _publish_service_component(
        args,
        stage,
        "service_cleanup",
        evidence="Removed and verified absence of the exact workflow-owned private service.",
        artifacts={"cleanup": args.output_uri},
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "cleanup"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    parser.add_argument("--terms-diagnostic-root-uri", default="")
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    parser.add_argument("--checkpoint-role", choices=("base", "adapted"), required=True)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--expected-gpu-type", required=True)
    parser.add_argument("--expected-compute-capability", required=True)
    parser.add_argument("--server-cpu", default="16")
    parser.add_argument("--server-memory", default="96Gi")
    parser.add_argument("--pull-secret", default="")
    parser.add_argument("--liveness-initial-delay-seconds", type=int, default=600)
    parser.add_argument("--gpu-node-selector-key", default="nebius.com/gpu-name")
    parser.add_argument("--gpu-node-selector-value", required=True)
    parser.add_argument("--service-cache-size", default="40Gi")
    parser.add_argument("--server-ready-timeout-seconds", type=float, default=1200)
    parser.add_argument("--cleanup-timeout-seconds", type=float, default=180)
    parser.add_argument("--poll-interval-seconds", type=float, default=5)
    parser.add_argument("--api-timeout-seconds", type=float, default=30)
    parser.add_argument("--http-timeout-seconds", type=float, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run service lifecycle. Args: argv. Returns: Exit status. Raises: OpenPIServiceError."""
    args = build_parser().parse_args(argv)
    return _start(args) if args.command == "start" else _cleanup(args)


if __name__ == "__main__":
    raise SystemExit(main())
