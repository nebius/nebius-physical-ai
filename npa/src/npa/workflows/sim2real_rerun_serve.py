"""Deploy a hosted Rerun web viewer for a completed Sim2Real run on Kubernetes."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from npa.cli.rerun import RERUN_VERSION
from npa.clients.config import StorageConfig, resolve_project_storage
from npa.clients.scoped_credentials import bucket_from_s3_uri

DEFAULT_RERUN_IMAGE = f"rerunio/rerun:{RERUN_VERSION}"
DEFAULT_AWS_CLI_IMAGE = "amazon/aws-cli:2.22.12"
DEFAULT_NAMESPACE = "default"
DEFAULT_PORT = 9090
DEFAULT_S3_PREFIX = "sim2real-b"
DEFAULT_CLUSTER_NAME = "npa-rtxpro-mk8s"
DEFAULT_SERVICE_TYPE = "LoadBalancer"
K8S_NAME_MAX_LEN = 63
K8S_NAME_PREFIX = "npa-sim2real-rerun"
ROLLOUT_TIMEOUT_SEC = 900
DEPLOYMENT_PROGRESS_DEADLINE_SEC = 900

STAGED_RUN_ID_RE = re.compile(
    r"^(?:sim2real-staged-[0-9]{8}t[0-9]{6}z|rtxpro-staged-[a-z0-9-]*[0-9]{8}t[0-9]{6}z)$",
    re.IGNORECASE,
)
PLACEHOLDER_RUN_ID_RE = re.compile(
    r"yyyymmdd|hhmmss|your-run-id|<run-id>|placeholder|example-run|tbd|xxxx",
    re.IGNORECASE,
)


class Sim2RealRerunServeError(ValueError):
    """Raised when rerun serve manifest generation or deployment fails."""


@dataclass(frozen=True)
class RerunServeConfig:
    run_id: str
    s3_bucket: str
    s3_prefix: str = DEFAULT_S3_PREFIX
    s3_endpoint: str = ""
    namespace: str = DEFAULT_NAMESPACE
    port: int = DEFAULT_PORT
    name: str = ""
    rerun_image: str = DEFAULT_RERUN_IMAGE
    aws_cli_image: str = DEFAULT_AWS_CLI_IMAGE
    service_type: str = DEFAULT_SERVICE_TYPE
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "eu-north1"
    rrd_s3_uri_override: str = ""

    @property
    def deployment_name(self) -> str:
        return self.name or deployment_name_for_run(self.run_id)

    @property
    def rrd_s3_uri(self) -> str:
        if self.rrd_s3_uri_override.strip():
            return self.rrd_s3_uri_override.strip()
        prefix = "/".join(part.strip("/") for part in (self.s3_prefix, self.run_id) if part)
        return f"s3://{self.s3_bucket}/{prefix}/reports/sim2real.rrd"

    @property
    def secret_name(self) -> str:
        return f"{self.deployment_name}-s3"


@dataclass(frozen=True)
class RerunServeResult:
    status: str
    run_id: str
    deployment_name: str
    namespace: str
    rrd_s3_uri: str
    service_type: str
    port: int
    cluster_url: str
    public_url: str
    port_forward_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_staged_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value:
        raise Sim2RealRerunServeError("--run-id is required")
    if PLACEHOLDER_RUN_ID_RE.search(value):
        raise Sim2RealRerunServeError(
            f"run-id looks like a template placeholder: {value!r}. "
            "Use a real id from submit output (e.g. sim2real-staged-20260615t180818z)."
        )
    if not STAGED_RUN_ID_RE.fullmatch(value):
        raise Sim2RealRerunServeError(
            "run-id must match sim2real-staged-YYYYMMDDtHHMMSSz with digit timestamps "
            f"(got {value!r})."
        )
    return value


def verify_rrd_exists_on_s3(
    config: RerunServeConfig,
    *,
    head_object: Callable[..., Any] | None = None,
) -> None:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    uri = config.rrd_s3_uri
    if not uri.startswith("s3://"):
        raise Sim2RealRerunServeError(f"invalid rrd s3 uri: {uri}")
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise Sim2RealRerunServeError(f"invalid rrd s3 uri: {uri}")

    if head_object is not None:
        try:
            head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "missing")
            raise Sim2RealRerunServeError(
                f"Rerun recording not found at {uri} ({code}). "
                "Wait for reports/sim2real.rrd on S3 before rerun serve."
            ) from exc
        return

    client_kwargs: dict[str, Any] = {
        "aws_access_key_id": config.aws_access_key_id,
        "aws_secret_access_key": config.aws_secret_access_key,
        "config": Config(signature_version="s3v4"),
        "region_name": config.aws_region,
    }
    if config.s3_endpoint:
        client_kwargs["endpoint_url"] = config.s3_endpoint
    client = boto3.client("s3", **client_kwargs)
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "missing")
        raise Sim2RealRerunServeError(
            f"Rerun recording not found at {uri} ({code}). "
            "Wait for reports/sim2real.rrd on S3 before rerun serve."
        ) from exc


def deployment_name_for_run(run_id: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", run_id.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        raise Sim2RealRerunServeError("run-id must contain at least one alphanumeric character")
    base = f"{K8S_NAME_PREFIX}-{slug}"
    if len(base) <= K8S_NAME_MAX_LEN:
        return base
    digest = re.sub(r"[^a-z0-9]", "", run_id.lower())[:8] or "run"
    trimmed = slug[: K8S_NAME_MAX_LEN - len(K8S_NAME_PREFIX) - len(digest) - 2].rstrip("-")
    return f"{K8S_NAME_PREFIX}-{trimmed}-{digest}"


def resolve_storage_bucket(storage: StorageConfig, *, override: str = "") -> str:
    if override.strip():
        value = override.strip()
        return bucket_from_s3_uri(value) if value.startswith("s3://") else value
    configured = storage.checkpoint_bucket
    if not configured:
        raise Sim2RealRerunServeError(
            "S3 bucket is not configured. Pass --s3-bucket or configure storage.bucket."
        )
    return bucket_from_s3_uri(configured) if configured.startswith("s3://") else configured


def rrd_s3_uri_from_report_uri(report_uri: str) -> str:
    uri = report_uri.strip()
    if not uri.startswith("s3://"):
        raise Sim2RealRerunServeError("--report-uri must be an s3:// URI")
    if uri.endswith("/sim2real-report.json"):
        return uri[: -len("sim2real-report.json")] + "sim2real.rrd"
    if uri.endswith("sim2real-report.json"):
        return uri.replace("sim2real-report.json", "sim2real.rrd")
    raise Sim2RealRerunServeError(
        "--report-uri must end with reports/sim2real-report.json"
    )


def resolve_cluster_name_from_config() -> str:
    import os

    import yaml

    for env_key in ("NPA_K8S_CONTEXT", "KUBECONTEXT"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value
    cfg_path = Path.home() / ".npa" / "config.yaml"
    if not cfg_path.exists():
        return ""
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except OSError:
        return ""
    storage = cfg.get("storage") or {}
    k8s_context = str(storage.get("k8s_context", "") or "").strip()
    if k8s_context:
        return k8s_context
    for proj in (cfg.get("projects") or {}).values():
        if isinstance(proj, dict) and proj.get("k8s_context"):
            return str(proj["k8s_context"]).strip()
    return ""


def build_rerun_serve_config(
    *,
    run_id: str,
    project: str | None = None,
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_S3_PREFIX,
    s3_endpoint: str = "",
    namespace: str = DEFAULT_NAMESPACE,
    port: int = DEFAULT_PORT,
    name: str = "",
    rerun_image: str = DEFAULT_RERUN_IMAGE,
    aws_cli_image: str = DEFAULT_AWS_CLI_IMAGE,
    service_type: str = DEFAULT_SERVICE_TYPE,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_region: str = "eu-north1",
    rrd_s3_uri: str = "",
    report_uri: str = "",
) -> RerunServeConfig:
    normalized_run_id = validate_staged_run_id(run_id)
    storage = resolve_project_storage(project)
    bucket = resolve_storage_bucket(storage, override=s3_bucket)
    endpoint = s3_endpoint.strip() or storage.endpoint_url or ""
    if not aws_access_key_id or not aws_secret_access_key:
        raise Sim2RealRerunServeError(
            "S3 credentials are required to sync sim2real.rrd into the cluster pod."
        )
    rrd_override = rrd_s3_uri.strip()
    if report_uri.strip():
        rrd_override = rrd_s3_uri_from_report_uri(report_uri)
    normalized_service = _normalize_service_type(service_type)
    return RerunServeConfig(
        run_id=normalized_run_id,
        s3_bucket=bucket,
        s3_prefix=s3_prefix.strip("/") or DEFAULT_S3_PREFIX,
        s3_endpoint=endpoint,
        namespace=namespace.strip() or DEFAULT_NAMESPACE,
        port=port,
        name=name.strip(),
        rerun_image=rerun_image.strip() or DEFAULT_RERUN_IMAGE,
        aws_cli_image=aws_cli_image.strip() or DEFAULT_AWS_CLI_IMAGE,
        service_type=normalized_service,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region.strip() or "eu-north1",
        rrd_s3_uri_override=rrd_override,
    )


def build_rerun_serve_manifest(config: RerunServeConfig) -> dict[str, Any]:
    labels = {
        "app": config.deployment_name,
        "app.kubernetes.io/name": "npa-sim2real-rerun",
        "app.kubernetes.io/instance": config.deployment_name,
        "npa.nebius.com/sim2real-run-id": _label_value(config.run_id),
    }
    secret_data = {
        "AWS_ACCESS_KEY_ID": _b64(config.aws_access_key_id),
        "AWS_SECRET_ACCESS_KEY": _b64(config.aws_secret_access_key),
    }
    if config.s3_endpoint:
        secret_data["AWS_ENDPOINT_URL"] = _b64(config.s3_endpoint)
        secret_data["AWS_ENDPOINT_URL_S3"] = _b64(config.s3_endpoint)
    secret_data["AWS_DEFAULT_REGION"] = _b64(config.aws_region)
    secret_data["S3_URI"] = _b64(config.rrd_s3_uri)

    init_script = """\
set -eu
aws s3 cp "${S3_URI}" /data/sim2real.rrd
test -s /data/sim2real.rrd
"""
    serve_command = (
        "rerun /data/sim2real.rrd --web-viewer "
        f"--web-viewer-port {config.port} --bind 0.0.0.0"
    )
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": config.secret_name,
                    "namespace": config.namespace,
                    "labels": labels,
                },
                "type": "Opaque",
                "data": secret_data,
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": config.deployment_name,
                    "namespace": config.namespace,
                    "labels": labels,
                },
                "spec": {
                    "replicas": 1,
                    "progressDeadlineSeconds": DEPLOYMENT_PROGRESS_DEADLINE_SEC,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": {"app.kubernetes.io/instance": config.deployment_name}},
                    "template": {
                        "metadata": {"labels": labels},
                        "spec": {
                            "initContainers": [
                                {
                                    "name": "sync-rrd",
                                    "image": config.aws_cli_image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["/bin/sh", "-c", init_script],
                                    "envFrom": [{"secretRef": {"name": config.secret_name}}],
                                    "volumeMounts": [{"name": "rrd-data", "mountPath": "/data"}],
                                }
                            ],
                            "containers": [
                                {
                                    "name": "rerun",
                                    "image": config.rerun_image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["/bin/sh", "-c", serve_command],
                                    "ports": [{"name": "http", "containerPort": config.port}],
                                    "readinessProbe": {
                                        "httpGet": {"path": "/", "port": "http"},
                                        "initialDelaySeconds": 5,
                                        "periodSeconds": 10,
                                        "timeoutSeconds": 5,
                                    },
                                    "resources": {
                                        "requests": {"cpu": "250m", "memory": "512Mi"},
                                        "limits": {"cpu": "2", "memory": "2Gi"},
                                    },
                                    "volumeMounts": [{"name": "rrd-data", "mountPath": "/data", "readOnly": True}],
                                }
                            ],
                            "volumes": [{"name": "rrd-data", "emptyDir": {}}],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": config.deployment_name,
                    "namespace": config.namespace,
                    "labels": labels,
                },
                "spec": {
                    "type": config.service_type,
                    "selector": {"app.kubernetes.io/instance": config.deployment_name},
                    "ports": [{"name": "http", "port": config.port, "targetPort": "http"}],
                },
            },
        ],
    }


def redact_rerun_serve_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(manifest))
    for item in redacted.get("items", []):
        if item.get("kind") == "Secret":
            item["data"] = {key: "<redacted>" for key in item.get("data", {})}
    return redacted


def rerun_serve_result(
    config: RerunServeConfig,
    *,
    status: str,
    public_url: str = "",
    kubeconfig: str = "",
) -> RerunServeResult:
    cluster_url = f"http://{config.deployment_name}.{config.namespace}.svc.cluster.local:{config.port}"
    port_forward = _port_forward_command(
        deployment_name=config.deployment_name,
        namespace=config.namespace,
        port=config.port,
        kubeconfig=kubeconfig,
    )
    return RerunServeResult(
        status=status,
        run_id=config.run_id,
        deployment_name=config.deployment_name,
        namespace=config.namespace,
        rrd_s3_uri=config.rrd_s3_uri,
        service_type=config.service_type,
        port=config.port,
        cluster_url=cluster_url,
        public_url=public_url,
        port_forward_command=port_forward,
    )


def apply_rerun_serve(
    config: RerunServeConfig,
    *,
    kubeconfig: str,
    kubectl: Callable[..., str] | None = None,
    wait_for_public_url: bool = True,
    public_url_timeout_sec: int = 300,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    get_service: Callable[[str, str, str], dict[str, Any] | None] | None = None,
) -> RerunServeResult:
    runner = kubectl or _default_kubectl
    getter = get_service or _default_get_service
    clock = now or time.monotonic
    waiter = sleep or time.sleep

    manifest = build_rerun_serve_manifest(config)
    verify_rrd_exists_on_s3(config)
    runner(["apply", "-f", "-"], stdin=json.dumps(manifest), kubeconfig=kubeconfig)
    try:
        runner(
            [
                "rollout",
                "status",
                f"deployment/{config.deployment_name}",
                "-n",
                config.namespace,
                f"--timeout={ROLLOUT_TIMEOUT_SEC}s",
            ],
            kubeconfig=kubeconfig,
        )
    except Sim2RealRerunServeError as exc:
        detail = _rollout_failure_diagnostics(config, kubeconfig, runner)
        if detail:
            raise Sim2RealRerunServeError(f"{exc}\n{detail}") from exc
        raise

    public_url = ""
    if config.service_type == "LoadBalancer" and wait_for_public_url:
        deadline = clock() + public_url_timeout_sec
        while clock() < deadline:
            service = getter("service", config.deployment_name, kubeconfig)
            host = _service_external_host(service or {})
            if host:
                public_url = f"http://{host}:{config.port}/"
                break
            waiter(5)

    return rerun_serve_result(config, status="deployed", public_url=public_url, kubeconfig=kubeconfig)


def destroy_rerun_serve(
    config: RerunServeConfig,
    *,
    kubeconfig: str,
    kubectl: Callable[..., str] | None = None,
) -> RerunServeResult:
    runner = kubectl or _default_kubectl
    for kind in ("service", "deployment", "secret"):
        runner(
            ["delete", kind, config.deployment_name if kind != "secret" else config.secret_name, "-n", config.namespace, "--ignore-not-found=true"],
            kubeconfig=kubeconfig,
        )
    return rerun_serve_result(config, status="deleted", kubeconfig=kubeconfig)


def resolve_kubeconfig_path(*, cluster_name: str, kubeconfig: str) -> str:
    import os

    if kubeconfig.strip():
        return kubeconfig.strip()
    env_kube = os.environ.get("KUBECONFIG", "").strip()
    if env_kube:
        for candidate in env_kube.split(os.pathsep):
            if candidate and Path(candidate).exists():
                return candidate
    profile = cluster_name.strip() or resolve_cluster_name_from_config() or DEFAULT_CLUSTER_NAME
    for filename in ("kubeconfig.resolved", "kubeconfig"):
        path = Path.home() / ".npa" / "clusters" / profile / filename
        if path.exists():
            return str(path)
    return ""


def require_kubeconfig(*, cluster_name: str, kubeconfig: str) -> str:
    resolved = resolve_kubeconfig_path(cluster_name=cluster_name, kubeconfig=kubeconfig)
    if not resolved:
        profile = cluster_name.strip() or resolve_cluster_name_from_config() or DEFAULT_CLUSTER_NAME
        raise Sim2RealRerunServeError(
            "No kubeconfig found. Pass --kubeconfig, export KUBECONFIG, or configure "
            f"~/.npa/clusters/{profile}/kubeconfig."
        )
    return resolved


def _rollout_failure_diagnostics(
    config: RerunServeConfig,
    kubeconfig: str,
    kubectl: Callable[..., str],
) -> str:
    lines: list[str] = []
    pod_cmd = [
        "get",
        "pods",
        "-n",
        config.namespace,
        "-l",
        f"app.kubernetes.io/instance={config.deployment_name}",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}",
    ]
    try:
        pod_names = kubectl(pod_cmd, kubeconfig=kubeconfig).strip().splitlines()
    except Sim2RealRerunServeError:
        return ""

    for pod_name in pod_names:
        if not pod_name:
            continue
        lines.append(f"pod: {pod_name}")
        for container in ("sync-rrd", "rerun"):
            log_cmd = [
                "logs",
                "-n",
                config.namespace,
                pod_name,
                "-c",
                container,
                "--tail=40",
            ]
            try:
                logs = kubectl(log_cmd, kubeconfig=kubeconfig).strip()
            except Sim2RealRerunServeError:
                logs = ""
            if logs:
                lines.append(f"--- {container} logs ---")
                lines.append(logs)
        describe_cmd = ["describe", "pod", "-n", config.namespace, pod_name]
        try:
            desc = kubectl(describe_cmd, kubeconfig=kubeconfig)
        except Sim2RealRerunServeError:
            desc = ""
        for marker in ("Init Containers:", "State:", "Reason:", "Message:", "Warning"):
            for raw in desc.splitlines():
                if marker in raw:
                    lines.append(raw.strip())
    if not lines:
        return (
            "Hint: init container sync-rrd pulls sim2real.rrd from S3 using credentials "
            f"in Secret {config.secret_name!r}. Check RUN_ID exists and S3 creds are valid."
        )
    return "\n".join(lines)


def _default_kubectl(
    args: list[str],
    *,
    stdin: str | None = None,
    kubeconfig: str = "",
) -> str:
    import shutil
    import subprocess

    if shutil.which("kubectl") is None:
        raise Sim2RealRerunServeError("kubectl is not installed or not on PATH")
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(args)
    try:
        result = subprocess.run(cmd, input=stdin, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise Sim2RealRerunServeError(f"kubectl command failed: {detail}") from exc
    return result.stdout


def _default_get_service(kind: str, name: str, kubeconfig: str) -> dict[str, Any] | None:
    import shutil
    import subprocess

    if shutil.which("kubectl") is None:
        return None
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(["get", kind, name, "-o", "json"])
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _service_external_host(service: dict[str, Any]) -> str:
    for item in service.get("status", {}).get("loadBalancer", {}).get("ingress", []) or []:
        host = str(item.get("ip") or item.get("hostname") or "").strip()
        if host:
            return host
    node_port = _node_port(service)
    external_ips = service.get("spec", {}).get("externalIPs", []) or []
    if external_ips:
        return str(external_ips[0]).strip()
    if node_port:
        return ""
    return ""


def _node_port(service: dict[str, Any]) -> int:
    ports = service.get("spec", {}).get("ports", []) or []
    for item in ports:
        node_port = item.get("nodePort")
        if isinstance(node_port, int) and node_port > 0:
            return node_port
    return 0


def _port_forward_command(*, deployment_name: str, namespace: str, port: int, kubeconfig: str) -> str:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(["port-forward", "-n", namespace, f"deployment/{deployment_name}", f"{port}:{port}"])
    return " ".join(cmd)


def _normalize_service_type(service_type: str) -> str:
    normalized = service_type.strip().lower()
    mapping = {
        "loadbalancer": "LoadBalancer",
        "lb": "LoadBalancer",
        "nodeport": "NodePort",
        "clusterip": "ClusterIP",
    }
    if normalized not in mapping:
        raise Sim2RealRerunServeError(
            "--service-type must be one of: LoadBalancer, NodePort, ClusterIP"
        )
    return mapping[normalized]


def _label_value(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value.strip())[:63].strip("-_.")
    return cleaned or "run"


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")
