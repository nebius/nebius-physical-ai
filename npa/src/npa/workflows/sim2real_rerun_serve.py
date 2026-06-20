"""Deploy a hosted Rerun web viewer for a completed Sim2Real run on Kubernetes."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

from npa.clients.config import StorageConfig, resolve_project_storage
from npa.clients.scoped_credentials import bucket_from_s3_uri

# rerunio/rerun:* is not published on Docker Hub; serve via PyPI bootstrap or a
# registry-built npa-sim2real-rerun-viewer image (see sim2real-build.sh).
DEFAULT_RERUN_BOOTSTRAP_IMAGE = "python:3.11-slim-bookworm"
DEFAULT_RERUN_IMAGE = DEFAULT_RERUN_BOOTSTRAP_IMAGE
DEFAULT_RERUN_VIEWER_TOOL = "sim2real-rerun-viewer"
DEFAULT_AWS_CLI_IMAGE = "amazon/aws-cli:2.22.12"
DEFAULT_NAMESPACE = "default"
DEFAULT_PORT = 9090
# Rerun web viewer binds here; nginx sidecar exposes DEFAULT_PORT with cache headers.
RERUN_INTERNAL_WEB_PORT = 9091
DEFAULT_GRPC_PORT = 9876
# Browser gRPC origin for kubectl port-forward (must match forwarded local ports).
DEFAULT_LOCAL_VIEWER_HOST = "127.0.0.1"
DEFAULT_NGINX_IMAGE = "nginx:1.27-alpine"
# Browser-cache static wasm/js (~40 MiB) so refresh does not re-download the app bundle.
RERUN_STATIC_CACHE_CONTROL = "public, max-age=604800, immutable"
# 0.31.x embeds localhost gRPC URLs and lacks --cors-allow-origin; remote LoadBalancer
# viewers stall around wasm load (~37%) then fail to connect. Pin serve pods to 0.32+.
DEFAULT_RERUN_SERVE_SDK_VERSION = "0.32.0"
DEFAULT_S3_PREFIX = "sim2real-b"
DEFAULT_CLUSTER_NAME = "npa-rtxpro-mk8s"
DEFAULT_SERVICE_TYPE = "LoadBalancer"
K8S_NAME_MAX_LEN = 63
K8S_NAME_PREFIX = "npa-sim2real-rerun"
DEFAULT_CLUSTER_VIEWER_SUFFIX = "viewer"
ROLLOUT_TIMEOUT_SEC = 900
DEPLOYMENT_PROGRESS_DEADLINE_SEC = 900
KUBECTL_DELETE_TIMEOUT_SEC = 60

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
    cluster_context: str = ""
    rerun_image: str = DEFAULT_RERUN_IMAGE
    aws_cli_image: str = DEFAULT_AWS_CLI_IMAGE
    service_type: str = DEFAULT_SERVICE_TYPE
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "eu-north1"
    rrd_s3_uri_override: str = ""
    auth_user: str = ""
    auth_password: str = ""

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_password and self.auth_user)

    @property
    def auth_secret_name(self) -> str:
        return f"{self.deployment_name}-auth"

    @property
    def htpasswd_line(self) -> str:
        """nginx basic-auth line using the {SHA} scheme (pure-Python, no apache2-utils)."""
        import hashlib

        digest = base64.b64encode(hashlib.sha1(self.auth_password.encode()).digest()).decode()
        return f"{self.auth_user}:{{SHA}}{digest}\n"

    @property
    def deployment_name(self) -> str:
        return self.name or deployment_name_for_cluster(self.cluster_context)

    @property
    def rrd_s3_uri(self) -> str:
        if self.rrd_s3_uri_override.strip():
            return self.rrd_s3_uri_override.strip()
        prefix = "/".join(part.strip("/") for part in (self.s3_prefix, self.run_id) if part)
        return f"s3://{self.s3_bucket}/{prefix}/reports/sim2real.rrd"

    @property
    def secret_name(self) -> str:
        return f"{self.deployment_name}-s3"

    @property
    def nginx_configmap_name(self) -> str:
        return f"{self.deployment_name}-nginx"


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
    local_url: str
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


def _k8s_name_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def deployment_name_for_cluster(cluster_context: str = "") -> str:
    """Return the shared mk8s Deployment/Service name for one cluster viewer."""

    context = cluster_context.strip()
    if not context:
        return f"{K8S_NAME_PREFIX}-{DEFAULT_CLUSTER_VIEWER_SUFFIX}"
    slug = _k8s_name_slug(context)
    if not slug:
        return f"{K8S_NAME_PREFIX}-{DEFAULT_CLUSTER_VIEWER_SUFFIX}"
    base = f"{K8S_NAME_PREFIX}-{slug}"
    if len(base) <= K8S_NAME_MAX_LEN:
        return base
    digest = re.sub(r"[^a-z0-9]", "", context.lower())[:8] or "ctx"
    trimmed = slug[: K8S_NAME_MAX_LEN - len(K8S_NAME_PREFIX) - len(digest) - 2].rstrip("-")
    return f"{K8S_NAME_PREFIX}-{trimmed}-{digest}"


def deployment_name_for_run(run_id: str) -> str:
    """Deprecated alias kept for tests; cluster viewers are not keyed by run_id."""

    del run_id
    return deployment_name_for_cluster("")


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


def default_rerun_image() -> str:
    """Return the Rerun viewer container image for mk8s serve deployments."""

    override = os.environ.get("NPA_SIM2REAL_RERUN_IMAGE", "").strip()
    if override:
        return override
    return DEFAULT_RERUN_BOOTSTRAP_IMAGE


def _rerun_image_has_preinstalled_cli(image: str) -> bool:
    lowered = image.strip().lower()
    return "npa-sim2real-rerun-viewer" in lowered or lowered.startswith("rerunio/rerun:")


def rerun_serve_sdk_version() -> str:
    override = os.environ.get("NPA_SIM2REAL_RERUN_SERVE_VERSION", "").strip()
    return override or DEFAULT_RERUN_SERVE_SDK_VERSION


def _rerun_remote_cors_flags() -> str:
    # rerun 0.32+ only; allow Nebius LoadBalancer origins (204.12.*) and any http host.
    return (
        "--cors-allow-origin 'http://*:*' "
        "--cors-allow-origin 'http://204.12.*:*' "
    )


RERUN_HTPASSWD_PATH = "/etc/nginx/auth/.htpasswd"


def build_rerun_nginx_config(
    *,
    external_port: int = DEFAULT_PORT,
    internal_port: int = RERUN_INTERNAL_WEB_PORT,
    cache_control: str = RERUN_STATIC_CACHE_CONTROL,
    auth_required: bool = False,
    htpasswd_path: str = RERUN_HTPASSWD_PATH,
) -> str:
    """Return nginx config: proxy static assets with long-lived browser cache.

    When ``auth_required`` the public viewer is gated behind HTTP basic-auth
    (``auth_basic`` + htpasswd file), so the cloud LoadBalancer URL needs
    credentials instead of being open to anyone with the IP.
    """

    auth_lines = ""
    health_block = ""
    if auth_required:
        auth_lines = (
            f'\n            auth_basic "NPA Sim2Real Rerun";'
            f'\n            auth_basic_user_file {htpasswd_path};'
        )
        # Unauthenticated health endpoint so the readiness probe passes (GET / is 401).
        health_block = (
            '\n        location = /healthz {'
            '\n            auth_basic off;'
            '\n            return 200 "ok";'
            '\n        }'
        )
    return f"""\
worker_processes 1;
error_log /dev/stderr warn;
pid /tmp/nginx.pid;
events {{ worker_connections 1024; }}
http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    access_log /dev/stdout;
    sendfile on;
    upstream rerun_web {{
        server 127.0.0.1:{internal_port};
    }}
    server {{
        listen {external_port};{health_block}
        location ~* \\.(wasm|js|ico|svg)$ {{
            proxy_pass http://rerun_web;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            add_header Cache-Control "{cache_control}" always;{auth_lines}
        }}
        location / {{
            proxy_pass http://rerun_web;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            add_header Cache-Control "no-cache" always;{auth_lines}
        }}
    }}
}}
"""


def _rerun_serve_command(config: RerunServeConfig) -> str:
    sdk_version = rerun_serve_sdk_version()
    base_cmd = (
        "rerun /data/sim2real.rrd --serve-web --web-viewer "
        f"--web-viewer-port {RERUN_INTERNAL_WEB_PORT} --port {DEFAULT_GRPC_PORT} --bind 0.0.0.0 "
        f"{_rerun_remote_cors_flags()}"
    )
    if _rerun_image_has_preinstalled_cli(config.rerun_image):
        return base_cmd
    return (
        f"python -m pip install --no-cache-dir 'rerun-sdk=={sdk_version}' "
        f"&& exec {base_cmd}"
    )


def public_viewer_url(host: str, *, http_port: int, grpc_port: int = DEFAULT_GRPC_PORT) -> str:
    """Return a viewer URL whose gRPC origin matches the HTTP page host."""

    host = host.strip()
    if not host:
        return ""
    proxy = f"rerun+http://{host}:{grpc_port}/proxy"
    return f"http://{host}:{http_port}/?url={quote(proxy, safe='')}"


def local_viewer_url(
    *,
    http_port: int,
    grpc_port: int = DEFAULT_GRPC_PORT,
    host: str = DEFAULT_LOCAL_VIEWER_HOST,
) -> str:
    """Return a port-forward URL (127.0.0.1) with matching local gRPC proxy origin."""

    return public_viewer_url(host, http_port=http_port, grpc_port=grpc_port)


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
    cluster_context: str = "",
    rerun_image: str = DEFAULT_RERUN_IMAGE,
    aws_cli_image: str = DEFAULT_AWS_CLI_IMAGE,
    service_type: str = DEFAULT_SERVICE_TYPE,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_region: str = "eu-north1",
    rrd_s3_uri: str = "",
    report_uri: str = "",
    auth_user: str = "",
    auth_password: str = "",
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
        cluster_context=cluster_context.strip(),
        rerun_image=rerun_image.strip() or default_rerun_image(),
        aws_cli_image=aws_cli_image.strip() or DEFAULT_AWS_CLI_IMAGE,
        service_type=normalized_service,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region.strip() or "eu-north1",
        rrd_s3_uri_override=rrd_override,
        auth_user=auth_user.strip(),
        auth_password=auth_password,
    )


def fetch_rrd_sync_token(
    config: RerunServeConfig,
    *,
    head_object: Callable[..., Any] | None = None,
) -> str:
    """Return an S3 ETag (or URI fallback) so serve rollouts re-sync updated .rrd files."""

    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    uri = config.rrd_s3_uri
    if not uri.startswith("s3://"):
        return uri
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    if head_object is not None:
        try:
            response = head_object(Bucket=bucket, Key=key)
        except ClientError:
            return uri
        etag = str(response.get("ETag") or "").strip('"')
        return etag or uri

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
        response = client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return uri
    etag = str(response.get("ETag") or "").strip('"')
    return etag or uri


def build_rerun_serve_manifest(
    config: RerunServeConfig,
    *,
    rrd_sync_token: str = "",
) -> dict[str, Any]:
    labels = {
        "app": config.deployment_name,
        "app.kubernetes.io/name": "npa-sim2real-rerun",
        "app.kubernetes.io/instance": config.deployment_name,
        "app.kubernetes.io/component": "cluster-viewer",
        "npa.nebius.com/sim2real-run-id": _label_value(config.run_id),
    }
    if config.cluster_context.strip():
        labels["npa.nebius.com/k8s-context"] = _label_value(config.cluster_context)
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
    serve_command = _rerun_serve_command(config)
    nginx_config = build_rerun_nginx_config(
        external_port=config.port, auth_required=config.auth_enabled
    )
    sync_token = (rrd_sync_token or config.run_id).strip()
    pod_annotations = {
        "npa.nebius.com/rrd-s3-uri": _label_value(config.rrd_s3_uri),
        "npa.nebius.com/rrd-sync-token": _label_value(sync_token),
    }
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
            *(
                [{
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": config.auth_secret_name,
                        "namespace": config.namespace,
                        "labels": labels,
                    },
                    "type": "Opaque",
                    "data": {".htpasswd": _b64(config.htpasswd_line)},
                }]
                if config.auth_enabled
                else []
            ),
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": config.nginx_configmap_name,
                    "namespace": config.namespace,
                    "labels": labels,
                },
                "data": {"nginx.conf": nginx_config},
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
                        "metadata": {"labels": labels, "annotations": pod_annotations},
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
                                    "name": "nginx",
                                    "image": DEFAULT_NGINX_IMAGE,
                                    "imagePullPolicy": "IfNotPresent",
                                    "ports": [{"name": "http", "containerPort": config.port}],
                                    "readinessProbe": {
                                        "httpGet": {
                                            "path": "/healthz" if config.auth_enabled else "/",
                                            "port": "http",
                                        },
                                        "initialDelaySeconds": 5,
                                        "periodSeconds": 5,
                                        "timeoutSeconds": 3,
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "nginx-config",
                                            "mountPath": "/etc/nginx/nginx.conf",
                                            "subPath": "nginx.conf",
                                        }
                                    ] + (
                                        [{
                                            "name": "nginx-auth",
                                            "mountPath": "/etc/nginx/auth",
                                            "readOnly": True,
                                        }] if config.auth_enabled else []
                                    ),
                                    "resources": {
                                        "requests": {"cpu": "50m", "memory": "64Mi"},
                                        "limits": {"cpu": "500m", "memory": "128Mi"},
                                    },
                                },
                                {
                                    "name": "rerun",
                                    "image": config.rerun_image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["/bin/sh", "-c", serve_command],
                                    "ports": [
                                        {
                                            "name": "web-internal",
                                            "containerPort": RERUN_INTERNAL_WEB_PORT,
                                        },
                                        {"name": "grpc", "containerPort": DEFAULT_GRPC_PORT},
                                    ],
                                    "readinessProbe": {
                                        "httpGet": {
                                            "path": "/",
                                            "port": RERUN_INTERNAL_WEB_PORT,
                                        },
                                        "initialDelaySeconds": (
                                            15
                                            if _rerun_image_has_preinstalled_cli(config.rerun_image)
                                            else 90
                                        ),
                                        "periodSeconds": 10,
                                        "timeoutSeconds": 5,
                                    },
                                    "resources": {
                                        "requests": {"cpu": "250m", "memory": "512Mi"},
                                        "limits": {"cpu": "2", "memory": "2Gi"},
                                    },
                                    "volumeMounts": [
                                        {"name": "rrd-data", "mountPath": "/data", "readOnly": True}
                                    ],
                                },
                            ],
                            "volumes": [
                                {"name": "rrd-data", "emptyDir": {}},
                                {
                                    "name": "nginx-config",
                                    "configMap": {"name": config.nginx_configmap_name},
                                },
                            ] + (
                                [{
                                    "name": "nginx-auth",
                                    "secret": {
                                        "secretName": config.auth_secret_name,
                                        "items": [{"key": ".htpasswd", "path": ".htpasswd"}],
                                    },
                                }] if config.auth_enabled else []
                            ),
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
                    "ports": [
                        {"name": "http", "port": config.port, "targetPort": "http"},
                        {
                            "name": "grpc",
                            "port": DEFAULT_GRPC_PORT,
                            "targetPort": "grpc",
                        },
                    ],
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
        http_port=config.port,
        grpc_port=DEFAULT_GRPC_PORT,
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
        local_url=local_viewer_url(http_port=config.port, grpc_port=DEFAULT_GRPC_PORT),
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

    verify_rrd_exists_on_s3(config)
    sync_token = fetch_rrd_sync_token(config)
    manifest = build_rerun_serve_manifest(config, rrd_sync_token=sync_token)
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
                public_url = public_viewer_url(
                    host, http_port=config.port, grpc_port=DEFAULT_GRPC_PORT
                )
                break
            waiter(5)

    return rerun_serve_result(config, status="deployed", public_url=public_url, kubeconfig=kubeconfig)


def destroy_rerun_serve(
    config: RerunServeConfig,
    *,
    kubeconfig: str,
    kubectl: Callable[..., str] | None = None,
    wait: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RerunServeResult:
    runner = kubectl or _default_kubectl
    notify = progress or (lambda message: print(message, flush=True))
    for kind in ("service", "deployment", "configmap", "secret"):
        if kind == "secret":
            name = config.secret_name
        elif kind == "configmap":
            name = config.nginx_configmap_name
        else:
            name = config.deployment_name
        notify(
            f"Deleting {kind}/{name} in namespace {config.namespace} "
            f"(wait={'true' if wait else 'false'}, "
            f"timeout={KUBECTL_DELETE_TIMEOUT_SEC}s)..."
        )
        runner(
            [
                "delete",
                kind,
                name,
                "-n",
                config.namespace,
                "--ignore-not-found=true",
                "--wait=true" if wait else "--wait=false",
                f"--request-timeout={KUBECTL_DELETE_TIMEOUT_SEC}s",
            ],
            kubeconfig=kubeconfig,
            timeout_sec=KUBECTL_DELETE_TIMEOUT_SEC + 5,
        )
    notify(
        f"Deleted shared cluster Rerun viewer {config.deployment_name} "
        f"(was serving run_id={config.run_id!r})"
    )
    return rerun_serve_result(config, status="deleted", kubeconfig=kubeconfig)


def in_cluster_kubernetes() -> bool:
    """True when running inside a Kubernetes pod with in-cluster API access."""

    return bool(os.environ.get("KUBERNETES_SERVICE_HOST", "").strip())


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
    timeout_sec: float | None = None,
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
        result = subprocess.run(
            cmd,
            input=stdin,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise Sim2RealRerunServeError(
            f"kubectl command timed out after {timeout_sec}s: {' '.join(args[:4])}"
        ) from exc
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


def _port_forward_command(
    *,
    deployment_name: str,
    namespace: str,
    http_port: int,
    grpc_port: int = DEFAULT_GRPC_PORT,
    kubeconfig: str,
) -> str:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(
        [
            "port-forward",
            "-n",
            namespace,
            f"deployment/{deployment_name}",
            f"{http_port}:{http_port}",
            f"{grpc_port}:{grpc_port}",
        ]
    )
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


def _env_bool(value: str, *, default: bool = True) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized not in {"0", "false", "no", "off"}


def should_auto_rerun_serve(
    *,
    rerun_enabled: bool,
    s3_bucket: str,
    upload_status: str,
    viz_status: str,
) -> bool:
    """Return True when finalize should deploy the hosted Rerun viewer on mk8s."""

    if not _env_bool(os.environ.get("NPA_SIM2REAL_RERUN_SERVE", "1")):
        return False
    if not rerun_enabled:
        return False
    if not s3_bucket.strip():
        return False
    if upload_status != "uploaded":
        return False
    if viz_status in {"disabled", "skipped"}:
        return False
    return True


def resolve_rerun_serve_credentials() -> tuple[str, str]:
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if access_key and secret_key:
        return access_key, secret_key
    try:
        from npa.clients.credentials import load_credentials

        creds = load_credentials()
        access_key = access_key or (creds.s3_access_key_id or "").strip()
        secret_key = secret_key or (creds.s3_secret_access_key or "").strip()
    except Exception:
        pass
    if not access_key or not secret_key:
        raise Sim2RealRerunServeError(
            "S3 credentials are required for auto rerun serve. Configure ~/.npa/credentials.yaml "
            "or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )
    return access_key, secret_key


def maybe_auto_rerun_serve(
    *,
    run_id: str,
    s3_bucket: str,
    s3_prefix: str = DEFAULT_S3_PREFIX,
    s3_endpoint: str = "",
    rerun_enabled: bool,
    upload_info: dict[str, Any],
    viz_info: dict[str, Any],
    k8s_kubeconfig: str = "",
    k8s_namespace: str = DEFAULT_NAMESPACE,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
) -> dict[str, Any]:
    """Deploy idempotent hosted Rerun for a completed run when gates pass.

    One shared Deployment/LoadBalancer per mk8s cluster (stable
    ``deployment_name`` and ``public_url``). Serving a new ``run_id`` updates the
    synced ``sim2real.rrd`` without allocating a new external IP. Degrades to
    ``skipped`` / ``blocked`` with a reason instead of failing the workflow.
    """

    upload_status = str(upload_info.get("status", ""))
    viz_status = str(viz_info.get("status", ""))
    if not should_auto_rerun_serve(
        rerun_enabled=rerun_enabled,
        s3_bucket=s3_bucket,
        upload_status=upload_status,
        viz_status=viz_status,
    ):
        return {
            "status": "skipped",
            "reason": (
                "auto rerun serve disabled or prerequisites missing "
                f"(rerun_enabled={rerun_enabled}, upload={upload_status!r}, viz={viz_status!r})"
            ),
        }

    try:
        normalized_run_id = validate_staged_run_id(run_id)
    except Sim2RealRerunServeError as exc:
        return {"status": "skipped", "reason": str(exc)}

    try:
        access_key, secret_key = (
            (aws_access_key_id.strip(), aws_secret_access_key.strip())
            if aws_access_key_id.strip() and aws_secret_access_key.strip()
            else resolve_rerun_serve_credentials()
        )
        cluster_context = resolve_cluster_name_from_config()
        serve_config = build_rerun_serve_config(
            run_id=normalized_run_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
            namespace=k8s_namespace,
            cluster_context=cluster_context,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        kubeconfig = resolve_kubeconfig_path(
            cluster_name=cluster_context,
            kubeconfig=k8s_kubeconfig,
        )
        if not kubeconfig and not in_cluster_kubernetes():
            return {
                "status": "blocked",
                "reason": (
                    "No kubeconfig for auto rerun serve. Run "
                    f"'npa workbench sim2real rerun serve --run-id {normalized_run_id}' "
                    "from an operator shell with cluster access."
                ),
                "rrd_s3_uri": serve_config.rrd_s3_uri,
                "deployment_name": serve_config.deployment_name,
            }
        result = apply_rerun_serve(serve_config, kubeconfig=kubeconfig)
    except Sim2RealRerunServeError as exc:
        return {"status": "blocked", "reason": str(exc), "run_id": run_id}

    payload = result.to_dict()
    if payload.get("public_url"):
        print(f"public_url: {payload['public_url']}", flush=True)
    if payload.get("local_url"):
        print(f"local_url: {payload['local_url']}", flush=True)
    if payload.get("port_forward_command"):
        print(f"port_forward: {payload['port_forward_command']}", flush=True)
    return payload
