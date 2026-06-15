from __future__ import annotations

import json

import pytest

from npa.clients.config import StorageConfig
from npa.workflows.sim2real_rerun_serve import (
    DEFAULT_PORT,
    DEFAULT_RERUN_IMAGE,
    DEFAULT_S3_PREFIX,
    Sim2RealRerunServeError,
    apply_rerun_serve,
    build_rerun_serve_config,
    build_rerun_serve_manifest,
    deployment_name_for_run,
    destroy_rerun_serve,
    redact_rerun_serve_manifest,
    resolve_storage_bucket,
)


def _storage() -> StorageConfig:
    return StorageConfig(
        checkpoint_bucket="s3://demo-bucket/checkpoints/",
        endpoint_url="https://storage.example",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )


def test_deployment_name_sanitizes_run_id() -> None:
    assert deployment_name_for_run("rtxpro-staged-20260615T040034Z") == (
        "npa-sim2real-rerun-rtxpro-staged-20260615t040034z"
    )


def test_build_config_resolves_bucket_and_s3_uri(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real_rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-abc",
        s3_prefix="sim2real-b",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    assert config.s3_bucket == "demo-bucket"
    assert config.rrd_s3_uri == "s3://demo-bucket/sim2real-b/sim2real-staged-abc/reports/sim2real.rrd"
    assert config.s3_endpoint == "https://storage.example"


def test_manifest_contains_init_sync_and_rerun_serve(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real_rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-abc",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    manifest = build_rerun_serve_manifest(config)
    kinds = [item["kind"] for item in manifest["items"]]
    assert kinds == ["Secret", "Deployment", "Service"]

    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    init_container = deployment["spec"]["template"]["spec"]["initContainers"][0]
    rerun_container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert init_container["name"] == "sync-rrd"
    assert "aws s3 cp" in init_container["command"][-1]
    assert rerun_container["image"] == DEFAULT_RERUN_IMAGE
    assert f"--web-viewer-port {DEFAULT_PORT}" in rerun_container["command"][-1]
    assert rerun_container["command"][-1].endswith("--bind 0.0.0.0")

    secret = next(item for item in manifest["items"] if item["kind"] == "Secret")
    assert secret["metadata"]["name"] == f"{config.deployment_name}-s3"
    assert "S3_URI" in secret["data"]

    service = next(item for item in manifest["items"] if item["kind"] == "Service")
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["spec"]["ports"][0]["port"] == DEFAULT_PORT


def test_redact_manifest_hides_secret_values(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real_rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-abc",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    manifest = build_rerun_serve_manifest(config)
    redacted = redact_rerun_serve_manifest(manifest)
    secret = next(item for item in redacted["items"] if item["kind"] == "Secret")
    assert secret["data"] == {key: "<redacted>" for key in secret["data"]}


def test_apply_rerun_serve_uses_kubectl_runner(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real_rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-abc",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    calls: list[tuple[list[str], str | None]] = []

    def fake_kubectl(args, *, stdin=None, kubeconfig=""):
        calls.append((args, stdin))
        return ""

    service = {
        "status": {"loadBalancer": {"ingress": [{"ip": "203.0.113.10"}]}},
        "spec": {"ports": [{"port": DEFAULT_PORT}]},
    }
    result = apply_rerun_serve(
        config,
        kubeconfig="/tmp/kubeconfig",
        kubectl=fake_kubectl,
        get_service=lambda *_args, **_kwargs: service,
        wait_for_public_url=True,
        public_url_timeout_sec=1,
        now=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert calls[0][0] == ["apply", "-f", "-"]
    assert json.loads(calls[0][1] or "{}")["kind"] == "List"
    assert calls[1][0][:3] == ["rollout", "status", f"deployment/{config.deployment_name}"]
    assert result.status == "deployed"
    assert result.public_url == f"http://203.0.113.10:{DEFAULT_PORT}/"
    assert "port-forward" in result.port_forward_command


def test_destroy_rerun_serve_deletes_resources(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real_rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-abc",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    deleted: list[list[str]] = []

    def fake_kubectl(args, *, stdin=None, kubeconfig=""):
        deleted.append(args)
        return ""

    result = destroy_rerun_serve(config, kubeconfig="/tmp/kubeconfig", kubectl=fake_kubectl)
    assert result.status == "deleted"
    assert deleted[0][:2] == ["delete", "service"]
    assert deleted[1][:2] == ["delete", "deployment"]
    assert deleted[2][:2] == ["delete", "secret"]
    assert deleted[2][2] == config.secret_name


def test_missing_bucket_raises_clear_error() -> None:
    storage = StorageConfig(
        checkpoint_bucket="",
        endpoint_url="https://storage.example",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )
    with pytest.raises(Sim2RealRerunServeError, match="S3 bucket"):
        resolve_storage_bucket(storage)


def test_invalid_service_type_raises() -> None:
    with pytest.raises(Sim2RealRerunServeError, match="service-type"):
        build_rerun_serve_config(
            run_id="abc",
            s3_bucket="bucket",
            service_type="ingress",
            aws_access_key_id="ak",
            aws_secret_access_key="sk",
        )
