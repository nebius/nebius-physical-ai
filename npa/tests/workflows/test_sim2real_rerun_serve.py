from __future__ import annotations

import json

import pytest

from npa.clients.config import StorageConfig
from npa.workflows.rerun_serve import (
    DEFAULT_GRPC_PORT,
    DEFAULT_NGINX_IMAGE,
    DEFAULT_PORT,
    DEFAULT_RERUN_IMAGE,
    RERUN_INTERNAL_WEB_PORT,
    RERUN_STATIC_CACHE_CONTROL,
    Sim2RealRerunServeError,
    apply_rerun_serve,
    build_rerun_nginx_config,
    build_rerun_serve_config,
    build_rerun_serve_manifest,
    deployment_name_for_cluster,
    deployment_name_for_run,
    destroy_rerun_serve,
    default_rerun_image,
    fetch_rrd_sync_token,
    in_cluster_kubernetes,
    local_viewer_url,
    maybe_auto_rerun_serve,
    public_viewer_url,
    redact_rerun_serve_manifest,
    resolve_storage_bucket,
    rrd_s3_uri_from_report_uri,
    should_auto_rerun_serve,
    validate_run_id,
    validate_staged_run_id,
)


def _storage() -> StorageConfig:
    return StorageConfig(
        checkpoint_bucket="s3://demo-bucket/checkpoints/",
        endpoint_url="https://storage.example",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )


def test_deployment_name_for_cluster_default_viewer() -> None:
    assert deployment_name_for_cluster() == "npa-rerun-viewer"
    assert deployment_name_for_run("rtxpro-staged-20260615T040034Z") == "npa-rerun-viewer"


def test_deployment_name_for_cluster_slugifies_context() -> None:
    assert deployment_name_for_cluster("npa-rtxpro-mk8s") == "npa-rerun-npa-rtxpro-mk8s"


def test_same_cluster_deployment_name_for_different_runs(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    first = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        cluster_context="npa-rtxpro-mk8s",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    second = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t235414z",
        cluster_context="npa-rtxpro-mk8s",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    assert first.deployment_name == second.deployment_name == "npa-rerun-npa-rtxpro-mk8s"
    assert first.rrd_s3_uri != second.rrd_s3_uri


def test_validate_run_id_rejects_placeholder() -> None:
    with pytest.raises(Sim2RealRerunServeError, match="placeholder"):
        validate_run_id("sim2real-staged-YYYYMMDDTHHMMSSz")


def test_validate_run_id_accepts_canonical_staged() -> None:
    assert validate_run_id("sim2real-staged-20260615t180818z") == (
        "sim2real-staged-20260615t180818z"
    )


def test_validate_run_id_accepts_custom_run_ids() -> None:
    # Non-staged ids (e2e, BYO-robot, custom) are real runs with artifacts on S3
    # and must be servable without renaming to the staged-loop shape.
    for rid in (
        "sim2real-e2e-main-20260627t195851z",
        "kinova-onboard-b5-20260627t190600z",
        "my_custom.run-01",
    ):
        assert validate_run_id(rid) == rid


def test_validate_run_id_rejects_path_traversal_and_separators() -> None:
    for bad in ("../etc/passwd", "a/b", "run id", "-leading-dash", ""):
        with pytest.raises(Sim2RealRerunServeError):
            validate_run_id(bad)


def test_validate_staged_run_id_is_backcompat_alias() -> None:
    assert validate_staged_run_id is validate_run_id


def test_build_config_accepts_custom_run_id(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-e2e-main-20260627t195851z",
        s3_prefix="sim2real-b",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    assert config.rrd_s3_uri == (
        "s3://demo-bucket/sim2real-b/sim2real-e2e-main-20260627t195851z/reports/sim2real.rrd"
    )


def test_default_rerun_image_prefers_generic_env_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_RERUN_IMAGE", "legacy/image:1")
    monkeypatch.setenv("NPA_RERUN_VIEWER_IMAGE", "generic/image:2")
    assert default_rerun_image() == "generic/image:2"


def test_manifest_sets_progress_deadline(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    manifest = build_rerun_serve_manifest(config)
    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    assert deployment["spec"]["progressDeadlineSeconds"] == 900


def test_build_config_resolves_bucket_and_s3_uri(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        s3_prefix="sim2real-b",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    assert config.s3_bucket == "demo-bucket"
    assert config.rrd_s3_uri == (
        "s3://demo-bucket/sim2real-b/sim2real-staged-20260615t180818z/reports/sim2real.rrd"
    )
    assert config.s3_endpoint == "https://storage.example"


def test_build_config_accepts_report_uri(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        report_uri="s3://demo-bucket/sim2real-b/sim2real-staged-20260615t180818z/reports/sim2real-report.json",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    assert config.rrd_s3_uri == "s3://demo-bucket/sim2real-b/sim2real-staged-20260615t180818z/reports/sim2real.rrd"


def test_rrd_s3_uri_from_report_uri() -> None:
    report = "s3://demo-bucket/sim2real-b/run-1/reports/sim2real-report.json"
    assert rrd_s3_uri_from_report_uri(report) == (
        "s3://demo-bucket/sim2real-b/run-1/reports/sim2real.rrd"
    )


def test_manifest_contains_init_sync_and_rerun_serve(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    manifest = build_rerun_serve_manifest(config)
    kinds = [item["kind"] for item in manifest["items"]]
    assert kinds == ["Secret", "ConfigMap", "Deployment", "Service"]

    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    init_container = deployment["spec"]["template"]["spec"]["initContainers"][0]
    containers = deployment["spec"]["template"]["spec"]["containers"]
    nginx_container = next(c for c in containers if c["name"] == "nginx")
    rerun_container = next(c for c in containers if c["name"] == "rerun")
    assert init_container["name"] == "sync-rrd"
    assert "aws s3 cp" in init_container["command"][-1]
    assert nginx_container["image"] == DEFAULT_NGINX_IMAGE
    assert nginx_container["ports"][0]["containerPort"] == DEFAULT_PORT
    assert rerun_container["image"] == DEFAULT_RERUN_IMAGE
    assert "pip install" in rerun_container["command"][-1]
    assert "rerun-sdk==0.32.0" in rerun_container["command"][-1]
    assert "--serve-web" in rerun_container["command"][-1]
    assert f"--web-viewer-port {RERUN_INTERNAL_WEB_PORT}" in rerun_container["command"][-1]
    assert f"--port {DEFAULT_GRPC_PORT}" in rerun_container["command"][-1]
    assert "--cors-allow-origin" in rerun_container["command"][-1]
    assert rerun_container["command"][-1].endswith("--cors-allow-origin 'http://*:*' ")

    configmap = next(item for item in manifest["items"] if item["kind"] == "ConfigMap")
    assert RERUN_STATIC_CACHE_CONTROL in configmap["data"]["nginx.conf"]
    assert f"127.0.0.1:{RERUN_INTERNAL_WEB_PORT}" in configmap["data"]["nginx.conf"]

    secret = next(item for item in manifest["items"] if item["kind"] == "Secret")
    assert secret["metadata"]["name"] == f"{config.deployment_name}-s3"
    assert "S3_URI" in secret["data"]

    service = next(item for item in manifest["items"] if item["kind"] == "Service")
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["spec"]["ports"][0]["port"] == DEFAULT_PORT
    assert service["spec"]["ports"][1]["port"] == DEFAULT_GRPC_PORT


def test_public_viewer_url_points_at_external_grpc_proxy() -> None:
    url = public_viewer_url("203.0.113.10", http_port=9090, grpc_port=9876)
    assert url.startswith("http://203.0.113.10:9090/?url=")
    assert "203.0.113.10" in url
    assert "9876" in url


def test_local_viewer_url_uses_loopback_grpc_origin() -> None:
    url = local_viewer_url(http_port=9090, grpc_port=9876)
    assert url == public_viewer_url("127.0.0.1", http_port=9090, grpc_port=9876)
    assert "127.0.0.1" in url
    assert "9876" in url
    assert "rerun%2Bhttp" in url or "rerun+http" in url


def test_manifest_uses_direct_rerun_for_prebuilt_image(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
        rerun_image="cr.eu-north1.nebius.cloud/demo/npa-rerun-viewer:0.31.4",
    )
    manifest = build_rerun_serve_manifest(config)
    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    rerun_container = next(
        c for c in deployment["spec"]["template"]["spec"]["containers"] if c["name"] == "rerun"
    )
    assert "pip install" not in rerun_container["command"][-1]
    assert rerun_container["command"][-1].startswith("rerun /data/sim2real.rrd --serve-web")


def test_build_rerun_nginx_config_sets_static_cache_headers() -> None:
    config_text = build_rerun_nginx_config()
    assert RERUN_STATIC_CACHE_CONTROL in config_text
    assert f"listen {DEFAULT_PORT}" in config_text
    assert f"127.0.0.1:{RERUN_INTERNAL_WEB_PORT}" in config_text
    assert "(wasm|js|ico|svg)" in config_text


def test_build_rerun_serve_manifest_includes_rrd_sync_annotation(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    manifest = build_rerun_serve_manifest(config, rrd_sync_token="abc123etag")
    deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
    annotations = deployment["spec"]["template"]["metadata"]["annotations"]
    assert annotations["npa.nebius.com/rrd-sync-token"] == "abc123etag"


def test_fetch_rrd_sync_token_uses_head_object_etag() -> None:
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        s3_bucket="demo-bucket",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    token = fetch_rrd_sync_token(
        config,
        head_object=lambda **_kwargs: {"ETag": '"etag-from-s3"'},
    )
    assert token == "etag-from-s3"


def test_redact_manifest_hides_secret_values(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    manifest = build_rerun_serve_manifest(config)
    redacted = redact_rerun_serve_manifest(manifest)
    secret = next(item for item in redacted["items"] if item["kind"] == "Secret")
    assert secret["data"] == {key: "<redacted>" for key in secret["data"]}


def test_apply_rerun_serve_uses_kubectl_runner(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    mocker.patch("npa.workflows.rerun_serve.verify_rrd_exists_on_s3")
    mocker.patch(
        "npa.workflows.rerun_serve.fetch_rrd_sync_token",
        return_value="etag-test",
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
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
    assert result.public_url == public_viewer_url(
        "203.0.113.10", http_port=DEFAULT_PORT, grpc_port=DEFAULT_GRPC_PORT
    )
    assert result.local_url == local_viewer_url(http_port=DEFAULT_PORT, grpc_port=DEFAULT_GRPC_PORT)
    assert "port-forward" in result.port_forward_command
    assert f"{DEFAULT_PORT}:{DEFAULT_PORT}" in result.port_forward_command
    assert f"{DEFAULT_GRPC_PORT}:{DEFAULT_GRPC_PORT}" in result.port_forward_command


def test_destroy_rerun_serve_deletes_resources(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    deleted: list[list[str]] = []
    messages: list[str] = []

    def fake_kubectl(args, *, stdin=None, kubeconfig="", timeout_sec=None):
        deleted.append(args)
        return ""

    result = destroy_rerun_serve(
        config,
        kubeconfig="/tmp/kubeconfig",
        kubectl=fake_kubectl,
        progress=messages.append,
    )
    assert result.status == "deleted"
    assert deleted[0][:2] == ["delete", "service"]
    assert "--wait=false" in deleted[0]
    assert deleted[1][:2] == ["delete", "deployment"]
    assert deleted[2][:2] == ["delete", "configmap"]
    assert deleted[2][2] == config.nginx_configmap_name
    assert deleted[3][:2] == ["delete", "secret"]
    assert deleted[3][2] == config.secret_name
    assert any("Deleting service/" in message for message in messages)
    assert any("Deleted shared cluster Rerun viewer" in message for message in messages)


def test_destroy_rerun_serve_wait_uses_kubectl_wait_true(mocker) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_project_storage",
        return_value=_storage(),
    )
    config = build_rerun_serve_config(
        run_id="sim2real-staged-20260615t180818z",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    deleted: list[list[str]] = []

    def fake_kubectl(args, *, stdin=None, kubeconfig="", timeout_sec=None):
        deleted.append(args)
        return ""

    destroy_rerun_serve(
        config,
        kubeconfig="/tmp/kubeconfig",
        kubectl=fake_kubectl,
        wait=True,
        progress=lambda _message: None,
    )
    assert all("--wait=true" in call for call in deleted)


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
            run_id="sim2real-staged-20260615t180818z",
            s3_bucket="bucket",
            service_type="ingress",
            aws_access_key_id="ak",
            aws_secret_access_key="sk",
        )


@pytest.mark.parametrize(
    ("rerun_enabled", "upload_status", "viz_status", "expected"),
    [
        (True, "uploaded", "reference", True),
        (False, "uploaded", "reference", False),
        (True, "skipped", "reference", False),
        (True, "uploaded", "disabled", False),
    ],
)
def test_should_auto_rerun_serve_gate(
    rerun_enabled: bool,
    upload_status: str,
    viz_status: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_SIM2REAL_RERUN_SERVE", raising=False)
    assert (
        should_auto_rerun_serve(
            rerun_enabled=rerun_enabled,
            s3_bucket="demo-bucket",
            upload_status=upload_status,
            viz_status=viz_status,
        )
        is expected
    )


def test_should_auto_rerun_serve_respects_disable_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_RERUN_SERVE", "0")
    assert not should_auto_rerun_serve(
        rerun_enabled=True,
        s3_bucket="demo-bucket",
        upload_status="uploaded",
        viz_status="reference",
    )


def test_maybe_auto_rerun_serve_skips_when_upload_missing() -> None:
    result = maybe_auto_rerun_serve(
        run_id="sim2real-staged-20260615t180818z",
        s3_bucket="demo-bucket",
        rerun_enabled=True,
        upload_info={"status": "skipped"},
        viz_info={"status": "reference"},
    )
    assert result["status"] == "skipped"


def test_maybe_auto_rerun_serve_uses_in_cluster_kubectl_without_kubeconfig_file(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_rerun_serve_credentials",
        return_value=("ak", "sk"),
    )
    mocker.patch(
        "npa.workflows.rerun_serve.build_rerun_serve_config",
        return_value=build_rerun_serve_config(
            run_id="sim2real-staged-20260615t180818z",
            s3_bucket="demo-bucket",
            aws_access_key_id="ak",
            aws_secret_access_key="sk",
        ),
    )
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_kubeconfig_path",
        return_value="",
    )
    apply = mocker.patch(
        "npa.workflows.rerun_serve.apply_rerun_serve",
        return_value=type(
            "Result",
            (),
            {
                "to_dict": lambda self: {
                    "status": "deployed",
                    "public_url": (
                        "http://203.0.113.10:9090/?url=rerun%2Bhttp%3A%2F%2F203.0.113.10"
                        "%3A9876%2Fproxy"
                    ),
                    "local_url": (
                        "http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1"
                        "%3A9876%2Fproxy"
                    ),
                    "port_forward_command": "kubectl port-forward -n default deployment/x 9090:9090 9876:9876",
                    "run_id": "sim2real-staged-20260615t180818z",
                    "deployment_name": "npa-rerun-npa-rtxpro-mk8s",
                }
            },
        )(),
    )

    result = maybe_auto_rerun_serve(
        run_id="sim2real-staged-20260615t180818z",
        s3_bucket="demo-bucket",
        rerun_enabled=True,
        upload_info={"status": "uploaded"},
        viz_info={"status": "reference"},
    )
    assert result["status"] == "deployed"
    apply.assert_called_once()
    assert apply.call_args.kwargs["kubeconfig"] == ""


def test_in_cluster_kubernetes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert not in_cluster_kubernetes()
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    assert in_cluster_kubernetes()


def test_maybe_auto_rerun_serve_deploys_and_prints_public_url(mocker, capsys) -> None:
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_rerun_serve_credentials",
        return_value=("ak", "sk"),
    )
    mocker.patch(
        "npa.workflows.rerun_serve.build_rerun_serve_config",
        return_value=build_rerun_serve_config(
            run_id="sim2real-staged-20260615t180818z",
            s3_bucket="demo-bucket",
            aws_access_key_id="ak",
            aws_secret_access_key="sk",
        ),
    )
    mocker.patch(
        "npa.workflows.rerun_serve.resolve_kubeconfig_path",
        return_value="/tmp/kubeconfig",
    )
    mocker.patch(
        "npa.workflows.rerun_serve.apply_rerun_serve",
        return_value=type(
            "Result",
            (),
            {
                "to_dict": lambda self: {
                    "status": "deployed",
                    "public_url": "http://203.0.113.10:9090/",
                    "local_url": "http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy",
                    "port_forward_command": "kubectl port-forward -n default deployment/x 9090:9090 9876:9876",
                    "run_id": "sim2real-staged-20260615t180818z",
                    "deployment_name": "npa-rerun-viewer",
                }
            },
        )(),
    )

    result = maybe_auto_rerun_serve(
        run_id="sim2real-staged-20260615t180818z",
        s3_bucket="demo-bucket",
        rerun_enabled=True,
        upload_info={"status": "uploaded"},
        viz_info={"status": "reference"},
        k8s_kubeconfig="/tmp/kubeconfig",
    )
    assert result["status"] == "deployed"
    out = capsys.readouterr().out
    assert "public_url: http://203.0.113.10:9090/" in out
    assert "local_url: http://127.0.0.1:9090/" in out
    assert "port_forward:" in out
