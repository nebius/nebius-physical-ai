"""Coverage tests for the detection-training CLI helpers and commands."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from npa.cli.workbench import detection_training as dt
from npa.cli.workbench.detection_training import app as dt_app


runner = CliRunner()


# ── tiny helpers ───────────────────────────────────────────────────────────


def test_fail_raises_typer_exit() -> None:
    import typer

    with pytest.raises(typer.Exit):
        dt.fail("boom")


def test_emit_text_and_json() -> None:
    import typer

    captured: list[str] = []
    monkey_echo = typer.echo
    try:
        typer.echo = lambda msg: captured.append(msg)
        dt.emit({"a": 1, "b": 2}, output=dt.OutputFormat.text)
        dt.emit({"a": 1}, output=dt.OutputFormat.text, text="custom")
        dt.emit({"a": 1, "b": 2}, output=dt.OutputFormat.json)
    finally:
        typer.echo = monkey_echo
    assert captured[0] == "a: 1\nb: 2"
    assert captured[1] == "custom"
    assert json.loads(captured[2]) == {"a": 1, "b": 2}


# ── resolve_endpoint / request_json error paths ────────────────────────────


def test_resolve_endpoint_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_DETECTION_TRAINING_ENDPOINT", "https://api.example/")
    assert dt.resolve_endpoint("") == "https://api.example"


def test_resolve_endpoint_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    monkeypatch.delenv("NPA_DETECTION_TRAINING_ENDPOINT", raising=False)
    with pytest.raises(typer.Exit):
        dt.resolve_endpoint("")


def test_resolve_endpoint_rejects_non_http() -> None:
    import typer

    with pytest.raises(typer.Exit):
        dt.resolve_endpoint("ftp://x")


def test_request_json_http_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    response = httpx.Response(500, text="oops", request=httpx.Request("GET", "http://x/"))

    def fake_request(*a, **kw):
        return response

    monkeypatch.setattr(dt.httpx, "request", fake_request)
    with pytest.raises(typer.Exit):
        dt.request_json("GET", "http://x", "/p", token_env="X")


def test_request_json_http_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    def fake_request(*a, **kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(dt.httpx, "request", fake_request)
    with pytest.raises(typer.Exit):
        dt.request_json("GET", "http://x", "/p", token_env="X")


def test_request_json_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    response = httpx.Response(200, text="not-json", request=httpx.Request("GET", "http://x/"))
    monkeypatch.setattr(dt.httpx, "request", lambda *a, **kw: response)
    with pytest.raises(typer.Exit):
        dt.request_json("GET", "http://x", "/p", token_env="X")


def test_request_json_non_object_body(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    response = httpx.Response(200, json=[1, 2], request=httpx.Request("GET", "http://x/"))
    monkeypatch.setattr(dt.httpx, "request", lambda *a, **kw: response)
    with pytest.raises(typer.Exit):
        dt.request_json("GET", "http://x", "/p", token_env="X")


def test_request_json_sets_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_request(method, url, headers=None, json=None, params=None, timeout=None):
        captured.update(method=method, url=url, headers=headers)
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setenv("TKN", "secret")
    monkeypatch.setattr(dt.httpx, "request", fake_request)

    result = dt.request_json("GET", "http://x", "/p", token_env="TKN")
    assert result == {"ok": True}
    assert captured["headers"]["Authorization"] == "Bearer secret"


# ── _image_registry / docker auth helpers ─────────────────────────────────


def test_image_registry_returns_host_for_full_image() -> None:
    assert dt._image_registry("registry.example.com/proj/npa:1") == "registry.example.com"
    assert dt._image_registry("localhost/foo") == "localhost"
    assert dt._image_registry("npa-foo:latest") == ""
    assert dt._image_registry("library/npa") == ""


def test_docker_auth_config_uses_auths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"auths": {"registry.example.com": {"auth": "Zm9vOmJhcg=="}}})
    )
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))

    out = dt._docker_auth_config("registry.example.com")
    assert out["auths"]["registry.example.com"]["auth"] == "Zm9vOmJhcg=="


def test_docker_auth_config_missing_file_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_auth_config_bad_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{ not json")
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))

    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_auth_config_uses_credential_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"credsStore": "helper"}))
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(dt.shutil, "which", lambda exe: f"/usr/bin/{exe}")

    def fake_run(cmd, input=None, text=False, capture_output=False, check=False):
        return SimpleNamespace(
            stdout=json.dumps({"Username": "alice", "Secret": "topsecret"}),
            stderr="",
        )

    monkeypatch.setattr(dt.subprocess, "run", fake_run)

    out = dt._docker_auth_config("registry.example.com")
    entry = out["auths"]["registry.example.com"]
    assert entry["username"] == "alice"
    assert entry["password"] == "topsecret"
    assert entry["auth"] == base64.b64encode(b"alice:topsecret").decode()


def test_docker_auth_config_helper_missing_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"credsStore": "helper"}))
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(dt.shutil, "which", lambda exe: None)
    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_auth_config_helper_called_process_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"credsStore": "helper"}))
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(dt.shutil, "which", lambda exe: f"/bin/{exe}")

    def fake_run(*a, **kw):
        raise subprocess.CalledProcessError(1, "x", output="", stderr="denied")

    monkeypatch.setattr(dt.subprocess, "run", fake_run)
    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_auth_config_helper_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"credsStore": "helper"}))
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(dt.shutil, "which", lambda exe: f"/bin/{exe}")
    monkeypatch.setattr(
        dt.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="garbage", stderr=""),
    )
    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_auth_config_helper_incomplete_creds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"credsStore": "helper"}))
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(dt.shutil, "which", lambda exe: f"/bin/{exe}")
    monkeypatch.setattr(
        dt.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps({"Username": "alice"}), stderr=""
        ),
    )
    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_auth_config_no_auth_or_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"auths": {}}))
    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(typer.Exit):
        dt._docker_auth_config("registry.example.com")


def test_docker_credential_helper_prefers_credhelpers() -> None:
    cfg = {"credHelpers": {"https://x": "h1"}, "credsStore": "store"}
    assert dt._docker_credential_helper(cfg, "x") == "h1"
    assert dt._docker_credential_helper({"credsStore": "store"}, "y") == "store"
    assert dt._docker_credential_helper({}, "y") == ""


# ── _kubectl / _resolve_kubeconfig ────────────────────────────────────────


def test_kubectl_dry_run_echoes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(dt.typer, "echo", lambda msg: captured.append(msg))
    out = dt._kubectl(["apply", "-f", "-"], dry_run=True, kubeconfig="/tmp/kc")
    assert out == ""
    assert "kubectl --kubeconfig /tmp/kc apply -f -" in captured[0]


def test_kubectl_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    def fake_run(*a, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(dt.subprocess, "run", fake_run)
    with pytest.raises(typer.Exit):
        dt._kubectl(["get", "po"])


def test_kubectl_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    def fake_run(*a, **kw):
        raise subprocess.CalledProcessError(1, "x", output="", stderr="forbidden")

    monkeypatch.setattr(dt.subprocess, "run", fake_run)
    with pytest.raises(typer.Exit):
        dt._kubectl(["get", "po"])


def test_kubectl_success_echoes_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(dt.typer, "echo", lambda msg: captured.append(msg))
    monkeypatch.setattr(
        dt.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(stdout="hello\n", stderr=""),
    )
    out = dt._kubectl(["get", "po"])
    assert out == "hello\n"
    assert captured == ["hello"]


def test_kubectl_capture_does_not_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(dt.typer, "echo", lambda msg: captured.append(msg))
    monkeypatch.setattr(
        dt.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(stdout="{}", stderr=""),
    )
    out = dt._kubectl(["get", "po"], capture=True)
    assert out == "{}"
    assert captured == []


def test_resolve_kubeconfig_passthrough_and_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert dt._resolve_kubeconfig(cluster_name="x", kubeconfig="  /tmp/kc  ") == "/tmp/kc"
    assert dt._resolve_kubeconfig(cluster_name="", kubeconfig="") == ""

    monkeypatch.setattr(dt.Path, "home", classmethod(lambda cls: tmp_path))
    kc = tmp_path / ".npa" / "clusters" / "alpha" / "kubeconfig"
    kc.parent.mkdir(parents=True)
    kc.write_text("apiVersion: v1")
    assert dt._resolve_kubeconfig(cluster_name="alpha", kubeconfig="") == str(kc)
    assert dt._resolve_kubeconfig(cluster_name="missing", kubeconfig="") == ""


def test_redact_manifest_redacts_secrets() -> None:
    manifest = {
        "items": [
            {"kind": "Secret", "data": {"a": "abc", "b": "def"}},
            {"kind": "Deployment", "data": {"keep": "value"}},
        ]
    }
    redacted = dt._redact_manifest(manifest)
    assert redacted["items"][0]["data"] == {"a": "<redacted>", "b": "<redacted>"}
    assert redacted["items"][1]["data"] == {"keep": "value"}


# ── CLI invocations ──────────────────────────────────────────────────────


def test_deploy_validation_port() -> None:
    result = runner.invoke(
        dt_app,
        ["deploy", "--port", "100", "--output-path", "s3://b/", "--gpu-type", "h100"],
    )
    assert result.exit_code == 1
    assert "--port must be between" in result.output


def test_deploy_validation_auth_mode() -> None:
    result = runner.invoke(
        dt_app,
        [
            "deploy",
            "--auth-mode",
            "weird",
            "--output-path",
            "s3://b/",
        ],
    )
    assert result.exit_code == 1
    assert "--auth-mode" in result.output


def test_deploy_requires_output_path() -> None:
    result = runner.invoke(dt_app, ["deploy", "--gpu-type", "h100"])
    assert result.exit_code == 1
    assert "--output-path is required" in result.output


def test_deploy_invalid_gpu_type() -> None:
    result = runner.invoke(
        dt_app,
        [
            "deploy",
            "--gpu-type",
            "unknown",
            "--output-path",
            "s3://b/",
        ],
    )
    assert result.exit_code == 1
    assert "--gpu-type must be h100 or l40s" in result.output


def test_deploy_destroy_invokes_kubectl_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_kubectl(args, *, stdin=None, dry_run=False, capture=False, kubeconfig=""):
        calls.append(args)
        return ""

    monkeypatch.setattr(dt, "_kubectl", fake_kubectl)
    monkeypatch.setattr(dt, "_resolve_kubeconfig", lambda **k: "/tmp/kc")

    result = runner.invoke(
        dt_app,
        ["deploy", "--destroy", "--output", "json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["status"] == "deleted"
    assert len(calls) == 3
    assert calls[0][0] == "delete"


def test_deploy_dry_run_emits_redacted_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dt, "_resolve_kubeconfig", lambda **k: "/tmp/kc")
    # _kubernetes_manifest builds a real plan; no kubectl should be invoked
    monkeypatch.setattr(
        dt, "_kubectl", lambda *a, **kw: pytest.fail("kubectl should not run")
    )

    result = runner.invoke(
        dt_app,
        [
            "deploy",
            "--dry-run",
            "--output-path",
            "s3://b/",
            "--gpu-type",
            "h100",
        ],
    )
    assert result.exit_code == 0, result.output
    # Output contains a manifest JSON dict
    assert '"items"' in result.output or '"kind"' in result.output


def test_train_service_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_request(method, endpoint, path, *, token_env, payload=None, params=None, timeout=30.0):
        captured.update(method=method, path=path, payload=payload)
        return {"run_id": "r-1", "status": "running"}

    monkeypatch.setattr(dt, "request_json", fake_request)
    result = runner.invoke(
        dt_app,
        [
            "train",
            "--view",
            "v",
            "--output-uri",
            "s3://b/",
            "--service",
            "--endpoint",
            "http://api.x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/train"
    assert captured["payload"]["view"] == "v"


def test_eval_service_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dt,
        "request_json",
        lambda *a, **kw: {"mAP": 0.9, "eval_run_id": "e1"},
    )
    result = runner.invoke(
        dt_app,
        [
            "eval",
            "--checkpoint-uri",
            "s3://b/c.pt",
            "--eval-view",
            "v",
            "--output-uri",
            "s3://b/eval/",
            "--service",
            "--endpoint",
            "http://api.x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mAP" in result.output or "eval_run_id" in result.output


def test_status_service_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_request(method, endpoint, path, *, token_env, payload=None, params=None, timeout=30.0):
        captured.update(method=method, path=path, params=params)
        return {"status": "running", "epochs_completed": 3}

    monkeypatch.setattr(dt, "request_json", fake_request)
    result = runner.invoke(
        dt_app,
        [
            "status",
            "--run-id",
            "r-1",
            "--service",
            "--endpoint",
            "http://api.x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/status"
    assert captured["params"] == {"run_id": "r-1"}


def test_system_info_service_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dt,
        "request_json",
        lambda *a, **kw: {"gpu": "h100", "cuda": "12.4"},
    )
    result = runner.invoke(
        dt_app,
        ["system-info", "--service", "--endpoint", "http://api.x"],
    )
    assert result.exit_code == 0, result.output


def test_list_service_mode_with_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dt,
        "request_json",
        lambda *a, **kw: {"runs": [{"run_id": "r1"}, {"run_id": "r2"}]},
    )
    result = runner.invoke(
        dt_app,
        ["list", "--service", "--endpoint", "http://api.x"],
    )
    assert result.exit_code == 0, result.output
    assert "r1" in result.output


def test_list_service_mode_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dt, "request_json", lambda *a, **kw: {"runs": []})
    result = runner.invoke(
        dt_app,
        ["list", "--service", "--endpoint", "http://api.x"],
    )
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_list_local_mode_calls_kubectl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dt, "_resolve_kubeconfig", lambda **k: "/tmp/kc")
    monkeypatch.setattr(
        dt,
        "_kubectl",
        lambda *a, **kw: json.dumps(
            {"items": [{"metadata": {"name": "dt-svc"}}, {"metadata": {"name": "dt-deploy"}}]}
        ),
    )
    result = runner.invoke(dt_app, ["list", "--output", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output[result.output.find("{"):])
    assert data["count"] == 2
    assert "dt-svc" in data["resources"]


# ── _service_env ──────────────────────────────────────────────────────────


def test_service_env_uses_creds_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TKN", "tk")
    monkeypatch.setenv("AWS_REGION", "eu-north1")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setattr(
        dt,
        "load_credentials",
        lambda: SimpleNamespace(
            s3_access_key_id="ak",
            s3_secret_access_key="sk",
            s3_endpoint="https://s3.example",
        ),
    )

    env = dt._service_env(
        input_path="lance://x",
        output_path="s3://b/",
        auth_mode="token",
        token_env="TKN",
        port=8080,
    )
    assert env["DETECTION_TRAINING_AUTH_MODE"] == "token"
    assert env["DETECTION_TRAINING_TOKEN"] == "tk"
    assert env["AWS_REGION"] == "eu-north1"
    assert env["AWS_ACCESS_KEY_ID"] == "ak"
    assert env["AWS_ENDPOINT_URL"] == "https://s3.example"
    assert env["NEBIUS_S3_ENDPOINT"] == "https://s3.example"


def test_service_env_token_required(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    monkeypatch.delenv("TKN", raising=False)
    monkeypatch.setattr(
        dt,
        "load_credentials",
        lambda: SimpleNamespace(s3_access_key_id="", s3_secret_access_key="", s3_endpoint=""),
    )
    with pytest.raises(typer.Exit):
        dt._service_env(
            input_path="x",
            output_path="y",
            auth_mode="token",
            token_env="TKN",
            port=8080,
        )
