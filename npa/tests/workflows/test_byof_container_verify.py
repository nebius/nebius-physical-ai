from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_byof_container_verify.py"
YAML_PATH = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-container-smoke-rtxpro.yaml"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_byof_container_verify", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sky_environment_honors_task_isolation(monkeypatch, tmp_path) -> None:
    module = _load_module()
    isolated = tmp_path / "sky-state"
    calls = []
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(isolated))
    monkeypatch.setattr(
        module,
        "sky_environment",
        lambda value: calls.append(value) or {"HOME": str(value / "home")},
    )

    assert module._sky_environment() == {"HOME": str(isolated / "home")}
    assert calls == [isolated]


def test_render_workflow_injects_solution_smoke_metadata(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://bucket/prefix",
        image="registry.example/npa-byof:demo",
        smoke_command="python -c 'print(42)'",
        solution_name="demo-solution",
        capability_name="demo-capability",
        smoke_artifact_name="demo_artifact.json",
    )

    task = docs[1]
    envs = task["envs"]
    assert envs["BYOF_SMOKE_COMMAND"] == "python -c 'print(42)'"
    assert envs["BYOF_SOLUTION_NAME"] == "demo-solution"
    assert envs["BYOF_CAPABILITY_NAME"] == "demo-capability"
    assert envs["BYOF_SMOKE_ARTIFACT_NAME"] == "demo_artifact.json"
    assert envs["BYOF_IMAGE"] == "registry.example/npa-byof:demo"
    assert envs["S3_OUTPUT_PREFIX"] == "s3://bucket/prefix/byof-demo/"
    assert envs["NPA_S3_BUCKET"] == "bucket"
    assert envs["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert "AWS_ACCESS_KEY_ID" not in envs
    assert "AWS_SECRET_ACCESS_KEY" not in envs
    assert "AWS_SESSION_TOKEN" not in envs
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in envs
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


def test_runtime_secret_channel_has_no_invented_wan_consent(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "probe-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "probe-secret")

    assert module.resolve_secret_envs(None, solution_name="wan2.2") == [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    # Wan's own NVIDIA gate was removed upstream, so it has no entry and nothing
    # is invented for it; HF_TOKEN is unset here and drops out too.
    assert module.resolve_secret_envs(["HF_TOKEN"], solution_name="wan2.2") == []


def test_openpi_runtime_acceptance_uses_secret_channel(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "probe-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "probe-secret")

    assert module.resolve_secret_envs(None, solution_name="openpi") == [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
    ]
    assert module.resolve_secret_envs(["HF_TOKEN"], solution_name="openpi") == [
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS"
    ]


def test_one_solutions_operator_answers_do_not_widen_anothers(monkeypatch) -> None:
    """Vendor answers are per-image, and a shared tuple made them global.

    With one tuple for every BYOF image, adding LTX's variables also forwarded
    them — and HF_TOKEN — into wan2-2 and open-dreamer runs whenever they were
    set in the operator's shell. Nothing broke visibly, which is why it needs a
    test rather than a review.
    """

    module = _load_module()
    for name in (
        "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "HF_TOKEN",
    ):
        monkeypatch.setenv(name, "set")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    wan = module.resolve_secret_envs(None, solution_name="wan2.2")
    ltx = module.resolve_secret_envs(None, solution_name="ltx2.5")
    other = module.resolve_secret_envs(None, solution_name="open-dreamer")

    assert not [name for name in wan if name.startswith("NPA_LTX_")]
    assert "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS" not in ltx
    assert "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS" in ltx
    # The entitlement the LTX container needs for both of its fetches.
    assert "HF_TOKEN" in ltx
    # A solution with no vendor answers of its own forwards none, including the
    # token: it has no gate that reads one.
    assert other == []


def test_output_storage_preflight_writes_reads_and_deletes(monkeypatch) -> None:
    module = _load_module()
    calls: list[tuple[str, str, str]] = []

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            calls.append(("list", Bucket, Prefix))
            assert MaxKeys == 1
            return {"Contents": []}

        def put_object(self, *, Bucket, Key, **_kwargs):
            assert _kwargs["IfNoneMatch"] == "*"
            calls.append(("put", Bucket, Key))

        def head_object(self, *, Bucket, Key):
            calls.append(("head", Bucket, Key))
            return {"ContentLength": 24}

        def delete_object(self, *, Bucket, Key):
            calls.append(("delete", Bucket, Key))

    monkeypatch.setenv("NPA_E2E_PROJECT", "demo-project")
    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda project, *, allow_host_creds, endpoint_url: (
            FakeS3()
            if project == "demo-project"
            and allow_host_creds
            and endpoint_url == "https://storage.override"
            else pytest.fail("unexpected S3 credential scope")
        ),
    )
    monkeypatch.setenv("NPA_BYOF_S3_ENDPOINT", "https://storage.override")

    module.preflight_output_storage(
        output_root="s3://bucket/prefix", run_id="byof-demo"
    )

    assert calls == [
        ("list", "bucket", "prefix/byof-demo/"),
        ("put", "bucket", "prefix/byof-demo/.npa-write-preflight"),
        ("head", "bucket", "prefix/byof-demo/.npa-write-preflight"),
        ("delete", "bucket", "prefix/byof-demo/.npa-write-preflight"),
    ]


def test_render_storage_env_honors_explicit_regional_endpoint(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_E2E_PROJECT", "demo-project")
    monkeypatch.setenv("NPA_BYOF_S3_ENDPOINT", "https://storage.correct-region")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.stale-region")

    def storage_env(project, *, allow_host_creds, endpoint_url):
        assert project == "demo-project"
        assert allow_host_creds is True
        assert endpoint_url == "https://storage.correct-region"
        return {"AWS_ENDPOINT_URL": endpoint_url}

    monkeypatch.setattr(module, "storage_env_for_project", storage_env)

    assert module._resolved_storage_env() == {
        "AWS_ENDPOINT_URL": "https://storage.correct-region"
    }

    docs = module.render_workflow(
        YAML_PATH,
        run_id="regional-endpoint",
        output_root="s3://project-bucket/byof",
    )
    assert docs[1]["envs"]["AWS_ENDPOINT_URL"] == "https://storage.correct-region"
    assert docs[1]["envs"]["NEBIUS_S3_ENDPOINT"] == "https://storage.correct-region"


def test_output_storage_preflight_fails_before_launch(monkeypatch) -> None:
    module = _load_module()

    class DeniedS3:
        def list_objects_v2(self, **_kwargs):
            return {"Contents": []}

        def put_object(self, **_kwargs):
            raise PermissionError("Access denied")

    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda *_args, **_kwargs: DeniedS3(),
    )

    with pytest.raises(RuntimeError, match="output storage preflight failed"):
        module.preflight_output_storage(
            output_root="s3://bucket/prefix", run_id="byof-demo"
        )


def test_output_storage_preflight_rejects_reused_run_prefix(monkeypatch) -> None:
    module = _load_module()

    class ExistingS3:
        def list_objects_v2(self, **_kwargs):
            return {"Contents": [{"Key": "prefix/byof-demo/result.json"}]}

    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda *_args, **_kwargs: ExistingS3(),
    )

    with pytest.raises(
        RuntimeError, match="refusing to reuse a non-empty BYOF run prefix"
    ):
        module.preflight_output_storage(
            output_root="s3://bucket/prefix", run_id="byof-demo"
        )


def test_wait_timeout_zero_checks_status_once(monkeypatch) -> None:
    module = _load_module()
    calls: list[str] = []
    status = type("Status", (), {"status": "RUNNING"})()
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_args, **_kwargs: calls.append("status") or status,
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda *_args: (_ for _ in ()).throw(AssertionError("slept")),
    )

    final, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=0, poll_interval=1
    )
    assert final.status == "RUNNING"
    assert calls == ["status"]
    assert diagnostics == {
        "mode": "immediate",
        "polls": 1,
        "statuses": ["RUNNING"],
        "terminal": False,
        "deadline_exhausted": False,
        "stuck_state": "RUNNING",
        "hint": "workflow is not terminal; inspect SkyPilot controller/job and pod events",
    }


def test_positive_wait_is_bounded_and_reports_stuck_state(monkeypatch) -> None:
    module = _load_module()
    clock = {"now": 100.0}
    status = type("Status", (), {"status": "PENDING"})()
    monkeypatch.setattr(module, "workflow_status", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    _, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=2, poll_interval=1
    )
    assert diagnostics["mode"] == "bounded"
    assert diagnostics["polls"] == 3
    assert diagnostics["deadline_exhausted"] is True
    assert diagnostics["stuck_state"] == "PENDING"


def test_negative_one_waits_until_terminal(monkeypatch) -> None:
    module = _load_module()
    statuses = iter(["PENDING", "RUNNING", "SUCCEEDED"])
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_args, **_kwargs: type("Status", (), {"status": next(statuses)})(),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    final, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=-1, poll_interval=1
    )
    assert final.status == "SUCCEEDED"
    assert diagnostics["mode"] == "indefinite"
    assert diagnostics["statuses"] == ["PENDING", "RUNNING", "SUCCEEDED"]
    assert diagnostics["terminal"] is True


def test_wait_timeout_less_than_negative_one_is_rejected() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="must be -1"):
        module._wait_for_terminal(
            "run", sky_bin="sky", wait_timeout=-2, poll_interval=1
        )


def test_render_workflow_normalizes_docker_image_for_summary(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        image="docker:registry.example/npa-byof:demo",
    )
    task = docs[1]
    assert task["envs"]["BYOF_IMAGE"] == "registry.example/npa-byof:demo"
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


def test_render_workflow_rejects_unresolved_endpoint_placeholder(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "${AWS_ENDPOINT_URL}")
    monkeypatch.setattr(
        module,
        "_resolved_storage_env",
        lambda: {"AWS_ENDPOINT_URL": "https://storage.from-project"},
    )
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://bucket/prefix",
    )
    assert docs[1]["envs"]["AWS_ENDPOINT_URL"] == "https://storage.from-project"


def test_normalize_output_root_strips_double_s3_prefix(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    assert (
        module._normalize_s3_bucket("s3://lerobot-demo/checkpoints/") == "lerobot-demo"
    )
    assert (
        module._normalize_output_root("s3://s3://lerobot-demo/checkpoints/")
        == "s3://lerobot-demo/checkpoints"
    )
    assert (
        module._normalize_output_root("s3://lerobot-demo/checkpoints/")
        == "s3://lerobot-demo/checkpoints"
    )
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://s3://lerobot-demo/checkpoints/",
    )
    assert (
        docs[1]["envs"]["S3_OUTPUT_PREFIX"]
        == "s3://lerobot-demo/checkpoints/byof-demo/"
    )
    assert docs[1]["envs"]["NPA_S3_BUCKET"] == "lerobot-demo"


def test_default_infra_uses_resolved_kubernetes_context(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_K8S_CONTEXT", "customer-mk8s")
    monkeypatch.delenv("NPA_BYOF_INFRA", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_INFRA", raising=False)
    assert module._default_infra() == "k8s/customer-mk8s"


def test_ensure_infra_enabled_runs_sky_check_for_kubernetes(monkeypatch) -> None:
    module = _load_module()
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"default": {"Kubernetes": ["compute"]}}', stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
    )

    assert seen == [
        ["/opt/sky", "api", "stop"],
        [
            "/opt/sky",
            "check",
            "kubernetes",
            "-o",
            "json",
            "--config",
            "/tmp/skypilot.yaml",
        ],
    ]


def test_ensure_infra_enabled_skips_non_kubernetes(monkeypatch) -> None:
    module = _load_module()
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="aws/us-east-1")
    assert called is False


def test_ensure_infra_enabled_rejects_zero_exit_with_disabled_provider(
    monkeypatch,
) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="{}", stderr=""
        ),
    )

    with pytest.raises(module.SkyPilotConfigError, match="did not enable compute"):
        module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


def test_ensure_infra_enabled_parses_json_after_api_startup_prose(
    monkeypatch,
) -> None:
    module = _load_module()
    calls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="",
            stderr=(
                "Failed to connect to local API server; starting one.\n"
                '{"default": {"Kubernetes": ["compute"]}}\n'
            ),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


def test_ensure_infra_enabled_examines_all_kubernetes_entries(monkeypatch) -> None:
    module = _load_module()
    calls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "Kubernetes": {"enabled": False, "capabilities": []},
                    "profiles": [
                        {
                            "selected": {
                                "Kubernetes": {
                                    "enabled": True,
                                    "capabilities": ["compute"],
                                }
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        ("", ""),
        ("not json", "still not json"),
        ('{"Kubernetes": []}', ""),
        ('{"Kubernetes": {"enabled": false, "capabilities": ["compute"]}}', ""),
        ('{"status": "error", "Kubernetes": ["compute"]}', ""),
        ('{"error": "authentication failed", "Kubernetes": ["compute"]}', ""),
        (
            '{"result": {"status": "error", "Kubernetes": ["compute"]}}',
            "",
        ),
        (
            '{"error": "stale provider state"}\n'
            '{"result": {"Kubernetes": ["compute"]}}',
            "",
        ),
    ],
)
def test_ensure_infra_enabled_rejects_empty_disabled_malformed_and_error_output(
    monkeypatch, stdout, stderr
) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(module.SkyPilotConfigError, match="did not enable compute"):
        module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


def test_ensure_infra_enabled_accepts_enabled_json_from_stdout_or_stderr(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    outputs = iter(
        [
            ('[{"Kubernetes": ["compute"]}]', ""),
            ("", 'startup prose\n{"default": {"Kubernetes": ["compute"]}}'),
        ]
    )

    def fake_run(cmd, **_kwargs):
        stdout, stderr = next(outputs)
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="kubernetes")
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="kubernetes")


def test_direct_launch_uses_sky_launch_with_down(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    rendered_yaml = tmp_path / "workflow.yaml"
    rendered_yaml.write_text("name: demo\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    rc = module._direct_launch(
        rendered_yaml=rendered_yaml,
        run_id="byof-demo",
        outputs={"summary": "s3://bucket/summary.json"},
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
        cleanup=True,
        secret_envs=["HF_TOKEN"],
    )

    assert rc == 0
    assert seen["cmd"] == [
        "/opt/sky",
        "launch",
        "--yes",
        "--cluster",
        "byof-demo",
        "--name",
        "byof-demo",
        "--down",
        "--infra",
        "k8s/customer-mk8s",
        "--config",
        "/tmp/skypilot.yaml",
        "--secret",
        "HF_TOKEN",
        str(rendered_yaml),
    ]
    output = capsys.readouterr().out
    assert '"mode": "direct-launch"' in output


def test_write_default_k8s_config_adds_pull_secrets(tmp_path) -> None:
    module = _load_module()
    config_path = module._write_default_k8s_config(tmp_path, "k8s/customer-mk8s")

    assert config_path
    text = Path(config_path).read_text(encoding="utf-8")
    assert "imagePullSecrets" in text
    assert "agent-sa" in text
    assert "npa-nebius-registry" not in text
    assert "allowed_contexts" in text
    assert "customer-mk8s" in text


def test_normalize_kubeconfig_current_context(monkeypatch, tmp_path) -> None:
    module = _load_module()
    source = tmp_path / "source-kubeconfig"
    source.write_text(
        """
apiVersion: v1
kind: Config
current-context: old-context
contexts:
- name: target-context
  context: {}
clusters: []
users: []
""".strip(),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("KUBECONFIG", str(source))
    monkeypatch.setenv("KUBECONTEXT", "target-context")

    module._normalize_kubeconfig_current_context(out)

    updated = Path(os.environ["KUBECONFIG"]).read_text(encoding="utf-8")
    assert "current-context: target-context" in updated
    assert str(out) in os.environ["KUBECONFIG"]


def test_submit_and_wait_restores_kubeconfig_after_direct_launch(
    monkeypatch, tmp_path
) -> None:
    """Temp kubeconfig under TemporaryDirectory must not leak into later sky jobs."""
    module = _load_module()
    original = str(tmp_path / "original-kubeconfig")
    Path(original).write_text("kind: Config\n", encoding="utf-8")
    monkeypatch.setenv("KUBECONFIG", original)
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "1")
    monkeypatch.setattr(module, "resolve_sky_bin", lambda *_a, **_k: "/opt/sky")
    monkeypatch.setattr(module, "_default_run_id", lambda: "byof-restore")
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_a, **_k: [
            {"name": "meta"},
            {"name": "task", "envs": {}, "resources": {}},
        ],
    )
    monkeypatch.setattr(module, "_write_yaml_documents", lambda *_a, **_k: None)

    def _leak_kubeconfig(tmp: Path) -> None:
        os.environ["KUBECONFIG"] = str(tmp / "leaked")

    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", _leak_kubeconfig
    )
    monkeypatch.setattr(module, "_default_infra", lambda: "k8s/demo")
    monkeypatch.setattr(
        module, "_write_default_k8s_config", lambda *_a, **_k: "/tmp/skypilot.yaml"
    )
    monkeypatch.setattr(module, "_ensure_infra_enabled", lambda **_k: None)
    monkeypatch.setattr(module, "preflight_output_storage", lambda **_k: None)
    monkeypatch.setattr(module, "_direct_launch", lambda **_k: 0)
    seen_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        seen_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "sky_environment", lambda *_a, **_k: os.environ.copy())

    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--direct-launch",
            "--output-root",
            "s3://bucket/prefix",
        ]
    )
    assert module._submit_and_wait(args) == 0
    assert os.environ.get("KUBECONFIG") == original
    assert ["/opt/sky", "api", "stop"] in seen_cmds
