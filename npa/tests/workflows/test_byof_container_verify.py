from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
LIBERO_YAML_PATH = YAML_PATH.with_name(
    "byof-solution-smoke-libero-b200-gpu.yaml"
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


def _indirect_submit_args(module, monkeypatch, tmp_path):
    config_path = tmp_path / "skypilot.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    monkeypatch.setattr(module, "resolve_sky_bin", lambda *_a, **_k: "/opt/sky")
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_a, **_k: [{"name": "task", "envs": {}, "resources": {}}],
    )
    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", lambda *_a: None
    )
    monkeypatch.setattr(
        module,
        "_write_default_k8s_config",
        lambda *_a, **_k: str(config_path),
    )
    monkeypatch.setattr(
        module,
        "verify_solution_payload_service_accounts",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(module, "preflight_output_storage", lambda **_k: None)
    monkeypatch.setattr(module, "_ensure_infra_enabled", lambda **_k: None)
    guard = SimpleNamespace(
        isolated_config_dir=None,
        mark_launched=lambda **_k: None,
        teardown=lambda: module.CleanupResult(),
    )
    monkeypatch.setattr(module, "SignalTeardown", lambda **_k: guard)
    monkeypatch.setattr(module, "install_teardown_signal_handlers", lambda *_a: None)
    monkeypatch.setattr(module, "restore_signal_handlers", lambda *_a: None)
    return module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--run-id",
            "human-run-name",
            "--output-root",
            "s3://bucket/prefix",
            "--no-direct-launch",
            "--no-cleanup",
        ]
    )


def test_render_workflow_injects_solution_smoke_metadata(monkeypatch) -> None:
    from npa.execution_preflight import skypilot_output_destinations

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
    assert json.loads(envs["NPA_EXECUTION_OUTPUTS"]) == [
        {"uri": "s3://bucket/prefix/byof-demo/", "kind": "directory"}
    ]
    assert skypilot_output_destinations(docs) == {
        "s3://bucket/prefix/byof-demo/": "directory"
    }
    assert envs["NPA_S3_BUCKET"] == "bucket"
    assert envs["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert "AWS_ACCESS_KEY_ID" not in envs
    assert "AWS_SECRET_ACCESS_KEY" not in envs
    assert "AWS_SESSION_TOKEN" not in envs
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in envs
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


def test_render_workflow_materializes_libero_payload_account(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})

    docs = module.render_workflow(
        LIBERO_YAML_PATH,
        run_id="libero-demo",
        output_root="s3://bucket/prefix",
        solution_name="libero",
    )

    task = docs[1]
    assert "kubernetes" not in task["resources"]
    assert task["config"]["kubernetes"]["pod_config"]["spec"][
        "serviceAccountName"
    ] == "npa-byof-libero-payload"


def test_libero_profile_refuses_missing_or_mistyped_solution_name(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_a, **_k: [{"execution": "serial"}, {"name": "task"}],
    )
    for solution_name in ("", "not-libero", "LIBERO", " libero", "libero "):
        args = module._parse_args(
            [
                "--yaml",
                str(LIBERO_YAML_PATH),
                "--run-id",
                "libero-identity-refusal",
                "--output-root",
                "s3://bucket/prefix",
                "--solution-name",
                solution_name,
                "--render-only",
            ]
        )
        with pytest.raises(ValueError, match="requires --solution-name libero"):
            module._submit_and_wait(args)


@pytest.mark.parametrize("run_id", ["short", "libero.bad", "../libero-escape"])
def test_libero_profile_refuses_unsafe_run_id_before_render_output(
    monkeypatch, run_id
) -> None:
    module = _load_module()
    args = module._parse_args(
        [
            "--yaml",
            str(LIBERO_YAML_PATH),
            "--run-id",
            run_id,
            "--solution-name",
            "libero",
            "--render-only",
        ]
    )
    with pytest.raises(ValueError, match="SkyPilot run_id"):
        module._submit_and_wait(args)


def test_libero_payload_account_triggers_contract_independent_of_filename() -> None:
    module = _load_module()
    args = SimpleNamespace(solution_name="", yaml_path=Path("generic.yaml"))
    documents = [
        {
            "config": {
                "kubernetes": {
                    "pod_config": {
                        "spec": {"serviceAccountName": "npa-byof-libero-payload"}
                    }
                }
            }
        }
    ]

    assert module._is_libero_invocation(args, documents) is True


def test_libero_isolated_scheduler_state_must_be_owner_private(tmp_path) -> None:
    module = _load_module()
    state = tmp_path / "isolated-state"
    state.mkdir()
    state.chmod(0o755)
    with pytest.raises(ValueError, match="owner-private"):
        module._libero_isolated_state_root(state)

    state.chmod(0o700)
    assert module._libero_isolated_state_root(state) == state.resolve()
    with pytest.raises(ValueError, match="requires an isolated"):
        module._libero_isolated_state_root(None)


def test_libero_runtime_binding_refuses_disabled_cleanup() -> None:
    module = _load_module()
    args = SimpleNamespace(
        solution_name="libero", yaml_path=LIBERO_YAML_PATH,
        direct_launch=False, cleanup=False,
    )
    documents = [{"execution": "serial"}, {"name": "task"}]

    with pytest.raises(ValueError, match="requires verified managed cleanup"):
        module._bind_libero_runtime_contract(
            args, documents, global_config={}, infra="k8s/context"
        )


def test_libero_runtime_binding_requires_managed_exact_node_and_separate_context(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    kubeconfig = tmp_path / "payload-kubeconfig"
    kubeconfig.write_text("payload proof\n", encoding="utf-8")
    kubeconfig.chmod(0o600)
    execution_kubeconfig = tmp_path / "execution-kubeconfig"
    execution_kubeconfig.write_text("execution proof\n", encoding="utf-8")
    monkeypatch.setenv("KUBECONFIG", str(execution_kubeconfig))
    monkeypatch.setattr(module, "_libero_payload_kubeconfig", lambda: kubeconfig)
    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda path, **_kwargs: (
            "payload-proof-context",
            "isolated-namespace",
            "6" * 64,
        )
        if path == kubeconfig
        else ("execution-context", "", "6" * 64),
    )
    rbac = {
        "service_account_uid_sha256": "1" * 64,
        "role_uid_sha256": "2" * 64,
        "role_binding_uid_sha256": "3" * 64,
        "rbac_spec_sha256": "4" * 64,
        "namespace_sha256": "5" * 64,
    }
    monkeypatch.setattr(module, "_libero_rbac_evidence", lambda *_a: dict(rbac))
    args = SimpleNamespace(solution_name="libero", direct_launch=False)
    documents = [{"execution": "serial"}, {"envs": {}}]
    expected_evidence = {
        **rbac,
        "cluster_identity_sha256": "6" * 64,
        "allowed_node_sha256": module.hashlib.sha256(b"worker").hexdigest(),
    }
    for name, value in expected_evidence.items():
        monkeypatch.setenv(f"NPA_LIBERO_EXPECTED_{name.upper()}", value)

    evidence = module._bind_libero_runtime_contract(
        args,
        documents,
        global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
        infra="k8s/execution-context",
    )

    assert evidence == expected_evidence
    assert documents[1]["envs"] == {
        f"NPA_LIBERO_EXPECTED_{key.upper()}": value
        for key, value in evidence.items()
    }

    for direct, allowed in (
        (True, {"names": ["worker"]}),
        (False, None),
        (False, {"names": ["worker-a", "worker-b"]}),
    ):
        args.direct_launch = direct
        with pytest.raises(ValueError):
            module._bind_libero_runtime_contract(
                args,
                [{"execution": "serial"}, {"envs": {}}],
                global_config={"kubernetes": {"allowed_nodes": allowed}},
                infra="k8s/execution-context",
            )

    args.direct_launch = False
    missing_variable = "NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256"
    monkeypatch.delenv(missing_variable)
    with pytest.raises(ValueError, match="owner-receipted hash"):
        module._bind_libero_runtime_contract(
            args,
            [{"execution": "serial"}, {"envs": {}}],
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
        )
    monkeypatch.setenv(missing_variable, expected_evidence["role_binding_uid_sha256"])
    monkeypatch.setenv("NPA_LIBERO_EXPECTED_ROLE_UID_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="differs for role_uid_sha256"):
        module._bind_libero_runtime_contract(
            args,
            [{"execution": "serial"}, {"envs": {}}],
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
        )
    monkeypatch.setenv(
        "NPA_LIBERO_EXPECTED_ROLE_UID_SHA256", expected_evidence["role_uid_sha256"]
    )
    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda path, **_kwargs: (
            ("payload-proof-context", "isolated-namespace", "6" * 64)
            if path == kubeconfig
            else ("execution-context", "", "7" * 64)
        ),
    )
    with pytest.raises(ValueError, match="different clusters"):
        module._bind_libero_runtime_contract(
            args,
            [{"execution": "serial"}, {"envs": {}}],
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
        )

    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda _path, **_kwargs: (
            "execution-context",
            "isolated-namespace",
            "6" * 64,
        ),
    )
    with pytest.raises(ValueError, match="explicitly separated"):
        module._bind_libero_runtime_contract(
            args,
            [{"execution": "serial"}, {"envs": {}}],
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
        )


def test_libero_payload_kubeconfig_must_be_private_regular_file(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    kubeconfig = tmp_path / "payload-kubeconfig"
    kubeconfig.write_text("proof\n", encoding="utf-8")
    kubeconfig.chmod(0o644)
    monkeypatch.setenv("NPA_LIBERO_PAYLOAD_KUBECONFIG", str(kubeconfig))

    with pytest.raises(ValueError, match="mode-private regular file"):
        module._libero_payload_kubeconfig()

    kubeconfig.chmod(0o600)
    assert module._libero_payload_kubeconfig() == kubeconfig.resolve()


def test_libero_sky_config_uses_only_explicit_mode_private_owner_input(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    args = SimpleNamespace(config_path="")
    config = tmp_path / "skypilot-config"
    config.write_text("kubernetes: {}\n", encoding="utf-8")
    config.chmod(0o644)
    monkeypatch.setenv("NPA_LIBERO_SKYPILOT_CONFIG", str(config))

    with pytest.raises(ValueError, match="mode-private regular file"):
        module._libero_global_config_path(args)

    config.chmod(0o600)
    assert module._libero_global_config_path(args) == str(config.resolve())
    monkeypatch.delenv("NPA_LIBERO_SKYPILOT_CONFIG")
    with pytest.raises(ValueError, match="owner-supplied"):
        module._libero_global_config_path(args)


def test_libero_context_contract_binds_server_and_ca_without_exposing_them(
    monkeypatch,
) -> None:
    module = _load_module()
    config = {
        "current-context": "payload-context",
        "contexts": [
            {
                "name": "payload-context",
                "context": {"cluster": "cluster", "namespace": "isolated"},
            }
        ],
        "clusters": [
            {
                "name": "cluster",
                "cluster": {
                    "server": "https://cluster.example",
                    "certificate-authority-data": "base64-ca",
                },
            }
        ],
    }
    seen: dict[str, object] = {}

    def kubectl_json(arguments, *, purpose, kubeconfig):
        seen.update(arguments=arguments, purpose=purpose, kubeconfig=kubeconfig)
        return config

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    path = Path("/private/payload-kubeconfig")
    context, namespace, identity = module._libero_context_contract(
        path, require_namespace=True
    )

    assert (context, namespace) == ("payload-context", "isolated")
    assert identity == module._sha256_json(
        {
            "server": "https://cluster.example",
            "certificate_authority_data": "base64-ca",
        }
    )
    assert seen == {
        "arguments": ["config", "view", "--minify", "--flatten", "--raw"],
        "purpose": "selected-context",
        "kubeconfig": path,
    }

    config["clusters"][0]["cluster"]["insecure-skip-tls-verify"] = True
    with pytest.raises(RuntimeError, match="strict TLS identity"):
        module._libero_context_contract(path, require_namespace=True)


def test_libero_rbac_evidence_refuses_role_or_binding_drift(monkeypatch) -> None:
    module = _load_module()
    namespace = "isolated-namespace"
    objects = {
        "serviceaccount": {
            "metadata": {"uid": "account-uid"},
        },
        "role": {
            "metadata": {"uid": "role-uid"},
            "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
        },
        "rolebinding": {
            "metadata": {"uid": "binding-uid"},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "npa-byof-libero-payload",
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "npa-byof-libero-pod-reader",
            },
        },
    }
    monkeypatch.setattr(
        module,
        "_libero_resource",
        lambda _kubeconfig, _context, _namespace, kind, _name: objects[kind],
    )

    kubeconfig = Path("/private/payload-kubeconfig")
    evidence = module._libero_rbac_evidence(
        kubeconfig, "payload-context", namespace
    )
    assert evidence["service_account_uid_sha256"] == module.hashlib.sha256(
        b"account-uid"
    ).hexdigest()

    objects["role"]["rules"][0]["verbs"] = ["get", "list"]
    with pytest.raises(RuntimeError, match="broader"):
        module._libero_rbac_evidence(kubeconfig, "payload-context", namespace)
    objects["role"]["rules"][0]["verbs"] = ["get"]
    objects["rolebinding"]["subjects"][0]["name"] = "default"
    with pytest.raises(RuntimeError, match="differs"):
        module._libero_rbac_evidence(kubeconfig, "payload-context", namespace)


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


def test_wait_preserves_isolated_scheduler_identity(monkeypatch, tmp_path) -> None:
    module = _load_module()
    isolated = tmp_path / "isolated"
    config = tmp_path / "config.yaml"
    observed: dict[str, object] = {}

    def status(job_id, **kwargs):
        observed.update(job_id=job_id, **kwargs)
        return SimpleNamespace(status="SUCCEEDED")

    monkeypatch.setattr(module, "workflow_status", status)
    module._wait_for_terminal(
        "73",
        sky_bin="sky",
        isolated_config_dir=isolated,
        config_path=config,
        wait_timeout=0,
        poll_interval=1,
    )

    assert observed == {
        "job_id": "73",
        "sky_bin": "sky",
        "isolated_config_dir": isolated,
        "config_path": config,
    }


def test_wait_treats_verified_absence_as_terminal_failure(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="ABSENT"),
    )

    final, diagnostics = module._wait_for_terminal(
        "73", sky_bin="sky", wait_timeout=-1, poll_interval=1
    )

    assert final.status == "ABSENT"
    assert diagnostics["terminal"] is True
    assert diagnostics["polls"] == 1


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


def test_managed_cleanup_cancels_and_drains_exact_job_before_down(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    calls: list[object] = []
    statuses = iter(["RUNNING", "CANCELLING", "CANCELLED"])
    isolated = tmp_path / "isolated"
    config = tmp_path / "config.yaml"

    def status(job_id, **kwargs):
        calls.append(("status", job_id, kwargs))
        return SimpleNamespace(status=next(statuses))

    def cancel(**kwargs):
        calls.append(("cancel", kwargs))
        return {"cancel_returncode": 0}

    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )
    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(module, "cancel_workflow_job", cancel)

    def verified_absence(**_kwargs):
        calls.append("verify-absent")
        result = module.CleanupResult()
        result.verified = True
        result.remote_absence_verified = True
        return result

    monkeypatch.setattr(
        module,
        "_verify_managed_clusters_absent",
        verified_absence,
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=isolated,
        config_path=config,
        poll_interval=1,
    )

    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert calls[-2:] == ["down", "verify-absent"]
    cancel_call = next(call for call in calls if call[0] == "cancel")[1]
    assert cancel_call == {
        "sky_bin": "sky",
        "job_id": "73",
        "run_id": "human-run-name",
        "isolated_config_dir": isolated,
        "config_path": config,
        "timeout": 10,
        "poll_seconds": 1.0,
        "also_down_cluster": False,
    }
    assert [
        call[1] for call in calls if isinstance(call, tuple) and call[0] == "status"
    ] == [
        "73",
        "73",
        "73",
    ]


def test_managed_cleanup_preserves_clusters_when_exact_cancel_fails(
    monkeypatch,
) -> None:
    module = _load_module()
    down = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="RUNNING"),
    )
    monkeypatch.setattr(
        module, "cancel_workflow_job", lambda **_k: {"cancel_returncode": 1}
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: down.append(True) or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert result.errors == ["exact managed-job cancellation failed"]
    assert down == []


def test_managed_cleanup_preserves_clusters_on_ambiguous_controller_status(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="FAILED_CONTROLLER"),
    )
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert result.errors == [
        "managed job did not reach a verified terminal or absent state; "
        "preserving its clusters"
    ]
    assert calls == ["cancel"]


def test_managed_cleanup_cancels_after_status_exception_then_proves_drain(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    statuses = iter(
        [subprocess.TimeoutExpired(["sky", "jobs", "queue"], 1), "CANCELLED"]
    )

    def status(*_args, **_kwargs):
        value = next(statuses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(status=value)

    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    monkeypatch.setattr(
        module,
        "_verify_managed_clusters_absent",
        lambda **_k: calls.append("verify") or module.CleanupResult(),
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is True
    assert calls == ["cancel", "down", "verify"]


def test_managed_cleanup_preserves_clusters_after_persistent_status_exception(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("status unavailable")),
    )
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert "could not be verified" in result.errors[0]
    assert calls == ["cancel"]


@pytest.mark.parametrize(
    "status_failure",
    [None, SimpleNamespace(), TypeError("malformed status")],
)
def test_managed_cleanup_cancels_on_malformed_or_unlisted_status_failure(
    monkeypatch, status_failure
) -> None:
    module = _load_module()
    calls: list[str] = []

    def status(*_args, **_kwargs):
        if isinstance(status_failure, Exception):
            raise status_failure
        return status_failure

    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert "could not be verified" in result.errors[0]
    assert calls == ["cancel"]


@pytest.mark.parametrize(
    ("stdout", "expected_error"),
    [
        ("not-json", "was not exact JSON"),
        ('{"unexpected": []}', "invalid schema"),
        ('{"clusters": [], "error": "denied"}', "invalid schema"),
        ('[{"status": "UP"}]', "invalid row"),
        ('[{"name": " human-run-name-worker "}]', "invalid row"),
        (
            '[{"name": "unrelated", "cluster": "human-run-name-worker"}]',
            "ambiguous name row",
        ),
        (
            '[{"name": "unrelated", "error": "permission denied"}]',
            "contradictory error metadata",
        ),
        ('[{"name": "human-run-name-worker"}]', "still contains"),
    ],
)
def test_post_teardown_inventory_refuses_ambiguous_or_present_state(
    monkeypatch, stdout, expected_error
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            ["sky", "status"], 0, stdout=stdout, stderr=""
        ),
    )

    result = module._verify_managed_clusters_absent(
        run_id="human-run-name",
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        timeout=10,
    )

    assert result.ok is False
    assert expected_error in result.errors[0]
    assert result.remote_absence_verified is False


def test_post_teardown_inventory_proves_exact_run_absence(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    isolated = tmp_path / "isolated"
    config = tmp_path / "config.yaml"
    observed = {}

    def run(cmd, **kwargs):
        observed.update(cmd=cmd, kwargs=kwargs)
        return subprocess.CompletedProcess(
            cmd, 0, stdout='[{"name": "unrelated-cluster"}]', stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module._verify_managed_clusters_absent(
        run_id="human-run-name",
        sky_bin="sky",
        isolated_config_dir=isolated,
        config_path=config,
        timeout=10,
    )

    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert observed["cmd"] == [
        "sky",
        "status",
        "--config",
        str(config),
        "--refresh",
        "--output",
        "json",
    ]
    assert observed["kwargs"]["env"]["HOME"] == str(isolated / "home")


def test_managed_cleanup_preserves_resources_without_scheduler_id() -> None:
    module = _load_module()
    down = []
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: down.append(True) or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert "preserving" in result.errors[0]
    assert down == []


def test_submit_waits_on_scheduler_id_not_human_run_name(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    isolated = tmp_path / "isolated-state"
    isolated.mkdir()
    isolated.chmod(0o700)
    args.isolated_config_dir = str(isolated)
    observed: dict[str, object] = {"resolve_calls": 0}

    def resolve_isolated(value):
        observed["resolve_calls"] = int(observed["resolve_calls"]) + 1
        assert value == str(isolated)
        return isolated.resolve()

    def guard_factory(**kwargs):
        observed["guard_isolated"] = kwargs["isolated_config_dir"]
        return SimpleNamespace(
            run_id="human-run-name",
            timeout=10,
            isolated_config_dir=kwargs["isolated_config_dir"],
            mark_launched=lambda **_k: None,
            teardown=lambda: module.CleanupResult(),
        )

    def submit(_path, run_id, **kwargs):
        observed["submitted_run_id"] = run_id
        observed["submitted_isolated"] = kwargs["isolated_config_dir"]
        return SimpleNamespace(job_id="73", log_paths={})

    def wait(scheduler_job_id, **kwargs):
        observed["waited_job_id"] = scheduler_job_id
        observed["waited_isolated"] = kwargs["isolated_config_dir"]
        return SimpleNamespace(status="SUCCEEDED"), {"terminal": True}

    monkeypatch.setattr(module, "resolve_isolated_config_dir", resolve_isolated)
    monkeypatch.setattr(module, "SignalTeardown", guard_factory)
    monkeypatch.setattr(module, "submit_workflow", submit)
    monkeypatch.setattr(module, "_wait_for_terminal", wait)

    assert module._submit_and_wait(args) == 0
    assert observed == {
        "submitted_run_id": "human-run-name",
        "submitted_isolated": isolated.resolve(),
        "waited_job_id": "73",
        "waited_isolated": isolated.resolve(),
        "guard_isolated": isolated.resolve(),
        "resolve_calls": 1,
    }


def test_submit_refuses_empty_scheduler_id(monkeypatch, tmp_path) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: pytest.fail("waiter must not run without a scheduler ID"),
    )

    with pytest.raises(RuntimeError, match="no scheduler job ID"):
        module._submit_and_wait(args)


def test_submit_returns_failure_when_exact_cleanup_is_not_verified(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="SUCCEEDED"),
    )

    def failed_teardown():
        result = module.CleanupResult()
        result.errors.append("cluster absence was not verified")
        return result

    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        isolated_config_dir=None,
        config_path=None,
        mark_launched=lambda **_k: None,
        teardown=failed_teardown,
    )
    monkeypatch.setattr(module, "SignalTeardown", lambda **_k: guard)

    assert module._submit_and_wait(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["cleanup"] == {
        "errors": ["cluster absence was not verified"],
        "ok": False,
        "remote_absence_verified": False,
        "resources_removed": [],
        "verified": False,
    }


def test_submit_requires_and_emits_verified_remote_cleanup(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: (SimpleNamespace(status="SUCCEEDED"), {"terminal": True}),
    )
    cleanup = module.CleanupResult()
    cleanup.verified = True
    cleanup.remote_absence_verified = True
    monkeypatch.setattr(
        module, "_cancel_then_teardown_managed_job", lambda *_a, **_k: cleanup
    )

    assert module._submit_and_wait(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["cleanup"] == {
        "errors": [],
        "ok": True,
        "remote_absence_verified": True,
        "resources_removed": [],
        "verified": True,
    }


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
    environments: list[Path | None] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"default": {"Kubernetes": ["compute"]}}', stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "sky_environment",
        lambda isolated: environments.append(isolated) or {},
    )
    isolated = Path("/owner/isolated-sky-state")
    module._ensure_infra_enabled(
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
        isolated_config_dir=isolated,
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
    assert environments == [isolated, isolated]


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
    assert "serviceAccountName: skypilot-service-account" in text
    assert "kubectl create secret docker-registry" not in text
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
    isolated = tmp_path / "isolated-state"
    isolated.mkdir()
    monkeypatch.setattr(
        module, "resolve_isolated_config_dir", lambda _value: isolated
    )
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
    config_path = tmp_path / "skypilot.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module, "_write_default_k8s_config", lambda *_a, **_k: str(config_path)
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
    environment_roots: list[Path | None] = []
    monkeypatch.setattr(
        module,
        "sky_environment",
        lambda root: environment_roots.append(root) or os.environ.copy(),
    )

    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--direct-launch",
            "--isolated-config-dir",
            str(isolated),
            "--output-root",
            "s3://bucket/prefix",
        ]
    )
    assert module._submit_and_wait(args) == 0
    assert os.environ.get("KUBECONFIG") == original
    assert ["/opt/sky", "api", "stop"] in seen_cmds
    assert environment_roots == [isolated]
