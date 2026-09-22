# npa: publication-enforcement=libero
"""Customer handoff stays explicit and the fixed job remains recoverable."""
from __future__ import annotations

import argparse
import base64
import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.skypilot.workflow import SkyPilotSubmitError
from npa.workflows.byof import libero_customer as customer

ROOT = Path(__file__).resolve().parents[3]


def driver():
    spec = importlib.util.spec_from_file_location("libero_customer_test", ROOT / "npa/scripts/run_libero_customer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", ["setup", "token", "credential", "sidecar", "privileged", "mount", "mode", "root_uid", "missing_uid"])
def test_customer_profile_rejects_workload_expansion(mutation):
    docs = list(yaml.safe_load_all(customer.PROFILE.read_text()))
    customer.validate_profile(docs)
    task = docs[1]
    pod = task["resources"]["kubernetes"]["pod_config"]["spec"]
    assert pod["securityContext"]["runAsUser"] == 1000
    if mutation == "setup":
        task["setup"] += "\necho altered\n"
    elif mutation == "token":
        pod["automountServiceAccountToken"] = True
    elif mutation == "credential":
        task["envs"]["AWS_SECRET_ACCESS_KEY"] = "synthetic-test"
    elif mutation == "sidecar":
        pod["containers"].append({"name": "other"})
    elif mutation == "privileged":
        pod["containers"][0]["securityContext"]["capabilities"]["add"].append("SYS_ADMIN")
    elif mutation == "mount":
        pod["volumes"].append({"name": "host", "hostPath": {"path": "/"}})
    elif mutation == "root_uid":
        pod["securityContext"]["runAsUser"] = 0
    elif mutation == "missing_uid":
        pod["securityContext"].pop("runAsUser")
    else:
        task["envs"][customer.MODE_ENV] = "unreviewed"
    with pytest.raises(ValueError):
        customer.validate_profile(docs)


def test_customer_profile_survives_standard_kubernetes_config_lift():
    docs = list(yaml.safe_load_all(customer.PROFILE.read_text()))
    signed_profile = driver().libero_executable_profile_bytes(docs)
    task = docs[1]
    task["config"] = {"kubernetes": task["resources"].pop("kubernetes")}
    task["resources"]["region"] = "unit-context"
    customer.validate_profile(docs)
    assert driver().libero_executable_profile_bytes(docs) == signed_profile


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAIL", "FAILED_PRECHECKS", "FAILED_CONTROLLER", "FAILED_NEW_KIND", "CANCELED", "STOPPED"])
def test_phase_wait_rejects_every_terminal_status(status):
    with pytest.raises(RuntimeError, match="terminated before"):
        driver()._require_running(status, "materialization")


def test_private_handoff_rejects_symlinks_and_shared_files(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"synthetic")
    path.chmod(0o600)
    assert customer.private_bytes(path) == b"synthetic"
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        customer.private_bytes(link)
    path.chmod(0o644)
    with pytest.raises(ValueError):
        customer.private_bytes(path)


def _submit_fixture(monkeypatch, tmp_path):
    from kubernetes import client, config

    module = driver()
    packet = tmp_path / "packet"
    packet.mkdir()
    request = {"run_id": "libero-synthetic-customer-test", "workflow_profile_sha256": module._sha(b"profile")}
    authorization = {"customer_identity_sha256": "a" * 64, "expires_at": "2099-01-01T00:00:00Z", "signature": {"public_key_sha256": "b" * 64}}
    target = {name: "synthetic-" + name for name in ("project", "context", "controller_context", "namespace", "namespace_uid", "allowed_node", "kubeconfig", "config_path", "deny_policy_uid", "fetch_policy_uid")}
    target["isolated_config_dir"] = str(tmp_path / "sky")
    target_path = tmp_path / "target.json"
    monkeypatch.setattr(module, "_json", lambda path: target if path == target_path else request)
    monkeypatch.setattr(module, "private_bytes", lambda path: (
        json.dumps(authorization).encode() if path.name == "customer-authorization.json" else
        base64.b64encode(b"k" * 32) if path.name == "customer-public-key.b64" else b"profile"
    ))
    monkeypatch.setattr(module, "image_manifest", lambda _env: ({}, {"candidate_image": "synthetic@sha256:" + "d" * 64}))
    monkeypatch.setattr(module, "_profile", lambda *_a: [{}, {"name": "synthetic", "envs": {}}])
    monkeypatch.setattr(module, "libero_executable_profile_bytes", lambda _docs: b"profile")
    monkeypatch.setattr(module, "validate_authorization", lambda *_a, **_k: (authorization, "e" * 64))
    monkeypatch.setattr(module, "controller_context", lambda *_a, **_k: target["controller_context"])
    namespace = SimpleNamespace(metadata=SimpleNamespace(uid=target["namespace_uid"], labels={"npa-libero-run": request["run_id"]}))
    account = SimpleNamespace(automount_service_account_token=False, secrets=[], metadata=SimpleNamespace(uid="account"))
    calls = []
    core = SimpleNamespace(
        read_namespace=lambda _: namespace,
        read_namespaced_service_account=lambda *_a: account,
        list_namespaced_pod=lambda _: SimpleNamespace(items=[]),
        create_namespaced_config_map=lambda *_a: SimpleNamespace(metadata=SimpleNamespace(name="key", uid="owned-key-uid")),
        delete_namespaced_config_map=lambda *_a, **_k: calls.append("delete-key"),
    )
    policies = []
    for name, field, egress in (("libero-deny-egress", "deny_policy_uid", []), ("libero-fetch-egress", "fetch_policy_uid", [{}])):
        policies.append(SimpleNamespace(metadata=SimpleNamespace(name=name, uid=target[field]), spec={"podSelector": {}, "policyTypes": ["Egress"], "egress": egress}))
    monkeypatch.setattr(config, "new_client_from_config", lambda **_k: SimpleNamespace(sanitize_for_serialization=lambda spec: spec))
    monkeypatch.setattr(client, "CoreV1Api", lambda _: core)
    monkeypatch.setattr(client, "NetworkingV1Api", lambda _: SimpleNamespace(list_namespaced_network_policy=lambda _: SimpleNamespace(items=policies)))
    monkeypatch.setattr(module, "stop_isolated_api", lambda _: calls.append("stop-api"))
    monkeypatch.setattr(module, "_module", lambda *_a: SimpleNamespace(LiberoAccessState=SimpleNamespace, LiberoRuntimeBinding=SimpleNamespace))
    args = argparse.Namespace(packet=packet, target=target_path, output=tmp_path / "result")
    return module, args, calls


def test_key_and_api_cleaned_when_preparation_fails_after_creation(monkeypatch, tmp_path):
    module, args, calls = _submit_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_module", lambda *_a: (_ for _ in ()).throw(ValueError("preparation")))
    with pytest.raises(ValueError, match="preparation"):
        module.submit(args)
    assert calls == ["stop-api", "delete-key"]
    assert json.loads((args.output / "cleanup.json").read_text())["ok"]


def test_customer_observer_uses_pinned_single_task_dag_name(monkeypatch, tmp_path):
    module, args, _ = _submit_fixture(monkeypatch, tmp_path)
    observed = {}

    def binding(**values):
        observed.update(values)
        return SimpleNamespace(**values)

    monkeypatch.setattr(module, "_module", lambda *_a: SimpleNamespace(
        LiberoAccessState=SimpleNamespace, LiberoRuntimeBinding=binding,
    ))

    def submit(_path, run_id, **_kwargs):
        # SkyPilot 0.12.2 assigns a single task the DAG name supplied by --name.
        assert observed["task_name"] == run_id == "libero-synthetic-customer-test"
        assert observed["task_name"] != "synthetic"
        raise SkyPilotSubmitError("synthetic prelaunch stop", launch_attempted=False)

    monkeypatch.setattr(module, "submit_workflow", submit)
    with pytest.raises(SkyPilotSubmitError, match="synthetic prelaunch stop"):
        module.submit(args)


@pytest.mark.parametrize("attempted", [False, None, True])
def test_submit_failure_preserves_ambiguous_launch_recovery(monkeypatch, tmp_path, attempted):
    module, args, calls = _submit_fixture(monkeypatch, tmp_path)
    def fail(*_a, **_k):
        _k["transaction_recorder"]({"libero_unverified_candidate_job_id": "42"})
        raise SkyPilotSubmitError("synthetic launch failure", launch_attempted=attempted)
    monkeypatch.setattr(module, "submit_workflow", fail)
    with pytest.raises(SkyPilotSubmitError):
        module.submit(args)
    receipt = json.loads((args.output / "cleanup.json").read_text())
    assert receipt["ok"] is (attempted is False)
    assert calls == (["stop-api", "delete-key"] if attempted is False else [])
    journal = args.output / "launch-transactions.jsonl"
    assert json.loads(journal.read_text()) == {"libero_unverified_candidate_job_id": "42"}
    assert journal.stat().st_mode & 0o777 == 0o600


def test_missing_customer_evidence_has_no_kubernetes_effect(monkeypatch, tmp_path):
    module = driver()
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "request.json").write_text("{}")
    (packet / "request.json").chmod(0o600)
    with pytest.raises(FileNotFoundError):
        module.submit(argparse.Namespace(packet=packet, target=tmp_path / "absent", output=tmp_path / "result"))
    assert not (tmp_path / "result").exists()


def test_authorize_uses_nonseekable_terminal_and_declines_without_writing(monkeypatch, tmp_path, capsys):
    import pty

    module = driver()
    request = {
        "status": "needs_customer_acceptance", "runtime_delivery": customer.MODE,
        "candidate_image": "synthetic-image", "runtime_manifest_sha256": "a" * 64,
        "workflow_profile_sha256": module._sha(b"profile"), "terms": [],
        "upstream_source_revision": "b" * 40, "run_id": "libero-synthetic-customer-test",
    }
    manifest = {"runtime_manifest_sha256": "a" * 64, "customer_acceptance": {"terms": []}}
    qualification = {"candidate_image": "synthetic-image", "upstream_source_revision": "b" * 40}
    monkeypatch.setattr(module, "_json", lambda _path: request)
    monkeypatch.setattr(module, "private_bytes", lambda _path: b"profile")
    monkeypatch.setattr(module, "image_manifest", lambda _env: (manifest, qualification))
    monkeypatch.setattr(module, "_profile", lambda *_args: [])
    monkeypatch.setattr(module, "libero_executable_profile_bytes", lambda _docs: b"profile")
    master, slave = pty.openpty()
    try:
        terminal_path = os.ttyname(slave)
        def open_terminal(path, *args, **kwargs):
            assert path == "/dev/tty"
            return builtins.open(
                terminal_path, *args, **kwargs,
                opener=lambda path, flags: os.open(path, flags | os.O_NOCTTY),
            )
        monkeypatch.setattr(module, "open", open_terminal, raising=False)
        os.write(master, b"\nDECLINE\n")
        module.authorize(argparse.Namespace(packet=tmp_path, signing_key=None))
        assert json.loads(capsys.readouterr().out) == {
            "status": "declined", "authorization_created": False,
        }
        assert not list(tmp_path.iterdir())
        assert b"Customer identity" in os.read(master, 8192)
    finally:
        os.close(master)
        os.close(slave)


def _context_fixture(tmp_path):
    config = {
        "contexts": [
            {"name": "unit-worker", "context": {"cluster": "unit-cluster", "user": "unit-user", "namespace": "unit-workload"}},
            {"name": "unit-controller", "context": {"cluster": "unit-cluster", "user": "unit-user", "namespace": "unit-management"}},
        ],
        "clusters": [{"name": "unit-cluster", "cluster": {"server": "https://cluster.example.invalid", "certificate-authority-data": "synthetic-ca"}}],
        "users": [{"name": "unit-user", "user": {"token": "synthetic-token"}}],
    }
    path = tmp_path / "kubeconfig"
    path.write_text(yaml.safe_dump(config))
    path.chmod(0o600)
    env = {"KUBECONFIG": str(path), customer.CONTROLLER_CONTEXT_ENV: "unit-controller"}
    return config, path, env


@pytest.mark.parametrize("mutation", ["cluster", "user", "namespace", "missing_namespace", "missing_controller", "same_context", "duplicate", "ca", "insecure", "http", "target_namespace", "extra_context", "extra_cluster", "extra_user"])
def test_controller_context_rejects_identity_or_namespace_drift(tmp_path, mutation):
    config, path, env = _context_fixture(tmp_path)
    context = config["contexts"][1]["context"]
    cluster = config["clusters"][0]["cluster"]
    if mutation in {"cluster", "user"}:
        context[mutation] = "different-" + mutation
    elif mutation == "namespace":
        context["namespace"] = "unit-workload"
    elif mutation == "missing_namespace":
        context.pop("namespace")
    elif mutation == "missing_controller":
        env.pop(customer.CONTROLLER_CONTEXT_ENV)
    elif mutation == "same_context":
        env[customer.CONTROLLER_CONTEXT_ENV] = "unit-worker"
    elif mutation == "duplicate":
        config["contexts"].append(config["contexts"][0])
    elif mutation.startswith("extra_"):
        kind = mutation.removeprefix("extra_") + "s"
        config[kind].append({**config[kind][0], "name": "unrelated-identity"})
    elif mutation == "ca":
        cluster.pop("certificate-authority-data")
    elif mutation == "insecure":
        cluster["insecure-skip-tls-verify"] = True
    elif mutation == "http":
        cluster["server"] = "http://cluster.example.invalid"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        customer.controller_context(env, "unit-worker", workload_namespace=(
            "different-workload" if mutation == "target_namespace" else "unit-workload"
        ))


def test_controller_config_separates_verified_contexts_without_worker_profile_change(tmp_path):
    from npa.orchestration.skypilot import workflow

    _, _, env = _context_fixture(tmp_path)
    docs = list(yaml.safe_load_all(customer.PROFILE.read_text()))
    profile = driver().libero_executable_profile_bytes(docs)
    base = tmp_path / "sky.yaml"
    base.write_text("kubernetes:\n  allowed_contexts: [unrelated-context]\n")
    configured = workflow._submission_global_config(
        SimpleNamespace(global_config_path=base), "kubernetes", "k8s/unit-worker",
        documents=docs, extra_env=env,
    )
    assert configured["kubernetes"]["allowed_contexts"] == ["unit-worker", "unit-controller"]
    assert configured["jobs"]["controller"]["resources"]["region"] == "unit-controller"
    assert configured["kubernetes"]["context_configs"] == {
        "unit-worker": {"remote_identity": "NO_UPLOAD"},
        "unit-controller": {"remote_identity": "LOCAL_CREDENTIALS"},
    }
    assert profile == driver().libero_executable_profile_bytes(docs)
    assert customer.controller_context(env, "unit-worker", workload_namespace="unit-workload") == "unit-controller"
    ordinary = workflow._submission_global_config(
        SimpleNamespace(global_config_path=base), "kubernetes", "k8s/unit-worker",
        documents=[{"resources": {"cloud": "kubernetes"}}], extra_env=env,
    )
    assert ordinary["kubernetes"]["allowed_contexts"] == ["unit-worker"]
    assert ordinary["jobs"]["controller"]["resources"]["region"] == "unit-worker"


@pytest.mark.parametrize("controller_drift", [None, "workload-context", "other-context", "gpu-controller", "worker-credentials", "missing-controller-credentials"])
def test_customer_preflight_preserves_only_verified_context_pair(monkeypatch, tmp_path, controller_drift):
    from npa import execution_preflight as preflight
    from npa.orchestration.skypilot import workflow

    _, _, env = _context_fixture(tmp_path)
    docs = list(yaml.safe_load_all(customer.PROFILE.read_text()))
    image = "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:" + "d" * 64
    docs[1]["resources"]["image_id"] = "docker:" + image
    docs[1]["envs"]["BYOF_IMAGE"] = image
    base = tmp_path / "sky.yaml"
    base.write_text("kubernetes:\n  pod_config:\n    spec:\n      serviceAccountName: skypilot-service-account\n")
    configured = workflow._submission_global_config(
        SimpleNamespace(global_config_path=base), "kubernetes", "k8s/unit-worker",
        documents=docs, extra_env=env,
    )
    controller = configured["jobs"]["controller"]["resources"]
    if controller_drift == "gpu-controller":
        controller["accelerators"] = "B200:1"
    elif controller_drift == "worker-credentials":
        configured["kubernetes"]["context_configs"]["unit-worker"]["remote_identity"] = "LOCAL_CREDENTIALS"
    elif controller_drift == "missing-controller-credentials":
        configured["kubernetes"]["context_configs"]["unit-controller"]["remote_identity"] = "SERVICE_ACCOUNT"
    elif controller_drift:
        controller["region"] = "unit-worker" if controller_drift == "workload-context" else "foreign-context"
    observed = {}
    def resolve_target(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(context=kwargs["context"], credentials=kwargs["credentials"])
    monkeypatch.setattr(preflight, "_validate_libero_runtime_authorization", lambda *_a, **_k: {})
    monkeypatch.setattr(preflight, "resolve_execution_target", resolve_target)
    monkeypatch.setattr(preflight, "verify_worker_environment", lambda *_a: None)
    monkeypatch.setattr(preflight, "verify_execution_target", lambda *_a, **_k: {"checks": {}})
    kwargs = dict(project="unit", infra="k8s/unit-worker", extra_env=env, global_config=configured)
    if controller_drift:
        expected = "controller-only kubeconfig" if "credentials" in controller_drift else "verified CPU context"
        with pytest.raises(preflight.ExecutionPreflightError, match=expected):
            preflight.preflight_skypilot_submission(docs, **kwargs)
        assert not observed
    else:
        _, report, injected = preflight.preflight_skypilot_submission(docs, **kwargs)
        assert report["checks"]["libero_customer_run"] is True
        assert not injected
        assert observed["context"] == "unit-worker"
        assert configured["kubernetes"]["allowed_contexts"] == ["unit-worker", "unit-controller"]
        assert configured["jobs"]["controller"]["resources"]["region"] == "unit-controller"
        assert docs[1]["resources"]["region"] == "unit-worker"
