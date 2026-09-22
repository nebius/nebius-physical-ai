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


@pytest.mark.parametrize("mutation", ["setup", "token", "credential", "sidecar", "privileged", "mount", "mode"])
def test_customer_profile_rejects_workload_expansion(mutation):
    docs = list(yaml.safe_load_all(customer.PROFILE.read_text()))
    customer.validate_profile(docs)
    task = docs[1]
    pod = task["resources"]["kubernetes"]["pod_config"]["spec"]
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
    else:
        task["envs"][customer.MODE_ENV] = "unreviewed"
    with pytest.raises(ValueError):
        customer.validate_profile(docs)


def test_customer_profile_survives_standard_kubernetes_config_lift():
    docs = list(yaml.safe_load_all(customer.PROFILE.read_text()))
    task = docs[1]
    task["config"] = {"kubernetes": task["resources"].pop("kubernetes")}
    task["resources"]["region"] = "unit-context"
    customer.validate_profile(docs)


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
    target = {name: "synthetic-" + name for name in ("project", "context", "namespace", "namespace_uid", "allowed_node", "kubeconfig", "config_path", "deny_policy_uid", "fetch_policy_uid")}
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


@pytest.mark.parametrize("attempted", [False, None, True])
def test_submit_failure_preserves_ambiguous_launch_recovery(monkeypatch, tmp_path, attempted):
    module, args, calls = _submit_fixture(monkeypatch, tmp_path)
    def fail(*_a, **_k):
        raise SkyPilotSubmitError("synthetic launch failure", launch_attempted=attempted)
    monkeypatch.setattr(module, "submit_workflow", fail)
    with pytest.raises(SkyPilotSubmitError):
        module.submit(args)
    receipt = json.loads((args.output / "cleanup.json").read_text())
    assert receipt["ok"] is (attempted is False)
    assert calls == (["stop-api", "delete-key"] if attempted is False else [])


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
