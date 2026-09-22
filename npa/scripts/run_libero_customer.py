#!/usr/bin/env python3
"""Run the fixed LIBERO proof through NPA after a direct customer handoff.

prepare is offline; authorize must be run by the actual customer. submit uses
NPA's standard managed SkyPilot job path and retrieves proof files before exact
job cleanup. It never accepts terms or creates customer signatures itself.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time
from typing import Any

import yaml

from npa.deploy.images import (
    libero_customer_acceptance_notification,
    libero_customer_authorization_signature_payload,
)
from npa.execution_preflight import libero_executable_profile_bytes
from npa.orchestration.skypilot import submit_workflow, workflow_status
from npa.orchestration.skypilot.cleanup import cleanup_launched_workflow
from npa.orchestration.skypilot.launch_transaction import is_terminal_failure_job_status
from npa.orchestration.skypilot.local_api import stop_isolated_api
from npa.workflows.byof.libero_customer import (
    CONTROLLER_CONTEXT_ENV, IMAGE_FILE_ENV, KEY_FILE_ENV, MODE, PROFILE, SECRET_NAMES,
    controller_context, image_manifest, private_bytes, validate_authorization, validate_profile,
)

REPO = Path(__file__).resolve().parents[2]
PHASE_ROOT = "/workspace/byof-runs"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(private_bytes(path))
    if not isinstance(value, dict):
        raise ValueError("private run input must be a JSON object")
    return value


def _write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _write_json(path: Path, value: Any) -> None:
    _write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _record_launch(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((json.dumps(value, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _profile(run_id: str, qualification: dict[str, Any]) -> list[dict[str, Any]]:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id) is None:
        raise ValueError("LIBERO requires an exact run ID")
    docs = list(yaml.safe_load_all(PROFILE.read_text()))
    task = docs[1]
    task["resources"]["image_id"] = "docker:" + qualification["candidate_image"]
    task["envs"]["BYOF_IMAGE"] = qualification["candidate_image"]
    task["envs"]["NPA_BYOF_RUN_ID"] = run_id
    task["envs"]["NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256"] = qualification["canonical_build_metadata_sha256"]
    validate_profile(docs)
    return docs


def prepare(args: argparse.Namespace) -> None:
    manifest, qualification = image_manifest({IMAGE_FILE_ENV: str(args.image_manifest)})
    docs = _profile(args.run_id, qualification)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    profile = libero_executable_profile_bytes(docs)
    notice = libero_customer_acceptance_notification(manifest)
    notice.update({
        "run_id": args.run_id, "workflow_profile_sha256": _sha(profile),
        "upstream_source_revision": qualification["upstream_source_revision"],
        "runtime_delivery": MODE,
        "resume": "Run this helper's authorize command yourself, then return customer-authorization.json and customer-public-key.b64 through the private handoff.",
        "refuse": "Decline, close the prompt, or provide no authorization; no workload is submitted.",
    })
    _write_json(args.output / "request.json", notice)
    _write(args.output / "profile.json", profile)
    _write(args.output / "image-manifest.json", private_bytes(args.image_manifest))
    print(json.dumps({"status": "needs_customer_acceptance", "prepared": True, "run_id": args.run_id}))


def authorize(args: argparse.Namespace) -> None:
    """Only the actual customer's interactive session may issue its evidence."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    request = _json(args.packet / "request.json")
    profile = private_bytes(args.packet / "profile.json")
    manifest, qualification = image_manifest({IMAGE_FILE_ENV: str(args.packet / "image-manifest.json")})
    if (
        request.get("status") != "needs_customer_acceptance"
        or request.get("runtime_delivery") != MODE
        or request.get("candidate_image") != qualification["candidate_image"]
        or request.get("runtime_manifest_sha256") != manifest["runtime_manifest_sha256"]
        or request.get("workflow_profile_sha256") != _sha(profile)
        or request.get("terms") != manifest["customer_acceptance"]["terms"]
        or request.get("upstream_source_revision") != qualification["upstream_source_revision"]
        or profile != libero_executable_profile_bytes(_profile(request["run_id"], qualification))
    ):
        raise ValueError("customer request differs from reviewed image, terms or profile")
    # Never accept --yes, an environment variable or piped standard input.
    # A terminal is not seekable, so text r+ (BufferedRandom) is unsupported.
    with open("/dev/tty", "r", encoding="utf-8") as terminal_input, open(
        "/dev/tty", "w", encoding="utf-8", buffering=1
    ) as terminal:
        terminal.write(json.dumps(request, indent=2) + "\n")
        terminal.write("Customer identity (only its SHA256 is recorded; empty declines): ")
        terminal.flush()
        identity = terminal_input.readline().strip()
        terminal.write("Confirm you have authority to accept every named agreement for this customer.\n")
        expected = "AUTHORIZE " + request["run_id"]
        terminal.write(f"Type {expected} to authorize this exact run, or anything else to decline: ")
        terminal.flush()
        decision = terminal_input.readline().strip()
    if not identity or decision != expected:
        print(json.dumps({"status": "declined", "authorization_created": False}))
        return
    now = datetime.now(timezone.utc)
    if args.signing_key:
        raw = private_bytes(args.signing_key)
        key = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("customer signing key must be Ed25519")
    else:
        key = Ed25519PrivateKey.generate()
        _write(args.packet / "customer-signing-key.pem", key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    authorization = {
        "schema": "npa.libero.customer-runtime-authorization.v2", "solution": "libero",
        "status": "authorized", "authorization_id": "libero-customer-" + secrets.token_hex(16),
        "customer_identity_sha256": _sha(identity), "run_id": request["run_id"],
        "candidate_image": request["candidate_image"],
        "runtime_manifest_sha256": request["runtime_manifest_sha256"],
        "workflow_profile_sha256": request["workflow_profile_sha256"],
        "upstream_source_revision": request["upstream_source_revision"],
        "terms": [{"id": term["id"], "version": term["version"]} for term in request["terms"]],
        "issuer": "customer", "evidence_type": "customer-controlled-signature",
        "customer_signer_public_key_b64": base64.b64encode(public).decode(),
        "acknowledged_at": now.isoformat(), "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=24)).isoformat(),
        "nonce": secrets.token_urlsafe(32),
    }
    signature = key.sign(libero_customer_authorization_signature_payload(authorization))
    authorization["signature"] = {
        "algorithm": "ed25519", "public_key_sha256": _sha(public),
        "signature_b64": base64.b64encode(signature).decode(),
    }
    _write_json(args.packet / "customer-authorization.json", authorization)
    _write(args.packet / "customer-public-key.b64", base64.b64encode(public))
    print(json.dumps({"status": "customer_authorized", "private_key_must_remain_with_customer": True}))


def _exec(target: dict[str, Any], pod: str, argv: list[str], *, data: bytes | None = None,
          output=None, check: bool = True) -> subprocess.CompletedProcess:
    command = ["kubectl", "--kubeconfig", target["kubeconfig"], "--context", target["context"],
               "--namespace", target["namespace"], "exec"]
    if data is not None:
        command.append("-i")
    command += [pod, "--", *argv]
    return subprocess.run(command, input=data, stdout=output or subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check)


def _put_phase(target: dict[str, Any], pod: str, run_id: str, filename: str,
               value: Any, *, group_read: bool = False) -> None:
    script = """import os,sys
p=sys.argv[1]
f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,int(sys.argv[2],8))
with os.fdopen(f,'wb') as s:s.write(sys.stdin.buffer.read())
if sys.argv[3]=='1':os.chown(p,-1,1001)
"""
    _exec(target, pod, ["/usr/local/bin/python", "-c", script,
          f"{PHASE_ROOT}/{run_id}/{filename}", "0440" if group_read else "0600", "1" if group_read else "0"],
          data=json.dumps(value, sort_keys=True).encode())


def _observe(core, binding, job_id: str, managed, target: dict[str, Any]):
    pod, evidence = managed._libero_payload_pod_record(binding, scheduler_job_id=job_id)
    spec = pod["spec"]
    node_name = spec.get("nodeName")
    if not node_name:
        raise LookupError("payload is not scheduled yet")
    if node_name != target["allowed_node"]:
        raise ValueError("payload was scheduled on a different node")
    node = core.read_node(node_name)
    product = (node.metadata.labels or {}).get("nvidia.com/gpu.product", "")
    if "B200" not in product.upper() or int((node.status.allocatable or {}).get("nvidia.com/gpu", "0")) < 1:
        raise ValueError("actual node does not advertise allocatable B200")
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    if not statuses:
        raise LookupError("payload container has not started")
    digest = binding.candidate_image.rsplit("@", 1)[1]
    if len(statuses) != 1 or re.findall(r"sha256:[0-9a-f]{64}", statuses[0].get("imageID", "")) != [digest]:
        raise ValueError("actual payload imageID differs from the qualified digest")
    resources = spec["containers"][0].get("resources") or {}
    if str((resources.get("limits") or {}).get("nvidia.com/gpu")) != "1":
        raise ValueError("payload did not allocate exactly one GPU")
    observation = {
        "pod_observed_image_digest": digest,
        "observation_method": "customer_controller_kubernetes_status_imageID",
        "pod_name_sha256": evidence["payload_pod_name_sha256"],
        "pod_uid_sha256": evidence["payload_pod_uid_sha256"],
        "namespace_sha256": _sha(target["namespace"]), "node_name_sha256": _sha(node_name),
        "actual_service_account": spec["serviceAccountName"],
        "service_account_uid_sha256": _sha(binding.access_state.service_account_uid),
        "controller_service_account_separated": True,
    }
    return pod, observation


def _proof(root: Path, observation: dict[str, Any]) -> dict[str, Any]:
    smoke = _module("npa_libero_proof_contract", REPO / "npa/docker/workbench/libero/libero_smoke.py")
    report = json.loads((root / "libero-smoke.json").read_bytes())
    train, split = report.get("training", {}), report.get("split", {})
    heldout, action = report.get("heldout_metrics", {}), report.get("reloaded_action", {})
    checkpoint = report.get("checkpoint", {})
    runtime = report.get("runtime", {})
    source, dataset, language = (report.get(name, {}) for name in ("source", "dataset", "task_language_model"))
    if not (
        report.get("status") == "passed" and report.get("exit_status") == 0
        and source.get("observed_revision") == smoke.SOURCE_REF
        and source.get("source_prune_path_absent") is True and source.get("git_objects_absent") is True
        and dataset.get("observed_sha256") == smoke.DATASET_SHA256
        and dataset.get("observed_size_bytes") == smoke.DATASET_SIZE
        and language.get("embedding_method") == "upstream_LIBERO_bert_pooler_output"
        and language.get("embedding_finite") is True
        and language.get("source_sha256") == smoke.TASK_EMBEDDING_SOURCE_SHA256
        and set(language.get("files", {})) == set(smoke.LANGUAGE_MODEL_FILES)
        and all(language["files"][name].get("observed_sha256") == record[1] for name, record in smoke.LANGUAGE_MODEL_FILES.items())
        and train.get("optimizer_steps") == train.get("requested_optimizer_steps") == 8
        and train.get("all_losses_finite") is True
        and math.isfinite(float(train.get("parameter_max_abs_delta", 0)))
        and float(train.get("parameter_max_abs_delta", 0)) > 0
        and split.get("disjoint") is True
        and split.get("train_demo_count") == 40 and split.get("heldout_demo_count") == 10
        and split.get("train_sample_count") == 4020 and split.get("heldout_sample_count") == 1048
        and heldout.get("evaluated_sample_count") == 1048
        and math.isfinite(float(heldout.get("negative_log_likelihood", float("nan"))))
        and action.get("finite") is True and action.get("evaluated_sample_count") == 1048
        and action.get("shape", [0])[-1] == 7 and checkpoint.get("strict_state_dict_load") is True
        and checkpoint.get("sha256") == _sha((root / "libero-bc-rnn-smoke.pth").read_bytes())
        and runtime.get("gpu_count") == 1 and runtime.get("compute_capability") == [10, 0]
        and "B200" in str(runtime.get("gpu_model", "")).upper()
        and all(runtime.get(name) == value for name, value in observation.items())
    ):
        raise ValueError("retrieved LIBERO proof does not establish real train/reload/held-out execution")
    return report


def _require_running(status: str, phase: str) -> None:
    if status.upper() == "SUCCEEDED" or is_terminal_failure_job_status(status):
        raise RuntimeError(f"customer managed job terminated before {phase}")


def submit(args: argparse.Namespace) -> None:
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    packet = args.packet.resolve()
    request = _json(packet / "request.json")
    authorization_bytes = private_bytes(packet / "customer-authorization.json")
    authorization = json.loads(authorization_bytes)
    environment = {
        IMAGE_FILE_ENV: str(packet / "image-manifest.json"),
        KEY_FILE_ENV: str(packet / "customer-public-key.b64"),
        SECRET_NAMES[0]: base64.b64encode(authorization_bytes).decode(),
        SECRET_NAMES[1]: _sha(authorization_bytes),
        SECRET_NAMES[2]: authorization["customer_identity_sha256"],
    }
    manifest, qualification = image_manifest(environment)
    docs = _profile(request["run_id"], qualification)
    profile = libero_executable_profile_bytes(docs)
    if _sha(profile) != request["workflow_profile_sha256"] or profile != private_bytes(packet / "profile.json"):
        raise ValueError("prepared customer profile changed")
    authorization, authorization_sha = validate_authorization(
        environment, image_manifest=manifest, run_id=request["run_id"], profile_sha256=_sha(profile)
    )
    target = _json(args.target)
    required = {"project", "context", "controller_context", "namespace", "namespace_uid", "allowed_node", "kubeconfig", "config_path", "isolated_config_dir", "deny_policy_uid", "fetch_policy_uid"}
    if set(target) != required or any(not isinstance(v, str) or not v for v in target.values()):
        raise ValueError("customer target requires the exact owner-provided fields")
    for field in ("kubeconfig", "config_path"):
        private_bytes(Path(target[field]))
    environment["KUBECONFIG"] = target["kubeconfig"]
    environment[CONTROLLER_CONTEXT_ENV] = target["controller_context"]
    controller_context(environment, target["context"], workload_namespace=target["namespace"])
    run_id = request["run_id"]
    runtime_dir = Path(target["isolated_config_dir"])
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    api_client = config.new_client_from_config(config_file=target["kubeconfig"], context=target["context"])
    core, network = client.CoreV1Api(api_client), client.NetworkingV1Api(api_client)
    namespace = core.read_namespace(target["namespace"])
    if namespace.metadata.uid != target["namespace_uid"] or (namespace.metadata.labels or {}).get("npa-libero-run") != run_id:
        raise ValueError("target namespace ownership differs")
    account = core.read_namespaced_service_account("npa-byof-libero-payload", target["namespace"])
    if account.automount_service_account_token is not False or account.secrets:
        raise ValueError("customer payload service account must have no token")
    if core.list_namespaced_pod(target["namespace"]).items:
        raise ValueError("customer namespace must be empty before managed submission")
    policies = {p.metadata.name: p for p in network.list_namespaced_network_policy(target["namespace"]).items}
    if set(policies) != {"libero-deny-egress", "libero-fetch-egress"}:
        raise ValueError("customer namespace network policy inventory differs")
    for name, field in (("libero-deny-egress", "deny_policy_uid"), ("libero-fetch-egress", "fetch_policy_uid")):
        policy = policies[name]
        spec = api_client.sanitize_for_serialization(policy.spec)
        expected_egress = [] if name == "libero-deny-egress" else [{}]
        if policy.metadata.uid != target[field] or spec.get("podSelector") != {} or spec.get("policyTypes") != ["Egress"] or (spec.get("egress") or []) != expected_egress:
            raise ValueError("customer namespace egress policy differs")
    key = private_bytes(packet / "customer-public-key.b64").decode()
    key_object = core.create_namespaced_config_map(target["namespace"], {
        "apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
        "metadata": {"name": "npa-byof-libero-customer-run-key", "labels": {"npa-libero-run": run_id}},
        "data": {"customer-run-public-key.b64": key},
    })
    job_id = ""
    cluster = ""
    cleanup = None
    launch_attempted = False
    try:
        envs = docs[1]["envs"]
        envs["NPA_LIBERO_EXECUTABLE_PROFILE_B64"] = base64.b64encode(profile).decode()
        envs["NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256"] = _sha(profile)
        envs["NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT"] = authorization["expires_at"]
        envs["NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256"] = authorization["signature"]["public_key_sha256"]
        prepared = args.output / "prepared.yaml"
        _write(prepared, yaml.safe_dump_all(docs, sort_keys=False).encode())
        managed = _module("npa_libero_managed_runner", REPO / "npa/scripts/run_byof_container_verify.py")
        state = managed.LiberoAccessState(
            kubeconfig=Path(target["kubeconfig"]), context=target["context"],
            namespace=target["namespace"], namespace_uid=target["namespace_uid"],
            service_account_uid=account.metadata.uid, role_uid="", role_binding_uid="", run_id=run_id,
        )
        binding = managed.LiberoRuntimeBinding(
            evidence={}, access_state=state, candidate_image=qualification["candidate_image"],
            task_name=run_id, customer_identity_sha256=authorization["customer_identity_sha256"], customer_run=True,
        )
        launch_attempted = True
        result = submit_workflow(
            prepared, run_id, isolated_config_dir=runtime_dir,
            config_path=Path(target["config_path"]), controller_backend="kubernetes",
            infra="k8s/" + target["context"], extra_env=environment,
            secret_envs=SECRET_NAMES, project=target["project"],
            transaction_recorder=lambda value: _record_launch(args.output / "launch-transactions.jsonl", value),
        )
        job_id = managed._exact_scheduler_job_id(result.job_id)
        _write_json(args.output / "submission.json", {"run_id": run_id, "scheduler_job_id": job_id})
        while True:
            status = workflow_status(job_id, isolated_config_dir=runtime_dir, config_path=Path(target["config_path"]))
            _require_running(status.status, "materialization")
            try:
                pod, observation = _observe(core, binding, job_id, managed, target)
            except LookupError:
                time.sleep(5)
                continue
            name = pod["metadata"]["name"]
            cluster = pod["metadata"]["labels"]["skypilot-cluster-name"]
            state = replace(state, payload_pod_name=name, payload_pod_uid=pod["metadata"]["uid"], skypilot_cluster_name=cluster)
            binding = replace(binding, access_state=state)
            ready = _exec(target, name, ["cat", f"{PHASE_ROOT}/{run_id}/materialized.json"], check=False)
            if ready.returncode == 0:
                if json.loads(ready.stdout) != {"run_id": run_id, "authorization_sha256": authorization_sha}:
                    raise ValueError("materialization phase differs from customer authorization")
                break
            time.sleep(5)
        probe_ip = socket.getaddrinfo("docs.nvidia.com", 443, type=socket.SOCK_STREAM)[0][4][0]
        probe = "import socket,sys; s=socket.create_connection((sys.argv[1],443),3); s.close()"
        _exec(target, name, ["/usr/local/bin/python", "-c", probe, probe_ip])
        network.delete_namespaced_network_policy("libero-fetch-egress", target["namespace"],
            body=client.V1DeleteOptions(preconditions=client.V1Preconditions(uid=target["fetch_policy_uid"])))
        while True:
            status = workflow_status(job_id, isolated_config_dir=runtime_dir, config_path=Path(target["config_path"]))
            _require_running(status.status, "offline training release")
            remaining = network.list_namespaced_network_policy(target["namespace"]).items
            if len(remaining) != 1 or remaining[0].metadata.uid != target["deny_policy_uid"]:
                raise ValueError("offline training policy inventory differs")
            denied = _exec(target, name, ["/usr/local/bin/python", "-c", probe, probe_ip], check=False)
            if denied.returncode != 0:
                break
            time.sleep(1)
        pod, observation = _observe(core, binding, job_id, managed, target)
        _put_phase(target, name, run_id, "observation.json", observation, group_read=True)
        phase = {"schema": "npa.libero.customer-controller-phase.v1", "phase": "training", "run_id": run_id,
                 "authorization_sha256": authorization_sha, "image": qualification["candidate_image"]}
        _put_phase(target, name, run_id, "training.json", phase)
        _write_json(args.output / "manager-observation.json", observation)
        while True:
            _observe(core, binding, job_id, managed, target)
            completed = _exec(target, name, ["cat", f"{PHASE_ROOT}/{run_id}/completed.json"], check=False)
            if completed.returncode == 0:
                inventory = json.loads(completed.stdout)
                break
            status = workflow_status(job_id, isolated_config_dir=runtime_dir, config_path=Path(target["config_path"]))
            _require_running(status.status, "sealed output")
            time.sleep(5)
        if inventory.get("run_id") != run_id or inventory.get("image") != qualification["candidate_image"] or inventory.get("authorization_sha256") != authorization_sha:
            raise ValueError("sealed output belongs to a different run")
        if inventory.get("schema") != "npa.libero.controller-output-inventory.v1" or inventory.get("smoke_exit_code") != 0:
            raise ValueError("runtime did not complete successfully")
        bootstrap = _module("npa_libero_bootstrap_limits", REPO / "npa/docker/workbench/libero/runtime-bootstrap.py")
        artifacts = args.output / "artifacts"
        artifacts.mkdir(mode=0o700)
        records = inventory.get("files") or []
        if len({r["name"] for r in records}) != len(records) or {r["name"] for r in records} != set(bootstrap.OUTPUT_ARTIFACT_SIZE_LIMITS):
            raise ValueError("sealed output inventory is incomplete or duplicated")
        if sum(r["size_bytes"] for r in records) > bootstrap.MAX_OUTPUT_BYTES:
            raise ValueError("sealed output exceeds the existing aggregate bound")
        for record in records:
            filename = record["name"]
            if filename not in bootstrap.OUTPUT_ARTIFACT_SIZE_LIMITS or not 0 <= record["size_bytes"] <= bootstrap.OUTPUT_ARTIFACT_SIZE_LIMITS[filename]:
                raise ValueError("sealed output inventory has an unsupported file")
            descriptor = os.open(artifacts / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                # The sealed supervisor inventory determines a hard transport
                # bound before any remote bytes reach the controller filesystem.
                _exec(target, name, ["head", "-c", str(record["size_bytes"] + 1),
                      f"{PHASE_ROOT}/{run_id}/output/{filename}"], output=output)
            data = (artifacts / filename).read_bytes()
            if len(data) != record["size_bytes"] or _sha(data) != record["sha256"]:
                raise ValueError("retrieved output differs from the sealed inventory")
        _observe(core, binding, job_id, managed, target)
        proof = _proof(artifacts, observation)
        _write_json(args.output / "output-inventory.json", inventory)
        phase["phase"] = "retrieved"
        _put_phase(target, name, run_id, "retrieved.json", phase)
        while True:
            status = workflow_status(job_id, isolated_config_dir=runtime_dir, config_path=Path(target["config_path"]))
            if status.status == "SUCCEEDED" or is_terminal_failure_job_status(status.status):
                if status.status != "SUCCEEDED":
                    raise RuntimeError("customer managed job did not complete successfully")
                break
            time.sleep(5)
        _write_json(args.output / "proof.json", {"status": "passed", "run_id": run_id, "image": qualification["candidate_image"], "optimizer_steps": proof["training"]["optimizer_steps"]})
    except BaseException as exc:
        if launch_attempted and getattr(exc, "launch_attempted", None) is False:
            launch_attempted = False
        raise
    finally:
        receipt = {"ok": False, "launch_attempted": launch_attempted,
                   "scheduler_job_id": job_id, "local_api_stopped": False,
                   "customer_key_removed": False, "errors": [], "resources_removed": []}
        safe_to_remove = not launch_attempted
        if job_id:
            try:
                cleanup = cleanup_launched_workflow(job_id, run_id, cluster=cluster,
                    isolated_config_dir=runtime_dir, config_path=Path(target["config_path"]),
                    teardown_cluster=bool(cluster))
                receipt.update(errors=cleanup.errors, resources_removed=cleanup.resources_removed)
                safe_to_remove = not cleanup.errors
            except Exception as exc:
                receipt["errors"].append(type(exc).__name__)
        elif launch_attempted:
            receipt["errors"].append("ambiguous_submission_preserved_for_recovery")
        # Retain the exact key and API when launch identity is unresolved.
        if safe_to_remove:
            try:
                stop_isolated_api(runtime_dir)
                receipt["local_api_stopped"] = True
            except Exception as exc:
                receipt["errors"].append(type(exc).__name__)
            try:
                core.delete_namespaced_config_map(key_object.metadata.name, target["namespace"],
                    body=client.V1DeleteOptions(preconditions=client.V1Preconditions(uid=key_object.metadata.uid)))
                receipt["customer_key_removed"] = True
            except ApiException as exc:
                if exc.status == 404:
                    receipt["customer_key_removed"] = True
                else:
                    receipt["errors"].append(type(exc).__name__)
        receipt["ok"] = safe_to_remove and not receipt["errors"]
        _write_json(args.output / "cleanup.json", receipt)
    if not receipt["ok"]:
        raise RuntimeError("customer job cleanup is incomplete; recovery evidence preserved")
    print(json.dumps({"status": "passed", "optimizer_steps": 8, "cleanup": "complete"}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--image-manifest", type=Path, required=True)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    authorize_parser = commands.add_parser("authorize")
    authorize_parser.add_argument("--packet", type=Path, required=True)
    authorize_parser.add_argument("--signing-key", type=Path)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--packet", type=Path, required=True)
    submit_parser.add_argument("--target", type=Path, required=True)
    submit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        {"prepare": prepare, "authorize": authorize, "submit": submit}[args.command](args)
    except Exception as exc:
        # Provider errors and local paths can contain private identifiers.
        print(json.dumps({"status": "refused", "error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
