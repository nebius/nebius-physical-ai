"""Contract and canonical live B200 E2E for all OpenPI pi0.5 Polaris modes.

The live test builds and pushes the pinned source through the canonical BYOF
runner, resolves the result to a registry digest, verifies the built bytes, then
pulls that digest for separate negative-terms and positive-inference workloads.
The same digest then runs the connected direct, cross-pod serve, real optimizer,
and held-out evaluation graph through the top-level npa.workflow controller.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from npa.clients.project_credentials import storage_env_for_project
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_byof_profile_path,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)
from npa.workflows.byof.openpi_service import (
    build_controller_rbac_manifests,
    controller_service_account_name,
    service_resource_names,
)

from .npa_workflow_live_helpers import live_bucket

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
OPENPI_SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "byof-openpi.yaml"
)
FOUR_MODE_SPEC = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "openpi-pi05-four-mode.yaml"
)
EXPECTED_CAPABILITIES = {
    "pi05_droid_jointpos_polaris_checkpoint_download",
    "pi05_droid_jointpos_polaris_direct_infer",
    "pi05_droid_jointpos_polaris_served_infer",
}


def _config() -> dict[str, object]:
    payload = yaml.safe_load(OPENPI_SPEC.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = payload.get("config")
    assert isinstance(config, dict)
    return config


def _planned_args(run_id: str) -> dict[str, str | bool]:
    plan = build_plan(load_spec(OPENPI_SPEC), run_id=run_id)
    assert len(plan.steps) == 1
    argv = list(plan.steps[0].argv)
    assert argv[:4] == ["npa", "workbench", "byof", "run"]
    result: dict[str, str | bool] = {}
    index = 4
    while index < len(argv):
        flag = argv[index]
        assert flag.startswith("--"), argv[index:]
        if index + 1 == len(argv) or argv[index + 1].startswith("--"):
            result[flag] = True
            index += 1
        else:
            result[flag] = argv[index + 1]
            index += 2
    return result


def _last_json(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    index = 0
    last: dict[str, object] | None = None
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            last = value
        index = max(end, start + 1)
    assert last is not None, text[-4000:]
    return last


def _s3_client(project: str | None):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    env = storage_env_for_project(
        project,
        allow_host_creds=True,
        endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
    )
    kwargs: dict[str, object] = {
        "endpoint_url": (env.get("AWS_ENDPOINT_URL") or "").strip() or None,
        "config": Config(s3={"addressing_style": "path"}),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    }
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
    return boto3.client("s3", **kwargs)


def _read_json(s3, bucket: str, key: str) -> dict[str, object]:
    value = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    assert isinstance(value, dict)
    return value


def _sha256_s3_object(s3, bucket: str, key: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
        if chunk:
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _assert_float64_trajectory(value: dict[str, object]) -> None:
    assert value["dtype"] == "float64"
    assert value["finite"] is True
    shape = list(value["shape"])
    assert int(shape[0]) >= 5 and shape[1:] == [8]
    targets = value.get("first_five_targets", [])
    assert len(targets) == 5
    assert all(len(row) == 8 for row in targets)
    assert all(math.isfinite(float(item)) for row in targets for item in row)


def _assert_service_objects_absent(
    *, artifact: dict[str, object], env: dict[str, str], namespace: str
) -> None:
    identity = artifact["cleanup_identity"]
    assert isinstance(identity, dict)
    names = identity["exact_names"]
    assert isinstance(names, dict)
    context = env["NPA_BYOF_K8S_CONTEXT"]
    kubeconfig = env["NPA_BYOF_KUBECONFIG"]
    for key, kind in (
        ("client_job", "job"),
        ("deployment", "deployment"),
        ("service", "service"),
        ("secret", "secret"),
    ):
        result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "--context",
                context,
                "get",
                kind,
                str(names[key]),
                "--namespace",
                namespace,
                "--ignore-not-found",
                "-o",
                "name",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert not result.stdout.strip(), (kind, names[key], result.stdout)
    pod_selector = str(identity["pod_selector"])
    pods = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            kubeconfig,
            "--context",
            context,
            "get",
            "pods",
            "--namespace",
            namespace,
            "--selector",
            pod_selector,
            "-o",
            "name",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert not pods.stdout.strip(), (pod_selector, pods.stdout)


def _assert_service_run_objects_absent(
    *, run_id: str, env: dict[str, str], namespace: str
) -> None:
    """Verify exact service identities are absent even when no success artifact exists."""

    names = service_resource_names(run_id)
    identity = {
        "cleanup_identity": {
            "exact_names": names,
            "pod_selector": (f"npa.nebius.ai/cleanup-owner={names['deployment']}"),
        }
    }
    _assert_service_objects_absent(
        artifact=identity,
        env=env,
        namespace=namespace,
    )


def _assert_controller_rbac_absent(
    *, run_id: str, env: dict[str, str], namespace: str
) -> None:
    name = controller_service_account_name(run_id)
    for kind in ("serviceaccount", "role", "rolebinding"):
        result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                env["NPA_BYOF_KUBECONFIG"],
                "--context",
                env["NPA_BYOF_K8S_CONTEXT"],
                "get",
                kind,
                name,
                "--namespace",
                namespace,
                "--ignore-not-found",
                "-o",
                "name",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert not result.stdout.strip(), (kind, name, result.stdout)


def _kubectl_can_i(
    kube: list[str],
    *,
    verb: str,
    resource: str,
    namespace: str,
    subject: str,
    env: dict[str, str],
) -> bool:
    """Return an exact live authorization answer from ``kubectl auth can-i``."""

    result = subprocess.run(
        [
            *kube,
            "auth",
            "can-i",
            verb,
            resource,
            "--namespace",
            namespace,
            "--as",
            subject,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    answer = result.stdout.strip().lower()
    if (result.returncode, answer) == (0, "yes"):
        return True
    if (result.returncode, answer) == (1, "no"):
        return False
    raise AssertionError(
        "kubectl auth can-i returned an uncertain result: "
        f"exit={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )


def _assert_controller_rbac_scope(
    *, run_id: str, env: dict[str, str], namespace: str, service_account: str
) -> dict[str, object]:
    """Exercise the live ServiceAccount boundary, including a foreign Secret."""

    kube = [
        "kubectl",
        "--kubeconfig",
        env["NPA_BYOF_KUBECONFIG"],
        "--context",
        env["NPA_BYOF_K8S_CONTEXT"],
    ]
    subject = f"system:serviceaccount:{namespace}:{service_account}"
    manifests = build_controller_rbac_manifests(
        run_id=run_id,
        namespace=namespace,
        service_account=service_account,
    )
    secret_rule = next(
        rule
        for rule in manifests["role"]["rules"]
        if rule["resources"] == ["secrets"] and "get" in rule["verbs"]
    )
    exact_secret = secret_rule["resourceNames"][0]
    foreign_secret = f"{service_account}-foreign-probe"

    def can_i(verb: str, resource: str) -> bool:
        return _kubectl_can_i(
            kube,
            verb=verb,
            resource=resource,
            namespace=namespace,
            subject=subject,
            env=env,
        )

    subprocess.run(
        [
            *kube,
            "create",
            "secret",
            "generic",
            foreign_secret,
            "--namespace",
            namespace,
            "--from-literal=scope-probe=not-readable",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        foreign_read = subprocess.run(
            [
                *kube,
                "get",
                "secret",
                foreign_secret,
                "--namespace",
                namespace,
                "--as",
                subject,
                "-o",
                "name",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        evidence = {
            "schema": "npa.workbench.openpi.service-controller-rbac-scope.v1",
            "exact_secret_get_allowed": can_i("get", f"secret/{exact_secret}"),
            "foreign_secret_get_allowed": can_i("get", f"secret/{foreign_secret}"),
            "foreign_secret_read_refused": foreign_read.returncode != 0,
            "pod_list_allowed": can_i("list", "pods"),
            "pod_log_get_allowed": can_i("get", "pods/log"),
            "pod_delete_allowed": can_i("delete", "pods"),
        }
        assert evidence == {
            "schema": "npa.workbench.openpi.service-controller-rbac-scope.v1",
            "exact_secret_get_allowed": True,
            "foreign_secret_get_allowed": False,
            "foreign_secret_read_refused": True,
            "pod_list_allowed": True,
            "pod_log_get_allowed": False,
            "pod_delete_allowed": False,
        }
        return evidence
    finally:
        subprocess.run(
            [
                *kube,
                "delete",
                "secret",
                foreign_secret,
                "--namespace",
                namespace,
                "--ignore-not-found",
                "--wait=true",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )


def _run_four_mode_workflow(
    *,
    run_id: str,
    image: str,
    bucket: str,
    project: str | None,
    registry: str,
    env: dict[str, str],
    config_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    target = resolve_byof_kubernetes_target(project)
    assert target.context and target.kubeconfig
    sky_bin = resolve_skypilot_bin()
    assert sky_bin
    isolated_config_dir = env.get("NPA_BYOF_SKY_ISOLATED_CONFIG_DIR", "").strip()
    assert isolated_config_dir, (
        "the fresh-cluster OpenPI gate requires isolated SkyPilot state via "
        "NPA_BYOF_SKY_ISOLATED_CONFIG_DIR"
    )
    assert Path(isolated_config_dir).is_absolute()
    task_config_path = env.get("NPA_BYOF_SKY_CONFIG_PATH", "").strip()
    assert task_config_path, (
        "the fresh-cluster OpenPI gate requires an exact-context SkyPilot "
        "config via NPA_BYOF_SKY_CONFIG_PATH"
    )
    assert Path(task_config_path).is_absolute()
    namespace = target.namespace or "default"
    service_account = controller_service_account_name(run_id)
    workflow_config = load_spec(FOUR_MODE_SPEC).config
    prefix = f"oss-solutions/openpi/{run_id}"
    command = [
        str(REPO_ROOT / "npa" / ".venv" / "bin" / "npa"),
        "workbench",
        "workflow",
        "submit",
        str(FOUR_MODE_SPEC),
        "--run-id",
        run_id,
        "--runtime",
        "--max-wait-seconds",
        "0",
        "--poll-seconds",
        "30",
        "--stage-src",
        "--sky-bin",
        sky_bin,
        "--isolated-config-dir",
        isolated_config_dir,
        "--infra",
        f"k8s/{target.context}",
        "--registry",
        registry,
        "--s3-endpoint",
        env["AWS_ENDPOINT_URL"],
        "--s3-bucket",
        bucket,
        "--var",
        f"bucket={bucket}",
        "--var",
        f"prefix={prefix}",
        "--var",
        f"runtime_image={image}",
        "--var",
        f"service_namespace={namespace}",
        "--var",
        f"service_account={service_account}",
        "--secret-env",
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
        "--secret-env",
        "AWS_ACCESS_KEY_ID",
        "--secret-env",
        "AWS_SECRET_ACCESS_KEY",
        "--output-format",
        "json",
    ]
    if project:
        command.extend(["--project", project])
    for key, value in sorted((config_overrides or {}).items()):
        command.extend(["--var", f"{key}={value}"])
    command.extend(["--config-path", task_config_path])
    rbac_base = [
        sys.executable,
        "-m",
        "npa.workflows.byof.openpi_service_rbac",
        "--run-id",
        run_id,
        "--namespace",
        namespace,
        "--service-account",
        service_account,
        "--kubeconfig",
        target.kubeconfig,
        "--context",
        target.context,
        "--delete-timeout-seconds",
        str(workflow_config["service_rbac_delete_timeout_seconds"]),
        "--poll-interval-seconds",
        str(workflow_config["service_poll_interval_seconds"]),
        "--api-timeout-seconds",
        str(workflow_config["service_api_timeout_seconds"]),
    ]
    apply = subprocess.run(
        [*rbac_base[:3], "apply", *rbac_base[3:]],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    if apply.returncode != 0:
        return apply
    rbac_scope: dict[str, object] = {}
    try:
        rbac_scope = _assert_controller_rbac_scope(
            run_id=run_id,
            env=env,
            namespace=namespace,
            service_account=service_account,
        )
        submit = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
    finally:
        delete = subprocess.run(
            [*rbac_base[:3], "delete", *rbac_base[3:]],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
    returncode = submit.returncode if delete.returncode == 0 else delete.returncode
    lifecycle = {
        "schema": "npa.workbench.openpi.four-mode-submit.v1",
        "status": "SUCCEEDED" if returncode == 0 else "FAILED",
        "workflow_returncode": submit.returncode,
        "controller_rbac_apply_passed": apply.returncode == 0,
        "controller_rbac_delete_passed": delete.returncode == 0,
        "controller_service_account": service_account,
        "controller_rbac_scope": rbac_scope,
    }
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout="\n".join(
            (apply.stdout, submit.stdout, delete.stdout, json.dumps(lifecycle))
        ),
        stderr="\n".join((apply.stderr, submit.stderr, delete.stderr)),
    )


def _failed_managed_job_logs(
    *, run_id: str, stage_fragment: str, env: dict[str, str]
) -> str:
    """Read the exact failed stage log without accepting queue/API uncertainty."""

    sky_bin = resolve_skypilot_bin()
    assert sky_bin
    queue = subprocess.run(
        [sky_bin, "jobs", "queue", "--all", "--output", "json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert queue.returncode == 0, queue.stderr
    try:
        records = json.loads(queue.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"managed-job queue was not JSON: {queue.stdout!r}"
        ) from exc
    assert isinstance(records, list)
    prefix = run_id.lower()
    matches = [
        record
        for record in records
        if str(record.get("job_name", "")).lower().startswith(prefix)
        and stage_fragment in str(record.get("job_name", "")).lower()
    ]
    assert len(matches) == 1, matches
    record = matches[0]
    assert str(record.get("status", "")).upper() == "FAILED", record
    job_id = str(record.get("job_id", "")).strip()
    assert job_id, record
    logs = subprocess.run(
        [sky_bin, "jobs", "logs", job_id, "--no-follow", "--tail", "200"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert logs.returncode == 0, logs.stderr
    combined = logs.stdout + "\n" + logs.stderr
    assert combined.strip(), f"managed job {job_id} returned no failure log"
    return combined


def _requested_gpu_pods(env: dict[str, str]) -> list[dict[str, object]]:
    context = env.get("NPA_BYOF_K8S_CONTEXT", "").strip()
    kubeconfig = env.get("NPA_BYOF_KUBECONFIG", "").strip()
    assert context and kubeconfig
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            kubeconfig,
            "--context",
            context,
            "get",
            "pods",
            "-A",
            "-o",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    pods = json.loads(result.stdout).get("items", [])
    requested: list[dict[str, object]] = []
    for pod in pods:
        gpu_count = sum(
            int(
                (container.get("resources", {}).get("requests", {}) or {}).get(
                    "nvidia.com/gpu", "0"
                )
            )
            for container in pod.get("spec", {}).get("containers", [])
        )
        if gpu_count:
            metadata = pod.get("metadata", {})
            requested.append(
                {
                    "namespace": metadata.get("namespace", "default"),
                    "name": metadata.get("name", ""),
                    "labels": metadata.get("labels", {}),
                    "annotations": metadata.get("annotations", {}),
                    "gpu_count": gpu_count,
                }
            )
    return requested


def _requested_gpu_count(env: dict[str, str]) -> int:
    return sum(int(pod["gpu_count"]) for pod in _requested_gpu_pods(env))


def _read_json_wait(
    s3, bucket: str, key: str, *, env: dict[str, str]
) -> dict[str, object]:
    """Wait without a fixed deadline while the isolated GPU job is alive."""

    while True:
        try:
            return _read_json(s3, bucket, key)
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code not in {"NoSuchKey", "404"}:
                raise
            if _requested_gpu_count(env) == 0:
                raise AssertionError(
                    "OpenPI workload released its GPU before publishing the required artifact"
                ) from exc
            time.sleep(5)


def _live_env(project: str | None, registry: str) -> dict[str, str]:
    env = dict(os.environ)
    env["NPA_E2E_PROJECT"] = project or env.get("NPA_E2E_PROJECT", "")
    env["NPA_REGISTRY"] = registry
    env.update(
        storage_env_for_project(
            project,
            allow_host_creds=True,
            endpoint_url=env.get("NPA_BYOF_S3_ENDPOINT", ""),
        )
    )
    target = resolve_byof_kubernetes_target(project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    sky_bin = resolve_skypilot_bin()
    if sky_bin:
        env["PATH"] = f"{Path(sky_bin).parent}:{env.get('PATH', '')}"
    return env


def _release_split_run(*, run_id: str, env: dict[str, str]) -> None:
    """Release the exact split-run pod and verify that its B200 is free.

    The direct-launch path requests teardown through ``sky launch --down`` and
    then deliberately stops its local Sky API server.  Kubernetes launches can
    return before that asynchronous teardown removes the pod, while issuing a
    second ``sky down`` races the stopped server.  Delete only a GPU pod whose
    Sky label matches this run, through the exact task kubeconfig/context, then
    use the same provider-side query as the release gate.
    """

    context = env.get("NPA_BYOF_K8S_CONTEXT", "").strip()
    kubeconfig = env.get("NPA_BYOF_KUBECONFIG", "").strip()
    assert context and kubeconfig
    while True:
        pods = _requested_gpu_pods(env)
        if not pods:
            return
        for pod in pods:
            labels = pod.get("labels")
            annotations = pod.get("annotations")
            assert isinstance(labels, dict)
            assert isinstance(annotations, dict)
            full_sky_name = str(annotations.get("skypilot-cluster-name", ""))
            assert labels.get("parent") == "skypilot"
            assert full_sky_name == run_id, (full_sky_name, run_id)
            name = str(pod["name"])
            namespace = str(pod["namespace"])
            assert name and namespace
            subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    kubeconfig,
                    "--context",
                    context,
                    "delete",
                    "pod",
                    name,
                    "--namespace",
                    namespace,
                    "--wait=true",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        time.sleep(5)


def _run_byof(
    cmd: list[str], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def _smoke_command(
    *,
    planned: dict[str, str | bool],
    project: str | None,
    image: str,
    run_id: str,
    output_root: str,
) -> list[str]:
    profile = resolve_byof_profile_path(str(planned["--yaml"]))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "npa" / "scripts" / "run_byof_container_verify.py"),
        "--image",
        image,
        "--run-id",
        run_id,
        "--output-root",
        output_root,
        "--yaml",
        str(profile),
        "--smoke-command",
        str(planned["--smoke-command"]),
        "--solution-name",
        str(planned["--solution-name"]),
        "--capability-name",
        str(planned["--capability-name"]),
        "--smoke-artifact-name",
        str(planned["--smoke-artifact-name"]),
        "--wait-timeout",
        str(planned["--wait-timeout"]),
        "--poll-interval",
        str(planned["--poll-interval"]),
        "--cleanup",
    ]
    config_path = skypilot_config_for_project(project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    return cmd


def _inspect_built_image(tag: str, *, build_command: str) -> dict[str, object]:
    inspected = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    image_config = json.loads(inspected.stdout)[0]["Config"]
    assert image_config["User"] == "ubuntu"
    assert image_config["Labels"]["npa.byof.repo"] == (
        "https://github.com/Physical-Intelligence/openpi.git"
    )
    assert image_config["Labels"]["npa.byof.ref"] == (
        "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    )
    assert (
        image_config["Labels"]["org.nebius.npa.skypilot-bootstrap-contract"]
        == "skypilot-0.12.2-v1"
    )
    assert image_config["Env"] and "HOME=/home/ubuntu" in image_config["Env"]
    assert not any(
        value.startswith("NPA_OPENPI_ACCEPT_GEMMA_TERMS=")
        or value.startswith("OPENPI_DATA_HOME=")
        for value in image_config.get("Env") or []
    )

    probe_code = r"""
import importlib.metadata
import json
from pathlib import Path

metadata = json.loads(Path('/opt/byof/npa_source_metadata.json').read_text())
build_metadata = json.loads(Path('/opt/byof/npa_build_metadata.json').read_text())
editable = []
for distribution in importlib.metadata.distributions():
    value = distribution.read_text('direct_url.json')
    if not value:
        continue
    try:
        direct = json.loads(value)
    except json.JSONDecodeError:
        continue
    if direct.get('dir_info', {}).get('editable') is True:
        editable.append({
            'distribution': distribution.metadata.get('Name', ''),
            'url': direct.get('url', ''),
        })
weight_suffixes = {'.ckpt', '.gguf', '.pt', '.pth', '.safetensors'}
baked_weights = [
    str(path)
    for path in Path('/opt/byof').rglob('*')
    if path.is_file() and path.suffix.lower() in weight_suffixes and path.stat().st_size > 1_000_000
]
cache_roots = [
    path for path in (
        Path('/workspace/openpi-cache'),
        Path('/root/.cache/openpi'),
        Path('/home/ubuntu/.cache/openpi'),
    ) if path.exists() and any(path.rglob('*'))
]
print(json.dumps({
    'source_metadata': metadata,
    'build_metadata': build_metadata,
    'editable_installs': editable,
    'baked_weight_files': baked_weights,
    'populated_checkpoint_cache_roots': [str(path) for path in cache_roots],
}))
"""
    payload_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            # Inspect every possible cache root, including /root, while the
            # image's configured runtime user remains asserted as ubuntu.
            "--user",
            "0:0",
            "--entrypoint",
            "/opt/venv/bin/python",
            tag,
            "-c",
            probe_code,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    payload = json.loads(payload_proc.stdout)
    assert payload["source_metadata"]["ref"] == (
        "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    )
    assert payload["build_metadata"] == {
        "schema": "npa.byof.build.v1",
        "build_command_executed": True,
        "build_command_sha256": hashlib.sha256(build_command.encode()).hexdigest(),
    }
    assert any(
        item["url"] == "file:///opt/byof" for item in payload["editable_installs"]
    )
    assert payload["baked_weight_files"] == []
    assert payload["populated_checkpoint_cache_roots"] == []

    bootstrap_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            tag,
            "-ceu",
            """
            test "$(id -u)" != 0
            test "$HOME" = /home/ubuntu
            test -w /tmp
            test -w "$HOME"
            test -d /run/sshd
            for command_name in sh sudo sshd rsync service; do
              command -v "$command_name" >/dev/null || test -x "/usr/sbin/$command_name"
            done
            sudo -n true
            test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)"
            """,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert bootstrap_proc.returncode == 0
    forwarding_proc = subprocess.run(
        ["docker", "run", "--rm", tag, "/bin/sh", "-c", "printf npa-forwarded"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert forwarding_proc.stdout == "npa-forwarded"

    elf_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cuobjdump",
            tag,
            "--list-elf",
            "/usr/local/bin/npa-openpi-sm100-probe",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert "sm_100" in (elf_proc.stdout + elf_proc.stderr).lower()
    return {
        "user": image_config["User"],
        "source_metadata": payload["source_metadata"],
        "build_command_sha256": payload["build_metadata"]["build_command_sha256"],
        "editable_install": True,
        "bootstrap_attestation": "skypilot-0.12.2-v1",
        "bootstrap_bytes_verified": True,
        "sm100_elf": True,
        "weights_baked": False,
        "terms_baked": False,
    }


def test_openpi_polaris_spec_plans_real_b200_serving() -> None:
    plan = build_plan(load_spec(OPENPI_SPEC), run_id="openpi-polaris-render")
    assert len(plan.steps) == 1
    step = plan.steps[0]
    rendered = " ".join(step.argv)
    config = _config()

    assert step.tool_ref == "workbench.byof.repo"
    assert config["repo_ref"] == "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-openpi-b200-gpu"
    assert "pi05_droid_jointpos_polaris" in rendered
    assert "WebsocketPolicyServer" in rendered
    assert "B200:1" in OPENPI_SPEC.read_text(encoding="utf-8")
    assert secret_env_hints_for_plan(plan.steps) == ("NPA_OPENPI_ACCEPT_GEMMA_TERMS",)
    profile = resolve_byof_profile_path(str(config["resource_profile_yaml"]))
    profile_text = profile.read_text(encoding="utf-8")
    assert "accelerators: B200:1" in profile_text
    assert 'NVIDIA_DRIVER_CAPABILITIES: "compute,utility"' in profile_text
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in profile_text


def test_openpi_split_live_env_uses_one_project_storage_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "incompatible-host-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "incompatible-host-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.stale-region")
    monkeypatch.setenv("NPA_BYOF_S3_ENDPOINT", "https://storage.correct-region")
    monkeypatch.setattr(
        f"{__name__}.storage_env_for_project",
        lambda project, *, allow_host_creds, endpoint_url: {
            "AWS_ACCESS_KEY_ID": "project-key",
            "AWS_SECRET_ACCESS_KEY": "project-secret",
            "AWS_ENDPOINT_URL": endpoint_url,
            "NEBIUS_S3_ENDPOINT": endpoint_url,
        },
    )
    monkeypatch.setattr(
        f"{__name__}.resolve_byof_kubernetes_target",
        lambda _project: SimpleNamespace(
            namespace="default", kubeconfig="", context=""
        ),
    )
    monkeypatch.setattr(f"{__name__}.resolve_skypilot_bin", lambda: "")

    env = _live_env("project", "registry-us.example/registry")

    assert env["AWS_ACCESS_KEY_ID"] == "project-key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "project-secret"
    assert env["AWS_ENDPOINT_URL"] == "https://storage.correct-region"


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, "yes\n", True), (1, "no\n", False)],
)
def test_kubectl_can_i_accepts_authoritative_yes_and_no_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        f"{__name__}.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], returncode, stdout=stdout, stderr=""
        ),
    )

    assert (
        _kubectl_can_i(
            ["kubectl"],
            verb="get",
            resource="secret/exact",
            namespace="default",
            subject="system:serviceaccount:default:controller",
            env={},
        )
        is expected
    )


def test_kubectl_can_i_rejects_uncertain_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        f"{__name__}.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout="", stderr="API unavailable"
        ),
    )

    with pytest.raises(AssertionError, match="uncertain result"):
        _kubectl_can_i(
            ["kubectl"],
            verb="get",
            resource="secret/exact",
            namespace="default",
            subject="system:serviceaccount:default:controller",
            env={},
        )


def test_openpi_four_mode_submit_uses_task_isolated_skypilot_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    isolated = tmp_path / "sky-state"
    monkeypatch.setattr(
        f"{__name__}.resolve_byof_kubernetes_target",
        lambda _project: SimpleNamespace(
            namespace="openpi",
            kubeconfig="/tmp/task-kubeconfig",
            context="task-context",
        ),
    )
    monkeypatch.setattr(f"{__name__}.resolve_skypilot_bin", lambda: "/opt/sky")
    monkeypatch.setattr(f"{__name__}.skypilot_config_for_project", lambda _: "")
    monkeypatch.setattr(
        f"{__name__}._assert_controller_rbac_scope",
        lambda **_kwargs: {"foreign_secret_read_refused": True},
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(f"{__name__}.subprocess.run", run)

    result = _run_four_mode_workflow(
        run_id="openpi-live",
        image="cr.example.invalid/project/openpi@sha256:" + "a" * 64,
        bucket="bucket",
        project="project",
        registry="cr.example.invalid/project",
        env={
            "AWS_ENDPOINT_URL": "https://storage.example.invalid",
            "NPA_BYOF_SKY_ISOLATED_CONFIG_DIR": str(isolated),
            "NPA_BYOF_SKY_CONFIG_PATH": str(tmp_path / "sky-config.yaml"),
        },
    )

    assert result.returncode == 0
    assert len(calls) == 3
    apply_command, _ = calls[0]
    command, kwargs = calls[1]
    delete_command, _ = calls[2]
    assert apply_command[3] == "apply"
    assert delete_command[3] == "delete"
    assert "--delete-timeout-seconds" in apply_command
    assert "--api-timeout-seconds" in delete_command
    expected_service_account = controller_service_account_name("openpi-live")
    assert apply_command[apply_command.index("--service-account") + 1] == (
        expected_service_account
    )
    assert command[command.index("--var") + 1] == "bucket=bucket"
    assert f"service_account={expected_service_account}" in command
    assert command[command.index("--isolated-config-dir") + 1] == str(isolated)
    assert command[command.index("--infra") + 1] == "k8s/task-context"
    assert command[command.index("--config-path") + 1] == str(
        tmp_path / "sky-config.yaml"
    )
    assert kwargs["cwd"] == str(REPO_ROOT)


def test_failed_managed_job_logs_selects_exact_failed_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    queue = [
        {
            "job_id": 40,
            "job_name": "other-run-04-cross_pod_se",
            "status": "FAILED",
        },
        {
            "job_id": 41,
            "job_name": "openpi-failure-04-cross_pod_se",
            "status": "FAILED",
        },
    ]

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "queue" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(queue), stderr=""
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="server pod is Unschedulable: selector mismatch",
            stderr="",
        )

    monkeypatch.setattr(f"{__name__}.resolve_skypilot_bin", lambda: "/opt/sky")
    monkeypatch.setattr(f"{__name__}.subprocess.run", run)

    logs = _failed_managed_job_logs(
        run_id="openpi-failure",
        stage_fragment="cross_pod_se",
        env={},
    )

    assert "Unschedulable" in logs
    assert calls[1] == [
        "/opt/sky",
        "jobs",
        "logs",
        "41",
        "--no-follow",
        "--tail",
        "200",
    ]


def test_openpi_split_run_waits_for_exact_context_gpu_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    pod_results = iter(
        [
            {
                "items": [
                    {
                        "metadata": {
                            "namespace": "default",
                            "name": "negative-run-hash-head",
                            "annotations": {"skypilot-cluster-name": "negative-run"},
                            "labels": {
                                "parent": "skypilot",
                                "skypilot-cluster-name": "negative-run-hash",
                            },
                        },
                        "spec": {
                            "containers": [
                                {"resources": {"requests": {"nvidia.com/gpu": "1"}}}
                            ]
                        },
                    }
                ]
            },
            {"items": []},
        ]
    )

    def run(cmd, **_kwargs):
        calls.append(cmd)
        if "delete" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(next(pod_results)), stderr=""
        )

    monkeypatch.setattr(f"{__name__}.subprocess.run", run)
    monkeypatch.setattr(f"{__name__}.time.sleep", lambda _seconds: None)

    _release_split_run(
        run_id="negative-run",
        env={
            "NPA_BYOF_K8S_CONTEXT": "task-context",
            "NPA_BYOF_KUBECONFIG": "/tmp/task-kubeconfig",
        },
    )

    assert len(calls) == 3
    assert calls[1] == [
        "kubectl",
        "--kubeconfig",
        "/tmp/task-kubeconfig",
        "--context",
        "task-context",
        "delete",
        "pod",
        "negative-run-hash-head",
        "--namespace",
        "default",
        "--wait=true",
    ]
    assert all("--kubeconfig" in call and "--context" in call for call in calls)


def test_openpi_artifact_wait_keeps_live_gpu_until_object_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingObject(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    values: list[dict[str, object] | Exception] = [
        MissingObject(),
        {"status": "success"},
    ]
    monkeypatch.setattr(
        f"{__name__}._read_json",
        lambda *_args: (
            (_ for _ in ()).throw(value)
            if isinstance((value := values.pop(0)), Exception)
            else value
        ),
    )
    monkeypatch.setattr(f"{__name__}._requested_gpu_count", lambda _env: 1)
    monkeypatch.setattr(f"{__name__}.time.sleep", lambda _seconds: None)

    assert _read_json_wait(object(), "bucket", "key", env={}) == {"status": "success"}


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_BYOF_OPENPI_LIVE_B200") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_BYOF_OPENPI_LIVE_B200=1 to run "
        "the digest-pinned OpenPI Polaris B200 smoke."
    ),
)
@pytest.mark.e2e
def test_openpi_polaris_live_b200_all_four_modes(
    e2e_project: str | None,
) -> None:
    assert os.environ.get("NPA_OPENPI_ACCEPT_GEMMA_TERMS") == "YES", (
        "scoped Gemma terms acceptance must be forwarded for this OpenPI run"
    )
    assert not os.environ.get("NPA_BYOF_OPENPI_REUSE_IMAGE", "").strip(), (
        "the canonical gate must build and push; use the runner manually for reuse debugging"
    )
    project_registry = os.environ.get("NPA_BYOF_OPENPI_REGISTRY", "").strip()
    assert project_registry, (
        "NPA_BYOF_OPENPI_REGISTRY must name an authenticated operator-controlled "
        "registry; restricted OpenPI bytes must not enter the official NPA GHCR namespace"
    )
    assert project_registry.rstrip("/").lower() != (
        "ghcr.io/nebius/nebius-physical-ai"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build_run_id = f"byof-openpi-polaris-build-{stamp}"
    negative_run_id = f"byof-openpi-polaris-terms-{stamp}"
    positive_run_id = f"byof-openpi-polaris-e2e-{stamp}"
    planned = _planned_args(positive_run_id)
    bucket = live_bucket(e2e_project)
    output_root = f"s3://{bucket}/oss-solutions/openpi"
    env = _live_env(e2e_project, project_registry)

    build_cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--repo-url",
        str(planned["--repo-url"]),
        "--repo-ref",
        str(planned["--repo-ref"]),
        "--base-profile",
        str(planned["--base-profile"]),
        "--base-image",
        str(planned["--base-image"]),
        "--build-command",
        str(planned["--build-command"]),
        "--project",
        e2e_project or "",
        "--registry",
        project_registry,
        "--workload",
        str(planned["--workload"]),
        "--smoke-command",
        str(planned["--smoke-command"]),
        "--solution-name",
        str(planned["--solution-name"]),
        "--capability-name",
        str(planned["--capability-name"]),
        "--smoke-artifact-name",
        str(planned["--smoke-artifact-name"]),
        "--run-id",
        build_run_id,
        "--skip-run",
    ]
    build_proc = _run_byof(build_cmd, env=env)
    build_combined = build_proc.stdout + "\n" + build_proc.stderr
    assert build_proc.returncode == 0, build_combined[-16000:]
    build = _last_json(build_proc.stdout)
    image = str(build["image"])
    image_tag = str(build["image_tag"])
    assert build["status"] == "ok"
    assert build["build"]["ok"] is True
    assert build["build"]["pushed"] is True
    assert build["run"] == {"skipped": True}
    assert image.startswith(project_registry.rstrip("/") + "/")
    assert re.search(r"@sha256:[0-9a-f]{64}$", image)
    assert build["build"]["runtime_image"] == image
    assert "uv pip install" in str(build["build_command"])
    assert "-e ." in str(build["build_command"])
    assert "nvcc -O2 -arch=sm_100" in str(build["build_command"])
    build_byte_evidence = _inspect_built_image(
        image_tag, build_command=str(build["build_command"])
    )
    print(json.dumps({"openpi_build_evidence": build_byte_evidence}, sort_keys=True))

    negative_env = dict(env)
    negative_env.pop("NPA_OPENPI_ACCEPT_GEMMA_TERMS", None)
    negative_proc = _run_byof(
        _smoke_command(
            planned=planned,
            project=e2e_project,
            image=image,
            run_id=negative_run_id,
            output_root=output_root,
        ),
        env=negative_env,
    )
    _release_split_run(run_id=negative_run_id, env=negative_env)
    s3 = _s3_client(e2e_project)
    negative_prefix = f"oss-solutions/openpi/{negative_run_id}/"
    try:
        negative_summary = _read_json(
            s3, bucket, negative_prefix + "npa_byof_summary.json"
        )
        negative_gate = _read_json(
            s3, bucket, negative_prefix + "openpi_terms_gate.json"
        )
    except Exception as exc:
        combined = negative_proc.stdout + "\n" + negative_proc.stderr
        raise AssertionError(
            "negative OpenPI workload did not publish its required gate evidence:\n"
            + combined[-16000:]
        ) from exc
    assert negative_proc.returncode != 0
    assert negative_summary["status"] == "failed"
    assert negative_summary["smoke_exit_code"] == 64
    assert negative_summary["image"] == image
    assert negative_gate == {
        "schema": "npa.workbench.openpi.terms-gate.v1",
        "status": "refused",
        "exit_code": 64,
        "checkpoint_fetch_started": False,
        "model_import_started": False,
    }
    negative_objects = s3.list_objects_v2(Bucket=bucket, Prefix=negative_prefix).get(
        "Contents", []
    )
    negative_keys = {str(item["Key"]) for item in negative_objects}
    assert (
        negative_prefix + "openpi_pi05_droid_jointpos_polaris_inference.json"
        not in negative_keys
    )
    negative_stderr = (
        s3.get_object(Bucket=bucket, Key=negative_prefix + "solution_smoke_stderr.log")[
            "Body"
        ]
        .read()
        .decode()
    )
    assert "requires scoped Gemma terms acceptance" in negative_stderr
    negative_gpu = (
        s3.get_object(Bucket=bucket, Key=negative_prefix + "nvidia_smi_list.txt")[
            "Body"
        ]
        .read()
        .decode()
        .strip()
        .splitlines()
    )
    assert len(negative_gpu) == 1 and "B200" in negative_gpu[0].upper()
    print(
        json.dumps(
            {
                "openpi_negative_gate_evidence": {
                    **negative_gate,
                    "gpu_product": "B200",
                }
            },
            sort_keys=True,
        )
    )

    # The connected graph is the acceptance surface for all four modes. It
    # deliberately reuses only the immutable digest returned by this fresh
    # builder; model/data/checkpoint bytes remain runtime-only.
    four_mode_run_id = f"openpi-pi05-four-mode-{stamp}"
    four_mode_env = dict(env)
    four_mode_env["NPA_OPENPI_ACCEPT_GEMMA_TERMS"] = "YES"
    four_mode_proc = _run_four_mode_workflow(
        run_id=four_mode_run_id,
        image=image,
        bucket=bucket,
        project=e2e_project,
        registry=project_registry,
        env=four_mode_env,
    )
    four_mode_combined = four_mode_proc.stdout + "\n" + four_mode_proc.stderr
    assert four_mode_proc.returncode == 0, four_mode_combined[-30000:]
    runtime_result = _last_json(four_mode_proc.stdout)
    assert str(runtime_result.get("status", "")).upper() in {
        "SUCCEEDED",
        "SUCCESS",
        "COMPLETED",
        "DONE",
    }, runtime_result
    assert runtime_result["controller_rbac_apply_passed"] is True
    assert runtime_result["controller_rbac_delete_passed"] is True
    _assert_controller_rbac_absent(
        run_id=four_mode_run_id,
        env=four_mode_env,
        namespace=resolve_byof_kubernetes_target(e2e_project).namespace or "default",
    )

    four_prefix = f"oss-solutions/openpi/{four_mode_run_id}/"
    negative = _read_json(s3, bucket, four_prefix + "reports/negative-terms-gate.json")
    dataset_manifest = _read_json(s3, bucket, four_prefix + "data/manifest.json")
    direct = _read_json(s3, bucket, four_prefix + "reports/direct-inference.json")
    serve = _read_json(
        s3, bucket, four_prefix + "reports/cross-pod-service/service.json"
    )
    serve_cleanup = _read_json(
        s3, bucket, four_prefix + "reports/cross-pod-service/cleanup.json"
    )
    training = _read_json(s3, bucket, four_prefix + "reports/training.json")
    checkpoint_manifest = _read_json(
        s3, bucket, four_prefix + "checkpoints/trained/manifest.json"
    )
    evaluation = _read_json(s3, bucket, four_prefix + "reports/heldout-evaluation.json")

    assert negative["schema"] == "npa.workbench.openpi.live-negative-terms-gate.v1"
    assert negative["status"] == "passed"
    assert negative["tested_child_status"] == "refused"
    assert negative["tested_child_exit_code"] == 64
    assert negative["runtime_image"] == image
    child_refusal = negative["tested_child_refusal"]
    assert child_refusal["schema"] == "npa.workbench.openpi.terms-gate.v1"
    assert child_refusal["status"] == "refused"
    assert child_refusal["declared_success_output_uri_untouched"] is True
    assert child_refusal["diagnostic_persistence"] == ("separate_attempt_scoped_uri")
    diagnostic_prefix = f"s3://{bucket}/{four_prefix}diagnostics/terms-refusals/"
    assert child_refusal["diagnostic_uri"].startswith(diagnostic_prefix)
    diagnostic_key = child_refusal["diagnostic_uri"].split(f"s3://{bucket}/", 1)[1]
    persisted_refusal = _read_json(s3, bucket, diagnostic_key)
    assert persisted_refusal["attempt_id"] == child_refusal["attempt_id"]
    assert child_refusal["checkpoint_fetch_started"] is False
    assert child_refusal["model_import_started"] is False
    assert dataset_manifest["schema"] == ("npa.workbench.openpi.mini-franka-dataset.v1")
    assert dataset_manifest["split_isolation"] == {
        "sample_id_intersection": [],
        "sample_hash_intersection": [],
        "disjoint": True,
    }
    assert dataset_manifest["redistribution"]["dataset"] == (
        "private_operator_object_storage_only"
    )
    dataset_hash, dataset_size = _sha256_s3_object(
        s3, bucket, four_prefix + "data/mini-franka-two-camera.npz"
    )
    assert dataset_hash == dataset_manifest["archive_sha256"]
    assert dataset_size == dataset_manifest["archive_size_bytes"]

    assert direct["schema"] == "npa.workbench.openpi.pi05-direct-inference.v2"
    assert direct["status"] == "passed"
    assert direct["runtime_image"] == image
    assert direct["checkpoint"]["weights_baked"] is False
    assert direct["redistribution"]["runtime_image"] == (
        "restricted_private_operator_registry"
    )
    assert direct["checkpoint"]["provenance"]["object_count"] == 27
    _assert_float64_trajectory(direct["trajectory"])

    assert serve["schema"] == "npa.workbench.openpi.pi05-cross-pod-service.v1"
    assert serve["status"] == "passed"
    assert serve["runtime_image"] == image
    assert serve["redistribution"]["runtime_image"] == (
        "restricted_private_operator_registry"
    )
    assert serve["topology"]["service_type"] == "ClusterIP"
    assert serve["topology"]["public_ingress"] is False
    assert serve["topology"]["separate_pods"] is True
    assert serve["topology"]["server_pod_uid"] != serve["topology"]["client_pod_uid"]
    assert serve["topology"]["server_gpu_request"] == 1
    assert serve["topology"]["client_gpu_request"] == 0
    assert serve["topology"]["client_created_after_clusterip_health"] is True
    assert serve["probes"]["readiness"]["passed"] is True
    assert serve["probes"]["liveness"]["configured"] is True
    assert serve["probes"]["clusterip_health"] == "OK"
    assert serve["request_count"] == 2
    assert serve["all_trajectories_float64_finite_t_ge_5_x_8"] is True
    serve_hardware = serve["hardware"]
    assert serve_hardware["gpu_count_allocated"] == 1
    assert all("B200" in item.upper() for item in serve_hardware["device_kinds"])
    assert serve_hardware["compute_capabilities"] == ["10.0"]
    assert serve_hardware["sm100_probe"]["passed"] is True
    for request in serve["client"]["requests"]:
        assert request["dtype"] == "float64"
        assert request["finite"] is True
        assert list(request["shape"])[0] >= 5
        assert list(request["shape"])[1:] == [8]
        assert re.fullmatch(r"[0-9a-f]{64}", request["first_five_targets_sha256"])
    assert serve_cleanup["schema"] == "npa.workbench.openpi.service-cleanup.v1"
    assert serve_cleanup["all_exact_resources_absent"] is True
    assert all(serve_cleanup["verified"].values())
    target = resolve_byof_kubernetes_target(e2e_project)
    _assert_service_objects_absent(
        artifact=serve,
        env=four_mode_env,
        namespace=target.namespace or "default",
    )

    failure_run_id = f"openpi-pi05-service-failure-{stamp}"
    failure_proc = _run_four_mode_workflow(
        run_id=failure_run_id,
        image=image,
        bucket=bucket,
        project=e2e_project,
        registry=project_registry,
        env=four_mode_env,
        config_overrides={
            "service_gpu_node_selector_value": "NO-SUCH-B200-LABEL",
            "service_server_ready_timeout_seconds": "60",
        },
    )
    failure_combined = failure_proc.stdout + "\n" + failure_proc.stderr
    assert failure_proc.returncode != 0, failure_combined[-30000:]
    failure_logs = _failed_managed_job_logs(
        run_id=failure_run_id,
        stage_fragment="cross_pod_se",
        env=four_mode_env,
    )
    assert "Unschedulable" in failure_logs, failure_logs[-30000:]
    _assert_controller_rbac_absent(
        run_id=failure_run_id,
        env=four_mode_env,
        namespace=target.namespace or "default",
    )
    _assert_service_run_objects_absent(
        run_id=failure_run_id,
        env=four_mode_env,
        namespace=target.namespace or "default",
    )
    failure_prefix = f"oss-solutions/openpi/{failure_run_id}/"
    failure_objects = s3.list_objects_v2(Bucket=bucket, Prefix=failure_prefix).get(
        "Contents", []
    )
    failure_keys = {str(item["Key"]) for item in failure_objects}
    assert failure_prefix + "reports/direct-inference.json" in failure_keys
    assert failure_prefix + "reports/cross-pod-service/service.json" not in failure_keys
    assert failure_prefix + "reports/cross-pod-service/cleanup.json" not in failure_keys
    print(
        json.dumps(
            {
                "openpi_service_failure_cleanup_evidence": {
                    "failure": "unschedulable_missing_gpu_label",
                    "service_success_artifact_absent": True,
                    "service_cleanup_artifact_absent": True,
                    "all_exact_resources_absent": True,
                    "controller_rbac_absent": True,
                }
            },
            sort_keys=True,
        )
    )

    assert training["schema"] == "npa.workbench.openpi.pi05-training.v1"
    assert training["status"] == "passed"
    assert training["runtime_image"] == image
    assert training["redistribution"]["trained_checkpoint"] == (
        "private_operator_object_storage_only"
    )
    optimization = training["optimization"]
    assert optimization["forward"] is True
    assert optimization["backward"] is True
    assert optimization["optimizer_steps"] >= 1
    assert optimization["trainable_state_changed"] is True
    assert (
        optimization["trainable_state_sha256_before"]
        != (optimization["trainable_state_sha256_after"])
    )
    assert math.isfinite(float(optimization["trainable_update_l2"]))
    assert float(optimization["trainable_update_l2"]) > 0
    assert optimization["all_metrics_finite"] is True
    assert len(optimization["metrics"]) == optimization["optimizer_steps"]
    assert all(
        math.isfinite(float(value))
        for row in optimization["metrics"]
        for value in row.values()
    )
    checkpoint = training["checkpoint"]
    assert checkpoint["reload_passed"] is True
    assert checkpoint["saved_step"] == optimization["optimizer_steps"]
    assert checkpoint["reloaded_train_state_step"] == checkpoint["saved_step"]
    assert (
        checkpoint["content_manifest_sha256"]
        == (checkpoint_manifest["content_manifest_sha256"])
    )
    records = checkpoint_manifest["files"]
    assert records and checkpoint_manifest["file_count"] == len(records)
    canonical_records = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(canonical_records).hexdigest()
        == (checkpoint_manifest["content_manifest_sha256"])
    )
    readback_size = 0
    for record in records:
        object_hash, object_size = _sha256_s3_object(
            s3,
            bucket,
            four_prefix + "checkpoints/trained/" + record["path"],
        )
        assert object_hash == record["sha256"]
        assert object_size == record["size"]
        readback_size += object_size
    assert readback_size == checkpoint_manifest["total_size_bytes"]

    assert evaluation["schema"] == ("npa.workbench.openpi.pi05-heldout-evaluation.v1")
    assert evaluation["status"] == "passed"
    assert evaluation["runtime_image"] == image
    assert evaluation["redistribution"]["trained_checkpoint"] == (
        "private_operator_object_storage_only"
    )
    assert evaluation["lineage"]["exact_training_checkpoint_consumed"] is True
    assert (
        evaluation["lineage"]["trained_checkpoint_manifest_sha256"]
        == (checkpoint_manifest["content_manifest_sha256"])
    )
    assert evaluation["split_isolation"]["sample_id_intersection"] == []
    assert evaluation["split_isolation"]["disjoint"] is True
    assert evaluation["split_isolation"]["normalization_source"] == (
        "training_split_only"
    )
    assert evaluation["schema_checks"]["all_passed"] is True
    metrics = evaluation["metrics"]
    assert metrics["sample_count"] == dataset_manifest["splits"]["heldout"]["count"]
    assert metrics["all_finite"] is True
    for key in (
        "heldout_model_loss_mean",
        "heldout_action_mae",
        "heldout_action_mse",
    ):
        assert math.isfinite(float(metrics[key]))
    assert len(metrics["heldout_model_loss_per_sample"]) == metrics["sample_count"]
    assert len(metrics["heldout_action_mae_per_dimension"]) == 8
    _assert_float64_trajectory(evaluation["reloaded_trajectory"])

    print(
        json.dumps(
            {
                "openpi_four_mode_evidence": {
                    "image_digest": image.rsplit("@", 1)[1],
                    "direct": direct["trajectory"],
                    "serve_requests": serve["client"]["requests"],
                    "optimizer": optimization,
                    "checkpoint_manifest_sha256": checkpoint_manifest[
                        "content_manifest_sha256"
                    ],
                    "evaluation_metrics": metrics,
                    "service_cleanup": serve_cleanup,
                }
            },
            sort_keys=True,
        )
    )

    positive_env = dict(env)
    positive_env["NPA_OPENPI_ACCEPT_GEMMA_TERMS"] = "YES"
    positive_proc = _run_byof(
        _smoke_command(
            planned=planned,
            project=e2e_project,
            image=image,
            run_id=positive_run_id,
            output_root=output_root,
        ),
        env=positive_env,
    )
    combined = positive_proc.stdout + "\n" + positive_proc.stderr
    assert positive_proc.returncode == 0, combined[-16000:]

    prefix = f"oss-solutions/openpi/{positive_run_id}/"
    try:
        summary = _read_json_wait(
            s3,
            bucket,
            prefix + "npa_byof_summary.json",
            env=positive_env,
        )
        artifact = _read_json_wait(
            s3,
            bucket,
            prefix + "openpi_pi05_droid_jointpos_polaris_inference.json",
            env=positive_env,
        )
    finally:
        _release_split_run(run_id=positive_run_id, env=positive_env)
    assert summary["status"] == "success"
    assert summary["smoke_exit_code"] == 0
    assert summary["image"] == image
    assert artifact["status"] == "passed"
    assert artifact["build"]["runtime_image"] == image
    assert artifact["build"]["build_metadata"] == {
        "schema": "npa.byof.build.v1",
        "build_command_executed": True,
        "build_command_sha256": hashlib.sha256(
            str(build["build_command"]).encode()
        ).hexdigest(),
    }
    assert artifact["build"]["editable_install"]["editable"] is True
    assert artifact["build"]["editable_install"]["url"] == "file:///opt/byof"
    assert artifact["build"]["nvcc_arch"] == "sm_100"
    assert artifact["build"]["weights_baked"] is False
    assert any("sm_100" in line.lower() for line in artifact["build"]["cuda_probe_elf"])
    assert set(artifact["capabilities_exercised"]) == EXPECTED_CAPABILITIES
    assert artifact["checkpoint"]["uri"].endswith("pi05_droid_jointpos_polaris")
    assert artifact["checkpoint"]["weights_baked"] is False
    assert artifact["checkpoint"]["provenance"]["object_count"] > 0
    assert artifact["checkpoint"]["provenance"]["total_size_bytes"] > 1_000_000_000
    assert artifact["terms"] == {
        "forwarded": True,
        "scope": "this_openpi_workload_only",
        "persisted": False,
    }
    assert artifact["runtime"]["visible_device_count"] == 1
    assert "B200" in artifact["runtime"]["device_kind"].upper()
    assert artifact["runtime"]["compute_capability"] == "10.0"
    assert len(artifact["runtime"]["nvidia_smi"]) == 1
    assert "B200" in artifact["runtime"]["nvidia_smi"][0].upper()
    for mode in ("direct", "served"):
        response = artifact["response"][mode]
        assert response["ok"] is True
        assert response["action_shape"][0] >= 5
        assert response["action_shape"][1] == 8
        assert response["action_dtype"] == "float64"
        assert response["finite"] is True
        assert response["latency_ms"] > 0
    assert artifact["response"]["served"]["healthz"] == "OK"
    assert artifact["response"]["served"]["server_infer_ms"] > 0
    assert artifact["response"]["served"]["execution_scope"] == "kubernetes_workload"
    assert artifact["response"]["served"]["client_origin"] == "same_pod_loopback"
    assert artifact["response"]["action_semantics"].startswith(
        "joint_position_targets_dims_0_6"
    )
    first_five = artifact["response"]["first_five_targets"]
    assert len(first_five) == 5 and all(len(row) == 8 for row in first_five)
    assert all(math.isfinite(float(value)) for row in first_five for value in row)
    assert artifact["training_evaluation"]["live_validated"] is False
    print(
        json.dumps(
            {
                "openpi_positive_inference_evidence": {
                    "image_digest": image.rsplit("@", 1)[1],
                    "compute_capability": artifact["runtime"]["compute_capability"],
                    "direct": artifact["response"]["direct"],
                    "served": artifact["response"]["served"],
                }
            },
            sort_keys=True,
        )
    )
