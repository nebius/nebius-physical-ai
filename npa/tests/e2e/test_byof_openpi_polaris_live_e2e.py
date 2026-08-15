"""Contract and canonical live B200 E2E for OpenPI pi0.5 Polaris serving.

The live test builds and pushes the pinned source through the canonical BYOF
runner, resolves the result to a registry digest, verifies the built bytes, then
pulls that digest for separate negative-terms and positive-inference workloads.
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

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_byof_profile_path,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)
from npa.workflows.sim2real.registry_auth import ensure_nebius_registry_pull_secret

from .npa_workflow_live_helpers import live_bucket

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
OPENPI_SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "byof-openpi.yaml"
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


def _refresh_pull_secrets(*, project: str | None, registry: str) -> None:
    """Materialize private-registry auth before the split negative/positive runs."""

    server = registry.removeprefix("docker:").split("/", 1)[0].strip()
    assert server.startswith("cr.") and ".nebius.cloud" in server
    target = resolve_byof_kubernetes_target(project)
    for namespace in sorted({target.namespace or "default", "default"}):
        for secret_name in ("agent-sa", "npa-nebius-registry"):
            ensure_nebius_registry_pull_secret(
                registry_server=server,
                secret_name=secret_name,
                namespace=namespace,
                kubeconfig=target.kubeconfig,
                k8s_context=target.context,
            )


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
    expected_prefix = run_id[:30]
    while True:
        pods = _requested_gpu_pods(env)
        if not pods:
            return
        for pod in pods:
            labels = pod.get("labels")
            assert isinstance(labels, dict)
            sky_name = str(labels.get("skypilot-cluster-name", ""))
            assert labels.get("parent") == "skypilot"
            assert sky_name.startswith(expected_prefix), (sky_name, run_id)
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


def test_openpi_split_live_runs_refresh_pull_secrets_for_selected_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        f"{__name__}.resolve_byof_kubernetes_target",
        lambda _project: SimpleNamespace(
            namespace="openpi",
            kubeconfig="/tmp/task-kubeconfig",
            context="task-context",
        ),
    )
    monkeypatch.setattr(
        f"{__name__}.ensure_nebius_registry_pull_secret",
        lambda **kwargs: calls.append(kwargs),
    )

    _refresh_pull_secrets(
        project="project", registry="cr.us-central1.nebius.cloud/registry"
    )

    assert calls == [
        {
            "registry_server": "cr.us-central1.nebius.cloud",
            "secret_name": secret_name,
            "namespace": namespace,
            "kubeconfig": "/tmp/task-kubeconfig",
            "k8s_context": "task-context",
        }
        for namespace in ("default", "openpi")
        for secret_name in ("agent-sa", "npa-nebius-registry")
    ]


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

    env = _live_env("project", "cr.us-central1.nebius.cloud/registry")

    assert env["AWS_ACCESS_KEY_ID"] == "project-key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "project-secret"
    assert env["AWS_ENDPOINT_URL"] == "https://storage.correct-region"


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
def test_openpi_polaris_live_b200_build_terms_and_served_inference(
    e2e_project: str | None,
) -> None:
    assert os.environ.get("NPA_OPENPI_ACCEPT_GEMMA_TERMS") == "YES", (
        "scoped Gemma terms acceptance must be forwarded for this OpenPI run"
    )
    assert not os.environ.get("NPA_BYOF_OPENPI_REUSE_IMAGE", "").strip(), (
        "the canonical gate must build and push; use the runner manually for reuse debugging"
    )
    saved_registry = resolve_container_registry(e2e_project)
    project_registry = (
        os.environ.get("NPA_BYOF_OPENPI_PROJECT_REGISTRY", "").strip() or saved_registry
    )
    assert project_registry.startswith("cr.") and ".nebius.cloud/" in project_registry

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
    _refresh_pull_secrets(project=e2e_project, registry=project_registry)

    negative_env = dict(env)
    negative_env["NPA_OPENPI_ACCEPT_GEMMA_TERMS"] = "NO"
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
    negative_summary = _read_json(s3, bucket, negative_prefix + "npa_byof_summary.json")
    negative_gate = _read_json(s3, bucket, negative_prefix + "openpi_terms_gate.json")
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
