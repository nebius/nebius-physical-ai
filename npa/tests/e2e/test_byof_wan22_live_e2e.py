"""Contract and gated live RTX PRO E2E for the Wan 2.2 BYOF solution spec.

The always-on test plans/renders the checked-in spec. The live test builds and
pushes the image from that spec's repo/ref/base/build command, launches its own
RTX PRO 6000 resource profile, then verifies the named JSON and directly decodes the
published MP4. No live capability is inferred from the local test.

Live gates (all required):

* ``NPA_INTEGRATION_E2E=1``
* ``NPA_BYOF_WAN22_LIVE_GPU=1``
* normal NPA project, registry, Kubernetes, and S3 operator configuration

After a build/push succeeds but pre-launch infrastructure validation fails,
``NPA_BYOF_WAN22_REUSE_IMAGE`` may point at that exact immutable run tag so the
retry exercises the pushed image without rebuilding it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)

from .npa_workflow_live_helpers import live_bucket

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
WAN_SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "byof-wan2.2.yaml"
)
PROFILE_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "byof" / "profiles"
EXPECTED_CAPABILITIES = {
    "wan2.2_ti2v_5b_text_to_video",
    "wan2.2_decoded_mp4_validation",
}


def _spec_payload() -> dict[str, object]:
    payload = yaml.safe_load(WAN_SPEC.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _spec_config() -> dict[str, object]:
    config = _spec_payload().get("config")
    assert isinstance(config, dict)
    return config


def _planned_byof_args(run_id: str) -> dict[str, str | bool]:
    """Return the planner-rendered BYOF arguments for the checked-in spec."""
    from npa.orchestration.npa_workflow import build_plan, load_spec

    steps = build_plan(load_spec(WAN_SPEC), run_id=run_id).to_dict().get("steps") or []
    assert len(steps) == 1, steps
    argv = [str(part) for part in (steps[0].get("argv") or [])]
    assert argv[:4] == ["npa", "workbench", "byof", "run"], argv
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


def _parse_last_json_blob(text: str) -> dict[str, object]:
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
    if last is None:
        raise AssertionError(f"BYOF runner returned no JSON summary:\n{text[-4000:]}")
    return last


def _s3_client(e2e_project: str | None):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    env = storage_env_for_project(e2e_project, allow_host_creds=True)
    kwargs: dict[str, object] = {
        "endpoint_url": (env.get("AWS_ENDPOINT_URL") or "").strip() or None,
        "config": Config(s3={"addressing_style": "path"}),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    }
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
    return boto3.client("s3", **kwargs)


def _read_s3_json(s3, bucket: str, key: str) -> dict[str, object]:
    payload = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    assert isinstance(payload, dict), key
    return payload


def _decode_mp4(path: Path) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    assert ffprobe and ffmpeg, (
        "live Wan E2E requires ffprobe and ffmpeg on the operator"
    )
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert decoded.returncode == 0, decoded.stderr
    payload = json.loads(probe.stdout)
    streams = payload.get("streams") or []
    assert len(streams) == 1, payload
    return streams[0]


def test_wan22_spec_plans_the_real_pinned_rtxpro_workload() -> None:
    from npa.orchestration.npa_workflow import build_plan, load_spec

    spec = load_spec(WAN_SPEC)
    plan = build_plan(spec, run_id="wan22-render-check")
    steps = plan.to_dict().get("steps") or []
    assert len(steps) == 1, steps
    step = steps[0]
    assert step.get("tool_ref") == "workbench.byof.repo"
    argv = [str(part) for part in (step.get("argv") or [])]
    rendered = " ".join(argv)
    config = _spec_config()

    assert config["repo_url"] == "https://github.com/Wan-Video/Wan2.2.git"
    assert config["repo_ref"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    assert config["base_image"] == "ubuntu:22.04"
    assert (
        config["resource_profile_yaml"]
        == "byof-solution-smoke-wan22-rtxpro-gpu"
    )
    assert "--wait-timeout 0" in rendered
    assert "WanTI2V" in rendered and "generator.generate(" in rendered
    assert "cv2.VideoCapture" in rendered and "wan2_2_ti2v_5b.mp4" in rendered
    assert str(config["prompt"]) in rendered
    assert "{{config." not in rendered
    for capability in EXPECTED_CAPABILITIES:
        assert capability in rendered

    payload = _spec_payload()
    resources = payload["resources"]["gpu"]
    assert (
        resources["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    assert resources["disk_size"] == 200
    profile = PROFILE_DIR / f"{config['resource_profile_yaml']}.yaml"
    assert profile.is_file()
    profile_text = profile.read_text(encoding="utf-8")
    assert "accelerators: RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in profile_text
    assert "disk_size: 200" in profile_text
    assert 'NVIDIA_DRIVER_CAPABILITIES: "compute,utility"' in profile_text


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_BYOF_WAN22_LIVE_GPU") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_BYOF_WAN22_LIVE_GPU=1 to build, "
        "push, and run the Wan 2.2 RTX PRO 6000 smoke."
    ),
)
@pytest.mark.e2e
def test_wan22_live_rtxpro_build_push_generate_and_decode(
    e2e_project: str | None,
    tmp_path: Path,
) -> None:
    config = _spec_config()
    registry = resolve_container_registry(e2e_project)
    assert registry, "NPA container registry could not be resolved"
    reuse_image = os.environ.get("NPA_BYOF_WAN22_REUSE_IMAGE", "").strip()
    if reuse_image:
        assert reuse_image.startswith(registry.rstrip("/") + "/"), reuse_image
        run_id = reuse_image.rsplit(":", 1)[-1]
        assert run_id.startswith("byof-wan22-e2e-"), run_id
    else:
        run_id = "byof-wan22-e2e-" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    planned = _planned_byof_args(run_id)
    image = reuse_image or f"{registry.rstrip('/')}/npa-byof:{run_id}"
    profile = PROFILE_DIR / f"{planned['--yaml']}.yaml"
    out_bucket = live_bucket(e2e_project)
    output_root = f"s3://{out_bucket}/oss-solutions/wan2.2"
    key_prefix = f"oss-solutions/wan2.2/{run_id}/"

    cmd = [
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
        "--image",
        image,
        "--workload",
        str(planned["--workload"]),
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
        "--run-id",
        run_id,
        "--output-root",
        output_root,
        "--wait-timeout",
        str(planned["--wait-timeout"]),
        "--poll-interval",
        str(planned["--poll-interval"]),
        "--cleanup",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    if reuse_image:
        cmd.extend(["--skip-build", "--skip-push"])

    env = dict(os.environ)
    env["NPA_E2E_PROJECT"] = e2e_project or env.get("NPA_E2E_PROJECT", "")
    env.setdefault("NPA_REGISTRY", registry)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined[-12000:]
    runner = _parse_last_json_blob(proc.stdout)
    assert runner["status"] == "ok"
    assert runner["image"] == image
    assert runner["repo_ref"] == planned["--repo-ref"]
    if reuse_image:
        assert runner["build"] == {"ok": True, "skipped": True}
    else:
        assert runner["build"] == {"ok": True, "pushed": True}
    assert image.startswith(registry.rstrip("/") + "/")

    s3 = _s3_client(e2e_project)
    summary = _read_s3_json(s3, out_bucket, key_prefix + "npa_byof_summary.json")
    artifact_name = str(planned["--smoke-artifact-name"])
    artifact = _read_s3_json(s3, out_bucket, key_prefix + artifact_name)
    inventory = _read_s3_json(
        s3, out_bucket, key_prefix + "wan2_2_runtime_inventory.json"
    )
    assert summary["status"] == "success"
    assert summary["solution_name"] == "wan2.2"
    assert summary["capability_name"] == "wan2.2_ti2v_5b_text_to_video"
    assert summary["smoke_artifact_name"] == artifact_name
    assert summary["smoke_exit_code"] == 0
    assert summary["image"] == image
    assert summary["metadata"]["repo"] == planned["--repo-url"]
    assert summary["metadata"]["ref"] == planned["--repo-ref"]

    assert artifact["schema"] == "npa.workbench.byof.wan2_2_ti2v_5b.v1"
    assert artifact["solution"] == "wan2.2"
    assert artifact["upstream_ref"] == planned["--repo-ref"]
    assert artifact["prompt"] == config["prompt"]
    assert artifact["model_ref"] == "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    assert artifact["runtime_inventory_filename"] == "wan2_2_runtime_inventory.json"
    assert set(artifact["capabilities_exercised"]) == EXPECTED_CAPABILITIES
    assert artifact["deferred"] == []
    assert artifact["requested"] == {
        "width": 1280,
        "height": 704,
        "frame_count": 17,
        "fps": 24.0,
        "inference_steps": 8,
    }
    observed = artifact["observed"]
    assert observed["width"] == 1280 and observed["height"] == 704
    assert observed["frame_count"] == 17
    assert float(observed["fps"]) > 0
    assert float(observed["max_spatial_std"]) >= 1.0
    assert int(observed["pixel_range"]) >= 4
    assert float(observed["mean_temporal_abs_delta"]) > 0.001
    topology = artifact["device_topology"]
    assert topology["cuda_device_count"] == 1
    assert len(topology["devices"]) == 1
    assert topology["devices"][0]["compute_capability"] == [12, 0]
    assert "sm_120" in topology["torch_cuda_arch_list"]
    assert topology["driver_versions"]
    assert topology["flash_attention_installed"] is False
    assert topology["sdpa_source_binding"] is True
    assert topology["sdpa_probe"]["finite"] is True
    assert (
        topology["attention_backend"]
        == "torch.nn.functional.scaled_dot_product_attention"
    )

    assert inventory["schema"] == "npa.workbench.byof.wan2_2_runtime_inventory.v1"
    assert inventory["source"]["ref"] == planned["--repo-ref"]
    assert inventory["baked_runtime"]["non_root"] is True
    assert inventory["baked_runtime"]["uid"] != 0
    assert inventory["baked_runtime"]["venv_readable"] is True
    assert inventory["baked_runtime"]["interpreter_accessible"] is True
    assert inventory["baked_runtime"]["large_checkpoint_shaped_files"] == []
    runtime_stack = inventory["runtime_stack"]
    assert runtime_stack["devices"][0]["compute_capability"] == [12, 0]
    assert "sm_120" in runtime_stack["torch_cuda_arch_list"]
    assert runtime_stack["driver_versions"]
    assert runtime_stack["torch_cuda"] == "12.8"
    assert runtime_stack["flash_attention_installed"] is False
    assert runtime_stack["sdpa_source_binding"] is True
    assert runtime_stack["sdpa_probe"]["finite"] is True
    assert inventory["runtime_acquisition"]["weights_baked"] is False
    assert (
        inventory["runtime_acquisition"]["model"]["ref"]
        == artifact["model_ref"]
    )
    installed = {
        package["name"].lower(): package["version"]
        for package in inventory["baked_runtime"]["python_packages"]
    }
    assert installed["wan"] == "2.2.0"
    assert installed["torch"] == "2.7.1+cu128"
    assert inventory["baked_runtime"]["os_packages"]

    video_key = key_prefix + str(artifact["output_filename"])
    video_head = s3.head_object(Bucket=out_bucket, Key=video_key)
    assert video_head["ContentLength"] == artifact["output_size_bytes"]
    assert video_head["ContentLength"] > 4096
    video_path = tmp_path / "wan2_2_ti2v_5b.mp4"
    s3.download_file(out_bucket, video_key, str(video_path))
    stream = _decode_mp4(video_path)
    assert int(stream["width"]) == 1280
    assert int(stream["height"]) == 704
    assert int(stream["nb_read_frames"]) == 17
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    assert denominator > 0 and numerator / denominator > 0
