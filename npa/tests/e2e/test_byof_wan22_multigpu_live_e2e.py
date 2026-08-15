"""Contract and gated real four-B200 E2E for distributed Wan 2.2 TI2V-5B.

The always-on test plans the dedicated checked-in spec. The live gate may reuse
the exact immutable image already accepted by the single-GPU Wan run because the
distributed behavior is an upstream/runtime contract; it still pulls that image
onto a B200 node and fails closed unless all four physical devices participate in
one `torch.distributed.run → instrumentation wrapper → runpy(generate.py)`
generation with NCCL, FULL_SHARD FSDP, and Ulysses.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.clients.config import resolve_container_registry
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)
from npa.workflows.wan_rerun import MULTI_GPU_LAYOUT

from .npa_workflow_live_helpers import live_bucket
from .test_byof_wan22_live_e2e import (
    _decode_mp4,
    _parse_last_json_blob,
    _read_s3_json,
    _s3_client,
    _verify_published_rrd,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
WAN_SPEC = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "byof-wan2.2-multigpu.yaml"
)
PROFILE_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "byof" / "profiles"
EXPECTED_CAPABILITIES = {
    "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
    "wan2.2_distributed_rank_topology_validation",
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


def test_wan22_multigpu_spec_plans_the_real_official_path() -> None:
    planned = _planned_byof_args("wan22-multigpu-render-check")
    config = _spec_config()
    smoke = str(planned["--smoke-command"])

    assert planned["--repo-ref"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    assert planned["--yaml"] == "byof-solution-smoke-wan22-b200-4gpu"
    assert planned["--base-profile"] == "prebuilt"
    assert planned["--base-image"] == "tool://wan2-2"
    assert planned["--wait-timeout"] == "-1"
    assert planned["--capability-name"] in EXPECTED_CAPABILITIES
    assert "--nproc_per_node=4" in smoke
    assert "--dit_fsdp --t5_fsdp --ulysses_size 4" in smoke
    assert "ShardingStrategy.FULL_SHARD" in smoke
    assert "ulysses_all_to_all_calls" in smoke
    assert "observer_final_barrier" in smoke
    assert 'export NCCL_CUMEM_ENABLE="0"' in smoke
    assert 'export NCCL_CUMEM_HOST_ENABLE="0"' in smoke
    assert 'export NCCL_NVLS_ENABLE="0"' in smoke
    assert 'export NCCL_SOCKET_IFNAME="=eth0"' in smoke
    assert 'export NCCL_SOCKET_FAMILY="AF_INET"' in smoke
    assert 'export NCCL_IB_DISABLE="1"' in smoke
    assert 'export TORCH_NCCL_USE_COMM_NONBLOCKING="1"' in smoke
    assert "wan2_2_multigpu_nccl_rank_{rank}.log" in smoke
    assert "wan2_2_multigpu_progress_rank_" in smoke
    assert "wan2_2_multigpu_nccl_summary.json" in smoke
    assert "process_group_destroyed" in smoke
    assert "ncclGetVersion" in smoke
    assert "wan2_2_multigpu_topology.json" in smoke
    encoded_prompt = base64.b64encode(str(config["prompt"]).encode()).decode()
    assert encoded_prompt in smoke
    assert str(config["prompt"]) not in smoke
    assert "{{config." not in smoke
    for capability in EXPECTED_CAPABILITIES:
        assert capability in smoke

    payload = _spec_payload()
    resources = payload["resources"]["gpu"]
    assert resources == {
        "cloud": "kubernetes",
        "accelerators": "B200:4",
        "cpus": 32,
        "memory": "256Gi",
        "disk_size": 200,
    }
    profile = PROFILE_DIR / f"{planned['--yaml']}.yaml"
    text = profile.read_text(encoding="utf-8")
    assert "accelerators: B200:4" in text
    assert "memory: 256+" in text
    assert 'NVIDIA_DRIVER_CAPABILITIES: "compute,utility"' in text


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_BYOF_WAN22_MULTIGPU_LIVE_GPU") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_BYOF_WAN22_MULTIGPU_LIVE_GPU=1 "
        "to run the real four-B200 distributed Wan smoke."
    ),
)
@pytest.mark.e2e
def test_wan22_live_four_b200_fsdp_ulysses_generate_and_decode(
    e2e_project: str | None,
    tmp_path: Path,
) -> None:
    config = _spec_config()
    planned = _planned_byof_args("wan22-multigpu-live-plan")
    registry = resolve_container_registry(e2e_project)
    assert registry, "NPA container registry could not be resolved"
    reuse_image = os.environ.get("NPA_BYOF_WAN22_MULTIGPU_REUSE_IMAGE", "").strip()
    assert reuse_image, (
        "the live acceptance run requires an explicitly digest-pinned image"
    )
    assert reuse_image.startswith(registry.rstrip("/") + "/"), reuse_image
    assert re.search(r"@sha256:[0-9a-f]{64}$", reuse_image), reuse_image
    run_id = "byof-wan22-multigpu-e2e-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    image = reuse_image
    profile = PROFILE_DIR / f"{planned['--yaml']}.yaml"
    out_bucket = live_bucket(e2e_project)
    output_root = f"s3://{out_bucket}/oss-solutions/wan2.2-multigpu"
    key_prefix = f"oss-solutions/wan2.2-multigpu/{run_id}/"

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
        image,
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
    assert skypilot_bin
    env["NPA_SKYPILOT_BIN"] = skypilot_bin
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
    assert proc.returncode == 0, combined[-20000:]
    runner = _parse_last_json_blob(proc.stdout)
    assert runner["status"] == "ok"
    assert runner["image"] == image
    assert runner["repo_ref"] == planned["--repo-ref"]
    assert runner["build"] == {"ok": True, "skipped": True}

    s3 = _s3_client(e2e_project)
    summary = _read_s3_json(s3, out_bucket, key_prefix + "npa_byof_summary.json")
    artifact = _read_s3_json(
        s3, out_bucket, key_prefix + "wan2_2_ti2v_5b_multigpu.json"
    )
    topology = _read_s3_json(
        s3, out_bucket, key_prefix + "wan2_2_multigpu_topology.json"
    )
    inventory = _read_s3_json(
        s3, out_bucket, key_prefix + "wan2_2_multigpu_runtime_inventory.json"
    )

    assert summary["status"] == "success"
    assert summary["solution_name"] == "wan2.2-multigpu"
    assert summary["capability_name"] == planned["--capability-name"]
    assert summary["smoke_exit_code"] == 0
    assert summary["image"] == image
    assert summary["metadata"]["ref"] == planned["--repo-ref"]

    assert artifact["schema"] == "npa.workbench.byof.wan2_2_ti2v_5b_multigpu.v1"
    assert artifact["solution"] == "wan2.2"
    assert artifact["upstream"]["ref"] == planned["--repo-ref"]
    assert artifact["upstream"]["entrypoint"].endswith(
        "--nproc_per_node=4 wan22_distributed_wrapper.py"
    )
    assert artifact["upstream"]["wrapper_execution"] == (
        "runpy.run_path('/opt/byof/generate.py', run_name='__main__')"
    )
    assert artifact["model"]["weights_baked"] is False
    assert artifact["generation"]["prompt"] == config["prompt"]
    assert set(artifact["capabilities_exercised"]) == EXPECTED_CAPABILITIES
    assert artifact["deferred"] == []
    distributed = artifact["distributed"]
    assert distributed["world_size"] == 4
    assert distributed["local_world_size"] == 4
    assert distributed["node_count"] == 1
    assert distributed["backend"] == "nccl"
    assert distributed["fsdp"] == {
        "enabled": True,
        "sharding_strategy": "FULL_SHARD",
        "t5": True,
        "dit": True,
    }
    assert distributed["ulysses"] == {
        "enabled": True,
        "size": 4,
        "num_attention_heads": 24,
    }

    assert topology["schema"] == "npa.workbench.byof.wan2_2_multigpu_topology.v1"
    ranks = topology["rank_evidence"]
    assert len(ranks) == 4
    assert {item["rank"] for item in ranks} == {0, 1, 2, 3}
    assert {item["local_rank"] for item in ranks} == {0, 1, 2, 3}
    assert len({item["device"]["uuid_sha256"] for item in ranks}) == 4
    assert len({item["hostname_sha256"] for item in ranks}) == 1
    for item in ranks:
        assert item["process_group_initialized"] is True
        assert item["backend"] == "nccl"
        assert item["nccl_all_reduce"]["finite"] is True
        assert item["nccl_all_reduce"]["observed_sum"] == 10.0
        assert item["current_cuda_device"] == item["local_rank"]
        assert item["device"]["compute_capability"] == [10, 0]
        assert "sm_100" in item["torch_cuda_arch_list"]
        assert len(item["fsdp_wrappers"]) == 2
        assert all(
            w["sharding_strategy"].endswith("FULL_SHARD") for w in item["fsdp_wrappers"]
        )
        assert any(
            w["sequence_parallel_active_before_wrap"] for w in item["fsdp_wrappers"]
        )
        assert item["ulysses_distributed_attention_calls"] > 0
        assert item["ulysses_all_to_all_calls"] > 0
        assert item["barrier_calls"] > 0
        assert item["observer_final_barrier"] is True
        assert item["nccl_cumem_enable"] == "0"
        assert item["nccl_cumem_host_enable"] == "0"
        assert item["nccl_nvls_enable"] == "0"
        assert item["nccl_socket_ifname"] == "=eth0"
        assert item["nccl_socket_family"] == "AF_INET"
        assert item["nccl_ib_disable"] == "1"
        assert item["torch_nccl_use_comm_nonblocking"] == "1"

    assert (
        inventory["schema"] == "npa.workbench.byof.wan2_2_multigpu_runtime_inventory.v1"
    )
    assert inventory["non_root"] is True
    assert inventory["weights_baked"] is False
    assert inventory["customer_data_in_image"] is False
    assert len(inventory["devices"]) == 4
    assert all(
        device["compute_capability"] == [10, 0] for device in inventory["devices"]
    )
    assert inventory["package_versions"]["nvidia-nccl-cu13"] == "2.29.7"

    output = artifact["output"]
    video_key = key_prefix + output["filename"]
    head = s3.head_object(Bucket=out_bucket, Key=video_key)
    assert head["ContentLength"] == output["size_bytes"]
    assert head["ContentLength"] > 4096
    video_path = tmp_path / output["filename"]
    s3.download_file(out_bucket, video_key, str(video_path))
    assert hashlib.sha256(video_path.read_bytes()).hexdigest() == output["sha256"]
    stream = _decode_mp4(video_path)
    assert int(stream["width"]) == 1280
    assert int(stream["height"]) == 704
    assert int(stream["nb_read_frames"]) == 17
    observed = artifact["observed"]
    assert observed["codec"] == "h264"
    assert float(observed["fps"]) > 0
    assert float(observed["max_spatial_std"]) >= 1.0
    assert int(observed["pixel_range"]) >= 4
    assert float(observed["mean_temporal_abs_delta"]) > 0.001
    manifest = _verify_published_rrd(
        s3,
        bucket=out_bucket,
        key_prefix=key_prefix,
        layout=MULTI_GPU_LAYOUT,
        run_id=run_id,
        video_path=video_path,
        expected_frame_count=17,
        expected_fps=float(observed["fps"]),
        expected_rank_count=4,
        tmp_path=tmp_path,
    )
    assert manifest["variant"] == "multigpu"
