"""Contract and gated live GPU E2E for the LTX-2.5 BYOF solution spec.

The always-on tests plan the checked-in spec and pin the properties that keep
the zero-payload claim honest. The live test consumes a separately built,
byte-scanned image by immutable digest, runs it on a GPU, and verifies the
published evidence — the decoded MP4, re-checked from the operator side.

Live gates (all required):

* ``NPA_INTEGRATION_E2E=1``
* ``NPA_LTX2_LIVE_GPU=1``
* ``NPA_LTX2_REUSE_IMAGE`` pinned to ``…@sha256:…``
* ``HF_TOKEN`` with access to the gated ``Lightricks/LTX-2.5`` repository
* normal NPA project, registry, Kubernetes, and S3 operator configuration

The token is read and never written. It is the operator's own entitlement,
granted by Lightricks after that operator accepted the terms on the gated
repository page, and it is what both fetches require — so a run without one
refuses in the pod exactly as it does here.

Status: no live run has happened. The image has not been built. See
``docs/workbench/ltx2.md`` for the runbook that produces the evidence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.workbench.ltx2.video_check import validate_video
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)

from .npa_workflow_live_helpers import live_bucket

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
LTX_SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "byof-ltx2.yaml"
)
PROFILE_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "byof" / "profiles"
EXPECTED_CAPABILITIES = {
    "ltx2_5_text_to_video",
    "ltx2_5_decoded_mp4_validation",
}


def _spec_payload() -> dict[str, object]:
    payload = yaml.safe_load(LTX_SPEC.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _spec_config() -> dict[str, object]:
    config = _spec_payload().get("config")
    assert isinstance(config, dict)
    return config


def _planned_byof_args(run_id: str) -> dict[str, str | bool]:
    """Return the planner-rendered BYOF arguments for the generate state."""

    from npa.orchestration.npa_workflow import build_plan, load_spec

    steps = build_plan(load_spec(LTX_SPEC), run_id=run_id).to_dict().get("steps") or []
    generate = next(
        step for step in steps if step.get("tool_ref") == "workbench.byof.repo"
    )
    argv = [str(part) for part in (generate.get("argv") or [])]
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


def test_ltx2_spec_plans_the_real_pinned_gpu_workload() -> None:
    from npa.orchestration.npa_workflow import build_plan, load_spec

    spec = load_spec(LTX_SPEC)
    plan = build_plan(spec, run_id="ltx2-render-check")
    steps = plan.to_dict().get("steps") or []
    generate = next(
        step for step in steps if step.get("tool_ref") == "workbench.byof.repo"
    )
    rendered = " ".join(str(part) for part in (generate.get("argv") or []))
    config = _spec_config()

    assert config["repo_url"] == "https://github.com/Lightricks/LTX-2.git"
    assert config["repo_ref"] == "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"
    assert config["base_profile"] == "prebuilt"
    assert config["base_image"] == "tool://ltx2"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-ltx2-rtxpro-gpu"
    assert "--wait-timeout -1" in rendered
    assert "ltx_pipelines.distilled" in rendered
    assert "video_check.validate_video(" in rendered
    for capability in EXPECTED_CAPABILITIES:
        assert capability in rendered

    encoded_prompt = base64.b64encode(str(config["prompt"]).encode()).decode()
    assert encoded_prompt in rendered
    assert str(config["prompt"]) not in rendered
    assert "{{config." not in rendered

    payload = _spec_payload()
    resources = payload["resources"]["gpu"]
    profile = PROFILE_DIR / f"{config['resource_profile_yaml']}.yaml"
    assert profile.is_file()
    profile_text = profile.read_text(encoding="utf-8")
    assert resources["accelerators"] in profile_text
    # The weight set alone is ~66 GiB, so a default 100 GB disk would fail the
    # pull rather than the generation, which is a much less obvious failure.
    assert resources["disk_size"] == 500
    assert "disk_size: 500" in profile_text


# The negative control that used to live here — "this file must never accept a
# vendor's terms" — is now
# `tests/guardrails/test_live_tests_never_declare_a_licence.py`, which applies it
# to every live test rather than to this one, and catches the dict-literal and
# `env=` shapes the local version missed.


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_LTX2_LIVE_GPU") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_LTX2_LIVE_GPU=1, plus your own "
        "HF_TOKEN with access to Lightricks/LTX-2.5, to run the LTX-2.5 GPU "
        "smoke."
    ),
)
@pytest.mark.e2e
def test_ltx2_live_gpu_generate_and_decode(
    e2e_project: str | None,
    tmp_path: Path,
) -> None:
    registry = resolve_container_registry(e2e_project)
    assert registry, "NPA container registry could not be resolved"

    # The operator's own entitlement, read and never written. Its absence fails
    # the test here for the same reason the container refuses: Lightricks grants
    # it to them, on the gated repository page, and it covers the source as well
    # as the weights. Run `npa workbench ltx2 terms` first.
    assert os.environ.get("HF_TOKEN", "").strip(), (
        "Lightricks/LTX-2.5 is a gated repository; export your own HF_TOKEN."
    )

    reuse_image = os.environ.get("NPA_LTX2_REUSE_IMAGE", "").strip()
    assert reuse_image, (
        "the live run requires an explicitly digest-pinned, byte-scanned image"
    )
    assert reuse_image.startswith(registry.rstrip("/") + "/"), reuse_image
    assert re.search(r"@sha256:[0-9a-f]{64}$", reuse_image), reuse_image

    run_id = "byof-ltx2-e2e-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    planned = _planned_byof_args(run_id)
    profile = PROFILE_DIR / f"{planned['--yaml']}.yaml"
    out_bucket = live_bucket(e2e_project)
    output_root = f"s3://{out_bucket}/oss-solutions/ltx2"
    key_prefix = f"oss-solutions/ltx2/{run_id}/"

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
        reuse_image,
        "--build-command",
        str(planned["--build-command"]),
        "--project",
        e2e_project or "",
        "--image",
        reuse_image,
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
    assert runner["image"] == reuse_image
    assert runner["build"] == {"ok": True, "skipped": True}

    s3 = _s3_client(e2e_project)
    artifact_name = str(planned["--smoke-artifact-name"])
    summary = _read_s3_json(s3, out_bucket, key_prefix + "npa_byof_summary.json")
    artifact = _read_s3_json(s3, out_bucket, key_prefix + artifact_name)

    assert summary["status"] == "success"
    assert summary["solution_name"] == "ltx2.5"
    assert summary["smoke_exit_code"] == 0
    assert summary["image"] == reuse_image
    assert summary["metadata"]["ref"] == planned["--repo-ref"]

    assert artifact["solution"] == "ltx2.5"
    assert set(artifact["capabilities_exercised"]) == EXPECTED_CAPABILITIES
    assert artifact["deferred"] == []
    # The zero-payload claim, restated by the run that used the image.
    assert artifact["weights_baked"] is False
    assert artifact["source_baked"] is False

    # Decode the published pixels with the same module the pod used, from the
    # operator side, so a passing in-pod check cannot be the only evidence.
    evidence = artifact["evidence"]["video"]
    video_key = key_prefix + "ltx2_5_text_to_video.mp4"
    video_head = s3.head_object(Bucket=out_bucket, Key=video_key)
    assert video_head["ContentLength"] == evidence["size_bytes"]
    video_path = tmp_path / "ltx2_5_text_to_video.mp4"
    s3.download_file(out_bucket, video_key, str(video_path))
    assert evidence["sha256"] == hashlib.sha256(video_path.read_bytes()).hexdigest()
    local = validate_video(video_path, min_frames=24)
    assert local.frame_count == evidence["frame_count"]
    assert local.codec == evidence["codec"]

    # The in-run proof that the image refuses without an entitlement, published
    # alongside the video. It ran before either fetch, on these exact bytes.
    refusal = s3.get_object(
        Bucket=out_bucket, Key=key_prefix + "ltx2_5_entitlement_refusal.txt"
    )["Body"].read()
    assert b"NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK" in refusal
