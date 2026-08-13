"""Contract and gated live GPU E2E for the LTX-2.5 BYOF solution spec.

The always-on tests plan the checked-in spec and pin the properties that make
the licence control real. The live test consumes a separately built, byte-scanned
image by immutable digest, runs it on a GPU, and verifies the published evidence
— including that the artifacts carry a provenance manifest and that
``npa workbench ltx2 gate`` reaches the disposition the operator declared.

Live gates (all required):

* ``NPA_INTEGRATION_E2E=1``
* ``NPA_LTX2_LIVE_GPU=1``
* ``NPA_LTX2_REUSE_IMAGE`` pinned to ``…@sha256:…``
* ``HF_TOKEN`` with access to the gated ``Lightricks/LTX-2.5`` repository
* the operator's own LTX-2.x declaration in the environment
* normal NPA project, registry, Kubernetes, and S3 operator configuration

The declaration is deliberately *not* set by this test. A test that exported
``NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES`` would be Nebius accepting Lightricks'
terms on the operator's behalf, which is the one thing the whole gate exists to
prevent — so the live test refuses to run instead, exactly as the container does.

Status: no live run has happened. The image has not been built. See
``docs/workbench/ltx2.md`` for the runbook that produces the evidence.
"""

from __future__ import annotations

import ast
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
from npa.workbench.ltx2.licensing import (
    ACCEPT_ENV,
    ENTITY_CLASS_ENV,
    PROVENANCE_SCHEMA,
    TRAINING_NON_COMMERCIAL_ONLY,
    TRAINING_PROHIBITED,
    USE_CLASS_ENV,
    USE_COMMERCIAL,
)
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
    "ltx2_5_license_gate_refusal",
    "ltx2_5_license_provenance_stamp",
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


DECLARATION_ENVS = {ACCEPT_ENV, ENTITY_CLASS_ENV, USE_CLASS_ENV}
DECLARATION_ALIASES = {"ACCEPT_ENV", "ENTITY_CLASS_ENV", "USE_CLASS_ENV"}
ENV_WRITE_METHODS = {"setenv", "setdefault", "putenv", "update"}


def _declaration_env(node: ast.expr) -> str | None:
    """Return the declaration variable a node names, whether spelled or aliased."""

    if isinstance(node, ast.Constant) and node.value in DECLARATION_ENVS:
        return str(node.value)
    if isinstance(node, ast.Name) and node.id in DECLARATION_ALIASES:
        return node.id
    return None


def test_the_live_tier_never_declares_on_the_operators_behalf() -> None:
    """Negative control: this file must never *write* a declaration variable.

    Exporting the operator's answers here would make every live run technically
    valid and legally meaningless: Nebius would be accepting Lightricks' terms
    on the operator's behalf, which is the one thing the gate exists to prevent.
    Reads are fine and necessary — the live test asserts on what the operator
    already declared — so this walks the AST for writes rather than grepping
    text, which would only catch the spelling a future edit happens to use.
    """

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    written: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                name = _declaration_env(target.slice)
                if name:
                    written.append(name)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ENV_WRITE_METHODS
            and node.args
        ):
            name = _declaration_env(node.args[0])
            if name:
                written.append(name)

    assert written == [], (
        f"this test assigns the operator's LTX declaration ({sorted(set(written))}); "
        "only the operator may answer those"
    )


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_LTX2_LIVE_GPU") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_LTX2_LIVE_GPU=1, plus your own "
        "LTX-2.x declaration and HF_TOKEN, to run the LTX-2.5 GPU smoke."
    ),
)
@pytest.mark.e2e
def test_ltx2_live_gpu_generate_gate_and_decode(
    e2e_project: str | None,
    tmp_path: Path,
) -> None:
    config = _spec_config()
    registry = resolve_container_registry(e2e_project)
    assert registry, "NPA container registry could not be resolved"

    # The operator's own answers, read and never written. Missing ones fail the
    # test here for the same reason the container refuses: nobody else may
    # answer them.
    declared_use = (os.environ.get(USE_CLASS_ENV) or "").strip().lower()
    assert (os.environ.get(ACCEPT_ENV) or "").strip().upper() == "YES", (
        f"{ACCEPT_ENV} must be YES: you, not this test, accept the LTX-2.x "
        "Community License Agreement. Run `npa workbench ltx2 terms` first."
    )
    assert (os.environ.get(ENTITY_CLASS_ENV) or "").strip(), ENTITY_CLASS_ENV
    assert declared_use, USE_CLASS_ENV
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

    licence = artifact["license"]
    assert licence["schema"] == PROVENANCE_SCHEMA
    assert licence["source"]["ref"] == planned["--repo-ref"]
    assert licence["license"]["osi_approved"] is False
    expected_disposition = (
        TRAINING_PROHIBITED
        if declared_use == USE_COMMERCIAL
        else TRAINING_NON_COMMERCIAL_ONLY
    )
    assert licence["restrictions"]["derived_model_training"] == expected_disposition

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

    # Finally the part that is not about video at all: the gate must reach the
    # declared disposition, and must fail the workflow when it is prohibited.
    manifest_uri = f"s3://{out_bucket}/{key_prefix}ltx2_provenance.json"
    stamp = subprocess.run(
        [
            "npa",
            "workbench",
            "ltx2",
            "stamp",
            "--run-id",
            run_id,
            "--manifest-uri",
            manifest_uri,
            "--output-uri",
            f"s3://{out_bucket}/{video_key}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert stamp.returncode == 0, stamp.stderr
    gate = subprocess.run(
        [
            "npa",
            "workbench",
            "ltx2",
            "gate",
            "--manifest-uri",
            manifest_uri,
            "--consumer",
            "LeRobot policy training",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if declared_use == USE_COMMERCIAL:
        assert gate.returncode != 0, (
            "a commercial declaration must stop the trainer; Attachment A(18)"
        )
    else:
        assert gate.returncode == 0, gate.stderr
        assert json.loads(gate.stdout)["allowed"] is True
