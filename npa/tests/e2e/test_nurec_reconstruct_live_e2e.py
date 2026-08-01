"""Live NuRec/NRE reconstruction e2e: submit the real workflow, assert the run tree.

Gated on real infrastructure. Requires an RT-core GPU on the target Kubernetes
context, `NGC_API_KEY` (for the `nre-ga` container), `HF_TOKEN` (for the
PhysicalAI capture), and S3 credentials.

    NPA_INTEGRATION_E2E=1 \\
    NPA_NUREC_E2E_BUCKET=<bucket> \\
    NPA_NUREC_E2E_NPA_SRC_S3_URI=s3://<bucket>/npa-src/<tag> \\
    NPA_NUREC_E2E_INFRA=k8s/<rt-core-context> \\
    npa/.venv/bin/python -m pytest npa/tests/e2e/test_nurec_reconstruct_live_e2e.py -q

The assertions are the Definition of Done for the capability: a renderable USDZ,
real quality metrics, novel-view renders, and a Rerun recording that the agent's
run picker selects as the run's preferred artifact.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "nurec-reconstruct.yaml"
SPEC = (
    ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "nurec-reconstruct.yaml"
)
CATEGORY = "neural-reconstruction"
DEFAULT_IMAGE = "nvcr.io/nvidia/nre/nre-ga:26.04"
#: Objects the run MUST publish for the capability to be considered delivered.
REQUIRED_SUFFIXES = (
    "/ncore/manifest.json",
    "/reconstruction/last.usdz",
    "/reconstruction/metrics.yaml",
    "/reports/sim2real.rrd",
    "/reports/final.json",
)


def _require(name: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    if not value:
        pytest.skip(f"{name} is required for the live NuRec e2e")
    return value


def _npa_bin() -> str:
    candidate = ROOT / "npa" / ".venv" / "bin" / "npa"
    return str(candidate) if candidate.exists() else "npa"


def _s3_client():
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=_require("AWS_ENDPOINT_URL"),
        aws_access_key_id=_require("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-north1"),
        config=Config(signature_version="s3v4"),
    )


def _list_run_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)
        keys.extend(str(item["Key"]) for item in response.get("Contents") or ())
        if not response.get("IsTruncated"):
            return keys
        token = response.get("NextContinuationToken")


def test_nurec_reconstruct_publishes_a_viewable_run(tmp_path: Path) -> None:
    if os.environ.get("NPA_INTEGRATION_E2E", "") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 to run the live NuRec e2e")

    bucket = _require("NPA_NUREC_E2E_BUCKET")
    npa_src = _require("NPA_NUREC_E2E_NPA_SRC_S3_URI")
    infra = os.environ.get("NPA_NUREC_E2E_INFRA", "").strip()
    base_prefix = os.environ.get("NPA_NUREC_E2E_PREFIX", "checkpoints").strip().strip("/")
    _require("NGC_API_KEY")
    _require("HF_TOKEN")
    endpoint = _require("AWS_ENDPOINT_URL")

    # The run id must embed the submit timestamp: the agent's run picker dates the
    # run from it (npa.workflows.artifacts._run_started_at).
    stamp = time.strftime("%Y%m%dt%H%M%S", time.gmtime())
    run_id = f"{CATEGORY}-struktur28-{stamp}z"
    run_uri = f"s3://{bucket}/{base_prefix}/{CATEGORY}/{run_id}".replace("//", "/").replace(
        "s3:/", "s3://"
    )

    command = [
        _npa_bin(),
        "workbench",
        "workflow",
        "submit",
        str(WORKFLOW),
        "--run-id",
        run_id,
        "--var",
        f"NPA_NUREC_IMAGE={os.environ.get('NPA_NUREC_E2E_IMAGE', DEFAULT_IMAGE)}",
        "--var",
        f"NPA_NUREC_RUN_ID={run_id}",
        "--var",
        f"NPA_NUREC_RUN_URI={run_uri}",
        "--var",
        f"NPA_SRC_S3_URI={npa_src}",
        "--var",
        f"AWS_ENDPOINT_URL={endpoint}",
        "--var",
        f"AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']}",
        "--var",
        f"AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}",
        "--var",
        f"HF_TOKEN={os.environ['HF_TOKEN']}",
        "--var",
        f"NGC_API_KEY={os.environ['NGC_API_KEY']}",
    ]
    if infra:
        command.extend(["--infra", infra])

    submit = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    evidence = tmp_path / "submit.log"
    evidence.write_text((submit.stdout or "") + (submit.stderr or ""), encoding="utf-8")
    assert submit.returncode == 0, f"submit failed; see {evidence}"
    # Never let a token reach the captured evidence.
    for secret in ("NGC_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        assert os.environ[secret] not in evidence.read_text(encoding="utf-8"), secret

    # The reconstruction itself is the long pole: ~14 GB image pull on a cold node,
    # then 30k 3DGUT steps, then the render pass and the upload. No budget cap is
    # imposed beyond this generous deadline.
    deadline = time.time() + float(os.environ.get("NPA_NUREC_E2E_MAX_WAIT_SECONDS", "7200"))
    s3 = _s3_client()
    prefix = f"{base_prefix}/{CATEGORY}/{run_id}/" if base_prefix else f"{CATEGORY}/{run_id}/"
    keys: list[str] = []
    while time.time() < deadline:
        keys = _list_run_keys(s3, bucket, prefix)
        if any(key.endswith("/reports/final.json") for key in keys):
            break
        time.sleep(30)

    missing = [suffix for suffix in REQUIRED_SUFFIXES if not any(k.endswith(suffix) for k in keys)]
    assert not missing, f"run {run_id} is missing {missing}; published: {sorted(keys)[:40]}"
    assert any("/novel_views/" in key and key.endswith(".png") for key in keys), (
        "the run published no novel-view frames"
    )

    # The agent must see the run as viewable, dated by its start, with the Rerun
    # recording auto-selected.
    from npa.workflows.artifacts import list_artifacts, list_runs, select_preferred_artifact

    category_prefix = f"{base_prefix}/{CATEGORY}" if base_prefix else CATEGORY
    page = list_runs(bucket, prefix=category_prefix, s3=s3)
    summary = next((item for item in page.runs if item.run_id == run_id), None)
    assert summary is not None, f"{run_id} not listed under {category_prefix}"
    assert summary.has_viewable is True
    # started_at must be the run-id-encoded SUBMIT time, not the newest artifact
    # write -- that is the whole point of embedding a timestamp in the run id.
    # Verified against a real run: id ...t051728z -> 2026-07-31T05:17:28+00:00 while
    # last_modified was 05:44:39.
    encoded = (
        f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
        f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}+00:00"
    )
    assert summary.started_at == encoded, f"{summary.started_at} != {encoded}"
    assert summary.started_at < summary.last_modified

    artifacts = list_artifacts(bucket, run_id, prefix=category_prefix, s3=s3)
    preferred = select_preferred_artifact(artifacts)
    assert preferred is not None
    assert preferred.key.endswith("/reports/sim2real.rrd")
    assert preferred.render == "rerun"

    # The recording must be real run data, never mistakable for the stock demo.
    from npa.cli.agent_recordings import is_stock_demo_recording, recording_has_run_entities

    local_rrd = tmp_path / "sim2real.rrd"
    s3.download_file(bucket, preferred.key, str(local_rrd))
    data = local_rrd.read_bytes()
    assert recording_has_run_entities(data) is True
    assert is_stock_demo_recording(data) is False
    assert b"novel_view" in data

    # And the run's own report must agree.
    report_key = next(key for key in keys if key.endswith("/reports/final.json"))
    local_report = tmp_path / "final.json"
    s3.download_file(bucket, report_key, str(local_report))
    report = json.loads(local_report.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["capability"] == CATEGORY
    assert report["has_rrd"] is True
    assert report["has_usdz"] is True
    assert report["has_novel_views"] is True


def test_nurec_declarative_spec_runs_multi_step_on_real_gpus(tmp_path: Path) -> None:
    """The declarative npa.workflow twin must run end to end, not just render.

    This is a materially different execution path from the single-pod SkyPilot
    task above: every state becomes its OWN pod, so nothing survives in /tmp and
    the NCore sequence and the trained USDZ have to travel through S3. That
    cross-pod handoff is the thing this test exists to protect -- a rendering-only
    check cannot see it fail.
    """
    if os.environ.get("NPA_INTEGRATION_E2E", "") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 to run the live NuRec e2e")

    bucket = _require("NPA_NUREC_E2E_BUCKET")
    npa_src = _require("NPA_NUREC_E2E_NPA_SRC_S3_URI")
    infra = os.environ.get("NPA_NUREC_E2E_INFRA", "").strip()
    base_prefix = os.environ.get("NPA_NUREC_E2E_PREFIX", "checkpoints").strip().strip("/")
    for name in ("NGC_API_KEY", "HF_TOKEN", "AWS_ENDPOINT_URL"):
        _require(name)

    stamp = time.strftime("%Y%m%dt%H%M%S", time.gmtime())
    run_id = f"nurec-npa-{stamp}z"
    prefix = f"{base_prefix}/{CATEGORY}/{run_id}" if base_prefix else f"{CATEGORY}/{run_id}"

    command = [
        _npa_bin(),
        "workbench",
        "workflow",
        "submit",
        str(SPEC),
        "--run-id",
        run_id,
        "--var",
        f"bucket={bucket}",
        "--var",
        f"prefix={prefix}",
        # The declarative path forwards credentials as SkyPilot secrets rather than
        # substituting them into the YAML.
        "--secret-env",
        "AWS_ACCESS_KEY_ID",
        "--secret-env",
        "AWS_SECRET_ACCESS_KEY",
        "--secret-env",
        "HF_TOKEN",
        "--secret-env",
        "NGC_API_KEY",
    ]
    if infra:
        command.extend(["--infra", infra])

    env = dict(os.environ, NPA_SRC_S3_URI=npa_src)
    submit = subprocess.run(command, capture_output=True, text=True, timeout=3600, env=env)
    evidence = tmp_path / "submit-declarative.log"
    evidence.write_text((submit.stdout or "") + (submit.stderr or ""), encoding="utf-8")
    assert submit.returncode == 0, f"submit failed; see {evidence}"
    for secret in ("NGC_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        assert os.environ[secret] not in evidence.read_text(encoding="utf-8"), secret

    deadline = time.time() + float(os.environ.get("NPA_NUREC_E2E_MAX_WAIT_SECONDS", "7200"))
    s3 = _s3_client()
    scan_prefix = f"{prefix}/"
    keys: list[str] = []
    while time.time() < deadline:
        keys = _list_run_keys(s3, bucket, scan_prefix)
        if any(key.endswith("/reports/final.json") for key in keys):
            break
        time.sleep(30)

    missing = [s for s in REQUIRED_SUFFIXES if not any(k.endswith(s) for k in keys)]
    assert not missing, f"{run_id} is missing {missing}; published: {sorted(keys)[:40]}"

    # The cross-pod handoff specifically: fetch must have published the WHOLE
    # sequence (meta-file + every shard, symlinks resolved) for reconstruct to
    # materialize in a different pod.
    sequence = [k for k in keys if "/ncore/sequence/" in k]
    assert sequence, "fetch did not publish the NCore sequence for the next pod"
    assert any(k.endswith(".json") for k in sequence), "sequence has no meta-file"
    assert sum(1 for k in sequence if k.endswith(".itar")) >= 2, (
        f"sequence looks incomplete, only: {sorted(sequence)}"
    )
    # ...and the later stages actually consumed it.
    assert any("/novel_views/" in k and k.endswith(".png") for k in keys)

    from npa.workflows.artifacts import list_artifacts, list_runs, select_preferred_artifact

    category_prefix = f"{base_prefix}/{CATEGORY}" if base_prefix else CATEGORY
    summary = next(
        (r for r in list_runs(bucket, prefix=category_prefix, s3=s3).runs if r.run_id == run_id),
        None,
    )
    assert summary is not None, f"{run_id} not listed under {category_prefix}"
    assert summary.has_viewable is True
    encoded = (
        f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
        f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}+00:00"
    )
    assert summary.started_at == encoded, f"{summary.started_at} != {encoded}"

    preferred = select_preferred_artifact(list_artifacts(bucket, run_id, prefix=category_prefix, s3=s3))
    assert preferred is not None
    assert preferred.key.endswith("/reports/sim2real.rrd")

    from npa.cli.agent_recordings import is_stock_demo_recording, recording_has_run_entities

    local_rrd = tmp_path / "declarative.rrd"
    s3.download_file(bucket, preferred.key, str(local_rrd))
    data = local_rrd.read_bytes()
    assert recording_has_run_entities(data) is True
    assert is_stock_demo_recording(data) is False

