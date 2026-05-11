from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


NPA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = NPA_ROOT.parent
MANIFEST_PATH = NPA_ROOT / "manifests" / "demo-8gpu-h200.yaml"


@pytest.fixture(scope="module")
def demo_stage_bucket(e2e_module_test_bucket: str) -> str:
    return e2e_module_test_bucket


@pytest.fixture(scope="module")
def demo_manifest() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST_PATH.read_text())


@pytest.mark.e2e
def test_demo_stage_stages_all_artifacts(
    demo_stage_bucket: str,
    demo_manifest: dict[str, Any],
    e2e_project: str | None,
    s3_helper,
) -> None:
    """`npa demo stage` uploads all manifest artifacts to the target bucket."""
    result = _run_demo_stage(demo_stage_bucket, e2e_project, output="json")
    assert result.returncode == 0, _format_result(result)

    staged = json.loads(result.stdout)
    assert staged["status"] == "ok"
    assert {item["name"] for item in staged["artifacts"]} == {
        artifact["name"] for artifact in demo_manifest["artifacts"]
    }

    for artifact in demo_manifest["artifacts"]:
        if artifact.get("is_prefix"):
            objects = s3_helper.list_object_summaries(
                demo_stage_bucket,
                _ensure_prefix(artifact["target_path"]),
            )
            assert len(objects) == artifact["expected_count"], artifact["name"]
            assert sum(int(obj["Size"]) for obj in objects) == artifact[
                "total_size_bytes"
            ]
            continue

        head = s3_helper.head_object(demo_stage_bucket, artifact["target_path"])
        assert head is not None, (
            f"Artifact {artifact['name']!r} missing at {artifact['target_path']}"
        )
        assert int(head["ContentLength"]) == artifact["size_bytes"]


@pytest.mark.e2e
def test_demo_stage_writes_sha256_metadata(
    demo_stage_bucket: str,
    demo_manifest: dict[str, Any],
    e2e_project: str | None,
    s3_helper,
) -> None:
    """`npa demo stage` writes x-amz-meta-sha256 metadata from the manifest."""
    result = _run_demo_stage(demo_stage_bucket, e2e_project)
    assert result.returncode == 0, _format_result(result)

    for artifact in demo_manifest["artifacts"]:
        if artifact.get("is_prefix"):
            continue

        actual_sha = s3_helper.get_sha256_metadata(
            demo_stage_bucket,
            artifact["target_path"],
        )
        assert actual_sha == artifact["sha256"], (
            f"sha256 mismatch for {artifact['name']}: "
            f"expected {artifact['sha256']}, got {actual_sha}"
        )


@pytest.mark.e2e
def test_demo_stage_is_idempotent(
    demo_stage_bucket: str,
    demo_manifest: dict[str, Any],
    e2e_project: str | None,
    s3_helper,
) -> None:
    """A second `npa demo stage` against the same bucket is a no-op."""
    first = _run_demo_stage(demo_stage_bucket, e2e_project, output="json")
    assert first.returncode == 0, _format_result(first)

    first_etags = {}
    for artifact in demo_manifest["artifacts"]:
        if artifact.get("is_prefix"):
            continue
        head = s3_helper.head_object(demo_stage_bucket, artifact["target_path"])
        assert head is not None, artifact["name"]
        first_etags[artifact["name"]] = head["ETag"]

    second = _run_demo_stage(demo_stage_bucket, e2e_project, output="json")
    assert second.returncode == 0, _format_result(second)
    second_payload = json.loads(second.stdout)
    assert all(item["action"] == "skip" for item in second_payload["artifacts"])

    for artifact in demo_manifest["artifacts"]:
        if artifact.get("is_prefix"):
            continue
        head = s3_helper.head_object(demo_stage_bucket, artifact["target_path"])
        assert head is not None, artifact["name"]
        assert head["ETag"] == first_etags[artifact["name"]], (
            f"Artifact {artifact['name']!r} was re-uploaded on second stage"
        )


@pytest.mark.e2e
def test_demo_verify_agrees_with_demo_stage(
    demo_stage_bucket: str,
    demo_manifest: dict[str, Any],
    e2e_project: str | None,
    s3_helper,
) -> None:
    """After `demo stage`, `demo verify` succeeds and detects missing artifacts."""
    stage = _run_demo_stage(demo_stage_bucket, e2e_project)
    assert stage.returncode == 0, _format_result(stage)

    verify = _run_npa(["demo", "verify", "--target-bucket", demo_stage_bucket])
    assert verify.returncode == 0, _format_result(verify)

    file_artifact = next(
        artifact for artifact in demo_manifest["artifacts"] if not artifact.get("is_prefix")
    )
    s3_helper.client.delete_object(
        Bucket=demo_stage_bucket,
        Key=file_artifact["target_path"],
    )

    verify_missing = _run_npa(["demo", "verify", "--target-bucket", demo_stage_bucket])
    assert verify_missing.returncode != 0, _format_result(verify_missing)


def _run_demo_stage(
    bucket: str,
    e2e_project: str | None,
    *,
    output: str = "text",
) -> subprocess.CompletedProcess[str]:
    args = [
        "demo",
        "stage",
        "--target-bucket",
        bucket,
        "--manifest",
        str(MANIFEST_PATH),
        "--output",
        output,
    ]
    if e2e_project:
        args.extend(
            [
                "--source-project",
                e2e_project,
                "--target-project",
                e2e_project,
            ]
        )
    return _run_npa(args, timeout=600)


def _run_npa(
    args: list[str],
    *,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_npa_executable(), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _npa_executable() -> str:
    script = Path(sys.executable).with_name("npa")
    if script.exists():
        return str(script)
    return "npa"


def _ensure_prefix(key: str) -> str:
    return key if key.endswith("/") else f"{key}/"


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
