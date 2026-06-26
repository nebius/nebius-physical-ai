from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


NPA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = NPA_ROOT.parent
MANIFEST_PATH = NPA_ROOT / "manifests" / "workbench" / "demo-8gpu-h200.yaml"


@pytest.fixture(scope="module")
def demo_stage_bucket(e2e_module_test_bucket: str) -> str:
    return e2e_module_test_bucket


@pytest.fixture
def demo_manifest(
    demo_stage_bucket: str,
    s3_helper,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    source_prefix = "demo-prestage-fixtures"
    output_path = tmp_path_factory.mktemp("demo-stage") / "demo-live-manifest.yaml"
    manifest = _materialize_demo_manifest(
        source_bucket=demo_stage_bucket,
        source_prefix=source_prefix,
        s3_helper=s3_helper,
    )
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {"path": output_path, "payload": manifest}


@pytest.mark.e2e
def test_demo_stage_stages_all_artifacts(
    demo_stage_bucket: str,
    demo_manifest: dict[str, Any],
    e2e_project: str | None,
    s3_helper,
) -> None:
    """`npa demo stage` uploads all manifest artifacts to the target bucket."""
    result = _run_demo_stage(
        demo_stage_bucket, e2e_project, demo_manifest["path"], output="json"
    )
    assert result.returncode == 0, _format_result(result)

    staged = json.loads(result.stdout)
    assert staged["status"] == "ok"
    assert {item["name"] for item in staged["artifacts"]} == {
        artifact["name"] for artifact in demo_manifest["payload"]["artifacts"]
    }

    for artifact in demo_manifest["payload"]["artifacts"]:
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
    result = _run_demo_stage(demo_stage_bucket, e2e_project, demo_manifest["path"])
    assert result.returncode == 0, _format_result(result)

    for artifact in demo_manifest["payload"]["artifacts"]:
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
    first = _run_demo_stage(
        demo_stage_bucket, e2e_project, demo_manifest["path"], output="json"
    )
    assert first.returncode == 0, _format_result(first)

    first_etags = {}
    for artifact in demo_manifest["payload"]["artifacts"]:
        if artifact.get("is_prefix"):
            continue
        head = s3_helper.head_object(demo_stage_bucket, artifact["target_path"])
        assert head is not None, artifact["name"]
        first_etags[artifact["name"]] = head["ETag"]

    second = _run_demo_stage(
        demo_stage_bucket, e2e_project, demo_manifest["path"], output="json"
    )
    assert second.returncode == 0, _format_result(second)
    second_payload = json.loads(second.stdout)
    assert all(item["action"] == "skip" for item in second_payload["artifacts"])

    for artifact in demo_manifest["payload"]["artifacts"]:
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
    stage = _run_demo_stage(demo_stage_bucket, e2e_project, demo_manifest["path"])
    assert stage.returncode == 0, _format_result(stage)

    verify = _run_demo_verify(demo_stage_bucket, e2e_project, demo_manifest["path"])
    assert verify.returncode == 0, _format_result(verify)

    file_artifact = next(
        artifact
        for artifact in demo_manifest["payload"]["artifacts"]
        if not artifact.get("is_prefix")
    )
    s3_helper.client.delete_object(
        Bucket=demo_stage_bucket,
        Key=file_artifact["target_path"],
    )

    verify_missing = _run_demo_verify(
        demo_stage_bucket, e2e_project, demo_manifest["path"]
    )
    assert verify_missing.returncode != 0, _format_result(verify_missing)


def _run_demo_stage(
    bucket: str,
    e2e_project: str | None,
    manifest_path: Path,
    *,
    output: str = "text",
) -> subprocess.CompletedProcess[str]:
    args = [
        "demo",
        "stage",
        "--target-bucket",
        bucket,
        "--manifest",
        str(manifest_path),
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


def _run_demo_verify(
    bucket: str,
    e2e_project: str | None,
    manifest_path: Path,
) -> subprocess.CompletedProcess[str]:
    args = [
        "demo",
        "verify",
        "--target-bucket",
        bucket,
        "--manifest",
        str(manifest_path),
    ]
    if e2e_project:
        args.extend(["--target-project", e2e_project])
    return _run_npa(args)


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


def _materialize_demo_manifest(
    *,
    source_bucket: str,
    source_prefix: str,
    s3_helper,
) -> dict[str, Any]:
    template = yaml.safe_load(MANIFEST_PATH.read_text())
    artifacts = template["artifacts"]
    rendered: list[dict[str, Any]] = []

    for artifact in artifacts:
        target_path = str(artifact["target_path"]).strip("/")
        source_uri = _source_uri(source_bucket, source_prefix, target_path, bool(artifact.get("is_prefix")))
        entry = dict(artifact)
        entry["source_uri"] = source_uri
        if artifact.get("is_prefix"):
            stats = _populate_prefix_artifact(s3_helper, source_bucket, source_prefix, target_path)
            entry["expected_count"] = stats["expected_count"]
            entry["total_size_bytes"] = stats["total_size_bytes"]
        else:
            payload = _payload_for_artifact(target_path)
            key = f"{source_prefix}/{target_path}"
            s3_helper.client.put_object(
                Bucket=source_bucket,
                Key=key,
                Body=payload,
                Metadata={"sha256": hashlib.sha256(payload).hexdigest()},
            )
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["size_bytes"] = len(payload)
        rendered.append(entry)

    file_artifacts = [entry for entry in rendered if not entry.get("is_prefix")]
    for entry in rendered:
        if not entry.get("is_prefix"):
            continue
        prefix = _ensure_prefix(str(entry["target_path"]).strip("/"))
        extra_count = 0
        extra_bytes = 0
        for file_entry in file_artifacts:
            file_path = str(file_entry["target_path"]).strip("/")
            if file_path.startswith(prefix):
                extra_count += 1
                extra_bytes += int(file_entry["size_bytes"])
        entry["expected_count"] = int(entry["expected_count"]) + extra_count
        entry["total_size_bytes"] = int(entry["total_size_bytes"]) + extra_bytes

    return {"version": template["version"], "artifacts": rendered}


def _populate_prefix_artifact(
    s3_helper,
    source_bucket: str,
    source_prefix: str,
    target_path: str,
) -> dict[str, int]:
    object_keys = [
        f"{source_prefix}/{target_path.rstrip('/')}/chunk-000.json",
        f"{source_prefix}/{target_path.rstrip('/')}/nested/chunk-001.json",
    ]
    total_size = 0
    for idx, key in enumerate(object_keys):
        payload = (
            json.dumps({"path": target_path, "index": idx}, sort_keys=True) + "\n"
        ).encode("utf-8")
        total_size += len(payload)
        s3_helper.client.put_object(Bucket=source_bucket, Key=key, Body=payload)
    return {"expected_count": len(object_keys), "total_size_bytes": total_size}


def _payload_for_artifact(target_path: str) -> bytes:
    return (
        json.dumps({"artifact": target_path, "kind": "demo-stage-fixture"}, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _source_uri(
    bucket: str,
    source_prefix: str,
    target_path: str,
    is_prefix: bool,
) -> str:
    leaf = f"{source_prefix}/{target_path.strip('/')}"
    if is_prefix and not leaf.endswith("/"):
        leaf = f"{leaf}/"
    return f"s3://{bucket}/{leaf}"


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
