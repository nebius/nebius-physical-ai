"""Read-only, provider/S3-byte-verified checkpoint-selection evidence check.

Backend-generic: validates whichever real sim backend's held-out validation
report the operator ran, as long as it emits ``per_env`` rows with
``object_position_m``/``goal_position_m`` (real, terminal-step positions —
not a minimum-over-episode distance) and feeds
``checkpoint_selection.select_best_checkpoint``. Submits, cancels, and mocks
nothing; the real training + validation job is owned end to end by whichever
operator-run worker produced the evidence this test reads back.

What each check actually proves, precisely:
  - Job status/platform/gpu_count/image come live from
    ``ServerlessClient.get_job``, not a local file. The image must be
    digest-pinned (``@sha256:...``); a tag is rejected.
  - The job's ``SOURCE_SHA256``/``SOURCE_REVISION``/``WORKER_NAME`` env values
    (narrowly extracted, never dumped alongside other keys) must match the
    operator-supplied expectations, tying the launched job to a specific
    reviewed source archive and revision.
  - ``candidates.json``/``selection.json``/``validation.json`` are fetched
    fresh from S3 and sha256-checked against a freshly fetched
    ``hashes.json`` before being trusted.
  - The source archive's own bytes are downloaded and sha256-checked against
    ``expected_source_sha`` (not just checked for existence at a hash-named
    key), then its internal ``source-hashes.json`` manifest is checked
    against every other member's actual bytes in memory. This proves the
    archive is internally self-consistent and is the exact bytes the job's
    ``SOURCE_SHA256`` env names; it does not independently prove those bytes
    are what actually executed inside the container.
  - Every candidate's checkpoint is verified by downloading the real bytes
    at its ``checkpoint_uri`` (bounded to live under the operator-supplied
    training root) and hashing them, not by comparing self-reported hash
    strings to each other.
  - Every per-env row must carry exactly 3 finite coordinates for both
    ``object_position_m`` and ``goal_position_m`` and a finite stored
    distance; the recomputed Euclidean distance must match both the row and
    the candidate's reported mean. No row is silently skipped.
  - The winner (checkpoint hash and full rank key) is independently
    recomputed by calling the current ``select_best_checkpoint`` on the raw
    candidates, forward and reversed.

Opt in with ``NPA_E2E_CHECKPOINT_SELECTION_EVIDENCE_CONFIG`` pointing to an
owner-only (mode 0600) JSON file:
  project_alias: project alias for ``s3_client_for_project``.
  project_id / job_id: the exact provider job to observe.
  output_uri: s3://bucket/prefix the training job published under. The
    validation bundle is at ``<output_uri>/validation/``; the reviewed
    source archive is at
    ``<output_uri>/validation-source/<expected_source_sha>.tar.gz``.
  expected_image_digest: exact ``registry/repo@sha256:...`` reference.
  expected_source_sha / expected_source_revision / expected_worker_name:
    the job's expected ``SOURCE_SHA256``/``SOURCE_REVISION``/``WORKER_NAME``
    env values.
  expected_gpu_platform / expected_gpu_count: the job's expected
    ``JobInfo.platform``/``gpu_count``.
  min_episodes: minimum real held-out episodes required per candidate.

This test never logs or asserts against the full ``JobInfo``, its ``raw``
payload, or any job environment wholesale — only single narrowly-extracted
values ever reach an assertion.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

_REQUIRED_KEYS = (
    "project_alias",
    "project_id",
    "job_id",
    "output_uri",
    "expected_image_digest",
    "expected_source_sha",
    "expected_source_revision",
    "expected_worker_name",
    "expected_gpu_platform",
    "expected_gpu_count",
    "min_episodes",
)

_IMAGE_PATHS = (
    ("spec", "container", "image"),
    ("spec", "image"),
    ("container", "image"),
    ("image",),
)

_POSITION_DIMS = 3
_DISTANCE_ATOL = 1e-6


def _config() -> dict[str, Any]:
    path = os.environ.get("NPA_E2E_CHECKPOINT_SELECTION_EVIDENCE_CONFIG", "")
    if not path:
        pytest.skip("NPA_E2E_CHECKPOINT_SELECTION_EVIDENCE_CONFIG not set")
    config_file = Path(path)
    assert config_file.stat().st_mode & 0o077 == 0, "live config must be owner-only"
    config = json.loads(config_file.read_text())
    missing = [key for key in _REQUIRED_KEYS if key not in config]
    assert not missing, f"config missing required keys: {missing}"
    return config


def _job_image(raw: dict[str, Any]) -> str:
    """Extract only the container image reference; never return ``raw``."""

    for path in _IMAGE_PATHS:
        node: Any = raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, str) and node:
            return node
    return ""


def _job_env_value(raw: dict[str, Any], key: str) -> str:
    """Extract exactly one named env value; never enumerate other keys."""

    spec = raw.get("spec") or {}
    environment = spec.get("environment_variables")
    if environment is None:
        environment = (spec.get("container") or {}).get("environment")
    if isinstance(environment, dict):
        value = environment.get(key)
        return value if isinstance(value, str) else ""
    if isinstance(environment, list):
        for entry in environment:
            if isinstance(entry, dict) and entry.get("name") == key:
                value = entry.get("value")
                return value if isinstance(value, str) else ""
    return ""


def _require_digest_pinned(image: str) -> None:
    assert "@sha256:" in image, (
        "job image is not digest-pinned (no @sha256: reference); a mutable "
        "tag cannot prove which exact bytes ran"
    )


def _assert_job_env_matches(raw: dict[str, Any], expected: dict[str, str]) -> None:
    for key, expected_value in expected.items():
        actual = _job_env_value(raw, key)
        assert actual == expected_value, (
            f"job env {key} is {actual!r}, expected {expected_value!r}"
        )


def _assert_job_matches_expected_provider_state(
    info: Any, config: dict[str, Any]
) -> None:
    assert info.status == "succeeded", f"job status is {info.status!r}, not succeeded"
    assert info.platform == config["expected_gpu_platform"], (
        f"job platform is {info.platform!r}, expected "
        f"{config['expected_gpu_platform']!r}"
    )
    assert info.gpu_count == int(config["expected_gpu_count"]), (
        f"job gpu_count is {info.gpu_count!r}, expected "
        f"{config['expected_gpu_count']!r}"
    )
    image = _job_image(info.raw)
    _require_digest_pinned(image)
    assert image == config["expected_image_digest"], (
        "job image is not the expected digest"
    )
    _assert_job_env_matches(
        info.raw,
        {
            "SOURCE_SHA256": config["expected_source_sha"],
            "SOURCE_REVISION": config["expected_source_revision"],
            "WORKER_NAME": config["expected_worker_name"],
        },
    )


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.strip("/")


def _euclidean_distance_m(
    object_position_m: list[float], goal_position_m: list[float]
) -> float:
    return math.sqrt(
        sum((o - g) ** 2 for o, g in zip(object_position_m, goal_position_m))
    )


def _assert_finite_vector(vector: Any, *, name: str, env_id: Any) -> None:
    assert isinstance(vector, list) and len(vector) == _POSITION_DIMS, (
        f"{env_id}: {name} must have exactly {_POSITION_DIMS} coordinates, "
        f"got {vector!r}"
    )
    for value in vector:
        assert isinstance(value, (int, float)) and math.isfinite(value), (
            f"{env_id}: {name} has a non-finite coordinate: {value!r}"
        )


def _row_distance(row: dict[str, Any]) -> tuple[float, float]:
    """Return (geometry-recomputed distance, stored distance); both required."""

    env_id = row.get("env_id")
    details = row.get("details") or {}
    assert "object_position_m" in details and "goal_position_m" in details, (
        f"{env_id}: row is missing object_position_m/goal_position_m"
    )
    object_position = details["object_position_m"]
    goal_position = details["goal_position_m"]
    _assert_finite_vector(object_position, name="object_position_m", env_id=env_id)
    _assert_finite_vector(goal_position, name="goal_position_m", env_id=env_id)
    stored = details.get("object_goal_distance_m")
    assert stored is not None and math.isfinite(float(stored)), (
        f"{env_id}: object_goal_distance_m is missing or non-finite"
    )
    return _euclidean_distance_m(object_position, goal_position), float(stored)


def _assert_row_geometry_consistent(
    per_env: list[dict[str, Any]], *, atol: float = _DISTANCE_ATOL
) -> None:
    for row in per_env:
        recomputed, stored = _row_distance(row)
        assert abs(recomputed - stored) <= atol, (
            f"{row.get('env_id')}: stored distance {stored} disagrees with "
            f"geometry recomputed from raw positions ({recomputed})"
        )


def _recompute_mean_distance(per_env: list[dict[str, Any]]) -> float:
    distances = [_row_distance(row)[0] for row in per_env]
    assert distances, "per_env has no rows"
    return sum(distances) / len(distances)


def _assert_env_ids_well_formed(
    per_env: list[dict[str, Any]], *, min_count: int
) -> None:
    env_ids = [row.get("env_id") for row in per_env]
    assert len(set(env_ids)) == len(env_ids), "duplicate env_id within one candidate"
    assert len(env_ids) >= min_count, (
        f"candidate has {len(env_ids)} episodes, expected at least {min_count}"
    )


def _assert_env_ids_aligned(candidates: list[dict[str, Any]]) -> None:
    id_sets = {
        frozenset(row["env_id"] for row in c["validation_report"]["per_env"])
        for c in candidates
    }
    assert len(id_sets) == 1, "candidates were evaluated on different env_id sets"


def _assert_episode_count_matches_summary(
    candidate: dict[str, Any], validation: dict[str, Any]
) -> None:
    per_env = candidate["validation_report"]["per_env"]
    expected = validation["episodes_per_candidate"]
    assert len(per_env) == expected, (
        f"candidate has {len(per_env)} episodes, validation summary claims {expected}"
    )


def _assert_reported_mean_matches_geometry(candidate: dict[str, Any]) -> None:
    summary = candidate["validation_report"]["success_summary"]
    assert "mean_object_goal_distance_m" in summary, (
        "success_summary is missing mean_object_goal_distance_m"
    )
    reported = summary["mean_object_goal_distance_m"]
    assert reported is not None and math.isfinite(float(reported)), (
        "reported mean_object_goal_distance_m is missing or non-finite"
    )
    recomputed = _recompute_mean_distance(candidate["validation_report"]["per_env"])
    assert abs(recomputed - float(reported)) <= _DISTANCE_ATOL, (
        "reported mean distance disagrees with geometry recomputed from raw positions"
    )


def _assert_candidate_evidence(
    candidate: dict[str, Any], validation: dict[str, Any], *, min_episodes: int
) -> None:
    per_env = candidate["validation_report"]["per_env"]
    _assert_env_ids_well_formed(per_env, min_count=min_episodes)
    _assert_episode_count_matches_summary(candidate, validation)
    _assert_row_geometry_consistent(per_env)
    _assert_reported_mean_matches_geometry(candidate)


def _get_bytes(s3: Any, bucket: str, key: str) -> bytes:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def _hash_object(s3: Any, bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    hasher = hashlib.sha256()
    try:
        for chunk in iter(lambda: body.read(1 << 20), b""):
            hasher.update(chunk)
    finally:
        body.close()
    return hasher.hexdigest()


def _verify_recorded_hash(
    name: str, data: bytes, recorded_hashes: dict[str, str]
) -> None:
    expected = recorded_hashes.get(name)
    assert expected, f"hashes.json has no recorded hash for {name}"
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected, (
        f"{name}: downloaded bytes hash {actual} does not match recorded hash {expected}"
    )


def _load_verified_bundle(s3: Any, bucket: str, prefix: str) -> dict[str, Any]:
    prefix = prefix.rstrip("/")
    recorded_hashes = json.loads(_get_bytes(s3, bucket, f"{prefix}/hashes.json"))
    bundle: dict[str, Any] = {}
    for name in ("candidates.json", "selection.json", "validation.json"):
        data = _get_bytes(s3, bucket, f"{prefix}/{name}")
        _verify_recorded_hash(name, data, recorded_hashes)
        bundle[name] = json.loads(data)
    return bundle


def _assert_checkpoint_under_training_root(
    checkpoint_uri: str, training_root: str
) -> None:
    root = training_root.rstrip("/") + "/"
    assert checkpoint_uri.startswith(root), (
        f"checkpoint_uri {checkpoint_uri!r} is not under the expected training "
        f"root {training_root!r}"
    )


def _assert_checkpoint_bytes_match(
    s3: Any, checkpoint_uri: str, expected_sha256: str, *, training_root: str
) -> None:
    _assert_checkpoint_under_training_root(checkpoint_uri, training_root)
    bucket, key = _split_s3_uri(checkpoint_uri)
    actual = _hash_object(s3, bucket, key)
    assert actual == expected_sha256, (
        f"checkpoint at {checkpoint_uri!r} hashes to {actual}, "
        f"candidate claims {expected_sha256}"
    )


def _assert_distinct_checkpoint_hashes(
    s3: Any, candidates: list[dict[str, Any]], *, training_root: str
) -> None:
    hashes = [c["checkpoint_sha256"] for c in candidates]
    assert len(set(hashes)) == len(hashes), (
        "candidates do not have distinct checkpoint hashes"
    )
    for candidate in candidates:
        _assert_checkpoint_bytes_match(
            s3,
            candidate["checkpoint_uri"],
            candidate["checkpoint_sha256"],
            training_root=training_root,
        )


def _assert_archive_manifest_self_consistent(
    archive_bytes: bytes, *, expected_worker_name: str
) -> None:
    """Verify every manifested member's real bytes hash to its recorded value.

    Reads members via ``tarfile.extractfile`` (in-memory); nothing is
    written to disk. Proves the archive is internally self-consistent, not
    that it is what actually ran in the container.
    """

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        manifest_file = tar.extractfile(tar.getmember("source-hashes.json"))
        assert manifest_file is not None, "source-hashes.json is not a regular file"
        manifest = json.loads(manifest_file.read())
        assert expected_worker_name in manifest, (
            f"source manifest does not list the expected worker "
            f"{expected_worker_name!r}"
        )
        assert any(name.startswith("src/") for name in manifest), (
            "source manifest has no staged src/ modules"
        )
        for name, expected_hash in manifest.items():
            member_file = tar.extractfile(tar.getmember(name))
            assert member_file is not None, f"{name} is not a regular file"
            actual_hash = hashlib.sha256(member_file.read()).hexdigest()
            assert actual_hash == expected_hash, (
                f"archive member {name} does not match its recorded manifest hash"
            )


def _assert_source_archive_verified(
    s3: Any,
    bucket: str,
    root_prefix: str,
    *,
    expected_sha: str,
    expected_worker_name: str,
) -> None:
    key = f"{root_prefix.rstrip('/')}/validation-source/{expected_sha}.tar.gz"
    data = _get_bytes(s3, bucket, key)
    actual_sha = hashlib.sha256(data).hexdigest()
    assert actual_sha == expected_sha, (
        f"source archive at {key!r} hashes to {actual_sha}, expected {expected_sha}"
    )
    _assert_archive_manifest_self_consistent(
        data, expected_worker_name=expected_worker_name
    )


def _assert_validation_summary(
    validation: dict[str, Any], config: dict[str, Any]
) -> None:
    assert validation.get("gpu"), "no GPU device name recorded for the validation run"
    assert validation["full_pipeline_executed"] is False
    assert validation["source_revision"] == config["expected_source_revision"], (
        "validation summary's self-reported source_revision does not match "
        "the operator-expected revision"
    )


def _assert_selection_matches_recompute(
    candidates: list[dict[str, Any]],
    selection: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint

    recomputed = select_best_checkpoint(copy.deepcopy(candidates))
    assert recomputed["checkpoint_sha256"] == selection["checkpoint_sha256"]
    assert recomputed["checkpoint_sha256"] == validation["selected_sha256"]
    assert recomputed["rank_key"] == selection["rank_key"]
    assert recomputed["rank_key"] == validation["rank_key"]

    reversed_result = select_best_checkpoint(list(reversed(copy.deepcopy(candidates))))
    assert reversed_result["checkpoint_sha256"] == recomputed["checkpoint_sha256"]
    assert reversed_result["rank_key"] == recomputed["rank_key"]


@pytest.mark.gpu
@pytest.mark.public_inputs
def test_real_checkpoint_selection_evidence_is_conclusive() -> None:
    """Read-only: never submits, cancels, or mocks the job it observes."""

    config = _config()

    from npa.clients.project_credentials import s3_client_for_project
    from npa.clients.serverless import ServerlessClient

    info = ServerlessClient().get_job(config["job_id"], config["project_id"])
    _assert_job_matches_expected_provider_state(info, config)

    s3 = s3_client_for_project(config["project_alias"])
    bucket, root_prefix = _split_s3_uri(config["output_uri"])
    bundle = _load_verified_bundle(s3, bucket, f"{root_prefix}/validation")
    _assert_source_archive_verified(
        s3,
        bucket,
        root_prefix,
        expected_sha=config["expected_source_sha"],
        expected_worker_name=config["expected_worker_name"],
    )

    candidates = bundle["candidates.json"]
    selection = bundle["selection.json"]
    validation = bundle["validation.json"]
    assert len(candidates) >= 2, "need at least two candidates to prove a real ranking"

    _assert_validation_summary(validation, config)
    _assert_env_ids_aligned(candidates)
    for candidate in candidates:
        _assert_candidate_evidence(
            candidate, validation, min_episodes=int(config["min_episodes"])
        )
    _assert_distinct_checkpoint_hashes(
        s3, candidates, training_root=config["output_uri"]
    )
    _assert_selection_matches_recompute(candidates, selection, validation)


def _build_test_archive(files: dict[str, bytes], manifest: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in {
            **files,
            "source-hashes.json": json.dumps(manifest).encode(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_config_rejects_group_or_world_readable_file(tmp_path) -> None:
    config_file = tmp_path / "evidence.json"
    config_file.write_text("{}")
    config_file.chmod(0o644)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "NPA_E2E_CHECKPOINT_SELECTION_EVIDENCE_CONFIG", str(config_file)
        )
        with pytest.raises(AssertionError, match="owner-only"):
            _config()


def test_config_rejects_missing_required_keys(tmp_path) -> None:
    config_file = tmp_path / "evidence.json"
    config_file.write_text(json.dumps({"project_alias": "p"}))
    config_file.chmod(0o600)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "NPA_E2E_CHECKPOINT_SELECTION_EVIDENCE_CONFIG", str(config_file)
        )
        with pytest.raises(AssertionError, match="missing required keys"):
            _config()


def test_job_image_prefers_nested_container_path() -> None:
    raw = {
        "spec": {"container": {"image": "registry/repo@sha256:aaa"}, "image": "other"}
    }
    assert _job_image(raw) == "registry/repo@sha256:aaa"


def test_job_image_returns_empty_when_absent() -> None:
    assert _job_image({"spec": {}}) == ""


def test_require_digest_pinned_rejects_tag_only_image() -> None:
    with pytest.raises(AssertionError, match="not digest-pinned"):
        _require_digest_pinned("registry/repo:latest")


def test_job_env_value_extracts_from_provider_environment_variables() -> None:
    raw = {
        "spec": {"environment_variables": [{"name": "SOURCE_REVISION", "value": "abc"}]}
    }
    assert _job_env_value(raw, "SOURCE_REVISION") == "abc"


def test_job_env_value_extracts_from_list_shaped_environment() -> None:
    raw = {
        "spec": {
            "container": {"environment": [{"name": "SOURCE_REVISION", "value": "abc"}]}
        }
    }
    assert _job_env_value(raw, "SOURCE_REVISION") == "abc"


def test_job_env_value_extracts_from_dict_shaped_environment() -> None:
    raw = {"spec": {"container": {"environment": {"WORKER_NAME": "worker.py"}}}}
    assert _job_env_value(raw, "WORKER_NAME") == "worker.py"


def test_job_env_value_returns_empty_for_unknown_key() -> None:
    raw = {"spec": {"container": {"environment": {"OTHER": "x"}}}}
    assert _job_env_value(raw, "SOURCE_SHA256") == ""


def test_job_env_mismatch_is_detected() -> None:
    raw = {"spec": {"container": {"environment": {"SOURCE_REVISION": "wrong"}}}}
    with pytest.raises(AssertionError, match="job env SOURCE_REVISION"):
        _assert_job_env_matches(raw, {"SOURCE_REVISION": "expected"})


def test_euclidean_distance_matches_manual_computation() -> None:
    assert _euclidean_distance_m([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_row_geometry_mismatch_is_detected() -> None:
    per_env = [
        {
            "env_id": "validation-0000",
            "details": {
                "object_position_m": [1.0, 0.0, 0.0],
                "goal_position_m": [0.0, 0.0, 0.0],
                "object_goal_distance_m": 0.5,  # corrupted: real geometry says 1.0
            },
        }
    ]
    with pytest.raises(AssertionError, match="disagrees with"):
        _assert_row_geometry_consistent(per_env)


def test_missing_position_is_rejected() -> None:
    row = {"env_id": "e0", "details": {"object_goal_distance_m": 0.0}}
    with pytest.raises(
        AssertionError, match="missing object_position_m/goal_position_m"
    ):
        _row_distance(row)


def test_nan_coordinate_is_rejected() -> None:
    row = {
        "env_id": "e0",
        "details": {
            "object_position_m": [math.nan, 0.0, 0.0],
            "goal_position_m": [0.0, 0.0, 0.0],
            "object_goal_distance_m": 0.0,
        },
    }
    with pytest.raises(AssertionError, match="non-finite coordinate"):
        _row_distance(row)


def test_wrong_dimensionality_is_rejected() -> None:
    row = {
        "env_id": "e0",
        "details": {
            "object_position_m": [0.0, 0.0],
            "goal_position_m": [0.0, 0.0, 0.0],
            "object_goal_distance_m": 0.0,
        },
    }
    with pytest.raises(AssertionError, match="exactly 3 coordinates"):
        _row_distance(row)


def test_mean_distance_recompute_requires_every_row_valid() -> None:
    per_env = [
        {
            "env_id": "e0",
            "details": {
                "object_position_m": [0.0, 0.0, 0.0],
                "goal_position_m": [0.0, 0.0, 1.0],
                "object_goal_distance_m": 1.0,
            },
        },
        {"env_id": "e1", "details": {}},
    ]
    with pytest.raises(AssertionError, match="missing object_position_m"):
        _recompute_mean_distance(per_env)


def test_reported_mean_missing_is_rejected() -> None:
    candidate = {
        "validation_report": {
            "success_summary": {},
            "per_env": [
                {
                    "env_id": "e0",
                    "details": {
                        "object_position_m": [0.0, 0.0, 0.0],
                        "goal_position_m": [0.0, 0.0, 0.0],
                        "object_goal_distance_m": 0.0,
                    },
                }
            ],
        }
    }
    with pytest.raises(AssertionError, match="missing mean_object_goal_distance_m"):
        _assert_reported_mean_matches_geometry(candidate)


def test_reported_mean_non_finite_is_rejected() -> None:
    candidate = {
        "validation_report": {
            "success_summary": {"mean_object_goal_distance_m": math.nan},
            "per_env": [
                {
                    "env_id": "e0",
                    "details": {
                        "object_position_m": [0.0, 0.0, 0.0],
                        "goal_position_m": [0.0, 0.0, 0.0],
                        "object_goal_distance_m": 0.0,
                    },
                }
            ],
        }
    }
    with pytest.raises(AssertionError, match="missing or non-finite"):
        _assert_reported_mean_matches_geometry(candidate)


def test_hash_mismatch_is_detected() -> None:
    with pytest.raises(AssertionError, match="does not match recorded hash"):
        _verify_recorded_hash(
            "candidates.json", b"tampered-bytes", {"candidates.json": "deadbeef"}
        )


def test_env_id_misalignment_is_detected() -> None:
    candidates = [
        {"validation_report": {"per_env": [{"env_id": "validation-0000"}]}},
        {"validation_report": {"per_env": [{"env_id": "validation-0001"}]}},
    ]
    with pytest.raises(AssertionError, match="different env_id sets"):
        _assert_env_ids_aligned(candidates)


def test_duplicate_env_ids_within_candidate_is_rejected() -> None:
    per_env = [{"env_id": "e0"}, {"env_id": "e0"}]
    with pytest.raises(AssertionError, match="duplicate env_id"):
        _assert_env_ids_well_formed(per_env, min_count=1)


def test_min_episodes_threshold_is_enforced() -> None:
    per_env = [{"env_id": "e0"}, {"env_id": "e1"}]
    with pytest.raises(AssertionError, match="expected at least"):
        _assert_env_ids_well_formed(per_env, min_count=5)


def test_episode_count_mismatch_with_validation_summary_is_rejected() -> None:
    candidate = {"validation_report": {"per_env": [{"env_id": "e0"}, {"env_id": "e1"}]}}
    validation = {"episodes_per_candidate": 5}
    with pytest.raises(AssertionError, match="validation summary claims"):
        _assert_episode_count_matches_summary(candidate, validation)


def test_duplicate_checkpoint_hash_is_detected_without_touching_s3() -> None:
    candidates = [
        {"checkpoint_sha256": "same", "checkpoint_uri": "s3://bucket/root/a/model.pt"},
        {"checkpoint_sha256": "same", "checkpoint_uri": "s3://bucket/root/b/model.pt"},
    ]
    with pytest.raises(AssertionError, match="distinct checkpoint hashes"):
        _assert_distinct_checkpoint_hashes(
            None, candidates, training_root="s3://bucket/root"
        )


def test_checkpoint_outside_training_root_is_rejected() -> None:
    with pytest.raises(AssertionError, match="not under the expected training root"):
        _assert_checkpoint_under_training_root(
            "s3://other-bucket/elsewhere/model.pt", "s3://bucket/root"
        )


def test_source_manifest_hash_mismatch_is_detected() -> None:
    files = {"src/worker.py": b"print('hello')", "worker.py": b"print('run')"}
    manifest = {name: hashlib.sha256(b"different-bytes").hexdigest() for name in files}
    archive = _build_test_archive(files, manifest)
    with pytest.raises(
        AssertionError, match="does not match its recorded manifest hash"
    ):
        _assert_archive_manifest_self_consistent(
            archive, expected_worker_name="worker.py"
        )


def test_source_manifest_missing_worker_is_detected() -> None:
    files = {"src/worker.py": b"print('hello')"}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    archive = _build_test_archive(files, manifest)
    with pytest.raises(AssertionError, match="does not list the expected worker"):
        _assert_archive_manifest_self_consistent(
            archive, expected_worker_name="missing_worker.py"
        )
