"""Exercise exact-tree queue admission against authenticated GitHub metadata."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ci_queue_evidence", ROOT / "npa/scripts/ci_queue_evidence.py"
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
REPOSITORY = "example/project"
BASE, HEAD, TESTED, QUEUED, TREE = (character * 40 for character in "abcde")
NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


def _event():
    return {
        "repository": {"full_name": REPOSITORY},
        "merge_group": {
            "base_ref": "refs/heads/main",
            "base_sha": BASE,
            "head_sha": QUEUED,
            "head_ref": f"refs/heads/gh-readonly-queue/main/pr-12-{BASE}",
        },
    }


def _run():
    return {
        "id": 100,
        "path": evidence._WORKFLOW,
        "event": "pull_request",
        "head_sha": HEAD,
        "repository": {"full_name": REPOSITORY},
        "pull_requests": [{"number": 12}],
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "run_started_at": (NOW - timedelta(minutes=20)).isoformat(),
    }


def _values():
    run = _run()
    jobs = evidence._REQUIRED_JOBS | {
        *(f"test-gate / pytest-3.12-shard-{index}" for index in range(1, 9)),
        "test-gate / coverage",
        "test-gate / browser-and-compatibility",
    }
    return {
        "pull": {
            "number": 12,
            "state": "open",
            "base": {"ref": "main"},
            "head": {"sha": HEAD},
        },
        "runs": [run],
        "jobs": [{"name": name, "conclusion": "success"} for name in sorted(jobs)],
        "artifacts": [
            {
                "name": f"validated-candidate-1-{TESTED}",
                "expired": False,
                "workflow_run": {"id": 100, "head_sha": HEAD},
            }
        ],
        "tested": {
            "sha": TESTED,
            "tree": {"sha": TREE},
            "parents": [{"sha": BASE}, {"sha": HEAD}],
        },
        "queued": {"sha": QUEUED, "tree": {"sha": TREE}},
        "current": deepcopy(run),
        "trees": {},
    }


@pytest.fixture
def metadata(monkeypatch):
    event, values = _event(), _values()

    def api(endpoint, *, pages=False):
        if endpoint.endswith("/pulls/12"):
            return values["pull"]
        if "/workflows/" in endpoint:
            assert pages
            return [{"workflow_runs": values["runs"]}]
        if "/jobs?" in endpoint:
            assert pages
            assert "/attempts/1/" in endpoint
            return [{"jobs": values["jobs"]}]
        if "/artifacts?" in endpoint:
            assert pages
            return [{"artifacts": values["artifacts"]}]
        if endpoint.endswith(f"/commits/{TESTED}"):
            return values["tested"]
        if endpoint.endswith(f"/commits/{QUEUED}"):
            return values["queued"]
        if endpoint.endswith("/runs/100"):
            return values["current"]
        if "/git/trees/" in endpoint:
            return values["trees"][endpoint.split("/git/trees/")[1].split("?")[0]]
        raise AssertionError(endpoint)

    monkeypatch.setattr(evidence, "_api", api)
    return event, values


def test_identical_tree_accepts_different_merge_commit_ids(metadata):
    event, _ = metadata
    assert evidence.verify(REPOSITORY, event, NOW) == {
        "mode": "reuse",
        "run_id": 100,
        "tested_sha": TESTED,
        "tree": TREE,
    }


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", None])
def test_every_required_job_must_succeed(metadata, conclusion):
    event, values = metadata
    for job in values["jobs"]:
        original = job["conclusion"]
        job["conclusion"] = conclusion
        with pytest.raises(ValueError):
            evidence.verify(REPOSITORY, event, NOW)
        job["conclusion"] = original


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", ".github/workflows/other.yml"),
        ("event", "push"),
        ("head_sha", BASE),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("pull_requests", []),
        ("repository", {"full_name": "other/project"}),
        ("run_started_at", (NOW - timedelta(hours=25)).isoformat()),
        ("run_started_at", (NOW + timedelta(minutes=1)).isoformat()),
    ],
)
def test_rejects_unrelated_incomplete_and_stale_runs(metadata, field, value):
    event, values = metadata
    values["runs"][0][field] = value
    with pytest.raises(ValueError):
        evidence.verify(REPOSITORY, event, NOW)


def test_newer_failed_run_cannot_inherit_an_older_pass(metadata):
    event, values = metadata
    newer = {**values["runs"][0], "id": 101, "conclusion": "failure"}
    values["runs"].append(newer)
    with pytest.raises(ValueError, match="latest PR validation"):
        evidence.verify(REPOSITORY, event, NOW)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "expired",
        "duplicate",
        "wrong_attempt",
        "wrong_run",
        "wrong_head",
        "bad_sha",
    ],
)
def test_receipt_is_unique_and_bound_to_the_successful_attempt(metadata, mutation):
    event, values = metadata
    artifact = values["artifacts"][0]
    if mutation == "missing":
        values["artifacts"] = []
    elif mutation == "expired":
        artifact["expired"] = True
    elif mutation == "duplicate":
        values["artifacts"].append(deepcopy(artifact))
    elif mutation == "wrong_attempt":
        artifact["name"] = f"validated-candidate-2-{TESTED}"
    elif mutation == "wrong_run":
        artifact["workflow_run"]["id"] = 101
    elif mutation == "wrong_head":
        artifact["workflow_run"]["head_sha"] = BASE
    else:
        artifact["name"] = "validated-candidate-1-invalid"
    with pytest.raises(ValueError):
        evidence.verify(REPOSITORY, event, NOW)


def test_queue_tree_must_match_including_preceding_queued_changes(metadata):
    event, values = metadata
    values["queued"]["tree"]["sha"] = "f" * 40
    with pytest.raises(ValueError, match="unvalidated changes"):
        evidence.verify(REPOSITORY, event, NOW)


def test_receipt_cannot_name_a_commit_unrelated_to_the_validated_head(metadata):
    event, values = metadata
    values["tested"]["parents"][1]["sha"] = BASE
    with pytest.raises(ValueError, match="tested PR merge commit"):
        evidence.verify(REPOSITORY, event, NOW)


def test_rerun_during_verification_invalidates_evidence(metadata):
    event, values = metadata
    values["current"]["run_attempt"] = 2
    with pytest.raises(ValueError, match="rerun"):
        evidence.verify(REPOSITORY, event, NOW)


def test_optional_reporting_does_not_hold_the_queue(metadata):
    event, values = metadata
    for run in (values["runs"][0], values["current"]):
        run.update(status="in_progress", conclusion=None)
    values["jobs"].append({"name": "ci-timing-report", "conclusion": None})
    assert evidence.verify(REPOSITORY, event, NOW)["tree"] == TREE


def test_prose_smoke_can_replace_full_tests_but_not_security(metadata):
    event, values = metadata
    values["jobs"] = [
        job for job in values["jobs"] if job["name"] in evidence._REQUIRED_JOBS
    ]
    values["jobs"].append({"name": "test-gate / pr-smoke", "conclusion": "success"})
    assert evidence.verify(REPOSITORY, event, NOW)["tree"] == TREE
    values["jobs"] = [
        job for job in values["jobs"] if job["name"] != "security-scanners"
    ]
    with pytest.raises(ValueError, match="Required PR checks"):
        evidence.verify(REPOSITORY, event, NOW)


def test_api_failure_never_becomes_reuse(metadata, monkeypatch):
    event, _ = metadata

    def unavailable(*args, **kwargs):
        raise OSError("GitHub unavailable")

    monkeypatch.setattr(evidence, "_api", unavailable)
    with pytest.raises(OSError):
        evidence.verify(REPOSITORY, event, NOW)


@pytest.mark.parametrize(
    "path,mode",
    [
        ("npa/src/npa/example.py", "retest"),
        ("npa/tests/cli/test_example.py", "retest"),
        ("npa/docker/workbench/base/Dockerfile", "full"),
        (".github/workflows/security-regression.yml", "full"),
        ("scripts/security_install.sh", "full"),
    ],
)
def test_changed_trees_rerun_tests_and_only_reuse_unchanged_images(
    metadata, path, mode
):
    event, values = metadata
    values["queued"]["tree"]["sha"] = "f" * 40
    entry = {"path": path, "mode": "100644", "type": "blob", "sha": HEAD}
    values["trees"] = {
        TREE: {"truncated": False, "tree": [entry]},
        "f" * 40: {"truncated": False, "tree": [{**entry, "sha": BASE}]},
    }
    namespace = evidence.runpy.run_path(
        str(ROOT / "scripts/ci_image_security_scope.py")
    )
    result = evidence.verify(
        REPOSITORY, event, NOW, namespace["needs_deep_image_security"]
    )
    assert result["mode"] == mode


def test_incomplete_tree_comparison_cannot_reuse_image_checks(metadata):
    event, values = metadata
    values["queued"]["tree"]["sha"] = "f" * 40
    values["trees"][TREE] = {"truncated": True, "tree": []}
    with pytest.raises(ValueError, match="incomplete tree"):
        evidence.verify(REPOSITORY, event, NOW, lambda paths: False)


@pytest.mark.parametrize("change", ["addition", "deletion", "mode", "type"])
def test_image_comparison_includes_every_kind_of_tree_change(metadata, change):
    event, values = metadata
    values["queued"]["tree"]["sha"] = "f" * 40
    path = "npa/docker/example/Dockerfile"
    entry = {"path": path, "mode": "100644", "type": "blob", "sha": HEAD}
    before, after = [entry], [entry.copy()]
    if change == "addition":
        before = []
    elif change == "deletion":
        after = []
    elif change == "mode":
        after[0]["mode"] = "100755"
    else:
        after[0]["type"] = "commit"
    values["trees"] = {
        TREE: {"truncated": False, "tree": before},
        "f" * 40: {"truncated": False, "tree": after},
    }
    result = evidence.verify(REPOSITORY, event, NOW, lambda paths: path in paths)
    assert result["mode"] == "full"
