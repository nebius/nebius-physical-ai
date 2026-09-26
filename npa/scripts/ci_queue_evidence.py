"""Verify completed PR validation for the exact tree proposed by the merge queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import runpy
import subprocess
from collections.abc import Callable


_WORKFLOW = ".github/workflows/security-regression.yml"
_REQUIRED_JOBS = {
    "pr-precheck",
    "gitleaks",
    "scan",
    "security-scanners",
    "hostile-input-runtime",
    "security-regression",
    "lint-gate / ruff",
    "lint-gate / docs-drift",
    "image-security / Image policy and complete-byte security",
    "image-security / Base image CVE inventory",
}
_SHA = re.compile(r"[0-9a-f]{40}")


def _api(endpoint: str, *, pages: bool = False):
    command = ["gh", "api", endpoint]
    if pages:
        command.extend(["--paginate", "--slurp"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validation_available(run: dict) -> bool:
    return (run["status"], run["conclusion"]) in {
        ("completed", "success"),
        ("in_progress", None),
        ("queued", None),
    }


def _queue_candidate(repository: str, event: dict) -> tuple[dict, dict]:
    group = event["merge_group"]
    match = re.fullmatch(
        r"refs/heads/gh-readonly-queue/main/pr-(\d+)-[0-9a-f]{40}",
        group["head_ref"],
    )
    _require(match is not None, "Unrecognized merge-queue candidate reference")
    _require(group["base_ref"] == "refs/heads/main", "Unexpected queue target")
    for field in ("base_sha", "head_sha"):
        _require(bool(_SHA.fullmatch(group[field])), "Invalid queue commit")
    pull = _api(f"repos/{repository}/pulls/{match[1]}")
    _require(pull["state"] == "open", "The queued PR is no longer open")
    _require(pull["base"]["ref"] == "main", "The PR target changed")
    return group, pull


def _latest_validation(repository: str, pull: dict, now: datetime) -> dict:
    head = pull["head"]["sha"]
    endpoint = (
        f"repos/{repository}/actions/workflows/security-regression.yml/runs"
        f"?event=pull_request&head_sha={head}&per_page=100"
    )
    runs = [run for page in _api(endpoint, pages=True) for run in page["workflow_runs"]]
    _require(bool(runs), "No PR validation exists for the current head")
    run = max(runs, key=lambda item: item["id"])
    _require(run["path"] == _WORKFLOW, "Unexpected validation workflow")
    _require(run["event"] == "pull_request", "Evidence is not PR validation")
    _require(run["head_sha"] == head, "The validated PR head changed")
    _require(run["repository"]["full_name"] == repository, "Wrong repository")
    _require(
        any(item["number"] == pull["number"] for item in run["pull_requests"]),
        "The validation run is not associated with this PR",
    )
    _require(
        _validation_available(run),
        "The latest PR validation has not completed successfully",
    )
    started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
    _require(
        timedelta(0) <= now - started <= timedelta(hours=24), "PR evidence is stale"
    )
    return run


def _test_scope_succeeded(jobs: list[dict]) -> bool:
    for job in jobs:
        if job["conclusion"] != "success":
            continue
        if job["name"] == "test-gate / test-scope":
            return True
        if job["name"] == "gitleaks" and any(
            step["name"] == "Select tests using the trusted base policy"
            and step.get("conclusion") == "success"
            for step in job.get("steps", [])
        ):
            return True
    return False


def _verify_jobs(repository: str, run: dict) -> None:
    endpoint = (
        f"repos/{repository}/actions/runs/{run['id']}"
        f"/attempts/{run['run_attempt']}/jobs?per_page=100"
    )
    jobs = [job for page in _api(endpoint, pages=True) for job in page["jobs"]]
    successful = {job["name"] for job in jobs if job["conclusion"] == "success"}
    _require(
        _REQUIRED_JOBS <= successful, "Required PR checks are missing or unsuccessful"
    )
    _require(_test_scope_succeeded(jobs), "Trusted test selection did not succeed")
    _require(
        all(
            job["conclusion"] in {"success", "skipped"}
            for job in jobs
            if job["name"] != "ci-timing-report"
        ),
        "The validation attempt contains an unsuccessful job",
    )
    full_suite = {
        *(f"test-gate / pytest-3.12-shard-{index}" for index in range(1, 9)),
        "test-gate / coverage",
        "test-gate / browser-and-compatibility",
    }
    _require(
        full_suite <= successful or "test-gate / pr-smoke" in successful,
        "Neither full test coverage nor the prose-only smoke completed",
    )


def _tested_commit(repository: str, run: dict) -> dict:
    pages = _api(
        f"repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100",
        pages=True,
    )
    prefix = f"validated-candidate-{run['run_attempt']}-"
    receipts = [
        artifact
        for page in pages
        for artifact in page["artifacts"]
        if artifact["name"].startswith(prefix) and not artifact["expired"]
    ]
    _require(
        len(receipts) == 1, "The successful attempt has no unique validation receipt"
    )
    receipt = receipts[0]
    revision = receipt["name"].removeprefix(prefix)
    _require(bool(_SHA.fullmatch(revision)), "Invalid validated commit identity")
    _require(
        receipt["workflow_run"]["id"] == run["id"], "Receipt belongs to another run"
    )
    _require(
        receipt["workflow_run"]["head_sha"] == run["head_sha"], "Receipt head differs"
    )
    commit = _api(f"repos/{repository}/git/commits/{revision}")
    _require(commit["sha"] == revision, "Validated commit identity differs")
    _require(
        len(commit["parents"]) == 2 and commit["parents"][1]["sha"] == run["head_sha"],
        "Receipt does not identify the tested PR merge commit",
    )
    return commit


def _tree_entries(repository: str, tree: str) -> dict:
    result = _api(f"repos/{repository}/git/trees/{tree}?recursive=1")
    _require(result.get("truncated") is False, "GitHub returned an incomplete tree")
    return {
        entry["path"]: (entry["mode"], entry["type"], entry["sha"])
        for entry in result["tree"]
    }


def _reuse_mode(repository: str, tested: dict, queued: dict, image_policy) -> str:
    if tested["tree"]["sha"] == queued["tree"]["sha"]:
        return "reuse"
    _require(image_policy is not None, "Cannot classify unvalidated changes")
    before = _tree_entries(repository, tested["tree"]["sha"])
    after = _tree_entries(repository, queued["tree"]["sha"])
    paths = [
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    ]
    return "full" if image_policy(paths) else "retest"


def _require_current_attempt(repository: str, run: dict) -> None:
    current = _api(f"repos/{repository}/actions/runs/{run['id']}")
    _require(
        current["run_attempt"] == run["run_attempt"] and _validation_available(current),
        "Validation was rerun while its evidence was being checked",
    )


def verify(
    repository: str,
    event: dict,
    now: datetime,
    image_policy: Callable[[list[str]], bool] | None = None,
) -> dict:
    """Verify PR evidence and choose which combined-tree checks must rerun.

    Args:
        repository: GitHub owner/repository receiving the merge-group event.
        event: Original GitHub merge-group event, never candidate-supplied data.
        now: Current timezone-aware time for evidence freshness.
        image_policy: Trusted-base classifier for changed image-security inputs.
    Returns:
        Validation mode and verified run, candidate, and tree identities.
    Raises:
        ValueError: Evidence is absent, stale, incomplete, or cannot be compared.
        KeyError: GitHub metadata is incomplete.
        subprocess.CalledProcessError: GitHub metadata cannot be read.
    """
    _require(bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repository)), "Invalid repository")
    _require(event["repository"]["full_name"] == repository, "Event repository differs")
    group, pull = _queue_candidate(repository, event)
    run = _latest_validation(repository, pull, now)
    _verify_jobs(repository, run)
    tested = _tested_commit(repository, run)
    queued = _api(f"repos/{repository}/git/commits/{group['head_sha']}")
    _require(queued["sha"] == group["head_sha"], "Queue commit identity differs")
    mode = _reuse_mode(repository, tested, queued, image_policy)
    _require_current_attempt(repository, run)
    return {
        "mode": mode,
        "run_id": run["id"],
        "tested_sha": tested["sha"],
        "tree": queued["tree"]["sha"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--event-path", required=True, type=Path)
    parser.add_argument("--github-output", required=True, type=Path)
    parser.add_argument("--image-policy", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        proof = verify(
            arguments.repository,
            json.loads(arguments.event_path.read_text()),
            datetime.now(timezone.utc),
            runpy.run_path(str(arguments.image_policy))["needs_deep_image_security"],
        )
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        # Missing proof must restore all checks, including during policy rollout.
        proof = {"mode": "full", "reason": str(error)}
    print(json.dumps(proof, sort_keys=True))
    with arguments.github_output.open("a") as output:
        output.write(f"mode={proof['mode']}\n")
        output.write(f"source_run={proof.get('run_id', '')}\n")


if __name__ == "__main__":
    _main()
