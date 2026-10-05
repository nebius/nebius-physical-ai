"""Recover current PR validation only when GitHub runner acquisition failed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import runpy
import subprocess


# Isolated Python omits the script directory from its import search path.
runner_startup_jobs = runpy.run_path(
    str(Path(__file__).with_name("ci_runner_startup.py"))
)["runner_startup_jobs"]


_WORKFLOW = ".github/workflows/security-regression.yml"


def _api(endpoint: str, *, method: str = "GET"):
    result = subprocess.run(
        ["gh", "api", "--method", method, endpoint],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def _pages(endpoint: str, key: str | None = None) -> list[dict]:
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        item
        for page in json.loads(result.stdout)
        for item in (page[key] if key else page)
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _latest_validation(repository: str, number: int) -> tuple[dict, dict]:
    pull = _api(f"repos/{repository}/pulls/{number}")
    _require(pull["state"] == "open", "The PR is no longer open")
    _require(pull["base"]["ref"] == "main", "The PR does not target main")
    runs = _pages(
        f"repos/{repository}/actions/workflows/security-regression.yml/runs"
        f"?event=pull_request&head_sha={pull['head']['sha']}&per_page=100",
        "workflow_runs",
    )
    matching = [
        run
        for run in runs
        if run.get("event") == "pull_request"
        and run.get("path") == _WORKFLOW
        and run.get("head_sha") == pull["head"]["sha"]
        and run.get("repository", {}).get("full_name") == repository
        and any(item["number"] == number for item in run.get("pull_requests", []))
    ]
    _require(bool(matching), "No validation run matches the current PR head")
    run = max(matching, key=lambda item: item["id"])
    association = next(
        item for item in run["pull_requests"] if item["number"] == number
    )
    _require(
        association.get("base", {}).get("sha") == pull["base"]["sha"],
        "The tested base changed; update the branch to create fresh PR validation",
    )
    return pull, run


def _recoverable_jobs(repository: str, run: dict) -> list[dict]:
    _require(run["status"] == "completed", "Validation is still active")
    _require(
        run["conclusion"] in {"failure", "cancelled", "startup_failure"},
        "Validation has no recoverable failed result",
    )
    jobs = _pages(
        f"repos/{repository}/actions/runs/{run['id']}"
        f"/attempts/{run['run_attempt']}/jobs?per_page=100",
        "jobs",
    )
    jobs = runner_startup_jobs(repository, jobs, _pages)
    _require(
        any(job["runner_startup_failure"] for job in jobs),
        "No confirmed GitHub runner-acquisition failure was found",
    )
    _require(
        all(_component_recoverable(job) for job in jobs),
        "A component failed for another or unknown reason; inspect the validation run",
    )
    return jobs


def _component_recoverable(job: dict) -> bool:
    if job["name"] == "ci-timing-report":
        return True
    if job.get("status") != "completed":
        return False
    if job["conclusion"] in {"success", "skipped"} or job["runner_startup_failure"]:
        return True
    if job["name"] != "security-regression" or job["conclusion"] != "failure":
        return False
    failures = [
        step["name"]
        for step in job.get("steps", [])
        if step.get("conclusion") not in {"success", "skipped"}
    ]
    return failures == ["Require every candidate gate and security audit"]


def _recover(repository: str, number: int, rerun: bool) -> dict:
    pull, run = _latest_validation(repository, number)
    jobs = _recoverable_jobs(repository, run)
    result = {
        "pr": number,
        "head_sha": pull["head"]["sha"],
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "runner_startup_failures": [
            job["name"] for job in jobs if job["runner_startup_failure"]
        ],
        "rerun_requested": False,
    }
    if not rerun:
        return result
    current_pull, current_run = _latest_validation(repository, number)
    _require(
        current_pull["head"]["sha"] == pull["head"]["sha"]
        and current_pull["base"]["sha"] == pull["base"]["sha"]
        and (current_run["id"], current_run["run_attempt"], current_run["status"])
        == (run["id"], run["run_attempt"], "completed"),
        "The PR or validation attempt changed during diagnosis",
    )
    # A complete new attempt is required for the merge queue's validation receipt.
    _api(f"repos/{repository}/actions/runs/{run['id']}/rerun", method="POST")
    result["rerun_requested"] = True
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository) or args.pr < 1:
        parser.error("Pass --repository owner/name and a positive --pr")
    try:
        result = _recover(args.repository, args.pr, args.rerun)
    except ValueError as error:
        parser.exit(1, f"Recovery refused: {error}\n")
    except (subprocess.CalledProcessError, KeyError, TypeError):
        parser.exit(
            1, "Recovery refused: GitHub validation metadata could not be verified\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
