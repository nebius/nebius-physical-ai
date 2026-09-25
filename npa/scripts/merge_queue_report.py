"""Explain merge-queue removals using GitHub metadata without reading candidate code."""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import os
import re
import subprocess


_WORKFLOW = ".github/workflows/security-regression.yml"
_FAILURES = {"failure", "timed_out", "action_required", "startup_failure", "stale"}
_REASONS = {
    "checks_failed": "Required checks failed",
    "failed_checks": "Required checks failed",
    "checks_timed_out": "Required checks did not finish before the queue deadline",
    "merge_conflict": "The merge candidate has conflicts",
    "validation_failed": "Merge-queue validation failed",
}
_QUERY = """
query($owner:String!, $name:String!, $number:Int!, $before:String) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      timelineItems(last:100, before:$before, itemTypes:[REMOVED_FROM_MERGE_QUEUE_EVENT]) {
        pageInfo { hasPreviousPage startCursor }
        nodes { ... on RemovedFromMergeQueueEvent {
          id createdAt reason beforeCommit { oid }
        } }
      }
    }
  }
}
"""
_OPEN_QUERY = """
query($owner:String!, $name:String!, $after:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:100, after:$after, states:OPEN) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        mergeQueueEntry { headCommit { oid } }
        timelineItems(last:1, itemTypes:[REMOVED_FROM_MERGE_QUEUE_EVENT]) {
          nodes { ... on RemovedFromMergeQueueEvent {
            id createdAt reason beforeCommit { oid }
          } }
        }
      }
    }
  }
}
"""


def _api(path: str, *, payload: dict | None = None, method: str = "GET") -> dict:
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command.extend(["--input", "-"])
    result = subprocess.run(
        command,
        input=None if payload is None else json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    response = json.loads(result.stdout)
    if isinstance(response, dict) and response.get("errors"):
        raise RuntimeError("GitHub could not read the merge-queue timeline")
    return response


def _pages(path: str, key: str | None = None) -> list[dict]:
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        capture_output=True,
        text=True,
        check=True,
    )
    pages = json.loads(result.stdout)
    return [item for page in pages for item in (page[key] if key else page)]


def _open_candidates(repository: str) -> list[tuple[int, str | None]]:
    owner, name = repository.split("/")
    variables = {"owner": owner, "name": name, "after": None}
    candidates = []
    while True:
        data = _api(
            "graphql",
            method="POST",
            payload={"query": _OPEN_QUERY, "variables": variables},
        )
        page = data["data"]["repository"]["pullRequests"]
        for pull in page["nodes"]:
            queue_head = (
                (pull.get("mergeQueueEntry") or {}).get("headCommit") or {}
            ).get("oid")
            if queue_head:
                candidates.append((pull["number"], queue_head))
            for removal in pull["timelineItems"]["nodes"]:
                if removal.get("reason") == "merged":
                    continue
                sha = (removal.get("beforeCommit") or {}).get("oid")
                if not queue_head or sha != queue_head:
                    candidates.append((pull["number"], sha))
        if not page["pageInfo"]["hasNextPage"]:
            return candidates
        variables["after"] = page["pageInfo"]["endCursor"]


def _removal(repository: str, number: int, candidate: str | None) -> dict | None:
    owner, name = repository.split("/")
    variables = {"owner": owner, "name": name, "number": number, "before": None}
    while True:
        data = _api(
            "graphql", method="POST", payload={"query": _QUERY, "variables": variables}
        )
        timeline = data["data"]["repository"]["pullRequest"]["timelineItems"]
        for removal in reversed(timeline["nodes"]):
            sha = (removal.get("beforeCommit") or {}).get("oid")
            if candidate is None or candidate == sha:
                return removal
        if not timeline["pageInfo"]["hasPreviousPage"]:
            return None
        variables["before"] = timeline["pageInfo"]["startCursor"]


def _candidate_run(repository: str, candidate: str | None) -> dict | None:
    if candidate is None:
        return None
    runs = _pages(
        f"repos/{repository}/actions/workflows/security-regression.yml/runs"
        f"?event=merge_group&head_sha={candidate}&per_page=100",
        "workflow_runs",
    )
    matching = [
        run
        for run in runs
        if run["event"] == "merge_group"
        and run["head_sha"] == candidate
        and run["path"] == _WORKFLOW
        and run["repository"]["full_name"] == repository
    ]
    return max(matching, key=lambda run: (run["id"], run["run_attempt"]), default=None)


def _text(value: object) -> str:
    text = " ".join(str(value).split())
    text = html.escape(text).replace("@", "&#64;")
    return re.sub(r"([\\`*_{}\[\]()#!|~])", r"\\\1", text)


def _attempt_at_removal(repository: str, run: dict, removed_at: str) -> dict:
    # A later rerun must not replace the job/step that rejected the candidate.
    while run["run_attempt"] > 1 and run["run_started_at"] > removed_at:
        run = _api(
            f"repos/{repository}/actions/runs/{run['id']}/attempts/{run['run_attempt'] - 1}"
        )
    return run


def _diagnosis(
    repository: str, number: int, candidate: str | None
) -> tuple[dict | None, dict | None]:
    removal = _removal(repository, number, candidate)
    if removal is not None:
        if removal.get("reason") == "merged":
            return None, None
        candidate = (removal.get("beforeCommit") or {}).get("oid")
    run = _candidate_run(repository, candidate)
    if removal is None and run and run.get("conclusion") in _FAILURES:
        # A failed candidate can be rebuilt before a dequeue event is available.
        removal = {
            "id": f"run-{run['id']}",
            "createdAt": run["updated_at"],
            "reason": "validation_failed",
            "beforeCommit": {"oid": candidate},
        }
    if removal is not None and run is not None:
        run = _attempt_at_removal(repository, run, removal["createdAt"])
    return removal, run


def _at_removal(job: dict, removed_at: str) -> str:
    completed = job.get("completed_at")
    if completed and completed <= removed_at:
        return job.get("conclusion") or "completed"
    # Never-started jobs can have synthetic start times; runner_id proves a start.
    if job.get("runner_id") and job.get("started_at", "") <= removed_at:
        return "running"
    if job.get("created_at", "") > removed_at:
        return "not yet scheduled"
    return "waiting for a runner"


def _wait(job: dict) -> str:
    if (
        not job.get("runner_id")
        or not job.get("started_at")
        or not job.get("created_at")
    ):
        return "—"
    start, created = (
        datetime.fromisoformat(job[key].replace("Z", "+00:00"))
        for key in ("started_at", "created_at")
    )
    seconds = max(0, round((start - created).total_seconds()))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def _job_rows(repository: str, run: dict, jobs: list[dict], removal: dict) -> list[str]:
    rows = []
    for job in jobs:
        if job["name"] in {"security-regression", "ci-timing-report"}:
            continue
        at_removal = _at_removal(job, removal["createdAt"])
        if at_removal in {"success", "skipped"}:
            continue
        failed = [
            step["name"]
            for step in job.get("steps", [])
            if step.get("conclusion") in _FAILURES
        ]
        url = (
            f"https://github.com/{repository}/actions/runs/{run['id']}/job/{job['id']}"
        )
        latest = job.get("conclusion") or job.get("status") or "pending"
        rows.append(
            f"| [{_text(job['name'])}]({url}) | {_text(at_removal)} | {_text(latest)} | "
            f"{_wait(job)} | {_text('; '.join(failed)) or '—'} |"
        )
    return rows


def _next_step(reason: str, jobs: list[dict]) -> str:
    failed_steps = [
        step["name"]
        for job in jobs
        for step in job.get("steps", [])
        if step.get("conclusion") in _FAILURES
    ]
    if "Check CI dependency pins" in failed_steps:
        return (
            "Bring current main into the branch, run "
            "`npa/.venv/bin/python npa/scripts/ci_requirements.py --update`, "
            "and commit the reviewed pins. Run `make precheck` and "
            "`make merge-precheck` before requeueing."
        )
    if any(job.get("conclusion") in _FAILURES for job in jobs):
        return "Open the failed component and step above, fix or reproduce that failure, then rerun validation before requeueing."
    if reason == "checks_timed_out":
        return (
            "Inspect runner waits and unfinished checks before requeueing. "
            "A later successful check does not undo the queue timeout."
        )
    return "Inspect the linked validation run and the queue removal reason before requeueing."


def _validation_lines(
    repository: str, removal: dict, run: dict | None, jobs: list[dict]
) -> list[str]:
    if run is None:
        return [
            "",
            "No matching validation run was found for this exact candidate; inspect the PR checks for missing required results.",
        ]
    url = f"https://github.com/{repository}/actions/runs/{run['id']}/attempts/{run['run_attempt']}"
    lines = [
        "",
        f"[Validation run, attempt {run['run_attempt']}]({url}) — latest result: **{_text(run.get('conclusion') or run['status'])}**.",
    ]
    rows = _job_rows(repository, run, jobs, removal)
    if rows:
        state_label = (
            "At observation"
            if removal.get("reason") == "validation_failed"
            else "At removal"
        )
        lines.extend(
            [
                "",
                f"| Check | {state_label} | Latest result | Runner wait | Failed step |",
                "| --- | --- | --- | ---: | --- |",
                *rows,
            ]
        )
    return lines


def _render(repository: str, removal: dict, run: dict | None, jobs: list[dict]) -> str:
    reason = removal.get("reason") or "unknown"
    candidate = (removal.get("beforeCommit") or {}).get("oid")
    marker = f"candidate:{candidate}" if candidate else f"removal:{removal['id']}"
    heading = (
        "Merge-queue check result"
        if reason == "validation_failed"
        else "Merge queue removed this PR"
    )
    timestamp = "Observed" if reason == "validation_failed" else "Removed"
    lines = [
        f"<!-- npa-merge-queue-{marker} -->",
        f"{heading}: **{_text(_REASONS.get(reason, reason.replace('_', ' ')))}**.",
        "",
        f"{timestamp} at {_text(removal['createdAt'])}. Candidate: `{candidate or 'unavailable'}`.",
        *_validation_lines(repository, removal, run, jobs),
    ]
    lines.extend(
        [
            "",
            "Cancelled siblings may follow matrix fail-fast; use the failed component and step to identify the original failure. "
            "Runner wait excludes dependency waiting before job creation. Only job metadata is shown; logs and artifacts are not copied.",
            "",
            "Next: " + _next_step(reason, jobs),
            "",
        ]
    )
    return "\n".join(lines)


def _publish(repository: str, number: int, body: str) -> None:
    endpoint = f"repos/{repository}/issues/{number}/comments"
    marker = body.splitlines()[0]
    comments = _pages(endpoint + "?per_page=100")
    existing = next(
        (
            comment
            for comment in comments
            if comment["user"]["login"] == "github-actions[bot]"
            and comment["body"].startswith(marker)
        ),
        None,
    )
    if existing and existing["body"] == body:
        return
    if existing:
        _api(
            f"repos/{repository}/issues/comments/{existing['id']}",
            method="PATCH",
            payload={"body": body},
        )
    else:
        _api(endpoint, method="POST", payload={"body": body})


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--pr", type=int)
    parser.add_argument("--candidate")
    parser.add_argument("--scan-open", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not args.repository or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository):
        parser.error("--repository must be owner/name")
    if args.candidate and not re.fullmatch(r"[0-9a-f]{40}", args.candidate):
        parser.error("--candidate must be a full commit SHA")
    if bool(args.pr) == args.scan_open or (args.pr is not None and args.pr < 1):
        parser.error("Pass either --pr with a positive number or --scan-open")
    if args.candidate and args.scan_open:
        parser.error("--candidate requires --pr")
    return args


def _report(repository: str, number: int, candidate: str | None, publish: bool) -> None:
    removal, run = _diagnosis(repository, number, candidate)
    if removal is None:
        print("No rejected merge-queue attempt to report.")
        return
    jobs = (
        []
        if run is None
        else _pages(
            f"repos/{repository}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100",
            "jobs",
        )
    )
    body = _render(repository, removal, run, jobs)
    if publish:
        _publish(repository, number, body)
    print(body)


def _main() -> None:
    args = _arguments()
    candidates = (
        _open_candidates(args.repository)
        if args.scan_open
        else [(args.pr, args.candidate)]
    )
    for number, candidate in candidates:
        _report(args.repository, number, candidate, args.publish)


if __name__ == "__main__":
    _main()
