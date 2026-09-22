"""Report workflow latency from GitHub job metadata without executing candidate code."""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import statistics


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    elapsed = (
        datetime.fromisoformat(end.replace("Z", "+00:00"))
        - datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds()
    return round(elapsed, 3) if elapsed >= 0 else None


def _job_timing(job: dict) -> dict:
    steps = [
        {
            "name": step["name"],
            "conclusion": step.get("conclusion"),
            "seconds": _seconds(step.get("started_at"), step.get("completed_at")),
        }
        for step in job.get("steps", [])
        if step.get("conclusion") != "skipped"
    ]
    setup = sum(
        step["seconds"] or 0
        for step in steps
        if step["name"].lower().startswith(("set up ", "install ", "create "))
    )
    # GitHub gives never-started cancelled jobs a synthetic started_at value.
    unstarted = job.get("runner_id") == 0 and not job.get("steps")
    executed = (
        job.get("conclusion") != "skipped"
        and not unstarted
        and bool(job.get("started_at"))
    )
    return {
        "id": job["id"],
        "name": job["name"],
        "conclusion": job.get("conclusion"),
        "runner_wait_seconds": None
        if not executed
        else _seconds(job.get("created_at"), job.get("started_at")),
        "execution_seconds": None
        if not executed
        else _seconds(job.get("started_at"), job.get("completed_at")),
        "setup_seconds": None if not executed else round(setup, 3),
        "steps": steps,
    }


def _report(run: dict, pages: list[dict]) -> dict:
    source_jobs = [
        job
        for page in pages
        for job in page["jobs"]
        if job["name"] != "ci-timing-report"
    ]
    gate = next(
        (job for job in source_jobs if job["name"] == "security-regression"), {}
    )
    jobs = [_job_timing(job) for job in source_jobs]
    waits = [
        job["runner_wait_seconds"]
        for job in jobs
        if job["runner_wait_seconds"] is not None
    ]
    return {
        "run_id": run["id"],
        "attempt": run["run_attempt"],
        "event": run["event"],
        "conclusion": gate.get("conclusion") or run.get("conclusion"),
        "head_sha": run["head_sha"],
        "wall_seconds": _seconds(
            run["created_at"], gate.get("completed_at") or run.get("updated_at")
        ),
        "median_runner_wait_seconds": statistics.median(waits) if waits else None,
        "max_runner_wait_seconds": max(waits) if waits else None,
        "total_job_execution_seconds": sum(
            job["execution_seconds"] or 0 for job in jobs
        ),
        "jobs": jobs,
    }


def _cell(value: object) -> str:
    return (
        html.escape(str(value))
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def _markdown(report: dict) -> str:
    lines = [
        "## CI timing report",
        "",
        f"Run {report['run_id']}, attempt {report['attempt']} "
        f"({_cell(report['event'])}; {_cell(report['conclusion'])}).",
        "",
        f"Validation elapsed: **{_duration(report['wall_seconds'])}**. "
        f"Median runner wait: **{_duration(report['median_runner_wait_seconds'])}**; "
        f"maximum: **{_duration(report['max_runner_wait_seconds'])}**.",
        "",
        "Runner wait is job creation → start; dependency waits before job creation "
        "are excluded. Execution includes setup and post-job cleanup. Setup is the "
        "subset of steps named Set up, Install, or Create. Parallel job times do not "
        "add up to validation elapsed time (run creation → required gate completion, "
        "including earlier attempts). Historical runs without that gate use the "
        "run's last update. Only the current attempt's jobs are listed; the timing "
        "report itself is excluded. Run cancellation can interrupt reporting.",
        "",
        "| Job | Result | Runner wait | Execution | Setup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for job in report["jobs"]:
        lines.append(
            f"| {_cell(job['name'])} | {_cell(job['conclusion'])} | "
            f"{_duration(job['runner_wait_seconds'])} | "
            f"{_duration(job['execution_seconds'])} | "
            f"{_duration(job['setup_seconds'])} |"
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    pages = json.loads(args.jobs.read_text())
    report = _report(run, pages)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "timing.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_directory / "timing.md").write_text(_markdown(report))


if __name__ == "__main__":
    _main()
