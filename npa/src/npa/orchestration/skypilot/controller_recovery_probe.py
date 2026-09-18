"""Verify one orphaned managed job inside its original controller before cancellation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import yaml


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read_database():
    paths = list(Path.home().glob(".sky/**/spot_jobs.db"))
    _require(len(paths) == 1, "original controller database is ambiguous")
    with sqlite3.connect(f"file:{paths[0]}?mode=ro", uri=True) as database:
        database.row_factory = sqlite3.Row
        jobs = [dict(row) for row in database.execute("SELECT * FROM job_info")]
        tasks = [dict(row) for row in database.execute("SELECT * FROM spot")]
    return jobs, tasks


def _verify_definition(job, task, expected):
    for key in ("user_hash", "workspace", "name"):
        _require(job[key] == expected[key], f"original caller/job {key} changed")
    for key in ("dag_yaml_content", "config_file_content"):
        _require(_digest(job[key]) == expected[key + "_sha256"], "original configuration changed")
    _require(task["submitted_at"] == expected["submitted_at"], "job submission identity changed")
    config = yaml.safe_load(job["config_file_content"])
    _require(config["kubernetes"]["allowed_contexts"] == [expected["context"]], "context is not exclusive")
    _require(config["jobs"]["controller"]["resources"]["region"] == expected["context"], "controller context differs")
    documents = list(yaml.safe_load_all(job["dag_yaml_content"]))
    definitions = [item for item in documents if isinstance(item, dict) and "resources" in item]
    _require(len(definitions) == 1, "multi-task controller recovery is unsupported")
    definition = definitions[0]
    env = definition["envs"]
    _require(env["NPA_WORKFLOW_RUN_ID"] == expected["run_id"], "run binding changed")
    _require(env["NPA_WORKFLOW_ATTEMPT_ID"] == expected["attempt_id"], "attempt binding changed")
    resources = json.loads(task["full_resources"])
    _require(resources["infra"] == "kubernetes/" + expected["context"], "worker context differs")
    images = resources["image_id"]
    _require(images == {expected["context"]: "docker:" + expected["image"]}, "immutable image changed")
    _require(definition["resources"]["image_id"] == images, "submitted image differs")
    _require(task["task_name"] == expected["name"], "task name differs")


def verify_job(jobs, tasks, expected):
    """Require an exclusive controller and an unchanged original execution binding.

    Args:
        jobs: Original controller job records.
        tasks: Original controller task records.
        expected: Private caller, configuration and durable attempt binding.
    Returns:
        The single verified native task record.
    Raises:
        ValueError: Identity is absent, shared, ambiguous or changed.
    """
    _require(len(jobs) == len(tasks) == 1, "shared or ambiguous controller; refusing recovery")
    job, task = jobs[0], tasks[0]
    _require(job["spot_job_id"] == task["spot_job_id"] == expected["job_id"], "immutable job ID changed")
    _verify_definition(job, task, expected)
    return task


def _main():
    expected = json.load(sys.stdin)
    jobs, tasks = _read_database()
    task = verify_job(jobs, tasks, expected)
    result = {"status": task["status"], "job_binding_verified": True, "cancel_requested": False}
    if expected.get("cancel"):
        import sky
        from sky.jobs import utils

        _require(sky.__version__ == "0.12.2", "unsupported controller version")
        # The native API signals exactly this job's controller; it owns worker cleanup.
        # Never edit its database, delete pods, or invoke a name/all selector.
        utils.cancel_jobs_by_id(
            [expected["job_id"]], current_workspace=expected["workspace"],
            user_hash=expected["user_hash"],
        )
        result["cancel_requested"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    _main()
