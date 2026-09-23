"""Exercise queue diagnostics, exact candidate association, and safe comment updates."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_queue_report as reporter  # noqa: E402


_REPOSITORY = "example/workbench"
_SHA = "a" * 40
_REMOVAL = {
    "id": "removal-id",
    "createdAt": "2026-09-21T02:20:00Z",
    "reason": "checks_timed_out",
    "beforeCommit": {"oid": _SHA},
}
_RUN = {
    "id": 12,
    "run_attempt": 2,
    "head_sha": _SHA,
    "head_branch": "gh-readonly-queue/main/pr-604-" + "b" * 40,
    "event": "merge_group",
    "path": reporter._WORKFLOW,
    "repository": {"full_name": _REPOSITORY},
    "status": "completed",
    "conclusion": "success",
}


def _job(**changes) -> dict:
    return {
        "id": 21,
        "name": "test-gate / shard-5",
        "conclusion": "success",
        "status": "completed",
        "runner_id": 8,
        "created_at": "2026-09-21T01:50:00Z",
        "started_at": "2026-09-21T02:10:00Z",
        "completed_at": "2026-09-21T02:22:00Z",
        "steps": [],
        **changes,
    }


def test_timeout_remains_a_timeout_when_every_check_eventually_passes() -> None:
    """Preserve the actual removal reason instead of misreporting later success."""
    body = reporter._render(_REPOSITORY, _REMOVAL, _RUN, [_job()])
    assert "before the queue deadline" in body
    assert "| running | success | 20m 00s |" in body
    assert "later successful check does not undo" in body
    assert "/actions/runs/12/attempts/2" in body
    assert "/actions/runs/12/job/21" in body


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"completed_at": "2026-09-21T02:19:00Z"}, "success"),
        (
            {"runner_id": 0, "steps": [], "conclusion": "cancelled"},
            "waiting for a runner",
        ),
        (
            {
                "created_at": "2026-09-21T02:21:00Z",
                "started_at": "2026-09-21T02:21:02Z",
            },
            "not yet scheduled",
        ),
        (
            {"runner_id": 0, "started_at": None, "completed_at": None},
            "waiting for a runner",
        ),
    ],
)
def test_job_state_is_reconstructed_at_removal(changes: dict, expected: str) -> None:
    """Never interpret GitHub's synthetic cancelled-job timestamps as execution."""
    assert reporter._at_removal(_job(**changes), _REMOVAL["createdAt"]) == expected


def test_failed_steps_are_named_without_copying_logs_or_blaming_siblings() -> None:
    """Report the actionable step and distinguish matrix cancellation."""
    removal = {**_REMOVAL, "reason": "failed_checks"}
    failure = _job(
        name="test-gate / test-scope",
        conclusion="failure",
        completed_at="2026-09-21T02:19:00Z",
        steps=[
            {
                "name": "Check CI dependency pins",
                "conclusion": "failure",
                "log": "private-log",
            }
        ],
    )
    sibling = _job(name="shard-2", conclusion="cancelled", runner_id=0)
    aggregate = _job(name="security-regression", conclusion="failure")
    body = reporter._render(_REPOSITORY, removal, _RUN, [failure, sibling, aggregate])
    assert "Check CI dependency pins" in body
    assert "Required checks failed" in body
    assert "ci_requirements.py --update" in body
    assert "Cancelled siblings may follow matrix fail-fast" in body
    assert "private-log" not in body
    assert "[security-regression]" not in body


def test_names_cannot_inject_links_mentions_or_multiline_markdown() -> None:
    """Treat candidate-controlled job labels as text in privileged comments."""
    name = "[fake](https://example.invalid) | @everyone\n<script>"
    body = reporter._render(_REPOSITORY, _REMOVAL, _RUN, [_job(name=name)])
    assert "[fake](https://example.invalid)" not in body
    assert "@everyone" not in body
    assert "<script>" not in body
    assert "\n<script>" not in body


def test_unknown_run_still_explains_the_removal() -> None:
    """Missing Actions metadata must not erase the queue's rejection reason."""
    body = reporter._render(_REPOSITORY, _REMOVAL, None, [])
    assert "before the queue deadline" in body
    assert "No matching validation run" in body
    assert "actions/runs" not in body


def test_failed_run_without_dequeue_event_is_reported_without_claiming_removal(
    monkeypatch,
) -> None:
    """A rebuilt or failing candidate still receives an actionable PR comment."""
    run = {
        **_RUN,
        "run_attempt": 1,
        "conclusion": "failure",
        "updated_at": _REMOVAL["createdAt"],
    }
    monkeypatch.setattr(reporter, "_removal", lambda *args: None)
    monkeypatch.setattr(reporter, "_candidate_run", lambda *args: run)
    removal, selected = reporter._diagnosis(_REPOSITORY, 604, _SHA)
    body = reporter._render(_REPOSITORY, removal, selected, [])
    assert "Merge-queue validation failed" in body
    assert "removed this PR" not in body
    actual_removal = reporter._render(_REPOSITORY, _REMOVAL, run, [])
    assert body.splitlines()[0] == actual_removal.splitlines()[0]


def test_candidate_selection_rejects_unrelated_runs(monkeypatch) -> None:
    """Require exact synthetic SHA, workflow, repository, and merge-group event."""
    invalid = [
        {**_RUN, "head_sha": "c" * 40},
        {**_RUN, "event": "pull_request"},
        {**_RUN, "path": ".github/workflows/unrelated.yml"},
        {**_RUN, "repository": {"full_name": "different/repo"}},
    ]
    monkeypatch.setattr(reporter, "_pages", lambda *args: invalid)
    assert reporter._candidate_run(_REPOSITORY, _SHA) is None
    invalid.append(_RUN)
    assert reporter._candidate_run(_REPOSITORY, _SHA) == _RUN


def test_later_rerun_cannot_erase_the_rejecting_attempt(monkeypatch) -> None:
    """Bind diagnosis to the attempt running when GitHub removed the candidate."""
    rerun = {**_RUN, "run_started_at": "2026-09-21T02:30:00Z"}
    original = {
        **_RUN,
        "run_attempt": 1,
        "conclusion": "failure",
        "run_started_at": "2026-09-21T01:50:00Z",
    }
    reads = []

    def api(path):
        reads.append(path)
        return original

    monkeypatch.setattr(reporter, "_api", api)
    assert (
        reporter._attempt_at_removal(_REPOSITORY, rerun, _REMOVAL["createdAt"])
        == original
    )
    assert reads == ["repos/example/workbench/actions/runs/12/attempts/1"]


def test_timeline_paginates_to_the_exact_rejected_candidate(monkeypatch) -> None:
    """Late workflow completion must update its own attempt, not a newer attempt."""
    cursors = []

    def api(path, *, payload, method):
        cursors.append(payload["variables"]["before"])
        first_page = len(cursors) == 1
        nodes = (
            [{**_REMOVAL, "beforeCommit": {"oid": "c" * 40}}]
            if first_page
            else [_REMOVAL]
        )
        return {
            "data": {
                "repository": {
                    "pullRequest": {
                        "timelineItems": {
                            "nodes": nodes,
                            "pageInfo": {
                                "hasPreviousPage": first_page,
                                "startCursor": "previous",
                            },
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(reporter, "_api", api)
    assert reporter._removal(_REPOSITORY, 604, _SHA) == _REMOVAL
    assert cursors == [None, "previous"]


def test_identical_bot_comment_is_not_reposted(monkeypatch) -> None:
    """Repeated events are idempotent for one queue removal."""
    body = reporter._render(_REPOSITORY, _REMOVAL, None, [])
    comments = [{"id": 1, "body": body, "user": {"login": "github-actions[bot]"}}]
    monkeypatch.setattr(reporter, "_pages", lambda *args: comments)
    writes = []
    monkeypatch.setattr(reporter, "_api", lambda *args, **kwargs: writes.append(kwargs))
    reporter._publish(_REPOSITORY, 604, body)
    assert writes == []


@pytest.mark.parametrize(
    ("login", "method"), [("github-actions[bot]", "PATCH"), ("contributor", "POST")]
)
def test_updates_only_its_own_bot_comment(monkeypatch, login: str, method: str) -> None:
    """Update late results in place while preserving human comments."""
    body = reporter._render(_REPOSITORY, _REMOVAL, None, [])
    comments = [
        {
            "id": 1,
            "body": body.splitlines()[0] + "\nold result",
            "user": {"login": login},
        }
    ]
    monkeypatch.setattr(reporter, "_pages", lambda *args: comments)
    writes = []
    monkeypatch.setattr(
        reporter, "_api", lambda *args, **kwargs: writes.append((args, kwargs))
    )
    reporter._publish(_REPOSITORY, 604, body)
    assert len(writes) == 1
    assert writes[0][1] == {"method": method, "payload": {"body": body}}
    assert writes[0][0][0].endswith(
        "comments/1" if method == "PATCH" else "604/comments"
    )


def test_poll_paginates_open_prs_and_deduplicates_queue_and_removal_candidates(
    monkeypatch,
) -> None:
    """Reconcile current open PRs without a history time limit or PR count cap."""
    cursors = []

    def api(path, *, payload, method):
        cursors.append(payload["variables"]["after"])
        first = len(cursors) == 1
        queue = {"headCommit": {"oid": _SHA}}
        nodes = (
            [
                {
                    "number": 604,
                    "mergeQueueEntry": queue,
                    "timelineItems": {"nodes": [_REMOVAL]},
                },
                {
                    "number": 605,
                    "mergeQueueEntry": None,
                    "timelineItems": {"nodes": []},
                },
            ]
            if first
            else [
                {
                    "number": 606,
                    "mergeQueueEntry": None,
                    "timelineItems": {"nodes": [_REMOVAL]},
                },
                {
                    "number": 607,
                    "mergeQueueEntry": queue,
                    "timelineItems": {"nodes": []},
                },
            ]
        )
        return {
            "data": {
                "repository": {
                    "pullRequests": {
                        "nodes": nodes,
                        "pageInfo": {"hasNextPage": first, "endCursor": "next"},
                    }
                }
            }
        }

    monkeypatch.setattr(reporter, "_api", api)
    assert reporter._open_candidates(_REPOSITORY) == [
        (604, _SHA),
        (606, _SHA),
        (607, _SHA),
    ]
    assert cursors == [None, "next"]


def test_scan_mode_reports_each_candidate_with_publication_explicit(
    monkeypatch,
) -> None:
    """The scheduler uses the same report path as a read-only local diagnosis."""
    monkeypatch.setattr(
        sys, "argv", ["report", "--repository", _REPOSITORY, "--scan-open"]
    )
    monkeypatch.setattr(
        reporter, "_open_candidates", lambda repo: [(604, _SHA), (605, None)]
    )
    calls = []
    monkeypatch.setattr(reporter, "_report", lambda *args: calls.append(args))
    reporter._main()
    assert calls == [(_REPOSITORY, 604, _SHA, False), (_REPOSITORY, 605, None, False)]


@pytest.mark.parametrize("removal", [None, {**_REMOVAL, "reason": "merged"}])
def test_successful_merges_and_missing_removals_do_not_comment(
    monkeypatch, removal
) -> None:
    """Normal queue success cannot generate a failure comment."""
    args = ["report", "--repository", _REPOSITORY, "--pr", "604", "--publish"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(reporter, "_removal", lambda *args: deepcopy(removal))
    writes = []
    monkeypatch.setattr(reporter, "_publish", lambda *args: writes.append(args))
    reporter._main()
    assert writes == []


def test_manual_diagnosis_does_not_publish_without_explicit_flag(
    monkeypatch, capsys
) -> None:
    """Historical validation is read-only unless publication is requested."""
    monkeypatch.setattr(
        sys, "argv", ["report", "--repository", _REPOSITORY, "--pr", "604"]
    )
    monkeypatch.setattr(reporter, "_removal", lambda *args: _REMOVAL)
    monkeypatch.setattr(reporter, "_candidate_run", lambda *args: None)
    writes = []
    monkeypatch.setattr(reporter, "_publish", lambda *args: writes.append(args))
    reporter._main()
    assert "queue deadline" in capsys.readouterr().out
    assert writes == []
