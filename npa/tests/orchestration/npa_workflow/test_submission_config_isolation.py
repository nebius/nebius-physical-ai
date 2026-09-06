"""Exercise config selection at real process startup, without inherited imports."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator

import pytest

from npa.clients import config
from npa.orchestration.npa_workflow import submission_state


_PROCESS = """
from dataclasses import asdict
import fcntl
import json
import sys
from npa.clients import config
from npa.orchestration.npa_workflow import submission_state as state

request = json.loads(sys.stdin.readline())
project = request.get("project", "demo")
run_id = "same-run"
operation = request["operation"]
if operation == "update":
    result = state.update_submission_state(project, run_id, request["updates"])
elif operation == "load":
    result = state.load_submission_state(project, run_id)
elif operation == "inspect":
    result = asdict(state.inspect_submission_state(project, run_id))
elif operation == "audit":
    result = asdict(state.audit_project_submissions(project))
elif operation == "path":
    result = {"receipt": str(state.submission_state_path(project, run_id)),
              "config": str(config.CONFIG_PATH)}
elif operation == "hold_lock":
    with state.submission_lock(project, run_id):
        print(json.dumps({"held": True}), flush=True)
        sys.stdin.readline()
    result = {"released": True}
elif operation == "probe_lock":
    # Keep the production lock context and the real kernel lock, but report
    # contention immediately so a broken cross-config lock cannot hang pytest.
    flock = fcntl.flock
    def nonblocking(fd, operation):
        if operation == fcntl.LOCK_EX:
            operation |= fcntl.LOCK_NB
        return flock(fd, operation)
    fcntl.flock = nonblocking
    try:
        with state.submission_lock(project, run_id):
            result = {"acquired": True}
    except BlockingIOError:
        result = {"acquired": False}
print(json.dumps(result), flush=True)
"""


def _environment(home: Path, root: str | Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(HOME=str(home), USERPROFILE=str(home))
    env.pop("NPA_CONFIG_DIR", None)
    if root is not None:
        env["NPA_CONFIG_DIR"] = str(root)
    return env


def _run(home: Path, root: str | Path | None, operation: str, **values: object) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _PROCESS],
        input=json.dumps({"operation": operation, **values}) + "\n",
        env=_environment(home, root),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


@contextmanager
def _held_lock(home: Path, root: Path) -> Iterator[None]:
    with subprocess.Popen(
        [sys.executable, "-c", _PROCESS],
        env=_environment(home, root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as holder:
        assert holder.stdin is not None
        assert holder.stdout is not None
        holder.stdin.write(json.dumps({"operation": "hold_lock"}) + "\n")
        holder.stdin.flush()
        try:
            assert json.loads(holder.stdout.readline()) == {"held": True}
            yield
        finally:
            stdout, stderr = holder.communicate("release\n")
            assert holder.returncode == 0, stderr
            assert json.loads(stdout) == {"released": True}


def test_receipt_writes_are_isolated_between_config_roots(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first, second = tmp_path / "first", tmp_path / "second"
    _run(home, first, "update", updates={"first_only": True})
    _run(home, second, "update", updates={"second_only": True})

    first_state = _run(home, first, "load")
    second_state = _run(home, second, "load")
    assert first_state["first_only"] is True
    assert "second_only" not in first_state
    assert second_state["second_only"] is True
    assert "first_only" not in second_state
    for root in (first, second):
        receipt = root / "workflow-submissions" / "demo" / "same-run.json"
        assert receipt.stat().st_mode & 0o777 == 0o600
        assert receipt.parent.stat().st_mode & 0o777 == 0o700
        assert receipt.with_suffix(".lock").stat().st_mode & 0o777 == 0o600
    assert not (home / ".npa").exists()


@pytest.mark.parametrize("operation", ["load", "inspect"])
def test_receipt_reads_do_not_cross_config_roots(tmp_path: Path, operation: str) -> None:
    home = tmp_path / "home"
    first, second = tmp_path / "first", tmp_path / "second"
    _run(home, first, "update", updates={"launch": {"status": "launching"}})

    result = _run(home, second, operation)

    if operation == "load":
        assert result == {}
    else:
        assert result == {"outcome": "absent", "payload": {}, "error": ""}
    assert not second.exists()


def test_project_audits_use_only_the_selected_config_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first, second = tmp_path / "first", tmp_path / "second"
    _run(home, first, "update", updates={"launch_state": "reserved"})
    _run(home, second, "update", updates={"launch": {"status": "launching"}})
    _run(home, first, "update", project="other", updates={"launch": {}})

    assert _run(home, first, "audit") == {
        "outcome": "not_submitted", "ledger_count": 1, "error": ""
    }
    assert _run(home, second, "audit") == {
        "outcome": "launch_evidence", "ledger_count": 1, "error": ""
    }
    assert _run(home, tmp_path / "empty", "audit") == {
        "outcome": "absent", "ledger_count": 0, "error": ""
    }


@pytest.mark.parametrize("same_root", [False, True], ids=["distinct-roots", "same-root"])
def test_submission_locks_are_scoped_to_config_root(tmp_path: Path, same_root: bool) -> None:
    home = tmp_path / "home"
    first = tmp_path / "first"
    contender = first if same_root else tmp_path / "second"

    with _held_lock(home, first):
        assert _run(home, contender, "probe_lock") == {"acquired": not same_root}

    assert _run(home, contender, "probe_lock") == {"acquired": True}


@pytest.mark.parametrize("root", [None, "", " \t "], ids=["unset", "empty", "whitespace"])
def test_default_config_keeps_home_submission_state(tmp_path: Path, root: str | None) -> None:
    home = tmp_path / "home"
    expected = home / ".npa"
    paths = _run(home, root, "path")
    assert Path(paths["config"]) == expected / "config.yaml"
    assert Path(paths["receipt"]) == expected / "workflow-submissions" / "demo" / "same-run.json"
    _run(home, root, "update", updates={"launch_state": "reserved"})
    assert _run(home, root, "inspect")["outcome"] == "found"
    assert _run(home, root, "audit")["outcome"] == "not_submitted"


def test_submission_state_reuses_selected_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected"
    ambient = tmp_path / "ambient"
    monkeypatch.setattr(config, "CONFIG_PATH", selected / "config.yaml")
    monkeypatch.setenv("NPA_CONFIG_DIR", str(ambient))

    submission_state.update_submission_state("demo", "same-run", {"launch_state": "reserved"})

    assert submission_state.submission_state_path("demo", "same-run") == (
        selected / "workflow-submissions" / "demo" / "same-run.json"
    )
    assert not ambient.exists()
