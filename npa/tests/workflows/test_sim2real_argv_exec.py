"""Unit tests for the argv-based (non-shell) component command execution path.

Covers issue #488: workflow component commands must execute via argv lists
with no shell involved, so shell metacharacters in operator-supplied commands
are inert argument data rather than an injection surface.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import npa.workflows.sim2real.engine as engine_module
import npa.workflows.sim2real.legacy_heldout as heldout_module
from npa.workflows.sim2real.legacy_components import (
    Sim2RealLoopError,
    _redact_command,
    _run_component_command,
    _split_component_command,
)
from npa.workflows.sim2real.models import Sim2RealLoopConfig


def _minimal_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


# --- _split_component_command ------------------------------------------------


def test_split_basic_command() -> None:
    argv, env = _split_component_command(
        "python3 -m npa.workflows.sim2real.byo_isaac_eval",
        component="heldout_eval",
    )
    assert argv == ["python3", "-m", "npa.workflows.sim2real.byo_isaac_eval"]
    assert env == {}


def test_split_quoting_preserves_spaces() -> None:
    argv, env = _split_component_command(
        "python3 script.py --name 'a b' --flag \"c d\"",
        component="test",
    )
    assert argv == ["python3", "script.py", "--name", "a b", "--flag", "c d"]
    assert env == {}


def test_split_shell_metacharacters_are_literal() -> None:
    """Pipes, redirects, && chains and $(...) stay literal argument text."""
    argv, env = _split_component_command(
        "run.sh 'a;b' \"c|d\" e&&f $(g) `h` i>j",
        component="test",
    )
    assert argv == ["run.sh", "a;b", "c|d", "e&&f", "$(g)", "`h`", "i>j"]
    assert env == {}


def test_split_leading_env_assignments_become_overrides() -> None:
    """The shell's `VAR=x cmd` prefix works without a shell; values are literal."""
    argv, env = _split_component_command(
        "FOO=bar BAZ='a b' EMPTY= python3 run.py",
        component="test",
    )
    assert argv == ["python3", "run.py"]
    assert env == {"FOO": "bar", "BAZ": "a b", "EMPTY": ""}


def test_split_env_assignment_only_after_argv_start_is_literal() -> None:
    argv, env = _split_component_command(
        "python3 run.py FOO=bar",
        component="test",
    )
    assert argv == ["python3", "run.py", "FOO=bar"]
    assert env == {}


def test_split_invalid_assignment_name_is_literal() -> None:
    argv, env = _split_component_command(
        "1FOO=bar python3 run.py",
        component="test",
    )
    assert argv == ["1FOO=bar", "python3", "run.py"]
    assert env == {}


@pytest.mark.parametrize("command", ["", "   ", "\t\n ", "FOO=bar"])
def test_split_empty_command_raises(command: str) -> None:
    with pytest.raises(Sim2RealLoopError, match="command is empty"):
        _split_component_command(command, component="test")


# --- _run_component_command --------------------------------------------------


def test_run_executes_argv(tmp_path: Path) -> None:
    invocation = _run_component_command(
        [sys.executable, "-c", "print('hello argv')"],
        cwd=tmp_path,
        env=_minimal_env(),
        component="test",
    )
    assert invocation["mode"] == "command"
    assert invocation["component"] == "test"
    assert invocation["returncode"] == 0
    assert "hello argv" in invocation["stdout"]


def test_run_rejects_shell_string(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="argv list, not a shell string"):
        _run_component_command(
            "echo should-not-run",  # type: ignore[arg-type]
            cwd=tmp_path,
            env=_minimal_env(),
            component="test",
        )


def test_run_failure_raises_with_component(tmp_path: Path) -> None:
    with pytest.raises(Sim2RealLoopError, match="boom-comp"):
        _run_component_command(
            [sys.executable, "-c", "raise SystemExit(3)"],
            cwd=tmp_path,
            env=_minimal_env(),
            component="boom-comp",
        )


def test_run_argv_injection_safe(tmp_path: Path) -> None:
    """Metacharacters in argv must be inert: echoed back literally, never run."""
    marker = tmp_path / "pwned-by-shell"
    nasty = [
        "; touch " + str(marker),
        "$(touch " + str(tmp_path / "pwned2") + ")",
        "`touch " + str(tmp_path / "pwned3") + "`",
        "a|b",
        "c&&d",
        'it\'s "quoted"',
        "line1\nline2",
    ]
    invocation = _run_component_command(
        [
            sys.executable,
            "-c",
            "import sys, json; print(json.dumps(sys.argv[1:]))",
            *nasty,
        ],
        cwd=tmp_path,
        env=_minimal_env(),
        component="injection-probe",
    )
    received = json.loads(invocation["stdout"])
    assert received == nasty
    assert not marker.exists()


def test_run_env_overrides_reach_process(tmp_path: Path) -> None:
    """Env assignments split from the command prefix land in the process env."""
    argv, env_overrides = _split_component_command(
        "NPA_TEST_MARKER=hello python3 run.py",
        component="test",
    )
    env = _minimal_env()
    env.update(env_overrides)
    invocation = _run_component_command(
        [sys.executable, "-c", "import os; print(os.environ['NPA_TEST_MARKER'])"],
        cwd=tmp_path,
        env=env,
        component="test",
    )
    assert argv == ["python3", "run.py"]
    assert invocation["stdout"].strip() == "hello"


# --- _redact_command ---------------------------------------------------------


def test_redact_command_argv_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "super-secret-token")
    redacted = _redact_command(["python3", "run.py", "--token", "super-secret-token"])
    assert "super-secret-token" not in redacted
    assert "<HF_TOKEN>" in redacted
    assert "run.py" in redacted


def test_redact_command_str_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NGC_API_KEY", "ngc-secret")
    assert _redact_command("curl -H ngc-secret") == "curl -H <NGC_API_KEY>"


# --- real caller migration ---------------------------------------------------


def test_byo_policy_command_reaches_runner_as_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The policy BYO hook (a real caller) now passes argv, not a shell string."""
    captured: dict = {}

    def _fake_run_component_command(command, *, cwd, env, component, **kwargs):
        captured["command"] = command
        captured["component"] = component
        captured["env"] = env
        return {"ok": True}

    monkeypatch.setattr(
        engine_module, "_run_component_command", _fake_run_component_command
    )
    monkeypatch.setattr(
        engine_module, "_read_component_json", lambda path, invocation: {}
    )
    config = Sim2RealLoopConfig(
        run_id="r",
        byo_policy_command=(
            "NPA_TEST_HOOK=1 python3 -m some.module --flag 'a b' --evil '$(x)'"
        ),
    )
    engine_module._run_policy_rollouts_via_command(
        config,
        actions_dir=tmp_path,
        outer_iteration=0,
        iteration=0,
        train_envs_uri="",
        checkpoint_uri="",
    )
    assert captured["component"] == "policy_actions"
    assert captured["command"] == [
        "python3",
        "-m",
        "some.module",
        "--flag",
        "a b",
        "--evil",
        "$(x)",
    ]
    assert captured["env"]["NPA_TEST_HOOK"] == "1"


class _Sentinel(Exception):
    pass


def _capture_heldout_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: Sim2RealLoopConfig
) -> tuple:
    captured: dict = {}

    def _fake_run_component_command(command, *, cwd, env, component, **kwargs):
        captured["command"] = command
        captured["env"] = env
        raise _Sentinel()

    monkeypatch.setattr(
        engine_module, "_run_component_command", _fake_run_component_command
    )
    monkeypatch.setattr(heldout_module, "_heldout_k8s_image_ready", lambda config: True)
    with pytest.raises(_Sentinel):
        heldout_module.run_heldout_eval(
            config,
            local_dir=tmp_path,
            inner_evidence={},
            outer_iteration=0,
        )
    return captured["command"], captured["env"]


def test_heldout_default_eval_uses_literal_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The internally generated Isaac eval default is argv from the start."""
    config = Sim2RealLoopConfig(
        run_id="r",
        sim_backend="isaac",
        byo_trainer_command="python3 -m some.trainer",
        s3_bucket="bucket",
    )
    argv, _env = _capture_heldout_argv(monkeypatch, tmp_path, config)
    assert argv == ["python3", "-m", "npa.workflows.sim2real.byo_isaac_eval"]


def test_heldout_byo_eval_string_split_to_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An operator BYO eval string is split to argv before execution."""
    config = Sim2RealLoopConfig(
        run_id="r",
        sim_backend="isaac",
        byo_trainer_command="python3 -m some.trainer",
        s3_bucket="bucket",
        byo_eval_command="MARKER_ENV=abc python3 -m my_eval --opt 'x y' --chain 'a&&b'",
    )
    argv, env = _capture_heldout_argv(monkeypatch, tmp_path, config)
    assert argv == ["python3", "-m", "my_eval", "--opt", "x y", "--chain", "a&&b"]
    assert env["MARKER_ENV"] == "abc"
