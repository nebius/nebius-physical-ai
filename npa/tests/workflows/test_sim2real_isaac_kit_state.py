"""Regression tests for the shared writable NVIDIA Kit state boundary."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from npa.workflows.sim2real import (
    interchange_isaac_runtime,
    interchange_teacher_runtime,
)
from npa.workflows.sim2real.isaac_kit_state import validated_kit_args


@pytest.mark.parametrize(
    "kit_args",
    (
        "",
        "--portable-root",
        "--portable-root=",
        "--portable-root relative/kit",
        "--portable-root /tmp/outside",
        "--portable-root {state}",
        "--portable-root {state}/kit --portable-root={state}/other",
        "--portable-root '{state}/kit",
    ),
)
def test_kit_args_reject_invalid_boundaries_before_isaac_import(
    tmp_path: Path, kit_args: str
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    environment = {
        "NPA_ISAAC_STATE_DIR": str(state),
        "NPA_ISAAC_KIT_ARGS": kit_args.format(state=state),
    }

    with pytest.raises(RuntimeError, match="portable|malformed"):
        validated_kit_args(environment)


def test_kit_args_preserve_exact_validated_launcher_string(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    arguments = f"--portable-root={state / 'kit'} --/structuredLog/enable=false"

    assert (
        validated_kit_args(
            {
                "NPA_ISAAC_STATE_DIR": str(state),
                "NPA_ISAAC_KIT_ARGS": arguments,
            }
        )
        == arguments
    )


@pytest.mark.parametrize(
    ("runtime", "runtime_arguments"),
    (
        (
            interchange_teacher_runtime,
            (
                "--train-episodes",
                "1",
                "--validation-episodes",
                "1",
                "--gold-episodes",
                "1",
            ),
        ),
        (
            interchange_isaac_runtime,
            ("--service-receipt-path", "service.json", "--decisions", "1"),
        ),
    ),
)
def test_generalized_runtimes_forward_validated_kit_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime: types.ModuleType,
    runtime_arguments: tuple[str, ...],
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    arguments = f"--portable-root {state / 'kit'} --/structuredLog/enable=false"
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", arguments)
    calls: list[dict[str, object]] = []

    class App:
        def close(self) -> None:
            return None

    class AppLauncher:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            self.app = App()

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = AppLauncher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setattr(runtime, "run", lambda _args: {"complete": True})
    output = tmp_path / runtime.__name__.rsplit(".", 1)[-1]
    common = (
        "--input-path",
        "input.json",
        "--preparation-path",
        "preparation.json",
        "--robot-usd-path",
        "robot.usd",
        "--robot-asset-manifest-path",
        "robot.json",
        "--output-path",
        str(output),
        "--seed",
        "1",
    )

    assert runtime.main([*common, *runtime_arguments]) == 0
    assert calls == [{"headless": True, "enable_cameras": True, "kit_args": arguments}]


def test_generalized_runtime_rejects_state_escape_before_app_launcher_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", "--portable-root /opt/isaac-cache/kit")
    monkeypatch.delitem(sys.modules, "isaaclab.app", raising=False)

    with pytest.raises(RuntimeError, match="inside NPA_ISAAC_STATE_DIR"):
        interchange_teacher_runtime.main(
            [
                "--input-path",
                "input.json",
                "--preparation-path",
                "preparation.json",
                "--robot-usd-path",
                "robot.usd",
                "--robot-asset-manifest-path",
                "robot.json",
                "--output-path",
                str(tmp_path / "output"),
                "--train-episodes",
                "1",
                "--validation-episodes",
                "1",
                "--gold-episodes",
                "1",
                "--seed",
                "1",
            ]
        )
    assert "isaaclab.app" not in sys.modules
