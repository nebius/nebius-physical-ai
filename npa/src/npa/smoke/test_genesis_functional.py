"""Standalone Genesis functional smoke checks.

This script builds a small scene with a Franka Panda, a plane, and a box,
then steps the simulation and reads back object state.

Run with:
    python -m npa.smoke.test_genesis_functional
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable

from npa.smoke._versions import supported_tool_version


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    gs: Any | None = None
    scene: Any | None = None
    box: Any | None = None
    franka: Any | None = None


def _package_version(module, distribution_name: str) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _flatten_numbers(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    out: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            out.append(float(item))

    visit(value)
    return out


def check_import_genesis(state: SmokeState) -> CheckResult:
    try:
        expected = supported_tool_version("genesis", __file__)
        state.gs = importlib.import_module("genesis")
        version = _package_version(state.gs, "genesis-world")
        if version != expected:
            return CheckResult(
                "import genesis",
                False,
                f"expected version: {expected}; found: {version}",
            )
        return CheckResult("import genesis", True, f"version: {version}")
    except Exception as exc:
        return CheckResult("import genesis", False, _format_exception(exc))


def check_build_franka_scene(state: SmokeState) -> CheckResult:
    if state.gs is None:
        return CheckResult("build Franka scene", False, "skipped because genesis import failed")

    try:
        gs = state.gs
        init_kwargs = {"logging_level": "warning"}
        if hasattr(gs, "cpu"):
            init_kwargs["backend"] = gs.cpu
        gs.init(**init_kwargs)

        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        box = scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0.0, 0.05))
        )
        franka = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.0))
        )
        scene.build()
        state.scene = scene
        state.box = box
        state.franka = franka
        return CheckResult("build Franka scene", True, "plane + box + Franka Panda built")
    except Exception as exc:
        return CheckResult("build Franka scene", False, _format_exception(exc))


def check_step_scene(state: SmokeState) -> CheckResult:
    if state.scene is None:
        return CheckResult("step scene 10 times", False, "skipped because scene build failed")

    try:
        for _ in range(10):
            state.scene.step()
        return CheckResult("step scene 10 times", True, "completed 10 scene.step() calls")
    except Exception as exc:
        return CheckResult("step scene 10 times", False, _format_exception(exc))


def check_read_body_state(state: SmokeState) -> CheckResult:
    if state.box is None:
        return CheckResult("read body position", False, "skipped because box entity is unavailable")

    try:
        pos = _flatten_numbers(state.box.get_pos())
        if len(pos) < 3:
            return CheckResult("read body position", False, f"expected >=3 values, got {pos}")
        if not all(math.isfinite(v) for v in pos[:3]):
            return CheckResult("read body position", False, f"non-finite position: {pos[:3]}")
        return CheckResult("read body position", True, f"box position: {pos[:3]}")
    except Exception as exc:
        return CheckResult("read body position", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    state = SmokeState()
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_import_genesis,
        check_build_franka_scene,
        check_step_scene,
        check_read_body_state,
    ]
    results: list[CheckResult] = []

    for check in checks:
        result = check(state)
        results.append(result)
        _print_result(result)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
