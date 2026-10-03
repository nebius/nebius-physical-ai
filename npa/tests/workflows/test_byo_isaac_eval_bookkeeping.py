"""Exercise the generated evaluator's mandatory metric and terminal bookkeeping."""

from __future__ import annotations

import ast
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from npa.workflows.sim2real import byo_isaac_eval as evaluator


class _Scene(dict):
    """Provide the scene mapping and origins used by the generated evaluator."""


def _metric_scene(torch: Any) -> _Scene:
    """Provide actual tensor state for one nearby stationary object."""

    scene = _Scene()
    scene.env_origins = torch.zeros((1, 3))
    scene["object"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.07]]),
            root_lin_vel_w=torch.zeros((1, 3)),
        )
    )
    scene["ee_frame"] = SimpleNamespace(
        data=SimpleNamespace(
            target_pos_w=torch.tensor([[[0.0, 0.0, 0.08]]]),
        )
    )
    return scene


def _metric_namespace(terminal: bool = False) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    scene = _metric_scene(torch)
    manager = SimpleNamespace(get_command=lambda name: torch.tensor([[0.0, 0.0, 0.09]]))
    namespace = {
        "env": SimpleNamespace(
            unwrapped=SimpleNamespace(scene=scene, command_manager=manager)
        ),
        "torch": torch,
        "np": np,
        "N": 1,
        "_step": 3,
        "active": np.array([True]),
        "newly_terminal": np.array([terminal]),
        "initial_obj_z": np.array([0.0]),
        "termination": np.array(["max_steps"], dtype=object),
    }
    for name in ("reach", "contact", "grasp", "lift", "place", "final_place"):
        namespace[name] = np.array([False])
    for name in ("stable_grasp_steps", "stable_place_steps", "max_stable_place_steps"):
        namespace[name] = np.array([2])
    for name in ("min_dist", "final_dist", "min_speed_in_strict_basin"):
        namespace[name] = np.array([0.01])
    namespace["prior"] = {
        name: value.copy()
        for name, value in namespace.items()
        if isinstance(value, np.ndarray)
    }
    return namespace


def _execute_generated_block(namespace: dict[str, Any], assignment: str) -> None:
    tree = ast.parse(evaluator.ISAAC_EVAL_SCRIPT)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "npa.workflows.sim2real.byo_isaac_eval"
    ]
    block = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and node.body
        and isinstance(node.body[0], ast.Assign)
        and ast.unparse(node.body[0]).startswith(assignment)
    )
    loop = ast.For(
        target=ast.Name(id="_test_once", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=[*imports, block],
        orelse=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    with tempfile.TemporaryDirectory(prefix="isaac-eval-test-") as directory:
        module_path = Path(directory) / "evaluation.py"
        module_path.write_text(ast.unparse(module), encoding="utf-8")
        namespace.update(runpy.run_path(str(module_path), init_globals=namespace))


def test_generated_capture_imports_contact_helper_and_records_placement() -> None:
    namespace = _metric_namespace()
    _execute_generated_block(namespace, "uenv = env.unwrapped")
    for name in ("reach", "contact", "grasp", "lift", "place", "final_place"):
        assert namespace[name].tolist() == [True]
    assert namespace["stable_place_steps"].tolist() == [3]
    assert namespace["final_dist"].tolist() == pytest.approx([0.02])


def test_generated_capture_restores_every_pre_reset_metric() -> None:
    namespace = _metric_namespace(terminal=True)
    _execute_generated_block(namespace, "uenv = env.unwrapped")
    for name, expected in namespace["prior"].items():
        if name != "termination":
            np.testing.assert_array_equal(namespace[name], expected)
    assert namespace["termination"].tolist() == ["task_or_timeout"]


@pytest.mark.parametrize("missing", ["object", "velocity", "commands"])
def test_generated_capture_rejects_missing_required_state(missing: str) -> None:
    namespace = _metric_namespace()
    environment = namespace["env"].unwrapped
    if missing == "object":
        del environment.scene["object"]
    elif missing == "velocity":
        del environment.scene["object"].data.root_lin_vel_w
    else:
        del environment.command_manager
    with pytest.raises(RuntimeError, match="metric capture failed at step 3"):
        _execute_generated_block(namespace, "uenv = env.unwrapped")


def test_generated_capture_does_not_hide_helper_errors(monkeypatch) -> None:
    def broken_contact(*args: Any) -> None:
        raise ValueError("contact bookkeeping failed")

    monkeypatch.setattr(evaluator, "manipulator_contact_signal", broken_contact)
    with pytest.raises(RuntimeError, match="metric capture failed") as caught:
        _execute_generated_block(_metric_namespace(), "uenv = env.unwrapped")
    assert isinstance(caught.value.__cause__, ValueError)


def test_generated_capture_rejects_unreadable_termination_flags() -> None:
    namespace = {"dones": None, "N": 1, "np": np, "_step": 3}
    with pytest.raises(RuntimeError, match="could not read episode termination"):
        _execute_generated_block(namespace, "done_np = dones.detach()")
