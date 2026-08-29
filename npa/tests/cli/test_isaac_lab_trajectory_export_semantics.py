from __future__ import annotations

import ast

import pytest

from npa.cli.isaac_lab import _build_train_trajectory_export_script


def _generated_tree(*, capture_rgb: bool) -> ast.Module:
    return ast.parse(
        _build_train_trajectory_export_script(
            "Isaac-Cartpole-v0",
            1,
            4,
            "/tmp/checkpoint.pt",
            "/tmp/trajectories",
            capture_rgb=capture_rgb,
        )
    )


def _step_loop(tree: ast.Module) -> ast.For:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step"
    )


def _call_lines(loop: ast.For, name: str) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(loop):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == name:
            lines.append(node.lineno)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == name:
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("capture_rgb", [False, True])
def test_generated_rollout_disables_kit_telemetry_before_launch(
    capture_rgb: bool,
) -> None:
    tree = _generated_tree(capture_rgb=capture_rgb)
    launcher = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AppLauncher"
    )
    launcher_line = launcher.lineno
    disable_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "os"
            and target.value.attr == "environ"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"
            for target in node.targets
        )
    )
    kit_args = next(
        keyword.value
        for keyword in launcher.keywords
        if keyword.arg == "kit_args"
    )
    rendered_args = ast.literal_eval(kit_args)

    assert disable_assignment.lineno < launcher_line
    assert rendered_args.startswith("--portable-root /tmp/npa-isaac-kit ")
    assert "--/structuredLog/enable=false" in rendered_args
    assert "--/telemetry/enableAnonymousData=false" in rendered_args
    assert "--/privacy/usage=false" in rendered_args
    assert "--/privacy/performance=false" in rendered_args
    assert "--/privacy/personalization=false" in rendered_args


@pytest.mark.parametrize("capture_rgb", [False, True])
def test_generated_rollout_records_pre_step_timeline(capture_rgb: bool) -> None:
    """Each exported row is state_t/action_t/(RGB_t) before step(action_t)."""

    tree = _generated_tree(capture_rgb=capture_rgb)
    loop = _step_loop(tree)
    step_line = _call_lines(loop, "_step_env")
    state_assignments = [
        node.lineno
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "state_values"
            for target in node.targets
        )
    ]
    state_appends = _call_lines(loop, "append")
    named_appends = {
        node.func.value.id: node.lineno
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"states", "actions_out", "rgb_frames"}
    }

    assert len(step_line) == 1
    assert state_assignments and max(state_assignments) < step_line[0]
    assert named_appends["states"] < step_line[0]
    assert named_appends["actions_out"] < step_line[0]
    assert state_appends
    if capture_rgb:
        assert _call_lines(loop, "_rgb_frame")[0] < step_line[0]
        assert named_appends["rgb_frames"] < step_line[0]

    capture_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "capture_rgb"
            for target in node.targets
        )
    )
    assert isinstance(capture_assignment.value, ast.Constant)
    assert capture_assignment.value.value is capture_rgb


@pytest.mark.parametrize("capture_rgb", [False, True])
def test_generated_rollout_guards_exported_timeline_lengths(capture_rgb: bool) -> None:
    tree = _generated_tree(capture_rgb=capture_rgb)
    comparisons = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Compare)]

    assert "len(actions_out) != len(states)" in comparisons
    if capture_rgb:
        assert "len(rgb_frames) != len(states)" in comparisons
