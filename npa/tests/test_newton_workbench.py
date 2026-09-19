"""Tests for the Newton physics engine workbench surface (issue #499).

Covers:
- workbench CLI wrappers import correctly via ``npa._sdk.make_cli_wrapper``
- the ``npa.cli.newton`` Typer app registers the three stub commands
- the workflow module exposes ``train_teacher`` / ``generate_demos`` / ``evaluate``
  with argument validation, stub-manifest plumbing, and clear
  not-yet-implemented errors
- real validation: a double-pendulum XPBD simulation on CPU using the
  installed Newton 1.2.1 package (no GPU required)
"""

from __future__ import annotations

import json

import pytest

typer = pytest.importorskip("typer")


def test_workbench_wrappers_import():
    import npa.workbench.newton as wb

    for name, module, callback in [
        ("train_teacher", "npa.cli.newton", "train_teacher_cmd"),
        ("generate_demos", "npa.cli.newton", "generate_demos_cmd"),
        ("eval", "npa.cli.newton", "eval_cmd"),
    ]:
        wrapper = getattr(wb, name)
        assert callable(wrapper)
        assert wrapper.__npa_cli_module__ == module
        assert wrapper.__npa_cli_callback__ == callback
        assert wrapper.__name__ == callback.removesuffix("_cmd")
    assert set(wb.__all__) == {"train_teacher", "generate_demos", "eval"}


def test_cli_app_registers_stub_commands():
    import npa.cli.newton as cli

    assert cli.app.info.name == "newton"
    command_names = {cmd.name for cmd in cli.app.registered_commands}
    assert command_names == {"train-teacher", "generate-demos", "eval"}


def test_cli_stubs_exit_not_implemented():
    from typer.testing import CliRunner

    import npa.cli.newton as cli

    runner = CliRunner()
    cases = [
        [
            "train-teacher",
            "--dataset-uri",
            "file:///tmp/ds",
            "--output-uri",
            "file:///tmp/out.json",
        ],
        [
            "generate-demos",
            "--checkpoint-uri",
            "file:///tmp/ckpt",
            "--output-uri",
            "file:///tmp/out.json",
        ],
        [
            "eval",
            "--checkpoint-uri",
            "file:///tmp/ckpt",
            "--dataset-uri",
            "file:///tmp/ds",
            "--output-uri",
            "file:///tmp/out.json",
        ],
    ]
    for argv in cases:
        result = runner.invoke(cli.app, argv)
        assert result.exit_code == 2, (
            f"{argv[0]} exited {result.exit_code}: {result.output}"
        )
        assert "not yet implemented" in result.output


def test_workflow_module_functions_exist():
    from npa.workflows.byof import newton_pipeline as pipe

    for name in ("train_teacher", "generate_demos", "evaluate"):
        assert callable(getattr(pipe, name)), f"missing {name}"
    assert issubclass(pipe.NewtonPipelineError, RuntimeError)
    pytest.importorskip("newton", reason="Newton package not installed")
    assert pipe.newton_version() == "1.2.1"


def _expect_stub(tmp_path, schema, func, **kwargs):
    from npa.workflows.byof import newton_pipeline as pipe

    out = tmp_path / "manifest.json"
    with pytest.raises(pipe.NewtonPipelineError, match="not yet implemented"):
        func(output_uri=f"file://{out}", **kwargs)
    manifest = json.loads(out.read_text())
    assert manifest["schema"] == schema
    assert manifest["status"] == "not_implemented"
    assert "499" in manifest["issue"]
    return manifest


def test_workflow_train_teacher_stub(tmp_path):
    from npa.workflows.byof import newton_pipeline as pipe

    manifest = _expect_stub(
        tmp_path,
        pipe.STUB_SCHEMA_TRAIN,
        pipe.train_teacher,
        dataset_uri="file:///tmp/ds",
        train_steps=5,
        seed=1,
    )
    assert manifest["inputs"]["train_steps"] == 5
    with pytest.raises(pipe.NewtonPipelineError, match="train_steps"):
        pipe.train_teacher(
            dataset_uri="file:///tmp/ds", output_uri="file:///tmp/o.json", train_steps=0
        )


def test_workflow_generate_demos_stub(tmp_path):
    from npa.workflows.byof import newton_pipeline as pipe

    manifest = _expect_stub(
        tmp_path,
        pipe.STUB_SCHEMA_DEMOS,
        pipe.generate_demos,
        checkpoint_uri="file:///tmp/ckpt",
        num_demos=3,
    )
    assert manifest["inputs"]["num_demos"] == 3
    with pytest.raises(pipe.NewtonPipelineError, match="checkpoint_uri"):
        pipe.generate_demos(checkpoint_uri="", output_uri="file:///tmp/o.json")


def test_workflow_eval_stub(tmp_path):
    from npa.workflows.byof import newton_pipeline as pipe

    manifest = _expect_stub(
        tmp_path,
        pipe.STUB_SCHEMA_EVAL,
        pipe.evaluate,
        checkpoint_uri="file:///tmp/ckpt",
        dataset_uri="file:///tmp/ds",
        num_episodes=2,
    )
    assert manifest["inputs"]["num_episodes"] == 2


def test_workflow_argparse_entrypoint():
    from npa.workflows.byof import newton_pipeline as pipe

    parser = pipe.build_parser()
    args = parser.parse_args(
        [
            "train-teacher",
            "--dataset-uri",
            "file:///tmp/ds",
            "--output-uri",
            "file:///tmp/o.json",
        ]
    )
    assert args.stage == "train-teacher"
    assert args.train_steps == 100
    with pytest.raises(pipe.NewtonPipelineError, match="not yet implemented"):
        pipe.main(
            [
                "eval",
                "--checkpoint-uri",
                "file:///tmp/ckpt",
                "--dataset-uri",
                "file:///tmp/ds",
                "--output-uri",
                "file:///tmp/o.json",
            ]
        )


def test_newton_double_pendulum_xpbd_cpu():
    """Real validation: double-pendulum XPBD sim on CPU with Newton 1.2.1.

    Two capsule links joined by revolute joints (world->link1, link1->link2),
    displaced from rest, stepped with SolverXPBD on the CPU device.  Asserts
    the bodies move under gravity and no NaN appears in the state.
    """
    import math

    wp = pytest.importorskip("warp", reason="Warp package not installed")
    newton = pytest.importorskip("newton", reason="Newton package not installed")

    wp.init()
    device = wp.get_device("cpu")

    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    link_a = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity())
    )
    link_b = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())
    )
    builder.add_shape_capsule(link_a, radius=0.05, half_height=0.5)
    builder.add_shape_capsule(link_b, radius=0.05, half_height=0.5)

    joint_0 = builder.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
    )
    joint_1 = builder.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
    )
    builder.add_articulation([joint_0, joint_1])
    builder.joint_q[0] = 0.3
    builder.joint_q[1] = 0.6

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    start_positions = state_0.body_q.numpy()[:, :3].copy()
    for _ in range(240):  # 1 simulated second at 240 Hz
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    end_positions = state_0.body_q.numpy()[:, :3]
    assert all(math.isfinite(x) for x in end_positions.ravel()), "NaN in body positions"
    displacement = float(abs(end_positions - start_positions).max())
    assert displacement > 1e-3, (
        f"pendulum did not swing (max displacement {displacement})"
    )
