"""Exercise actual replay encoding, untrusted JSON boundaries, and native solver setup."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest

from npa.workflows import physical_augmentation_demo as demo
from npa.workflows.physical_augmentation_contract import make_recipe
from npa.workflows.physical_augmentation_runtime import _configure_solver


def test_contact_resolution_preserves_control_and_camera_frequency():
    from npa.workflows.physical_augmentation_runtime import _configure_clock

    config = NS(sim=NS(dt=0.01, render_interval=2), decimation=2)
    _configure_clock(config, make_recipe("test", 42, 1, 600))
    assert config.sim.dt == 0.0025
    assert config.decimation == config.sim.render_interval == 8
    assert config.sim.dt * config.decimation == 0.02


def test_solver_enables_native_velocity_iterations_without_relaxing_acceptance():
    robot, object_body = NS(), NS()
    config = NS(
        sim=NS(physics=NS(solver_type=1)),
        scene=NS(
            robot=NS(spawn=NS(articulation_props=robot)),
            object=NS(spawn=NS(rigid_props=object_body)),
        ),
    )
    recipe = make_recipe("test", 42, 1, 600)
    _configure_solver(config, recipe)
    assert (
        robot.solver_velocity_iteration_count
        == object_body.solver_velocity_iteration_count
        == 1
    )
    assert (
        robot.solver_position_iteration_count
        == object_body.solver_position_iteration_count
        == 16
    )
    assert config.sim.physics.enable_external_forces_every_iteration is True
    assert recipe["success"]["max_speed_m_s"] == 0.03
    config.sim.physics.solver_type = 0
    with pytest.raises(ValueError, match="TGS"):
        _configure_solver(config, recipe)


def test_html_keeps_script_delimiters_inside_json_data(tmp_path):
    payload = {"untrusted": '</script><script>alert("x")</script>', "value": 7}
    path = tmp_path / "demo.html"
    demo._html(payload, path)
    document = path.read_text()
    embedded = document.split('<script id="demo-data" type="application/json">')[
        1
    ].split("</script>")[0]
    assert json.loads(embedded) == payload
    assert payload["untrusted"] not in document
    assert document.count("</script>") == 2


def _recording(root, condition, count):
    folder = root / condition
    folder.mkdir()
    frames = np.zeros((count, 36, 64, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(count)[:, None, None] * 7
    arrays = {
        "rgb": frames,
        "actions": np.tile([0.5, 0, 0.2, 1, 0, 0, 0, -1], (count, 1)),
        "state": np.zeros((count, 9)),
        "object": np.tile([0.5, 0, 0.025], (count, 1)),
        "next_object": np.tile([0.5, 0, 0.2], (count, 1)),
        "next_tcp": np.tile([0.5, 0, 0.2], (count, 1)),
        "next_velocity": np.zeros((count, 3)),
    }
    if condition == "slippery":
        arrays["next_velocity"][:, 0] = 0.04
    for name, values in arrays.items():
        np.save(folder / f"{name}.npy", values)
    return {
        "condition": condition,
        "attempt": 0,
        "seed": 42,
        "source": condition,
        "success": condition != "slippery",
        "length": count,
        "initial_object_m": [0.5, 0, 0.025],
    }


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="media tools unavailable",
)
def test_real_encoding_retains_failed_trial_and_decodes_comparison(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output with spaces"
    source.mkdir()
    output.mkdir()
    recipe = make_recipe("test", 42, 1, 600)
    recipe["presentation"].update(width=64, height=36)
    (source / "recipe.json").write_text(json.dumps(recipe))
    attempts = [_recording(source, condition, 32) for condition in demo.ORDER]
    identity = {"control_dt": 0.02, "runtime_version": "test-fixture"}
    evidence = demo.build_demo(source, output, recipe, attempts, identity)
    assert evidence["attempted"] == 4 and evidence["accepted"] == 3
    assert len(evidence["videos"]) == 4
    assert int(evidence["film"]["nb_read_frames"]) == 20
    assert evidence["film"]["width"] == 1920
    document = (output / "demo.html").read_text()
    payload = json.loads(
        document.split('<script id="demo-data" type="application/json">')[1].split(
            "</script>"
        )[0]
    )
    assert payload["trials"][-1]["success"] is False
    assert payload["trials"][-1]["video"].startswith("data:video/mp4;base64,")
    assert payload["trials"][0]["telemetry"]["hold_steps"][-1] == 32
    assert payload["trials"][-1]["telemetry"]["hold_steps"][-1] == 0
    with pytest.raises(ValueError, match="differs"):
        demo._verify_video(
            output / "demo.mp4", {"length": 19}, {"width": 1920, "height": 1280}, 30
        )
