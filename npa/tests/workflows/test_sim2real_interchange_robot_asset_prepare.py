from __future__ import annotations

import json
from pathlib import Path

from npa.workflows.sim2real import interchange_robot_asset_prepare as stage
from npa.workflows.sim2real import isaac_robot_asset


def _spec() -> dict:
    return {
        "asset_root_uri": "s3://private/source/",
        "source_relative_path": "robot.urdf",
        "source_tree_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "source_format": "urdf",
        "embodiment_digest": "c" * 64,
        "resolved_usd_uri": "s3://private/resolved/recipe/robot.usd",
        "resolved_manifest_uri": "s3://private/resolved/recipe/manifest.json",
        "expected_action_dim": 8,
        "expected_observation_dim": None,
        "conversion_recipe": isaac_robot_asset.conversion_recipe("urdf"),
        "conversion_recipe_sha256": isaac_robot_asset.conversion_recipe_sha256("urdf"),
    }


def test_stage_converts_preparation_robot_and_publishes_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    spec = _spec()
    published: dict[str, object] = {}
    monkeypatch.setattr(
        stage,
        "read_json",
        lambda uri, *, directory: {"robot_asset": {"robot_spec": spec}},
    )
    closed: list[bool] = []
    app = type("App", (), {"close": lambda self: closed.append(True)})()
    monkeypatch.setattr(
        stage,
        "_launch_and_convert",
        lambda: (
            {
                "usd_sha256": "d" * 64,
                "asset_tree_sha256": "e" * 64,
            },
            app,
        ),
    )
    monkeypatch.setattr(
        stage,
        "write_json",
        lambda uri, payload, *, directory: published.update(
            {"uri": uri, "payload": payload}
        ),
    )
    args = stage.build_parser().parse_args(
        [
            "--run-id",
            "run-1",
            "--preparation-uri",
            "s3://private/preparation.json",
            "--output-uri",
            "s3://private/robot-asset.json",
        ]
    )

    receipt = stage.run(args)

    assert receipt["status"] == "passed"
    assert receipt["conversion_recipe_sha256"] == spec["conversion_recipe_sha256"]
    assert published == {
        "uri": "s3://private/robot-asset.json",
        "payload": receipt,
    }
    assert closed == [True]
    assert json.loads(stage.os.environ["NPA_BYO_ROBOT_SPEC_JSON"]) == spec


def test_stage_publishes_through_real_workflow_uri_contract(monkeypatch) -> None:
    spec = _spec()
    preparation_uri = "s3://private/preparation.json"
    output_uri = "s3://private/robot-asset.json"
    objects = {
        preparation_uri: json.dumps({"robot_asset": {"robot_spec": spec}}).encode()
    }

    class Storage:
        def download_file(self, uri: str, target: str) -> None:
            destination = Path(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(objects[uri])

        def upload_file(self, source: str, uri: str) -> str:
            objects[uri] = Path(source).read_bytes()
            return uri

    monkeypatch.setattr("npa.workflows.sim2real.workflow_io.storage", lambda: Storage())
    monkeypatch.setattr(
        stage,
        "_launch_and_convert",
        lambda: (
            {"usd_sha256": "d" * 64, "asset_tree_sha256": "e" * 64},
            type("App", (), {"close": lambda self: None})(),
        ),
    )
    args = stage.build_parser().parse_args(
        [
            "--run-id",
            "run-1",
            "--preparation-uri",
            preparation_uri,
            "--output-uri",
            output_uri,
        ]
    )

    receipt = stage.run(args)

    assert json.loads(objects[output_uri]) == receipt


def test_stage_rejects_preparation_without_robot_spec(monkeypatch) -> None:
    monkeypatch.setattr(stage, "read_json", lambda uri, *, directory: {})
    args = stage.build_parser().parse_args(
        [
            "--run-id",
            "run-1",
            "--preparation-uri",
            "s3://private/preparation.json",
            "--output-uri",
            "s3://private/robot-asset.json",
        ]
    )

    try:
        stage.run(args)
    except RuntimeError as error:
        assert "resolved RobotSpec" in str(error)
    else:
        raise AssertionError("missing RobotSpec was accepted")
