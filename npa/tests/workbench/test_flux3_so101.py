"""Pinned SO-101 runner and workflow contract."""

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.skypilot_render import tool_image_key
from npa.workbench.lerobot.flux3_so101.runner import _losses, run, training_command


def test_training_command_and_image_route():
    args = training_command(
        Path("/tmp/work"), Path("/tmp/policy"), Path("/tmp/base"), 4
    )
    assert "--config_path=/opt/lerobot/examples/flux3/lora.json" in args
    assert (
        '--rename_map={"observation.images.front":"observation.images.scene"}' in args
    )
    assert "--steps=4" in args
    assert "--log_freq=4" in args
    assert "--log_freq=20" in training_command(
        Path("/tmp/work"), Path("/tmp/policy"), Path("/tmp/base"), 60000
    )
    entry = TOOL_CATALOG["workbench.lerobot.flux3_so101_finetune"]
    assert entry.argv_template[:4] == [
        "npa",
        "workbench",
        "lerobot",
        "flux3-so101-finetune",
    ]
    assert tool_image_key(entry.name) == "lerobot-flux3"


def test_loss_and_input_fail_closed(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("step: 1 loss: 0.5\nstep: 2 loss: 0.3\n")
    assert _losses(log)["last"] == 0.3
    log.write_text("step: 1 loss: nan\n")
    with pytest.raises(ValueError, match="finite"):
        _losses(log)
    log.write_text("step: 1 loss: 0.5\nstep: 2 loss: nan\n")
    with pytest.raises(ValueError, match="finite"):
        _losses(log)
    with pytest.raises(ValueError, match="multiple"):
        run(output_path="s3://example/run/", run_id="smoke", steps=5)
    with pytest.raises(ValueError, match="run_id"):
        run(output_path="s3://example/run/", run_id="../escape", steps=4)


def test_publish_requires_raw_and_ema_and_checksums(tmp_path):
    import hashlib
    import json

    from npa.workbench.lerobot.flux3_so101.publish import publish

    source = tmp_path / "trainer"
    checkpoint = source / "checkpoints" / "000004"
    checkpoint.mkdir(parents=True)
    (source / "checkpoints" / "last").symlink_to("000004", target_is_directory=True)
    for variant in ("pretrained_model", "pretrained_model_ema"):
        directory = checkpoint / variant
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(variant.encode())
        (directory / "adapter_config.json").write_text("{}")
    result = tmp_path / "result"
    result.mkdir()
    record = publish(
        source, result, "calibration-v1", "black-forest-labs/flux-3-action-so101"
    )[0]
    for variant in ("pretrained_model", "pretrained_model_ema"):
        config = json.loads(
            (
                result / "checkpoints/000004" / variant / "adapter_config.json"
            ).read_text()
        )
        assert (
            config["base_model_name_or_path"] == "black-forest-labs/flux-3-action-so101"
        )
    assert record["adapter_sha256"] == hashlib.sha256(b"pretrained_model").hexdigest()
    assert (
        record["ema_adapter_sha256"]
        == hashlib.sha256(b"pretrained_model_ema").hexdigest()
    )
    assert (result / "checkpoints/000004/COMPLETE.json").is_file()


def test_cli_invokes_shared_runner(monkeypatch):
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.workbench.lerobot import flux3_so101

    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)
        return {"status": "trained", "microsteps": kwargs["steps"]}

    monkeypatch.setattr(flux3_so101, "run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "lerobot",
            "flux3-so101-finetune",
            "--output-path",
            "s3://example-bucket/runs/smoke/",
            "--run-id",
            "smoke",
            "--steps",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert called == {
        "output_path": "s3://example-bucket/runs/smoke/",
        "run_id": "smoke",
        "steps": 4,
    }


def test_remote_checkpoint_marker_is_uploaded_last(tmp_path):
    import hashlib
    import json

    from npa.workbench.lerobot.flux3_so101.runner import _upload_checkpoint

    target = tmp_path / "checkpoints/000004"
    for variant in ("pretrained_model", "pretrained_model_ema"):
        directory = target / variant
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_bytes(variant.encode())
    (target / "COMPLETE.json").write_text("{}")
    record = {
        "directory": "000004",
        "adapter_sha256": hashlib.sha256(b"pretrained_model").hexdigest(),
        "ema_adapter_sha256": hashlib.sha256(b"pretrained_model_ema").hexdigest(),
    }

    class Storage:
        objects = {}
        order = []

        def upload_file(self, local, uri):
            self.objects[uri] = Path(local).read_bytes()
            self.order.append(uri)

        def read_bytes_with_etag(self, uri):
            return self.objects[uri], "etag"

    storage = Storage()
    _upload_checkpoint(storage, tmp_path, record, "s3://example/run/")
    assert storage.order[-1].endswith("/COMPLETE.json")
    assert json.loads(storage.objects[storage.order[-1]]) == {}
