"""Exercise source integrity, cycle isolation, runtime routing, and failure publication."""

import json
from pathlib import Path

import pytest
import yaml

from npa.workflows.antioch_posttrain import artifacts, dataset, runner
from npa.orchestration.npa_workflow.skypilot_render import tool_image_key, tool_vendor_interpreters


def _frames():
    rows = []
    for cycle in range(6):
        for phase in dataset.PHASE_LABELS:
            rows.append({"frame": len(rows), "sim_s": (len(rows) + 1) / 30,
                         "phase": phase, "placed": cycle + int(phase == "RETRACT")})
    rows.append({"frame": len(rows), "sim_s": (len(rows) + 1) / 30,
                 "phase": "BATCH_COMPLETE", "placed": 6})
    return rows


def test_entire_carton_cycles_are_disjoint_including_post_placement_retract():
    records = dataset.frame_records(_frames(), 1)
    for split, cycles in dataset.SPLIT_CYCLES.items():
        selected = [row for row in records if row["split"] == split]
        assert {row["cycle"] for row in selected} == set(cycles)
        assert {row["label"] for row in selected} == set(range(5))
    assert next(row for row in records if row["placed"] == 5 and row["phase"] == "RETRACT")["split"] == "validation"
    assert records[-1]["split"] == "test"


@pytest.mark.parametrize("mutation", [
    {"frame": 7}, {"sim_s": float("nan")}, {"sim_s": -1},
    {"phase": "INVENTED"}, {"placed": 9}, {"phase": "BATCH_COMPLETE"},
])
def test_invalid_frame_telemetry_fails_closed(mutation):
    frames = _frames()
    frames[0].update(mutation)
    with pytest.raises(ValueError):
        dataset.frame_records(frames, 1)


@pytest.mark.parametrize("stride", [0, -1])
def test_invalid_sampling_stride_is_rejected(stride):
    with pytest.raises(ValueError):
        dataset.frame_records(_frames(), stride)


def test_missing_class_in_any_split_is_rejected():
    records = dataset.frame_records(_frames(), 1)
    records = [row for row in records if not (row["split"] == "test" and row["label"] == 2)]
    with pytest.raises(ValueError, match="operation classes"):
        dataset._check_splits(records)


def test_checksum_corruption_prevents_consuming_stage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "measurements.json").write_text("{}")
    target = tmp_path / "published"
    artifacts.publish(source, str(target))
    assert artifacts.materialize(str(target), tmp_path / "scratch") == target
    (target / "measurements.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifacts.materialize(str(target), tmp_path / "scratch")


def test_symlink_cannot_enter_published_bundle(tmp_path):
    (tmp_path / "real").write_text("data")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValueError, match="symlinks"):
        artifacts.tree_hashes(tmp_path)


def test_s3_completion_manifest_is_last_and_readback_is_verified(tmp_path, monkeypatch):
    (tmp_path / "data").write_bytes(b"native data")
    uploaded = {}
    class Client:
        def upload_file(self, file, uri):
            uploaded[uri] = Path(file).read_bytes()

        def read_bytes_with_etag(self, uri):
            return uploaded[uri], "etag"
    monkeypatch.setattr(artifacts.StorageClient, "from_environment", lambda: Client())
    artifacts.publish(tmp_path, "s3://example-bucket/test/")
    assert list(uploaded)[-1].endswith("checksums.json")
    assert json.loads(uploaded[list(uploaded)[-1]])["data"] == artifacts.file_hash(tmp_path / "data")


def test_failed_quality_report_is_published_with_nonzero_exit(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_text("input")
    sealed = tmp_path / "sealed"
    artifacts.publish(source, str(sealed))
    def failed_stage(args, source, output, root):
        artifacts.write_json(output / "evaluation.json", {"all_passed": False})
        return False
    monkeypatch.setattr(runner, "_execute", failed_stage)
    result = runner.main(["evaluate", "--input-path", str(sealed), "--checkpoint-path", "unused",
                          "--output-path", str(tmp_path / "report")])
    assert result == 1
    assert json.loads((tmp_path / "report/evaluation.json").read_text()) == {"all_passed": False}


@pytest.mark.parametrize("stage", ["prepare", "train", "evaluate"])
def test_stages_use_existing_torch_runtime_and_vendor_interpreter(stage):
    reference = f"workflow.antioch_posttrain.{stage}"
    assert tool_image_key(reference) == "lerobot"
    assert tool_vendor_interpreters(reference) == ("/opt/lerobot/venv/bin/python",)


def test_workflow_pins_the_kubernetes_compatible_lerobot_variant():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "npa/src/npa/deploy/lerobot_version_manifest.json").read_text())
    digest = manifest["versions"]["0.6.0"]["image_digest"]
    spec = yaml.safe_load((root / "workflows/testing/antioch-posttrain.yaml").read_text())
    for resource in spec["resources"].values():
        assert resource["image"] == f"ghcr.io/nebius/nebius-physical-ai/npa-lerobot@{digest}"


def test_native_resnet_has_five_outputs_and_trainable_backbone():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from npa.workflows.antioch_posttrain.model import build_model

    model = build_model(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 5)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_score_reports_actual_confusion_and_macro_f1():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from npa.workflows.antioch_posttrain.model import score

    class FixedClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.tensor([[9., 0., 0., 0., 0.], [9., 0., 0., 0., 0.]])
    loader = [(torch.zeros(2, 3, 224, 224, dtype=torch.uint8), torch.tensor([0, 1]))]
    result = score(FixedClassifier(), loader, "cpu")
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"][0][0] == 1
    assert result["confusion_matrix"][1][0] == 1
    assert result["macro_f1"] == pytest.approx((2 / 3) / 5)


def test_training_refuses_cpu_before_fetching_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from npa.workflows.antioch_posttrain import training

    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "build_model", lambda **kwargs: pytest.fail("weights must not be fetched"))
    with pytest.raises(RuntimeError, match="Nebius CUDA worker"):
        training.train_model(tmp_path, tmp_path / "output", 20, 32, 42)


def test_checkpoint_dataset_mismatch_refuses_before_deserialization(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from npa.workflows.antioch_posttrain import evaluation

    artifacts.write_json(tmp_path / "data/checksums.json", {"frame": "digest"})
    artifacts.write_json(tmp_path / "checkpoint/training.json", {"dataset_checksums_sha256": "different"})
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: pytest.fail("weights must not load"))
    with pytest.raises(ValueError, match="different dataset"):
        evaluation.evaluate_model(tmp_path / "data", tmp_path / "checkpoint", tmp_path / "report")
