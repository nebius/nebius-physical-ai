from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = (
    ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "bdd100k-pipeline.yaml"
)
WRAPPER_PATH = ROOT / "npa" / "scripts" / "run_bdd100k_pipeline.py"

EXPECTED_TASK_ORDER = [
    "bdd100k-ingest",
    "bdd100k-backfill-cpu",
    "bdd100k-backfill-clip",
    "bdd100k-create-mvs",
    "bdd100k-train-rider",
    "bdd100k-train-nighttime",
    "bdd100k-train-distant",
    "bdd100k-eval-rider",
    "bdd100k-eval-nighttime",
    "bdd100k-eval-distant",
    "bdd100k-fiftyone-app",
]
EXPECTED_YAML_SHA256 = "edbedfd91a380543345da0d43c43c7fa55894c2474bd2434171d4876ba2243cf"
EXPECTED_LANCEDB_IMAGE = "docker:cr.eu-north1.nebius.cloud/<your-registry-id>/npa-lancedb:0.30.3"
EXPECTED_DETECTION_IMAGE = (
    "docker:cr.eu-north1.nebius.cloud/<your-registry-id>/"
    "npa-detection-training:bdd100k-golden-eval-smoke-20260614T210000Z"
)
SYNTHETIC_BDD100K_LABEL_MAP = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "bike": 7,
    "traffic light": 8,
    "traffic sign": 9,
}
REAL_BDD100K_LABEL_MAP = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}


def _docs() -> list[dict]:
    return [doc for doc in yaml.safe_load_all(YAML_PATH.read_text(encoding="utf-8")) if doc is not None]


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_bdd100k_pipeline", WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bdd100k_pipeline_yaml_has_expected_logical_stages_and_resources() -> None:
    docs = _docs()
    assert docs[0] == {"name": "bdd100k-pipeline", "execution": "serial"}
    tasks = docs[1:]
    assert [task["name"] for task in tasks] == EXPECTED_TASK_ORDER

    logical_stage_counts = {
        "ingest": 1,
        "cpu_backfill": 1,
        "clip_backfill": 1,
        "materialized_views": 1,
        "training": 3,
        "evaluation": 3,
        "fiftyone_app": 1,
    }
    assert len(tasks) == sum(logical_stage_counts.values())

    by_name = {task["name"]: task for task in tasks}
    for name in ("bdd100k-ingest", "bdd100k-backfill-cpu", "bdd100k-create-mvs"):
        assert by_name[name]["resources"]["cloud"] == "kubernetes"
        assert by_name[name]["resources"]["cpus"] == 4
        assert by_name[name]["resources"]["memory"] == 16

    clip = by_name["bdd100k-backfill-clip"]["resources"]
    assert clip == {
        "cloud": "kubernetes",
        "accelerators": "H100:1",
        "cpus": 8,
        "memory": 32,
        "image_id": EXPECTED_LANCEDB_IMAGE,
    }

    for name in ("bdd100k-train-rider", "bdd100k-train-nighttime", "bdd100k-train-distant"):
        resources = by_name[name]["resources"]
        assert resources["cloud"] == "kubernetes"
        assert resources["accelerators"] == "H100:1"
        assert resources["cpus"] == 16
        assert resources["memory"] == 64
        assert resources["image_id"] == EXPECTED_DETECTION_IMAGE

    for name in ("bdd100k-eval-rider", "bdd100k-eval-nighttime", "bdd100k-eval-distant"):
        resources = by_name[name]["resources"]
        assert resources["cloud"] == "kubernetes"
        assert resources["accelerators"] == "H100:1"
        assert resources["cpus"] == 8
        assert resources["memory"] == 32
        assert resources["image_id"] == EXPECTED_DETECTION_IMAGE

    fiftyone = by_name["bdd100k-fiftyone-app"]
    assert fiftyone["resources"] == {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": 16,
        "ports": 5151,
        "image_id": "docker:cr.eu-north1.nebius.cloud/<your-registry-id>/npa-fiftyone:<fiftyone-image-tag>",
    }
    assert fiftyone["envs"]["FIFTYONE_DEFAULT_APP_ADDRESS"] == "0.0.0.0"
    assert fiftyone["envs"]["FIFTYONE_DEFAULT_APP_PORT"] == "5151"


def test_bdd100k_pipeline_wrapper_renders_run_id_and_submits_in_order(monkeypatch, tmp_path, capsys) -> None:
    wrapper = _load_wrapper_module()
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(sky_bin.stat().st_mode | stat.S_IXUSR)
    captured = {}

    def fake_submit_workflow(yaml_path, run_id, **kwargs):
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        captured["docs"] = [doc for doc in yaml.safe_load_all(Path(yaml_path).read_text(encoding="utf-8")) if doc is not None]
        return wrapper.WorkflowResult(status="SUBMITTED", job_id="42", returncode=0, log_paths={"config": str(tmp_path / "config.yaml")})

    def fake_workflow_status(job_id, **kwargs):
        return wrapper.WorkflowResult(status="SUCCEEDED", job_id=job_id, returncode=0)

    monkeypatch.setattr(wrapper, "submit_workflow", fake_submit_workflow)
    monkeypatch.setattr(wrapper, "workflow_status", fake_workflow_status)

    rc = wrapper.main(
        [
            "--yaml",
            str(YAML_PATH),
            "--run-id",
            "bdd100k-test-run",
            "--synthetic",
            "5000",
            "--sky-bin",
            str(sky_bin),
            "--poll-interval",
            "0",
        ]
    )

    assert rc == 0
    capsys.readouterr()
    assert captured["run_id"] == "bdd100k-test-run"
    assert [doc["name"] for doc in captured["docs"][1:]] == EXPECTED_TASK_ORDER
    for doc in captured["docs"][1:]:
        envs = doc["envs"]
        assert envs["NPA_PIPELINE_RUN_ID"] == "bdd100k-test-run"
        assert envs["S3_PREFIX"] == "bdd100k-pipeline/bdd100k-test-run"
        assert envs["LANCE_URI"] == f"s3://{wrapper.DEFAULT_BUCKET}/bdd100k-pipeline/bdd100k-test-run/lancedb/"
        if "BDD100K_SYNTHETIC_ROWS" in envs:
            assert envs["BDD100K_SYNTHETIC_ROWS"] == "5000"


def test_bdd100k_pipeline_mock_endpoint_validation(capsys, tmp_path) -> None:
    wrapper = _load_wrapper_module()
    output = tmp_path / "mock.json"

    rc = wrapper.main(
        [
            "--yaml-path",
            str(YAML_PATH),
            "--mock-endpoints",
            "--run-id",
            "bdd100k-mock-run",
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    capsys.readouterr()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert [item["path"] for item in summary["lancedb_requests"] if item["method"] == "POST"] == [
        "/import-bdd100k",
        "/backfill",
        "/backfill",
        "/backfill",
        "/backfill",
        "/backfill",
        "/backfill",
        "/create-mv",
        "/create-mv",
        "/create-mv",
    ]
    assert [item["path"] for item in summary["detection_requests"] if item["method"] == "POST"] == [
        "/train",
        "/train",
        "/train",
        "/eval",
        "/eval",
        "/eval",
    ]
    train_payloads = [item["payload"] for item in summary["detection_requests"] if item["path"] == "/train"]
    assert len(train_payloads) == 3
    for payload in train_payloads:
        assert "num_classes" not in payload
        assert payload["label_map"] == SYNTHETIC_BDD100K_LABEL_MAP


def test_bdd100k_pipeline_documents_synthetic_and_real_label_maps() -> None:
    text = YAML_PATH.read_text(encoding="utf-8")
    synthetic_line = f"  BDD100K_LABEL_MAP: '{json.dumps(SYNTHETIC_BDD100K_LABEL_MAP, separators=(',', ':'))}'"
    real_line = f"  # BDD100K_LABEL_MAP: '{json.dumps(REAL_BDD100K_LABEL_MAP, separators=(',', ':'))}'"

    assert text.count("# Synthetic BDD100K data - category names match the synthetic data generator.") == 3
    assert text.count(synthetic_line) == 3
    assert text.count("# Real BDD100K data - uncomment the line below and comment the line above.") == 3
    assert text.count(real_line) == 3


def test_bdd100k_pipeline_yaml_snapshot_hash() -> None:
    digest = hashlib.sha256(YAML_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_YAML_SHA256
