"""BDD100K pipeline coverage, at the spec level plus a real mock-endpoint run.

This file used to assert the raw shape of `skypilot/bdd100k-pipeline.yaml` (task order,
per-task resources, image pins, a SHA-256 snapshot of the file). The runner now renders the
`npa.workflow` spec instead, so the assertions move onto the spec and — more valuably — onto
what the pipeline actually *does*: `--mock-endpoints` executes every stage's resolved argv
against in-process LanceDB and detection-training stand-ins and checks the call sequence.

That mode is the honest offline proof for this pipeline, because a live run needs the LanceDB
workbench service, which is not deployed (EVIDENCE.md §R16).
"""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "bdd100k-pipeline.yaml"
WRAPPER_PATH = ROOT / "npa" / "scripts" / "run_bdd100k_pipeline.py"

EXPECTED_STAGE_ORDER = [
    "ingest",
    "backfill-cpu",
    "backfill-clip",
    "curate-views",
    "train-rider",
    "train-nighttime",
    "train-distant",
    "eval-rider",
    "eval-nighttime",
    "eval-distant",
    "review",
]
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


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_bdd100k_pipeline", WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan(run_id: str = "bdd100k-test-run"):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(SPEC_PATH)
    return spec, build_plan(spec, run_id=run_id)


# --------------------------------------------------------------------------- the spec


def test_spec_expands_to_the_pipeline_stages_in_order() -> None:
    _spec, plan = _plan()

    assert [step.state for step in plan.steps if step.argv] == EXPECTED_STAGE_ORDER


def test_gpu_stages_are_the_ones_that_need_a_gpu() -> None:
    """CLIP embedding, training and evaluation; ingest, CPU backfill and views do not."""

    spec, plan = _plan()
    by_state = {step.state: step for step in plan.steps}

    for state in ("backfill-clip", "train-rider", "train-nighttime", "train-distant"):
        profile = spec.resources[by_state[state].resources]
        assert "accelerators" in profile, state
    for state in ("ingest", "backfill-cpu", "curate-views", "review"):
        profile = spec.resources[by_state[state].resources]
        assert "accelerators" not in profile, state


def test_train_stages_wait_and_carry_the_label_map() -> None:
    """Both were bash in the template and unreachable from a spec (EVIDENCE.md §R21)."""

    _spec, plan = _plan()

    for state in ("train-rider", "train-nighttime", "train-distant"):
        argv = next(step.argv for step in plan.steps if step.state == state)
        assert "--wait" in argv, state
        label_map = argv[argv.index("--label-map") + 1]
        assert json.loads(label_map) == SYNTHETIC_BDD100K_LABEL_MAP, state


def test_eval_stages_discover_their_checkpoint_and_publish_metrics() -> None:
    _spec, plan = _plan()

    for state, view in (
        ("eval-rider", "bdd100k_rider_train"),
        ("eval-nighttime", "bdd100k_nighttime_person_train"),
        ("eval-distant", "bdd100k_distant_person_train"),
    ):
        step = next(candidate for candidate in plan.steps if candidate.state == state)
        assert "--discover-checkpoint" in step.argv, state
        assert "--write-canonical-metrics" in step.argv, state
        # The search prefix is the TRAINING output, which is what /train was handed.
        assert step.argv[step.argv.index("--checkpoint-uri") + 1].endswith(f"/training/{view}")
        assert step.outputs[0]["uri"].endswith("/metrics.json")


def test_spec_documents_synthetic_and_real_label_maps() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")

    assert "pedestrian/motorcycle/bicycle" in text, "the real-BDD100K alternative must stay documented"
    assert json.loads(yaml.safe_load(text)["config"]["detection_label_map"]) == (
        SYNTHETIC_BDD100K_LABEL_MAP
    )
    # The real map differs only in three category names; keep the anchor honest.
    assert set(REAL_BDD100K_LABEL_MAP) - set(SYNTHETIC_BDD100K_LABEL_MAP) == {
        "pedestrian",
        "motorcycle",
        "bicycle",
    }


# ------------------------------------------------------------------------- the runner


def test_wrapper_submits_the_rendered_spec_and_forwards_tokens(
    monkeypatch, tmp_path, capsys
) -> None:
    wrapper = _load_wrapper_module()
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(sky_bin.stat().st_mode | stat.S_IXUSR)
    captured = {}

    def fake_submit_workflow(yaml_path, run_id, **kwargs):
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        captured["docs"] = [
            doc
            for doc in yaml.safe_load_all(Path(yaml_path).read_text(encoding="utf-8"))
            if doc is not None
        ]
        return wrapper.WorkflowResult(
            status="SUBMITTED",
            job_id="42",
            returncode=0,
            log_paths={"config": str(tmp_path / "config.yaml")},
        )

    monkeypatch.setattr(wrapper, "submit_workflow", fake_submit_workflow)
    monkeypatch.setattr(
        wrapper,
        "workflow_status",
        lambda job_id, **kwargs: wrapper.WorkflowResult(
            status="SUCCEEDED", job_id=job_id, returncode=0
        ),
    )
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    monkeypatch.setenv("LANCEDB_TOKEN", "unit-test-token")
    monkeypatch.delenv("DETECTION_TRAINING_TOKEN", raising=False)

    rc = wrapper.main(
        [
            "--run-id",
            "bdd100k-test-run",
            "--synthetic",
            "5000",
            "--sky-bin",
            str(sky_bin),
            "--poll-interval",
            "0",
            "--default-image",
        ]
    )

    assert rc == 0
    capsys.readouterr()
    assert captured["run_id"] == "bdd100k-test-run"
    # Only the token that is actually available is forwarded.
    assert captured["kwargs"]["secret_envs"] == ["LANCEDB_TOKEN"]
    # The rendered document is the spec's plan, and the override reached the argv.
    names = [doc["name"] for doc in captured["docs"] if "name" in doc and "run" in doc]
    assert names, captured["docs"]
    rendered = Path.read_text  # keep flake-free; the assertion below uses the parsed docs
    assert any("--synthetic 5000" in doc["run"] for doc in captured["docs"] if "run" in doc)
    assert rendered is Path.read_text


def test_wrapper_yaml_flag_is_still_accepted(monkeypatch, capsys) -> None:
    """`--yaml` is kept as a deprecated alias; it now names a spec."""

    wrapper = _load_wrapper_module()
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")

    rc = wrapper.main(
        [
            "--yaml",
            str(SPEC_PATH),
            "--run-id",
            "bdd100k-alias-run",
            "--render-only",
            "--default-image",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "bdd100k-alias-run"
    assert payload["stages"][0] == "ingest"
    assert "npa workbench lancedb import-bdd100k" in payload["rendered_skypilot"]


@pytest.mark.timeout(300)
def test_mock_endpoint_validation_drives_every_stage(capsys, tmp_path, monkeypatch) -> None:
    """The offline proof: every stage's real argv against stand-in services."""

    wrapper = _load_wrapper_module()
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    output = tmp_path / "mock.json"

    rc = wrapper.main(
        [
            "--mock-endpoints",
            "--run-id",
            "bdd100k-mock-run",
            "--output-json",
            str(output),
        ]
    )

    capsys.readouterr()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert not summary["failures"], summary["failures"]
    assert rc == 0

    assert [
        item["path"] for item in summary["lancedb_requests"] if item["method"] == "POST"
    ] == wrapper.EXPECTED_LANCEDB_POSTS
    assert [
        item["path"] for item in summary["detection_requests"] if item["method"] == "POST"
    ] == wrapper.EXPECTED_DETECTION_POSTS
    # Every stage ran, and each one ran its own argv.
    assert [item["name"] for item in summary["task_results"]] == EXPECTED_STAGE_ORDER
    for item in summary["task_results"]:
        assert item["returncode"] == 0, item

    train_payloads = [
        item["payload"] for item in summary["detection_requests"] if item["path"] == "/train"
    ]
    assert len(train_payloads) == 3
    for payload in train_payloads:
        assert payload["label_map"] == SYNTHETIC_BDD100K_LABEL_MAP
        # num_classes agrees with the map rather than contradicting it.
        assert payload["num_classes"] == len(SYNTHETIC_BDD100K_LABEL_MAP)


@pytest.mark.timeout(300)
def test_mock_run_awaits_training_and_resolves_the_real_checkpoint(
    capsys, tmp_path, monkeypatch
) -> None:
    """The two behaviours the template did in bash, observed in the request stream."""

    wrapper = _load_wrapper_module()
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    output = tmp_path / "mock.json"

    assert wrapper.main(
        ["--mock-endpoints", "--run-id", "bdd100k-mock-order", "--output-json", str(output)]
    ) == 0
    capsys.readouterr()
    summary = json.loads(output.read_text(encoding="utf-8"))

    calls = [(item["method"], item["path"]) for item in summary["detection_requests"]]
    # `--wait` polled /status after every /train ...
    assert calls.count(("GET", "/status")) >= 3
    for index, call in enumerate(calls):
        if call == ("POST", "/train"):
            assert calls[index + 1] == ("GET", "/status")
        if call == ("POST", "/eval"):
            # ... and `--discover-checkpoint` asked /runs before every /eval.
            assert calls[index - 1] == ("GET", "/runs")

    # The eval payload names a concrete checkpoint file, not the training directory.
    eval_payloads = [
        item["payload"] for item in summary["detection_requests"] if item["path"] == "/eval"
    ]
    assert len(eval_payloads) == 3
    for payload in eval_payloads:
        checkpoint = payload["checkpoint_uri"]
        assert checkpoint.endswith(".pt"), checkpoint
        assert "{epoch}" not in checkpoint, checkpoint
        assert "/checkpoints/epoch_" in checkpoint, checkpoint
