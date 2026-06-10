"""Tests for the BDD100K pipeline generator and its committed artifact."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
GEN_PATH = ROOT / "npa" / "scripts" / "generate_bdd100k_pipeline.py"
ARTIFACT = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "bdd100k-pipeline.generated.yaml"

_spec = importlib.util.spec_from_file_location("generate_bdd100k_pipeline", GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


def test_committed_artifact_is_in_sync_with_generator() -> None:
    # Regenerate with: npa/.venv/bin/python npa/scripts/generate_bdd100k_pipeline.py \
    #   --runner cli --out npa/workflows/workbench/skypilot/bdd100k-pipeline.generated.yaml
    assert ARTIFACT.read_text(encoding="utf-8") == gen.render("cli")


def test_pipeline_structure_is_thin_and_vanilla() -> None:
    docs = [d for d in yaml.safe_load_all(ARTIFACT.read_text()) if d]
    header, tasks = docs[0], docs[1:]
    assert header == {"name": "bdd100k-pipeline", "execution": "serial"}
    names = [t["name"] for t in tasks]
    assert names == [
        "bdd100k-ingest", "bdd100k-backfill-cpu", "bdd100k-backfill-clip", "bdd100k-create-mvs",
        "bdd100k-train-rider", "bdd100k-train-nighttime", "bdd100k-train-distant",
        "bdd100k-eval-rider", "bdd100k-eval-nighttime", "bdd100k-eval-distant",
    ]
    for task in tasks:
        # Vanilla, stock base image; no custom workflow image.
        assert task["resources"]["image_id"] == "docker:python:3.11-slim"
        # No GPU on task pods: GPU work lives in the in-cluster services.
        assert "accelerators" not in task["resources"]
        # Thin run: calls the npa CLI, not inline curl/jq.
        assert "npa workbench" in task["run"]
        assert "curl" not in task["run"]


def test_train_and_eval_use_new_cli_flags() -> None:
    docs = [d for d in yaml.safe_load_all(ARTIFACT.read_text()) if d]
    by_name = {t["name"]: t for t in docs if "name" in t and "run" in t}
    for view in ("rider", "nighttime", "distant"):
        train = by_name[f"bdd100k-train-{view}"]["run"]
        assert "--label-map" in train and "--wait" in train
        eval_run = by_name[f"bdd100k-eval-{view}"]["run"]
        assert "--from-view-latest" in eval_run


def test_both_runners_produce_same_dag() -> None:
    cli = [d.get("name") for d in yaml.safe_load_all(gen.render("cli")) if d]
    curl = [d.get("name") for d in yaml.safe_load_all(gen.render("curl")) if d]
    assert cli == curl
