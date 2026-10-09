"""Verify full Cartesian fanout, fixed prompts, private previews and replay safety."""

import copy
import itertools
import json
from types import SimpleNamespace

import pytest

from npa.workflows.video_sweep import (
    artifacts,
    execution,
    matrix,
    matrix_view,
    operator,
    planning,
)


@pytest.fixture
def sweep():
    return {
        "base": {
            "hint": "private warehouse hint",
            "edge_threshold": "medium",
            "num_steps": 35,
        },
        "axes": {
            "seed": [23, 41],
            "control_guidance": [1.0, 1.5],
            "guidance": [3.0, 5.0, 7.0],
        },
    }


def test_cartesian_product_and_worker_coverage(sweep):
    rows = matrix.expand(sweep)
    expected = set(itertools.product([23, 41], [1.0, 1.5], [3.0, 5.0, 7.0]))
    assert {(r["seed"], r["control_guidance"], r["guidance"]) for r in rows} == expected
    assert len(rows) == 12
    summary = matrix.describe(sweep, 2, 3)
    assert len(summary["jobs"]) == 24
    assert [sum(j["worker"] == w for j in summary["jobs"]) for w in range(3)] == [
        8,
        8,
        8,
    ]
    for source in (1, 2):
        assert {
            j["combination"] for j in summary["jobs"] if j["source"] == source
        } == set(range(1, 13))
    reordered = {**sweep, "axes": dict(reversed(list(sweep["axes"].items())))}
    assert matrix.expand(reordered) == rows


@pytest.mark.parametrize(
    "change",
    [
        {"axes": {}},
        {"axes": {"seed": []}},
        {"axes": {"seed": 23}},
        {"axes": {"seed": [23, 23]}},
        {"axes": {"control_guidance": [1, 1.0]}},
        {"axes": {"seed": [True]}},
        {"axes": {"seed": [-1]}},
        {"axes": {"guidance": [float("nan")]}},
        {"axes": {"unknown": [1]}},
        {"axes": {"hint": ["overlap"]}},
        {"extra": 1},
    ],
)
def test_invalid_grid_fails_before_generation(sweep, change):
    sweep.update(change)
    with pytest.raises(ValueError):
        matrix.expand(sweep)


@pytest.fixture
def planned(sweep, monkeypatch):
    descriptions, merges = [], []

    def describe(uri, *_):
        descriptions.append(uri)
        return {"uri": uri, "sha256": artifacts.digest(uri), "description": "scene"}

    def merge(source, variant, *_):
        merges.append((source["uri"], variant["hint"]))
        return f"Stable prompt {len(merges)}", {"model": "test"}

    monkeypatch.setattr(planning, "_describe_source", describe)
    monkeypatch.setattr(planning, "_merge", merge)
    sources = {
        "schema": "npa.video_sweep.sources.v1",
        "clips": ["source-a", "source-b"],
    }
    manifest = {
        "schema": "npa.video_sweep.variants.v3",
        "generator": "cosmos3-nano",
        "sweep": sweep,
    }
    planning._validate_inputs(sources, manifest)
    args = SimpleNamespace(
        root_uri="test",
        run_id="test",
        workers=2,
        samples=4,
        reasoner_model="test",
        merge_model="test",
    )
    items = planning._expand_items(args, sources, manifest, None)
    plan = {
        "schema": "npa.video_sweep.plan.v1",
        "run_id": "test",
        "workers": 2,
        "samples": 4,
        "generator": "cosmos3-nano",
        "sweep": sweep,
        "items": items,
    }
    return args, plan, descriptions, merges


def test_parameter_sweep_holds_source_and_prompt_fixed(planned):
    _, plan, descriptions, merges = planned
    assert len(descriptions) == len(merges) == 2
    assert len(plan["items"]) == len({r["id"] for r in plan["items"]}) == 24
    assert {r["prompt"] for r in plan["items"][:12]} == {"Stable prompt 1"}
    assert {r["prompt"] for r in plan["items"][12:]} == {"Stable prompt 2"}
    matrix.validate_plan(plan)


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "changed", "source", "identity"]
)
def test_matrix_plan_tampering_fails_closed(planned, corruption):
    _, plan, *_ = planned
    if corruption == "missing":
        plan["items"].pop()
    elif corruption == "duplicate":
        plan["items"][1] = copy.deepcopy(plan["items"][0])
    elif corruption == "changed":
        plan["sweep"]["axes"]["guidance"][0] = 9.0
    elif corruption == "source":
        plan["items"][1]["source"] = {"sha256": "foreign"}
    else:
        plan["items"][0]["id"] = "foreign"
    with pytest.raises(ValueError):
        matrix.validate_plan(plan)


def test_workers_execute_and_join_every_matrix_cell(planned, monkeypatch):
    args, plan, *_ = planned
    stored = {"test/plan.json": plan}
    monkeypatch.setattr(execution, "read_json", stored.__getitem__)
    monkeypatch.setattr(execution, "write_json", stored.__setitem__)
    observed = []

    def generate(item, *_):
        observed.append(item["id"])
        return {"id": item["id"], "engine": "cosmos3-nano"}

    monkeypatch.setattr(execution, "generate_candidate", generate)
    for worker in range(2):
        args.worker = worker
        execution.generate(args)
    assert len(observed) == len(set(observed)) == 24
    assert {r["id"] for r in execution._join(args, plan)} == set(observed)
    stored["test/workers/0.json"]["items"].pop()
    with pytest.raises(ValueError, match="Missing"):
        execution._join(args, plan)


def test_preview_is_offline_and_private(tmp_path, monkeypatch, capsys):
    config, output = tmp_path / "config.json", tmp_path / "preview"
    operator._initialize(config)
    monkeypatch.setattr(
        operator, "_credentials", lambda *_: pytest.fail("Unexpected credentials")
    )
    monkeypatch.setattr(
        operator, "_invoke", lambda *_: pytest.fail("Unexpected provider call")
    )
    assert (
        operator.main(["plan", "--config", str(config), "--output-dir", str(output)])
        == 0
    )
    assert "8 candidates" in capsys.readouterr().out
    summary = json.loads((output / "matrix.json").read_text())
    assert len(summary["jobs"]) == 8
    page = (output / "index.html").read_text()
    assert "Plan only" in page and "control_guidance" in page and "guidance" in page
    assert "<bucket>" not in page and "preserve the vehicle" not in page
    assert (output / "index.html").stat().st_mode & 0o777 == 0o600


def test_text_axes_stay_private_and_recorded_cells_select_their_clip(sweep):
    sweep["base"].pop("hint")
    sweep["axes"]["prompt"] = ["private prompt </script>", "another private prompt"]
    summary = matrix.describe(sweep, 1, 2)
    assert summary["axes"]["prompt"] == ["Prompt 1", "Prompt 2"]
    candidates = [{"accepted": False, "score": 0.15} for _ in summary["jobs"]]
    page = matrix_view.render(summary, candidates)
    assert "private prompt" not in page
    assert page.count("data-matrix-candidate=") == 24
    assert 'data-matrix-candidate="23"' in page
    with pytest.raises(ValueError):
        matrix_view.render(summary, candidates[:-1])


def test_resume_refuses_changed_axes_before_writing_any_inventory(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.json"
    operator._initialize(path)
    config = json.loads(path.read_text())
    stored = {}
    monkeypatch.setattr(artifacts, "exists", stored.__contains__)
    monkeypatch.setattr(artifacts, "read_json", stored.__getitem__)
    monkeypatch.setattr(artifacts, "write_json", stored.__setitem__)
    operator._stage_inputs(config)
    before = copy.deepcopy(stored)
    changed = copy.deepcopy(config)
    changed["sweep"]["axes"]["seed"].append(100)
    with pytest.raises(ValueError, match="new run ID"):
        operator._stage_inputs(changed)
    assert stored == before
    operator._stage_inputs(config)


@pytest.mark.parametrize("native", [True, False])
def test_existing_explicit_operator_configs_remain_valid(tmp_path, native):
    path = tmp_path / "config.json"
    operator._initialize(path)
    config = json.loads(path.read_text())
    config.pop("sweep")
    if native:
        config["variants"] = [
            {
                "prompt": "Preserve the warehouse",
                "seed": 23,
                "edge_threshold": "medium",
                "control_guidance": 1.0,
                "guidance": 3.0,
                "num_steps": 35,
            }
        ]
    else:
        config.pop("generator")
        config["variants"] = [
            {
                "hint": "Preserve the warehouse",
                "seed": 23,
                "control": "edge",
                "control_weight": 0.5,
                "guidance": 3.0,
            }
        ]
    path.write_text(json.dumps(config))
    assert operator._load(path, offline=True) == config


@pytest.mark.parametrize("both", [True, False])
def test_operator_requires_one_variant_definition(tmp_path, both):
    path = tmp_path / "config.json"
    operator._initialize(path)
    config = json.loads(path.read_text())
    if both:
        config["variants"] = matrix.expand(config["sweep"])
    else:
        config.pop("sweep")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        operator._load(path, offline=True)
