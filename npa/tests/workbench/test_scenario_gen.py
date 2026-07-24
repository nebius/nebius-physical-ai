from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.workbench.scenario_gen import app as scenario_gen_app
from npa.workbench.scenario_gen.generation import (
    ScenarioGenError,
    generate_scenarios,
)
from npa.workbench.scenario_gen.ranking import ScenarioRankError, rank_scenarios
from npa.workbench.scenario_gen.schemas import (
    ADVERSARIAL_SET_SCHEMA,
    GenerateRequest,
    RankRequest,
)

runner = CliRunner()


def _fake_backend(request: GenerateRequest, seed: int) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": "adv-0000",
            "seed": seed,
            "perturbation": {"friction": 0.1, "mass_scale": 0.1},
            "failure_score": 0.2,
            "metrics": {"predicted_violation_rate": 0.2},
        },
        {
            "scenario_id": "adv-0001",
            "seed": seed + 1,
            "perturbation": {"friction": 0.9, "mass_scale": 0.9},
            "failure_score": 0.9,
            "metrics": {"predicted_violation_rate": 0.9},
        },
    ]


def _generate_request(tmp_path: Path, **overrides: Any) -> GenerateRequest:
    payload = {
        "policy_uri": "s3://bucket/policy.ckpt",
        "base_config_uri": "s3://bucket/base.json",
        "output_uri": str(tmp_path / "adv"),
        "num_scenarios": 2,
        "workflow_run": "run-xyz",
        "visualize": False,
    }
    payload.update(overrides)
    return GenerateRequest(**payload)


def test_generate_scenarios_ranks_and_threads_lineage(tmp_path: Path) -> None:
    response = generate_scenarios(
        _generate_request(tmp_path),
        run_id="run-1",
        adversary_backend=_fake_backend,
    )

    assert response.scenario_count == 2
    assert response.adversarial_set_schema == ADVERSARIAL_SET_SCHEMA
    assert response.top_severity == 0.9
    assert response.lineage.workflow_run == "run-xyz"
    assert response.lineage.policy_uri == "s3://bucket/policy.ckpt"

    manifest = json.loads(Path(response.manifest_uri).read_text())
    assert manifest["schema"] == ADVERSARIAL_SET_SCHEMA
    # Highest-severity scenario is ranked first.
    assert manifest["scenarios"][0]["scenario_id"] == "adv-0001"
    assert manifest["lineage"]["input_uris"] == [
        "s3://bucket/policy.ckpt",
        "s3://bucket/base.json",
    ]
    # Per-scenario config artifacts are emitted.
    for scenario in manifest["scenarios"]:
        assert Path(scenario["config_uri"]).exists()


def test_generate_scenarios_default_backend_is_deterministic(tmp_path: Path) -> None:
    first = generate_scenarios(_generate_request(tmp_path / "a", num_scenarios=4, seed=7), run_id="r")
    second = generate_scenarios(_generate_request(tmp_path / "b", num_scenarios=4, seed=7), run_id="r")
    assert first.scenario_count == 4
    first_scenarios = json.loads(Path(first.manifest_uri).read_text())["scenarios"]
    second_scenarios = json.loads(Path(second.manifest_uri).read_text())["scenarios"]
    # Same seed => same mined perturbations and severities, independent of output path.
    assert [s["perturbation"] for s in first_scenarios] == [s["perturbation"] for s in second_scenarios]
    assert first.top_severity == second.top_severity


def test_generate_scenarios_empty_backend_raises(tmp_path: Path) -> None:
    with pytest.raises(ScenarioGenError):
        generate_scenarios(
            _generate_request(tmp_path),
            adversary_backend=lambda _request, _seed: [],
        )


def test_generate_emits_rerun_rrd(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    response = generate_scenarios(
        _generate_request(tmp_path, num_scenarios=4, visualize=True),
        run_id="run-viz",
        adversary_backend=_fake_backend,
    )
    assert response.viz_uri.endswith("scenarios.rrd")
    rrd = Path(response.viz_uri)
    assert rrd.exists() and rrd.stat().st_size > 0


def test_render_adversarial_rrd_direct(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    from npa.workbench.scenario_gen.generation import generate_scenarios as gen
    from npa.workbench.scenario_gen.visualization import render_adversarial_rrd

    response = gen(_generate_request(tmp_path, visualize=False), run_id="r", adversary_backend=_fake_backend)
    manifest = json.loads(Path(response.manifest_uri).read_text())
    from npa.workbench.scenario_gen.schemas import ScenarioRecord

    records = [ScenarioRecord.model_validate(s) for s in manifest["scenarios"]]
    uri = render_adversarial_rrd(records, output_uri=str(tmp_path / "viz"), task_name="Isaac-Cartpole-v0", run_id="r")
    assert uri.endswith("scenarios.rrd")
    assert Path(uri).exists() and Path(uri).stat().st_size > 0


def test_generate_skips_viz_when_rerun_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.scenario_gen.visualization as viz

    def _no_rerun() -> Any:
        raise ImportError("rerun not installed")

    monkeypatch.setattr(viz, "_import_rerun", _no_rerun)
    response = generate_scenarios(
        _generate_request(tmp_path, visualize=True),
        run_id="run-noviz",
        adversary_backend=_fake_backend,
    )
    assert response.viz_uri == ""
    # Generation still succeeds and writes the JSON manifest.
    assert Path(response.manifest_uri).exists()


def test_generate_request_validation() -> None:
    with pytest.raises(ValueError):
        GenerateRequest(policy_uri="", base_config_uri="s3://b/x", output_uri="s3://b/o")
    with pytest.raises(ValueError):
        GenerateRequest(policy_uri="s3://b/p", base_config_uri="s3://b/x", output_uri="s3://b/o", num_scenarios=0)


def test_rank_scenarios_orders_by_weighted_score(tmp_path: Path) -> None:
    gen = generate_scenarios(
        _generate_request(tmp_path),
        run_id="run-1",
        adversary_backend=_fake_backend,
    )
    ranked = rank_scenarios(
        RankRequest(input_uri=gen.manifest_uri, output_uri=str(tmp_path / "ranked"), top_k=1),
        run_id="rank-1",
    )
    assert ranked.ranked_count == 1
    assert ranked.top_scenarios[0]["scenario_id"] == "adv-0001"
    payload = json.loads(Path(ranked.ranked_manifest_uri).read_text())
    assert payload["source_uri"] == gen.manifest_uri
    assert payload["lineage"]["produced_by"] == "workbench.scenario_gen.rank"


def test_rank_scenarios_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(ScenarioRankError):
        rank_scenarios(
            RankRequest(input_uri=str(tmp_path / "missing.json"), output_uri=str(tmp_path / "out"))
        )


def test_rank_scenarios_empty_set_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": ADVERSARIAL_SET_SCHEMA, "scenarios": []}))
    with pytest.raises(ScenarioRankError):
        rank_scenarios(RankRequest(input_uri=str(manifest), output_uri=str(tmp_path / "out")))


def test_generate_endpoint_success_and_status_list(tmp_path: Path) -> None:
    from npa.workbench.scenario_gen.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    response = client.post(
        "/generate",
        json={
            "policy_uri": "s3://bucket/policy.ckpt",
            "base_config_uri": "s3://bucket/base.json",
            "output_uri": str(tmp_path / "adv"),
            "num_scenarios": 3,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    run_id = body["run_id"]
    assert body["scenario_count"] == 3

    status = client.get("/status", params={"run_id": run_id})
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    listing = client.get("/list")
    assert listing.status_code == 200
    assert any(run["run_id"] == run_id for run in listing.json()["runs"])


def test_generate_endpoint_failure_returns_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.scenario_gen.service as service_module
    from npa.workbench.scenario_gen.service import create_app

    def boom(_body: Any) -> Any:
        raise ScenarioGenError("adversary backend produced no scenarios")

    monkeypatch.setattr(service_module, "generate_scenarios", boom)
    client = TestClient(create_app(auth_mode="none"))
    response = client.post(
        "/generate",
        json={
            "policy_uri": "s3://bucket/policy.ckpt",
            "base_config_uri": "s3://bucket/base.json",
            "output_uri": str(tmp_path / "adv"),
        },
    )
    assert response.status_code == 400
    assert "no scenarios" in response.json()["detail"]


def test_rank_endpoint_success_and_failure(tmp_path: Path) -> None:
    from npa.workbench.scenario_gen.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    gen = generate_scenarios(_generate_request(tmp_path), run_id="run-1", adversary_backend=_fake_backend)

    ok = client.post(
        "/rank",
        json={"input_uri": gen.manifest_uri, "output_uri": str(tmp_path / "ranked"), "top_k": 2},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["ranked_count"] == 2

    missing = client.post(
        "/rank",
        json={"input_uri": str(tmp_path / "nope.json"), "output_uri": str(tmp_path / "ranked")},
    )
    assert missing.status_code == 400


def test_health_and_system_info_endpoints() -> None:
    from npa.workbench.scenario_gen.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/system-info").json()
    assert info["tool"] == "scenario_gen"


def test_token_auth_rejects_missing_and_invalid_tokens() -> None:
    from npa.workbench.scenario_gen.service import create_app

    client = TestClient(create_app(auth_mode="token", token="s3cr3t"))
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/health", headers={"Authorization": "Bearer s3cr3t"})
    assert ok.status_code == 200


def test_cli_generate_and_rank_local(tmp_path: Path) -> None:
    gen_out = tmp_path / "adv"
    result = runner.invoke(
        scenario_gen_app,
        [
            "generate",
            "--policy-uri",
            "s3://bucket/policy.ckpt",
            "--input-path",
            "s3://bucket/base.json",
            "--output-path",
            str(gen_out),
            "--num-scenarios",
            "4",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest_uri = json.loads(result.output)["manifest_uri"]

    ranked = runner.invoke(
        scenario_gen_app,
        [
            "rank",
            "--input-path",
            manifest_uri,
            "--output-path",
            str(tmp_path / "ranked"),
            "--top-k",
            "2",
            "--output",
            "json",
        ],
    )
    assert ranked.exit_code == 0, ranked.output
    assert json.loads(ranked.output)["ranked_count"] == 2


def test_cli_and_sdk_do_not_import_heavy_ml_dependencies_at_module_level() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    cli_source = (npa_root / "src/npa/cli/workbench/scenario_gen.py").read_text()
    sdk_source = (npa_root / "src/npa/sdk/workbench/scenario_gen.py").read_text()
    for source in (cli_source, sdk_source):
        assert "import torch" not in source
        assert "import genesis" not in source
        assert "import isaac" not in source


def test_sdk_workbench_namespace_exports_scenario_gen() -> None:
    from npa.sdk import workbench

    assert workbench.scenario_gen.__name__ == "npa.sdk.workbench.scenario_gen"
    assert hasattr(workbench.scenario_gen, "generate")
    assert hasattr(workbench.scenario_gen, "rank")
