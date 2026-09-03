from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.workflows.paidf_upstream import (
    PAIDF_ORCHESTRATION_REVISION,
    PHYSICAL_AI_DATA_FACTORY_REVISION,
    SCHEMA,
    upstream_contract,
    write_upstream_contract,
)


def test_upstream_contract_distinguishes_sources_and_orchestrators() -> None:
    payload = upstream_contract("cosmos-transfer2.5")

    assert payload["schema"] == SCHEMA
    sources = {item["repository"]: item for item in payload["sources"]}
    ecosystem = sources["https://github.com/NVIDIA/physical-ai-data-factory"]
    scaler = sources["https://github.com/NVIDIA/paidf-orchestration"]
    assert ecosystem["revision"] == PHYSICAL_AI_DATA_FACTORY_REVISION
    assert ecosystem["licenses"] == ["CC-BY-4.0", "Apache-2.0"]
    assert ecosystem["upstream_orchestrator"] == "NVIDIA OSMO"
    assert scaler["revision"] == PAIDF_ORCHESTRATION_REVISION
    assert scaler["licenses"] == ["Apache-2.0"]
    assert scaler["role"] == "airflow-kubernetes-scaled-iaa-and-evg-reference"
    assert all(item["executed_by_npa"] is False for item in sources.values())
    assert payload["npa_integration"]["orchestrator"] == "SkyPilot"
    assert payload["npa_integration"]["components"]["execution"] == (
        "workbench.cosmos2.transfer_execute"
    )


def test_cosmos3_contract_names_real_framework_command() -> None:
    payload = upstream_contract("cosmos3-video2video")
    assert payload["npa_integration"]["components"]["execution"] == (
        "workbench.cosmos3.generate_video_variants"
    )


def test_write_upstream_contract_local(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "reports" / "upstream.json"
    result = write_upstream_contract("cosmos-transfer2.5", str(target))

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written == {key: value for key, value in result.items() if key != "written_uri"}
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported PAIDF workflow variant"):
        upstream_contract("airflow")


@pytest.mark.parametrize(
    ("filename", "variant", "successor"),
    [
        ("physical-ai-data-factory.yaml", "cosmos-transfer2.5", "generate-configs"),
        ("paidf-cosmos3.yaml", "cosmos3-video2video", "prepare-input"),
    ],
)
def test_shipped_workflows_record_upstream_before_processing(
    filename: str, variant: str, successor: str
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (
            repo_root
            / "npa"
            / "workflows"
            / "workbench"
            / "npa-workflows"
            / filename
        ).read_text(encoding="utf-8")
    )

    assert workflow["initial"] == "record-upstream"
    state = workflow["states"]["record-upstream"]
    assert state["run"]["argv"][-2:] == [
        variant,
        "{{config.upstream_contract_uri}}",
    ]
    assert state["outputs"] == [
        {
            "uri": "{{config.upstream_contract_uri}}",
            "schema": SCHEMA,
        }
    ]
    assert state["next"] == successor
    assert workflow["states"][successor]["needs"] == ["record-upstream"]
