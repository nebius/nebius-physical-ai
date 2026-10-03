"""Contracts for the reconstructed OpenVLA-OFT qualification workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.workflows import lorafleet_oft_adapters as qualification


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "lorafleet-oft-adapters.yaml"


def _workflow() -> dict[str, object]:
    """Load the checked-in workflow fixture."""
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_workflow_has_five_connected_substantive_byof_stages() -> None:
    """Keep the manager-required five-stage real artifact chain intact."""
    workflow = _workflow()
    states = workflow["states"]
    assert isinstance(states, dict)
    chain = [
        "verify-inputs",
        "reconstruct-weights",
        "published-baseline-rollouts",
        "reconstructed-adapter-rollouts",
        "compare-and-visualize",
    ]
    assert workflow["initial"] == chain[0]
    assert list(states) == chain
    for current, following in zip(chain, chain[1:]):
        state = states[current]
        assert state["toolRef"] == "workbench.byof.repo"
        assert state["next"] == following
        assert state["outputs"]
    assert states[chain[-1]]["terminal"] is True


def test_workflow_consumes_exact_prior_stage_artifact_uris() -> None:
    """Ensure each later calculation receives a declared prior result."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '"{{config.verification_uri}}"' in text
    assert '"{{config.reconstruction_uri}}"' in text
    assert '"{{config.baseline_uri}}"' in text
    assert '"{{config.reconstructed_uri}}"' in text
    assert "--verification-uri" in text
    assert "--reconstruction-uri" in text
    assert "--baseline-uri" in text
    assert "--reconstructed-uri" in text


def test_reconstruction_metadata_rejects_non_exact_or_wrong_rank() -> None:
    """Reject arithmetic that could silently merge factors over target weights."""
    suite = qualification.SUITES["spatial"]
    valid = {
        "source": {"repo_id": suite.source_repo, "revision": suite.source_revision},
        "rank": 64,
        "ab_parameters": 221656576,
        "non_lora_exact": True,
    }
    qualification._validate_reconstruction_metadata(valid, suite)
    invalid = {**valid, "non_lora_exact": False}
    with pytest.raises(ValueError, match="exact non-LoRA"):
        qualification._validate_reconstruction_metadata(invalid, suite)
    wrong_rank = {**valid, "rank": 32}
    with pytest.raises(ValueError, match="rank-64"):
        qualification._validate_reconstruction_metadata(wrong_rank, suite)


def test_s3_parser_requires_complete_s3_object_uri() -> None:
    """Avoid accidentally treating a local path or a bucket root as stage input."""
    assert qualification._s3_parts("s3://bucket/prefix/result.json") == (
        "bucket",
        "prefix/result.json",
    )
    with pytest.raises(ValueError, match="concrete s3"):
        qualification._s3_parts("s3://bucket")
    with pytest.raises(ValueError, match="concrete s3"):
        qualification._s3_parts("/tmp/result.json")


def test_publisher_base_hash_extraction_is_complete() -> None:
    """Keep recursive publisher verification independent from JSON nesting shape."""
    payload = json.loads(
        '{"shards":[{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}'
    )
    assert qualification._base_expected_hashes(payload) == {"a" * 64}


def test_rrd_is_written_and_verified_with_the_runtime_python(tmp_path: Path) -> None:
    """Keep the final RRD check independent of a shell PATH entry."""
    episodes = [{"success": True}, {"success": False}]
    baseline = {"episodes": episodes}
    reconstructed = {"episodes": list(reversed(episodes))}
    output = tmp_path / "comparison.json"
    assert qualification._rrd(output, baseline, reconstructed) == "comparison.rrd"
    assert (tmp_path / "comparison.rrd").is_file()
