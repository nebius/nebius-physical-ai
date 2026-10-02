"""Verify automatic SDG routing and exported training artifacts with real models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.sdk.workbench.token_factory import SdgRequest, sdg
from npa.clients.token_factory import TokenFactoryClient
from npa.workbench.token_factory.sdg_protocol import FAST_MODEL, REASONING_MODEL
from npa.workbench.token_factory.sdg_protocol import _review

pytestmark = pytest.mark.token_factory_e2e


def test_live_reviewer_rejects_contradictory_answer(tmp_path):
    if os.environ.get("NPA_TOKEN_FACTORY_SDG_LIVE") != "1":
        pytest.skip("Set NPA_TOKEN_FACTORY_SDG_LIVE=1 for real hosted SDG inference")
    pair = {
        "instruction": "A box contains two red balls and three blue balls. How many draws without replacement guarantee a red ball? Explain why.",
        "response": "Three draws guarantee a red ball. In the worst case all three blue balls come first. Three draws are therefore insufficient; the fourth draw must be red.",
    }
    review, trace = _review(
        TokenFactoryClient(), "Explain the worst-case guarantee.", "", pair
    )
    (tmp_path / "negative-review.json").write_text(
        json.dumps({"candidate": pair, "review": review, "call": trace}, indent=2)
        + "\n"
    )
    assert trace["status"] == "completed"
    assert review["accept"] is False
    assert review["consistent"] is False


def test_live_open_weight_sdg_routing_and_dataset(tmp_path):
    if os.environ.get("NPA_TOKEN_FACTORY_SDG_LIVE") != "1":
        pytest.skip("Set NPA_TOKEN_FACTORY_SDG_LIVE=1 for real hosted SDG inference")
    example = Path(__file__).resolve().parents[2] / "examples/token-factory-sdg"
    output = tmp_path / "sdg"
    report = sdg(
        SdgRequest(
            input_path=str(example / "seeds.jsonl"),
            context_path=str(example / "reference.md"),
            output_path=str(output),
        )
    )
    _assert_dataset(output, report)


def _assert_dataset(output, report):
    records = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    rows = [
        json.loads(line) for line in (output / "dataset.jsonl").read_text().splitlines()
    ]
    assert report["status"] == "completed"
    assert len(records) == report["seed_count"] == 6
    assert report["error_count"] == 0
    assert report["accepted_count"] == len(rows) > 0
    assert report["accepted_count"] + report["rejected_count"] == 6
    accepted_ids = {
        record["id"] for record in records if record["status"] == "accepted"
    }
    assert {row["id"] for row in rows} == accepted_ids
    assert report["routing_status_counts"] == {"accepted": 6}
    assert report["generation_model_counts"] == {FAST_MODEL: 3, REASONING_MODEL: 3}
    for record in records:
        expected = (
            FAST_MODEL if record["id"].startswith("paraphrase") else REASONING_MODEL
        )
        assert record["routing"]["selected_model"] == record["served_model"] == expected
        assert record["review"]["accept"] == (record["status"] == "accepted")
        assert all(call["status"] == "completed" for call in record["calls"])
        assert all(call["usage"]["total_tokens"] > 0 for call in record["calls"])
    assert {
        record["served_model"] for record in records if record["status"] == "accepted"
    } == {FAST_MODEL, REASONING_MODEL}
    for filename, metadata in report["artifacts"].items():
        assert (
            hashlib.sha256((output / filename).read_bytes()).hexdigest()
            == metadata["sha256"]
        )
