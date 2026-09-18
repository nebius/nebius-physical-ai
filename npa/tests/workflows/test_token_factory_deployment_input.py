"""Tests for the durable prompt producer used by the Agent UI template."""

from __future__ import annotations

from pathlib import Path

from npa.workbench.token_factory import _load_prompts
from npa.workflows.token_factory_deployment_input import (
    PROMPT_SCHEMA,
    deployment_review_prompt_records,
    write_deployment_review_prompts,
)


def test_deployment_review_prompt_records_are_nonempty_and_stable() -> None:
    records = deployment_review_prompt_records()

    assert [record["id"] for record in records] == [
        "delivery-plan",
        "readiness-risks",
        "validation-checklist",
    ]
    assert all(record["prompt"].strip() for record in records)


def test_written_prompt_jsonl_is_accepted_by_the_generate_reader(tmp_path: Path) -> None:
    output = tmp_path / "prompts.jsonl"

    result = write_deployment_review_prompts(str(output))

    assert result == {
        "schema": PROMPT_SCHEMA,
        "prompt_count": 3,
        "output_uri": str(output),
    }
    assert _load_prompts(output) == [
        (record["id"], record["prompt"])
        for record in deployment_review_prompt_records()
    ]


def test_prompt_writer_uploads_the_same_jsonl_to_a_durable_uri(tmp_path: Path) -> None:
    class FakeStorage:
        def __init__(self) -> None:
            self.payload = ""
            self.target = ""

        def upload_file(self, source: str, target: str) -> str:
            self.payload = Path(source).read_text(encoding="utf-8")
            self.target = target
            return target

    storage = FakeStorage()
    target = "s3://workflow-artifacts/prompts.jsonl"

    result = write_deployment_review_prompts(target, storage_client=storage)

    assert result["output_uri"] == target
    assert storage.target == target
    materialized = tmp_path / "prompts.jsonl"
    materialized.write_text(storage.payload, encoding="utf-8")
    assert len(_load_prompts(materialized)) == 3
