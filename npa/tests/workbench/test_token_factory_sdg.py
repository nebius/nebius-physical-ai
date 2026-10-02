"""Exercise SDG routing, quality rejection, provenance, and artifact contracts."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.token_factory import TokenFactoryClient, TokenFactoryConfig
from npa.sdk.workbench.token_factory import SdgRequest, sdg
from npa.workbench.token_factory import TokenFactoryToolError
from npa.workbench.token_factory import sdg as pipeline
from npa.workbench.token_factory.sdg_protocol import FAST_MODEL, REASONING_MODEL


def _request(tmp_path, prompts=("simple", "reasoning"), **kwargs):
    source = tmp_path / "seeds.jsonl"
    source.write_text(
        "".join(
            json.dumps({"id": str(i), "prompt": prompt}) + "\n"
            for i, prompt in enumerate(prompts)
        )
    )
    return SdgRequest(
        input_path=str(source), output_path=str(tmp_path / "output"), **kwargs
    )


@pytest.fixture
def provider():
    calls = []
    state = {}

    def handle(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": model}
                        for model in state.get("models", (FAST_MODEL, REASONING_MODEL))
                    ]
                },
            )
        body = json.loads(request.content)
        calls.append(body)
        stage = body["response_format"]["json_schema"]["name"]
        content = json.loads(body["messages"][-1]["content"])
        answer = {
            "Route": {
                "task_type": "reasoning"
                if "reasoning" in content["seed"]
                else "transformation",
                "reason": "synthetic route",
            },
            "Pair": {
                "instruction": "Explain " + content["seed"],
                "response": "Synthetic answer " + content["seed"],
            },
            "Review": {
                "reason": "synthetic review",
                "follows_seed": True,
                "self_contained": True,
                "correct": True,
                "consistent": True,
            },
        }[stage]
        response = {
            "id": f"synthetic-{len(calls)}",
            "model": body["model"],
            "choices": [
                {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_cache_hit_tokens": 64,
            },
        }
        if state.get("mutate"):
            response = state["mutate"](stage, body, response)
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = TokenFactoryClient(
            TokenFactoryConfig(
                api_key="synthetic-key", base_url="https://provider.invalid/v1"
            ),
            http_client=transport,
        )
        yield client, calls, state


def _read_rows(tmp_path, name):
    return [
        json.loads(line)
        for line in (tmp_path / "output" / name).read_text().splitlines()
    ]


def test_sdk_routes_generates_reviews_and_publishes_hash_bound_artifacts(
    tmp_path, provider
):
    client, calls, _ = provider
    report = sdg(_request(tmp_path), client=client)
    assert report["status"] == "completed"
    assert report["accepted_count"] == 2
    assert report["generation_model_counts"] == {FAST_MODEL: 1, REASONING_MODEL: 1}
    assert len(calls) == 6
    assert all("max_tokens" not in call for call in calls)
    assert all(call["response_format"]["json_schema"]["strict"] for call in calls)
    assert calls[1]["messages"][0] == calls[4]["messages"][0]
    rows = _read_rows(tmp_path, "dataset.jsonl")
    assert all(
        [message["role"] for message in row["messages"]] == ["user", "assistant"]
        for row in rows
    )
    for filename, artifact in report["artifacts"].items():
        assert (
            hashlib.sha256((tmp_path / "output" / filename).read_bytes()).hexdigest()
            == artifact["sha256"]
        )
    assert all(
        item["cached_tokens"] == item["request_count"] * 64
        for item in report["usage_by_stage"]
    )
    assert "synthetic-key" not in (tmp_path / "output" / "provenance.jsonl").read_text()


def test_selected_generator_failure_uses_remaining_eligible_model(tmp_path, provider):
    client, _, state = provider
    state["mutate"] = lambda stage, body, result: (
        {**result, "model": "wrong/model"}
        if stage == "Pair" and body["model"] == REASONING_MODEL
        else result
    )
    report = sdg(_request(tmp_path, ("reasoning",)), client=client)
    record = _read_rows(tmp_path, "provenance.jsonl")[0]
    assert report["accepted_count"] == 1
    assert record["routing"]["selected_model"] == REASONING_MODEL
    assert record["served_model"] == FAST_MODEL
    assert [call["status"] for call in record["calls"]] == [
        "completed",
        "error",
        "completed",
        "completed",
    ]


@pytest.mark.parametrize(
    "check",
    [False, "yes"],
)
def test_review_rejections_and_invalid_judgments_never_enter_training_data(
    tmp_path, provider, check
):
    client, _, state = provider

    def mutate(stage, body, result):
        if stage == "Review":
            result["choices"][0]["message"]["content"] = json.dumps(
                {
                    "reason": "synthetic failed check",
                    "follows_seed": True,
                    "self_contained": True,
                    "correct": True,
                    "consistent": check,
                }
            )
        return result

    state["mutate"] = mutate
    report = sdg(_request(tmp_path, ("simple",)), client=client)
    assert report["status"] == "failed"
    assert report["accepted_count"] == 0
    assert _read_rows(tmp_path, "dataset.jsonl") == []
    assert len(_read_rows(tmp_path, "rejected.jsonl")) == 1


def test_duplicate_pairs_are_retained_as_rejections(tmp_path, provider):
    client, _, _ = provider
    report = sdg(_request(tmp_path, ("simple", "simple")), client=client)
    assert report["accepted_count"] == report["rejected_count"] == 1
    assert (
        _read_rows(tmp_path, "rejected.jsonl")[0]["reason"]
        == "duplicate_instruction_answer"
    )


def test_missing_cache_counters_remain_unknown(tmp_path, provider):
    client, _, state = provider
    state["mutate"] = lambda stage, body, result: {
        key: value for key, value in result.items() if key != "usage"
    }
    report = sdg(_request(tmp_path, ("simple",)), client=client)
    assert all(
        item["cached_tokens"] is None and item["cache_counters_reported"] == 0
        for item in report["usage_by_stage"]
    )


def test_truncated_generation_is_quarantined_without_review(tmp_path, provider):
    client, calls, state = provider

    def mutate(stage, body, result):
        if stage == "Pair":
            result["choices"][0]["finish_reason"] = "length"
        return result

    state["mutate"] = mutate
    report = sdg(_request(tmp_path, ("simple",)), client=client)
    assert report["status"] == "failed"
    assert report["error_count"] == 1
    assert _read_rows(tmp_path, "dataset.jsonl") == []
    assert all(
        call["response_format"]["json_schema"]["name"] != "Review" for call in calls
    )


def test_s3_artifacts_use_real_serialization_and_publish_report_last(
    tmp_path, provider, monkeypatch
):
    from npa.clients.storage import StorageClient

    request = _request(tmp_path)
    writes = {}

    def upload(local_path, uri):
        from pathlib import Path

        writes[uri] = Path(local_path).read_bytes()
        return uri

    storage = SimpleNamespace(
        download_path=lambda uri, target: request.input_path, upload_file=upload
    )
    monkeypatch.setattr(StorageClient, "from_environment", lambda: storage)
    report = sdg(
        SdgRequest(
            input_path="s3://example-bucket/seeds.jsonl",
            output_path="s3://example-bucket/run",
        ),
        client=provider[0],
    )
    assert list(writes)[-1] == "s3://example-bucket/run/report.json"
    assert len(writes) == 4
    for filename, metadata in report["artifacts"].items():
        assert (
            hashlib.sha256(writes["s3://example-bucket/run/" + filename]).hexdigest()
            == metadata["sha256"]
        )


def test_invalid_router_choice_cannot_inject_an_unapproved_model(tmp_path, provider):
    client, calls, state = provider

    def mutate(stage, body, result):
        if stage == "Route":
            result["choices"][0]["message"]["content"] = (
                '{"task_type":"unapproved/model","reason":"injected"}'
            )
        return result

    state["mutate"] = mutate
    report = sdg(_request(tmp_path, ("simple",)), client=client)
    assert report["routing_status_counts"] == {"unavailable": 1}
    assert {call["model"] for call in calls} <= {FAST_MODEL, REASONING_MODEL}


def test_models_must_be_available_before_inference(tmp_path, provider):
    client, calls, state = provider
    state["models"] = [FAST_MODEL]
    with pytest.raises(TokenFactoryToolError, match="Both SDG"):
        sdg(_request(tmp_path), client=client)
    assert calls == []


@pytest.mark.parametrize(
    "body",
    [
        "",
        "not json",
        '{"id":"one","prompt":""}',
        '{"id":"one","prompt":"ok"}\n{"id":"one","prompt":"again"}',
    ],
)
def test_invalid_seed_files_fail_before_client_creation(tmp_path, monkeypatch, body):
    request = _request(tmp_path)
    (tmp_path / "seeds.jsonl").write_text(body)
    monkeypatch.setattr(
        pipeline, "TokenFactoryClient", lambda: pytest.fail("paid client initialized")
    )
    with pytest.raises(TokenFactoryToolError):
        sdg(request)


def test_dry_run_requires_neither_provider_key_nor_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "TokenFactoryClient", lambda: pytest.fail("paid client initialized")
    )
    report = sdg(_request(tmp_path, dry_run=True, router="jev"))
    assert report["status"] == "planned"
    assert report["inference_performed"] is False
    assert not (tmp_path / "output").exists()


def test_jev_is_optional_but_explicit_selection_requires_its_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "load_credentials", lambda: SimpleNamespace(tokens={})
    )
    monkeypatch.setattr(
        pipeline, "TokenFactoryClient", lambda: pytest.fail("paid client initialized")
    )
    with pytest.raises(TokenFactoryToolError, match="TYPESAFE_API_KEY"):
        sdg(_request(tmp_path, router="jev"))


def test_cli_requires_s3_handoffs_and_emits_one_json_failure_document(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "token-factory",
            "sdg",
            "--input-path",
            str(tmp_path / "seeds.jsonl"),
            "--output-path",
            "s3://example-bucket/sdg",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["result"] == "error"


def test_cli_maps_configuration_to_the_shared_pipeline(monkeypatch):
    from npa.cli.workbench import token_factory as commands

    seen = []
    monkeypatch.setattr(
        commands,
        "run_sdg",
        lambda request: seen.append(request) or {"status": "planned"},
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "token-factory",
            "sdg",
            "--input-path",
            "s3://example-bucket/seeds.jsonl",
            "--output-path",
            "s3://example-bucket/sdg",
            "--dry-run",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "planned"}
    assert seen[0].dry_run is True
