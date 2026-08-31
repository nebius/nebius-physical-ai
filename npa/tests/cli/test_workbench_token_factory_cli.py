from __future__ import annotations

import json
from pathlib import Path

import httpx
from PIL import Image
from typer.testing import CliRunner

import npa.cli.workbench.token_factory as cli_token_factory
from npa.cli.main import app
from npa.clients.token_factory import TokenFactoryClient, resolve_config
import npa.workbench.token_factory as tool

runner = CliRunner()


def _install_fake_client(monkeypatch, reply: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    config = resolve_config(api_key="test-key", environ={})
    client = TokenFactoryClient(config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(tool, "_default_client", lambda: client)


def test_token_factory_help() -> None:
    result = runner.invoke(app, ["workbench", "token-factory", "--help"])
    assert result.exit_code == 0
    assert "Token Factory" in result.output


def test_token_factory_status_reports_base_url() -> None:
    result = runner.invoke(app, ["workbench", "token-factory", "status", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["provider"] == "nebius-token-factory"
    assert payload["base_url"].startswith("https://api.tokenfactory.nebius.com")


def test_token_factory_list_capabilities() -> None:
    result = runner.invoke(app, ["workbench", "token-factory", "list", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    names = {cap["name"] for cap in payload["capabilities"]}
    assert {"caption", "generate", "batch-generate", "batch-status"} <= names


def test_token_factory_verify_without_key_fails(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", tmp_path / "missing.yaml")
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    result = runner.invoke(app, ["workbench", "token-factory", "verify"])
    assert result.exit_code == 1
    assert "NEBIUS_TOKEN_FACTORY_KEY is not set" in result.output


def test_token_factory_verify_with_key_reports_authenticated(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-key")

    class _FakeClient:
        def list_models(self):
            return ["model-a", "model-b"]

    monkeypatch.setattr(tool, "_default_client", lambda: _FakeClient())

    result = runner.invoke(app, ["workbench", "token-factory", "verify", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["authenticated"] is True
    assert payload["model_count"] == 2
    assert payload["base_url"].startswith("https://api.tokenfactory.nebius.com")


def test_token_factory_caption_writes_local_json(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, "a caption")
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(images / "frame.png")
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "workbench",
            "token-factory",
            "caption",
            "--input-path",
            str(images),
            "--output-path",
            str(output),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["image_count"] == 1
    written = output / "captions.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["captions"][0]["caption"] == "a caption"


def test_token_factory_reason_writes_scene_json(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, "<think>inspect the scene</think>1. approach the box\n2. grasp it")
    scene = tmp_path / "scene"
    scene.mkdir()
    Image.new("RGB", (16, 16), (200, 40, 40)).save(scene / "frame.png")
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "workbench",
            "token-factory",
            "reason",
            "--input-path",
            str(scene),
            "--output-path",
            str(output),
            "--task",
            "How does the robot pick up the red box?",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model"] == "nvidia/Cosmos3-Super-Reasoner"
    assert payload["image_count"] == 1
    written = output / "scene_reasoning.json"
    assert written.exists()
    written_payload = json.loads(written.read_text(encoding="utf-8"))
    assert "grasp" in written_payload["analysis"]
    assert "<think>" not in written_payload["analysis"]
    assert "reasoning" not in written_payload


def test_token_factory_generate_writes_jsonl(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, "generated")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "p1", "prompt": "hi"}), encoding="utf-8")
    output = tmp_path / "gen"

    result = runner.invoke(
        app,
        [
            "workbench",
            "token-factory",
            "generate",
            "--input-path",
            str(prompts),
            "--output-path",
            str(output),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    written = output / "generations.jsonl"
    assert written.exists()
    row = json.loads(written.read_text(encoding="utf-8").splitlines()[0])
    assert row == {"id": "p1", "prompt": "hi", "completion": "generated"}


def _batch_result(status: str, **overrides):
    """A BatchResult as the tool layer returns it.

    The CLI layer's own responsibility is which artifact a result warrants, so
    these tests drive it from a result object rather than a fake HTTP API; the
    request/response contract is covered in tests/workbench.
    """

    fields = {
        "status": status,
        "input_path": "prompts.jsonl",
        "output_path": "out",
        "result_uri": "out/generations.jsonl",
        "model": "openai/gpt-oss-120b",
        "operation_id": "batch__test-0001",
        "operation_status": "queued" if status == "pending" else "succeeded",
        "completion_window": "24h",
        "prompt_count": 1,
        "generation_count": 0,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    fields.update(overrides)
    return tool.BatchResult(**fields)


def test_token_factory_batch_generate_no_wait_writes_operation_handle(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "batch"
    monkeypatch.setattr(
        cli_token_factory,
        "batch_generate",
        lambda **kwargs: _batch_result(
            "pending",
            output_path=str(output),
            result_uri=str(output / "generations.jsonl"),
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "token-factory",
            "batch-generate",
            "--input-path",
            str(tmp_path / "prompts.jsonl"),
            "--output-path",
            str(output),
            "--no-wait",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pending"
    handle = json.loads((output / "batch_operation.json").read_text(encoding="utf-8"))
    assert handle["operation_id"] == "batch__test-0001"
    assert handle["operation_status"] == "queued"
    # A pending batch must not leave an empty generations.jsonl behind for a
    # downstream stage to read as "no results".
    assert not (output / "generations.jsonl").exists()


def test_token_factory_batch_generate_completed_writes_generations(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "batch"
    monkeypatch.setattr(
        cli_token_factory,
        "batch_generate",
        lambda **kwargs: _batch_result(
            "completed",
            output_path=str(output),
            result_uri=str(output / "generations.jsonl"),
            generation_count=1,
            generations=[tool.GenerationItem(id="p1", prompt="hi", completion="batched answer")],
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "token-factory",
            "batch-generate",
            "--input-path",
            str(tmp_path / "prompts.jsonl"),
            "--output-path",
            str(output),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "completed"
    row = json.loads((output / "generations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row == {"id": "p1", "prompt": "hi", "completion": "batched answer"}
    assert not (output / "batch_operation.json").exists()


def test_token_factory_batch_status_reports_failure_as_exit_one(monkeypatch, tmp_path: Path) -> None:
    def _raise(**kwargs):
        raise tool.TokenFactoryToolError("model 'x' is not enabled for batch inference")

    monkeypatch.setattr(cli_token_factory, "batch_collect", _raise)

    result = runner.invoke(
        app,
        [
            "workbench",
            "token-factory",
            "batch-status",
            "--operation-id",
            "batch__test-0001",
            "--output-path",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "not enabled for batch inference" in result.output
