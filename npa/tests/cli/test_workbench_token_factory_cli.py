from __future__ import annotations

import json
from pathlib import Path

import httpx
from PIL import Image
from typer.testing import CliRunner

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
    assert {"caption", "generate"} <= names


def test_token_factory_verify_without_key_fails(monkeypatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    result = runner.invoke(app, ["workbench", "token-factory", "verify"])
    assert result.exit_code == 1
    assert "NEBIUS_API_KEY is not set" in result.output


def test_token_factory_verify_with_key_reports_authenticated(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

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
    _install_fake_client(monkeypatch, "1. approach the box\n2. grasp it")
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
    assert "grasp" in json.loads(written.read_text(encoding="utf-8"))["analysis"]


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
