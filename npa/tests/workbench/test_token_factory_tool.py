from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from npa.clients.token_factory import (
    DEFAULT_REASONER_MODEL,
    TokenFactoryClient,
    resolve_config,
)
from npa.workbench.token_factory import (
    TokenFactoryToolError,
    caption_images,
    generate_text,
    reason_scene,
)


def _client(reply: str) -> TokenFactoryClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _capturing_client(reply: str, captured: dict) -> TokenFactoryClient:
    import json as _json

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_caption_images_writes_manifest(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_image(images / "a.png", (10, 20, 30))
    _write_image(images / "b.jpg", (200, 100, 50))
    output = tmp_path / "out"

    result = caption_images(
        input_path=str(images),
        output_path=str(output),
        client=_client("a clear caption"),
    )

    assert result.status == "completed"
    assert result.image_count == 2
    assert {item.image for item in result.captions} == {"a.png", "b.jpg"}
    assert all(item.caption == "a clear caption" for item in result.captions)
    assert result.result_uri.endswith("/captions.json")


def test_caption_images_respects_max_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    for index in range(5):
        _write_image(images / f"frame-{index}.png", (index * 10, 0, 0))

    result = caption_images(
        input_path=str(images),
        output_path=str(tmp_path / "out"),
        max_images=2,
        client=_client("cap"),
    )

    assert result.image_count == 2


def test_caption_images_no_images_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TokenFactoryToolError):
        caption_images(input_path=str(empty), output_path=str(tmp_path / "out"), client=_client("x"))


def test_generate_text_from_jsonl(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "prompt": "Write a task instruction"}),
                json.dumps({"prompt": "Another prompt"}),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gen"

    result = generate_text(
        input_path=str(prompts),
        output_path=str(output),
        client=_client("generated text"),
    )

    assert result.prompt_count == 2
    assert result.generations[0].id == "p1"
    assert result.generations[1].id == "item-0002"
    assert all(item.completion == "generated text" for item in result.generations)
    assert result.result_uri.endswith("/generations.jsonl")


def test_generate_text_from_txt_lines(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")

    result = generate_text(
        input_path=str(prompts),
        output_path=str(tmp_path / "gen"),
        client=_client("ok"),
    )

    assert result.prompt_count == 2
    assert [item.prompt for item in result.generations] == ["first prompt", "second prompt"]


def test_generate_text_missing_prompt_field_raises(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    with pytest.raises(TokenFactoryToolError):
        generate_text(input_path=str(prompts), output_path=str(tmp_path / "gen"), client=_client("x"))


def test_reason_scene_sends_images_and_returns_plan(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "a.png", (10, 20, 30))
    _write_image(scene / "b.png", (40, 50, 60))
    captured: dict = {}

    result = reason_scene(
        input_path=str(scene),
        output_path=str(tmp_path / "out"),
        task="What should the robot do here?",
        client=_capturing_client("1. approach 2. grasp", captured),
    )

    assert result.status == "completed"
    assert result.model == DEFAULT_REASONER_MODEL
    assert result.image_count == 2
    assert result.images == ["a.png", "b.png"]
    assert result.analysis == "1. approach 2. grasp"
    assert result.result_uri.endswith("/scene_reasoning.json")
    # One request carries the task text plus both images.
    content = captured["body"]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "What should the robot do here?"}
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 2


def test_reason_scene_respects_max_images(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    for index in range(5):
        _write_image(scene / f"f-{index}.png", (index * 10, 0, 0))

    result = reason_scene(
        input_path=str(scene),
        output_path=str(tmp_path / "out"),
        max_images=3,
        client=_client("plan"),
    )

    assert result.image_count == 3


def test_reason_scene_no_images_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TokenFactoryToolError):
        reason_scene(input_path=str(empty), output_path=str(tmp_path / "out"), client=_client("x"))
