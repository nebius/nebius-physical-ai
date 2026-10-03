"""Exercise the composed caption helper, controls, and durable failures."""

from __future__ import annotations

import json

import httpx
from PIL import Image
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.token_factory import (
    TokenFactoryClient,
    default_chat_extra,
    resolve_config,
    thinking_chat_extra,
)
import npa.workbench.token_factory as tool

_CASES = {
    "positive": ["A green circle is visible."],
    "exact": ["NO IMAGE RECEIVED."],
    "nested": ["**“'no image received.'”**"],
    "mixed": [
        "A green circle is visible.",
        "**“'no image received.'”**",
        "A red square is visible.",
    ],
}


def _install_transport(monkeypatch, replies, requests):
    def handler(request):
        body = json.loads(request.content)
        reply = replies[len(requests)]
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
            },
        )

    client = TokenFactoryClient(
        resolve_config(api_key="test-key", environ={}),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(tool, "_default_client", lambda: client)


def _caption_args(tmp_path, model, thinking, count):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(count):
        Image.new("RGB", (16, 16), "green").save(images / f"{index}.png")
    args = [
        "workbench",
        "token-factory",
        "caption",
        "--input-path",
        str(images),
        "--output-path",
        str(tmp_path / "out"),
        "--output",
        "json",
        "--model",
        model,
    ]
    if thinking is not None:
        args.append("--thinking" if thinking else "--no-thinking")
    return args


@pytest.mark.parametrize("thinking", [None, True, False])
@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize(
    "model",
    ["MiniMaxAI/MiniMax-M3", "nvidia/Nemotron-3_5-Lightning", "vendor/custom-vision"],
)
def test_caption_thinking_composes_with_availability(
    monkeypatch, tmp_path, thinking, case, model
):
    replies, requests = _CASES[case], []
    _install_transport(monkeypatch, replies, requests)
    result = CliRunner().invoke(
        app, _caption_args(tmp_path, model, thinking, len(replies))
    )
    expected_failures = sum(
        tool._is_image_unavailable_answer(reply) for reply in replies
    )
    assert result.exit_code == (1 if expected_failures else 0), result.output
    payload = json.loads(result.output)
    persisted = json.loads((tmp_path / "out/captions.json").read_text())
    assert payload["captions"] == persisted["captions"]
    assert payload["failed_count"] == persisted["failed_count"] == expected_failures
    assert payload["image_count"] == len(replies) == len(requests)
    assert payload["status"] == ("failed" if expected_failures else "completed")
    assert persisted["status"] == payload["status"]
    _assert_records(payload, requests, replies, model, thinking)


def _assert_records(payload, requests, replies, model, thinking):
    expected = (
        default_chat_extra(model)
        if thinking is None
        else thinking_chat_extra(model, thinking)
    )
    for body, item, reply in zip(requests, payload["captions"], replies):
        assert body.get("chat_template_kwargs") == expected.get("chat_template_kwargs")
        assert body["model"] == model
        assert (
            tool.CAPTION_AVAILABILITY_DIRECTIVE
            in body["messages"][0]["content"][0]["text"]
        )
        assert item["caption"] == reply
        unavailable = tool._is_image_unavailable_answer(reply)
        assert item["status"] == ("image_unavailable" if unavailable else "completed")
    assert payload["captions"][-1]["caption"] == replies[-1]
