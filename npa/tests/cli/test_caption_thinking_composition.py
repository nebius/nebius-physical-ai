"""Exercise the composed caption helper, controls, and durable failures."""

from __future__ import annotations

import json
import ast
from pathlib import Path

import httpx
from PIL import Image
import pytest
from typer.testing import CliRunner
from typer import Exit

from npa.cli.main import app
from npa.clients.token_factory import (
    TokenFactoryClient,
    default_chat_extra,
    resolve_config,
    thinking_chat_extra,
)
import npa.workbench.token_factory as tool
from npa.sdk.workbench import token_factory as sdk

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


def test_owned_caption_live_protocol_requires_explicit_opt_in():
    path = (
        Path(tool.__file__).resolve().parents[4]
        / "tests/e2e/test_token_factory_caption_thinking_live.py"
    )
    module = ast.parse(path.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        )
    )
    assert {
        node.attr
        for node in ast.walk(assignment.value)
        if isinstance(node, ast.Attribute)
    } >= {"e2e", "token_factory_e2e"}


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
    expected_failures = 0 if case == "positive" else 1
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
        unavailable = reply in {"NO IMAGE RECEIVED.", "**“'no image received.'”**"}
        assert item["status"] == ("image_unavailable" if unavailable else "completed")
    assert payload["captions"][-1]["caption"] == replies[-1]


def _direct_caption_args(args, thinking):
    return {
        "input_path": args[args.index("--input-path") + 1],
        "output_path": args[args.index("--output-path") + 1],
        "model": "moonshotai/Kimi-K3",
        "thinking": thinking,
    }


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("path", ["cli", "sdk", "module"])
def test_kimi_thinking_refuses_before_transport_and_artifacts(
    monkeypatch, tmp_path, thinking, path
):
    requests = []
    _install_transport(monkeypatch, ["answer"], requests)
    args = _caption_args(tmp_path, "moonshotai/Kimi-K3", thinking, 1)
    direct = _direct_caption_args(args, thinking)
    if path == "cli":
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 1
        assert "reasoning_effort" in result.output
    elif path == "sdk":
        with pytest.raises(Exit) as failure:
            sdk.caption(**direct)
        assert failure.value.exit_code == 1
    else:
        with pytest.raises(tool.TokenFactoryToolError, match="reasoning_effort"):
            tool.caption_images(**direct)
    assert requests == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("path", ["cli", "sdk", "module"])
def test_kimi_omitted_control_keeps_known_request_profile(monkeypatch, tmp_path, path):
    requests = []
    _install_transport(monkeypatch, ["answer"], requests)
    args = _caption_args(tmp_path, "moonshotai/Kimi-K3", None, 1)
    direct = _direct_caption_args(args, None)
    if path == "cli":
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
    elif path == "sdk":
        sdk.caption(**direct)
    else:
        assert tool.caption_images(**direct).status == "completed"
    assert len(requests) == 1
    assert requests[0]["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in requests[0]
    assert "temperature" not in requests[0]


@pytest.mark.parametrize("thinking", [None, True, False])
def test_actual_sdk_partial_failure_is_persisted_and_raises(
    monkeypatch, tmp_path, capsys, thinking
):
    replies, requests = _CASES["mixed"], []
    _install_transport(monkeypatch, replies, requests)
    args = _caption_args(tmp_path, "MiniMaxAI/MiniMax-M3", thinking, len(replies))
    with pytest.raises(Exit) as failure:
        sdk.caption(
            input_path=args[args.index("--input-path") + 1],
            output_path=args[args.index("--output-path") + 1],
            model="MiniMaxAI/MiniMax-M3",
            thinking=thinking,
            output="json",
        )
    assert failure.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads((tmp_path / "out/captions.json").read_text())
    assert payload["written_uri"] == str(tmp_path / "out/captions.json")
    assert {key: value for key, value in payload.items() if key != "written_uri"} == (
        persisted
    )
    assert payload["status"] == "failed"
    assert payload["failed_count"] == 1
    assert len(requests) == 3
    _assert_records(payload, requests, replies, "MiniMaxAI/MiniMax-M3", thinking)
