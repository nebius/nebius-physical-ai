"""Public caption SDK callbacks preserve result-level failure and its artifact."""

import importlib.util
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest
import typer

from npa.clients.token_factory import TokenFactoryClient, TokenFactoryConfig
from npa.sdk.workbench import token_factory as sdk
from npa.workbench import token_factory


@pytest.mark.parametrize(
    "replies,dry_run",
    [
        (["A red square."], False),
        (["NO IMAGE RECEIVED."], False),
        (["A red square.", "NO IMAGE RECEIVED.", "A blue circle."], False),
        (["NO IMAGE RECEIVED."], True),
    ],
)
def test_caption_sdk_preserves_callback_exit_and_durable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, replies, dry_run: bool
) -> None:
    inputs = tmp_path / "images"
    inputs.mkdir()
    for index in range(len(replies)):
        Image.new("RGB", (12, 12), (index, 0, 0)).save(inputs / f"{index:02}.png")
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        reply = replies[len(calls)]
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "MiniMaxAI/MiniMax-M3",
                "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
            },
        )

    client = TokenFactoryClient(
        config=TokenFactoryConfig(
            base_url="https://example.test/v1", api_key="synthetic-test-key"
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(token_factory, "_default_client", lambda: client)
    target = tmp_path / "captions.json"
    kwargs = dict(
        input_path=str(inputs),
        output_path=str(target),
        output="json",
        dry_run=dry_run,
    )
    failures = replies.count("NO IMAGE RECEIVED.")
    if failures:
        with pytest.raises(typer.Exit) as error:
            sdk.caption(**kwargs)
        assert error.value.exit_code == 1
    else:
        assert sdk.caption(**kwargs) is None
    emitted = json.loads(capsys.readouterr().out)
    assert len(calls) == len(replies)
    assert emitted["failed_count"] == failures
    assert emitted["status"] == ("failed" if failures else "completed")
    assert [item["caption"] for item in emitted["captions"]] == replies
    assert [item["status"] for item in emitted["captions"]] == [
        "image_unavailable" if reply == "NO IMAGE RECEIVED." else "completed"
        for reply in replies
    ]
    if dry_run:
        assert emitted["dry_run"] is True
        assert "written_uri" not in emitted
        assert not target.exists()
    else:
        assert emitted.pop("written_uri") == str(target)
        assert json.loads(target.read_text()) == emitted


def test_live_positive_control_skips_before_reading_frozen_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "e2e/test_token_factory_image_availability_live.py"
    )
    spec = importlib.util.spec_from_file_location("caption_live_hygiene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order = []

    def unavailable():
        order.append("credential_skip")
        pytest.skip("Synthetic unavailable credentials")

    def forbidden_read(_):
        order.append("fixture_read")
        raise AssertionError("Frozen image must not be read before credential skip")

    monkeypatch.setattr(module, "_client", unavailable)
    monkeypatch.setattr(module, "_input", forbidden_read)
    with pytest.raises(pytest.skip.Exception, match="Synthetic unavailable"):
        module.test_live_caption_identifies_frozen_shapes(tmp_path)
    assert order == ["credential_skip"]
