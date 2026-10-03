"""Exercise hosted image availability with frozen positive and no-image controls."""

from dataclasses import asdict
import json
from pathlib import Path

import httpx
import pytest

from npa.clients.token_factory import TokenFactoryClient, resolve_config
from npa.workbench import token_factory

pytestmark = pytest.mark.token_factory_e2e


def _client() -> TokenFactoryClient:
    if not resolve_config(require_api_key=False).api_key:
        pytest.skip("Token Factory live credentials are unavailable")
    return TokenFactoryClient()


def _input(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    source = root / "docs/testing/evidence/token-factory-image-availability-sentinel"
    image = tmp_path / "three-shapes.png"
    image.write_bytes((source / "three-shapes-submitted.png").read_bytes())
    return image


def test_live_caption_identifies_frozen_shapes(tmp_path: Path) -> None:
    result = token_factory.caption_images(
        input_path=str(_input(tmp_path)),
        output_path=str(tmp_path / "caption.json"),
        temperature=0,
        client=_client(),
    )
    (tmp_path / "caption-result.json").write_text(json.dumps(asdict(result)))
    assert result.status == "completed" and result.failed_count == 0
    assert result.model == "MiniMaxAI/MiniMax-M3"
    assert result.image_count == 1
    caption = result.captions[0].caption.lower()
    assert all(
        term in caption
        for term in ("red", "blue", "green", "square", "circle", "triangle")
    )


def test_live_no_image_response_fails_caption_replay(tmp_path: Path) -> None:
    client = _client()
    instruction = token_factory._caption_request_instruction(
        token_factory.DEFAULT_CAPTION_INSTRUCTION
    )
    response = client.chat_completion(
        model="MiniMaxAI/MiniMax-M3",
        messages=[{"role": "user", "content": [{"type": "text", "text": instruction}]}],
        temperature=0,
    )
    (tmp_path / "no-image-response.json").write_text(json.dumps(response))
    assert response["model"] == "MiniMaxAI/MiniMax-M3"
    assert response["choices"][0]["finish_reason"] == "stop"
    reply = response["choices"][0]["message"]["content"]
    assert reply.strip().casefold() == "no image received."
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    replay = TokenFactoryClient(http_client=httpx.Client(transport=transport))
    result = token_factory.caption_images(
        input_path=str(_input(tmp_path)),
        output_path=str(tmp_path / "replay.json"),
        client=replay,
    )
    (tmp_path / "replay-result.json").write_text(json.dumps(asdict(result)))
    assert result.status == "failed" and result.failed_count == 1
    assert result.captions[0].status == "image_unavailable"
    assert result.captions[0].caption == reply.strip()
