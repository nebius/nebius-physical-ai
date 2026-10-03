"""Retain real caption responses for a frozen tri-state thinking protocol."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
import pytest

from npa.clients.token_factory import TokenFactoryClient, resolve_config
from npa.workbench.token_factory import TokenFactoryToolError, caption_images

pytestmark = pytest.mark.token_factory_e2e
_INSTRUCTION = (
    "Describe only the visible shapes and their colors. State when no shape is visible."
)
_MODELS = ("MiniMaxAI/MiniMax-M3", "openbmb/MiniCPM-V-4_5")


def _evidence_path(tmp_path: Path) -> Path:
    configured = os.environ.get("NPA_CAPTION_THINKING_EVIDENCE_DIR")
    evidence = Path(configured) if configured else tmp_path / "evidence"
    evidence.mkdir(parents=True, mode=0o700)
    return evidence


def _write_controls(evidence: Path) -> tuple[Path, Path]:
    positive = evidence / "green-disk.png"
    blank = evidence / "blank.png"
    image = Image.new("RGB", (192, 192), "white")
    ImageDraw.Draw(image).ellipse((48, 48, 144, 144), fill="green")
    image.save(positive)
    Image.new("RGB", (192, 192), "white").save(blank)
    return positive, blank


def _transport_client(evidence: Path, records: list[dict]) -> TokenFactoryClient:
    def request(event):
        records.append({"request": json.loads(event.content)})

    def response(event):
        event.read()
        records[-1].update(
            http_status=event.status_code,
            raw_response=event.text,
            response_sha256=hashlib.sha256(event.content).hexdigest(),
        )
        (evidence / f"transport-{len(records):02d}.json").write_text(
            json.dumps(records[-1], indent=2) + "\n"
        )

    transport = httpx.Client(event_hooks={"request": [request], "response": [response]})
    return TokenFactoryClient(http_client=transport)


def _call_caption(evidence, client, model, thinking, image):
    case = f"{model.rsplit('/', 1)[-1]}-{thinking}-{image.stem}"
    result = {
        "model": model,
        "thinking": thinking,
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    try:
        caption = caption_images(
            input_path=str(image),
            output_path=str(evidence / case),
            model=model,
            thinking=thinking,
            instruction=_INSTRUCTION,
            temperature=0.0,
            client=client,
        )
        result.update(status="completed", caption=caption.captions[0].caption)
    except TokenFactoryToolError as exc:
        result.update(status="failed", error=str(exc))
    (evidence / f"result-{case}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_live_thinking_controls_keep_every_provider_outcome(tmp_path):
    if not resolve_config(require_api_key=False).api_key:
        pytest.skip("Token Factory credential is required")
    available = TokenFactoryClient().list_models()
    assert set(_MODELS) <= set(available)
    evidence = _evidence_path(tmp_path)
    positive, blank = _write_controls(evidence)
    protocol = {
        "models": _MODELS,
        "thinking": [None, False, True],
        "instruction": _INSTRUCTION,
        "temperature": 0.0,
        "limitation": "Synthetic image capability and control propagation only; no quality calibration.",
    }
    (evidence / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    client = _transport_client(evidence, records)
    outcomes = []
    for model in _MODELS:
        for thinking in (None, False, True):
            outcomes.append(_call_caption(evidence, client, model, thinking, positive))
        outcomes.append(_call_caption(evidence, client, model, False, blank))
    (evidence / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    _assert_controls(records, outcomes)


def _assert_controls(records, outcomes):
    assert len(records) == len(outcomes) == 8
    for record, outcome in zip(records, outcomes):
        assert record["http_status"] == 200
        request = record["request"]
        model, thinking = outcome["model"], outcome["thinking"]
        if model == "MiniMaxAI/MiniMax-M3":
            expected = "enabled" if thinking is True else "disabled"
            assert request["chat_template_kwargs"] == {"thinking_mode": expected}
        elif thinking is None:
            assert "chat_template_kwargs" not in request
        else:
            assert request["chat_template_kwargs"] == {"thinking": thinking}
        response = json.loads(record["raw_response"])
        assert response["model"] == model
        if outcome["status"] == "failed":
            assert "no visible caption" in outcome["error"]
        else:
            assert outcome["caption"].strip()
