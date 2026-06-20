"""Live Nebius Token Factory API tests.

These are first-class live tests: they hit the real Token Factory endpoint and
require a real ``NEBIUS_TOKEN_FACTORY_KEY``. They self-skip when no key is configured, so
they are safe to leave in the suite. Run explicitly with:

    NEBIUS_TOKEN_FACTORY_KEY=... npa/.venv/bin/python -m pytest \
        npa/tests/e2e/test_token_factory_e2e.py -v

They live under ``tests/e2e`` (excluded from the default unit run via
``--ignore=tests/e2e``) and are marked ``token_factory_e2e``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from npa.clients.token_factory import (
    DEFAULT_REASONER_MODEL,
    DEFAULT_TEXT_MODEL,
    TokenFactoryClient,
    resolve_config,
)
from npa.workbench.token_factory import reason_scene

pytestmark = pytest.mark.token_factory_e2e


def _require_key() -> str:
    config = resolve_config(require_api_key=False)
    if not config.api_key:
        pytest.skip(
            "Live Token Factory test requires NEBIUS_TOKEN_FACTORY_KEY in the environment "
            "or ~/.npa/credentials.yaml (tokens.NEBIUS_TOKEN_FACTORY_KEY)."
        )
    return config.api_key


def _has_model(client: TokenFactoryClient, model_id: str) -> bool:
    try:
        models = client.list_models()
    except Exception:  # pragma: no cover - network dependent
        return False
    return model_id in models


def _write_scene_image(path: Path) -> Path:
    image = Image.new("RGB", (640, 480), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 360, 640, 480], fill=(120, 90, 60))  # floor
    draw.rectangle([260, 250, 380, 360], fill=(180, 40, 40))  # a box on the floor
    draw.rectangle([60, 120, 160, 360], fill=(60, 60, 180))  # a wall/shelf
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_live_list_models_authenticates() -> None:
    _require_key()
    models = TokenFactoryClient().list_models()
    assert isinstance(models, list)
    assert models, "Token Factory returned no models for this key"


def test_live_text_chat_completion() -> None:
    _require_key()
    client = TokenFactoryClient()
    model = DEFAULT_TEXT_MODEL if _has_model(client, DEFAULT_TEXT_MODEL) else client.list_models()[0]
    text = client.chat_completion_text(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=16,
    )
    assert isinstance(text, str)
    assert text.strip()


def test_live_cosmos_super_reasoner_scene_plan(tmp_path: Path) -> None:
    _require_key()
    client = TokenFactoryClient()
    if not _has_model(client, DEFAULT_REASONER_MODEL):
        pytest.skip(
            f"{DEFAULT_REASONER_MODEL} is not available for this key; "
            "check `npa workbench token-factory models`."
        )
    scene = tmp_path / "scene"
    _write_scene_image(scene / "frame.png")

    result = reason_scene(
        input_path=str(scene),
        output_path=str(tmp_path / "out"),
        task="What objects are in this scene and what steps should a robot take to pick up the red box?",
        model=DEFAULT_REASONER_MODEL,
        max_tokens=512,
    )

    assert result.status == "completed"
    assert result.model == DEFAULT_REASONER_MODEL
    assert result.image_count == 1
    assert result.analysis.strip(), "reasoner returned an empty analysis"
