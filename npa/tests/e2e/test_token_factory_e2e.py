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

import json
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from npa.clients.token_factory import (
    DEFAULT_REASONER_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TokenFactoryClient,
    resolve_config,
)
from npa.workbench.token_factory import reason_scene

pytestmark = pytest.mark.token_factory_e2e


def test_live_provider_contract_recheck(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Recheck real outputs, thinking controls and structured-output drift."""
    _require_key()
    from npa.live_verification.token_factory_contract import run_contract

    additional = tuple(filter(None, (
        value.strip() for value in os.environ.get("NPA_TF_RECHECK_REQUIRED_MODELS", "").split(",")
    )))
    report = run_contract(
        TokenFactoryClient(), additional_models=additional,
        expected_json_behavior=os.environ.get("NPA_TF_RECHECK_JSON_BASELINE", "malformed_json"),
    )
    request.node.user_properties.append(("provider_contract", report))
    (tmp_path / "provider-contract.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = [check["check"] for check in report["checks"] if not check["passed"]]
    assert report["passed"], f"Provider contract drift: {', '.join(failures)}"


def _require_key() -> str:
    config = resolve_config(require_api_key=False)
    if not config.api_key:
        pytest.skip(
            "Live Token Factory test requires NEBIUS_TOKEN_FACTORY_KEY in the environment "
            "or ~/.npa/credentials.yaml (tokens.NEBIUS_TOKEN_FACTORY_KEY)."
        )
    return config.api_key


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
    model = DEFAULT_TEXT_MODEL
    text = client.chat_completion_text(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],

    )
    assert isinstance(text, str)
    assert text.strip()


def test_live_default_reasoner_scene_plan(tmp_path: Path) -> None:
    _require_key()
    scene = tmp_path / "scene"
    _write_scene_image(scene / "frame.png")

    result = reason_scene(
        input_path=str(scene),
        output_path=str(tmp_path / "out"),
        task="What objects are in this scene and what steps should a robot take to pick up the red box?",
        model=DEFAULT_REASONER_MODEL,

    )

    assert result.status == "completed"
    assert result.model == DEFAULT_REASONER_MODEL
    assert result.image_count == 1
    assert result.analysis.strip(), "reasoner returned an empty analysis"


@pytest.mark.parametrize("access_path", ["cli", "sdk"])
def test_live_generate_inventory_artifact(tmp_path: Path, access_path: str) -> None:
    """Exercise default selection, every prompt, and the durable output contract."""
    _require_key()
    from npa.cli.main import app
    from npa.sdk.workbench import token_factory
    from typer.testing import CliRunner

    source = tmp_path / "inventory.jsonl"
    counts = [(12, 8), (9, 4), (23, 17), (6, 11), (32, 9), (19, 1)]
    source.write_text("".join(json.dumps({
        "id": f"inventory-{index}",
        "prompt": f"There are {red} red crates and {blue} blue crates. "
        "Return only JSON with integer keys red, blue, total, without a code fence.",
    }) + "\n" for index, (red, blue) in enumerate(counts)))
    output = tmp_path / "generations.jsonl"
    if access_path == "cli":
        result = CliRunner().invoke(app, [
            "workbench", "token-factory", "generate", "--input-path", str(source),
            "--output-path", str(output), "--temperature", "0", "--output", "json",
        ])
        assert result.exit_code == 0, result.output
        stdout = result.output
    else:
        captured = StringIO()
        with redirect_stdout(captured):
            token_factory.generate(input_path=str(source), output_path=str(output),
                                   temperature=0, output="json")
        stdout = captured.getvalue()
    payload = json.loads(stdout)
    assert payload["status"] == "completed"
    assert payload["model"] == DEFAULT_TEXT_MODEL
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == payload["prompt_count"] == len(counts)
    for index, (row, (red, blue)) in enumerate(zip(rows, counts)):
        assert row["id"] == f"inventory-{index}"
        assert json.loads(row["completion"]) == {"red": red, "blue": blue, "total": red + blue}


def _shape_frame(path: Path, *, red_inside: bool) -> Path:
    """Synthetic diagram, not a simulated robot or a policy success claim."""
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((360, 130, 610, 430), outline="green", width=12)
    draw.ellipse((40, 30, 130, 120), fill="blue")
    x = 420 if red_inside else 170
    draw.rectangle((x, 230, x + 90, 320), fill="red")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_live_caption_and_reason_saved_artifacts(tmp_path: Path) -> None:
    _require_key()
    from npa.cli.main import app
    from typer.testing import CliRunner

    frames = tmp_path / "frames"
    for index, inside in enumerate((False, False, True)):
        _shape_frame(frames / f"frame-{index:03d}.png", red_inside=inside)
    runner = CliRunner()
    for command, filename in (("caption", "captions.json"), ("reason", "plan.json")):
        target = tmp_path / filename
        args = ["workbench", "token-factory", command, "--input-path", str(frames),
                "--output-path", str(target), "--output", "json"]
        if command == "reason":
            args += ["--task", "Describe the red square, blue circle, and green outline "
                     "in these diagrams and their change across ordered frames."]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        payload = json.loads(target.read_text())
        assert payload["status"] == "completed"
        assert payload["model"] == DEFAULT_VISION_MODEL
        assert payload["image_count"] == 3
        texts = ([row["caption"] for row in payload["captions"]] if command == "caption"
                 else [payload["analysis"]])
        assert all(all(color in text.lower() for color in ("red", "blue", "green"))
                   for text in texts)


@pytest.mark.parametrize("inside", [True, False])
def test_live_visual_judge_distinguishes_completion(tmp_path: Path, inside: bool) -> None:
    _require_key()
    from npa.workbench.vlm_eval import evaluate_vlm, write_result
    from dataclasses import asdict

    frames = tmp_path / "rollout"
    for index, state in enumerate((False, False, inside)):
        _shape_frame(frames / f"frame-{index:03d}.png", red_inside=state)
    output = tmp_path / "evaluation.json"
    result = evaluate_vlm(
        input_path=str(frames), output_path=str(output), backend="api",
        task="Move the red square fully inside the green rectangular outline by the final frame.",
        frame_selection="sequence",
    )
    write_result(asdict(result), result_uri=result.result_uri)
    saved = json.loads(output.read_text())
    assert saved["model"] == DEFAULT_VISION_MODEL
    assert saved["frame_count"] == 3
    assert saved["passed"] is inside
    assert saved["rationale"].strip()


def test_live_attribute_question_and_vision_chain(tmp_path: Path) -> None:
    _require_key()
    from dataclasses import asdict
    from npa.workbench.cosmos_evaluator.attribute_verification import verify_attributes

    frame = _shape_frame(tmp_path / "scene.png", red_inside=True)
    result = verify_attributes(
        clip_id="synthetic-diagram", frame=frame,
        selected_variables={"square_color": "red", "circle_color": "blue"},
        variable_options={"square_color": ["red", "yellow", "purple"],
                          "circle_color": ["blue", "orange", "black"]},
    )
    (tmp_path / "attributes.json").write_text(json.dumps(asdict(result), indent=2))
    assert result.question_model == DEFAULT_TEXT_MODEL
    assert result.vlm_model == DEFAULT_VISION_MODEL
    assert result.total_checks == result.passed_checks == 2
    assert result.passed
    assert all(check.question and check.vlm_answer and not check.error for check in result.checks)
