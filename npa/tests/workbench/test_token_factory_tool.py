from __future__ import annotations

import base64
import json
from io import BytesIO
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
    CAPTION_AVAILABILITY_DIRECTIVE,
    DEFAULT_CAPTION_INSTRUCTION,
    CaptionItem,
    CaptionResult,
    TokenFactoryToolError,
    _is_image_unavailable_answer,
    caption_images,
    generate_text,
    reason_scene,
)


def _client(reply: str) -> TokenFactoryClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(
        config, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _capturing_client(reply: str, captured: dict) -> TokenFactoryClient:
    import json as _json

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request_count"] = captured.get("request_count", 0) + 1
        captured["body"] = _json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(
        config, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


class _SequenceClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[dict] = []

    def chat_completion_text(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.replies[len(self.calls) - 1]


_SENTINEL_WRAPPERS = (
    ("plain", "", ""),
    ("markdown-asterisks", "**", "**"),
    ("markdown-underscores", "__", "__"),
    ("markdown-italic-asterisk", "*", "*"),
    ("markdown-italic-underscore", "_", "_"),
    ("ascii-single", "'", "'"),
    ("ascii-double", '"', '"'),
    ("smart-single", "‘", "’"),
    ("smart-double", "“", "”"),
)
_PRESENTATION_WRAPPERS = _SENTINEL_WRAPPERS[1:]
_SENTINEL_CORE_VARIANTS = (
    ("without-period", "NO IMAGE RECEIVED"),
    ("with-period", "NO IMAGE RECEIVED."),
)


def _nested_sentinel(depth: int) -> str:
    wrappers = (("“", "”"), ("_", "_"), ("*", "*"), ('"', '"'))
    answer = "NO IMAGE RECEIVED."
    for index in range(depth):
        opening, closing = wrappers[index % len(wrappers)]
        answer = f"{opening}  {answer}  {closing}"
    return answer


_ACCEPTED_SENTINEL_FORMATS = (
    [
        pytest.param(f"{opening}{core}{closing}", id=f"{name}-{core_name}")
        for name, opening, closing in _SENTINEL_WRAPPERS
        for core_name, core in _SENTINEL_CORE_VARIANTS
    ]
    + [
        pytest.param(
            f"{opening}  NO IMAGE RECEIVED.  {closing}",
            id=f"{name}-inner-whitespace",
        )
        for name, opening, closing in _SENTINEL_WRAPPERS
        if name != "plain"
    ]
    + [
        pytest.param(
            f"{opening}nO iMaGe ReCeIvEd.{closing}",
            id=f"{name}-case-variation",
        )
        for name, opening, closing in _SENTINEL_WRAPPERS
    ]
    + [
        pytest.param(
            f"{outer_opening}  {inner_opening}{core}{inner_closing}  {outer_closing}",
            id=f"nested-{outer_name}-{inner_name}-{core_name}",
        )
        for outer_name, outer_opening, outer_closing in _PRESENTATION_WRAPPERS
        for inner_name, inner_opening, inner_closing in _PRESENTATION_WRAPPERS
        for core_name, core in _SENTINEL_CORE_VARIANTS
    ]
    + [
        pytest.param('**"NO IMAGE RECEIVED."**', id="trigger-bold-around-quote"),
        pytest.param('"**NO IMAGE RECEIVED.**"', id="trigger-quote-around-bold"),
        pytest.param("***NO IMAGE RECEIVED.***", id="trigger-triple-asterisk"),
        pytest.param("___NO IMAGE RECEIVED.___", id="triple-underscore"),
    ]
    + [
        pytest.param(_nested_sentinel(depth), id=f"nested-depth-{depth}")
        for depth in range(1, 17)
    ]
)


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
    assert all(item.status == "completed" for item in result.captions)
    assert result.failed_count == 0
    assert result.result_uri.endswith("/captions.json")


@pytest.mark.parametrize(
    ("instruction", "expected_instruction"),
    [
        (None, DEFAULT_CAPTION_INSTRUCTION),
        ("Count the colored shapes.", "Count the colored shapes."),
    ],
)
def test_caption_images_appends_availability_directive(
    tmp_path: Path,
    instruction: str | None,
    expected_instruction: str,
) -> None:
    image = tmp_path / "frame.png"
    _write_image(image, (10, 20, 30))
    captured: dict = {}
    kwargs = {} if instruction is None else {"instruction": instruction}

    result = caption_images(
        input_path=str(image),
        output_path=str(tmp_path / "out"),
        client=_capturing_client("grounded caption", captured),
        **kwargs,
    )

    expected_request = f"{expected_instruction}\n\n{CAPTION_AVAILABILITY_DIRECTIVE}"
    prompt = captured["body"]["messages"][0]["content"][0]["text"]
    assert result.instruction == expected_instruction
    assert result.availability_directive == CAPTION_AVAILABILITY_DIRECTIVE
    assert result.request_instruction == expected_request
    assert prompt == expected_request
    content = captured["body"]["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert isinstance(url, str)
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    image_bytes = base64.b64decode(url.removeprefix(prefix), validate=True)
    assert image_bytes
    with Image.open(BytesIO(image_bytes)) as submitted:
        assert submitted.format == "PNG"
        assert submitted.width > 0
        assert submitted.height > 0
        submitted.verify()


def test_caption_dataclasses_preserve_positional_bindings() -> None:
    item = CaptionItem("frame.png", "grounded caption")
    result = CaptionResult(
        "completed",
        "frames",
        "output",
        "output/captions.json",
        "vision-model",
        "caption instruction",
        1,
        "2026-01-01T00:00:00+00:00",
        [item],
    )

    assert (item.image, item.caption, item.status) == (
        "frame.png",
        "grounded caption",
        "completed",
    )
    assert result.captions == [item]
    assert result.failed_count == 0
    assert result.availability_directive == CAPTION_AVAILABILITY_DIRECTIVE
    assert result.request_instruction == ""


def test_caption_images_marks_exact_sentinel_and_continues(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    for name in ("a.png", "b.png", "c.png"):
        _write_image(images / name, (10, 20, 30))
    client = _SequenceClient(
        ["first grounded caption", " \nno image received.\n ", "third grounded caption"]
    )

    result = caption_images(
        input_path=str(images),
        output_path=str(tmp_path / "out"),
        client=client,
    )

    assert len(client.calls) == 3
    assert result.status == "failed"
    assert result.image_count == 3
    assert result.failed_count == 1
    assert [item.image for item in result.captions] == ["a.png", "b.png", "c.png"]
    assert [item.status for item in result.captions] == [
        "completed",
        "image_unavailable",
        "completed",
    ]
    assert result.captions[1].caption == "no image received."


@pytest.mark.parametrize("reply", _ACCEPTED_SENTINEL_FORMATS)
def test_caption_images_normalizes_closed_whole_answer_formats(
    tmp_path: Path,
    reply: str,
) -> None:
    image = tmp_path / "images" / "frame.png"
    _write_image(image, (10, 20, 30))

    result = caption_images(
        input_path=str(image.parent),
        output_path=str(tmp_path / "out"),
        client=_SequenceClient([f" \n{reply}\n "]),
    )

    assert result.status == "failed"
    assert result.failed_count == 1
    assert result.captions[0] == CaptionItem(
        image="frame.png",
        caption=reply,
        status="image_unavailable",
    )


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("The sign says NO IMAGE RECEIVED.", id="longer-answer"),
        pytest.param('"The sign says NO IMAGE RECEIVED."', id="wrapped-longer-answer"),
        pytest.param(
            "NO IMAGE RECEIVED. Additional explanation.", id="sentinel-prefix"
        ),
        pytest.param("NO IMAGE RECEIVED!", id="arbitrary-punctuation"),
        pytest.param("NO IMAGE RECEIVED..", id="doubled-period"),
        pytest.param("NO IMAGE RECEIVED…", id="unicode-ellipsis"),
        pytest.param("NO IMAGE RECEIVED。", id="unicode-full-stop"),
        pytest.param("**NO IMAGE RECEIVED**.", id="punctuation-after-bold"),
        pytest.param('"NO IMAGE RECEIVED".', id="punctuation-after-quote"),
        pytest.param("`NO IMAGE RECEIVED.`", id="inline-code"),
        pytest.param("```NO IMAGE RECEIVED.```", id="code-fence"),
        pytest.param("*NO IMAGE RECEIVED.**", id="unmatched-asterisk-emphasis"),
        pytest.param("__NO IMAGE RECEIVED._", id="unmatched-underscore-emphasis"),
        pytest.param('“NO IMAGE RECEIVED."', id="mismatched-smart-ascii-double"),
        pytest.param('"NO IMAGE RECEIVED.”', id="mismatched-ascii-smart-double"),
        pytest.param("‘NO IMAGE RECEIVED.'", id="mismatched-smart-ascii-single"),
        pytest.param("'NO IMAGE RECEIVED.’", id="mismatched-ascii-smart-single"),
        pytest.param(
            "I'm sorry, I cannot see any image.", id="natural-language-paraphrase"
        ),
    ],
)
def test_caption_images_leaves_out_of_contract_answers_completed(
    tmp_path: Path,
    reply: str,
) -> None:
    image = tmp_path / "images" / "frame.png"
    _write_image(image, (10, 20, 30))

    result = caption_images(
        input_path=str(image.parent),
        output_path=str(tmp_path / "out"),
        client=_SequenceClient([reply]),
    )

    assert result.status == "completed"
    assert result.failed_count == 0
    assert result.captions[0] == CaptionItem(
        image="frame.png",
        caption=reply,
        status="completed",
    )


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("", id="empty"),
        pytest.param("**", id="bold-wrapper-only"),
        pytest.param("****", id="nested-asterisk-wrapper-only"),
        pytest.param('""', id="quote-wrapper-only"),
        pytest.param('"  *  _  “”  _  *  "', id="deep-wrapper-only"),
    ],
)
def test_wrapper_only_answer_is_a_classifier_only_non_match(reply: str) -> None:
    assert _is_image_unavailable_answer(reply) is False


def test_caption_images_preserves_actual_client_empty_response_error(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "frame.png"
    _write_image(image, (10, 20, 30))
    captured: dict = {}

    with pytest.raises(TokenFactoryToolError, match="response missing"):
        caption_images(
            input_path=str(image.parent),
            output_path=str(tmp_path / "out"),
            client=_capturing_client("", captured),
        )
    assert captured["request_count"] == 1
    assert not (tmp_path / "out" / "captions.json").exists()


def test_caption_images_does_not_substring_match_sentinel(tmp_path: Path) -> None:
    image = tmp_path / "images" / "frame.png"
    _write_image(image, (10, 20, 30))
    reply = 'The sign in the image says "NO IMAGE RECEIVED."'

    result = caption_images(
        input_path=str(image.parent),
        output_path=str(tmp_path / "out"),
        client=_client(reply),
    )

    assert result.status == "completed"
    assert result.failed_count == 0
    assert result.captions[0] == CaptionItem(
        image="frame.png",
        caption=reply,
        status="completed",
    )


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
        caption_images(
            input_path=str(empty),
            output_path=str(tmp_path / "out"),
            client=_client("x"),
        )


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
    assert [item.prompt for item in result.generations] == [
        "first prompt",
        "second prompt",
    ]


def test_generate_text_missing_prompt_field_raises(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    with pytest.raises(TokenFactoryToolError):
        generate_text(
            input_path=str(prompts),
            output_path=str(tmp_path / "gen"),
            client=_client("x"),
        )


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


def test_reason_scene_strips_think_block_from_analysis(tmp_path: Path) -> None:
    # Cosmos 3 reasoners prefix the answer with an inline <think> trace; it must
    # not surface in the stored analysis artifact.
    scene = tmp_path / "scene"
    _write_image(scene / "a.png", (10, 20, 30))

    result = reason_scene(
        input_path=str(scene),
        output_path=str(tmp_path / "out"),
        client=_client(
            "<think>\nthe cup is in front of the cube\n</think>\n1. clear the cup 2. grasp"
        ),
    )

    assert "<think>" not in result.analysis
    assert result.analysis == "1. clear the cup 2. grasp"


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
        reason_scene(
            input_path=str(empty),
            output_path=str(tmp_path / "out"),
            client=_client("x"),
        )
