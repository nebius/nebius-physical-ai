"""Nebius Token Factory workbench tool.

Hackathon-ready building blocks that call Nebius Token Factory (an
OpenAI-compatible hosted-inference API) natively, with the same
``--input-path`` / ``--output-path`` S3 contract as every other workbench tool.

Two capabilities, both zero-GPU because inference is hosted:

- ``caption_images``: caption / annotate a folder of images (or rollout frames)
  with a hosted vision model and write a JSON manifest.
- ``generate_text``: batch text generation / transformation from a JSONL of
  prompts (for example synthetic task or scene-prompt generation for Cosmos and
  sim variation) and write a JSONL of completions.

All Token Factory request, auth, and endpoint logic lives in
``npa.clients.token_factory``; this module only shapes inputs and outputs.
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from PIL import Image

from npa.clients.token_factory import (
    DEFAULT_REASONER_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TokenFactoryClient,
    TokenFactoryError,
)

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient

DEFAULT_CAPTION_INSTRUCTION = (
    "Describe this image in one or two sentences. Focus on the objects, the "
    "scene, and any action taking place. Be concrete and factual."
)
DEFAULT_GENERATE_SYSTEM_PROMPT = (
    "You are a helpful assistant generating concise, high-quality text for a "
    "physical-AI dataset. Respond with the requested content only."
)
DEFAULT_REASON_SYSTEM_PROMPT = (
    "You are a physical-AI reasoning assistant for a mobile robot. Analyze the "
    "scene images carefully and reason about objects, spatial layout, motion, "
    "and physical interactions. Then produce a concrete, ordered plan of action "
    "the robot can execute to complete the requested task, calling out "
    "preconditions, hazards, and failure cases."
)
DEFAULT_REASON_TASK = (
    "Describe this scene and give a step-by-step plan of action a robot should "
    "follow to operate safely and usefully here."
)
DEFAULT_MAX_IMAGES = 50
DEFAULT_MAX_TOKENS = 512
DEFAULT_REASON_MAX_IMAGES = 8
DEFAULT_REASON_MAX_TOKENS = 1024
CAPTION_RESULT_FILENAME = "captions.json"
GENERATE_RESULT_FILENAME = "generations.jsonl"
REASON_RESULT_FILENAME = "scene_reasoning.json"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}


class TokenFactoryToolError(ValueError):
    """Raised when a Token Factory tool request is invalid."""


@dataclass(frozen=True)
class CaptionItem:
    image: str
    caption: str


@dataclass(frozen=True)
class CaptionResult:
    status: str
    input_path: str
    output_path: str
    result_uri: str
    model: str
    instruction: str
    image_count: int
    generated_at: str
    captions: list[CaptionItem] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationItem:
    id: str
    prompt: str
    completion: str


@dataclass(frozen=True)
class GenerateResult:
    status: str
    input_path: str
    output_path: str
    result_uri: str
    model: str
    prompt_count: int
    generated_at: str
    generations: list[GenerationItem] = field(default_factory=list)


@dataclass(frozen=True)
class ReasonResult:
    status: str
    input_path: str
    output_path: str
    result_uri: str
    model: str
    task: str
    image_count: int
    images: list[str]
    analysis: str
    generated_at: str


__all__ = [
    "CaptionItem",
    "CaptionResult",
    "GenerateResult",
    "GenerationItem",
    "ReasonResult",
    "TokenFactoryToolError",
    "caption_images",
    "caption_result_uri_for",
    "generate_result_uri_for",
    "generate_text",
    "list_models",
    "reason_result_uri_for",
    "reason_scene",
    "write_captions",
    "write_generations",
    "write_reason",
]


def list_models(*, client: TokenFactoryClient | None = None) -> list[str]:
    """Return the model IDs available to the configured Token Factory key."""

    active = client or _default_client()
    try:
        return active.list_models()
    except TokenFactoryError as exc:
        raise TokenFactoryToolError(str(exc)) from exc


def caption_images(
    *,
    input_path: str,
    output_path: str,
    model: str = DEFAULT_VISION_MODEL,
    instruction: str = DEFAULT_CAPTION_INSTRUCTION,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    client: TokenFactoryClient | None = None,
) -> CaptionResult:
    """Caption every image under ``input_path`` with a hosted vision model."""

    _require(input_path, "input_path")
    _require(output_path, "output_path")
    if max_images <= 0:
        raise TokenFactoryToolError("--max-images must be positive")
    effective_model = model or DEFAULT_VISION_MODEL
    effective_instruction = (instruction or DEFAULT_CAPTION_INSTRUCTION).strip()
    active = client or _default_client()

    with _materialized_input(input_path) as local_input:
        image_paths = _discover_image_paths(local_input)[:max_images]
        if not image_paths:
            raise TokenFactoryToolError(
                f"No images found in {input_path}. Expected files with suffixes: "
                f"{', '.join(sorted(IMAGE_SUFFIXES))}."
            )
        captions: list[CaptionItem] = []
        for image_path in image_paths:
            label = _relative_label(image_path, local_input)
            data_url = _image_to_data_url(image_path)
            try:
                text = active.chat_completion_text(
                    model=effective_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": effective_instruction},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except TokenFactoryError as exc:
                raise TokenFactoryToolError(f"captioning {label} failed: {exc}") from exc
            captions.append(CaptionItem(image=label, caption=text.strip()))

    return CaptionResult(
        status="completed",
        input_path=input_path,
        output_path=output_path,
        result_uri=caption_result_uri_for(output_path),
        model=effective_model,
        instruction=effective_instruction,
        image_count=len(captions),
        generated_at=_now(),
        captions=captions,
    )


def generate_text(
    *,
    input_path: str,
    output_path: str,
    model: str = DEFAULT_TEXT_MODEL,
    system_prompt: str = DEFAULT_GENERATE_SYSTEM_PROMPT,
    max_prompts: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    client: TokenFactoryClient | None = None,
) -> GenerateResult:
    """Generate a completion for each prompt in a JSONL/text input file."""

    _require(input_path, "input_path")
    _require(output_path, "output_path")
    effective_model = model or DEFAULT_TEXT_MODEL
    effective_system = (system_prompt or DEFAULT_GENERATE_SYSTEM_PROMPT).strip()
    active = client or _default_client()

    with _materialized_input(input_path) as local_input:
        prompts = _load_prompts(local_input)
    if not prompts:
        raise TokenFactoryToolError(
            f"No prompts found in {input_path}. Expected a .jsonl file with "
            '{"id": ..., "prompt": ...} objects or a .txt file with one prompt per line.'
        )
    if max_prompts and max_prompts > 0:
        prompts = prompts[:max_prompts]

    generations: list[GenerationItem] = []
    for item_id, prompt in prompts:
        messages: list[dict[str, Any]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})
        try:
            text = active.chat_completion_text(
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TokenFactoryError as exc:
            raise TokenFactoryToolError(f"generation for {item_id!r} failed: {exc}") from exc
        generations.append(GenerationItem(id=item_id, prompt=prompt, completion=text.strip()))

    return GenerateResult(
        status="completed",
        input_path=input_path,
        output_path=output_path,
        result_uri=generate_result_uri_for(output_path),
        model=effective_model,
        prompt_count=len(generations),
        generated_at=_now(),
        generations=generations,
    )


def reason_scene(
    *,
    input_path: str,
    output_path: str,
    task: str = DEFAULT_REASON_TASK,
    model: str = DEFAULT_REASONER_MODEL,
    system_prompt: str = DEFAULT_REASON_SYSTEM_PROMPT,
    max_images: int = DEFAULT_REASON_MAX_IMAGES,
    max_tokens: int = DEFAULT_REASON_MAX_TOKENS,
    temperature: float = 0.2,
    client: TokenFactoryClient | None = None,
) -> ReasonResult:
    """Reason over scene images with a hosted physical-AI reasoner.

    Sends the scene images plus the task to the reasoning model (default
    ``nvidia/Cosmos3-Super-Reasoner``) in a single request and returns the
    model's scene understanding and plan of action. Built for the "walk the
    robot to a scene, ask what to do" physical-common-sense loop.
    """

    _require(input_path, "input_path")
    _require(output_path, "output_path")
    if max_images <= 0:
        raise TokenFactoryToolError("--max-images must be positive")
    effective_model = model or DEFAULT_REASONER_MODEL
    effective_task = (task or DEFAULT_REASON_TASK).strip()
    effective_system = (system_prompt or DEFAULT_REASON_SYSTEM_PROMPT).strip()
    active = client or _default_client()

    with _materialized_input(input_path) as local_input:
        image_paths = _discover_image_paths(local_input)[:max_images]
        if not image_paths:
            raise TokenFactoryToolError(
                f"No scene images found in {input_path}. Expected files with suffixes: "
                f"{', '.join(sorted(IMAGE_SUFFIXES))}."
            )
        labels = [_relative_label(path, local_input) for path in image_paths]
        content: list[dict[str, Any]] = [{"type": "text", "text": effective_task}]
        for image_path in image_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}}
            )
        messages: list[dict[str, Any]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": content})
        try:
            analysis = active.chat_completion_text(
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TokenFactoryError as exc:
            raise TokenFactoryToolError(f"scene reasoning failed: {exc}") from exc

    return ReasonResult(
        status="completed",
        input_path=input_path,
        output_path=output_path,
        result_uri=reason_result_uri_for(output_path),
        model=effective_model,
        task=effective_task,
        image_count=len(labels),
        images=labels,
        analysis=analysis.strip(),
        generated_at=_now(),
    )


def caption_result_uri_for(output_path: str) -> str:
    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{CAPTION_RESULT_FILENAME}"


def generate_result_uri_for(output_path: str) -> str:
    if output_path.endswith((".jsonl", ".json")):
        return output_path
    return output_path.rstrip("/") + f"/{GENERATE_RESULT_FILENAME}"


def reason_result_uri_for(output_path: str) -> str:
    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{REASON_RESULT_FILENAME}"


def write_captions(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text(body, result_uri=result_uri, filename=CAPTION_RESULT_FILENAME, storage_client=storage_client)


def write_generations(
    generations: Sequence[dict[str, Any]],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in generations)
    return _write_text(body, result_uri=result_uri, filename=GENERATE_RESULT_FILENAME, storage_client=storage_client)


def write_reason(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text(body, result_uri=result_uri, filename=REASON_RESULT_FILENAME, storage_client=storage_client)


def _write_text(
    body: str,
    *,
    result_uri: str,
    filename: str,
    storage_client: "StorageClient | None",
) -> str:
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-token-factory-") as tmp:
            local_path = Path(tmp) / filename
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix not in {".json", ".jsonl"}:
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _default_client() -> TokenFactoryClient:
    try:
        return TokenFactoryClient()
    except TokenFactoryError as exc:
        raise TokenFactoryToolError(str(exc)) from exc


@contextmanager
def _materialized_input(input_path: str) -> Iterator[Path]:
    if not input_path.startswith("s3://"):
        yield Path(input_path)
        return

    from npa.clients.storage import StorageClient

    with tempfile.TemporaryDirectory(prefix="npa-token-factory-input-") as tmp:
        local = StorageClient.from_environment().download_path(input_path, tmp)
        yield Path(local)


def _discover_image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )


def _relative_label(image_path: Path, base: Path) -> str:
    try:
        return str(image_path.relative_to(base))
    except ValueError:
        return image_path.name


def _image_to_data_url(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((768, 768))
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_prompts(local_input: Path) -> list[tuple[str, str]]:
    path = _resolve_prompt_file(local_input)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return _parse_jsonl_prompts(text)
    if path.suffix.lower() == ".json":
        return _parse_json_prompts(text)
    return [
        (f"line-{index:04d}", line.strip())
        for index, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


def _resolve_prompt_file(local_input: Path) -> Path:
    if local_input.is_file():
        return local_input
    if local_input.is_dir():
        for name in ("prompts.jsonl", "prompts.json", "prompts.txt"):
            candidate = local_input / name
            if candidate.is_file():
                return candidate
        for suffix in (".jsonl", ".json", ".txt"):
            matches = sorted(local_input.glob(f"*{suffix}"))
            if matches:
                return matches[0]
    raise TokenFactoryToolError(
        f"No prompt file found at {local_input}. Expected a .jsonl, .json, or .txt file."
    )


def _parse_jsonl_prompts(text: str) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TokenFactoryToolError(f"prompt file line {index} is not valid JSON") from exc
        prompts.append(_prompt_from_object(payload, index))
    return prompts


def _parse_json_prompts(text: str) -> list[tuple[str, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TokenFactoryToolError("prompt file is not valid JSON") from exc
    if isinstance(payload, dict):
        payload = payload.get("prompts") or payload.get("items") or []
    if not isinstance(payload, list):
        raise TokenFactoryToolError("prompt JSON must be a list or have a 'prompts' list")
    return [_prompt_from_object(item, index) for index, item in enumerate(payload, start=1)]


def _prompt_from_object(payload: Any, index: int) -> tuple[str, str]:
    if isinstance(payload, str):
        return (f"item-{index:04d}", payload.strip())
    if not isinstance(payload, dict):
        raise TokenFactoryToolError(f"prompt item {index} must be a string or object")
    prompt = payload.get("prompt") or payload.get("text") or payload.get("instruction")
    if not prompt:
        raise TokenFactoryToolError(f"prompt item {index} must include a 'prompt' field")
    item_id = str(payload.get("id") or payload.get("name") or f"item-{index:04d}").strip()
    return (item_id or f"item-{index:04d}", str(prompt).strip())


def _require(value: str, name: str) -> None:
    if not value:
        raise TokenFactoryToolError(f"{name} is required")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
