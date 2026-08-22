"""Nebius Token Factory workbench tool.

Hackathon-ready building blocks that call Nebius Token Factory (an
OpenAI-compatible hosted-inference API) natively, with the same
``--input-path`` / ``--output-path`` S3 contract as every other workbench tool.

Capabilities, all zero-GPU because inference is hosted:

- ``caption_images``: caption / annotate a folder of images (or rollout frames)
  with a hosted vision model and write a JSON manifest.
- ``generate_text``: real-time text generation / transformation from a JSONL of
  prompts (for example synthetic task or scene-prompt generation for Cosmos and
  sim variation) and write a JSONL of completions.
- ``batch_generate`` / ``batch_collect``: the same prompt file and the same
  ``generations.jsonl`` output, run through Token Factory batch inference
  instead. Cheaper per token and unbounded in prompt count, but asynchronous:
  results arrive within a completion window rather than immediately.

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
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence
from uuid import uuid4

from PIL import Image

from npa.clients.token_factory import (
    BATCH_TERMINAL_STATUSES,
    DEFAULT_BATCH_MODEL,
    DEFAULT_BATCH_POLL_INTERVAL_S,
    DEFAULT_BATCH_TIMEOUT_S,
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_REASONER_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TokenFactoryClient,
    TokenFactoryError,
    split_reasoning,
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
BATCH_OPERATION_FILENAME = "batch_operation.json"
#: Folder the temporary request datasets are created under, so a batch run's
#: server-side scratch state is identifiable and separable from real datasets.
BATCH_DATASET_FOLDER = "/npa-batch"
BATCH_CUSTOM_ID_COLUMN = "custom_id"
BATCH_MESSAGES_COLUMN = "messages"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}


#: Callback invoked with each operation record while polling a batch.
OperationObserver = Callable[[dict[str, Any]], None] | None


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
class BatchUsage:
    """Token counts reported by the batch responses, when the model reports them."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class BatchResult:
    """Outcome of a batch-inference run.

    ``status`` is ``"completed"`` when generations were collected, or
    ``"pending"`` when the operation is still running and only ``operation_id``
    is meaningful. A pending result is not a failure: batch inference is
    asynchronous by design, so ``operation_id`` is the handle to collect later.
    """

    status: str
    input_path: str
    output_path: str
    result_uri: str
    model: str
    operation_id: str
    operation_status: str
    completion_window: str
    prompt_count: int
    generation_count: int
    generated_at: str
    #: ``request_counts`` from the batch record: total, completed, failed, and
    #: invalid rows. The only real progress signal while a batch is pending.
    request_counts: dict[str, int] = field(default_factory=dict)
    usage: BatchUsage = field(default_factory=BatchUsage)
    generations: list[GenerationItem] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


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
    "BatchResult",
    "BatchUsage",
    "CaptionItem",
    "CaptionResult",
    "GenerateResult",
    "GenerationItem",
    "ReasonResult",
    "TokenFactoryToolError",
    "batch_collect",
    "batch_generate",
    "batch_operation_uri_for",
    "caption_images",
    "caption_result_uri_for",
    "generate_result_uri_for",
    "generate_text",
    "list_models",
    "reason_result_uri_for",
    "reason_scene",
    "write_batch_operation",
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


def batch_generate(
    *,
    input_path: str,
    output_path: str,
    model: str = DEFAULT_BATCH_MODEL,
    system_prompt: str = DEFAULT_GENERATE_SYSTEM_PROMPT,
    max_prompts: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    wait: bool = True,
    poll_interval_s: float = DEFAULT_BATCH_POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    keep_datasets: bool = False,
    client: TokenFactoryClient | None = None,
    on_poll: OperationObserver = None,
) -> BatchResult:
    """Generate a completion for every prompt through Token Factory batch inference.

    Takes the same prompt file as :func:`generate_text` and produces the same
    ``generations.jsonl``, so a batch stage is a drop-in replacement for a
    real-time one. The difference is economics and latency: batch tokens are
    cheaper and the prompt count is not bounded by how long a stage can sit in a
    request loop, but the answers arrive asynchronously.

    With ``wait=False`` this returns as soon as the operation is accepted, with
    ``status="pending"`` and an ``operation_id`` for :func:`batch_collect`.
    """

    _require(input_path, "input_path")
    _require(output_path, "output_path")
    effective_model = model or DEFAULT_BATCH_MODEL
    effective_system = (system_prompt or "").strip()
    effective_window = (completion_window or DEFAULT_COMPLETION_WINDOW).strip()
    if max_tokens <= 0:
        raise TokenFactoryToolError("--max-tokens must be positive")
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

    rows = []
    for item_id, prompt in prompts:
        messages: list[dict[str, Any]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})
        rows.append(
            {
                BATCH_CUSTOM_ID_COLUMN: item_id,
                BATCH_MESSAGES_COLUMN: json.dumps(messages),
            }
        )

    dataset_name = f"npa-batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    try:
        dataset = active.create_dataset(
            name=dataset_name,
            folder=BATCH_DATASET_FOLDER,
            columns={BATCH_CUSTOM_ID_COLUMN: "string", BATCH_MESSAGES_COLUMN: "string"},
            rows=rows,
        )
    except TokenFactoryError as exc:
        raise TokenFactoryToolError(f"uploading {len(rows)} batch prompts failed: {exc}") from exc

    dataset_id = str(dataset.get("id") or "")
    dataset_version = str(dataset.get("current_version") or "")
    if not dataset_id or not dataset_version:
        raise TokenFactoryToolError(
            "Token Factory dataset response is missing 'id' or 'current_version'"
        )

    try:
        operation = active.create_batch_inference(
            model=effective_model,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            messages_column=BATCH_MESSAGES_COLUMN,
            custom_id_column=BATCH_CUSTOM_ID_COLUMN,
            max_tokens=max_tokens,
            completion_window=effective_window,
        )
    except TokenFactoryError as exc:
        _cleanup_datasets(active, [dataset_id])
        hint = ""
        if "text2text" in str(exc):
            # Batch inference accepts text models only, so a vision model is
            # rejected outright rather than failing per row later.
            hint = (
                f" Model {effective_model!r} is not a text model. Batch inference "
                "accepts text-to-text models only; caption images with "
                "`npa workbench token-factory caption` instead."
            )
        raise TokenFactoryToolError(f"starting batch inference failed: {exc}{hint}") from exc

    operation_id = str(operation.get("id") or "")
    if not operation_id:
        _cleanup_datasets(active, [dataset_id])
        raise TokenFactoryToolError("Token Factory operation response is missing 'id'")

    prompt_lookup = {item_id: prompt for item_id, prompt in prompts}

    if not wait:
        # The operation reads the source dataset server-side, so it must outlive
        # this process. batch_collect cleans both datasets up.
        return BatchResult(
            status="pending",
            input_path=input_path,
            output_path=output_path,
            result_uri=generate_result_uri_for(output_path),
            model=effective_model,
            operation_id=operation_id,
            operation_status=str(operation.get("status") or "queued"),
            completion_window=effective_window,
            prompt_count=len(prompts),
            generation_count=0,
            generated_at=_now(),
        )

    return _await_and_collect(
        active,
        operation_id=operation_id,
        input_path=input_path,
        output_path=output_path,
        model=effective_model,
        completion_window=effective_window,
        prompt_count=len(prompts),
        prompt_lookup=prompt_lookup,
        source_dataset_id=dataset_id,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
        keep_datasets=keep_datasets,
        on_poll=on_poll,
    )


def batch_collect(
    *,
    operation_id: str,
    output_path: str,
    input_path: str = "",
    wait: bool = True,
    poll_interval_s: float = DEFAULT_BATCH_POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    keep_datasets: bool = False,
    client: TokenFactoryClient | None = None,
    on_poll: OperationObserver = None,
) -> BatchResult:
    """Collect the results of a batch operation started earlier.

    With ``wait=False`` this reports the current status without blocking, so a
    caller can poll on its own schedule. Prompts are recovered from the
    operation's source dataset, so the original prompt file is not needed.
    """

    _require(operation_id, "operation_id")
    _require(output_path, "output_path")
    active = client or _default_client()

    try:
        operation = active.get_operation(operation_id)
    except TokenFactoryError as exc:
        raise TokenFactoryToolError(f"reading operation {operation_id} failed: {exc}") from exc

    source_dataset_id = _source_dataset_id(operation)
    prompt_lookup = _recover_prompts(active, source_dataset_id)
    model = str((operation.get("params") or {}).get("model") or "")
    window = str((operation.get("params") or {}).get("completion_window") or "")

    if not wait and str(operation.get("status") or "") not in BATCH_TERMINAL_STATUSES:
        return BatchResult(
            status="pending",
            input_path=input_path,
            output_path=output_path,
            result_uri=generate_result_uri_for(output_path),
            model=model,
            operation_id=operation_id,
            operation_status=str(operation.get("status") or "unknown"),
            completion_window=window,
            prompt_count=len(prompt_lookup),
            generation_count=0,
            generated_at=_now(),
            request_counts=_request_counts(_batch_record(active, operation_id)),
        )

    return _await_and_collect(
        active,
        operation_id=operation_id,
        input_path=input_path,
        output_path=output_path,
        model=model,
        completion_window=window,
        prompt_count=len(prompt_lookup),
        prompt_lookup=prompt_lookup,
        source_dataset_id=source_dataset_id,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
        keep_datasets=keep_datasets,
        on_poll=on_poll,
    )


def _await_and_collect(
    client: TokenFactoryClient,
    *,
    operation_id: str,
    input_path: str,
    output_path: str,
    model: str,
    completion_window: str,
    prompt_count: int,
    prompt_lookup: dict[str, str],
    source_dataset_id: str,
    poll_interval_s: float,
    timeout_s: float,
    keep_datasets: bool,
    on_poll: OperationObserver,
) -> BatchResult:
    try:
        operation = client.wait_for_operation(
            operation_id,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            on_poll=on_poll,
        )
    except TokenFactoryError as exc:
        raise TokenFactoryToolError(str(exc)) from exc

    status = str(operation.get("status") or "unknown")
    if status != "succeeded":
        explanation = _explain_failed_operation(
            client, operation, model=model, operation_id=operation_id
        )
        # The operation is terminal, so its scratch datasets are dead weight. A
        # timeout does not reach here: that operation is still running and still
        # needs its source rows.
        if not keep_datasets:
            _cleanup_datasets(client, [_destination_dataset_id(operation), source_dataset_id])
        raise TokenFactoryToolError(explanation)

    batch = _batch_record(client, operation_id)
    destination_id = _destination_dataset_id(operation)
    exported = _read_results(client, batch, destination_id, operation_id=operation_id)

    generations, failures, usage = _parse_batch_export(exported, prompt_lookup)
    if not generations and not failures:
        raise TokenFactoryToolError(
            f"batch operation {operation_id} succeeded but its result dataset "
            "contained no parsable responses"
        )

    if not keep_datasets:
        _cleanup_datasets(client, [destination_id, source_dataset_id])

    return BatchResult(
        status="completed",
        input_path=input_path,
        output_path=output_path,
        result_uri=generate_result_uri_for(output_path),
        model=model,
        operation_id=operation_id,
        operation_status=status,
        completion_window=completion_window,
        prompt_count=prompt_count or len(generations) + len(failures),
        generation_count=len(generations),
        generated_at=_now(),
        request_counts=_request_counts(batch),
        usage=usage,
        generations=generations,
        failures=failures,
    )


def _explain_failed_operation(
    client: TokenFactoryClient,
    operation: dict[str, Any],
    *,
    model: str,
    operation_id: str,
) -> str:
    """Build the most specific failure message the API will give up.

    The operations endpoint is the least informative source here: a failed batch
    reports a single empty error string. The per-row reason lives in the batch
    record's error file, so that is tried first.
    """

    status = str(operation.get("status") or "unknown")
    message = f"batch operation {operation_id} ended with status {status!r}"

    batch = _batch_record(client, operation_id)
    detail = _batch_error_text(client, batch)
    if detail:
        counts = _request_counts(batch)
        invalid = counts.get("invalid") or 0
        total = counts.get("total") or 0
        if invalid and total:
            message = f"{message} with {invalid}/{total} rows rejected"
        hint = ""
        if "not a known batch endpoint routing key" in detail:
            hint = (
                f" Model {model!r} is not available for batch inference, even "
                "though it may serve real-time requests. Try another model, or "
                "run the same prompts through `npa workbench token-factory generate`."
            )
        return f"{message}: {detail}{hint}"

    errors = client.operation_errors(operation_id)
    if errors:
        return f"{message}: {'; '.join(errors)}"
    return f"{message}, and Token Factory reported no error detail"


def _read_results(
    client: TokenFactoryClient,
    batch: dict[str, Any],
    destination_id: str,
    *,
    operation_id: str,
) -> str:
    """Return the raw result rows for a succeeded batch.

    The batch record's output file is the OpenAI-standard batch result JSONL, so
    it is preferred; exporting the destination dataset is the fallback for when
    the batch view is unavailable.
    """

    file_id = str(batch.get("output_file_id") or "")
    if file_id:
        try:
            return client.download_file(file_id)
        except TokenFactoryError:
            pass
    if not destination_id:
        raise TokenFactoryToolError(
            f"batch operation {operation_id} succeeded but exposed neither an "
            "output file nor a destination dataset to read results from"
        )
    try:
        return client.export_dataset(destination_id, output_format="jsonl")
    except TokenFactoryError as exc:
        raise TokenFactoryToolError(f"reading batch results failed: {exc}") from exc


def _batch_record(client: TokenFactoryClient, operation_id: str) -> dict[str, Any]:
    """Return the OpenAI-compatible batch view of an operation, or an empty dict.

    Only ever used to enrich a result or a failure message, so an unavailable
    batch view must not turn into the error the caller sees.
    """

    try:
        return client.get_batch(operation_id)
    except TokenFactoryError:
        return {}


def _request_counts(batch: dict[str, Any]) -> dict[str, int]:
    counts = batch.get("request_counts")
    if not isinstance(counts, dict):
        return {}
    return {key: int(value) for key, value in counts.items() if isinstance(value, int)}


def _batch_error_text(client: TokenFactoryClient, batch: dict[str, Any]) -> str:
    file_id = str(batch.get("error_file_id") or "")
    if not file_id:
        return ""
    try:
        return " ".join(client.download_file(file_id).split())[:800]
    except TokenFactoryError:
        return ""


def _parse_batch_export(
    exported: str,
    prompt_lookup: dict[str, str],
) -> tuple[list[GenerationItem], list[dict[str, Any]], BatchUsage]:
    """Turn an exported result dataset into generations, failures, and usage.

    The exported rows wrap an OpenAI-shaped response, but the wrapper key has
    varied between response bodies, so each candidate location is tried rather
    than assuming one.
    """

    generations: list[GenerationItem] = []
    failures: list[dict[str, Any]] = []
    prompt_tokens = completion_tokens = total_tokens = 0

    for index, line in enumerate(exported.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            failures.append({"id": f"row-{index:04d}", "error": "result row is not valid JSON"})
            continue
        if not isinstance(row, dict):
            failures.append({"id": f"row-{index:04d}", "error": "result row is not an object"})
            continue

        item_id = str(row.get("custom_id") or row.get("id") or f"row-{index:04d}")
        body = _response_body(row)
        if body is None:
            failures.append({"id": item_id, "error": "result row has no response body"})
            continue

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)

        error = body.get("error") or row.get("error")
        if error:
            failures.append({"id": item_id, "error": _stringify(error)})
            continue

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            failures.append({"id": item_id, "error": "response has no choices"})
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            failures.append({"id": item_id, "error": "response choice has no message"})
            continue
        visible, _reasoning = split_reasoning(message)
        if not visible:
            failures.append({"id": item_id, "error": "response had no visible content"})
            continue
        generations.append(
            GenerationItem(
                id=item_id,
                prompt=prompt_lookup.get(item_id, ""),
                completion=visible.strip(),
            )
        )

    return (
        generations,
        failures,
        BatchUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def _response_body(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("response", "body", "result", "output"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            # The standard batch row nests the completion one level deeper under
            # "body", and a per-row failure puts its "error" in the same place.
            inner = value.get("body")
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    inner = None
            if isinstance(inner, dict) and ("choices" in inner or "error" in inner):
                return inner
            return value
    if "choices" in row:
        return row
    return None


def _source_dataset_id(operation: dict[str, Any]) -> str:
    for entry in operation.get("src") or []:
        if isinstance(entry, dict) and entry.get("id"):
            return str(entry["id"])
    return ""


def _destination_dataset_id(operation: dict[str, Any]) -> str:
    for entry in operation.get("dst") or []:
        if isinstance(entry, dict) and entry.get("id"):
            return str(entry["id"])
    return ""


def _recover_prompts(client: TokenFactoryClient, dataset_id: str) -> dict[str, str]:
    """Best-effort recovery of prompt text from a batch run's source dataset.

    Prompts are only used to enrich the output artifact, so a source dataset
    that has already been deleted degrades the artifact rather than failing the
    collection.
    """

    if not dataset_id:
        return {}
    try:
        exported = client.export_dataset(dataset_id, output_format="jsonl")
    except TokenFactoryError:
        return {}

    prompts: dict[str, str] = {}
    for line in exported.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        item_id = str(row.get(BATCH_CUSTOM_ID_COLUMN) or "")
        raw = row.get(BATCH_MESSAGES_COLUMN)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None
        if not item_id or not isinstance(raw, list):
            continue
        for message in reversed(raw):
            if isinstance(message, dict) and message.get("role") == "user":
                prompts[item_id] = str(message.get("content") or "")
                break
    return prompts


def _cleanup_datasets(client: TokenFactoryClient, dataset_ids: Sequence[str]) -> None:
    """Delete the scratch datasets a batch run created, never raising.

    Leaving them behind only wastes project storage, so a cleanup failure must
    not mask the result the caller came for.
    """

    for dataset_id in dataset_ids:
        if not dataset_id:
            continue
        try:
            client.delete_dataset(dataset_id)
        except TokenFactoryError:
            continue


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


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


def batch_operation_uri_for(output_path: str) -> str:
    """Where a pending batch run records its operation handle.

    Sits beside the eventual ``generations.jsonl`` so a later ``batch-status``
    call, or a human, can find the operation id without the submitting process.
    """

    if output_path.endswith((".jsonl", ".json")):
        return output_path.rsplit("/", 1)[0] + f"/{BATCH_OPERATION_FILENAME}"
    return output_path.rstrip("/") + f"/{BATCH_OPERATION_FILENAME}"


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


def write_batch_operation(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text(
        body,
        result_uri=result_uri,
        filename=BATCH_OPERATION_FILENAME,
        storage_client=storage_client,
    )


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
