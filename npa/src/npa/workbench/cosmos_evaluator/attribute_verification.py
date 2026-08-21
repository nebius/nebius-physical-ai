"""Cosmos Evaluator attribute-verification check driven by Nebius Token Factory.

Upstream (``checks/attribute_verification`` in
https://github.com/nvidia-cosmos/cosmos-evaluator, Apache-2.0, Copyright (c)
2026 NVIDIA CORPORATION & AFFILIATES) verifies that an augmented clip really
shows the attributes it was asked for, in two hops:

1. ``LLMQuestionGenerator`` asks an LLM for one multiple-choice question per
   attribute, constrained so the options come from that attribute's option list
   and the requested value is among them (guided JSON schema, with tolerant
   parsing when the endpoint does not support structured output); then
2. ``VLMVerifier`` extracts a frame from the augmented clip and asks a VLM to
   answer each question, scoring a pass when the answer letter matches.

Both hops are plain OpenAI-compatible chat completions in upstream, configured
by endpoint + model, so NPA runs them against Nebius Token Factory: zero GPU,
same protocol, same prompts, same JSON schema, same answer parsing. For the
Physical AI Data Factory this maps exactly onto the blueprint's own data — the
sampled appearance combo is ``selected_variables`` and the sampler's full
``APPEARANCE_VARIABLES`` table is ``variable_options``.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError

_log = logging.getLogger(__name__)

# Verbatim from upstream checks/cosmos_evaluator.yaml so question generation and
# answering behave the same way against a different endpoint.
QUESTION_SYSTEM_PROMPT = (
    "You are an expert at creating multiple choice verification questions.\n"
    "Your task is to generate a simple, direct question that can verify a specific "
    "attribute in a video frame.\n"
    "The question must have 2-4 answer options and test for a specific visual attribute.\n"
    "The question should be answerable by looking at a single frame from the video.\n"
    "Output your response as a single JSON object with no additional text or formatting.\n"
)
VERIFY_SYSTEM_PROMPT = (
    "You are an expert vision model tasked with answering multiple choice questions about images.\n"
    "The image may be a left-to-right beginning, middle, and end contact sheet. "
    "Require the requested visual attribute to be consistent across its panels.\n"
    "Analyze the image carefully and select the single best answer from the provided options.\n"
    "Respond with ONLY a single letter (A, B, C, or D) corresponding to your answer.\n"
    "Do not include any explanation or additional text.\n"
)

# Upstream's guided-JSON response schema for one question.
QUESTION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "verification_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "variable": {"type": "string", "description": "Name of the variable being verified"},
                "value": {"type": "string", "description": "The selected value to verify"},
                "question": {"type": "string", "description": "The multiple choice question text"},
                "options": {
                    "type": "object",
                    "properties": {
                        "A": {"type": "string"},
                        "B": {"type": "string"},
                        "C": {"type": "string"},
                        "D": {"type": "string"},
                    },
                    "required": ["A", "B"],
                    "additionalProperties": False,
                },
                "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            },
            "required": ["variable", "value", "question", "options", "correct_answer"],
            "additionalProperties": False,
        },
    },
}

REQUIRED_QUESTION_FIELDS = ("variable", "value", "question", "options", "correct_answer")
ANSWER_LETTERS = ("A", "B", "C", "D")
DEFAULT_QUESTION_MAX_TOKENS = 2048
DEFAULT_VERIFY_MAX_TOKENS = 10
ATTRIBUTE_SAMPLE_POLICIES = frozenset({"ranking", "holdout"})


@dataclass(frozen=True)
class AttributeVerificationCheck:
    """Upstream's per-question record."""

    variable: str
    value: str
    question: str
    options: dict[str, str]
    expected_answer: str
    vlm_answer: str
    passed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttributeVerificationResult:
    """Upstream's ``AttributeVerificationResult`` plus the score the gate reads."""

    clip_id: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    score: float
    question_model: str
    vlm_model: str
    checks: list[AttributeVerificationCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload


def verify_attributes(
    *,
    clip_id: str,
    video: str | Path | None = None,
    frame: str | Path | None = None,
    selected_variables: dict[str, str],
    variable_options: dict[str, Sequence[str]] | None = None,
    question_model: str = "",
    vlm_model: str = "",
    client: Any | None = None,
    max_tokens: int = DEFAULT_VERIFY_MAX_TOKENS,
    sample_policy: str = "ranking",
) -> AttributeVerificationResult:
    """Verify that ``video`` (or ``frame``) shows every selected attribute value.

    Exactly one of ``video`` or ``frame`` is required. ``client`` defaults to a
    :class:`~npa.clients.token_factory.TokenFactoryClient`; tests pass a double.
    """

    from npa.clients.token_factory import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL

    if not selected_variables:
        raise CosmosEvaluatorError("attribute verification needs at least one selected variable")
    if (video is None) == (frame is None):
        raise CosmosEvaluatorError("pass exactly one of video= or frame=")

    active = client if client is not None else _default_client()
    llm_model = question_model or DEFAULT_TEXT_MODEL
    vision_model = vlm_model or DEFAULT_VISION_MODEL
    options_table = {key: list(values) for key, values in (variable_options or {}).items()}

    if sample_policy not in ATTRIBUTE_SAMPLE_POLICIES:
        raise CosmosEvaluatorError("sample_policy must be ranking or holdout")

    with _frame_image(
        video=video, frame=frame, sample_policy=sample_policy
    ) as image_path:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        data_url = f"data:{media_type};base64,{image_b64}"

        checks: list[AttributeVerificationCheck] = []
        for variable, value in selected_variables.items():
            options = options_table.get(variable) or [str(value)]
            checks.append(
                _verify_one(
                    client=active,
                    variable=str(variable),
                    value=str(value),
                    options=[str(option) for option in options],
                    data_url=data_url,
                    llm_model=llm_model,
                    vision_model=vision_model,
                    max_tokens=max_tokens,
                )
            )

    passed_checks = sum(1 for check in checks if check.passed and check.error is None)
    failed_checks = len(checks) - passed_checks
    score = (passed_checks / len(checks)) if checks else 0.0
    return AttributeVerificationResult(
        clip_id=clip_id,
        passed=failed_checks == 0,
        total_checks=len(checks),
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        score=round(score, 6),
        question_model=llm_model,
        vlm_model=vision_model,
        checks=checks,
    )


def _verify_one(
    *,
    client: Any,
    variable: str,
    value: str,
    options: list[str],
    data_url: str,
    llm_model: str,
    vision_model: str,
    max_tokens: int,
) -> AttributeVerificationCheck:
    try:
        question = generate_question(
            client=client,
            variable=variable,
            value=value,
            options=options,
            model=llm_model,
        )
    except Exception as exc:  # noqa: BLE001 - one bad question must not drop the batch
        _log.warning("question generation failed for %r: %s", variable, exc, exc_info=True)
        return AttributeVerificationCheck(
            variable=variable,
            value=value,
            question="",
            options={},
            expected_answer="",
            vlm_answer="",
            passed=False,
            error=f"question generation failed: {exc}"[:300],
        )

    try:
        answer = answer_question(
            client=client,
            question=question["question"],
            options=question["options"],
            data_url=data_url,
            model=vision_model,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - record the failure, keep the batch
        _log.warning("VLM verification failed for %r: %s", variable, exc, exc_info=True)
        return AttributeVerificationCheck(
            variable=variable,
            value=value,
            question=str(question["question"]),
            options=dict(question["options"]),
            expected_answer=str(question["correct_answer"]),
            vlm_answer="",
            passed=False,
            error=f"vlm verification failed: {exc}"[:300],
        )

    expected = str(question["correct_answer"]).upper()
    return AttributeVerificationCheck(
        variable=variable,
        value=value,
        question=str(question["question"]),
        options=dict(question["options"]),
        expected_answer=expected,
        vlm_answer=answer,
        passed=answer == expected,
    )


def generate_question(
    *,
    client: Any,
    variable: str,
    value: str,
    options: Sequence[str],
    model: str,
    max_tokens: int = DEFAULT_QUESTION_MAX_TOKENS,
) -> dict[str, Any]:
    """Ask the LLM for one multiple-choice question about ``variable``.

    Tries upstream's guided-JSON schema first and falls back to upstream's
    tolerant text parsing when the endpoint rejects ``response_format``.
    """

    prompt = question_user_prompt(variable=variable, value=value, options=options)
    messages = [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = ""
    try:
        text = client.chat_completion_text(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            response_format=QUESTION_SCHEMA,
        )
    except Exception as exc:
        # Only retry the one thing a retry can fix. A 429, an auth failure, or a
        # timeout would fail again identically, so retrying doubles the load on an
        # endpoint already in trouble and buries the real cause under a message
        # about guided JSON.
        if not _looks_like_unsupported_response_format(exc):
            raise
        _log.info("guided JSON unsupported by %s (%s); retrying unstructured", model, exc)
        text = client.chat_completion_text(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
    question = parse_question_response(text)
    return normalize_question(question, variable=variable, value=value, options=options)


#: Substrings OpenAI-compatible endpoints use when they cannot honour a
#: ``response_format`` schema. Matched case-insensitively against the error text
#: because the wording, not the exception type, is what varies between servers.
_UNSUPPORTED_RESPONSE_FORMAT_MARKERS = (
    "response_format",
    "guided",
    "json_schema",
    "json schema",
    "structured output",
)


def _looks_like_unsupported_response_format(exc: BaseException) -> bool:
    """True when the endpoint rejected the schema itself, not the request."""

    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    # A rejected schema is a client error; 429 and 5xx are load, not capability.
    if status is not None and not 400 <= int(status) < 429:
        return False
    if status is not None and int(status) in {401, 403, 408}:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _UNSUPPORTED_RESPONSE_FORMAT_MARKERS)


def answer_question(
    *,
    client: Any,
    question: str,
    options: dict[str, str],
    data_url: str,
    model: str,
    max_tokens: int = DEFAULT_VERIFY_MAX_TOKENS,
) -> str:
    """Ask the VLM to answer ``question`` about the frame, returning a letter."""

    formatted = format_question(question, options)
    text = client.chat_completion_text(
        model=model,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return parse_answer_letter(text, offered=options.keys())


def question_user_prompt(*, variable: str, value: str, options: Sequence[str]) -> str:
    """Upstream's per-variable user prompt for question generation."""

    all_options = list(options)
    return f"""Generate a verification question for the following variable:
            Variable: {variable}
            Selected value: {value}
            All possible values: {all_options}

            Create ONE multiple choice question that verifies if the selected value is present in the video frame.

            Requirements:
            1. The question must have 2-4 answer options (A, B, C, D)
            2. Each option must be taken from this list of all possible values: {all_options}.
                Options not in the list are not allowed.
            3. The selected value '{value}' MUST be one of the options
            4. The question should be simple and direct
            5. Options should include other possible values from the list above
            6. Format the correct answer as a single letter (A, B, C, or D)

            Output MUST be a single JSON object with the following structure:
            {{
                "variable": "{variable}",
                "value": "{value}",
                "question": "Question text?",
                "options": {{"A": "option1", "B": "option2", etc.}},
                "correct_answer": "B"
            }}

            Do NOT include any explanatory text, markdown fences, or comments. Output ONLY the JSON object."""


def format_question(question: str, options: dict[str, str]) -> str:
    """Upstream's rendering of a question plus its lettered options."""

    options_text = "\n".join(f"{key}) {value}" for key, value in sorted(options.items()))
    return f"{question}\n{options_text}\n\nAnswer with only a single letter (A, B, C, or D)."


def parse_question_response(text: str) -> dict[str, Any]:
    """Extract the question object from a model reply.

    Mirrors upstream's tolerance: drop any reasoning block, prefer a fenced code
    block, strip comment artifacts, then scan for the first decodable JSON value.
    """

    if not text:
        raise CosmosEvaluatorError("question generation returned no content")
    think_close = text.rfind("</think>")
    base = text[think_close + len("</think>") :].strip() if think_close != -1 else text.strip()
    fenced = re.search(r"```(?:json|javascript|js|python)?([\s\S]*?)```", base, flags=re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else base
    candidate = re.sub(r"```[a-zA-Z]*", "", candidate)
    candidate = re.sub(r"```", "", candidate)
    candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.DOTALL)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", candidate):
        try:
            payload, _ = decoder.raw_decode(candidate, idx=match.start())
        except json.JSONDecodeError:
            continue
        unwrapped = _unwrap_question(payload)
        if unwrapped is not None:
            return unwrapped
    raise CosmosEvaluatorError("could not parse a verification question from the model output")


def _unwrap_question(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if all(key in payload for key in REQUIRED_QUESTION_FIELDS):
            return payload
        for key in ("question", "item", "data", "result", "output"):
            inner = payload.get(key)
            if isinstance(inner, dict) and all(field in inner for field in REQUIRED_QUESTION_FIELDS):
                return inner
    if isinstance(payload, list) and len(payload) == 1:
        return _unwrap_question(payload[0])
    return None


def normalize_question(
    question: dict[str, Any],
    *,
    variable: str,
    value: str,
    options: Sequence[str],
) -> dict[str, Any]:
    """Validate a generated question and repair the answer key when needed.

    Upstream trusts the generator's ``correct_answer``. A wrong key would make
    every clip fail for reasons that have nothing to do with the pixels, so the
    letter is re-derived from the option whose text matches the requested value,
    and a question that dropped the requested value is rejected outright.
    """

    raw_options = question.get("options")
    if not isinstance(raw_options, dict) or len(raw_options) < 2:
        raise CosmosEvaluatorError(f"generated question for {variable!r} has no usable options")
    letters = {
        str(key).strip().upper(): str(text).strip()
        for key, text in raw_options.items()
        if str(key).strip().upper() in ANSWER_LETTERS and str(text).strip()
    }
    if len(letters) < 2:
        raise CosmosEvaluatorError(f"generated question for {variable!r} has fewer than two lettered options")

    wanted = str(value).strip().casefold()
    matches = [letter for letter, text in letters.items() if text.strip().casefold() == wanted]
    if not matches:
        raise CosmosEvaluatorError(
            f"generated question for {variable!r} omits the requested value {value!r}"
        )
    text = str(question.get("question") or "").strip()
    if not text:
        raise CosmosEvaluatorError(f"generated question for {variable!r} has no question text")
    return {
        "variable": variable,
        "value": value,
        "question": text,
        "options": letters,
        "correct_answer": sorted(matches)[0],
        "all_options": list(options),
    }


def parse_answer_letter(text: str, *, offered: Iterable[str] | None = None) -> str:
    """Upstream's answer parsing: a standalone A-D, else ``UNKNOWN``.

    ``offered`` restricts the answer to the letters the question actually listed.
    Upstream always scans for A-D, so a three-option question answered "D" is read
    as a concrete wrong answer; that is indistinguishable from a model that
    ignored the options, and a live Token Factory VLM does return it. Rejecting an
    unoffered letter as ``UNKNOWN`` keeps a malformed answer from reading as
    evidence about the pixels.
    """

    if not text:
        return "UNKNOWN"
    allowed = {str(letter).strip().upper() for letter in offered} if offered else set(ANSWER_LETTERS)
    allowed &= set(ANSWER_LETTERS)
    if not allowed:
        return "UNKNOWN"
    upper = text.upper()
    for match in re.finditer(r"\b([A-D])\b", upper):
        if match.group(1) in allowed:
            return match.group(1)
    stripped = upper.strip()
    return stripped if stripped in allowed else "UNKNOWN"


def _default_client() -> Any:
    from npa.clients.token_factory import TokenFactoryClient, TokenFactoryError

    try:
        return TokenFactoryClient()
    except TokenFactoryError as exc:
        raise CosmosEvaluatorError(str(exc)) from exc


class _FrameImage:
    """Yield a still or a beginning/middle/end contact sheet for the clip."""

    def __init__(
        self,
        *,
        video: str | Path | None,
        frame: str | Path | None,
        sample_policy: str = "ranking",
    ) -> None:
        self._video = Path(video) if video is not None else None
        self._frame = Path(frame) if frame is not None else None
        self._sample_policy = sample_policy
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self._frame is not None:
            if not self._frame.is_file():
                raise CosmosEvaluatorError(f"frame not found: {self._frame}")
            return self._frame
        assert self._video is not None
        if not self._video.is_file():
            raise CosmosEvaluatorError(f"video not found: {self._video}")
        self._tmp = tempfile.TemporaryDirectory(prefix="npa-cosmos-eval-")
        try:
            out = Path(self._tmp.name) / "representative_frames.jpg"
            _write_representative_contact_sheet(
                self._video, out, sample_policy=self._sample_policy
            )
        except BaseException:
            self.__exit__()
            raise
        return out

    def __exit__(self, *_exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def _frame_image(
    *,
    video: str | Path | None,
    frame: str | Path | None,
    sample_policy: str = "ranking",
) -> _FrameImage:
    return _FrameImage(
        video=video, frame=frame, sample_policy=sample_policy
    )


def _write_representative_contact_sheet(
    video: Path, output: Path, *, sample_policy: str = "ranking"
) -> None:
    """Decode deterministic ranking or disjoint holdout frames for verification."""

    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise CosmosEvaluatorError(
            "representative video verification requires PyAV and Pillow"
        ) from exc

    try:
        with av.open(str(video)) as container:
            frame_count = sum(1 for _frame in container.decode(video=0))
        if frame_count < 1:
            raise CosmosEvaluatorError("the augmented video decoded zero frames")
        targets = _representative_frame_targets(frame_count, sample_policy)
        frames: list[Image.Image] = []
        with av.open(str(video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in targets:
                    frames.append(frame.to_image().convert("RGB"))
    except CosmosEvaluatorError:
        raise
    except Exception as exc:  # noqa: BLE001 - sanitized evaluator boundary
        raise CosmosEvaluatorError(
            "could not decode representative frames from the augmented video"
        ) from exc
    if len(frames) != len(targets):
        raise CosmosEvaluatorError(
            "the augmented video changed frame count while representative frames were decoded"
        )

    normalized: list[Image.Image] = []
    for frame in frames:
        resized = frame.copy()
        resized.thumbnail((384, 384), Image.Resampling.LANCZOS)
        normalized.append(resized)
    width = sum(frame.width for frame in normalized)
    height = max(frame.height for frame in normalized)
    sheet = Image.new("RGB", (width, height), color=(18, 18, 18))
    offset = 0
    for frame in normalized:
        sheet.paste(frame, (offset, (height - frame.height) // 2))
        offset += frame.width
    sheet.save(output, format="JPEG", quality=95, subsampling=0)


def _representative_frame_targets(
    frame_count: int, sample_policy: str
) -> set[int]:
    """Return deterministic ranking or strictly disjoint holdout frame indices."""

    if frame_count < 1:
        raise CosmosEvaluatorError("the augmented video decoded zero frames")
    ranking_targets = {0, (frame_count - 1) // 2, frame_count - 1}
    if sample_policy == "ranking":
        return ranking_targets
    if sample_policy != "holdout":
        raise CosmosEvaluatorError("sample_policy must be ranking or holdout")
    available = [
        index for index in range(frame_count) if index not in ranking_targets
    ]
    if not available:
        raise CosmosEvaluatorError(
            "holdout verification needs a frame not used by ranking"
        )
    target_count = min(3, len(available))
    return {
        available[min(len(available) - 1, int(i * len(available) / target_count))]
        for i in range(target_count)
    }
