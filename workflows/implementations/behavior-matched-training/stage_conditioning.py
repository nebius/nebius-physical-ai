"""Validate and apply frozen parent stage-controller replay traces."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

TASK_NUM_STAGES = {0: 5, 1: 6, 22: 9}
TRACE_SCHEMA = "npa.behavior.rlc-stage-replay.v2"


@dataclasses.dataclass(frozen=True, order=True)
class FrameKey:
    """Identify one replan frame within a released episode."""

    task_id: int
    episode_index: int
    episode_relative_frame: int


@dataclasses.dataclass(frozen=True)
class ReplayRow:
    """Hold the frozen controller state and its provenance for one frame."""

    key: FrameKey
    timestamp: float
    teacher_stage: int
    replay_stage: int
    post_update_stage: int
    raw_argmax: int
    raw_logits: tuple[float, ...]
    served_argmax: int
    served_logits: tuple[float, ...]
    history_before: tuple[int, ...]
    history_after: tuple[int, ...]
    task0_reset_applied: bool
    stage_count: int
    observation_sha256: str
    action_sha256: str


@dataclasses.dataclass
class FilterState:
    """Represent the native three-vote stage filter state."""

    stage: int = 0
    history: deque[int] = dataclasses.field(default_factory=lambda: deque(maxlen=3))


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: The file cannot be opened or read.
    """

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal_time_stage(frame: int, episode_length: int, task_id: int) -> int:
    """Compute the pinned equal-duration stage label.

    Args:
        frame: Episode-relative frame index.
        episode_length: Stored frame count for the episode.
        task_id: Supported BEHAVIOR task ID.

    Returns:
        Zero-based stage bin.

    Raises:
        ValueError: Inputs do not identify a valid frame and task.
    """

    if task_id not in TASK_NUM_STAGES or episode_length <= 0 or frame < 0:
        raise ValueError("invalid task, episode length, or frame")
    stages = TASK_NUM_STAGES[task_id]
    return min(int(frame / (episode_length / stages)), stages - 1)


def update_filter(
    state: FilterState, raw_prediction: int, task_id: int
) -> tuple[FilterState, bool]:
    """Apply the native correction and hysteresis ordering.

    Args:
        state: Controller state before the action call.
        raw_prediction: Valid-task stage argmax.
        task_id: Supported BEHAVIOR task ID.

    Returns:
        Updated state and whether the radio reset fired.

    Raises:
        ValueError: The task is unsupported.
    """

    if task_id not in TASK_NUM_STAGES:
        raise ValueError(f"unsupported task: {task_id}")
    stage = state.stage
    history = deque(state.history, maxlen=3)
    reset = task_id == 0 and stage == 4
    if reset:
        stage = 2
        history.clear()
    history.append(int(raw_prediction))
    maximum = TASK_NUM_STAGES[task_id] - 1
    if len(history) == 3 and stage + 1 <= maximum:
        votes = Counter(history)
        if votes[stage + 1] >= 2 or votes[stage + 2] == 3:
            stage += 1
            history.clear()
        elif stage > 0 and votes[stage - 1] == 3:
            stage -= 1
            history.clear()
    stage = min(max(stage, 0), TASK_NUM_STAGES[task_id] - 1)
    return FilterState(stage=stage, history=history), reset


def replay_episode(
    *,
    task_id: int,
    episode_index: int,
    episode_length: int,
    logits: Sequence[Sequence[float]],
    served_logits: Sequence[Sequence[float]],
    sample_digests: Sequence[tuple[str, str]],
    stride: int = 20,
) -> list[ReplayRow]:
    """Replay an episode using each pre-update stage as its condition.

    Args:
        task_id: Supported BEHAVIOR task ID.
        episode_index: Released episode identifier.
        episode_length: Stored frame count.
        logits: Auxiliary raw 15-way logits at each replan frame.
        served_logits: Canonical batch-one valid-stage logits at each frame.
        sample_digests: Observation and action hashes per frame.
        stride: Replan cadence in frames.

    Returns:
        Chronological replay rows.

    Raises:
        ValueError: Counts, logits, or task metadata are invalid.
    """

    frames = list(range(0, episode_length, stride))
    if (
        len(frames) != len(logits)
        or len(frames) != len(served_logits)
        or len(frames) != len(sample_digests)
    ):
        raise ValueError(
            "logits and sample digests are required for every replan frame"
        )
    state = FilterState()
    rows: list[ReplayRow] = []
    for frame, scores, served, digests in zip(
        frames, logits, served_logits, sample_digests, strict=True
    ):
        scores = tuple(float(score) for score in scores)
        if len(scores) != 15 or not np.isfinite(scores).all():
            raise ValueError("stage logits must contain 15 finite values")
        served = tuple(float(score) for score in served)
        if len(served) != TASK_NUM_STAGES[task_id] or not np.isfinite(served).all():
            raise ValueError("served logits must contain every valid task stage")
        row, state = _replay_frame(
            task_id,
            episode_index,
            episode_length,
            frame,
            scores,
            served,
            digests,
            state,
        )
        rows.append(row)
    return rows


def _replay_frame(
    task_id: int,
    episode_index: int,
    episode_length: int,
    frame: int,
    scores: tuple[float, ...],
    served: tuple[float, ...],
    digests: tuple[str, str],
    state: FilterState,
) -> tuple[ReplayRow, FilterState]:
    before = tuple(state.history)
    condition = state.stage
    raw_argmax = int(np.argmax(scores))
    prediction = int(np.argmax(served))
    updated, reset = update_filter(state, prediction, task_id)
    row = ReplayRow(
        key=FrameKey(task_id, episode_index, frame),
        timestamp=frame / 30.0,
        teacher_stage=equal_time_stage(frame, episode_length, task_id),
        replay_stage=condition,
        post_update_stage=updated.stage,
        raw_argmax=raw_argmax,
        raw_logits=scores,
        served_argmax=prediction,
        served_logits=served,
        history_before=before,
        history_after=tuple(updated.history),
        task0_reset_applied=reset,
        stage_count=TASK_NUM_STAGES[task_id],
        observation_sha256=digests[0],
        action_sha256=digests[1],
    )
    return row, updated


def write_trace(
    path: Path,
    rows: Iterable[ReplayRow],
    *,
    split: str,
    identities: Mapping[str, str],
) -> dict[str, object]:
    """Write a canonical replay trace and its identity receipt.

    Args:
        path: Destination JSONL path.
        rows: Chronological replay records.
        split: ``training`` or ``holdout``.
        identities: Frozen source identities.

    Returns:
        Trace path, row count, digest, split, and ordered-key digest.

    Raises:
        OSError: The trace cannot be written.
        ValueError: The split or rows violate the trace contract.
    """

    if split not in {"training", "holdout"}:
        raise ValueError("trace split must be training or holdout")
    rows = sorted(rows, key=lambda row: row.key)
    _validate_rows(rows)
    key_sha256 = _ordered_key_sha256(row.key for row in rows)
    header = {
        "schema": TRACE_SCHEMA,
        "split": split,
        "identities": dict(sorted(identities.items())),
        "ordered_key_sha256": key_sha256,
    }
    lines = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    for row in rows:
        payload = dataclasses.asdict(row)
        payload["key"] = dataclasses.asdict(row.key)
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n")
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256(path),
        "split": split,
        "ordered_key_sha256": key_sha256,
    }


def load_trace(
    path: Path,
    *,
    split: str,
    identities: Mapping[str, str],
) -> dict[FrameKey, ReplayRow]:
    """Load a replay trace after checking split and source identities.

    Args:
        path: Source JSONL path.
        split: Expected dataset split.
        identities: Expected frozen source identities.

    Returns:
        Replay rows keyed by immutable frame identity.

    Raises:
        OSError: The trace cannot be read.
        ValueError: The header, rows, or ordered-key digest differ.
    """

    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records:
        raise ValueError("replay trace is empty")
    header = records[0]
    if {key: header.get(key) for key in ("identities", "schema", "split")} != {
        "identities": dict(sorted(identities.items())),
        "schema": TRACE_SCHEMA,
        "split": split,
    }:
        raise ValueError("replay trace header differs from the frozen contract")
    rows = []
    for payload in records[1:]:
        key = FrameKey(**payload.pop("key"))
        for field in (
            "raw_logits",
            "served_logits",
            "history_before",
            "history_after",
        ):
            payload[field] = tuple(payload[field])
        rows.append(ReplayRow(key=key, **payload))
    _validate_rows(rows)
    if header.get("ordered_key_sha256") != _ordered_key_sha256(row.key for row in rows):
        raise ValueError("replay trace ordered key identity differs")
    return {row.key: row for row in rows}


def _ordered_key_sha256(keys: Iterable[FrameKey]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(
            f"{key.task_id}:{key.episode_index}:{key.episode_relative_frame}\n".encode()
        )
    return digest.hexdigest()


def _validate_rows(rows: Sequence[ReplayRow]) -> None:
    seen: set[FrameKey] = set()
    previous: FrameKey | None = None
    for row in rows:
        key = row.key
        if key in seen:
            raise ValueError(f"duplicate replay key: {key}")
        seen.add(key)
        if key.task_id not in TASK_NUM_STAGES or key.episode_relative_frame < 0:
            raise ValueError(f"invalid replay key: {key}")
        if key.episode_relative_frame % 20:
            raise ValueError(f"replay key is off the 20-frame cadence: {key}")
        stages = TASK_NUM_STAGES[key.task_id]
        values = (row.teacher_stage, row.replay_stage, row.post_update_stage)
        if any(not 0 <= value < stages for value in values):
            raise ValueError(f"out-of-range stage in replay row: {key}")
        if not 0 <= row.raw_argmax < 15:
            raise ValueError(f"out-of-range raw argmax in replay row: {key}")
        if len(row.raw_logits) != 15 or not np.isfinite(row.raw_logits).all():
            raise ValueError(f"invalid logits in replay row: {key}")
        if (
            len(row.served_logits) != stages
            or not np.isfinite(row.served_logits).all()
            or row.served_argmax != int(np.argmax(row.served_logits))
        ):
            raise ValueError(f"invalid canonical served logits in replay row: {key}")
        if (
            row.stage_count != stages
            or abs(row.timestamp - key.episode_relative_frame / 30.0) > 5e-4
        ):
            raise ValueError(f"stage count or timestamp differs in replay row: {key}")
        for digest in (row.observation_sha256, row.action_sha256):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"invalid sample identity digest in replay row: {key}")
        if previous is not None and key <= previous:
            raise ValueError("replay rows must be in canonical key order")
        previous = key


@dataclasses.dataclass(frozen=True)
class ApplyStageCondition:
    """Verify the native teacher label, then install the selected action condition."""

    arm: str

    def __call__(self, data: dict) -> dict:
        if self.arm not in {"teacher", "replay"}:
            raise ValueError(f"unknown stage arm: {self.arm}")
        prompt = np.asarray(data["tokenized_prompt"]).copy()
        mask = np.asarray(data["tokenized_prompt_mask"])
        if mask.dtype != np.bool_ or mask.shape != prompt.shape or not mask.all():
            raise ValueError("native prompt mask differs from the frozen contract")
        task_id = int(prompt[0])
        if task_id not in TASK_NUM_STAGES:
            raise ValueError("native prompt contains an unsupported task ID")
        teacher = int(data["teacher_stage"])
        replay = int(data["replay_stage"])
        if (
            not 0 <= teacher < TASK_NUM_STAGES[task_id]
            or not 0 <= replay < TASK_NUM_STAGES[task_id]
        ):
            raise ValueError("stage label is outside the task stage range")
        if prompt.shape == (1,):
            prompt = np.asarray([int(prompt[0]), teacher], dtype=np.int32)
        elif prompt.shape != (2,) or int(prompt[1]) != teacher:
            raise ValueError("native equal-time stage differs from frozen trace")
        condition = teacher if self.arm == "teacher" else replay
        prompt[1] = condition
        return {
            **data,
            "tokenized_prompt": prompt,
            "tokenized_prompt_mask": np.ones(2, dtype=bool),
        }
