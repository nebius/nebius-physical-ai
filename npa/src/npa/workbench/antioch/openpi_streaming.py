"""Bounded soft-real-time orchestration for continuous OpenPI control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

from .openpi_bridge import OpenPIBridgeError, safe_position_targets, validate_actions

_T = TypeVar("_T")


class StreamingPolicyTransport(Protocol):
    @property
    def generation(self) -> int: ...

    @property
    def reconnect_count(self) -> int: ...

    def ping(self) -> None: ...

    def infer(self, observation: Mapping[str, object]) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class StreamingConfig:
    observation_hz: float = 10.0
    policy_request_hz: float = 2.0
    control_hz: float = 10.0
    executed_targets_per_chunk: int = 5
    maximum_observation_age_seconds: float = 0.75
    maximum_response_age_seconds: float = 1.5
    inference_deadline_seconds: float = 10.0
    ping_interval_seconds: float = 5.0
    reconnect_initial_backoff_seconds: float = 0.5
    reconnect_maximum_backoff_seconds: float = 8.0
    safe_hold_behavior: str = "hold-current"
    minimum_ready_cycles: int = 3
    minimum_ready_seconds: float = 5.0
    maximum_joint_delta_rad: float = 0.08

    def __post_init__(self) -> None:
        positive = {
            "observation_hz": self.observation_hz,
            "policy_request_hz": self.policy_request_hz,
            "control_hz": self.control_hz,
            "maximum_observation_age_seconds": self.maximum_observation_age_seconds,
            "maximum_response_age_seconds": self.maximum_response_age_seconds,
            "inference_deadline_seconds": self.inference_deadline_seconds,
            "ping_interval_seconds": self.ping_interval_seconds,
            "reconnect_initial_backoff_seconds": self.reconnect_initial_backoff_seconds,
            "reconnect_maximum_backoff_seconds": self.reconnect_maximum_backoff_seconds,
            "minimum_ready_seconds": self.minimum_ready_seconds,
            "maximum_joint_delta_rad": self.maximum_joint_delta_rad,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError(
                "streaming rates, ages, deadlines, and limits must be positive"
            )
        if not 1 <= self.executed_targets_per_chunk <= 15:
            raise ValueError("executed targets per chunk must be between 1 and 15")
        if self.minimum_ready_cycles < 1:
            raise ValueError("minimum ready cycles must be positive")
        if self.safe_hold_behavior not in {"hold-current", "no-action"}:
            raise ValueError("safe hold behavior must be hold-current or no-action")


@dataclass(frozen=True)
class ObservationEnvelope:
    sequence: int
    epoch: int
    monotonic_seconds: float
    value: Mapping[str, object]


@dataclass(frozen=True)
class ResponseEnvelope:
    observation_sequence: int
    epoch: int
    observation_monotonic_seconds: float
    completed_monotonic_seconds: float
    actions: np.ndarray


@dataclass(frozen=True)
class ControlDecision:
    target: np.ndarray | None
    source: str
    epoch: int
    observation_sequence: int | None = None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[max(index, 0)])


class StreamingPolicyLoop:
    """Keep one request in flight and expose bounded receding-horizon decisions."""

    def __init__(
        self,
        transport: StreamingPolicyTransport,
        *,
        config: StreamingConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.config = config or StreamingConfig()
        self._clock = clock
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_observation: ObservationEnvelope | None = None
        self._responses: deque[ResponseEnvelope] = deque(maxlen=1)
        self._actions: deque[np.ndarray] = deque(
            maxlen=self.config.executed_targets_per_chunk
        )
        self._active_sequence: int | None = None
        self._active_epoch = 0
        self._epoch = 0
        self._sequence = 0
        self._first_observation_at: float | None = None
        self._last_observation_at: float | None = None
        self._next_request_at = 0.0
        self._last_ping_at = float("-inf")
        self._failure_streak = 0
        self._started_at = self._clock()
        self._underrun_active = False
        self._metrics: dict[str, int] = {
            "observations": 0,
            "dropped_observations": 0,
            "policy_requests": 0,
            "policy_round_trips": 0,
            "reconnects": 0,
            "rejected_responses": 0,
            "stale_responses": 0,
            "malformed_or_unsafe_responses": 0,
            "inference_timeouts": 0,
            "transport_failures": 0,
            "response_queue_replacements": 0,
            "chunks_superseded": 0,
            "chunk_underruns": 0,
            "safe_holds": 0,
            "control_steps": 0,
            "safely_applied_targets": 0,
            "render_ticks": 0,
        }
        self._inference_latencies: deque[float] = deque(maxlen=2048)
        self._observation_ages: deque[float] = deque(maxlen=2048)
        self._response_ages: deque[float] = deque(maxlen=2048)

    @property
    def epoch(self) -> int:
        with self._condition:
            return self._epoch

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker, name="openpi-policy-stream", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self.transport.close()
        if self._thread is not None:
            self._thread.join(timeout=self.config.inference_deadline_seconds + 1.0)
            if self._thread.is_alive():
                raise OpenPIBridgeError(
                    "policy worker did not stop within the inference deadline"
                )

    def record_render_tick(self) -> None:
        with self._condition:
            self._metrics["render_ticks"] += 1

    def publish_observation(
        self,
        value: Mapping[str, object],
        *,
        monotonic_seconds: float | None = None,
    ) -> int:
        now = self._clock() if monotonic_seconds is None else monotonic_seconds
        with self._condition:
            self._sequence += 1
            if self._latest_observation is not None:
                self._metrics["dropped_observations"] += 1
            self._latest_observation = ObservationEnvelope(
                sequence=self._sequence,
                epoch=self._epoch,
                monotonic_seconds=now,
                value=value,
            )
            if self._first_observation_at is None:
                self._first_observation_at = now
            self._last_observation_at = now
            self._metrics["observations"] += 1
            self._condition.notify_all()
            return self._sequence

    def reset_epoch(self, *, reconnects: int = 0) -> None:
        with self._condition:
            self._epoch += 1
            self._latest_observation = None
            self._responses.clear()
            self._metrics["reconnects"] += max(reconnects, 1)
            self._condition.notify_all()

    def policy_cycle(self, *, monotonic_seconds: float | None = None) -> bool:
        now = self._clock() if monotonic_seconds is None else monotonic_seconds
        with self._condition:
            if self._latest_observation is None or now < self._next_request_at:
                return False
            observation = self._latest_observation
            self._latest_observation = None
            self._next_request_at = now + 1.0 / self.config.policy_request_hz
            observation_age = now - observation.monotonic_seconds
            self._observation_ages.append(max(observation_age, 0.0))
            if observation_age > self.config.maximum_observation_age_seconds:
                self._metrics["rejected_responses"] += 1
                self._metrics["stale_responses"] += 1
                return False
            self._metrics["policy_requests"] += 1
            epoch = self._epoch
        reconnects_before = self.transport.reconnect_count
        started = self._clock()
        try:
            if started - self._last_ping_at >= self.config.ping_interval_seconds:
                self.transport.ping()
                self._last_ping_at = started
            generation_before = self.transport.generation
            raw_actions = self.transport.infer(observation.value)
        except Exception:
            completed = self._clock()
            with self._condition:
                self._inference_latencies.append(completed - started)
                self._metrics["rejected_responses"] += 1
                self._metrics["transport_failures"] += 1
                if completed - started >= self.config.inference_deadline_seconds:
                    self._metrics["inference_timeouts"] += 1
                backoff = min(
                    self.config.reconnect_maximum_backoff_seconds,
                    self.config.reconnect_initial_backoff_seconds
                    * 2**self._failure_streak,
                )
                self._failure_streak += 1
                self._next_request_at = max(self._next_request_at, completed + backoff)
            reconnect_delta = self.transport.reconnect_count - reconnects_before
            self.reset_epoch(reconnects=max(reconnect_delta, 1))
            return False
        completed = self._clock()
        self._failure_streak = 0
        latency = completed - started
        response_age = completed - observation.monotonic_seconds
        generation_changed = self.transport.generation != generation_before
        try:
            actions = validate_actions({"actions": raw_actions})
        except OpenPIBridgeError:
            with self._condition:
                self._inference_latencies.append(latency)
                self._metrics["rejected_responses"] += 1
                self._metrics["malformed_or_unsafe_responses"] += 1
            return False
        with self._condition:
            self._inference_latencies.append(latency)
            if (
                generation_changed
                or epoch != self._epoch
                or observation.epoch != self._epoch
            ):
                self._metrics["rejected_responses"] += 1
                reconnect_delta = self.transport.reconnect_count - reconnects_before
                self._epoch += 1
                self._latest_observation = None
                self._responses.clear()
                self._metrics["reconnects"] += max(reconnect_delta, 1)
                return False
            if (
                latency > self.config.inference_deadline_seconds
                or response_age > self.config.maximum_response_age_seconds
            ):
                self._metrics["rejected_responses"] += 1
                self._metrics["stale_responses"] += 1
                return False
            response = ResponseEnvelope(
                observation_sequence=observation.sequence,
                epoch=epoch,
                observation_monotonic_seconds=observation.monotonic_seconds,
                completed_monotonic_seconds=completed,
                actions=actions,
            )
            if self._responses:
                self._metrics["response_queue_replacements"] += 1
                self._responses.clear()
            self._responses.append(response)
            self._response_ages.append(response_age)
            self._metrics["policy_round_trips"] += 1
        return True

    def _worker(self) -> None:
        while not self._stop.is_set():
            if self.policy_cycle():
                continue
            with self._condition:
                if self._stop.is_set():
                    return
                timeout = (
                    0.25
                    if self._latest_observation is None
                    else max(0.001, self._next_request_at - self._clock())
                )
                self._condition.wait(timeout=min(timeout, 0.25))

    def next_control_decision(
        self,
        current_joints: np.ndarray,
        current_gripper: float,
        *,
        monotonic_seconds: float | None = None,
    ) -> ControlDecision:
        now = self._clock() if monotonic_seconds is None else monotonic_seconds
        joints = np.asarray(current_joints, dtype=np.float64)
        with self._condition:
            self._metrics["control_steps"] += 1
            if self._active_epoch != self._epoch:
                self._actions.clear()
                self._active_sequence = None
                self._active_epoch = self._epoch
            response = self._responses.pop() if self._responses else None
            self._responses.clear()
        if response is not None:
            age = now - response.observation_monotonic_seconds
            if (
                response.epoch != self.epoch
                or age > self.config.maximum_response_age_seconds
            ):
                with self._condition:
                    self._metrics["rejected_responses"] += 1
                    self._metrics["stale_responses"] += 1
            else:
                try:
                    targets = safe_position_targets(
                        response.actions,
                        joints,
                        max_joint_delta_rad=self.config.maximum_joint_delta_rad,
                        execute_steps=self.config.executed_targets_per_chunk,
                    )
                except OpenPIBridgeError:
                    with self._condition:
                        self._metrics["rejected_responses"] += 1
                        self._metrics["malformed_or_unsafe_responses"] += 1
                    self._actions.clear()
                else:
                    if self._actions:
                        with self._condition:
                            self._metrics["chunks_superseded"] += 1
                    self._actions = deque(
                        np.array(target, copy=True) for target in targets
                    )
                    self._active_sequence = response.observation_sequence
                    self._active_epoch = response.epoch
                    self._underrun_active = False
        if self._actions:
            return ControlDecision(
                target=self._actions.popleft(),
                source="policy",
                epoch=self._active_epoch,
                observation_sequence=self._active_sequence,
            )
        with self._condition:
            if not self._underrun_active:
                self._metrics["chunk_underruns"] += 1
                self._underrun_active = True
            self._metrics["safe_holds"] += 1
        if self.config.safe_hold_behavior == "hold-current":
            target = np.concatenate(
                [joints, np.asarray([np.clip(current_gripper, 0.0, 1.0)])]
            )
            return ControlDecision(target=target, source="hold", epoch=self.epoch)
        return ControlDecision(target=None, source="no-action", epoch=self.epoch)

    def mark_applied(self, decision: ControlDecision) -> None:
        if decision.source != "policy":
            return
        if decision.epoch != self.epoch:
            raise OpenPIBridgeError(
                "refusing to count a target from a stale control epoch"
            )
        with self._condition:
            self._metrics["safely_applied_targets"] += 1

    def apply_if_current(
        self, decision: ControlDecision, apply: Callable[[np.ndarray], _T]
    ) -> _T:
        """Atomically reject epoch changes before a policy target reaches physics."""

        if decision.source != "policy" or decision.target is None:
            raise OpenPIBridgeError("only a policy target can use guarded application")
        with self._condition:
            if decision.epoch != self._epoch:
                self._metrics["rejected_responses"] += 1
                self._metrics["stale_responses"] += 1
                raise OpenPIBridgeError(
                    "refusing to apply a target from a stale control epoch"
                )
            result = apply(decision.target)
            self._metrics["safely_applied_targets"] += 1
            return result

    def pending_counts(self) -> dict[str, int]:
        with self._condition:
            return {
                "observations": int(self._latest_observation is not None),
                "responses": len(self._responses),
                "actions": len(self._actions),
            }

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def metrics_snapshot(
        self, *, monotonic_seconds: float | None = None
    ) -> dict[str, object]:
        now = self._clock() if monotonic_seconds is None else monotonic_seconds
        with self._condition:
            elapsed = max(now - self._started_at, 1e-9)
            metrics: dict[str, object] = dict(self._metrics)
            metrics.update(
                {
                    "elapsed_seconds": elapsed,
                    "observation_fps": self._metrics["observations"] / elapsed,
                    "control_step_fps": self._metrics["control_steps"] / elapsed,
                    "render_fps": self._metrics["render_ticks"] / elapsed,
                    "inference_latency_ms_p50": 1000
                    * _percentile(self._inference_latencies, 0.5),
                    "inference_latency_ms_p95": 1000
                    * _percentile(self._inference_latencies, 0.95),
                    "observation_age_ms_p50": 1000
                    * _percentile(self._observation_ages, 0.5),
                    "observation_age_ms_p95": 1000
                    * _percentile(self._observation_ages, 0.95),
                    "response_age_ms_p50": 1000 * _percentile(self._response_ages, 0.5),
                    "response_age_ms_p95": 1000
                    * _percentile(self._response_ages, 0.95),
                    "metric_sample_window": {
                        "capacity": 2048,
                        "inference": len(self._inference_latencies),
                        "observation_age": len(self._observation_ages),
                        "response_age": len(self._response_ages),
                    },
                    "epoch": self._epoch,
                    "latest_observation_sequence": self._sequence,
                    "camera_frame_span_seconds": (
                        0.0
                        if self._first_observation_at is None
                        or self._last_observation_at is None
                        else max(
                            self._last_observation_at - self._first_observation_at,
                            0.0,
                        )
                    ),
                    "ready": (
                        elapsed >= self.config.minimum_ready_seconds
                        and self._metrics["policy_round_trips"]
                        >= self.config.minimum_ready_cycles
                        and self._metrics["safely_applied_targets"]
                        >= self.config.minimum_ready_cycles
                        and self._sequence >= self.config.minimum_ready_cycles
                        and self._first_observation_at is not None
                        and self._last_observation_at is not None
                        and self._last_observation_at > self._first_observation_at
                    ),
                }
            )
            return metrics
