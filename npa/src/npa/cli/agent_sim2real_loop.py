"""Autonomous Sim2Real outer-loop orchestration for the NPA agent backend.

Where ``agent_actions`` provides a generic bounded tool loop, this module drives
the *specific* sim-to-real outer loop: launch sim -> run eval -> read gate
metrics -> diagnose failure mode -> adjust config -> re-run. It mirrors the
staged engine's Stage-11 threshold gate (``promote_checkpoint`` vs
``loop_back``; see ``npa/src/npa/workflows/sim2real/engine.py``) without
re-implementing the engine.

Cost/safety contract (see ``docs/architecture/agent-competitive-plan.md``):

- Every GPU-spending step (``launch``) passes through the confirmation gate. The
  loop *proposes* the drive; the operator confirms with a token before any run
  starts. A model turn can never auto-launch GPU work.
- A stage/iteration is only marked complete when the injected ``status``
  callable (the real ``workflows/sim2real/status`` / ``runs/{run_id}`` surface)
  confirms it. No fabricated run data.

All collaborators are injected callables so the orchestration unit-tests with
zero GPU / infra / model access. The VM backend wires the real engine + status
APIs; tests inject deterministic fakes. The module is embedded verbatim into the
agent VM backend by ``agent.py`` (same mechanism as the other agent modules).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

# Normalized outer-loop decisions (mirror engine threshold_decision output).
DECISION_PROMOTE = "promote_checkpoint"
DECISION_LOOP_BACK = "loop_back"

# Terminal reasons for the drive.
STOP_PROMOTED = "promoted"
STOP_EXHAUSTED = "iterations_exhausted"
STOP_NEEDS_CONFIRMATION = "needs_confirmation"
STOP_UNCONFIRMED_STATUS = "status_unconfirmed"
STOP_INSUFFICIENT_SIGNAL = "insufficient_gate_signal"
STOP_NO_ADJUSTMENT = "no_config_adjustment"
STOP_ERROR = "error"

DEFAULT_MAX_ITERATIONS = 3


def drive_action_digest(action: Any) -> str:
    """Stable short digest binding a confirmation token to a specific drive config."""
    try:
        payload = json.dumps(action or {}, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = str(action)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def evaluate_gate(gate_result: Any) -> dict[str, Any]:
    """Derive a normalized promote/loop-back decision from gate metrics.

    Mirrors the engine's Stage-11 rule ``success_rate >= threshold ->
    promote_checkpoint``. Accepts either an explicit ``decision`` field or a
    ``success_rate``/``threshold`` pair. Returns ``{decision, success_rate,
    threshold, promoted}``.
    """
    data = gate_result if isinstance(gate_result, dict) else {}
    success_rate = data.get("success_rate")
    threshold = data.get("threshold")
    try:
        success_rate = float(success_rate)
    except (TypeError, ValueError):
        success_rate = None
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = None

    explicit = str(data.get("decision") or "").strip()
    has_signal = bool(explicit) or (success_rate is not None and threshold is not None)
    if explicit.startswith("promote"):
        promoted = True
    elif explicit.startswith("loop_back"):
        promoted = False
    elif success_rate is not None and threshold is not None:
        promoted = success_rate >= threshold
    else:
        promoted = False
    return {
        "decision": DECISION_PROMOTE if promoted else DECISION_LOOP_BACK,
        "success_rate": success_rate,
        "threshold": threshold,
        "promoted": promoted,
        "has_signal": has_signal,
    }


def _status_confirms_run(status: Any, run_id: str) -> bool:
    """A run is confirmed only when the authoritative status echoes its run_id.

    Guards against fabricating progress: if the status surface does not report a
    matching, non-idle run, we do not mark the iteration complete.
    """
    if not isinstance(status, dict):
        return False
    if not status.get("ok", True):
        return False
    candidates: list[str] = []
    sim_viz = status.get("sim_viz")
    if isinstance(sim_viz, dict):
        candidates.append(str(sim_viz.get("run_id") or ""))
    run = status.get("run")
    if isinstance(run, dict):
        candidates.append(str(run.get("run_id") or ""))
    candidates.append(str(status.get("run_id") or ""))
    latest = status.get("latest_submit")
    if isinstance(latest, dict):
        candidates.append(str(latest.get("run_id") or ""))
    target = str(run_id or "").strip()
    if not target:
        return any(c.strip() for c in candidates)
    return any(c.strip() == target for c in candidates)


def drive_sim2real_loop(
    goal: str,
    *,
    config: dict[str, Any],
    launch: Callable[[dict[str, Any]], Any],
    status: Callable[[str], Any],
    gate: Callable[[str, int], Any],
    diagnose: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
    adjust: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
    confirm_token: str = "",
    session_token: str = "",
    confirm_digest: str = "",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    require_gate_signal: bool = True,
    confirmation_ok: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Drive the Sim2Real outer loop with confirmation gates and real status.

    Returns a trace dict::

        {ok, goal, iterations: [...], decision, final_run_id, stopped_reason,
         needs_confirmation, proposed_action, reply}

    Each iteration records the launch, the confirmed status, the gate metrics,
    the promote/loop-back decision + reason, and (on loop-back) the diagnosis and
    config adjustment.
    """
    cfg = dict(config) if isinstance(config, dict) else {}
    gate_ok = confirmation_ok
    if gate_ok is None:
        def gate_ok(token: str, expected: str) -> bool:  # noqa: ANN001
            token = str(token or "").strip()
            expected = str(expected or "").strip()
            return bool(token) and bool(expected) and token == expected

    try:
        max_iters = max(1, int(max_iterations))
    except (TypeError, ValueError):
        max_iters = DEFAULT_MAX_ITERATIONS

    # GPU-spending gate: driving the loop launches real runs. Require an explicit
    # confirmation token bound to this exact drive config before any launch.
    proposed = {"action": "drive_sim2real", "config": cfg}
    digest = drive_action_digest(proposed)
    token_ok = gate_ok(confirm_token, session_token)
    digest_ok = (not confirm_digest) or confirm_digest == digest
    if not (token_ok and digest_ok):
        proposed_with_digest = dict(proposed)
        proposed_with_digest["digest"] = digest
        return {
            "ok": True,
            "goal": str(goal or ""),
            "iterations": [],
            "decision": None,
            "final_run_id": "",
            "stopped_reason": STOP_NEEDS_CONFIRMATION,
            "needs_confirmation": True,
            "proposed_action": proposed_with_digest,
            "reply": (
                "Driving the Sim2Real loop launches GPU runs and needs explicit "
                "confirmation. Re-send with the confirmation token issued for this "
                "exact drive config to start."
            ),
        }

    iterations: list[dict[str, Any]] = []
    decision_value: str | None = None
    final_run_id = ""
    stopped_reason = STOP_EXHAUSTED

    for iteration in range(1, max_iters + 1):
        record: dict[str, Any] = {"iteration": iteration}
        try:
            launched = launch(cfg)
        except Exception as exc:  # noqa: BLE001 - surface launch failure
            record["status"] = "error"
            record["error"] = f"launch failed: {exc}"
            iterations.append(record)
            stopped_reason = STOP_ERROR
            break
        launched_dict = launched if isinstance(launched, dict) else {}
        run_id = str(launched_dict.get("run_id") or cfg.get("run_id") or "").strip()
        record["launch"] = launched_dict
        record["run_id"] = run_id

        # Only mark progress when the authoritative status confirms the run.
        try:
            run_status = status(run_id)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = f"status read failed: {exc}"
            iterations.append(record)
            stopped_reason = STOP_ERROR
            break
        confirmed = _status_confirms_run(run_status, run_id)
        record["status_confirmed"] = confirmed
        record["status"] = run_status if isinstance(run_status, dict) else {}
        if not confirmed:
            record["reason"] = "status did not confirm the launched run"
            iterations.append(record)
            stopped_reason = STOP_UNCONFIRMED_STATUS
            break

        final_run_id = run_id or final_run_id

        try:
            gate_result = gate(run_id, iteration)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"gate read failed: {exc}"
            iterations.append(record)
            stopped_reason = STOP_ERROR
            break
        evaluation = evaluate_gate(gate_result)
        record["gate"] = gate_result if isinstance(gate_result, dict) else {}
        record["decision"] = evaluation["decision"]
        sr = evaluation["success_rate"]
        th = evaluation["threshold"]
        decision_value = evaluation["decision"]

        if evaluation["promoted"]:
            record["reason"] = (
                f"success_rate={sr} >= threshold={th}: promote checkpoint"
            )
            iterations.append(record)
            stopped_reason = STOP_PROMOTED
            break

        # No real gate metrics yet: do NOT loop back and relaunch another GPU run
        # on a fabricated/absent signal. Stop and report insufficient signal.
        if require_gate_signal and not evaluation.get("has_signal"):
            record["reason"] = "no gate success_rate available — stopping (no fabricated loop-back)"
            iterations.append(record)
            stopped_reason = STOP_INSUFFICIENT_SIGNAL
            break

        record["reason"] = (
            f"success_rate={sr} < threshold={th}: loop back and adjust"
        )
        # Diagnose the failure mode and adjust config for the next iteration.
        diagnosis: dict[str, Any] = {}
        if diagnose is not None:
            try:
                raw = diagnose(record["gate"], record["status"])
                diagnosis = raw if isinstance(raw, dict) else {"notes": str(raw)}
            except Exception as exc:  # noqa: BLE001
                diagnosis = {"error": f"diagnose failed: {exc}"}
        record["diagnosis"] = diagnosis
        if iteration >= max_iters:
            iterations.append(record)
            continue
        # Never relaunch identical GPU work: only continue when an adjustment
        # actually changed the config, otherwise a re-run cannot improve the gate.
        if adjust is None:
            record["reason"] = record["reason"] + " (no adjuster wired — stopping)"
            iterations.append(record)
            stopped_reason = STOP_NO_ADJUSTMENT
            break
        try:
            new_cfg = adjust(cfg, diagnosis)
        except Exception as exc:  # noqa: BLE001
            record["adjust_error"] = str(exc)
            iterations.append(record)
            stopped_reason = STOP_ERROR
            break
        if not isinstance(new_cfg, dict) or new_cfg == cfg:
            record["reason"] = record["reason"] + " (adjustment produced no change — stopping)"
            iterations.append(record)
            stopped_reason = STOP_NO_ADJUSTMENT
            break
        cfg = new_cfg
        record["adjusted_config"] = dict(new_cfg)
        iterations.append(record)

    ok = stopped_reason in {
        STOP_PROMOTED,
        STOP_EXHAUSTED,
        STOP_NEEDS_CONFIRMATION,
        STOP_INSUFFICIENT_SIGNAL,
        STOP_NO_ADJUSTMENT,
    }
    reply = _summarize(goal, iterations, decision_value, stopped_reason, final_run_id)
    return {
        "ok": ok,
        "goal": str(goal or ""),
        "iterations": iterations,
        "decision": decision_value,
        "final_run_id": final_run_id,
        "stopped_reason": stopped_reason,
        "needs_confirmation": False,
        "proposed_action": None,
        "reply": reply,
    }


def _summarize(
    goal: str,
    iterations: list[dict[str, Any]],
    decision: str | None,
    stopped_reason: str,
    final_run_id: str,
) -> str:
    lines = ["**Autonomous Sim2Real drive** (grounded on live status):"]
    lines.append(f"- **iterations_run**: `{len(iterations)}`")
    if final_run_id:
        lines.append(f"- **final_run_id**: `{final_run_id}`")
    if decision:
        lines.append(f"- **final_decision**: `{decision}`")
    lines.append(f"- **stopped_reason**: `{stopped_reason}`")
    for record in iterations:
        it = record.get("iteration")
        dec = record.get("decision") or record.get("reason") or record.get("status")
        lines.append(f"  - iteration `{it}`: `{dec}`")
    if stopped_reason == STOP_PROMOTED:
        lines.append("- Checkpoint promoted — gate threshold met on a confirmed run.")
    elif stopped_reason == STOP_UNCONFIRMED_STATUS:
        lines.append("- Stopped: live status did not confirm the launched run (no fabricated progress).")
    elif stopped_reason == STOP_INSUFFICIENT_SIGNAL:
        lines.append("- Stopped: no gate success_rate on the confirmed run yet (no fabricated loop-back / relaunch).")
    elif stopped_reason == STOP_EXHAUSTED:
        lines.append("- Exhausted outer iterations without meeting the gate threshold.")
    return "\n".join(lines)
