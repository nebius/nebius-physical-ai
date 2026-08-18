"""Grounded GPU allocation fallback routes for the shipped agent backend."""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Callable

try:
    from agent_backend import gpu_allocation_fallback as fallback
except ImportError:  # repository tests
    from npa.agent_backend import gpu_allocation_fallback as fallback


@dataclass
class GpuAllocationDeps:
    mutate_state: Callable[[Callable[[dict], Any]], Any]
    action_digest: Callable[[Any], str]


def register_gpu_allocation_routes(
    app: Any, deps: GpuAllocationDeps, http_error: Any
) -> None:
    """Register zero-token attempt and consent routes with injected backend state."""

    @app.post("/agent/gpu-allocation/attempt")
    def gpu_allocation_attempt(payload: dict) -> dict:
        body = payload if isinstance(payload, dict) else {}
        logical = str(body.get("logical_allocation") or "").strip()
        request = body.get("request") if isinstance(body.get("request"), dict) else {}
        if not logical or not request:
            raise http_error(
                status_code=400, detail="logical_allocation and request are required"
            )
        logical_ref = fallback.logical_allocation_ref(logical)
        failure = body.get("failure") if isinstance(body.get("failure"), dict) else {}

        def mutate(state: dict) -> dict:
            records = state.get("gpu_allocation_fallback")
            records = records if isinstance(records, dict) else {}
            current = records.get(logical_ref)
            advanced, decision = fallback.record_attempt(
                current if isinstance(current, dict) else None,
                logical_allocation=logical,
                request=request,
                failure_code=str(failure.get("code") or ""),
                failure_message=str(failure.get("message") or ""),
                evidence=(
                    body.get("evidence")
                    if isinstance(body.get("evidence"), dict)
                    else None
                ),
                candidate=(
                    body.get("preemptible_candidate")
                    if isinstance(body.get("preemptible_candidate"), dict)
                    else None
                ),
                success=bool(body.get("success")),
            )
            confirm_token = ""
            if decision.get("prompt"):
                proposed = (
                    decision.get("proposed_action")
                    if isinstance(decision.get("proposed_action"), dict)
                    else {}
                )
                unsigned = {
                    key: value for key, value in proposed.items() if key != "digest"
                }
                digest = deps.action_digest(unsigned)
                proposed["digest"] = digest
                decision["proposed_action"] = proposed
                advanced["pending_action_digest"] = digest
                confirm_token = secrets.token_hex(8)
                act_state = state.get("agent_act")
                act_state = act_state if isinstance(act_state, dict) else {}
                act_state["confirm_token"] = confirm_token
                act_state["confirm_digest"] = digest
                act_state["pending_action"] = proposed
                state["agent_act"] = act_state
            records[logical_ref] = advanced
            state["gpu_allocation_fallback"] = records
            response = {
                "ok": True,
                "grounded": True,
                "usage": {"total_tokens": 0},
                "decision": decision,
                "allocation": fallback.public_state(advanced),
            }
            if confirm_token:
                response["needs_confirmation"] = True
                response["confirm_token"] = confirm_token
            return response

        return deps.mutate_state(mutate)

    @app.post("/agent/gpu-allocation/consent")
    def gpu_allocation_consent(payload: dict) -> dict:
        body = payload if isinstance(payload, dict) else {}
        logical = str(body.get("logical_allocation") or "").strip()
        if not logical or not isinstance(body.get("accept"), bool):
            raise http_error(
                status_code=400,
                detail="logical_allocation and boolean accept are required",
            )
        accepted = bool(body["accept"])
        logical_ref = fallback.logical_allocation_ref(logical)

        def mutate(state: dict) -> dict:
            records = state.get("gpu_allocation_fallback")
            records = records if isinstance(records, dict) else {}
            current = records.get(logical_ref)
            if not isinstance(current, dict):
                raise http_error(
                    status_code=409, detail="no tracked GPU allocation fallback"
                )
            pending_action_digest = str(current.get("pending_action_digest") or "")
            if not pending_action_digest:
                raise http_error(
                    status_code=409,
                    detail="no GPU allocation fallback is awaiting consent",
                )
            confirmed_digest = ""
            if accepted:
                supplied = str(body.get("confirm_token") or "").strip()
                act_state = state.get("agent_act")
                act_state = act_state if isinstance(act_state, dict) else {}
                session_token = str(act_state.get("confirm_token") or "")
                session_digest = str(act_state.get("confirm_digest") or "")
                pending = act_state.get("pending_action")
                pending = pending if isinstance(pending, dict) else {}
                pending_digest = str(pending.get("digest") or "")
                unsigned = {
                    key: value for key, value in pending.items() if key != "digest"
                }
                if (
                    not supplied
                    or supplied != session_token
                    or not session_digest
                    or session_digest != pending_digest
                    or session_digest != pending_action_digest
                    or str(pending.get("logical_allocation_ref") or "") != logical_ref
                    or deps.action_digest(unsigned) != session_digest
                ):
                    raise http_error(
                        status_code=403,
                        detail="invalid or expired confirmation for GPU pool switch",
                    )
                confirmed_digest = session_digest
                act_state["confirm_token"] = ""
                act_state["confirm_digest"] = ""
                act_state["pending_action"] = None
                state["agent_act"] = act_state
            try:
                advanced = fallback.record_consent(
                    current,
                    accepted=accepted,
                    confirmed_action_digest=confirmed_digest,
                )
            except ValueError as exc:
                raise http_error(status_code=409, detail=str(exc)) from exc
            records[logical_ref] = advanced
            state["gpu_allocation_fallback"] = records
            return {
                "ok": True,
                "grounded": True,
                "usage": {"total_tokens": 0},
                "allocation": fallback.public_state(advanced),
            }

        return deps.mutate_state(mutate)
