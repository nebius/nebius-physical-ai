"""Grounded GPU allocation fallback routes for the shipped agent backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from agent_backend import gpu_allocation_fallback as fallback
except ImportError:  # repository tests
    from npa.agent_backend import gpu_allocation_fallback as fallback


@dataclass
class GpuAllocationDeps:
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    issue_confirmation: Callable[[dict, str], str]
    peek_confirmation: Callable[[], tuple[str, str, dict | None]]
    consume_confirmation: Callable[[], tuple[str, str, dict | None]]
    action_digest: Callable[[Any], str]


def register_gpu_allocation_routes(app: Any, deps: GpuAllocationDeps, http_error: Any) -> None:
    """Register zero-token attempt and consent routes with injected backend state."""

    @app.post("/agent/gpu-allocation/attempt")
    def gpu_allocation_attempt(payload: dict) -> dict:
        body = payload if isinstance(payload, dict) else {}
        logical = str(body.get("logical_allocation") or "").strip()
        request = body.get("request") if isinstance(body.get("request"), dict) else {}
        if not logical or not request:
            raise http_error(status_code=400, detail="logical_allocation and request are required")
        state = deps.load_state()
        records = state.get("gpu_allocation_fallback")
        records = records if isinstance(records, dict) else {}
        logical_ref = fallback.logical_allocation_ref(logical)
        current = records.get(logical_ref)
        failure = body.get("failure") if isinstance(body.get("failure"), dict) else {}
        advanced, decision = fallback.record_attempt(
            current if isinstance(current, dict) else None,
            logical_allocation=logical,
            request=request,
            failure_code=str(failure.get("code") or ""),
            failure_message=str(failure.get("message") or ""),
            evidence=body.get("evidence") if isinstance(body.get("evidence"), dict) else None,
            candidate=(
                body.get("preemptible_candidate")
                if isinstance(body.get("preemptible_candidate"), dict)
                else None
            ),
            success=bool(body.get("success")),
        )
        if decision.get("prompt"):
            proposed = (
                decision.get("proposed_action")
                if isinstance(decision.get("proposed_action"), dict)
                else {}
            )
            unsigned = {key: value for key, value in proposed.items() if key != "digest"}
            digest = deps.action_digest(unsigned)
            proposed["digest"] = digest
            decision["proposed_action"] = proposed
            advanced["pending_action_digest"] = digest
        records[logical_ref] = advanced
        state["gpu_allocation_fallback"] = records
        deps.save_state(state)
        response = {
            "ok": True,
            "grounded": True,
            "usage": {"total_tokens": 0},
            "decision": decision,
            "allocation": fallback.public_state(advanced),
        }
        if decision.get("prompt"):
            proposed = decision["proposed_action"]
            response["needs_confirmation"] = True
            response["confirm_token"] = deps.issue_confirmation(proposed, proposed["digest"])
        return response

    @app.post("/agent/gpu-allocation/consent")
    def gpu_allocation_consent(payload: dict) -> dict:
        body = payload if isinstance(payload, dict) else {}
        logical = str(body.get("logical_allocation") or "").strip()
        if not logical or not isinstance(body.get("accept"), bool):
            raise http_error(status_code=400, detail="logical_allocation and boolean accept are required")
        accepted = bool(body["accept"])
        state = deps.load_state()
        records = state.get("gpu_allocation_fallback")
        records = records if isinstance(records, dict) else {}
        logical_ref = fallback.logical_allocation_ref(logical)
        current = records.get(logical_ref)
        if not isinstance(current, dict):
            raise http_error(status_code=409, detail="no tracked GPU allocation fallback")
        pending_action_digest = str(current.get("pending_action_digest") or "")
        if not pending_action_digest:
            raise http_error(status_code=409, detail="no GPU allocation fallback is awaiting consent")
        confirmed_digest = ""
        if accepted:
            supplied = str(body.get("confirm_token") or "").strip()
            session_token, session_digest, pending = deps.peek_confirmation()
            pending = pending if isinstance(pending, dict) else {}
            pending_digest = str(pending.get("digest") or "")
            unsigned = {key: value for key, value in pending.items() if key != "digest"}
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
            consumed_token, consumed_digest, consumed_pending = deps.consume_confirmation()
            if (
                consumed_token != session_token
                or consumed_digest != session_digest
                or consumed_pending != pending
            ):
                raise http_error(
                    status_code=403,
                    detail="invalid or expired confirmation for GPU pool switch",
                )
            confirmed_digest = consumed_digest
            # ``consume_confirmation`` persists the cleared single-use gate.
            # Continue from that fresh state so saving the allocation cannot
            # accidentally restore the consumed token from our earlier snapshot.
            state = deps.load_state()
            records = state.get("gpu_allocation_fallback")
            records = records if isinstance(records, dict) else {}
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
        deps.save_state(state)
        return {
            "ok": True,
            "grounded": True,
            "usage": {"total_tokens": 0},
            "allocation": fallback.public_state(advanced),
        }
