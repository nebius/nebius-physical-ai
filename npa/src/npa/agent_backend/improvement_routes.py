"""Zero-token improvement APIs and non-disruptive action hooks."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence
import uuid
import re
import sqlite3

try:
    from agent_backend.improvements import ImprovementError, ImprovementStore, lesson_context, store_from_config
    from agent_backend.trajectory import current_episode_id, current_session_id
except ImportError:
    from npa.agent_backend.improvements import ImprovementError, ImprovementStore, lesson_context, store_from_config
    from npa.agent_backend.trajectory import current_episode_id, current_session_id


LOGGER = logging.getLogger(__name__)


@dataclass
class ImprovementDeps:
    store: Callable[[], ImprovementStore | None]


class ImprovementRuntime:
    """Best-effort feedback never repeats or invalidates an already finished tool.

    Opt in with an owner-only NPA_AGENT_IMPROVEMENT_CONFIG file. Configuration
    errors produce a visible pending status; absence is explicitly disabled.
    """

    def __init__(self, store: Callable[[], ImprovementStore | None] | None = None):
        self._injected_store = store

    def store(self) -> ImprovementStore | None:
        if self._injected_store:
            return self._injected_store()
        path = os.environ.get("NPA_AGENT_IMPROVEMENT_CONFIG", "")
        if not path:
            return None
        # Reload the small protected configuration each time. Stat-only caches
        # miss same-size rewrites, permission changes and reviewer revocation;
        # redaction configuration may also change independently of this file.
        return store_from_config(Path(path))

    def prepare(self, targets: Sequence[str]) -> dict:
        request_id = current_episode_id() or uuid.uuid4().hex
        try:
            store = self.store()
            if store is None:
                return {"status": "disabled", "request_id": request_id, "lessons": [], "context": ""}
            lessons = store.consume_lessons(targets, request_id=request_id)
            return {"status": "ready", "request_id": request_id, "lessons": lessons, "context": lesson_context(lessons)}
        except Exception:
            return {"status": "pending", "request_id": request_id, "lessons": [], "context": ""}

    def context(self, targets: Sequence[str]) -> str:
        try:
            store = self.store()
            return lesson_context(store.matching_verified_lessons(targets)) if store else ""
        except Exception:
            return ""

    def targets(self, skills: Sequence[str], user_text: str) -> list[str]:
        """Match explicitly named configured components, without a model call."""
        targets = list(skills)
        try:
            store = self.store()
            if store:
                for component in store.scopes:
                    if re.search(r"(?<!\w)" + re.escape(component) + r"(?!\w)", user_text, re.IGNORECASE):
                        targets.append(component)
        except Exception:
            LOGGER.debug("Improvement target matching unavailable; retaining selected skills")
        return list(dict.fromkeys(targets))

    def record(self, result: dict, prepared: dict | None = None) -> dict:
        prepared = prepared or {"request_id": current_episode_id() or uuid.uuid4().hex, "lessons": []}
        request_id = prepared["request_id"]
        try:
            store = self.store()
            if store is None:
                return {"status": "disabled"}
            items = store.observe_action(result, episode_id=current_episode_id() or request_id,
                                         session_id=current_session_id())
            outcome = "confirmation" if result.get("needs_confirmation") else (
                "succeeded" if result.get("ok") is True else "failed" if result.get("ok") is False else "unknown"
            )
            store.record_lesson_outcome(prepared["lessons"], request_id=request_id, outcome=outcome)
            return {"status": "recorded", "item_ids": [item["id"] for item in items],
                    "lesson_item_ids": [lesson["item_id"] for lesson in prepared["lessons"]]}
        except Exception:
            # A pending receipt is visible to the caller. It does not assert that
            # queue persistence succeeded, and never re-executes the action.
            return {"status": "pending", "reason": "feedback_storage_or_evidence_unavailable"}


def register_improvement_routes(app: Any, deps: ImprovementDeps, http_error: Any) -> None:
    """Mutation accepts protected receipt references, never pass/fail assertions.

    Owner names are coordination labels. A claim's opaque lease and generation
    fence worker writes. Independent review is supplied only by a configured
    local evidence adapter, outside the shared agent HTTP identity.
    """

    def invoke(operation: Callable[[ImprovementStore], Any]) -> dict:
        try:
            store = deps.store()
            if store is None:
                return {"ok": True, "status": "disabled", "grounded": True, "usage": {"total_tokens": 0}}
            result = operation(store)
            return {"ok": True, "status": "ready", "grounded": True, "usage": {"total_tokens": 0}, "result": result}
        except sqlite3.Error:
            raise http_error(status_code=503, detail="improvement storage unavailable") from None
        except (ImprovementError, KeyError, TypeError, ValueError, OSError):
            # File locations, report contents and exception messages stay private.
            raise http_error(status_code=409, detail="improvement scope, ownership or evidence check failed") from None

    def ownership(payload: dict) -> dict:
        return {name: payload[name] for name in ("owner", "generation", "claim_token")}

    @app.get("/agent/improvements")
    def list_improvements() -> dict:
        return invoke(lambda store: store.list_items())

    @app.get("/agent/improvements/lessons")
    def list_lessons(target: str) -> dict:
        return invoke(lambda store: store.matching_verified_lessons([target]))

    @app.get("/agent/improvements/{item_id}")
    def improvement_history(item_id: str) -> dict:
        return invoke(lambda store: store.history(item_id))

    @app.post("/agent/improvements/reconcile")
    def reconcile_improvements(payload: dict) -> dict:
        return invoke(lambda store: store.observe_action(payload["result"], episode_id=payload["episode_id"]))

    @app.post("/agent/improvements/{item_id}/claim")
    def claim_improvement(item_id: str, payload: dict) -> dict:
        return invoke(lambda store: store.claim(item_id, owner=payload["owner"], version=payload["version"]))

    @app.post("/agent/improvements/{item_id}/release")
    def release_improvement(item_id: str, payload: dict) -> dict:
        return invoke(lambda store: store.release(item_id, **ownership(payload)))

    @app.post("/agent/improvements/{item_id}/validation")
    def validate_improvement(item_id: str, payload: dict) -> dict:
        return invoke(lambda store: store.record_validation(item_id, evidence_ref=payload["evidence_ref"], **ownership(payload)))

    @app.post("/agent/improvements/{item_id}/review")
    def review_improvement(item_id: str, payload: dict) -> dict:
        return invoke(lambda store: store.review(item_id, evidence_ref=payload["evidence_ref"]))
