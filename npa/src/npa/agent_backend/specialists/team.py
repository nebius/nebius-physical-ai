"""Coordinate durable specialist tasks, optional Jev selection and isolated workers."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import re
import sqlite3
import uuid

from npa.agent_backend.model_router import classify_generation_model
from npa.agent_backend.trajectory import redact
from npa.clients.credentials import load_credentials

from .config import TeamConfig, fingerprint
from .graph import build_graph, initial_state
from .observations import _dismiss
from .store import TaskStore, UncertainOperation, _private_file
from .storage_errors import StorageFailure
from .tools import WorkbenchTools


class SpecialistTeam:
    """Manage one host's specialist tasks through the same CLI, SDK and HTTP behavior.

    Args: config: Validated team policy. clients: Optional profile/client mapping for tests.
    Returns: Team coordinator; workers run separately or through work_once.
    Raises: ValueError, OSError: Runtime storage or configuration is invalid.
    """

    def __init__(self, config: TeamConfig, *, clients=None):
        self.config = config
        self.store = TaskStore(config.state_directory)
        self.clients = clients or {}

    def submit(
        self,
        goal: str,
        *,
        specialist: str = "auto",
        task_id: str = "",
        parent_id: str = "",
    ):
        """Enqueue an idempotent task with a fixed profile and durable routing receipt.

        Args: goal: Operator request. specialist: Explicit profile or auto. task_id: Retry identity.
            parent_id: Completed task whose answer supplies follow-up context.
        Returns: Persisted task record; duplicate requests reuse the original assignment.
        Raises: ValueError, KeyError: Request, parent or profile is invalid.
        """
        identity = task_id or str(uuid.uuid4())
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identity)
            or not goal.strip()
        ):
            raise ValueError("task id and nonempty goal are required")
        request = hashlib.sha256(
            json.dumps([goal, specialist, parent_id]).encode()
        ).hexdigest()
        try:
            previous = self.store._get(identity)
        except KeyError:
            previous = None
        if previous:
            if previous["request"] != request:
                raise ValueError("task id already belongs to a different request")
            return previous
        if parent_id:
            goal = self._followup(parent_id, goal)
        profile, route = self._route(goal, specialist)
        return self.store._submit(
            identity, profile.name, fingerprint(profile), request, redact(goal), route
        )

    def status(self, task_id: str = ""):
        """Read task results and observed worker heartbeats without credential values.

        Args: task_id: Exact task, or empty for the team overview.
        Returns: Task with events, or profiles and all tasks.
        Raises: KeyError: The requested task is unknown.
        """
        if task_id:
            task = self.store._get(task_id)
            return {
                **task,
                "events": self.store._events(task_id),
                "calls": self.store._calls(task_id),
            }
        paused, workers = self.store._paused_profiles(), self.store._workers()
        profiles = [
            {
                "name": item.name,
                "model": item.model,
                "description": item.description,
                "paused": item.name in paused,
                "worker": workers.get(item.name),
            }
            for item in self.config.profiles
        ]
        return {"profiles": profiles, "tasks": self.store._list()}

    def pause(self, *, task_id: str = "", specialist: str = "", paused: bool = True):
        """Pause or resume at the next model/tool boundary without interrupting effects.

        Args: task_id: Task to control. specialist: Alternatively control a whole profile.
            paused: True pauses; false resumes.
        Returns: Current controlled object status.
        Raises: ValueError, KeyError: Target is invalid or needs reconciliation.
        """
        if bool(task_id) == bool(specialist):
            raise ValueError("select exactly one task or specialist")
        if task_id:
            return self.store._pause_task(task_id, paused)
        self.config.profile(specialist)
        self.store._pause_profile(specialist, paused)
        return {"specialist": specialist, "paused": paused}

    def reconcile(
        self,
        task_id: str,
        *,
        call_id: str = "",
        result: dict | None = None,
        retry: bool = False,
    ):
        """Record an operator-verified effect result or authorize retry after inspection.

        Args: task_id: Blocked task. call_id: Interrupted effect, when present.
            result: Verified result to reuse. retry: Explicit authorization to retry.
        Returns: Requeued task; the model and grants remain unchanged.
        Raises: ValueError, KeyError: Reconciliation is incomplete or inappropriate.
        """
        if result is None and not retry:
            raise ValueError("supply a verified result or explicitly authorize retry")
        self.store._reconcile(task_id, call_id, redact(result), retry)
        return self.store._get(task_id)

    def patch(self, task_id: str) -> str:
        """Return a task's recorded source edits for human review.

        Args: task_id: Existing task identity.
        Returns: Unified diff, empty when no file was edited.
        Raises: KeyError, OSError: Task is unknown or artifact cannot be read.
        """
        self.store._get(task_id)
        path = self.store.directory / "artifacts" / task_id / "changes.diff"
        return path.read_text() if path.exists() else ""

    def dismiss_interrupted_observation(self, task_id: str, call_id: str):
        """Record a lost, explicitly classified observation as failed without replay.

        Args: task_id: Task awaiting attention. call_id: Its interrupted observation.
        Returns: Failed observation receipt; the task remains needs_attention.
        Raises: ValueError, KeyError: Policy, task or call is ineligible.
            BlockingIOError, StorageFailure: Ownership or durable storage is unavailable.
        """
        profile = self.config.profile(self.store._get(task_id)["profile"])
        with self._ownership(profile.name):
            return _dismiss(self.store, profile, task_id, call_id)

    def cancel(self, task_id: str):
        """Stop a task after its current node without undoing external effects.

        Args: task_id: Existing task to cancel.
        Returns: Cancelled task with existing tool receipts preserved.
        Raises: KeyError, ValueError: Task is unknown or already completed.
        """
        if self.store._get(task_id)["status"] == "completed":
            raise ValueError("completed tasks cannot be cancelled")
        self.store._update(task_id, "cancelled")
        return self.store._get(task_id)

    def work_once(self, specialist: str):
        """Advance one durable node while holding exclusive host-local profile ownership.

        Args: specialist: Exact configured profile.
        Returns: Updated task, or None when the queue is idle or paused.
        Raises: BlockingIOError: Another worker owns this profile.
            ValueError, OSError: Profile or durable storage is invalid.
        """
        profile = self.config.profile(specialist)
        with self._ownership(profile.name):
            self.store._heartbeat(profile.name)
            task = self.store._next(profile.name)
            if task is None:
                return None
            if task["policy"] != fingerprint(profile):
                self.store._update(
                    task["id"],
                    "needs_attention",
                    error="Profile changed; restore its original policy before resuming",
                )
                return self.store._get(task["id"])
            return self._advance(profile, task)

    def _advance(self, profile, task):
        self.store._update(task["id"], "running")
        if self.store._get(task["id"])["status"] == "cancelled":
            return self.store._get(task["id"])
        try:
            with self._graph(profile, task["id"]) as graph:
                options = {"configurable": {"thread_id": task["id"]}}
                snapshot = graph.get_state(options)
                if snapshot.values and not snapshot.next:
                    return self._complete(task["id"], snapshot.values)
                state = (
                    None if snapshot.values else initial_state(profile, task["goal"])
                )
                graph.invoke(state, options, durability="sync")
                snapshot = graph.get_state(options)
                if not snapshot.next:
                    return self._complete(task["id"], snapshot.values)
        except (
            RuntimeError,
            ValueError,
            OSError,
            KeyError,
            TypeError,
            IndexError,
        ) as error:
            self._attention(task["id"], error)
        return self.store._get(task["id"])

    def _attention(self, task_id, error):
        # Never include raw provider, filesystem or SQLite messages.
        detail = (
            redact(str(error))
            if isinstance(error, (ValueError, UncertainOperation, StorageFailure))
            else type(error).__name__
        )
        event = {"type": "needs_attention", "error": type(error).__name__}
        if isinstance(error, StorageFailure):
            event["storage_failure"] = error.diagnostic
        self.store._update(task_id, "needs_attention", error=detail)
        self.store._event(task_id, event)

    def _complete(self, task_id, values):
        if self.store._originals(task_id):
            profile = self.config.profile(self.store._get(task_id)["profile"])
            WorkbenchTools(profile, self.store, task_id)._save_patch()
        self.store._update(task_id, "completed", result=values.get("answer", ""))
        return self.store._get(task_id)

    @contextmanager
    def _graph(self, profile, task_id):
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = self.store.directory / (task_id + ".checkpoints.sqlite")
        _private_file(path)
        connection = None
        try:
            connection = sqlite3.connect(path, check_same_thread=False)
            saver = SqliteSaver(
                connection, serde=JsonPlusSerializer(pickle_fallback=False)
            )
            tools = WorkbenchTools(profile, self.store, task_id)
            yield build_graph(
                profile, tools, saver, client=self.clients.get(profile.name)
            )
        except sqlite3.Error as error:
            raise StorageFailure("graph_checkpoint", error) from None
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def _ownership(self, profile):
        import fcntl

        path = self.store.directory / (profile + ".lock")
        _private_file(path)
        with path.open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _route(self, goal, specialist):
        if specialist != "auto":
            return self.config.profile(specialist), {
                "status": "explicit",
                "specialist": specialist,
            }
        fallback = self.config.default_profile
        decision = {"status": "default", "specialist": fallback}
        if self.config.router == "jev":
            key = os.environ.get(
                "TYPESAFE_API_KEY", ""
            ) or load_credentials().tokens.get("TYPESAFE_API_KEY", "")
            candidates = {item.name: item.description for item in self.config.profiles}
            decision = classify_generation_model(redact(goal), candidates, api_key=key)
            fallback = decision.get("selected_model") or fallback
            decision = {**decision, "specialist": fallback}
        return self.config.profile(fallback), decision

    def _followup(self, parent_id, goal):
        parent = self.store._get(parent_id)
        if parent["status"] != "completed":
            raise ValueError("follow-ups require a completed parent task")
        return f"Previous task {parent_id}:\n{parent['result']}\n\nFollow-up:\n{goal}"
