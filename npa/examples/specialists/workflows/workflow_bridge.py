"""Give an Astra coordinator scoped Workbench tools and optional durable delegation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time
import uuid

from npa.agent_backend.specialists.config import fingerprint, load_config
from npa.agent_backend.specialists.observations import _dismiss
from npa.agent_backend.specialists.store import TaskStore, UncertainOperation
from npa.agent_backend.specialists.storage_errors import StorageFailure
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools

TERMINAL = {"completed", "needs_attention", "cancelled"}


def _receipt_event(event, include_read_content):
    if include_read_content or event.get("type") != "tool":
        return event
    if event.get("name") != "read_file":
        return event
    result = event.get("result", {})
    content = result.get("content")
    if not isinstance(content, str):
        return event
    summary = {key: value for key, value in result.items() if key != "content"}
    summary.update(content_omitted=True, content_bytes=len(content.encode("utf-8")))
    return {**event, "result": summary}


def _coordinator_store(team, directory):
    store = TaskStore(Path(directory) / "coordinator-state")
    for profile in team.config.profiles:
        identity = "coordinator-" + profile.name
        store._submit(
            identity, profile.name, fingerprint(profile), "coordinator", "", {}
        )
        store._update(identity, "completed")
    store._submit("supervisor", "supervisor", "", "coordinator", "", {})
    store._update("supervisor", "completed")
    return store


class _Bridge:
    def __init__(self, config_path, directory, hybrid):
        self.team = SpecialistTeam(load_config(config_path))
        self.store = _coordinator_store(self.team, directory)
        self.hybrid = hybrid

    async def _recorded(self, name, arguments, action):
        try:
            return await self._journaled(name, arguments, action)
        except StorageFailure as error:
            return {
                "ok": False,
                "error": str(error),
                "error_type": "StorageFailure",
                "storage_failure": error.diagnostic,
                "receipt_persisted": False,
                "effect_outcome": "unknown; inspect original receipts before recovery",
            }

    async def _journaled(self, name, arguments, action):
        identity = str(uuid.uuid4())
        invocation = {"name": name, "arguments": arguments}
        self.store._begin_call("supervisor", identity, invocation)
        self.store._event(
            "supervisor", {"type": "tool_started", "call_id": identity, **invocation}
        )
        try:
            result = await action()
        except (
            ValueError,
            KeyError,
            BlockingIOError,
            UncertainOperation,
            StorageFailure,
        ) as error:
            result = {
                "ok": False,
                "error": str(error),
                "error_type": type(error).__name__,
            }
            if isinstance(error, StorageFailure):
                result.update(
                    storage_failure=error.diagnostic, operation_receipt_persisted=False
                )
        self.store._finish_call("supervisor", identity, result)
        self.store._event(
            "supervisor",
            {"type": "tool", "call_id": identity, "name": name, "result": result},
        )
        return result

    def _require_idle(self, specialist):
        tasks = [
            task for task in self.team.store._list() if task["profile"] == specialist
        ]
        if any(task["status"] not in {"completed", "cancelled"} for task in tasks):
            raise ValueError("specialist owns this workspace; inspect its status first")
        self._require_resolved_effects(specialist)

    def _require_resolved_effects(self, specialist):
        tasks = [
            task for task in self.team.store._list() if task["profile"] == specialist
        ]
        calls = self.store._calls("coordinator-" + specialist)
        calls.extend(
            call for task in tasks for call in self.team.store._calls(task["id"])
        )
        if any(call["status"] != "completed" for call in calls):
            raise UncertainOperation(
                "workspace has an uncertain tool effect; reconcile externally"
            )

    def _invoke(self, specialist, name, arguments):
        profile = self.team.config.profile(specialist)
        with self.team._ownership(specialist):
            self._require_access(profile, name, arguments)
            executor = WorkbenchTools(profile, self.store, "coordinator-" + specialist)
            call = {
                "id": str(uuid.uuid4()),
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            return executor.execute(call)

    def _require_access(self, profile, name, arguments):
        operation = (
            profile.operations.get(arguments.get("name"))
            if name == "run_operation"
            else None
        )
        if operation is None or not operation.observation_only:
            return self._require_idle(profile.name)
        tasks = [
            task for task in self.team.store._list() if task["profile"] == profile.name
        ]
        if any(task["status"] in {"queued", "running"} for task in tasks):
            raise ValueError("specialist owns this workspace; inspect its status first")
        if any(task["policy"] != fingerprint(profile) for task in tasks):
            raise ValueError("task policy changed; restore its original policy")
        if self.store._get("coordinator-" + profile.name)["policy"] != fingerprint(
            profile
        ):
            raise ValueError("coordinator policy changed; restore its original policy")

    def _dismiss_observation(self, task_id, call_id):
        if not self.hybrid:
            raise ValueError("observation dismissal is unavailable in astra-only")
        profile = self.team.config.profile(self.team.store._get(task_id)["profile"])
        with self.team._ownership(profile.name):
            if any(
                call["status"] != "completed"
                for call in self.store._calls("coordinator-" + profile.name)
            ):
                raise UncertainOperation(
                    "coordinator has an uncertain effect; reconcile externally"
                )
            return _dismiss(self.team.store, profile, task_id, call_id)

    async def _direct(self, specialist, name, arguments):
        return await self._recorded(
            name,
            {"specialist": specialist, **arguments},
            lambda: asyncio.to_thread(self._invoke, specialist, name, arguments),
        )

    async def _batch(self, specialists, name):
        if len(specialists) != len(set(specialists)):
            raise ValueError("each specialist must appear once")
        results = await asyncio.gather(
            *(
                self._direct(specialist, "run_operation", {"name": name})
                for specialist in specialists
            )
        )
        return dict(zip(specialists, results))

    def _delegate(self, specialist, task_id, goal):
        if not self.hybrid:
            raise ValueError("delegation is unavailable in astra-only")
        self.team.config.profile(specialist)
        with self.team._ownership(specialist):
            existing = {task["id"] for task in self.team.store._list()}
            if task_id not in existing:
                self._require_idle(specialist)
            task = self.team.submit(goal, specialist=specialist, task_id=task_id)
            return {
                "task_id": task["id"],
                "specialist": task["profile"],
                "status": task["status"],
            }

    def _take_over(self, task_id):
        if not self.hybrid:
            raise ValueError("takeover is unavailable in astra-only")
        specialist = self.team.store._get(task_id)["profile"]
        with self.team._ownership(specialist):
            task = self.team.store._get(task_id)
            if task["status"] != "needs_attention":
                raise ValueError(
                    "takeover requires a task awaiting attention at a durable boundary"
                )
            self._require_resolved_effects(specialist)
            self.team.store._event(
                task_id,
                {
                    "type": "coordinator_takeover_requested",
                    "previous_status": task["status"],
                    "previous_error": task["error"],
                    "remote_workloads_cancelled": False,
                },
            )
            cancelled = self.team.cancel(task_id)
            return {
                "task_id": task_id,
                "specialist": specialist,
                "status": cancelled["status"],
                "remote_workloads_cancelled": False,
            }

    def _status(self, task_id, after_sequence=0, include_read_content=False):
        if after_sequence < 0:
            raise ValueError("after_sequence must be nonnegative")
        task = self.team.status(task_id)
        events = task["events"]
        return {
            "task_id": task["id"],
            "specialist": task["profile"],
            "status": task["status"],
            "paused": task["paused"],
            "result": task["result"],
            "error": task["error"],
            "last_sequence": events[-1]["sequence"] if events else 0,
            "events": [
                _receipt_event(event, include_read_content)
                for event in events
                if event["sequence"] > after_sequence
            ],
            "uncertain_calls": [
                call for call in task["calls"] if call["status"] != "completed"
            ],
        }

    async def _wait(self, task_id, after_sequence, observation_seconds):
        if not 0 < observation_seconds <= 60:
            raise ValueError(
                "observation_seconds must be greater than zero and at most 60"
            )
        deadline = time.monotonic() + observation_seconds
        while True:
            result = self._status(task_id, after_sequence)
            changed = result["last_sequence"] > after_sequence
            if changed or result["status"] in TERMINAL or result["paused"]:
                return {**result, "changed": True}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {**result, "changed": False}
            await asyncio.sleep(min(1, remaining))

    def _team_attention(self, task_ids, after_sequences):
        states, attention = {}, []
        for task_id in task_ids:
            cursor = after_sequences.get(task_id, 0)
            observed = self._status(task_id, cursor)
            states[task_id] = {
                key: observed[key]
                for key in ("specialist", "status", "paused", "last_sequence", "error")
            }
            if observed["paused"] or (
                observed["status"] in TERMINAL and observed["last_sequence"] > cursor
            ):
                attention.append(task_id)
        return {
            "tasks": states,
            "attention_task_ids": attention,
            "after_sequences": {
                task_id: state["last_sequence"] for task_id, state in states.items()
            },
        }

    async def _wait_all(self, task_ids, after_sequences, observation_seconds):
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must contain distinct tasks")
        if set(after_sequences) - set(task_ids) or any(
            type(value) is not int or value < 0 for value in after_sequences.values()
        ):
            raise ValueError("after_sequences must contain nonnegative task cursors")
        if not 0 < observation_seconds <= 60:
            raise ValueError("observation_seconds must be in (0, 60]")
        deadline = time.monotonic() + observation_seconds
        while True:
            result = self._team_attention(task_ids, after_sequences)
            remaining = deadline - time.monotonic()
            if result["attention_task_ids"] or remaining <= 0:
                return result
            await asyncio.sleep(min(1, remaining))


def _register_reads(server, bridge):
    @server.tool()
    async def read_file(
        specialist: str, path: str, start_line: int = 1, end_line: int | None = None
    ) -> dict:
        """Read scoped source lines with the full-file hash for subsequent edits.

        Args: specialist: Configured workspace. path: Relative authorized file.
            start_line, end_line: Inclusive 1-based range; omitted end reads to EOF.
        Returns: Workbench read receipt. Raises: None; failures are receipts.
        """
        return await bridge._direct(
            specialist,
            "read_file",
            {"path": path, "start_line": start_line, "end_line": end_line},
        )

    @server.tool()
    async def list_files(specialist: str, path: str) -> dict:
        """List a scoped task directory.

        Args: specialist: Configured workspace. path: Relative authorized directory.
        Returns: Workbench listing receipt. Raises: None; failures are receipts.
        """
        return await bridge._direct(specialist, "list_files", {"path": path})


def _register_edits(server, bridge):
    @server.tool()
    async def edit_file(
        specialist: str, path: str, expected_sha256: str, old: str, new: str
    ) -> dict:
        """Replace exact text after verifying a scoped file's observed hash.

        Args: specialist: Workspace. path: File. expected_sha256: Read hash.
            old, new: Exact replacement text.
        Returns: Workbench edit receipt. Raises: None; failures are receipts.
        """
        return await bridge._direct(
            specialist,
            "edit_file",
            {
                "path": path,
                "expected_sha256": expected_sha256,
                "old": old,
                "new": new,
            },
        )


def _register_operations(server, bridge):
    @server.tool()
    async def run_operation(specialist: str, name: str) -> dict:
        """Execute one configured operation, preserving stdout and exit status.

        Args: specialist: Workspace. name: Operator-configured operation.
        Returns: Workbench command receipt. Raises: None; failures are receipts.
        """
        return await bridge._direct(specialist, "run_operation", {"name": name})

    @server.tool()
    async def run_operations(specialists: list[str], name: str) -> dict:
        """Execute one operation across distinct workspaces concurrently.

        Args: specialists: Distinct workspace names. name: Configured operation.
        Returns: Per-workspace receipts. Raises: ValueError: Repeated workspace.
        """
        return await bridge._recorded(
            "run_operations",
            {"specialists": specialists, "name": name},
            lambda: bridge._batch(specialists, name),
        )


def _register_delegation(server, bridge):
    @server.tool()
    async def delegate(specialist: str, task_id: str, goal: str) -> dict:
        """Enqueue work for an independent model worker using an idempotent task ID.

        Args: specialist: Configured role. task_id: Stable request ID. goal: Task.
        Returns: Durable assignment receipt. Raises: None; failures are receipts.
        """
        arguments = {"specialist": specialist, "task_id": task_id, "goal": goal}
        return await bridge._recorded(
            "delegate",
            arguments,
            lambda: asyncio.to_thread(
                bridge._delegate,
                specialist,
                task_id,
                goal,
            ),
        )


def _register_status(server, bridge):
    @server.tool()
    async def specialist_status(
        task_id: str, after_sequence: int = 0, include_read_content: bool = False
    ) -> dict:
        """Observe state and receipts without repeating previously read source text.

        Args: task_id: Delegated task. after_sequence: Last consumed event cursor.
            include_read_content: Include original read_file text; defaults to false.
        Returns: Current state and subsequent events. Raises: None; failures are receipts.
        """
        arguments = {
            "task_id": task_id,
            "after_sequence": after_sequence,
            "include_read_content": include_read_content,
        }
        return await bridge._recorded(
            "specialist_status",
            arguments,
            lambda: asyncio.to_thread(
                bridge._status,
                task_id,
                after_sequence,
                include_read_content,
            ),
        )


def _register_wait(server, bridge):
    @server.tool()
    async def wait_specialist(
        task_id: str, after_sequence: int = 0, observation_seconds: float = 30
    ) -> dict:
        """Wait locally for new evidence without making model calls or stopping jobs.

        Args: task_id: Task. after_sequence: Event cursor. observation_seconds:
            Observation interval in (0, 60]; this is not a workload deadline.
        Returns: Task evidence with a changed flag. Raises: None; failures are receipts.
        """
        arguments = {
            "task_id": task_id,
            "after_sequence": after_sequence,
            "observation_seconds": observation_seconds,
        }
        return await bridge._recorded(
            "wait_specialist",
            arguments,
            lambda: bridge._wait(
                task_id,
                after_sequence,
                observation_seconds,
            ),
        )


def _register_takeover(server, bridge):
    @server.tool()
    async def take_over(task_id: str) -> dict:
        """Take control of a blocked specialist after inspecting its resolved receipts.

        Args: task_id: A needs_attention task with no uncertain tool effects.
        Returns: Local cancellation receipt; remote workflows are untouched.
        Raises: None; active ownership and unresolved effects return failure receipts.
        """
        return await bridge._recorded(
            "take_over",
            {"task_id": task_id},
            lambda: asyncio.to_thread(
                bridge._take_over,
                task_id,
            ),
        )


def _register_team_wait(server, bridge):
    @server.tool()
    async def wait_specialists(
        task_ids: list[str],
        after_sequences: dict[str, int] | None = None,
        observation_seconds: float = 60,
    ) -> dict:
        """Wait for any specialist to finish or need attention; return compact states.

        Routine worker model/tool events do not wake the coordinator. Inspect
        specialist_status for full receipts on attention, then remove ended tasks.
        Args: task_ids: Distinct tasks. after_sequences: Last returned cursors.
            observation_seconds: Observation interval in (0, 60], not a job deadline.
        Returns: Compact states, attention IDs and new cursors. Raises: None;
            validation failures are retained as receipts.
        """
        cursors = after_sequences or {}
        arguments = {
            "task_ids": task_ids,
            "after_sequences": cursors,
            "observation_seconds": observation_seconds,
        }
        return await bridge._recorded(
            "wait_specialists",
            arguments,
            lambda: bridge._wait_all(task_ids, cursors, observation_seconds),
        )


def _register_observation_recovery(server, bridge):
    @server.tool()
    async def dismiss_interrupted_observation(task_id: str, call_id: str) -> dict:
        """Discard a lost observation result under its original operator declaration.

        Args: task_id: Task awaiting attention. call_id: Interrupted observation.
        Returns: Failed observation receipt; no replay, requeue or external effect.
        Raises: None; unsafe calls and mixed uncertainty return failure receipts.
        """
        return await bridge._recorded(
            "dismiss_interrupted_observation",
            {"task_id": task_id, "call_id": call_id},
            lambda: asyncio.to_thread(bridge._dismiss_observation, task_id, call_id),
        )


def _server(config_path, directory, hybrid=False):
    from mcp.server.mcpserver import MCPServer

    bridge = _Bridge(config_path, directory, hybrid)
    server = MCPServer("workbench-workflow-experiment")
    _register_reads(server, bridge)
    _register_edits(server, bridge)
    _register_operations(server, bridge)
    if hybrid:
        _register_delegation(server, bridge)
        _register_status(server, bridge)
        _register_wait(server, bridge)
        _register_team_wait(server, bridge)
        _register_takeover(server, bridge)
        if any(
            operation.observation_only
            for profile in bridge.team.config.profiles
            for operation in profile.operations.values()
        ):
            _register_observation_recovery(server, bridge)
    return server


if __name__ == "__main__":
    os.umask(0o077)
    _server(sys.argv[1], sys.argv[2], sys.argv[3] == "astra-tofa").run(
        transport="stdio"
    )
