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
from npa.agent_backend.specialists.store import TaskStore, UncertainOperation
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools

TERMINAL = {"completed", "needs_attention", "cancelled"}


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
        identity = str(uuid.uuid4())
        invocation = {"name": name, "arguments": arguments}
        self.store._begin_call("supervisor", identity, invocation)
        self.store._event(
            "supervisor", {"type": "tool_started", "call_id": identity, **invocation}
        )
        try:
            result = await action()
        except (ValueError, KeyError, BlockingIOError, UncertainOperation) as error:
            result = {
                "ok": False,
                "error": str(error),
                "error_type": type(error).__name__,
            }
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
            self._require_idle(specialist)
            executor = WorkbenchTools(profile, self.store, "coordinator-" + specialist)
            call = {
                "id": str(uuid.uuid4()),
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            return executor.execute(call)

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

    def _status(self, task_id, after_sequence=0):
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
            "events": [event for event in events if event["sequence"] > after_sequence],
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


def _register_reads(server, bridge):
    @server.tool()
    async def read_file(specialist: str, path: str) -> dict:
        """Read a scoped task file with its SHA-256.

        Args: specialist: Configured workspace. path: Relative authorized file.
        Returns: Workbench read receipt. Raises: None; failures are receipts.
        """
        return await bridge._direct(specialist, "read_file", {"path": path})

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

    @server.tool()
    async def specialist_status(task_id: str, after_sequence: int = 0) -> dict:
        """Observe task state, answer, new receipts and uncertain effects.

        Args: task_id: Delegated task. after_sequence: Last consumed event cursor.
        Returns: Current state and subsequent events. Raises: None; failures are receipts.
        """
        arguments = {"task_id": task_id, "after_sequence": after_sequence}
        return await bridge._recorded(
            "specialist_status",
            arguments,
            lambda: asyncio.to_thread(
                bridge._status,
                task_id,
                after_sequence,
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


def _server(config_path, directory, hybrid=False):
    from mcp.server.mcpserver import MCPServer

    bridge = _Bridge(config_path, directory, hybrid)
    server = MCPServer("workbench-workflow-experiment")
    _register_reads(server, bridge)
    _register_edits(server, bridge)
    _register_operations(server, bridge)
    if hybrid:
        _register_delegation(server, bridge)
        _register_wait(server, bridge)
        _register_takeover(server, bridge)
    return server


if __name__ == "__main__":
    os.umask(0o077)
    _server(sys.argv[1], sys.argv[2], sys.argv[3] == "astra-tofa").run(
        transport="stdio"
    )
