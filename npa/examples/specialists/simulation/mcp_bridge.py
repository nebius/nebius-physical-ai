"""Give a single Codex baseline the same scoped Workbench tools as the specialists."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

from npa.agent_backend.specialists.config import load_config
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


def _server(path):
    from mcp.server.mcpserver import MCPServer

    team = SpecialistTeam(load_config(path))
    server = MCPServer("workbench-experiment")

    async def invoke(specialist, name, arguments):
        profile = team.config.profile(specialist)
        executor = WorkbenchTools(profile, team.store, specialist)
        call = {
            "id": str(uuid.uuid4()),
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        return await asyncio.to_thread(executor.execute, call)

    _register_reads(server, invoke)
    _register_writes(server, invoke)
    _register_batches(server, invoke)
    return server


def _register_reads(server, invoke):
    @server.tool()
    async def read_file(
        specialist: str, path: str, start_line: int = 1, end_line: int | None = None
    ) -> dict:
        """Read authorized source lines with the full-file SHA-256.

        Args: specialist: Task name. path: Workspace-relative file.
            start_line, end_line: Inclusive 1-based range; omitted end reads to EOF.
        Returns: The identical Workbench read receipt used by specialists.
        Raises: ValueError: Scope or task is invalid.
        """
        return await invoke(
            specialist,
            "read_file",
            {"path": path, "start_line": start_line, "end_line": end_line},
        )

    @server.tool()
    async def list_files(specialist: str, path: str) -> dict:
        """List files under an authorized directory.

        Args: specialist: Task name. path: Workspace-relative directory.
        Returns: The identical Workbench listing receipt used by specialists.
        Raises: ValueError: Scope or task is invalid.
        """
        return await invoke(specialist, "list_files", {"path": path})


def _register_writes(server, invoke):
    @server.tool()
    async def edit_file(
        specialist: str, path: str, expected_sha256: str, old: str, new: str
    ) -> dict:
        """Replace one exact occurrence in an authorized file after checking its hash.

        Args: specialist: Task. path: File. expected_sha256: Read hash. old, new: Replacement.
        Returns: The identical Workbench edit receipt used by specialists.
        Raises: ValueError: Scope, hash or replacement is invalid.
        """
        return await invoke(
            specialist,
            "edit_file",
            {"path": path, "expected_sha256": expected_sha256, "old": old, "new": new},
        )

    @server.tool()
    async def run_operation(specialist: str, name: str) -> dict:
        """Run validate or simulate for one task; independent tasks may run concurrently.

        Args: specialist: Task. name: Configured operation name.
        Returns: Real stdout, stderr, exit status and operation receipt.
        Raises: ValueError: Task or operation is not configured.
        """
        return await invoke(specialist, "run_operation", {"name": name})


def _register_batches(server, invoke):
    @server.tool()
    async def run_operations(specialists: list[str], name: str) -> dict:
        """Fan out the same allowed operation across independent tasks concurrently.

        Args: specialists: Distinct task names. name: Configured operation.
        Returns: Per-task receipts from the unchanged Workbench executor.
        Raises: ValueError: A task is repeated or outside the configured team.
        """
        if len(specialists) != len(set(specialists)):
            raise ValueError("each task must appear once")
        results = await asyncio.gather(
            *(
                invoke(specialist, "run_operation", {"name": name})
                for specialist in specialists
            )
        )
        return dict(zip(specialists, results))


if __name__ == "__main__":
    os.umask(0o077)
    _server(sys.argv[1]).run(transport="stdio")
