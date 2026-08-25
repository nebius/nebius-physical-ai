"""Guard the provider-neutral NPA workflow operations contract for agents."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GUIDE = REPO_ROOT / "docs" / "workbench" / "agent-workflow-operations.md"
_CONSOLE_FENCE_RE = re.compile(r"```console\n(.*?)```", re.DOTALL)


def test_agent_workflow_journey_uses_only_fixed_npa_subprocesses() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    commands = [
        line.strip()
        for block in _CONSOLE_FENCE_RE.findall(text)
        for line in block.splitlines()
        if line.strip()
    ]

    assert commands
    assert all(command.startswith("npa ") for command in commands)
    assert not any(
        forbidden in command
        for command in commands
        for forbidden in (" kubectl ", " sky ", " tmux ", "curl ", "localhost")
    )
    assert not any("--model" in command or "--provider" in command for command in commands)


def test_agent_workflow_journey_covers_the_bounded_lifecycle() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for operation in (
        "GET /api/tools",
        "POST /api/workflows/draft",
        "npa workbench workflow validate-spec",
        "npa workbench workflow plan-spec",
        "npa workbench health preflight",
        "npa provision-if-absent",
        "npa workbench workflow submit",
        "--resume-run <run-id>",
        "npa workbench workflow status",
        "npa workbench workflow logs",
        "--max-output-chars 32768",
        "npa workbench workflow artifacts",
        "npa workbench workflow cancel",
        "npa cluster down",
    ):
        assert operation in text

    assert "apiVersion: npa.workflow/v0.0.1" in text
    assert "caller owns its reasoning system" in text
    assert "always retain\n`--runtime`" in text
