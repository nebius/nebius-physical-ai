"""Operator-goal scenarios for the agent task-eval harness.

Each scenario declares a goal and an expected end-state. The harness
(`harness.py`) executes it against the real, pure agent modules with mocked
collaborators (0 tokens) and scores success/steps/tokens.

Kinds:
- ``grounded``     — zero-token intent router + grounded reply.
- ``workflow``     — draft + validate an npa.workflow spec via the grounded path.
- ``action_loop``  — Phase-B bounded tool-calling loop (mocked planner + tools).
- ``sim2real_loop``— Phase-C autonomous drive (mocked launch/status/gate).
- ``semantic``     — Phase-D fallthrough for a paraphrase the regex misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    goal: str
    expected: dict[str, Any] = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    # ── grounded Q&A (0 tokens) ──────────────────────────────────────────────
    Scenario(
        id="grounded_status",
        kind="grounded",
        goal="what is the current sim2real status",
        expected={"intent": "sim2real_status"},
    ),
    Scenario(
        id="grounded_tools_catalog",
        kind="grounded",
        goal="show me the workbench tools catalog",
        expected={"intent": "tools_catalog"},
    ),
    Scenario(
        id="grounded_cosmos_caps",
        kind="grounded",
        goal="what does cosmos support",
        expected={"intent": "cosmos_capabilities"},
    ),
    # ── workflow draft + validate ────────────────────────────────────────────
    Scenario(
        id="workflow_vlm_rl_draft",
        kind="workflow",
        goal="create a vlm-rl workflow with an outer loop gate",
        expected={"intent": "create_vlm_rl_workflow"},
    ),
    # ── Phase-B action loop ──────────────────────────────────────────────────
    Scenario(
        id="action_status_then_answer",
        kind="action_loop",
        goal="check the live sim-viz status and summarize it",
        expected={"tool": "sim_viz_status", "stopped_reason": "done"},
    ),
    Scenario(
        id="action_gpu_needs_confirmation",
        kind="action_loop",
        goal="launch a sim2real run now",
        expected={"needs_confirmation": True, "tool": "sim2real_submit"},
    ),
    # ── Phase-C sim2real drive ───────────────────────────────────────────────
    Scenario(
        id="sim2real_promote",
        kind="sim2real_loop",
        goal="drive the sim2real loop until the gate passes",
        expected={"decision": "promote_checkpoint", "stopped_reason": "promoted"},
    ),
    Scenario(
        id="sim2real_needs_confirmation",
        kind="sim2real_loop",
        goal="autonomously drive sim2real",
        expected={"needs_confirmation": True},
    ),
    # ── Phase-D semantic fallthrough ─────────────────────────────────────────
    Scenario(
        id="semantic_watch_paraphrase",
        kind="semantic",
        goal="could you keep an eye on the simulation for me",
        expected={"intent": "watch_sim"},
    ),
    # ── Blueprint Phase-H retrieval/grounding ────────────────────────────────
    Scenario(
        id="retrieval_genesis_doc",
        kind="retrieval",
        goal="genesis gpu physics simulator",
        expected={
            "uri": "docs/genesis.md",
            "corpus": [
                ("docs/genesis.md", "Genesis", "Genesis is a GPU physics simulator for robotics training."),
                ("docs/storage.md", "Storage", "Configure S3 object storage buckets and credentials."),
            ],
        },
    ),
]
