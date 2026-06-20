"""Catalog of workbench tools referenced by ``toolRef`` in NPA workflow specs."""

from __future__ import annotations

from dataclasses import dataclass

from npa.orchestration.npa_workflow.errors import NpaWorkflowError


@dataclass(frozen=True)
class ToolEntry:
    name: str
    argv_template: list[str]
    description: str = ""


TOOL_CATALOG: dict[str, ToolEntry] = {
    "workbench.vlm_eval.run": ToolEntry(
        name="workbench.vlm_eval.run",
        description="Score rollout directories with the VLM eval workbench tool.",
        argv_template=[
            "npa",
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "{{config.rollouts_uri}}",
            "--output-path",
            "{{config.scores_uri}}",
            "--backend",
            "{{config.vlm_backend}}",
        ],
    ),
    "workbench.token_factory.reason": ToolEntry(
        name="workbench.token_factory.reason",
        description="Run Cosmos reasoner over scene inputs.",
        argv_template=[
            "npa",
            "workbench",
            "token-factory",
            "reason",
            "--input-path",
            "{{config.scene_uri}}",
            "--output-path",
            "{{config.plan_uri}}",
        ],
    ),
    "workbench.cosmos2.transfer": ToolEntry(
        name="workbench.cosmos2.transfer",
        description="Cosmos Transfer augment stage.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos2",
            "transfer",
            "--input-path",
            "{{config.trigger_uri}}",
            "--output-path",
            "{{config.augment_uri}}",
        ],
    ),
    "workbench.sim2real_envgen.raw_shard": ToolEntry(
        name="workbench.sim2real_envgen.raw_shard",
        description="Generate raw simulation env shard.",
        argv_template=[
            "python",
            "-m",
            "npa.workflows.sim2real_envgen",
            "raw-shard",
            "--output-uri",
            "{{config.raw_envs_uri}}",
            "--env-count",
            "{{config.env_count}}",
        ],
    ),
}


def validate_tool_ref(tool_ref: str) -> ToolEntry:
    entry = TOOL_CATALOG.get(tool_ref)
    if entry is None:
        known = ", ".join(sorted(TOOL_CATALOG))
        raise NpaWorkflowError(f"unknown toolRef {tool_ref!r} (known: {known})")
    return entry


def argv_for_tool(tool_ref: str) -> list[str]:
    return list(validate_tool_ref(tool_ref).argv_template)
