"""Public defaults must use reproducible source and the governed image inventory."""

from pathlib import Path
import re
import subprocess

import pytest

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    public_release_manifest,
)
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    resolve_task_image,
    tool_image_key,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
IMPACT_REPORT = (
    REPO_ROOT / "docs/workbench/validation/public-default-quarantine-impact-20261005.md"
)


def _task_tool(task: dict) -> str:
    image = str(task["resources"].get("image") or "")
    if image.startswith("tool://"):
        return image.removeprefix("tool://")
    for tool, name in CONTAINER_IMAGE_NAMES.items():
        if f"/{name}:" in image:
            return tool
    return tool_image_key(task["tool_ref"]) or ""


def _blocked_shipped_defaults() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    blocked, required = set(), set()
    options = SkypilotRenderOptions()
    for path in iter_npa_workflow_specs():
        spec = load_spec(path)
        for decision in ("promote_checkpoint", "loop_back"):
            try:
                plan = build_plan(
                    spec, run_id="quarantine-audit", assume_decision=decision
                )
            except NpaWorkflowError as exc:
                assert "requires an explicit registry-qualified immutable image" in str(
                    exc
                )
                keys = spec.config["required_immutable_images"]
                assert keys and any(f"config.{key}" in str(exc) for key in keys)
                required.update((str(path.relative_to(REPO_ROOT)), key) for key in keys)
                continue
            for step in plan.steps:
                task = build_scheduler_task(spec, step, run_id="quarantine-audit")
                try:
                    resolve_task_image(
                        task["tool_ref"], task["resources"], options=options
                    )
                except NpaWorkflowError as exc:
                    assert "quarantined" in str(exc) or "no accepted release" in str(
                        exc
                    )
                    tool = _task_tool(task)
                    assert tool, "A blocked default must identify its image tool"
                    blocked.add((str(path.relative_to(REPO_ROOT)), tool))
    return blocked, required


def test_shipped_defaults_match_documented_quarantine_impact(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.invalid/operator/private")
    current = IMPACT_REPORT.read_text().split("## Current correction, 2026-10-06\n", 1)[
        1
    ]
    images, inputs = current.split("### Required exact operator image inputs\n", 1)
    rows = re.findall(r"\| `(workflows/[^`]+\.yaml)` \| `([^`]+)` \|", images)
    documented = {(path, tool) for path, tools in rows for tool in tools.split(", ")}
    assert documented
    required_rows = re.findall(r"\| `(workflows/[^`]+\.yaml)` \| `([^`]+)` \|", inputs)
    documented_inputs = {
        (path, key) for path, keys in required_rows for key in keys.split(", ")
    }
    assert _blocked_shipped_defaults() == (documented, documented_inputs)


@pytest.mark.parametrize(
    "tool", sorted(public_release_manifest()["workflow_validation_candidates"])
)
def test_public_candidate_source_is_in_main_history(tool: str) -> None:
    entry = public_release_manifest()["workflow_validation_candidates"][tool]
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            entry["development_sha"],
            "origin/main",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{tool} source revision is outside main history"


@pytest.mark.parametrize(
    "tool", sorted(public_release_manifest()["workflow_validation_candidates"])
)
def test_default_rendering_uses_governed_public_candidate_without_overrides(
    tool: str,
) -> None:
    entry = public_release_manifest()["workflow_validation_candidates"][tool]
    for tool_ref in entry["validation_tool_refs"]:
        image = resolve_task_image(tool_ref, {}, options=SkypilotRenderOptions())
        assert image.startswith("ghcr.io/nebius/nebius-physical-ai/")
        assert image.endswith("@" + entry["published_digest"])


def test_operator_guide_identifies_both_public_and_private_pin_sources() -> None:
    guide = (
        REPO_ROOT / "docs/workbench/guides/physical-ai-data-factory-deploy.md"
    ).read_text()
    image_section = guide.split("## 4. ", 1)[1].split("## 5. ", 1)[0]
    assert "public_release_manifest.json" in image_section
    assert "supported_tool_version()" in image_section
    assert "old public bytes remain quarantined" in image_section


@pytest.mark.parametrize(
    "path",
    [
        "workflows/testing/physical-ai-data-factory.yaml",
        "workflows/testing/nvidia-paidf-vda-cosmos-transfer25.yaml",
        "workflows/main/paidf-cosmos3.yaml",
    ],
)
def test_paidf_defaults_execute_submitted_adapters(path: str, monkeypatch) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import source_overlay_requested

    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    spec = load_spec(REPO_ROOT / path)
    assert spec.config["source_overlay"] is True
    assert source_overlay_requested(spec.config)
