"""Hosted-model defaults that cannot import the client must mirror it exactly.

``npa.clients.token_factory`` is the source of truth. The agent VM embeds
``agent_routing`` / ``agent_workflow`` verbatim, the content-agents image copies a
handful of files without httpx, and scripts, examples, and workflow specs carry
literals. Each mirror is one row here, so the next model rotation is one edit
per row instead of an archaeology exercise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from npa.clients import token_factory as tf

ROOT = Path(__file__).resolve().parents[2]


def test_importable_mirrors() -> None:
    from npa.cli import agent, agent_routing, agent_workflow
    from npa.workflows import content_agents

    assert agent_routing.VISION_MODEL == tf.DEFAULT_VISION_MODEL
    assert agent_routing.REASONING_MODEL == tf.DEFAULT_REASONER_MODEL
    assert agent_routing.STANDARD_MODEL == tf.DEFAULT_TEXT_MODEL
    assert agent.DEFAULT_LLM_MODEL == tf.DEFAULT_REASONER_MODEL
    assert tf.DEFAULT_VISION_MODEL in agent.DEFAULT_LLM_MODELS
    spec = agent_workflow._data_factory_spec()
    assert spec["config_runtime"]["caption_model"] == tf.DEFAULT_VISION_MODEL
    assert content_agents.DEFAULT_MODEL == tf.DEFAULT_VISION_MODEL
    assert content_agents.VISION_MODEL_ENV == tf.VISION_MODEL_ENV
    assert content_agents.DEFAULT_BASE_URL == tf.DEFAULT_BASE_URL


SPECS = ROOT / "npa/workflows/workbench/npa-workflows"
TEXT_MIRRORS = [
    ("npa/examples/isaac_franka_token_factory_reason.py", r'DEFAULT_MODEL = "([^"]+)"', tf.DEFAULT_REASONER_MODEL),
    ("npa/scripts/run_agent_local.py", r'llm_model="([^"]+)"', tf.DEFAULT_REASONER_MODEL),
    ("npa/scripts/audit_agent_capabilities.py", r'llm_model="([^"]+)"', tf.DEFAULT_REASONER_MODEL),
    *[
        (str(path.relative_to(ROOT)), r'caption_model: "?([^"\n]+)"?', tf.DEFAULT_VISION_MODEL)
        for path in sorted(SPECS.glob("*.yaml"))
        if "caption_model:" in path.read_text()
    ],
    *[
        (str(path.relative_to(ROOT)), r"reason_model: \"?([^\"\n]+)\"?", tf.DEFAULT_REASONER_MODEL)
        for path in sorted(SPECS.glob("*.yaml"))
        if "reason_model:" in path.read_text()
    ],
]


@pytest.mark.parametrize("relpath,pattern,expected", TEXT_MIRRORS, ids=lambda v: str(v)[:60])
def test_text_mirrors(relpath: str, pattern: str, expected: str) -> None:
    text = (ROOT / relpath).read_text()
    values = {m.strip() for m in re.findall(pattern, text)}
    assert values == {expected}, f"{relpath}: {values} != {expected}"
