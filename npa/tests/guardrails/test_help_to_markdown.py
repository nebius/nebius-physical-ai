"""Regression coverage for wrapped Rich help in generated CLI Markdown."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "_help_to_markdown.py"


def _module():
    spec = importlib.util.spec_from_file_location("help_to_markdown", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapped_option_descriptions_are_joined_without_phantom_flags() -> None:
    converter = _module()
    raw = """
╭─ Options ─────────────────────────────────────────────────────────────╮
│ --provision  --no-provision  Auto-create a bucket.                   │
│                              Use --no-provision to enter credentials. │
│                              [default: provision]                     │
│ --help                     Show this message and exit.                │
╰───────────────────────────────────────────────────────────────────────╯
"""

    options = converter._extract_options(converter._clean_help(raw))

    assert options == [
        (
            "--provision",
            "--no-provision  Auto-create a bucket. Use --no-provision to enter "
            "credentials. [default: provision]",
        ),
        ("--help", "Show this message and exit."),
    ]


def test_wrapped_command_descriptions_are_joined() -> None:
    converter = _module()
    raw = """
╭─ Commands ────────────────────────────────────────────────────────────╮
│ actions  Generate action-conditioned train envs for a                │
│          representative slice.                                       │
│ split    Generate deterministic train/heldout manifests.             │
╰───────────────────────────────────────────────────────────────────────╯
"""

    commands = converter._extract_commands(converter._clean_help(raw))

    assert commands == [
        ("actions", "Generate action-conditioned train envs for a representative slice."),
        ("split", "Generate deterministic train/heldout manifests."),
    ]
