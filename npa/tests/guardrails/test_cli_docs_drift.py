"""Guardrail: npa command mentions in onboarding docs/skills resolve in the CLI.

Scans ``docs/workbench/``, ``workflows/guides/``, and ``skills/`` for
``npa ...`` command mentions and asserts each one resolves to a real command
in the Typer tree. This catches renamed or removed commands that docs and
skills still reference (e.g. the stale ``npa workbench retargeting workflow``
from #520).

The check is deliberately scoped to onboarding-relevant roots: the full docs
tree contains historical design notes that intentionally reference commands
that never shipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer.main

from npa.cli.main import app


REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOTS = (
    REPO_ROOT / "docs" / "workbench",
    REPO_ROOT / "workflows" / "guides",
    REPO_ROOT / "skills",
)

# Documented `npa ...` stems that do not resolve to a CLI command, each with a
# reason. Keep entries precise: a stem here is either a positional argument
# that reads like a subcommand, a prose fragment the extractor over-matched,
# or a known-stale reference tracked by its own issue.
ALLOWLIST: dict[tuple[str, ...], str] = {
    # Positional arguments (spec paths, spec names, tool names, file names)
    # that the extractor reads as subcommand tokens.
    ("workbench", "workflow", "submit", "workflows"): "positional spec-path arg",
    ("workbench", "workflow", "validate-spec", "workflows"): "positional spec-path arg",
    ("workbench", "workflow", "plan-spec", "workflows"): "positional spec-path arg",
    ("workbench", "workflow", "preflight-images", "workflows"): "positional spec-path arg",
    ("workbench", "workflow", "validate-spec", "alpamayo2-super-inference"): "positional spec-name arg",
    ("workbench", "workflow", "plan-spec", "alpamayo2-super-inference"): "positional spec-name arg",
    ("workbench", "workflow", "submit", "alpamayo2-super-inference"): "positional spec-name arg",
    ("workbench", "workflow", "submit", "physical-ai-data-factory"): "positional spec-name arg",
    ("workbench", "golden-eval", "show", "cosmos3-ray-serve"): "positional target arg",
    ("workbench", "golden-eval", "run-all", "lerobot", "groot"): "positional tool-name args",
    ("rerun", "host", "recording"): "positional file arg (recording.rrd)",
    ("rerun", "share", "recording"): "positional file arg (recording.rrd)",
    # Prose fragments the extractor over-matched, not command invocations.
    ("workbench", "health", "preflight", "for", "the", "selected"): "prose: 'for the selected services'",
    ("workbench", "health", "preflight", "for", "the", "selected", "project"): "prose: 'for the selected project'",
    ("storage", "helpers", "or"): "prose: 'npa storage helpers or ...'",
    ("agent", "without", "workflow"): "prose: 'npa agent without workflow ...'",
    ("workflow", "pod"): "prose: 'npa workflow pod' (a Kubernetes pod)",
    # Deliberate negative example: the doc says not to invent this tool.
    ("workbench", "data-factory"): "negative example in physical-ai-data-factory/SKILL.md",
    # Passthrough subcommands: `npa studio` forwards to npa.studio.run's argparse dispatcher (search/create), invisible to the Typer tree walk.
    ("studio", "search"): "passthrough dispatched by npa.studio.run",
    # Known-stale references tracked by their own issues; remove the entry
    # when the referenced issue lands.
    ("workbench", "retargeting", "workflow"): "stale command; #520 tracks fixing the docs",
}


def _command_paths() -> set[tuple[str, ...]]:
    """Every valid command path in the Typer tree, including groups."""

    def walk(cmd: object, prefix: tuple[str, ...]) -> None:
        for name, sub in getattr(cmd, "commands", {}).items():
            path = (*prefix, name)
            paths.add(path)
            walk(sub, path)

    paths: set[tuple[str, ...]] = set()
    walk(typer.main.get_command(app), ())
    return paths


def _mentioned_stems() -> dict[tuple[str, ...], list[str]]:
    """Map each documented `npa ...` stem to the files that mention it."""
    top_level = sorted({path[0] for path in _command_paths()})
    # Same-line only: a newline usually starts a new command. Continuation
    # tokens exclude a bare `npa` so adjacent mentions stay separate.
    pattern = re.compile(
        r"\bnpa[ \t]+((?:" + "|".join(top_level) + r")"
        r"(?:[ \t]+(?!npa\b)[a-z][a-z0-9-]*)*)"
    )
    stems: dict[tuple[str, ...], list[str]] = {}
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.md")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                for match in pattern.finditer(line):
                    stem = tuple(match.group(1).split())
                    stems.setdefault(stem, []).append(str(path.relative_to(REPO_ROOT)))
    return stems


def test_documented_npa_commands_resolve() -> None:
    paths = _command_paths()
    failures: list[str] = []
    for stem, files in sorted(_mentioned_stems().items()):
        if stem in paths or stem in ALLOWLIST:
            continue
        failures.append(
            f"`npa {' '.join(stem)}` does not resolve in the CLI "
            f"(mentioned in {files[0]}; {len(files)} file(s) total)"
        )
    assert not failures, (
        "Documented npa commands that do not exist in the Typer tree:\n"
        + "\n".join(failures)
        + "\nFix the docs/skills, or add a documented ALLOWLIST entry."
    )


def test_allowlist_entries_are_documented() -> None:
    # Every allowlist entry must carry a reason so the list does not become a
    # dumping ground for unexamined misses.
    undocumented = [stem for stem, reason in ALLOWLIST.items() if not reason.strip()]
    assert not undocumented, f"ALLOWLIST entries without a reason: {undocumented}"
