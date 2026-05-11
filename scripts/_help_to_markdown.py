#!/usr/bin/env python3
"""Convert Typer/Rich help output to deterministic Markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOX_LINE_RE = re.compile(r"^[╭╰├┤┬┴┼─╮╯│\s]+$")
ASCII_TRANSLATION = str.maketrans(
    {
        "→": "->",
        "—": "-",
        "–": "-",
        "‑": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
    }
)


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: _help_to_markdown.py <help-output> <name> <command-path>",
            file=sys.stderr,
        )
        return 2

    help_path = Path(sys.argv[1])
    name = sys.argv[2]
    command_path = sys.argv[3]
    raw = help_path.read_text()
    lines = _clean_help(raw)
    options = _extract_options(lines)
    commands = _extract_commands(lines)

    print(f"# `{command_path}`")
    print()
    print("## Command Tree")
    print()
    print("```text")
    print("\n".join(lines).strip())
    print("```")
    print()
    print("## Options")
    print()
    if options:
        print("| Option | Description |")
        print("| --- | --- |")
        for option, description in options:
            print(f"| `{option}` | {description or '-'} |")
    else:
        print("No command-specific options are listed by `--help`.")
    print()
    print("## Subcommands")
    print()
    if commands:
        print("| Command | Description |")
        print("| --- | --- |")
        for command, description in commands:
            print(f"| `{command}` | {description or '-'} |")
    else:
        print("No subcommands are listed by `--help`.")
    print()
    print("## Examples")
    print()
    print("```bash")
    print(f"{command_path} --help")
    if commands:
        first = commands[0][0]
        print(f"{command_path} {first} --help")
    print("```")
    print()
    print(
        f"Regenerate this page with `bash scripts/build_docs.sh` after changing `{name}`."
    )
    return 0


def _clean_help(raw: str) -> list[str]:
    cleaned: list[str] = []
    for line in raw.splitlines():
        line = ANSI_RE.sub("", line).rstrip()
        if any(char in line for char in "╭╰╮╯─"):
            section = _section_name(line)
            if section:
                line = section
            else:
                continue
        if BOX_LINE_RE.match(line):
            continue
        line = line.replace("│", " ").strip()
        line = line.translate(ASCII_TRANSLATION)
        line = line.encode("ascii", "ignore").decode("ascii")
        if not line:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        cleaned.append(re.sub(r"\s{2,}", "  ", line))
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def _section_name(line: str) -> str:
    for name in ("Options", "Commands", "Arguments"):
        if name in line:
            return name
    return ""


def _extract_options(lines: list[str]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        parts = re.split(r"\s{2,}", stripped, maxsplit=1)
        option = parts[0].replace("`", "")
        description = parts[1].replace("|", "\\|") if len(parts) > 1 else ""
        options.append((option, description))
    return _dedupe(options)


def _extract_commands(lines: list[str]) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    ignored = {
        "Usage:",
        "Options",
        "Commands",
        "Arguments",
        "Nebius",
        "No",
        "Regenerate",
    }
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        parts = re.split(r"\s{2,}", stripped, maxsplit=1)
        if len(parts) < 2:
            continue
        command = parts[0]
        if command in ignored or not re.match(r"^[a-z][a-z0-9-]*$", command):
            continue
        commands.append((command, parts[1].replace("|", "\\|")))
    return _dedupe(commands)


def _dedupe(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for key, value in items:
        if key in seen:
            continue
        seen.add(key)
        result.append((key, value))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
