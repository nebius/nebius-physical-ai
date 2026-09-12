"""Parse repository documentation links and literal NPA command examples offline."""

from __future__ import annotations

import os
import ast
import re
import shlex
import subprocess
from collections import Counter
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from typer.core import TyperGroup, TyperOption
import yaml

_MARKDOWN = MarkdownIt().enable("table")
_SHELL_LANGUAGES = {"bash", "sh", "shell", "console"}
_SHELL_BOUNDARIES = {"|", "||", "&&", ";", ">", ">>", "2>&1"}
_NON_USER_DOCS = {"archive", "architecture", "cli", "security", "testing"}


def _documentation_paths(root: Path) -> list[Path]:
    paths = {root / "README.md", root / "CONTRIBUTING.md"}
    for directory in ("docs", "workflows"):
        paths.update((root / directory).rglob("*.md"))
    for directory in ("npa", "deploy", "workbench", "research"):
        for parent, directories, files in os.walk(root / directory):
            directories[:] = [
                name for name in directories
                if not name.startswith(".") and name not in {"vendor", "node_modules"}
            ]
            if "README.md" in files:
                paths.add(Path(parent) / "README.md")
                paths.update(Path(parent).glob("*.md"))
    return sorted(paths)


def _has_current_commands(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if relative.parts[0] == "research" or path.name == "CONTRIBUTING.md":
        return False
    return not (
        relative.parts[0] == "docs"
        and len(relative.parts) > 2
        and relative.parts[1] in _NON_USER_DOCS
    )


class _HtmlReferences(HTMLParser):
    def __init__(self, content: str):
        super().__init__()
        self.links: list[str] = []
        self.anchors: set[str] = set()
        self.feed(content)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.anchors.add(values["id"])
        if tag == "a" and values.get("name"):
            self.anchors.add(values["name"])
        destination = values.get("href") if tag == "a" else values.get("src")
        if tag in {"a", "img", "source", "video"} and destination:
            self.links.append(destination)


def _inline_text(token) -> str:
    return "".join(
        child.content for child in token.children or []
        if child.type in {"text", "code_inline", "image"}
    )


@lru_cache(maxsize=None)
def _anchors(path: Path) -> set[str]:
    tokens = _MARKDOWN.parse(path.read_text(encoding="utf-8"))
    anchors: set[str] = set()
    counts: Counter[str] = Counter()
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            label = _inline_text(tokens[index + 1]).lower()
            slug = "".join(char for char in label if char.isalnum() or char in "-_ ")
            slug = slug.replace(" ", "-")
            unique = slug + (f"-{counts[slug]}" if counts[slug] else "")
            while unique in anchors:
                counts[slug] += 1
                unique = f"{slug}-{counts[slug]}"
            counts[slug] += 1
            anchors.add(unique)
        for child in [token, *(token.children or [])]:
            if child.type in {"html_block", "html_inline"}:
                anchors.update(_HtmlReferences(child.content).anchors)
    return anchors


def _links(content: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for token in _MARKDOWN.parse(content):
        line = (token.map or [0])[0] + 1
        for child in [token, *(token.children or [])]:
            if child.type in {"link_open", "image"}:
                found.append((line, child.attrGet("href") or child.attrGet("src")))
            elif child.type in {"html_block", "html_inline"}:
                found.extend((line, href) for href in _HtmlReferences(child.content).links)
    return found


def _link_errors(path: Path) -> list[str]:
    errors = []
    for line, href in _links(path.read_text(encoding="utf-8")):
        uri = urlsplit(href)
        if uri.scheme or uri.netloc or "$" in href:
            continue
        target = (path.parent / unquote(uri.path)).resolve() if uri.path else path
        if not target.exists():
            errors.append(f"line {line}: missing file: {href}")
        elif (
            uri.fragment and target.suffix.lower() == ".md"
            and unquote(uri.fragment) not in _anchors(target)
        ):
            errors.append(f"line {line}: missing anchor: {href}")
    return errors


def _shell_commands(content: str) -> list[tuple[int, list[str]]]:
    commands = []
    for token in _MARKDOWN.parse(content):
        if token.type != "fence" or token.info.strip() not in _SHELL_LANGUAGES:
            continue
        lines = token.content.splitlines(keepends=True)
        pending = ""
        start = 0
        for offset, line in enumerate(lines):
            if not pending:
                start = (token.map or [0])[0] + offset + 2
            pending += line.rstrip("\n").removesuffix("\\")
            if line.rstrip("\n").endswith("\\"):
                pending += " "
                continue
            command = pending.strip().removeprefix("$ ")
            argv = _npa_argv(command)
            if argv:
                commands.append((start, argv))
            pending = ""
    return commands


def _npa_argv(command: str) -> list[str]:
    command = re.sub(r"\$\([^)]*\)", "command-output", command)
    try:
        argv = shlex.split(command, comments=True)
    except ValueError:
        # Heredoc bodies may span quoted lines; Bash validates the whole block.
        return []
    if argv and argv[0] == "env":
        argv = argv[1:]
    while argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
        argv = argv[1:]
    if argv and Path(argv[0]).name == "npa":
        return ["npa", *argv[1:]]
    return []


def _python_syntax_errors(path: Path) -> list[str]:
    errors = []
    for token in _MARKDOWN.parse(path.read_text(encoding="utf-8")):
        if token.type != "fence" or token.info.strip() not in {"python", "py"}:
            continue
        try:
            ast.parse(token.content)
        except SyntaxError as exc:
            line = (token.map or [0])[0] + (exc.lineno or 1) + 1
            errors.append(f"line {line}: {exc.msg}")
    return errors


def _shell_syntax_errors(path: Path, bash: str) -> list[str]:
    errors = []
    for token in _MARKDOWN.parse(path.read_text(encoding="utf-8")):
        if token.type != "fence" or token.info.strip() not in {"bash", "sh", "shell"}:
            continue
        result = subprocess.run(
            [bash, "-n"], input=token.content, text=True, capture_output=True,
        )
        if result.returncode:
            line = (token.map or [0])[0] + 2
            errors.append(f"line {line}: {result.stderr.strip()}")
    return errors


def _option_parameters(command) -> dict:
    return {
        spelling: parameter
        for parameter in command.params if isinstance(parameter, TyperOption)
        for spelling in [*parameter.opts, *parameter.secondary_opts]
    }


def _command_error(root, argv: list[str]) -> str:
    command = root
    names = ["npa"]
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument in _SHELL_BOUNDARIES or argument == "--" or argument.startswith("..."):
            break
        if argument.startswith("-"):
            option = argument.split("=", 1)[0]
            if option == "--help":
                break
            parameter = _option_parameters(command).get(option)
            if parameter is None:
                return f"{' '.join(names)}: unknown option {option}"
            index += 1 if parameter.is_flag or "=" in argument else 1 + parameter.nargs
            continue
        if isinstance(command, TyperGroup):
            if argument.startswith(("$", "<", "[")):
                break
            context = command.context_class(command, info_name=names[-1])
            child = command.get_command(context, argument)
            if child is None:
                return f"{' '.join(names)}: unknown command {argument}"
            command = child
            names.append(argument)
        index += 1
    return ""


def _workflow_variable_errors(root: Path, argv: list[str]) -> list[str]:
    if argv[1:3] != ["workbench", "workflow"]:
        return []
    paths = [
        root / value for value in argv
        if value.startswith("workflows/") and value.endswith(".yaml")
    ]
    variables = [
        argv[index + 1].split("=", 1)[0]
        for index, value in enumerate(argv[:-1]) if value == "--var"
    ]
    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"missing workflow specification: {path.relative_to(root)}")
            continue
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or spec.get("apiVersion") != "npa.workflow/v0.0.1":
            continue
        for variable in variables:
            if variable not in spec.get("config", {}):
                errors.append(f"{path.name}: undeclared configuration variable {variable}")
    return errors
