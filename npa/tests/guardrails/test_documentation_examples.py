"""Keep user documentation navigable and its literal CLI names aligned with NPA."""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest
from typer.main import get_command

from npa.cli.main import app

from documentation_examples import (
    _anchors,
    _command_error,
    _documentation_paths,
    _has_current_commands,
    _link_errors,
    _python_syntax_errors,
    _shell_commands,
    _shell_syntax_errors,
    _workflow_variable_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS = _documentation_paths(REPO_ROOT)
COMMAND_DOCUMENTS = [path for path in DOCUMENTS if _has_current_commands(path, REPO_ROOT)]


@pytest.fixture(scope="module")
def cli_tree():
    """Build command metadata without invoking callbacks or cloud operations."""
    return get_command(app)


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_documented_relative_links_resolve(path: Path) -> None:
    errors = _link_errors(path)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "path", COMMAND_DOCUMENTS, ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_documented_literal_commands_and_options_exist(path: Path, cli_tree) -> None:
    errors = []
    for line, argv in _shell_commands(path.read_text(encoding="utf-8")):
        error = _command_error(cli_tree, argv)
        if error:
            errors.append(f"line {line}: {error}")
        errors.extend(
            f"line {line}: {error}" for error in _workflow_variable_errors(REPO_ROOT, argv)
        )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "path", COMMAND_DOCUMENTS, ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_documented_shell_and_python_examples_parse_without_execution(path: Path) -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required to parse the documented shell examples")
    errors = _shell_syntax_errors(path, bash) + _python_syntax_errors(path)
    assert not errors, "\n".join(errors)


def test_shell_syntax_check_rejects_redirection_placeholders_without_running_code(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for shell syntax validation")
    source = tmp_path / "README.md"
    marker = tmp_path / "must-not-exist"
    source.write_text(
        f"```bash\ntouch '{marker}'\nnpa configure --project-id <project-id>\n```\n",
        encoding="utf-8",
    )
    assert len(_shell_syntax_errors(source, bash)) == 1
    source.write_text(f"```bash\ntouch '{marker}'\n```\n", encoding="utf-8")
    assert _shell_syntax_errors(source, bash) == []
    assert not marker.exists()


def test_documentation_inventory_excludes_vendor_and_virtualenv(tmp_path: Path) -> None:
    for relative in (
        "npa/.venv/README.md", "deploy/cluster/vendor/tool/README.md",
        "npa/src/npa/new-tool/README.md", "docs/new-guide.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    paths = _documentation_paths(tmp_path)
    assert tmp_path / "npa/src/npa/new-tool/README.md" in paths
    assert tmp_path / "docs/new-guide.md" in paths
    assert not any(".venv" in path.parts or "vendor" in path.parts for path in paths)


def test_link_check_covers_references_images_html_and_anchors(tmp_path: Path) -> None:
    target = tmp_path / "next page.md"
    target.write_text("# Setup\n\n## Setup\n\n<a id='manual'></a>\n", encoding="utf-8")
    source = tmp_path / "README.md"
    source.write_text(
        "[next][guide]\n\n[guide]: next%20page.md#setup-1\n"
        "[manual](next%20page.md#manual)\n"
        "[broken](next%20page.md#gone)\n"
        "![missing image](image.png)\n"
        "<a href='gone.md'>missing HTML link</a>\n"
        "<img src='gone.svg' alt='missing HTML image'>\n"
        "```md\n[example](ignore.md)\n```\n"
        "[external](https://example.com)\n",
        encoding="utf-8",
    )
    errors = _link_errors(source)
    assert len(errors) == 4
    for destination in ("#gone", "image.png", "gone.md", "gone.svg"):
        assert any(destination in error for error in errors)


def test_heading_anchors_handle_formatting_and_duplicate_collisions(tmp_path: Path) -> None:
    source = tmp_path / "headings.md"
    source.write_text(
        "# Run `npa` & **inspect**\n# Setup\n# Setup-1\n# Setup\n"
        "```html\n<a id='example-only'></a>\n```\n",
        encoding="utf-8",
    )
    assert _anchors(source) == {"run-npa--inspect", "setup", "setup-1", "setup-2"}


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("npa registry delete", "unknown command registry"),
        ("npa workbench nurec check --json", "unknown option --json"),
        ("npa workbench workflow submit '$SPEC' --invented", "unknown option --invented"),
        ("npa workbench workflow submit demo.yaml --var 'prompt=--literal' --runtime", ""),
        ("npa --help", ""),
        ("npa workbench nurec check --output json | jq .", ""),
    ],
)
def test_command_check_detects_retired_names_without_running_them(
    command: str, message: str, cli_tree,
) -> None:
    _, argv = _shell_commands(f"```bash\n{command}\n```\n")[0]
    assert message in _command_error(cli_tree, argv)
    if not message:
        assert _command_error(cli_tree, argv) == ""


def test_shell_parser_preserves_line_numbers_and_continuations() -> None:
    content = "# Example\n\n```bash\n# comment\nnpa workbench workflow submit \\\n  '$SPEC' --runtime\n$ npa --version\n```\n"
    assert _shell_commands(content) == [
        (5, ["npa", "workbench", "workflow", "submit", "$SPEC", "--runtime"]),
        (7, ["npa", "--version"]),
    ]


@pytest.mark.parametrize("prefix", ["NPA_DEBUG=1 ", "env NPA_DEBUG=1 ", "npa/.venv/bin/"])
def test_command_check_also_covers_environment_and_interpreter_prefixes(
    prefix: str, cli_tree,
) -> None:
    _, argv = _shell_commands(f"```bash\n{prefix}npa registry delete\n```\n")[0]
    assert _command_error(cli_tree, argv) == "npa: unknown command registry"


def test_workflow_examples_cannot_silently_override_an_unused_variable(tmp_path: Path) -> None:
    path = tmp_path / "workflows/main/example.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "apiVersion: npa.workflow/v0.0.1\nconfig:\n  bucket: example-bucket\n",
        encoding="utf-8",
    )
    command = ["npa", "workbench", "workflow", "submit", "workflows/main/example.yaml"]
    assert _workflow_variable_errors(tmp_path, [*command, "--var", "bucket=example-bucket"]) == []
    assert _workflow_variable_errors(
        tmp_path, [*command, "--var", "NPA_SIM2REAL_BUCKET=example-bucket"]
    ) == ["example.yaml: undeclared configuration variable NPA_SIM2REAL_BUCKET"]


def test_package_readme_python_preview_runs_without_cloud_credentials(
    tmp_path: Path,
) -> None:
    from markdown_it import MarkdownIt

    blocks = MarkdownIt().parse((REPO_ROOT / "npa/README.md").read_text())
    examples = [
        block.content for block in blocks
        if block.type == "fence" and block.info == "python" and "build_plan" in block.content
    ]
    assert len(examples) == 1
    example = tmp_path / "readme_preview.py"
    example.write_text(
        "import socket\n"
        "def refuse_network(*args, **kwargs):\n"
        "    raise RuntimeError('The README planning example must not contact a service')\n"
        "socket.socket.connect = refuse_network\n\n" + examples[0],
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", str(example)], cwd=REPO_ROOT,
        env={"PATH": os.defpath, "NPA_CONFIG_DIR": str(tmp_path / "empty-config")},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "generate workbench.cosmos3.generate"
