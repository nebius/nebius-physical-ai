"""Access-key inventory must never request or publish secret-bearing list JSON."""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_TEXT_ROOTS = (
    REPO_ROOT / "docs",
    REPO_ROOT / "npa" / "src",
    REPO_ROOT / "research",
    REPO_ROOT / "scripts",
    REPO_ROOT / "skills",
)


def _public_text() -> str:
    chunks = [
        (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
    ]
    for root in PUBLIC_TEXT_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".md",
                ".py",
                ".sh",
                ".yaml",
                ".yml",
            }:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def test_public_examples_never_use_raw_json_for_access_key_lists() -> None:
    text = _public_text()
    commands = list(
        re.finditer(
            r"(?m)^\s*(?:\$\s*)?nebius\s+iam\s+v2\s+access-key\s+list\b",
            text,
            re.IGNORECASE,
        )
    )

    assert commands
    for command in commands:
        window = text[command.start() : command.start() + 900]
        assert not re.search(
            r"--format\s+(?:['\"])?json(?:['\"])?(?:\s|$)",
            window,
            re.IGNORECASE,
        ), window


def test_every_shell_or_documented_access_key_list_selects_fields_in_cli() -> None:
    text = _public_text()
    occurrences = list(
        re.finditer(
            r"(?m)^\s*(?:\$\s*)?nebius\s+iam\s+v2\s+access-key\s+list\b",
            text,
            re.IGNORECASE,
        )
    )

    assert occurrences, "expected the safe recovery example and research wrapper"
    for match in occurrences:
        window = text[match.start() : match.start() + 900]
        assert re.search(r"--format\s+['\"]?jsonpath=", window, re.IGNORECASE), window


def test_python_inventory_does_not_call_generic_raw_json_runner() -> None:
    source = (REPO_ROOT / "npa" / "src" / "npa" / "clients" / "nebius.py").read_text(
        encoding="utf-8"
    )

    unsafe = re.compile(
        r"_run_json\(\s*\[\s*['\"]iam['\"]\s*,\s*['\"]v2['\"]\s*,\s*"
        r"['\"]access-key['\"]\s*,\s*['\"]list['\"]",
        re.DOTALL,
    )
    assert unsafe.search(source) is None
    assert "_ACCESS_KEY_LIST_JSONPATH" in source


def test_canary_secrets_are_absent_from_public_and_generated_docs() -> None:
    assert "NPA_CANARY_SECRET_" not in _public_text()
