from __future__ import annotations

from pathlib import Path

from npa.cli.agent_source_embed import embedded_module_source


def test_embedded_module_source_strips_multiline_docstring_and_future_import(
    tmp_path: Path,
) -> None:
    source = tmp_path / "embedded.py"
    source.write_text(
        '''\n"""A module docstring\nthat occupies several lines.\n"""\n\nfrom __future__ import (\n    annotations,\n)\n\nVALUE: int = 7\n'''.lstrip(),
        encoding="utf-8",
    )

    embedded = embedded_module_source(source)

    assert "__future__" not in embedded
    assert "module docstring" not in embedded
    assert "VALUE: int = 7" in embedded
    compile("def wrapper():\n" + "".join(f"    {line}" for line in embedded.splitlines(True)), "<embedded>", "exec")


def test_embedded_module_source_keeps_non_prologue_strings(tmp_path: Path) -> None:
    source = tmp_path / "embedded.py"
    source.write_text('VALUE = "keep me"\n', encoding="utf-8")

    assert embedded_module_source(source) == 'VALUE = "keep me"\n'
