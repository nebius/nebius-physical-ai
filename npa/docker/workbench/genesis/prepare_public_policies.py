"""Retain native ACT factories without eagerly importing excluded policy families."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from importlib.util import cache_from_source
from pathlib import Path
import py_compile


SOURCE_HASHES = {
    "lerobot/policies/__init__.py": "dfdb10aa986c259e9ceb13dfa1bde6a4919c5ff0884bee19174afcebdeb4db42",
    "lerobot/policies/factory.py": "79caf2df45cce4d1b1551a156063f2098b5fa29ffc9e3ed62de60fc64d5f1c95",
}
UNSUPPORTED = "This public Genesis image supports native ACT only; use a qualified operator image for other policy families."
POLICY_MODIFICATION_NOTICE = (
    "# Modified by Nebius: restricts this public Genesis runtime to native ACT policies.\n"
)


def _statement(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def _act_branch(statement: ast.If) -> ast.If:
    while True:
        condition = ast.unparse(statement.test)
        if condition in {
            "name == 'act'",
            "policy_type == 'act'",
            "isinstance(policy_cfg, ACTConfig)",
        }:
            return statement
        if len(statement.orelse) != 1 or not isinstance(statement.orelse[0], ast.If):
            raise RuntimeError("reviewed factory has no unique ACT branch")
        statement = statement.orelse[0]


def _reject(condition: str) -> ast.If:
    return _statement(
        f"if {condition}:\n    raise NotImplementedError({UNSUPPORTED!r})"
    )


def _processor_body(function: ast.FunctionDef) -> list[ast.stmt]:
    docstring, pretrained, branches, returning = function.body
    if ast.unparse(pretrained.test) != "pretrained_path":
        raise RuntimeError("reviewed processor factory changed")
    # The first nested block is specific to unsupported GR00T processors.
    if ast.unparse(pretrained.body[0].test) != "isinstance(policy_cfg, GrootConfig)":
        raise RuntimeError("reviewed pretrained processor branch changed")
    pretrained.body.pop(0)
    return [
        docstring,
        _reject("not isinstance(policy_cfg, ACTConfig)"),
        pretrained,
        *_act_branch(branches).body,
        returning,
    ]


def _factory(tree: ast.Module) -> ast.Module:
    kept = []
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and any(
                alias.name.endswith("Config") and alias.name != "ACTConfig"
                for alias in node.names
            )
            and str(node.module).startswith("lerobot.policies.")
        ):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name in {"get_policy_class", "make_policy_config"}:
                docstring, branch = node.body
                act = _act_branch(branch)
                act.orelse = [_statement(f"raise NotImplementedError({UNSUPPORTED!r})")]
                node.body = [docstring, act]
            elif node.name == "make_pre_post_processors":
                node.body = _processor_body(node)
            elif node.name == "make_policy":
                node.body.insert(
                    1, _reject("cfg.type != 'act' or getattr(cfg, 'use_peft', False)")
                )
        kept.append(node)
    tree.body = kept
    return tree


def _replacement(name: str, original: bytes) -> bytes:
    text = original.decode()
    tree = ast.parse(text)
    header = text[: text.index("from ")]
    if name.endswith("__init__.py"):
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "act.configuration_act"
        ]
        if len(imports) != 1:
            raise RuntimeError("reviewed ACT registration changed")
        tree.body = [*imports, _statement("__all__ = ['ACTConfig']")]
    else:
        tree = _factory(tree)
    return (
        header
        + POLICY_MODIFICATION_NOTICE
        + ast.unparse(ast.fix_missing_locations(tree))
        + "\n"
    ).encode()


def _changes(root: Path) -> list[tuple[str, bytes, bytes]]:
    changes = []
    for name, expected in SOURCE_HASHES.items():
        original = (root / name).read_bytes()
        if hashlib.sha256(original).hexdigest() != expected:
            raise RuntimeError(f"unreviewed policy source: {name}")
        changes.append((name, original, _replacement(name, original)))
    return changes


def _record(root: Path, changes: list[tuple[str, bytes, bytes]]) -> str:
    rows = list(
        csv.reader(io.StringIO((root / "lerobot-0.4.4.dist-info/RECORD").read_text()))
    )
    for name, original, replacement in changes:
        selected = [row for row in rows if row[0] == name]
        bytecode = (
            Path(cache_from_source(str(root / name))).relative_to(root).as_posix()
        )
        if sum(row[0] == bytecode for row in rows) > 1:
            raise RuntimeError(f"duplicate policy bytecode RECORD: {bytecode}")
        original_hash = (
            base64.urlsafe_b64encode(hashlib.sha256(original).digest())
            .decode()
            .rstrip("=")
        )
        if len(selected) != 1 or selected[0][1:] != [
            "sha256=" + original_hash,
            str(len(original)),
        ]:
            raise RuntimeError(f"invalid original policy RECORD: {name}")
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(replacement).digest())
            .decode()
            .rstrip("=")
        )
        selected[0][1:] = ["sha256=" + digest, str(len(replacement))]
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def prepare(root: Path, *, check_only: bool) -> list[dict]:
    changes = _changes(root)
    record = _record(root, changes)
    if not check_only:
        for name, _, replacement in changes:
            (root / name).write_bytes(replacement)
            py_compile.compile(str(root / name), doraise=True)
        record = _bytecode_records(root, changes, record)
        (root / "lerobot-0.4.4.dist-info/RECORD").write_text(record)
    return [
        {
            "path": name,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "public_sha256": hashlib.sha256(replacement).hexdigest(),
            "record_updated": not check_only,
        }
        for name, original, replacement in changes
    ]


def _bytecode_records(
    root: Path, changes: list[tuple[str, bytes, bytes]], record: str
) -> str:
    rows = list(csv.reader(io.StringIO(record)))
    for name, _, _ in changes:
        path = Path(cache_from_source(str(root / name)))
        relative = path.relative_to(root).as_posix()
        matching = [row for row in rows if row[0] == relative]
        if len(matching) > 1:
            raise RuntimeError(f"duplicate policy bytecode RECORD: {relative}")
        content = path.read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode()
            .rstrip("=")
        )
        row = [relative, "sha256=" + digest, str(len(content))]
        if matching:
            matching[0][:] = row
        else:
            rows.append(row)
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.site_packages, check_only=arguments.check_only)))


if __name__ == "__main__":
    _main()
