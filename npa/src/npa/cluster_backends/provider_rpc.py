"""Supply missing provider RPC deadlines in the owned materialized recipe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

_FIELDS = frozenset({"timeout", "per_retry_timeout", "auth_timeout"})


@dataclass
class _Provider:
    path: Path
    attributes: set[str]
    insertion: int | None = None
    json_block: dict[str, Any] | None = None


def _read(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("RPC defaults require regular nonsymlinked Terraform inputs")
    return path.read_text()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hcl_providers(path: Path, text: str) -> list[_Provider]:
    import hcl2
    from lark import Tree
    from lark.exceptions import LarkError

    try:
        body = hcl2.parses_to_tree(text).children[0]
    except LarkError:
        raise ValueError(f"Invalid Terraform HCL configuration: {path.name}") from None
    providers = []
    for block in body.children:
        if not isinstance(block, Tree) or block.data != "block":
            continue
        if block.children[0].children[0] != "provider":
            continue
        label = block.children[1]
        raw = text[label.meta.start_pos : label.meta.end_pos]
        if (json.loads(raw) if raw.startswith('"') else raw) != "nebius":
            continue
        contents = next(
            child for child in block.children if getattr(child, "data", "") == "body"
        )
        attributes = {
            str(child.children[0].children[0])
            for child in contents.children
            if getattr(child, "data", "") == "attribute"
        }
        if "alias" not in attributes:
            providers.append(_Provider(path, attributes, block.meta.end_pos - 1))
    return providers


def _json_providers(path: Path, text: str) -> tuple[list[_Provider], dict[str, Any]]:
    try:
        configuration = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid Terraform JSON configuration: {path.name}") from None
    if not isinstance(configuration, dict):
        raise ValueError("Terraform JSON configuration must be an object")
    providers = configuration.get("provider", {})
    if not isinstance(providers, dict):
        raise ValueError("Terraform JSON provider configuration must be an object")
    entries = providers.get("nebius", [])
    entries = [entries] if isinstance(entries, dict) else entries
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise ValueError("Terraform JSON Nebius provider blocks must be objects")
    return [
        _Provider(path, set(entry), json_block=entry)
        for entry in entries
        if "alias" not in entry
    ], configuration


def _is_override(path: Path) -> bool:
    stem = path.name.removesuffix(".json").removesuffix(".tf")
    return stem == "override" or stem.endswith("_override")


def _inspect(
    workdir: Path,
) -> tuple[_Provider, set[str], dict[Path, str], dict[Path, Any]]:
    texts, documents, ordinary, attributes = {}, {}, [], set()
    for path in sorted(set(workdir.glob("*.tf")) | set(workdir.glob("*.tf.json"))):
        texts[path] = _read(path)
        if path.name.endswith(".tf.json"):
            blocks, documents[path] = _json_providers(path, texts[path])
        else:
            blocks = _hcl_providers(path, texts[path])
        for block in blocks:
            attributes.update(block.attributes)
            if not _is_override(path):
                ordinary.append(block)
    if len(ordinary) != 1:
        raise ValueError(
            "RPC defaults require exactly one unaliased Nebius provider block"
        )
    return ordinary[0], attributes, texts, documents


def _replace(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".npa-rpc-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_provider_rpc_deadlines(
    workdir: Path, timeout_minutes: int
) -> dict[str, Any]:
    """Derive omitted SDK request deadlines from the existing NPA apply budget.

    Args:
        workdir: Owned materialized Terraform root, before apply provenance freezes.
        timeout_minutes: Existing NPA apply subprocess budget, in minutes.
    Returns:
        Receipt binding the materialized change and preserved operator settings.
    Raises:
        ValueError: Inputs are invalid, ambiguous, or changed during inspection.
        OSError: The owned recipe cannot be inspected or written.
    """
    if type(timeout_minutes) is not int or timeout_minutes <= 0:
        raise ValueError("NPA apply timeout must be a positive number of minutes")
    provider, attributes, texts, documents = _inspect(workdir)
    defaults = {key: f"{timeout_minutes}m" for key in sorted(_FIELDS - attributes)}
    original = texts[provider.path]
    changed = original
    if defaults and provider.json_block is not None:
        provider.json_block.update(defaults)
        changed = json.dumps(documents[provider.path], indent=2) + "\n"
    elif defaults:
        assert provider.insertion is not None
        insertion = "\n" + "".join(
            f'  {key} = "{value}"\n' for key, value in defaults.items()
        )
        changed = (
            original[: provider.insertion] + insertion + original[provider.insertion :]
        )
    if any(_read(path) != text for path, text in texts.items()):
        raise ValueError(
            "Terraform configuration changed during RPC deadline inspection"
        )
    if changed != original:
        _replace(provider.path, changed)
    return {
        "apply_timeout_minutes": timeout_minutes,
        "inserted_defaults": defaults,
        "preserved_operator_fields": sorted(_FIELDS & attributes),
        "source_sha256": {path.name: _digest(text) for path, text in texts.items()},
        "materialized_provider_sha256": _digest(changed),
    }
