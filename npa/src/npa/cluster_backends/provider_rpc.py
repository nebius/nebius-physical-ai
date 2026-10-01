"""Supply missing provider RPC deadlines in the owned materialized recipe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import stat
from pathlib import Path
import tempfile
from typing import Any

_FIELDS = frozenset({"timeout", "per_retry_timeout", "auth_timeout"})


@dataclass
class _Provider:
    path: Path
    attributes: set[str]
    insertion: int | None = None
    opening: int | None = None
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

    parser = getattr(hcl2, "parses_to_tree", None)
    if not callable(parser):
        raise ImportError("Terraform parser API is unavailable")
    try:
        body = parser(text).children[0]
    except LarkError:
        raise ValueError(
            f"Unsupported or invalid Terraform HCL configuration: {path.name}"
        ) from None
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
            opening = next(
                child
                for child in block.children
                if getattr(child, "type", "") == "LBRACE"
            )
            providers.append(
                _Provider(path, attributes, block.meta.end_pos - 1, opening.end_pos)
            )
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
            os.fchmod(output.fileno(), stat.S_IMODE(path.stat().st_mode))
            output.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _materialized_text(
    provider: _Provider,
    original: str,
    defaults: dict[str, str],
    documents: dict[Path, Any],
) -> str:
    if not defaults:
        return original
    if provider.json_block is not None:
        document = documents[provider.path]
        providers = document["provider"]
        entries = providers["nebius"]
        merged = [
            {**entry, **defaults} if entry is provider.json_block else entry
            for entry in ([entries] if isinstance(entries, dict) else entries)
        ]
        rewritten = {
            **providers,
            "nebius": merged[0] if isinstance(entries, dict) else merged,
        }
        return json.dumps({**document, "provider": rewritten}, indent=2) + "\n"
    assert provider.insertion is not None and provider.opening is not None
    insertion = "\n" + "".join(
        f'  {key} = "{value}"\n' for key, value in defaults.items()
    )
    return (
        original[: provider.opening]
        + "\n"
        + original[provider.opening : provider.insertion]
        + insertion
        + original[provider.insertion :]
    )


def _advisory(timeout_minutes: int, reason: str) -> dict[str, Any]:
    return {
        "status": "advisory",
        "reason_code": reason,
        "apply_timeout_minutes": timeout_minutes,
        "inserted_defaults": {},
        "configuration_changed": False,
    }


def _configure(workdir: Path, timeout_minutes: int) -> dict[str, Any]:
    if workdir.is_symlink():
        return _advisory(timeout_minutes, "unsafe_recipe_path")
    provider, attributes, texts, documents = _inspect(workdir)
    defaults = {key: f"{timeout_minutes}m" for key in sorted(_FIELDS - attributes)}
    original = texts[provider.path]
    changed = _materialized_text(provider, original, defaults, documents)
    if any(_read(path) != text for path, text in texts.items()):
        return _advisory(timeout_minutes, "configuration_changed_during_inspection")
    if changed != original:
        _replace(provider.path, changed)
    return {
        "status": "configured" if changed != original else "unchanged",
        "apply_timeout_minutes": timeout_minutes,
        "inserted_defaults": defaults,
        "preserved_operator_fields": sorted(_FIELDS & attributes),
        "source_sha256": {path.name: _digest(text) for path, text in texts.items()},
        "materialized_provider_sha256": _digest(changed),
    }


def configure_provider_rpc_deadlines(
    workdir: Path, timeout_minutes: int
) -> dict[str, Any]:
    """Supply advisory RPC defaults without blocking an upstream recipe.

    Args:
        workdir: Owned materialized Terraform root, before provenance freezes.
        timeout_minutes: Existing positive NPA apply subprocess budget in minutes.
    Returns:
        Sidecar receipt describing atomic defaults or a sanitized advisory reason.
        Unsupported inputs remain byte-identical for Terraform's own validation.
    Raises:
        ValueError: The requested apply timeout is invalid.
    """
    if type(timeout_minutes) is not int or timeout_minutes <= 0:
        raise ValueError("NPA apply timeout must be a positive number of minutes")
    try:
        return _configure(workdir, timeout_minutes)
    except OSError:
        return _advisory(timeout_minutes, "recipe_io_unavailable")
    except (ValueError, TypeError, IndexError, KeyError, StopIteration, AttributeError):
        return _advisory(timeout_minutes, "unsupported_recipe_shape")
    except ImportError:
        return _advisory(timeout_minutes, "parser_unavailable")
    except Exception:
        return _advisory(timeout_minutes, "deadline_configuration_unavailable")
