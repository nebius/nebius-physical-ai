"""Edit scoped JSON values without exposing their previous redacted contents."""

from __future__ import annotations

import json
import re

_SOURCE_EDIT_TOOLS = frozenset({"edit_file", "replace_json"})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON pointer editing rejects duplicate object keys")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("JSON pointer editing requires finite standard JSON values")


def _parts(pointer):
    if pointer == "":
        return []
    if not pointer.startswith("/") or re.search(r"~(?![01])", pointer):
        raise ValueError("json_pointer must be a valid RFC 6901 pointer")
    return [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]


def _key(parent, part):
    if isinstance(parent, dict) and part in parent:
        return part
    if isinstance(parent, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
        index = int(part)
        if index < len(parent):
            return index
    raise ValueError("json_pointer does not identify an existing value")


def _replace_json_text(content, pointer, old, new):
    root = json.loads(
        content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    parts = _parts(pointer)
    parent = None
    value = root
    for part in parts:
        parent = value
        key = _key(parent, part)
        value = parent[key]
    if not isinstance(value, str):
        raise ValueError("json_pointer must identify a string value")
    if not old or value.count(old) != 1:
        raise ValueError("old text must match exactly once in the selected JSON string")
    replacement = value.replace(old, new, 1)
    if parent is None:
        root = replacement
    else:
        parent[key] = replacement
    suffix = "\n" if content.endswith("\n") else ""
    return json.dumps(root, ensure_ascii=False, allow_nan=False, indent=2) + suffix


def _replace_json_value(content, pointer, replacement):
    root = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    parts = _parts(pointer)
    parent = None
    value = root
    for part in parts:
        parent = value
        key = _key(parent, part)
        value = parent[key]
    if parent is None:
        root = replacement
    else:
        parent[key] = replacement
    suffix = "\n" if content.endswith("\n") else ""
    return json.dumps(root, ensure_ascii=False, allow_nan=False, indent=2) + suffix
