"""Select public development image builds from committed recipe and source inputs."""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import shlex
import subprocess
from collections.abc import Collection
from pathlib import Path, PurePosixPath

import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    DEVELOPMENT_BUILD_QUARANTINE_TOOLS,
)

_LOG = logging.getLogger(__name__)
_WORKBENCH = "npa/docker/workbench/"
_CONTRACT = _WORKBENCH + "packaging-contract.yaml"
_CATALOG = _WORKBENCH + "blackwell-dc-images.json"
_SHA = re.compile(r"[0-9a-f]{40}")
_FULL_REBUILD_INPUTS = {"npa/.dockerignore", "npa/src/npa/workflow_build.py"}
# NCore assembles OCI bytes through a Python publisher as well as its Dockerfile.
_ASSEMBLY_INPUTS = {
    "ncore": {
        "npa/scripts/publish_ncore_oci.py",
        "npa/scripts/ncore_publication",
        "npa/src/npa/deploy/ncore*",
    }
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def _read(root: Path, revision: str, path: str) -> str:
    return _git(root, "show", f"{revision}:{path}")


def _contract(root: Path, revision: str) -> dict:
    contract = yaml.safe_load(_read(root, revision, _CONTRACT))
    if not isinstance(contract, dict) or not isinstance(contract.get("images"), dict):
        raise ValueError("Invalid image packaging contract")
    if not all(isinstance(entry, dict) for entry in contract["images"].values()):
        raise ValueError("Invalid image packaging entry")
    return contract


def _catalog_entries(document: dict) -> dict[str, dict]:
    if not isinstance(document, dict) or not isinstance(document.get("images"), list):
        raise ValueError("Invalid image catalog")
    entries = {}
    for entry in document["images"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("dockerfile"), str):
            raise ValueError("Invalid image catalog entry")
        if entry["dockerfile"] in entries:
            raise ValueError("Ambiguous image catalog entry")
        entries[entry["dockerfile"]] = entry
    return entries


def _changed_entries(before: dict, after: dict) -> set[str]:
    if {k: v for k, v in before.items() if k != "images"} != {
        k: v for k, v in after.items() if k != "images"
    }:
        raise ValueError("Shared image metadata changed")
    return {
        key
        for key in before["images"].keys() | after["images"].keys()
        if before["images"].get(key) != after["images"].get(key)
    }


def _metadata_tools(root: Path, before: str, head: str, paths: set[str]) -> set[str]:
    old, new = _contract(root, before), _contract(root, head)
    selected = _changed_entries(old, new) if _CONTRACT in paths else set()
    if _CATALOG not in paths:
        return selected
    documents = [
        json.loads(_read(root, revision, _CATALOG)) for revision in (before, head)
    ]
    catalogs = [dict(doc, images=_catalog_entries(doc)) for doc in documents]
    dockerfiles = _changed_entries(*catalogs)
    known = {entry.get("dockerfile") for entry in new["images"].values()}
    if dockerfiles - known:
        raise ValueError("Unmapped image catalog change")
    return selected | {
        tool
        for tool, entry in new["images"].items()
        if entry.get("dockerfile") in dockerfiles
    }


def _copy_sources(arguments: str) -> set[str]:
    while arguments.startswith("--"):
        flag, separator, arguments = arguments.partition(" ")
        if not separator or "=" not in flag and flag not in {"--link", "--parents"}:
            raise ValueError("Unsupported Dockerfile COPY option")
        if flag.startswith("--from="):
            return set()
        arguments = arguments.lstrip()
    words = (
        json.loads(arguments) if arguments.startswith("[") else shlex.split(arguments)
    )
    if not isinstance(words, list) or len(words) < 2:
        raise ValueError("Unsupported Dockerfile COPY operands")
    paths = set()
    for word in words[:-1]:
        if not isinstance(word, str) or any(
            marker in word for marker in ("$", "<<", "://")
        ):
            raise ValueError("Dynamic Dockerfile source")
        path = PurePosixPath(word)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Non-relative Dockerfile source")
        paths.add("npa" if str(path) == "." else "npa/" + str(path))
    return paths


def _dockerfile_inputs(text: str) -> set[str]:
    if re.search(r"^\s*#\s*escape\s*=\s*`", text, re.MULTILINE):
        raise ValueError("Unsupported Dockerfile escape directive")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    instructions = "\n".join(lines).replace("\\\n", " ")
    paths = set()
    for line in instructions.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        instruction, arguments = parts
        if instruction.upper() in {"COPY", "ADD"}:
            paths.update(_copy_sources(arguments.lstrip()))
        if instruction.upper() == "RUN":
            for mount in re.findall(r"--mount=([^\s]+)", arguments):
                fields = dict(
                    part.split("=", 1) for part in mount.split(",") if "=" in part
                )
                if fields.get("type", "bind") == "bind" and "from" not in fields:
                    raise ValueError("Unmapped build-context mount")
    return paths


def _matches(path: str, sources: Collection[str]) -> bool:
    ancestors = [path, *(str(parent) for parent in PurePosixPath(path).parents)]
    return any(
        fnmatch.fnmatchcase(parent, source)
        for source in sources
        for parent in ancestors
    )


def _source_paths(paths: set[str]) -> set[str]:
    # The publisher stages the authoring catalog into package data before building.
    return paths | {
        "npa/src/npa/" + path
        for path in paths
        if path.startswith("workflows/") and path.endswith(".yaml")
    }


def _image_inputs(root: Path, head: str, dockerfile: str) -> set[str]:
    path = _WORKBENCH + dockerfile
    sources = _dockerfile_inputs(_read(root, head, path))
    return sources | {path, path + ".dockerignore"}


def _select_changed(root: Path, before: str, head: str, entries: dict) -> set[str]:
    paths = set(
        _git(root, "diff", "--name-only", "--no-renames", "-z", before, head).split(
            "\0"
        )
    ) - {""}
    selected = _metadata_tools(root, before, head, paths)
    tree = _git(root, "ls-tree", "-r", "-z", head, "--", "npa")
    # A source can traverse a symlink or submodule before its literal COPY path.
    # Do not claim complete dependencies for contexts containing such indirection.
    if any(row.startswith(("120000 ", "160000 ")) for row in tree.split("\0")):
        raise ValueError("Indirect image build context")
    inputs = {
        key: _image_inputs(root, head, entry["dockerfile"])
        | _ASSEMBLY_INPUTS.get(key, set())
        for key, entry in entries.items()
    }
    sources = _source_paths(paths)
    for key, dependencies in inputs.items():
        directory = _WORKBENCH + entries[key]["dockerfile"].split("/", 1)[0] + "/"
        if any(
            path.startswith(directory) or _matches(path, dependencies)
            for path in sources
        ):
            selected.add(key)
    for path in paths - {_CONTRACT, _CATALOG}:
        if path in _FULL_REBUILD_INPUTS:
            return set(entries)
        if path.startswith(_WORKBENCH) and not any(
            path.startswith(_WORKBENCH + entry["dockerfile"].split("/", 1)[0] + "/")
            or _matches(path, inputs[key])
            for key, entry in entries.items()
        ):
            raise ValueError("Unmapped shared workbench input")
    return selected & entries.keys()


def select_public_image_builds(root: Path, before: str, head: str) -> list[str]:
    """Select affected eligible images, rebuilding all when dependency proof is incomplete.

    Args:
        root: Repository containing the committed image build inputs.
        before: Previous main SHA from the push event; missing history selects all.
        head: Full immutable source SHA to build.

    Returns:
        Sorted public development tools; an empty list means no images changed.

    Raises:
        ValueError: The head SHA or current packaging contract is invalid.
        OSError: Git cannot be executed.
        subprocess.CalledProcessError: Current packaging inputs cannot be read.
    """
    if not _SHA.fullmatch(head):
        raise ValueError("Image selection requires a full source SHA")
    entries = {
        key: entry
        for key, entry in _contract(root, head)["images"].items()
        if entry.get("redistribution") == "public"
        and (key in CONTAINER_IMAGE_NAMES or key in DEVELOPMENT_BUILD_QUARANTINE_TOOLS)
    }
    if not _SHA.fullmatch(before) or set(before) == {"0"}:
        return sorted(entries)
    try:
        return sorted(_select_changed(root, before, head, entries))
    except (
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
    ):
        _LOG.warning(
            "Image input scope is uncertain; rebuilding every eligible public image"
        )
        return sorted(entries)
