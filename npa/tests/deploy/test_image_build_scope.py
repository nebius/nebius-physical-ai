"""Exercise public build selection against real committed source changes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from npa.deploy.image_build_scope import (
    _dockerfile_inputs,
    select_public_image_builds,
)

_CONTRACT = "npa/docker/workbench/packaging-contract.yaml"
_CATALOG = "npa/docker/workbench/blackwell-dc-images.json"
_PUBLIC = ["lerobot", "sam2", "sam3"]


def _write(root: Path, path: str, content: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _commit(root: Path) -> str:
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture inputs")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    entries = {
        tool: {"dockerfile": f"{tool}/Dockerfile", "redistribution": "public"}
        for tool in _PUBLIC
    }
    entries["sonic"] = {
        "dockerfile": "sonic/Dockerfile",
        "redistribution": "restricted",
    }
    _write(tmp_path, _CONTRACT, yaml.safe_dump({"version": 1, "images": entries}))
    _write(
        tmp_path, _CATALOG, json.dumps({"version": 1, "images": list(entries.values())})
    )
    for tool in entries:
        recipe = f"FROM scratch\nCOPY docker/workbench/{tool}/runtime.py /runtime.py\n"
        if tool == "lerobot":
            recipe += "COPY pyproject.toml /app/\nCOPY src/npa /app/src/npa\n"
        _write(tmp_path, f"npa/docker/workbench/{tool}/Dockerfile", recipe)
        _write(tmp_path, f"npa/docker/workbench/{tool}/runtime.py", "# Original\n")
    _write(tmp_path, "npa/pyproject.toml", "# Original\n")
    _write(tmp_path, "npa/src/npa/workbench/runtime.py", "# Original\n")
    return tmp_path, _commit(tmp_path)


def _selection(repository: tuple[Path, str]) -> list[str]:
    root, before = repository
    return select_public_image_builds(root, before, _commit(root))


def test_isolated_tool_change_selects_one_image(repository):
    root, _ = repository
    _write(root, "npa/docker/workbench/sam3/runtime.py", "# Updated\n")
    assert _selection(repository) == ["sam3"]


def test_new_tool_and_shared_source_select_consumers_not_the_whole_inventory(
    repository,
):
    root, before = repository
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    del contract["images"]["sam3"]
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    before = _commit(root)
    contract["images"]["sam3"] = {
        "dockerfile": "sam3/Dockerfile",
        "redistribution": "public",
    }
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    _write(root, "npa/src/npa/deploy/images.py", "# Register the new tool\n")
    assert _selection((root, before)) == ["lerobot", "sam3"]


@pytest.mark.parametrize(
    "path", ["npa/pyproject.toml", "npa/src/npa/workbench/runtime.py"]
)
def test_shared_package_input_selects_the_image_that_copies_it(repository, path):
    root, _ = repository
    _write(root, path, "# Updated shared source\n")
    assert _selection(repository) == ["lerobot"]


def test_packaging_change_is_scoped_to_its_entry(repository):
    root, _ = repository
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    contract["images"]["sam3"]["notes"] = "Updated contract"
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    assert _selection(repository) == ["sam3"]


def test_catalog_change_is_scoped_to_its_dockerfile(repository):
    root, _ = repository
    catalog = json.loads((root / _CATALOG).read_text())
    catalog["images"][2]["validation"] = "pending-gpu"
    _write(root, _CATALOG, json.dumps(catalog))
    assert _selection(repository) == ["sam3"]


def test_metadata_formatting_only_does_not_rebuild(repository):
    root, _ = repository
    _write(
        root, _CATALOG, json.dumps(json.loads((root / _CATALOG).read_text()), indent=2)
    )
    assert _selection(repository) == []


@pytest.mark.parametrize(
    "field", ["as_of", "registry_note", "verdicts", "validation_states"]
)
def test_catalog_description_changes_preserve_recipe_scope(repository, field):
    root, _ = repository
    catalog = json.loads((root / _CATALOG).read_text())
    glossary = field in {"verdicts", "validation_states"}
    catalog[field] = {"pending-build": "Original wording"} if glossary else "Original"
    _write(root, _CATALOG, json.dumps(catalog))
    before = _commit(root)
    catalog[field] = {"pending-build": "Clarified wording"} if glossary else "Updated"
    _write(root, _CATALOG, json.dumps(catalog))
    _write(root, "npa/docker/workbench/sam3/runtime.py", "# Updated\n")
    assert _selection((root, before)) == ["sam3"]


def test_catalog_descriptions_still_rebuild_images_that_copy_catalog_bytes(repository):
    root, _ = repository
    path = "npa/docker/workbench/sam2/Dockerfile"
    _write(
        root,
        path,
        "FROM scratch\nCOPY docker/workbench/blackwell-dc-images.json /catalog.json\n",
    )
    before = _commit(root)
    catalog = json.loads((root / _CATALOG).read_text())
    catalog["as_of"] = "2026-09-25T00:00:00Z"
    _write(root, _CATALOG, json.dumps(catalog))
    assert _selection((root, before)) == ["sam2"]


@pytest.mark.parametrize("value", [{"new-state": "New meaning"}, {"pending-build": []}])
def test_changed_catalog_states_or_invalid_glossaries_keep_full_rebuild(
    repository, value
):
    root, _ = repository
    catalog = json.loads((root / _CATALOG).read_text())
    catalog["validation_states"] = {"pending-build": "Original wording"}
    _write(root, _CATALOG, json.dumps(catalog))
    before = _commit(root)
    catalog["validation_states"] = value
    _write(root, _CATALOG, json.dumps(catalog))
    assert _selection((root, before)) == _PUBLIC


@pytest.mark.parametrize("metadata", [_CONTRACT, _CATALOG])
def test_global_metadata_changes_keep_full_rebuild(repository, metadata):
    root, _ = repository
    content = yaml.safe_load((root / metadata).read_text())
    content["version"] = 2
    _write(root, metadata, json.dumps(content))
    assert _selection(repository) == _PUBLIC


@pytest.mark.parametrize(
    "path", ["npa/docker/workbench/common/new-policy.sh", "npa/.dockerignore"]
)
def test_unknown_shared_inputs_keep_full_rebuild(repository, path):
    root, _ = repository
    _write(root, path, "# New shared input\n")
    assert _selection(repository) == _PUBLIC


@pytest.mark.parametrize("before", ["", "0" * 40, "f" * 40, "--invalid"])
def test_missing_or_invalid_history_rebuilds_all_public_images(repository, before):
    root, head = repository
    assert select_public_image_builds(root, before, head) == _PUBLIC


def test_restricted_image_is_never_selected(repository):
    root, _ = repository
    _write(
        root, "npa/docker/workbench/sonic/runtime.py", "# Updated restricted input\n"
    )
    assert "sonic" not in _selection(repository)


def test_deleted_and_renamed_sources_still_select_the_consumer(repository):
    root, _ = repository
    (root / "npa/src/npa/workbench/runtime.py").rename(
        root / "npa/src/npa/workbench/renamed.py"
    )
    assert _selection(repository) == ["lerobot"]


def test_staged_workflow_catalog_selects_package_consumers(repository):
    root, _ = repository
    _write(root, "workflows/main/new.yaml", "# New staged workflow\n")
    assert _selection(repository) == ["lerobot"]


def test_non_image_docs_select_no_images(repository):
    root, _ = repository
    _write(root, "docs/guide.md", "Documentation only\n")
    assert _selection(repository) == []


@pytest.mark.parametrize(
    "copy",
    [
        "COPY --chmod=0444 docker/workbench/common/config.json /config.json",
        'COPY ["docker/workbench/common/config.json", "/config.json"]',
        "COPY docker/workbench/common/*.json /config/",
        "COPY docker/workbench/* /config/",
        "COPY\tdocker/workbench/common/config.json /config.json",
        "COPY --chown=app:app \\\n# Preserve Dockerfile continuation comments\n docker/workbench/common/config.json /config.json",
    ],
)
def test_shared_copy_sources_are_tracked(repository, copy):
    root, _ = repository
    _write(root, "npa/docker/workbench/sam3/Dockerfile", "FROM scratch\n" + copy + "\n")
    _write(root, "npa/docker/workbench/common/config.json", "{}")
    before = _commit(root)
    _write(root, "npa/docker/workbench/common/config.json", '{"changed": true}')
    assert _selection((root, before)) == ["sam3"]


@pytest.mark.parametrize(
    "recipe",
    [
        "FROM scratch\nCOPY ${SOURCE} /app\n",
        "FROM scratch\nRUN --mount=type=bind,source=${SOURCE},target=/src build\n",
        'FROM scratch\nRUN --mount=source="some path",target=/src build\n',
        "FROM scratch\nRUN --mount=source=src,src=other,target=/src build\n",
        "FROM scratch\nRUN --mount=source=src,source=other,target=/src build\n",
        "FROM scratch\nRUN --mount=type=${TYPE},target=/src build\n",
        "FROM scratch\nRUN --mount=source=../outside,target=/src build\n",
        "FROM scratch\nCOPY https://example.invalid/input /app\n",
        "FROM scratch\nADD ${SOURCE} /app\n",
        "# escape=`\nFROM scratch\nCOPY src `/app\n",
        "FROM scratch\nCOPY <<EOF /config\nembedded\nEOF\n",
    ],
)
def test_unrecognized_build_inputs_fall_back_to_all(repository, recipe):
    root, _ = repository
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe)
    assert _selection(repository) == _PUBLIC


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_remote_add_keeps_local_inputs_and_recipe_changes_scoped(repository, scheme):
    """Track local files and recipe edits alongside remote downloads.

    Args:
        repository: Committed image-input fixture.
        scheme: Explicit remote URL scheme.
    Returns:
        None.
    Raises:
        AssertionError: Build selection misses inputs or selects unrelated images.
    """
    root, _ = repository
    recipe = (
        "FROM scratch\n"
        f"ADD --checksum=sha256:${{DIGEST}} {scheme}://example.invalid/${{VERSION}} /remote\n"
        "COPY docker/workbench/common/config.json /config.json\n"
    )
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe)
    _write(root, "npa/docker/workbench/common/config.json", "{}")
    before = _commit(root)
    _write(root, "docs/guide.md", "Prose only\n")
    assert _selection((root, before)) == []
    _write(root, "npa/docker/workbench/common/config.json", '{"changed": true}')
    assert _selection((root, before)) == ["sam3"]
    before = _git(root, "rev-parse", "HEAD")
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe + "ENV UPDATED=1\n")
    assert _selection((root, before)) == ["sam3"]


@pytest.mark.parametrize(
    ("mount", "package_consumers"),
    [
        ("type=bind,source=src,target=/src,readonly", ["lerobot"]),
        ("source=src,target=/src", ["lerobot"]),
        ("src=src,target=/src", ["lerobot"]),
        ("TyPe=BIND,SoUrCe=src,target=/src", ["lerobot"]),
        ("Src=src,target=/src", ["lerobot"]),
        ("type=BIND,source=src,target=/src", ["lerobot"]),
        ("type=bind,target=/src", ["lerobot", "sam3"]),
    ],
)
def test_bind_mounts_track_local_context_inputs(repository, mount, package_consumers):
    """Rebuild consumers of a static bind mount when its source changes.

    Args:
        repository: Committed image-input fixture.
        mount: Supported bind-mount syntax.
        package_consumers: Images affected by an unrelated package input.
    Returns:
        None.
    Raises:
        AssertionError: Bind-mount dependencies are selected incorrectly.
    """
    root, _ = repository
    _write(
        root,
        "npa/docker/workbench/sam3/Dockerfile",
        f"FROM scratch\nRUN --mount={mount} build\n",
    )
    before = _commit(root)
    _write(root, "docs/guide.md", "Prose only\n")
    assert _selection((root, before)) == []
    _write(root, "npa/pyproject.toml", "# Changed package metadata\n")
    assert _selection((root, before)) == package_consumers
    _write(root, "npa/src/npa/workbench/runtime.py", "# Updated\n")
    assert _selection((root, before)) == ["lerobot", "sam3"]


@pytest.mark.parametrize(
    "mount",
    [
        "src=other,Source=src,target=/src",
        "source=other,SOURCE=src,target=/src",
        "type=cache,TYPE=bind,source=src,target=/src",
    ],
)
def test_case_insensitive_mount_aliases_restore_full_rebuild(repository, mount):
    """Keep ambiguous mount declarations from hiding changed consumers.

    Args:
        repository: Committed image-input fixture.
        mount: Case-insensitive duplicate fields or source aliases.
    Returns:
        None.
    Raises:
        AssertionError: An ambiguous mount silently omits a consumer.
    """
    root, _ = repository
    _write(
        root,
        "npa/docker/workbench/sam3/Dockerfile",
        f"FROM ubuntu:22.04\nRUN --mount={mount} cat /src/npa/workbench/runtime.py\n",
    )
    _write(root, "npa/other/placeholder", "Unused\n")
    before = _commit(root)
    _write(root, "npa/src/npa/workbench/runtime.py", "# Changed consumed source\n")
    _write(root, "npa/docker/workbench/sam2/runtime.py", "# Trigger publication\n")
    assert _selection((root, before)) == _PUBLIC


@pytest.mark.parametrize(
    ("source", "lookalike"),
    [
        ("config[0].json", "config0.json"),
        ("config?.json", "config1.json"),
        ("config*.json", "config-extra.json"),
        ("folder[0]/config.json", "folder0/config.json"),
    ],
)
def test_bind_sources_match_literal_paths_without_selecting_lookalikes(
    repository, source, lookalike
):
    """Track literal mount paths while excluding files that only match as globs.

    Args:
        repository: Committed image-input fixture.
        source: Literal bind source beneath the shared directory.
        lookalike: Unconsumed path that matches the source as a glob.
    Returns:
        None.
    Raises:
        AssertionError: Selection misses a consumer or treats its source as a glob.
    """
    root, _ = repository
    recipe = (
        "FROM ubuntu:22.04\n"
        f"RUN --mount=source=shared/{source},target=/config cat /config\n"
    )
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe)
    _write(root, f"npa/shared/{source}", "{}\n")
    _write(root, f"npa/shared/{lookalike}", "{}\n")
    before = _commit(root)
    _write(root, f"npa/shared/{source}", '{"changed": true}\n')
    _write(root, "npa/docker/workbench/sam2/runtime.py", "# Publish source change\n")
    assert _selection((root, before)) == ["sam2", "sam3"]
    before = _git(root, "rev-parse", "HEAD")
    _write(root, f"npa/shared/{lookalike}", '{"unconsumed": true}\n')
    _write(root, "npa/docker/workbench/sam2/runtime.py", "# Publish unrelated change\n")
    assert _selection((root, before)) == ["sam2"]


def test_non_context_mounts_do_not_hide_local_dependencies(repository):
    """Retain local stage inputs alongside stage and cache mounts.

    Args:
        repository: Committed image-input fixture.
    Returns:
        None.
    Raises:
        AssertionError: Stage inputs disappear from build selection.
    """
    root, _ = repository
    recipe = (
        "FROM scratch AS source\nCOPY src/npa /source\nFROM scratch\n"
        "RUN --mount=from=source,source=/source,target=/app build\n"
        "RUN --mount=type=cache,target=/cache build\n"
    )
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe)
    before = _commit(root)
    _write(root, "npa/src/npa/workbench/runtime.py", "# Updated\n")
    assert _selection((root, before)) == ["lerobot", "sam3"]


def test_robomimic_sources_use_its_actual_nested_build_context(repository):
    """Resolve robomimic inputs inside the publisher's nested build context.

    Args:
        repository: Committed image-input fixture.
    Returns:
        None.
    Raises:
        AssertionError: Dependencies escape or ignore the actual build context.
    """
    root, _ = repository
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    contract["images"]["robomimic"] = {
        "dockerfile": "robomimic/Dockerfile",
        "redistribution": "public",
    }
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    _write(
        root,
        "npa/docker/workbench/robomimic/Dockerfile",
        "FROM scratch\nCOPY . /app\n"
        "RUN --mount=source=build-inputs,target=/inputs,readonly build\n",
    )
    before = _commit(root)
    _write(root, "npa/src/npa/workbench/runtime.py", "# Updated\n")
    assert _selection((root, before)) == ["lerobot"]
    assert _dockerfile_inputs(
        (root / "npa/docker/workbench/robomimic/Dockerfile").read_text(),
        "npa/docker/workbench/robomimic",
    ) == {
        "npa/docker/workbench/robomimic",
        "npa/docker/workbench/robomimic/build-inputs",
    }
    before = _git(root, "rev-parse", "HEAD")
    _write(root, "npa/docker/workbench/robomimic/build-inputs/source.tar", "Updated\n")
    assert _selection((root, before)) == ["robomimic"]


def test_shipped_public_recipes_do_not_force_unrelated_full_rebuilds():
    """Require dependency parsing for every committed eligible public recipe.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: A shipped recipe would restore unrelated image builds.
    """
    from npa.deploy.image_build_scope import _image_inputs
    from npa.deploy.images import (
        CONTAINER_IMAGE_NAMES,
        DEVELOPMENT_BUILD_QUARANTINE_TOOLS,
    )

    root = Path(__file__).resolve().parents[3]
    head = _git(root, "rev-parse", "HEAD")
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    eligible = CONTAINER_IMAGE_NAMES.keys() | DEVELOPMENT_BUILD_QUARANTINE_TOOLS
    for tool, entry in contract["images"].items():
        if entry.get("redistribution") == "public" and tool in eligible:
            _image_inputs(root, head, entry["dockerfile"])


def test_stage_copy_does_not_hide_original_context_dependencies(repository):
    root, _ = repository
    recipe = "FROM scratch AS source\nCOPY src/npa /source\nFROM scratch\nCOPY --from=source /source /app\n"
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe)
    before = _commit(root)
    _write(root, "npa/src/npa/workbench/runtime.py", "# Updated\n")
    assert _selection((root, before)) == ["lerobot", "sam3"]


def test_symbolic_link_dependencies_use_full_fallback(repository):
    root, _ = repository
    (root / "npa/src/npa/shared.py").symlink_to("../../pyproject.toml")
    assert _selection(repository) == _PUBLIC


def test_invalid_current_contract_fails_instead_of_selecting_nothing(repository):
    root, before = repository
    _write(root, _CONTRACT, "images: []\n")
    with pytest.raises(ValueError, match="packaging contract"):
        select_public_image_builds(root, before, _commit(root))


def test_invalid_head_cannot_be_used_as_a_git_option(repository):
    root, before = repository
    with pytest.raises(ValueError, match="full source SHA"):
        select_public_image_builds(root, before, "--help")


@pytest.mark.parametrize(
    ("event", "explicit", "expected"),
    [
        ("push", "", ["sam3"]),
        ("schedule", "", _PUBLIC),
        ("workflow_dispatch", "sam2", ["sam2"]),
        ("workflow_dispatch", "ncore", None),
    ],
)
def test_workflow_resolver_uses_scope_and_preserves_schedule_and_dispatch(
    repository, event, explicit, expected
):
    root, before = repository
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    contract["images"]["ncore"] = {
        "dockerfile": "ncore/Dockerfile",
        "redistribution": "public",
    }
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    _write(root, "npa/docker/workbench/ncore/Dockerfile", "FROM scratch\n")
    before = _commit(root)
    _write(root, "npa/docker/workbench/sam3/runtime.py", "# Updated\n")
    head = _commit(root)
    result = _resolve_workflow_plan(root, before, head, event, explicit)
    if expected is None:
        assert result.returncode != 0
        assert "NCore Actions publication is quarantined" in result.stderr
        assert "build_matrix=" not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert int(outputs["build_count"]) == len(expected)
    assert [entry["tool"] for entry in json.loads(outputs["build_matrix"])] == expected
    assert outputs["cleanup_count"] == "0"


def _resolve_workflow_plan(root, before, head, event, explicit):
    import os
    import sys

    workflow = (
        Path(__file__).resolve().parents[3]
        / ".github/workflows/publish-public-images.yml"
    )
    spec = yaml.safe_load(workflow.read_text())
    step = next(
        step for step in spec["jobs"]["resolve"]["steps"] if step.get("id") == "plan"
    )
    script = (
        step["run"].split("<<'PY' >> \"$GITHUB_OUTPUT\"\n", 1)[1].rsplit("\nPY", 1)[0]
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "TARGET": "ghcr.io/nebius/nebius-physical-ai",
            "DEVELOPMENT_SHA": head,
            "EVENT_BEFORE": before,
            "EVENT_NAME": event,
            "BUILD_TOOLS": explicit,
            "CLEANUP_TOOLS": "",
            "LEROBOT_VERSION": "",
        },
    )


def test_ncore_custom_assembly_inputs_select_ncore(repository):
    root, _ = repository
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    contract["images"]["ncore"] = {
        "dockerfile": "ncore/Dockerfile",
        "redistribution": "public",
    }
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    _write(root, "npa/docker/workbench/ncore/Dockerfile", "FROM scratch\n")
    _write(root, "npa/scripts/ncore_publication/build.py", "# Original\n")
    before = _commit(root)
    _write(root, "npa/scripts/ncore_publication/build.py", "# Updated assembler\n")
    assert _selection((root, before)) == ["ncore"]


def test_copy_through_symbolic_directory_uses_full_fallback(repository):
    root, _ = repository
    (root / "npa/linked").symlink_to("src/npa", target_is_directory=True)
    _write(
        root,
        "npa/docker/workbench/sam3/Dockerfile",
        "FROM scratch\nCOPY linked/workbench/runtime.py /runtime.py\n",
    )
    before = _commit(root)
    _write(root, "npa/src/npa/workbench/runtime.py", "# Updated target\n")
    assert _selection((root, before)) == _PUBLIC


def test_new_development_candidate_is_selected_without_release_registration(repository):
    root, before = repository
    contract = yaml.safe_load((root / _CONTRACT).read_text())
    contract["images"]["gymnasium-robotics"] = {
        "dockerfile": "gymnasium-robotics/Dockerfile",
        "redistribution": "public",
    }
    _write(root, _CONTRACT, yaml.safe_dump(contract))
    _write(root, "npa/docker/workbench/gymnasium-robotics/Dockerfile", "FROM scratch\n")
    assert _selection((root, before)) == ["gymnasium-robotics"]
    head = _git(root, "rev-parse", "HEAD")
    assert select_public_image_builds(root, "", head) == [
        "gymnasium-robotics",
        *_PUBLIC,
    ]
