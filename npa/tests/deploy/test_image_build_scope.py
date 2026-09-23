"""Exercise public build selection against real committed source changes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from npa.deploy.image_build_scope import select_public_image_builds

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
        "FROM scratch\nRUN --mount=type=bind,source=src,target=/src build\n",
        "FROM scratch\nRUN --mount=source=src,target=/src build\n",
        "# escape=`\nFROM scratch\nCOPY src `/app\n",
        "FROM scratch\nCOPY <<EOF /config\nembedded\nEOF\n",
    ],
)
def test_unrecognized_build_inputs_fall_back_to_all(repository, recipe):
    root, _ = repository
    _write(root, "npa/docker/workbench/sam3/Dockerfile", recipe)
    assert _selection(repository) == _PUBLIC


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
    ],
)
def test_workflow_resolver_uses_scope_and_preserves_schedule_and_dispatch(
    repository, event, explicit, expected
):
    import os
    import sys

    root, before = repository
    _write(root, "npa/docker/workbench/sam3/runtime.py", "# Updated\n")
    head = _commit(root)
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
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
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
    outputs = dict(line.split("=", 1) for line in result.splitlines())
    assert int(outputs["build_count"]) == len(expected)
    assert [entry["tool"] for entry in json.loads(outputs["build_matrix"])] == expected
    assert outputs["cleanup_count"] == "0"


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
