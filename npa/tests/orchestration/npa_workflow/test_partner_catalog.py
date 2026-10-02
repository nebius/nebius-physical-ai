"""Partner workflows remain discoverable in source, packages, and worker uploads."""

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import blueprints
from npa.orchestration.npa_workflow.src_staging import staged_source_files
from npa.workflow_build import catalog_files, stage_catalog


SPEC = "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n"
PARTNER = Path("partners/acme/train.yaml")


def _write(path: Path, text: str = SPEC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def catalog(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "repo/npa"
    source = package.parent / "workflows"
    _write(package / "pyproject.toml", "[project]\nname='npa'\n")
    _write(package / "src/npa/__init__.py", "")
    _write(source / "main/main.yaml")
    _write(source / "testing/testing.yaml")
    _write(source / PARTNER)
    return package, source


@pytest.mark.parametrize("installed", [False, True])
def test_discovery_and_resolution_include_partner_catalog(
    catalog: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, installed: bool
) -> None:
    package, source = catalog
    if installed:
        stage_catalog(package)
        source = package / "src/npa/workflows"
    monkeypatch.setattr(
        blueprints, "_REPO_ROOT", package / "absent" if installed else package.parent
    )
    monkeypatch.setattr(blueprints, "_PACKAGE_ROOT", package / "src/npa")
    _write(source / "partners/acme/examples/excluded.yaml")
    _write(source / "partners/acme/config.yaml", "not_a_workflow: true\n")
    # Stale generated copies must not shadow an authoritative source catalog.
    _write(package / "src/npa/workflows/testing/train.yaml", "stale: true\n")

    specs = blueprints.iter_npa_workflow_specs()

    assert {path.name for path in specs} == {"main.yaml", "testing.yaml", "train.yaml"}
    assert blueprints.resolve_npa_workflow_spec("train.yaml") == source / PARTNER
    assert blueprints.resolve_npa_workflow_spec("excluded.yaml") is None


def test_build_catalog_preserves_partner_path_and_cleans_stale_copies(
    catalog: tuple[Path, Path],
) -> None:
    package, source = catalog
    destination = package / "src/npa/workflows"
    stale = _write(destination / "testing/train.yaml")
    retired = _write(destination / "partners/retired/old.yaml")
    _write(source / "partners/acme/examples/excluded.yaml")

    assert catalog_files(package)[PARTNER] == source / PARTNER
    assert stage_catalog(package) == 3
    assert (destination / PARTNER).read_bytes() == (source / PARTNER).read_bytes()
    assert not stale.exists()
    assert not retired.exists()
    assert not (destination / "partners/acme/examples/excluded.yaml").exists()


def test_build_catalog_falls_back_to_packaged_partner_files(
    catalog: tuple[Path, Path],
) -> None:
    package, source = catalog
    stage_catalog(package)
    source.rename(source.with_name("unavailable"))

    assert catalog_files(package)[PARTNER] == package / "src/npa/workflows" / PARTNER
    assert stage_catalog(package) == 3


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("relative", ["partners", "partners/acme"])
def test_build_rejects_linked_partner_directories(
    catalog: tuple[Path, Path], side: str, relative: str
) -> None:
    package, source = catalog
    root = source if side == "source" else package / "src/npa/workflows"
    linked = root / relative
    linked.parent.mkdir(parents=True, exist_ok=True)
    outside = package.parent / "external"
    if linked.exists():
        linked.rename(outside)
    else:
        outside.mkdir()
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        stage_catalog(package)


def test_worker_upload_includes_partner_yaml_and_omits_stale_or_private_files(
    catalog: tuple[Path, Path],
) -> None:
    package, source = catalog
    _write(package / "src/npa/workflows/testing/train.yaml", "stale: true\n")
    _write(package / "src/npa/workflows/partners/retired/old.yaml")
    _write(source / "partners/acme/credentials.yaml", "private_fixture: true\n")
    _write(source / "partners/acme/examples/excluded.yaml")
    _write(source / "partners/acme/README.md", "Operator documentation\n")
    outside = _write(package.parent / "external.yaml")
    (source / "partners/acme/linked.yaml").symlink_to(outside)

    files = staged_source_files(package)
    workflows = {
        path for path in files if path.parts[:3] == ("src", "npa", "workflows")
    }

    assert workflows == {
        Path("src/npa/workflows/main/main.yaml"),
        Path("src/npa/workflows/testing/testing.yaml"),
        Path("src/npa/workflows") / PARTNER,
    }
    assert files[Path("src/npa/workflows") / PARTNER] == source / PARTNER
