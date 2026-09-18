"""Hatch hook including the repository workflow catalog in distributions.

The canonical source is outside this Python project's root. Source distributions,
staged worker sources, and container build contexts carry the same catalog under
``src/npa/workflows`` so they can build without a sibling repository checkout.
"""

from pathlib import Path
import shutil


def catalog_directories(catalog: Path) -> tuple[Path, ...]:
    """List main, testing, and individual partner workflow directories.

    Args:
        catalog: Repository or packaged workflow catalog root.
    Returns:
        Main/testing paths plus existing immediate partner directories.
    Raises:
        ValueError: A catalog directory is a symbolic link.
    """
    partners = catalog / "partners"
    if catalog.is_symlink() or partners.is_symlink():
        raise ValueError("Workflow catalog directories must not be symbolic links")
    directories = [catalog / "main", catalog / "testing"]
    if partners.is_dir():
        directories.extend(
            path for path in sorted(partners.iterdir())
            if path.is_dir() or path.is_symlink()
        )
    if any(path.is_symlink() for path in directories):
        raise ValueError("Workflow catalog directories must not be symbolic links")
    return tuple(directories)


def catalog_files(root: Path) -> dict[Path, Path]:
    """Map tier-relative YAML paths to the complete available source catalog."""

    for catalog in (root.parent / "workflows", root / "src/npa/workflows"):
        directories = catalog_directories(catalog)
        if not all((catalog / tier).is_dir() for tier in ("main", "testing")):
            continue
        return {
            path.relative_to(catalog): path
            for directory in directories
            for path in sorted(directory.glob("*.yaml"))
            if path.is_file() and not path.is_symlink()
        }
    return {}


def stage_catalog(package_root: Path) -> int:
    """Mirror catalog YAMLs into the generated, narrow container build context."""

    root = package_root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src/npa").is_dir():
        raise ValueError("--package-root must contain pyproject.toml and src/npa")
    files = catalog_files(root)
    if not files:
        raise ValueError(
            "The supported workflows/main and workflows/testing catalog is missing"
        )
    destination = root / "src/npa/workflows"
    if destination.is_symlink():
        raise ValueError(
            "The generated catalog destination must not be a symbolic link"
        )
    for directory in catalog_directories(destination):
        directory.mkdir(parents=True, exist_ok=True)
        for previous in directory.glob("*.yaml"):
            if previous.relative_to(destination) not in files:
                previous.unlink()
    for relative, source in files.items():
        target = destination / relative
        if target.is_symlink():
            raise ValueError("Generated workflow YAMLs must not be symbolic links")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    return len(files)


def get_build_hook():
    # Keep the staging CLI usable without importing a build-only dependency.
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class WorkflowCatalogHook(BuildHookInterface):
        def initialize(self, version: str, build_data: dict) -> None:
            files = catalog_files(Path(self.root))
            # Runtime-only source contexts can omit the authoring catalog.
            # Official image builds stage it first using the CLI below.
            destination = (
                "npa/workflows" if self.target_name == "wheel" else "src/npa/workflows"
            )
            includes = build_data.setdefault("force_include", {})
            for relative, path in files.items():
                includes[str(path)] = f"{destination}/{relative.as_posix()}"

    return WorkflowCatalogHook


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-catalog", action="store_true", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = stage_catalog(args.package_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Staged {count} supported workflow YAMLs for the package build context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
