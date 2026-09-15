"""Hatch hook including the repository workflow catalog in distributions.

The canonical source is outside this Python project's root. Source distributions,
staged worker sources, and container build contexts carry the same catalog under
``src/npa/workflows`` so they can build without a sibling repository checkout.
"""

from pathlib import Path
import shutil


def catalog_files(root: Path) -> dict[Path, Path]:
    """Map tier-relative YAML paths to the complete available source catalog."""

    for catalog in (root.parent / "workflows", root / "src/npa/workflows"):
        if catalog.is_symlink() or any(
            (catalog / tier).is_symlink() for tier in ("main", "testing")
        ):
            raise ValueError("Workflow catalog directories must not be symbolic links")
        if not all((catalog / tier).is_dir() for tier in ("main", "testing")):
            continue
        return {
            Path(tier) / path.name: path
            for tier in ("main", "testing")
            for path in sorted((catalog / tier).glob("*.yaml"))
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
    for tier in ("main", "testing"):
        if (destination / tier).is_symlink():
            raise ValueError("Generated catalog tiers must not be symbolic links")
        (destination / tier).mkdir(parents=True, exist_ok=True)
        for previous in (destination / tier).glob("*.yaml"):
            if previous.relative_to(destination) not in files:
                previous.unlink()
    for relative, source in files.items():
        target = destination / relative
        if target.is_symlink():
            raise ValueError("Generated workflow YAMLs must not be symbolic links")
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
