"""Remove dependencies intentionally excluded from the public Envgen runtime."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


EXCLUDED_DEPENDENCIES = (
    ("genesis-world", "tetgen"),
    ("lerobot", "wandb"),
)


def remove_dependency(distribution: str, dependency: str) -> None:
    """Remove exactly one unconditional Requires-Dist edge from installed metadata."""
    dist = metadata.distribution(distribution)
    metadata_path = Path(dist._path) / "METADATA"  # type: ignore[attr-defined]
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    prefix = f"requires-dist: {dependency}".lower()
    filtered = [line for line in lines if not line.lower().startswith(prefix)]
    if len(filtered) != len(lines) - 1:
        raise RuntimeError(
            f"expected exactly one {distribution} dependency on {dependency}"
        )
    metadata_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for distribution_name, dependency_name in EXCLUDED_DEPENDENCIES:
        remove_dependency(distribution_name, dependency_name)
