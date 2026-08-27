"""Materialize runtime-fetched guardrail blocklist files without symlinks."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def materialize_symlinks(root: Path) -> int:
    """Replace every symlink below *root* with a regular copy of its target."""

    replaced = 0
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if not target.is_file():
            raise RuntimeError(f"guardrail symlink target is not a file: {path}")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(target, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        replaced += 1
    remaining = [str(path) for path in root.rglob("*") if path.is_symlink()]
    if remaining:
        raise RuntimeError(f"guardrail runtime tree still contains symlinks: {remaining}")
    return replaced


def main() -> int:
    from huggingface_hub import snapshot_download

    repository = os.environ.get(
        "NPA_COSMOS3_SERVE_GUARDRAIL_MODEL", "nvidia/Cosmos-1.0-Guardrail"
    )
    revision = os.environ.get("NPA_COSMOS3_SERVE_GUARDRAIL_REVISION")
    if not revision:
        raise RuntimeError("missing pinned guardrail revision")
    snapshot = Path(
        snapshot_download(
            repository,
            revision=revision,
            allow_patterns=["blocklist/*"],
            token=os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        )
    )
    if snapshot.name != revision:
        raise RuntimeError(
            f"guardrail snapshot resolved to {snapshot.name}, expected {revision}"
        )
    blocklist = snapshot / "blocklist"
    if not blocklist.is_dir():
        raise RuntimeError("guardrail snapshot has no blocklist runtime data")
    replaced = materialize_symlinks(blocklist)
    regular_files = [path for path in blocklist.rglob("*") if path.is_file()]
    if not regular_files:
        raise RuntimeError("guardrail blocklist contains no runtime files")
    print(
        f"[npa-cosmos3-serving] prepared pinned guardrail blocklist "
        f"with {len(regular_files)} regular runtime files "
        f"({replaced} newly materialized)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
