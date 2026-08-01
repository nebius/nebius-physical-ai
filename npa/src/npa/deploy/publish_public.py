"""Mirror the OSS-redistributable workbench images to a public registry.

Nebius Container Registry does not support anonymous/public pulls, so "public
exposure" of the workbench means mirroring the publicly-redistributable image
subset to a public-capable registry (e.g. GHCR ``ghcr.io/<org>/<repo>``).

This tool is license-guarded: it only ever copies tools reported by
``images.publicly_publishable_tools()`` and hard-refuses the Omniverse-Kit
images (``isaac-lab``, ``sonic``, ``groot`` / ``sonic-mujoco``), which are
NVIDIA-proprietary and must not be redistributed to third parties.

Example (dry run first, then execute):

    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai --dry-run
    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass

from npa.deploy.images import (
    OMNIVERSE_RESTRICTED_TOOLS,
    container_image_for_tool,
    is_publicly_redistributable,
    omniverse_restricted_image_names,
    primary_container_registry,
    public_container_registry,
    publicly_publishable_tools,
)


@dataclass(frozen=True)
class PublishItem:
    tool: str
    source_ref: str
    target_ref: str


def build_publish_plan(
    *,
    target_registry: str,
    source_registry: str | None = None,
) -> list[PublishItem]:
    """Return the (source -> target) copy plan for the public image subset.

    Raises ``ValueError`` if an Omniverse-restricted tool ever leaks into the
    plan (defense in depth around the license boundary).
    """
    if not target_registry.strip():
        raise ValueError("target_registry is required")
    source_registry = source_registry or primary_container_registry()
    target = target_registry.rstrip("/")

    plan: list[PublishItem] = []
    for tool in publicly_publishable_tools():
        if not is_publicly_redistributable(tool) or tool in OMNIVERSE_RESTRICTED_TOOLS:
            raise ValueError(
                f"refusing to publish restricted (Omniverse Kit) tool {tool!r} to a public registry"
            )
        source_ref = container_image_for_tool(tool, registry=source_registry)
        image = source_ref.rsplit("/", 1)[-1]  # npa-<tool>:<tag>
        plan.append(PublishItem(tool=tool, source_ref=source_ref, target_ref=f"{target}/{image}"))
    return plan


def _crane_copy(item: PublishItem) -> None:
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError("crane not found on PATH; install go-containerregistry crane")
    subprocess.run([crane, "copy", item.source_ref, item.target_ref], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=public_container_registry(),
        help="Target public registry (e.g. ghcr.io/nebius/nebius-physical-ai); "
        "defaults to $NPA_PUBLIC_REGISTRY.",
    )
    parser.add_argument(
        "--source-registry",
        default=None,
        help="Source registry to copy from (defaults to the primary Nebius registry).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying.")
    args = parser.parse_args(argv)

    if not (args.target or "").strip():
        parser.error("no target registry; pass --target or set NPA_PUBLIC_REGISTRY")

    plan = build_publish_plan(target_registry=args.target, source_registry=args.source_registry)
    restricted = ", ".join(omniverse_restricted_image_names())
    print(f"Publishing {len(plan)} OSS image(s) to {args.target.rstrip('/')}")
    print(f"Excluded (NVIDIA Omniverse Kit, not for public registries): {restricted}")
    for item in plan:
        print(f"  {item.source_ref}  ->  {item.target_ref}")
    if args.dry_run:
        print("(dry run — nothing copied)")
        return 0
    for item in plan:
        _crane_copy(item)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
