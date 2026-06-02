#!/usr/bin/env python3
"""Dev-VM-runnable registry existence check for SkyPilot workflow images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from npa.guardrails.skypilot import (
    image_refs_for_workflows,
    inspect_image_exists,
    resolve_workflow_image,
    unresolved_image_placeholders,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--registry-id", default=os.environ.get("NPA_REGISTRY_ID", ""))
    parser.add_argument("--strict-placeholders", action="store_true")
    args = parser.parse_args(argv)

    workflow_dir = args.repo_root / "npa" / "workflows" / "workbench" / "skypilot"
    images = image_refs_for_workflows(sorted(workflow_dir.glob("*.yaml")))
    unresolved = sorted({image for image in images if unresolved_image_placeholders(image)})
    if unresolved and not args.strict_placeholders:
        print(
            json.dumps(
                {
                    "status": "SEAM",
                    "reason": "workflow images still contain operator placeholders",
                    "unresolved_count": len(unresolved),
                    "checked_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.registry_id:
        print(
            json.dumps(
                {
                    "status": "SEAM",
                    "reason": "NPA_REGISTRY_ID is required for live registry inspection",
                    "checked_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    missing: list[str] = []
    checked: list[str] = []
    for image in images:
        resolved = resolve_workflow_image(image, registry_id=args.registry_id)
        if unresolved_image_placeholders(resolved):
            missing.append(resolved)
            continue
        try:
            ok = inspect_image_exists(resolved)
        except RuntimeError as exc:
            print(
                json.dumps(
                    {"status": "SEAM", "reason": str(exc), "checked_count": len(checked)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        checked.append(resolved)
        if not ok:
            missing.append(resolved)

    status = "WORKS" if not missing else "BLOCKED"
    print(
        json.dumps(
            {
                "status": status,
                "checked_count": len(checked),
                "missing": missing,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
