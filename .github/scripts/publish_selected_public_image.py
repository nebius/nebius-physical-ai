"""Publish one image through the repository's license-guarded mirror path."""

from __future__ import annotations

import argparse

from npa.deploy.publish_public import (
    _crane_copy,
    _preflight_or_explain,
    build_publish_plan,
    verify_public,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--source-registry", default=None)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--mode", choices=("plan", "preflight", "publish", "verify"), required=True
    )
    args = parser.parse_args()

    # Build the complete plan first so the normal redistribution guard runs before
    # selection. The selector then prevents a one-image release from touching unrelated
    # packages whose source and target digests happen to differ.
    complete = build_publish_plan(
        target_registry=args.target,
        source_registry=args.source_registry or None,
    )
    selected = [item for item in complete if item.tool == args.tool]
    if len(selected) != 1:
        parser.error(
            f"tool {args.tool!r} is not exactly one publicly publishable workbench image"
        )
    item = selected[0]
    print(f"Selected license-guarded image: {item.source_ref} -> {item.target_ref}")

    if args.mode == "plan":
        return 0
    if args.mode == "verify":
        failures = verify_public(selected)
        return 1 if failures else 0

    publishable = _preflight_or_explain(selected)
    if not publishable:
        return 1
    if args.mode == "preflight":
        return 0

    copied = _crane_copy(item)
    print("Copied 1 image." if copied else "Already current; copied 0 images.")
    failures = verify_public(selected)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
