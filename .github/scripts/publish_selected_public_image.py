"""Publish selected images through the repository's license-guarded mirror path."""

from __future__ import annotations

import argparse
import re

from npa.deploy.publish_public import (
    _crane_copy,
    _mark_copy_phase_complete,
    _preflight_or_explain,
    build_publish_plan,
    verify_public,
)


def _parse_tools(values: list[str]) -> list[str]:
    """Return an ordered, duplicate-free selector from repeated/CSV CLI values."""

    tools = [
        tool for value in values for tool in re.split(r"[\s,]+", value.strip()) if tool
    ]
    if not tools:
        raise ValueError("at least one workbench tool is required")
    duplicates = sorted({tool for tool in tools if tools.count(tool) > 1})
    if duplicates:
        raise ValueError("duplicate workbench tool(s): " + ", ".join(duplicates))
    return tools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        action="append",
        required=True,
        help=(
            "One or more publicly publishable workbench tools. May be repeated or "
            "provided as a comma/space-separated value."
        ),
    )
    parser.add_argument("--development-sha", default=None)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--mode",
        choices=("plan", "preflight", "publish", "verify"),
        required=True,
    )
    args = parser.parse_args()
    try:
        requested_tools = _parse_tools(args.tool)
    except ValueError as exc:
        parser.error(str(exc))

    # Build the complete plan first so the normal redistribution guard runs before
    # selection. The selector then prevents a scoped release from touching unrelated
    # packages whose source and target digests happen to differ.
    complete = build_publish_plan(
        target_registry=args.target,
        development_git_sha=args.development_sha or None,
    )
    by_tool = {item.tool: item for item in complete}
    unknown = [tool for tool in requested_tools if tool not in by_tool]
    if unknown:
        parser.error(
            "not publicly publishable workbench tool(s): " + ", ".join(unknown)
        )
    selected = [by_tool[tool] for tool in requested_tools]
    print(f"Selected {len(selected)} license-guarded image(s):")
    for item in selected:
        print(f"  {item.source_ref} -> {item.target_ref}")

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

    # Preflight returns digest-frozen items. Do not fall back to ``selected`` here:
    # those are mutable tag-form plan entries and the copy guard correctly rejects them.
    copied = sum(_crane_copy(item) for item in publishable)
    _mark_copy_phase_complete()
    print(
        f"Copied {copied} of {len(publishable)} image(s); "
        f"{len(publishable) - copied} already current."
    )
    failures = verify_public(publishable)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
