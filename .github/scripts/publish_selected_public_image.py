"""Publish selected images through the repository's license-guarded mirror path."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace

from npa.deploy.images import development_image_for_tool, development_tag
from npa.deploy.publish_public import (
    PublishItem,
    _crane_copy,
    _crane_digest,
    _crane_json,
    _mark_copy_phase_complete,
    _preflight_or_explain,
    anonymous_digest,
    build_publish_plan,
    classify_preflight_failure,
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


def _validate_release_override(
    tools: list[str], tag: str, digest: str, sha: str | None
) -> None:
    if not tag and not digest:
        return
    if len(tools) != 1 or not tag or not digest or not sha:
        raise ValueError(
            "an additive release requires exactly one tool, --release-tag, "
            "--expected-source-digest, and explicit --development-sha"
        )
    development_tag(sha)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag) or any(
        token in {"dev", "latest", "stable", "main", "master", "nightly", "edge", "head"}
        for token in re.split(r"[-_.]", tag.lower())
    ):
        raise ValueError("release tag must be a safe, immutable additive tag")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("expected source digest must be a full sha256 digest")


def _require_additive_source(item: PublishItem, sha: str, digest: str) -> None:
    if not item.source_ref.endswith("@" + digest):
        raise RuntimeError("source differs from the GPU-accepted expected digest")
    config = _crane_json(["config", item.source_ref])
    labels = (config.get("config") or {}).get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != sha:
        raise RuntimeError("source revision differs from the accepted development SHA")
    ok, observed = anonymous_digest(item.source_ref)
    if not ok or observed != digest:
        raise RuntimeError("accepted source digest is not anonymously verified")


def _require_additive_target(item: PublishItem, digest: str) -> None:
    ok, observed = _crane_digest(item.target_ref)
    if ok:
        if observed != digest:
            raise RuntimeError("refusing to overwrite an existing additive release tag")
    elif classify_preflight_failure(observed) != "missing":
        raise RuntimeError("cannot prove additive target absence or matching digest")


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
    parser.add_argument(
        "--release-tag", default="",
        help="New immutable release tag for exactly one tool; never overwrites other bytes.",
    )
    parser.add_argument(
        "--expected-source-digest", default="",
        help="Exact public development digest already accepted by real GPU validation.",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--mode",
        choices=("plan", "preflight", "publish", "verify"),
        required=True,
    )
    args = parser.parse_args()
    try:
        requested_tools = _parse_tools(args.tool)
        _validate_release_override(
            requested_tools, args.release_tag, args.expected_source_digest,
            args.development_sha,
        )
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
    if args.release_tag:
        selected = [replace(
            selected[0],
            source_ref=development_image_for_tool(
                requested_tools[0], registry=args.target, git_sha=args.development_sha,
            ),
            target_ref=selected[0].target_ref.rsplit(":", 1)[0] + ":" + args.release_tag,
        )]
    print(f"Selected {len(selected)} license-guarded image(s):")
    for item in selected:
        print(f"  {item.source_ref} -> {item.target_ref}")

    if args.mode == "plan":
        return 0
    if args.mode == "verify" and not args.release_tag:
        failures = verify_public(selected)
        return 1 if failures else 0

    publishable = _preflight_or_explain(selected)
    if not publishable:
        return 1
    if args.release_tag:
        _require_additive_source(
            publishable[0], args.development_sha, args.expected_source_digest
        )
        _require_additive_target(publishable[0], args.expected_source_digest)
    if args.mode == "preflight":
        return 0

    # Preflight returns digest-frozen items. Do not fall back to ``selected`` here:
    # those are mutable tag-form plan entries and the copy guard correctly rejects them.
    if args.mode != "verify":
        copied = sum(
            _crane_copy(item, allow_replace=False) if args.release_tag else _crane_copy(item)
            for item in publishable
        )
        _mark_copy_phase_complete()
        print(
            f"Copied {copied} of {len(publishable)} image(s); "
            f"{len(publishable) - copied} already current."
        )
    failures = verify_public(publishable)
    if args.release_tag:
        ok, observed = anonymous_digest(publishable[0].target_ref)
        if not ok or observed != args.expected_source_digest:
            raise RuntimeError("additive release is not anonymously verified at the accepted digest")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
