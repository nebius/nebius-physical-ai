"""Verify persisted Open3D inputs before deriving another artifact from them.

Object prefixes are mutable. A recording must bind to the exact registration,
manifest and geometry used by its reconstruction, rather than trusting whatever
bytes occupy those names later.
"""

from __future__ import annotations

import re
from typing import Any

from .artifacts import Open3dError, canonical, sha256_bytes, validate_pose_graph
from .schemas import RegistrationManifest


def verify_digest(payload: bytes, expected: Any, *, what: str) -> str:
    """Require a recorded SHA256 and matching bytes, returning the verified hash."""

    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise Open3dError(
            f"{what} has no valid recorded SHA256; regenerate the upstream artifact"
        )
    actual = sha256_bytes(payload)
    if actual != expected:
        raise Open3dError(f"{what} does not match its recorded SHA256")
    return actual


def verify_registration(
    report: dict[str, Any], graph: dict[str, Any], manifest: RegistrationManifest
) -> dict[str, str]:
    """Bind the graph and current manifest to the registration that used them."""

    if report.get("kind") != "multiway" or not isinstance(
        report.get("pose_graph"), dict
    ):
        raise Open3dError("recording requires a published multiway registration")
    if canonical(graph) != canonical(report["pose_graph"]):
        raise Open3dError("pose graph does not match the published registration")
    validate_pose_graph(graph, fragment_count=len(manifest.fragments))
    if [node.get("fragment_id") for node in graph["nodes"]] != [
        fragment.id for fragment in manifest.fragments
    ]:
        raise Open3dError("pose graph fragment identities do not match the manifest")
    manifest_hash = verify_digest(
        canonical(manifest.model_dump(mode="json")),
        report.get("manifest_sha256"),
        what="registration manifest",
    )
    if report.get("voxel_size") != manifest.voxel_size:
        raise Open3dError("registration scale does not match the manifest")
    return {
        "manifest_sha256": manifest_hash,
        "pose_graph_sha256": sha256_bytes(canonical(graph)),
    }
