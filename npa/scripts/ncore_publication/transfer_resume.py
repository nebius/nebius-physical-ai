"""Explicit owner-pinned resumption of a previously accepted registry transfer."""

from image_byte_scan import core as W, prepare as P

from . import handoff
from .process import file_sha


def _identity(args, build, graph, verification):
    return {
        "source_sha": build["source_sha"],
        "image": build["image"],
        "image_digest": build["image_digest"],
        "archive_sha256": verification["archive_sha256"],
        "build_receipt_sha256": file_sha(args.analysis_root / "build/build.json"),
        "acceptance_sha256": file_sha(args.acceptance),
        "graph": handoff.graph_blobs(graph, build["image_digest"]),
    }


def require_prior_transfer(args, build, graph, verification):
    """Verify an explicit private receipt and its independently retained hash.

    Args:
        args: Publication arguments including the receipt path and pinned SHA-256.
        build: Currently checked original build receipt.
        graph: Freshly verified complete graph.
        verification: Fresh verification of the original OCI archive.
    Returns:
        None. Fresh publication gates and anonymous verification remain required.
    Raises:
        ValueError: The selected prior transfer is unbound, foreign or altered.
    """
    path = getattr(args, "resume_transfer", None)
    W.require(path is not None, "development_tag_preexisted_acceptance")
    binding = P.binding(path)
    W.require(
        binding["sha256"] == args.resume_transfer_sha256,
        "resume_transfer_receipt_changed",
    )
    receipt = W.bound_json(binding)
    W.require(
        receipt.get("schema") == "npa.ncore.completed-transfer.v1"
        and receipt.get("status") == "copied_not_published"
        and receipt.get("identity") == _identity(args, build, graph, verification),
        "resume_transfer_identity_differs",
    )
    pre = W.bound_json(receipt["prepublication"])
    W.require(
        pre.get("status") == "pass"
        and all(
            pre.get(key) == build[key]
            for key in ("source_sha", "image_digest", "archive_sha256")
        ),
        "resume_transfer_prepublication_differs",
    )


def save_transfer(args, directory, build, graph, verification):
    """Retain a private continuation receipt after the exact tag is observed.

    Args:
        args: Accepted publication inputs.
        directory: Fresh private transfer output directory.
        build: Current supported build receipt.
        graph: Complete verified graph.
        verification: Verification of the unchanged original archive.
    Returns:
        None; this is not an anonymous publication or release receipt.
    Raises:
        ValueError, OSError: Evidence cannot be bound or saved privately.
    """
    W.write_private_json(
        directory,
        "completed-transfer.json",
        {
            "schema": "npa.ncore.completed-transfer.v1",
            "status": "copied_not_published",
            "identity": _identity(args, build, graph, verification),
            "prepublication": P.binding(directory.parent / "prepublication.json"),
        },
    )
