"""Bind a private NCore package-admin handoff without accepting a publication."""

import re

from image_byte_scan import core as W, oci_graph as G, prepare as P


_ACTION = (
    "Package admin: refresh the exact validated inventory immediately before action; "
    "in GitHub organization nebius, container package nebius-physical-ai/npa-ncore, "
    "open Package settings > Danger Zone > Change visibility > Public. "
    "Then retry publish with the same original OCI archive and a new evidence directory; "
    "anonymous full graph, hash and byte verification is mandatory."
)
_GATE_FILES = (
    "prepublication.json", "graph.json", "source-guards.json", "source-guards-snapshot.json",
    "buildx.spdx.json", "bytes/report.json", "inspection.json", "inventory.json",
    "source-delivery.log", "payload.json", "payload-history.json", "trivy-policy.json",
    "trivy-all.json", "selected-base/selected.receipt.json", "components/receipt.json",
    "local-image-binding.json", "image-only.log", "entrypoint-bootstrap.log", "bash-bootstrap.log",
)


class AdministratorHandoffRequired(ValueError):
    """Signal a validated visibility handoff that still fails publication.

    Args:
        source_sha: Full source commit identity.
        image_digest: Exact immutable development index identity.
        receipt_sha256: Hash of the private handoff receipt file.
    Returns:
        Exception carrying only the minimal public handoff fields.
    Raises:
        ValueError: Any field is not a plain, complete lowercase identity.
    """

    def __init__(self, source_sha, image_digest, receipt_sha256):
        _identity(source_sha, r"[0-9a-f]{40}")
        _identity(image_digest, r"sha256:[0-9a-f]{64}")
        _identity(receipt_sha256, r"[0-9a-f]{64}")
        super().__init__("ncore_administrator_handoff_required")
        self.source_sha = source_sha
        self.image_digest = image_digest
        self.receipt_sha256 = receipt_sha256


def _identity(value, pattern):
    W.require(type(value) is str and re.fullmatch(pattern, value), "invalid_ncore_handoff_identity")


def public_summary(error):
    """Render only revalidated handoff identities and a fixed package-admin action.

    Args:
        error: Typed handoff failure, never an arbitrary exception to format.
    Returns:
        Public-safe text, or None if the exception or its fields are invalid.
    Raises:
        None.
    """
    if type(error) is not AdministratorHandoffRequired:
        return None
    values = vars(error)
    fields = {"source_sha": r"[0-9a-f]{40}", "image_digest": r"sha256:[0-9a-f]{64}",
              "receipt_sha256": r"[0-9a-f]{64}"}
    for key, pattern in fields.items():
        value = values.get(key)
        if type(value) is not str or re.fullmatch(pattern, value) is None:
            return None
    return ("NCore OCI administrator-handoff-required "
            f"receipt_sha256={values['receipt_sha256']} source_sha={values['source_sha']} "
            f"image_digest={values['image_digest']}\n" + _ACTION)


def graph_blobs(graph, digest):
    """Select strictly typed blob identities from the already verified OCI graph.

    Args:
        graph: Graph returned by the existing complete OCI validation.
        digest: Exact publication index identity.
    Returns:
        Validated digest, media type and size descriptors.
    Raises:
        ValueError: Graph identities or descriptor population are malformed.
    """
    _identity(digest, r"sha256:[0-9a-f]{64}")
    receipt = graph["receipt"]
    W.require(receipt["image_index_digest"] == digest
              and receipt["runtime_platform"] == {"os": "linux", "architecture": "amd64"},
              "invalid_ncore_handoff_graph")
    rows = receipt["blobs"]
    W.require(type(rows) is list and rows, "invalid_ncore_handoff_graph")
    allowed = {G.INDEX, G.MANIFEST, G.CONFIG, G.EMPTY, G.INTOTO, G.ATTESTATION, *G.LAYERS}
    seen = set()
    for row in rows:
        W.require(type(row) is dict and set(row) == {"digest", "size", "mediaType"},
                  "invalid_ncore_handoff_blob")
        _identity(row["digest"], r"sha256:[0-9a-f]{64}")
        W.require(type(row["mediaType"]) is str and row["mediaType"] in allowed
                  and type(row["size"]) is int and row["size"] >= 0
                  and row["digest"] not in seen, "invalid_ncore_handoff_blob")
        seen.add(row["digest"])
    W.require({graph["image_manifest_digest"], graph["image_config_digest"]} <= seen,
              "incomplete_ncore_handoff_graph")
    return rows


def _evidence(args, directory, build, verification):
    files = {name: directory.parent / name for name in _GATE_FILES}
    files.update({"build_receipt": args.analysis_root / "build/build.json",
                  "build_metadata": args.analysis_root / "build/buildx.json"})
    for name in ("visibility.json", "versions.json", "visibility-refresh.json",
                 "versions-refresh.json", "handoff-index.json"):
        files["registry/" + name] = directory / name
    bindings = {name: P.binding(path) for name, path in files.items()}
    W.require(W.bound_json(bindings["build_receipt"]) == build
              and bindings["build_metadata"]["sha256"] == build["metadata"]["sha256"]
              and W.bound_json(bindings["graph.json"]) == verification,
              "ncore_handoff_evidence_changed")
    W.require(W.bound_json(bindings["prepublication.json"]) == {
        "status": "pass", "source_sha": build["source_sha"], "image_digest": build["image_digest"],
        "archive_sha256": build["archive_sha256"], "release_acceptance": False,
    }, "ncore_handoff_prepublication_required")
    return {name: binding["sha256"] for name, binding in bindings.items()}


def _build_identity(args, build, graph, verification):
    from . import gates

    source, digest = build["source_sha"], build["image_digest"]
    _identity(source, r"[0-9a-f]{40}")
    W.require(source == args.source_sha and build["image"] == gates.eligibility(source),
              "invalid_ncore_handoff_source")
    W.require(verification["valid"] is True and verification["image_index_digest"] == digest
              and verification["expected_image_id"] == digest
              and verification["archive_sha256"] == build["archive_sha256"], "invalid_ncore_handoff_verification")
    for key in ("image_manifest_digest", "image_config_digest"):
        W.require(graph[key] == verification[key], "invalid_ncore_handoff_verification")
    return source, digest


def _save_receipt(directory, receipt):
    identity = W.write_private_json(directory, "administrator-handoff.json", receipt)
    binding = P.binding(directory / "administrator-handoff.json")
    with W.bound_open(binding) as (_, _, observed):
        W.require(W.stat_fingerprint(observed) == identity, "ncore_handoff_receipt_changed")
    return binding["sha256"]


def require_administrator(args, directory, build, graph, verification, inventory):
    """Write a hash-bound private receipt after successful gates and inventory checks.

    Args:
        args: Validated publication arguments for the original archive.
        directory: Fresh private registry evidence directory.
        build: Current supported build receipt.
        graph: Actual validated original OCI graph.
        verification: Actual graph verification from completed prepublication gates.
        inventory: Fully paginated and rechecked exact package inventory.
    Returns:
        None; this boundary never accepts publication or release.
    Raises:
        AdministratorHandoffRequired: Receipt written; package-admin action is required.
        ValueError, OSError: An identity or required evidence file is invalid.
    """
    from . import artifact

    source, digest = _build_identity(args, build, graph, verification)
    blobs = graph_blobs(graph, digest)
    evidence = _evidence(args, directory, build, verification)
    artifact.assert_unchanged(args.analysis_root / "build/image.oci.tar", verification)
    receipt = {
        "schema": "npa.ncore.visibility-admin-handoff.v1", "source_sha": source,
        "image": build["image"], "image_digest": digest,
        "platform_digest": graph["image_manifest_digest"], "config_digest": graph["image_config_digest"],
        "archive_sha256": verification["archive_sha256"], "graph": blobs,
        "graph_sha256": W.sha(W.canonical(graph["receipt"])),
        "package_inventory": inventory, "package_inventory_sha256": W.sha(W.canonical(inventory)),
        "evidence_sha256": evidence, "administrator_action": _ACTION,
        "status": "administrator-handoff-required", "release_acceptance": False,
    }
    identity = _save_receipt(directory, receipt)
    raise AdministratorHandoffRequired(source, digest, identity)
