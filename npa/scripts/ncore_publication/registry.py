"""Transfer the original NCore OCI graph and compare a complete anonymous copy."""

import subprocess

from image_byte_scan import core as W, oci_graph as G
from . import artifact, handoff
from .diagnostics import phase, run_phase
from .process import ROOT, public_environment, run, write_json

PACKAGE_API = "/orgs/nebius/packages/container/nebius-physical-ai%2Fnpa-ncore"


@phase("registry-tag-lookup")
def _observed(reference, output, authfile):
    argv = ["skopeo", "inspect", "--raw", "--authfile", str(authfile), "docker://" + reference]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False, env=public_environment())
    output.write_bytes(result.stdout)
    output.with_suffix(".stderr").write_bytes(result.stderr)
    if result.returncode == 0:
        W.require(result.stdout, "empty_registry_manifest")
        return "sha256:" + W.sha(result.stdout)
    # A denied/throttled/unreachable lookup is never evidence of absence.
    detail = result.stderr.decode(errors="replace").lower()
    denied = any(word in detail for word in ("unauthorized", "denied", "forbidden", "429", "tls"))
    W.require(not denied and ("manifest unknown" in detail or "name unknown" in detail),
              "registry_lookup_failed_not_proven_absent")
    return None


def _require_equal_or_absent(observed, digest):
    W.require(observed is None or observed == digest, "immutable_tag_has_divergent_bytes")


def transfer(args, directory, build, graph, verification):
    """Copy all original manifests with preservation enabled after all gates.

    Args:
        args: Validated CLI arguments with a private registry auth file.
        directory: Fresh private transfer evidence directory.
        build: Genuine supported build receipt.
        graph: Original verified OCI graph.
        verification: Actual completed prepublication graph receipt.
    Returns:
        Exact publication digest, after anonymous equality and full-byte checks.
    Raises:
        handoff.AdministratorHandoffRequired: Validated private package needs its admin.
        ValueError, OSError: Existing bytes differ, transfer or readback fails.
    """
    image, digest = build["image"], build["image_digest"]
    archive = args.analysis_root / "build/image.oci.tar"
    artifact.assert_unchanged(archive, verification)
    run(["skopeo", "inspect", "--raw", "oci-archive:" + str(archive)], directory / "local-index.json")
    W.require("sha256:" + W.sha((directory / "local-index.json").read_bytes()) == digest,
              "transfer_source_selection_differs")
    observed = _observed(image, directory / "existing.json", args.authfile)
    _require_equal_or_absent(observed, digest)
    if observed is None:
        # The workflow serializes this immutable SHA. Recheck immediately before
        # copy; an already equal tag never causes any registry write.
        observed = _observed(image, directory / "before-copy.json", args.authfile)
        _require_equal_or_absent(observed, digest)
        if observed is None:
            run_phase("registry-copy", _copy, args, directory, image, digest, archive, verification)
    run_phase("registry-visibility", _public_visibility, args, directory, build, graph, verification)
    run_phase("anonymous-verification", _readback, args, directory, image, digest, graph)
    artifact.assert_unchanged(archive, verification)
    return digest


def _copy(args, directory, image, digest, archive, verification):
    artifact.assert_unchanged(archive, verification)
    exact = image.rsplit(":", 1)[0] + "@" + digest
    run(["skopeo", "copy", "--all", "--preserve-digests", "--dest-authfile", str(args.authfile),
         "--digestfile", str(directory / "copied-digest"), "oci-archive:" + str(archive),
         "docker://" + exact], directory / "copy.log", env=public_environment())
    W.require((directory / "copied-digest").read_text().strip() == digest, "pushed_index_changed")
    # Uploading the potentially large graph uses a digest destination. Recheck
    # the tag after that upload, then attach only this exact registry graph.
    observed = _observed(image, directory / "before-tag.json", args.authfile)
    _require_equal_or_absent(observed, digest)
    if observed == digest:
        return
    run(["skopeo", "copy", "--all", "--preserve-digests",
         "--src-authfile", str(args.authfile), "--dest-authfile", str(args.authfile),
         "--digestfile", str(directory / "tagged-digest"), "docker://" + exact,
         "docker://" + image], directory / "tag.log", env=public_environment())
    W.require((directory / "tagged-digest").read_text().strip() == digest, "tagged_index_changed")


def _public_visibility(args, directory, build, graph, verification):
    run(["gh", "api", PACKAGE_API], directory / "visibility.json")
    package = W.json_object((directory / "visibility.json").read_bytes())
    if package["visibility"] == "public":
        return
    inventory = _package_inventory(package, directory / "versions.json", build, graph)
    # Detect inventory drift during the check. The administrator must still
    # refresh immediately before the UI action; these reads cannot lock GHCR.
    run(["gh", "api", PACKAGE_API], directory / "visibility-refresh.json")
    refreshed = W.json_object((directory / "visibility-refresh.json").read_bytes())
    W.require(_package_inventory(refreshed, directory / "versions-refresh.json", build, graph)
              == inventory, "private_package_inventory_changed")
    W.require(_observed(build["image"], directory / "handoff-index.json", args.authfile)
              == build["image_digest"], "private_package_tag_changed")
    handoff.require_administrator(args, directory, build, graph, verification, inventory)


def _package_inventory(package, output, build, graph):
    W.require(package.get("visibility") in {"private", "internal"}
              and package.get("package_type") == "container"
              and package.get("name") == "nebius-physical-ai/npa-ncore"
              and package.get("owner", {}).get("login") == "nebius"
              and type(package.get("version_count")) is int and package["version_count"] > 0,
              "invalid_private_package_inventory")
    run(["gh", "api", "--paginate", "--slurp", PACKAGE_API + "/versions?per_page=100"], output)
    pages = W.json_object(output.read_bytes())
    W.require(type(pages) is list and pages and all(type(page) is list and page for page in pages),
              "incomplete_private_package_inventory")
    rows = [_version_identity(row) for page in pages for row in page]
    W.require(len(rows) == package["version_count"]
              and len({row["id"] for row in rows}) == len(rows)
              and len({row["digest"] for row in rows}) == len(rows), "incomplete_private_package_inventory")
    digest = build["image_digest"]
    blobs = handoff.graph_blobs(graph, digest)
    # GHCR versions are manifests, not config/layer blobs. Every manifest in
    # this graph must be present, and only its publication index may be tagged.
    expected = {digest, *(row["digest"] for row in blobs
                           if row["mediaType"] in {G.INDEX, G.MANIFEST, G.ATTESTATION})}
    tag = build["image"].rsplit(":", 1)[1]
    W.require({row["digest"] for row in rows} == expected
              and all(row["tags"] == ([tag] if row["digest"] == digest else []) for row in rows),
              "private_package_contains_unvalidated_versions")
    return {"visibility": package["visibility"], "versions": sorted(rows, key=lambda row: row["digest"])}


def _version_identity(row):
    W.require(type(row) is dict and type(row.get("id")) is int and row["id"] > 0
              and type(row.get("name")) is str and W.DIGEST.fullmatch(row["name"]),
              "invalid_private_package_version")
    metadata = row.get("metadata")
    W.require(type(metadata) is dict and metadata.get("package_type") == "container"
              and type(metadata.get("container")) is dict, "invalid_private_package_version")
    tags = metadata["container"].get("tags")
    W.require(type(tags) is list and all(type(tag) is str for tag in tags), "invalid_private_package_tags")
    return {"id": row["id"], "digest": row["name"], "tags": tags}


def _readback(args, directory, image, digest, graph):
    auth = directory / "anonymous.json"
    write_json(auth, {"auths": {}})
    # Explicit no-creds and an empty auth file prohibit fallback to Docker or
    # containers credential helpers; every blob is downloaded, not only HEADed.
    archive = directory / "anonymous.oci.tar"
    with phase("anonymous-copy"):
        run(["skopeo", "copy", "--all", "--preserve-digests", "--src-no-creds",
             "--src-authfile", str(auth), "docker://" + image,
             "oci-archive:" + str(archive)], directory / "anonymous-copy.log", env=public_environment())
    with phase("anonymous-graph"):
        observed_graph, verification = artifact.inspect(archive, digest)
        for key in ("image_manifest_digest", "image_config_digest"):
            W.require(observed_graph[key] == graph[key], "anonymous_image_identity_changed")
        W.require(observed_graph["receipt"]["blobs"] == graph["receipt"]["blobs"], "anonymous_graph_changed")
    run_phase("anonymous-tag-check", _anonymous_tag, directory, image, digest, auth)
    from .gates import byte_scan

    run_phase("byte-scan", byte_scan, args, directory, archive, digest, verification)
    write_json(directory / "published.json", {
        "image_digest": digest, "platform_digest": graph["image_manifest_digest"],
        "config_digest": graph["image_config_digest"], "graph": graph["receipt"]["blobs"],
        "anonymous_archive_sha256": verification["archive_sha256"],
        "release_acceptance": False,
    })


def _anonymous_tag(directory, image, digest, auth):
    run(["skopeo", "inspect", "--raw", "--no-creds", "--authfile", str(auth),
         "docker://" + image], directory / "anonymous-index.json", env=public_environment())
    W.require("sha256:" + W.sha((directory / "anonymous-index.json").read_bytes()) == digest,
              "anonymous_tag_changed")
