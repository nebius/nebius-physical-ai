"""Transfer the original NCore OCI graph and compare a complete anonymous copy."""

import json
import subprocess

from image_byte_scan import core as W
from . import artifact
from .process import ROOT, public_environment, run, write_json

PACKAGE_API = "/orgs/nebius/packages/container/nebius-physical-ai%2Fnpa-ncore"


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
            _copy(args, directory, image, digest, archive, verification)
    _public_visibility(directory, image, digest, graph)
    _readback(args, directory, image, digest, graph)
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


def _public_visibility(directory, image, digest, graph):
    run(["gh", "api", PACKAGE_API], directory / "visibility.json")
    if json.loads((directory / "visibility.json").read_bytes())["visibility"] == "public":
        return
    run(["gh", "api", "--paginate", "--slurp", PACKAGE_API + "/versions?per_page=100"], directory / "versions.json")
    allowed = {digest, *(row["digest"] for row in graph["receipt"]["blobs"])}
    tag = image.rsplit(":", 1)[1]
    versions = [row for page in json.loads((directory / "versions.json").read_bytes()) for row in page]
    W.require(versions and all(row["name"] in allowed
              and set(row["metadata"]["container"]["tags"]) <= {tag} for row in versions),
              "private_package_contains_unvalidated_versions")
    run(["gh", "api", "--method", "PATCH", PACKAGE_API, "-f", "visibility=public"], directory / "make-public.log")


def _readback(args, directory, image, digest, graph):
    auth = directory / "anonymous.json"
    write_json(auth, {"auths": {}})
    # Explicit no-creds and an empty auth file prohibit fallback to Docker or
    # containers credential helpers; every blob is downloaded, not only HEADed.
    archive = directory / "anonymous.oci.tar"
    run(["skopeo", "copy", "--all", "--preserve-digests", "--src-no-creds",
         "--src-authfile", str(auth), "docker://" + image,
         "oci-archive:" + str(archive)], directory / "anonymous-copy.log", env=public_environment())
    observed_graph, verification = artifact.inspect(archive, digest)
    for key in ("image_manifest_digest", "image_config_digest"):
        W.require(observed_graph[key] == graph[key], "anonymous_image_identity_changed")
    W.require(observed_graph["receipt"]["blobs"] == graph["receipt"]["blobs"], "anonymous_graph_changed")
    _anonymous_tag(directory, image, digest, auth)
    from .gates import byte_scan

    byte_scan(args, directory, archive, digest, verification)
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
