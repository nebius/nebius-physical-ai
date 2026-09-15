"""Require local NCore buildx statements and shipped source to match the commit."""

import json
import re
import subprocess
import tarfile
from urllib.parse import parse_qsl, unquote, urlsplit

from image_byte_scan import core as W
from .process import ROOT

SPDX = "https://spdx.dev/Document"
SLSA = {"https://slsa.dev/provenance/v0.2", "https://slsa.dev/provenance/v1"}
BUILDKIT_TYPES = {
    "https://slsa.dev/provenance/v0.2": "https://mobyproject.org/buildkit@v1",
    "https://slsa.dev/provenance/v1": "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md",
}
NATIVE_SOURCES = {
    "opt/ncore/bin/native-bootstrap.sh": "npa/docker/workbench/ncore/native-bootstrap.sh",
    "opt/ncore/native/native_bootstrap.py": "npa/docker/workbench/ncore/native_bootstrap.py",
    "opt/ncore/native/native-bootstrap-lock.json": "npa/docker/workbench/ncore/native-bootstrap-lock.json",
    "opt/ncore/native/_public_https.py": "npa/src/npa/_public_https.py",
}


def verify(index, config, blobs, graph, sha):
    """Enforce release-compatible buildx predicates without release acceptance.

    Args:
        index: Exact publication index.
        config: Exact runtime config.
        blobs: Digest-keyed decoded OCI documents.
        graph: Verified graph.
        sha: Full reviewed build commit.
    Returns:
        The genuine embedded SPDX predicate.
    Raises:
        ValueError, KeyError, TypeError: Required bindings or statements differ.
    """
    platform = graph["image_manifest_digest"]
    _config(config, sha)
    statements = {}
    for descriptor in index["manifests"]:
        if descriptor["digest"] == platform:
            continue
        _attestation(descriptor, blobs, platform, statements)
    sbom = statements.pop(SPDX, {})
    W.require(isinstance(sbom.get("packages"), list) and sbom["packages"], "embedded_sbom_required")
    W.require(statements and set(statements) <= SLSA, "slsa_provenance_required")
    for kind, predicate in statements.items():
        _source_parameters(kind, predicate, sha)
    return sbom


def _config(config, sha):
    runtime = config["config"]
    user = runtime.get("User", "").split(":", 1)[0]
    W.require(user and user != "root" and not (user.isdecimal() and int(user) == 0), "nonroot_required")
    labels = runtime.get("Labels", {})
    W.require(labels.get("org.opencontainers.image.revision") == sha, "config_source_revision")
    W.require(labels.get("org.nebius.npa.skypilot-bootstrap-contract") == "skypilot-0.12.2-v1", "bootstrap_contract_required")
    source_lock = json.loads((ROOT / "npa/docker/workbench/ncore/source-lock.json").read_bytes())
    W.require(labels.get("npa.ncore.revision") == source_lock["ncore"]["revision"], "upstream_revision")
    serialized = json.dumps(config)
    for value in ("OMNI_KIT_ACCEPT_EULA=YES", "ISAACSIM_ACCEPT_EULA=YES"):
        W.require(value not in serialized, "cached_acceptance_in_config_or_history")


def _attestation(descriptor, blobs, platform, statements):
    annotations = descriptor.get("annotations", {})
    W.require(descriptor.get("platform") == {"os": "unknown", "architecture": "unknown"}
              and annotations.get("vnd.docker.reference.type") == "attestation-manifest"
              and annotations.get("vnd.docker.reference.digest") == platform, "unbound_attestation")
    manifest = blobs[descriptor["digest"]]
    if "subject" in manifest:
        W.require(manifest["subject"]["digest"] == platform, "attestation_subject")
    for layer in manifest["layers"]:
        kind = layer.get("annotations", {}).get("in-toto.io/predicate-type")
        W.require(kind in SLSA | {SPDX} and kind not in statements, "unknown_or_duplicate_predicate")
        statement = blobs[layer["digest"]]
        W.require(statement.get("_type") in {"https://in-toto.io/Statement/v0.1", "https://in-toto.io/Statement/v1"}
                  and statement.get("predicateType") == kind, "invalid_statement")
        subjects = statement.get("subject")
        W.require(isinstance(subjects, list) and subjects
                  and all(s.get("digest", {}).get("sha256") == platform[7:] for s in subjects), "statement_subject")
        W.require(isinstance(statement.get("predicate"), dict), "invalid_predicate")
        statements[kind] = statement["predicate"]


def _source_parameters(kind, predicate, sha):
    if kind.endswith("/v0.2"):
        definition = predicate
        parameters = predicate.get("invocation", {}).get("parameters", {})
        dependencies = predicate.get("materials")
    else:
        definition = predicate.get("buildDefinition", {})
        parameters = definition.get("externalParameters", {}).get("request", {})
        dependencies = definition.get("resolvedDependencies")
    W.require(definition.get("buildType") == BUILDKIT_TYPES[kind] and isinstance(dependencies, list) and dependencies,
              "provenance_dependencies_required")
    for dependency in dependencies:
        W.require(isinstance(dependency.get("uri"), str) and dependency["uri"]
                  and any(re.fullmatch(pattern, str(dependency.get("digest", {}).get(algorithm)))
                          for algorithm, pattern in (("sha256", "[0-9a-f]{64}"), ("sha1", "[0-9a-f]{40}"))),
                  "provenance_dependency_digest")
    args = parameters.get("args", {})
    values = [args[key] for key in ("build-arg:SOURCE_SHA", "build-arg:NPA_SOURCE_SHA") if key in args]
    W.require(values and all(value == sha for value in values), "mode_max_exact_source_sha_required")
    materials = _base_material(dependencies)
    W.require(_sbom_generator()[0] in materials, "sbom_generator_material_required")


def _base_material(dependencies):
    lock = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    reference, digest = lock["base_image"].split("@")
    name, version = reference.rsplit(":", 1)
    expected = f"docker/{name}@{version}"
    dockerfile = (ROOT / "npa/docker/workbench/ncore/Dockerfile").read_text()
    syntax = re.findall(r"^# syntax=([^\s]+)$", dockerfile, re.MULTILINE)
    W.require(len(syntax) == 1, "provenance_frontend_reference_required")
    frontend = "docker/" + syntax[0].replace(":", "@")
    generator, generator_digest = _sbom_generator()
    seen = set()
    matches = []
    for dependency in dependencies:
        name, qualifiers = _material_reference(dependency)
        W.require(name in {expected, frontend, generator} and name not in seen, "unexpected_provenance_material")
        seen.add(name)
        if name == expected:
            W.require(qualifiers.get("platform") == "linux/amd64"
                      and qualifiers.get("digest", digest) == digest, "provenance_base_reference")
            matches.append(dependency["digest"])
        if name == generator:
            W.require(dependency["digest"] == {"sha256": generator_digest[7:]},
                      "provenance_sbom_generator_digest")
        # The frontend is tag-selected by the committed Dockerfile. Its digest
        # is structurally bound, not independently pinned or producer-signed.
    W.require(matches == [{"sha256": digest.removeprefix("sha256:")}], "locked_base_material_required")
    return seen


def _sbom_generator():
    build = (ROOT / "npa/docker/workbench/ncore/build.sh").read_text()
    references = re.findall(r'^SBOM_GENERATOR="([^"\n]+)"$', build, re.MULTILINE)
    W.require(len(references) == 1, "committed_sbom_generator_required")
    reference, digest = references[0].split("@")
    W.require(re.fullmatch(r"sha256:[0-9a-f]{64}", digest), "pinned_sbom_generator_required")
    return "docker/" + reference.replace(":", "@"), digest


def _material_reference(dependency):
    uri = urlsplit(dependency["uri"])
    pairs = parse_qsl(uri.query, strict_parsing=True) if uri.query else []
    qualifiers = dict(pairs)
    digest = dependency["digest"]
    W.require(uri.scheme == "pkg" and not uri.netloc and not uri.fragment
              and len(pairs) == len(qualifiers) and set(qualifiers) <= {"digest", "platform"}
              and qualifiers.get("platform", "linux/amd64") == "linux/amd64"
              and set(digest) == {"sha256"} and W.SHA.fullmatch(digest["sha256"])
              and qualifiers.get("digest", "sha256:" + digest["sha256"]) == "sha256:" + digest["sha256"],
              "provenance_material_reference")
    return unquote(uri.path), qualifiers


def shipped_source(path, sha):
    """Bind shipped NPA code, locks and the Dockerfile to committed source bytes.

    Args:
        path: Verified single-layer root filesystem archive.
        sha: Full reviewed source commit used for the build context.
    Returns:
        Number of compared shipped source/lock files.
    Raises:
        ValueError, OSError, KeyError: Source is absent or differs.
    """
    required = {"usr/share/doc/npa-ncore/npa-source-sha",
                "usr/share/doc/npa-ncore/recipes/Dockerfile",
                "opt/ncore/base-sources/recipes/base-source-lock.json",
                "usr/share/doc/npa-ncore/runtime-lock.json", *NATIVE_SOURCES}
    native_seen = set()
    count = 0
    with tarfile.open(path) as archive:
        for member in archive:
            name = W.safe_name(member.name)
            if name in NATIVE_SOURCES or name.startswith("opt/ncore/native/"):
                W.require(name in NATIVE_SOURCES and name not in native_seen,
                          "unexpected_native_source_population")
                native_seen.add(name)
            if member.isdir():
                continue
            source = _source_path(name)
            if not source and name not in required:
                continue
            W.require(member.isfile(), "shipped_source_must_be_regular")
            raw = archive.extractfile(member).read()
            if name == "usr/share/doc/npa-ncore/npa-source-sha":
                W.require(raw == (sha + "\n").encode(), "shipped_source_sha")
            else:
                W.require(W.sha(raw) == _committed_digest(source, sha), "shipped_source_changed")
            required.discard(name)
            count += 1
    W.require(not required and count > 4 + len(NATIVE_SOURCES), "shipped_source_population_missing")
    return count


def _committed_digest(source, sha):
    # Unrelated worktree edits never enter git archive and must not invalidate
    # the shipped-source comparison. Read the exact committed blob as well.
    relative = source.relative_to(ROOT).as_posix()
    result = subprocess.run(["git", "show", f"{sha}:{relative}"], cwd=ROOT,
                            capture_output=True, check=False)
    W.require(result.returncode == 0, "shipped_source_not_in_commit")
    return W.sha(result.stdout)


def _source_path(name):
    if name in NATIVE_SOURCES:
        return ROOT / NATIVE_SOURCES[name]
    npa = "opt/npa/src/npa/"
    if name.startswith(npa):
        relative = name.removeprefix(npa)
        if relative in {"cli/entry.py", "__main__.py"}:
            relative = "workflows/ncore.py"
        return ROOT / "npa/src/npa" / relative
    paths = {
        "usr/share/doc/npa-ncore/recipes/Dockerfile": "Dockerfile",
        "usr/share/doc/npa-ncore/recipes/stage_upstream.py": "stage_upstream.py",
        "opt/ncore/base-sources/recipes/base-source-lock.json": "base-source-lock.json",
        "opt/ncore/base-sources/recipes/base_sources.py": "base_sources.py",
        "opt/ncore/base-sources/recipes/BASE-SOURCES.md": "BASE-SOURCES.md",
        "opt/venv/bin/npa": "npa-entrypoint",
    }
    if name in {"usr/share/doc/npa-ncore/recipes/_public_https.py",
                "opt/ncore/base-sources/recipes/_public_https.py"}:
        return ROOT / "npa/src/npa/_public_https.py"
    if name.startswith("opt/ncore/bin/"):
        return ROOT / "npa/docker/workbench/ncore" / name.removeprefix("opt/ncore/bin/")
    document = name.removeprefix("usr/share/doc/npa-ncore/")
    if (document in {"runtime-lock.json", "source-lock.json", "REDISTRIBUTION.md", "BASE-SOURCES.md"}
            or ("/" not in document and document.endswith("requirements.lock"))):
        return ROOT / "npa/docker/workbench/ncore" / document
    return ROOT / "npa/docker/workbench/ncore" / paths[name] if name in paths else None
