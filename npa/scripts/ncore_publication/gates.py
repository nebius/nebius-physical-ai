"""Compose existing native byte, source, payload and Trivy gates for NCore."""

import json
import tarfile

import yaml

from image_byte_scan import core as W
from npa.deploy import images
from npa.deploy.publish_public import _TRIVY_CONTAINER_IMAGE
from . import artifact, bootstrap, components, provenance
from .diagnostics import phase, run_phase
from .process import guard_command, guard_snapshot, verify_guard_execution
from .process import ROOT, PYTHON, committed_source, file_sha, public_environment, run, write_json


def eligibility(sha):
    """Require public redistribution while preserving independent quarantine.

    Args:
        sha: Full reviewed source SHA.
    Returns:
        Exact official NCore development reference.
    Raises:
        ValueError: Classification or the development reference is invalid.
    """
    contract = yaml.safe_load((ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text())
    W.require(contract["images"]["ncore"]["redistribution"] == "public"
              and images.is_publicly_redistributable("ncore"), "ncore_redistribution_must_be_public")
    return images.development_image_for_tool("ncore", git_sha=sha,
        registry="ghcr.io/nebius/nebius-physical-ai")


def source_guards(directory, sha):
    """Execute the established packaging/license guards before building/pushing.

    Args:
        directory: Private output directory.
        sha: Full reviewed commit whose complete snapshot the tests execute.
    Returns:
        None.
    Raises:
        ValueError, OSError: Any existing guard fails.
    """
    snapshot, digest = guard_snapshot(directory, sha)
    env = public_environment()
    env["TMPDIR"] = str(directory)
    run(guard_command(snapshot, directory), directory / "source-guards.log", env=env, cwd=snapshot)
    verify_guard_execution(json.loads((directory / "source-guards.json").read_bytes()))
    write_json(directory / "source-guards-snapshot.json", {
        "source_sha": sha, "archive_sha256": digest,
        "scope": "complete-committed-repository", "execution_report": "source-guards.json",
        "python_site_startup": False, "pytest_plugin_autoload": False,
    })


def byte_scan(args, directory, archive, digest, verification):
    """Authorize and run the unchanged complete-byte scanner on original OCI.

    Args:
        args: CLI roots and private scanner receipts.
        directory: New gate output directory.
        archive: Original buildx OCI artifact.
        digest: Exact publication index identity.
        verification: Actual native graph verification receipt.
    Returns:
        None.
    Raises:
        ValueError, OSError: Authorization, native tools or complete scan fails.
    """
    write_json(directory / "graph.json", verification)
    common = ["--analysis-root", str(args.analysis_root), "--trusted-root", str(ROOT)]
    authorization = directory / "authorization"
    argv = [str(PYTHON), "npa/scripts/image_byte_scan/prepare.py", "authorize", *common,
            "--tools-receipt", str(args.analysis_root / "tools/dependency-receipt.json"),
            "--native-receipt", str(args.analysis_root / "native/dependencies.json"),
            "--archive", str(archive), "--verification-report", str(directory / "graph.json"),
            "--expected-image-id", digest, "--output-dir", str(authorization),
            "--policy-mode", args.policy_mode]
    if args.policy_mode == "exact-literals":
        argv.extend(["--literal-inventory", str(args.literal_inventory),
                     "--literal-matching-policy", "exact-substring-v1"])
    run(argv, directory / "authorize.log")
    run([str(PYTHON), "npa/scripts/scan_image_bytes.py", *common,
         "--authorization", str(authorization / "authorization.json"),
         "--output-dir", str(directory / "bytes")], directory / "bytes.log")
    report = json.loads((directory / "bytes/report.json").read_bytes())
    W.require(report.get("complete") is True and report.get("valid") is True
              and report.get("helper_joined") is True, "complete_byte_scan_required")


def verify(args, directory, build):
    """Run every local artifact gate; no registry write is reachable here.

    Args:
        args: Validated CLI arguments including source and private source inputs.
        directory: New gate evidence directory.
        build: Build receipt produced by the supported local build command.
    Returns:
        Graph and verification receipts after all gates succeed.
    Raises:
        ValueError, OSError, ImportError: A required gate or integration is absent.
    """
    with phase("source-binding"):
        eligibility(args.source_sha)
        W.require(committed_source(args.source_sha) == build["context_sha256"], "build_context_changed")
    run_phase("source-guards", source_guards, directory, args.source_sha)
    archive = args.analysis_root / "build/image.oci.tar"
    digest = build["image_digest"]
    graph, verification = _graph_provenance(args, directory, build, archive)
    run_phase("byte-scan", byte_scan, args, directory, archive, digest, verification)
    run_phase("inspection-archives", artifact.inspection_archives, archive, digest, graph, directory)
    run_phase("shipped-source", provenance.shipped_source, directory / "rootfs.tar", args.source_sha)
    run_phase("source-delivery", _source_delivery, args, directory)
    run_phase("payload", _payload, directory)
    run_phase("payload-history", _payload_history, directory, graph)
    run_phase("image-security", _security, directory, graph)
    run_phase("selected-base", _selected_base, directory, digest, graph)
    run_phase("components", components.verify, directory, digest, graph, args.source_sha)
    run_phase("bootstrap", bootstrap.verify, directory, graph["image_config_digest"], args.bootstrap_source)
    with phase("source-recheck"):
        artifact.assert_unchanged(archive, verification)
        W.require(committed_source(args.source_sha) == build["context_sha256"], "source_changed_during_gates")
    return graph, verification


def _graph_provenance(args, directory, build, archive):
    digest = build["image_digest"]
    with phase("oci-graph"):
        graph, verification = artifact.inspect(archive, digest)
        W.require(verification["archive_sha256"] == build["archive_sha256"], "build_archive_changed")
    with phase("provenance"):
        index, config, blobs = artifact.documents(archive, digest, graph)
        sbom = provenance.verify(index, config, blobs, graph, args.source_sha)
        write_json(directory / "buildx.spdx.json", sbom)
    return graph, verification


def _source_delivery(args, directory):
    base = [str(PYTHON), "npa/docker/workbench/ncore/base_sources.py"]
    run([*base, "inventory", "--image-archive", str(directory / "inspection.tar"),
         "--output", str(directory / "inventory.json")], directory / "inventory.log")
    # Metadata is deliberately mandatory and separate from delivered source.
    # Older base_sources without the metadata integration must fail here.
    run([*base, "verify", "--lock", str(ROOT / "npa/docker/workbench/ncore/base-source-lock.json"),
         "--annex", str(args.annex), "--native", str(args.native_source),
         "--metadata", str(args.metadata), "--keyring", str(args.keyring),
         "--inventory", str(directory / "inventory.json")], directory / "source-delivery.log")


def _payload(directory):
    run([str(PYTHON), "npa/scripts/scan_image_omniverse_payload.py",
         "--tarball", str(directory / "inspection.tar"),
         "--json", str(directory / "payload.json")], directory / "payload.log")
    payload = json.loads((directory / "payload.json").read_bytes())
    W.require(payload.get("format") == "npa_restricted_payload_scan_v2"
              and payload.get("scan_complete") is True
              and payload.get("history_only") is False
              and payload.get("verdict") == "clean"
              and type(payload.get("entries_scanned")) is int
              and payload["entries_scanned"] > 0, "complete_payload_scan_required")
    for field in ("payload_hits", "history_hits", "weight_shaped_paths"):
        W.require(isinstance(payload.get(field), list), "payload_report_population_required")
    W.require(not payload["payload_hits"] and not payload["history_hits"], "ncore_restricted_payload")
    # The scanner labels .pth paths as weight-shaped, including Python's
    # ordinary source-path hook. Authenticate this one exact text file; no
    # other weight-shaped name or executable .pth content is accepted.
    hook = "opt/venv/lib/python3.12/site-packages/npa-source.pth"
    W.require(payload["weight_shaped_paths"] == [hook], "unexpected_weight_shaped_paths")
    with tarfile.open(directory / "rootfs.tar") as archive:
        members = [member for member in archive if W.safe_name(member.name) == hook]
        W.require(len(members) == 1 and members[0].isfile()
                  and members[0].size == len(b"/opt/npa/src\n"), "python_import_hook_required")
        W.require(archive.extractfile(members[0]).read() == b"/opt/npa/src\n", "python_import_hook_changed")


def _payload_history(directory, graph):
    from scan_image_omniverse_payload import classify_history

    digest = graph["image_config_digest"]
    with tarfile.open(directory / "inspection.tar") as archive:
        members = [member for member in archive if member.name == digest[7:] + ".json"]
        W.require(len(members) == 1 and members[0].isfile(), "payload_original_config_required")
        raw = archive.extractfile(members[0]).read()
    W.require("sha256:" + W.sha(raw) == digest, "payload_config_digest")
    history = W.json_object(raw).get("history")
    W.require(isinstance(history, list) and history, "payload_history_population_required")
    hits = []
    for position, entry in enumerate(history):
        W.require(isinstance(entry, dict) and isinstance(entry.get("created_by"), str),
                  "payload_history_command_required")
        reason = classify_history(entry["created_by"])
        if reason:
            hits.append({"entry": position, "why": reason, "command": entry["created_by"]})
    write_json(directory / "payload-history.json", {
        "schema": "npa.ncore.payload-history.v1", "config_digest": digest,
        "classifier_sha256": file_sha(ROOT / "npa/scripts/scan_image_omniverse_payload.py"),
        "source": "digest-bound-original-config", "entries_classified": len(history),
        "history_hits": hits, "complete": True, "valid": not hits,
        "layer_paths_report": "payload.json",
    })
    W.require(not hits, "ncore_restricted_history")


def _security(directory, graph):
    # Pin the existing supported scanner and prevent ambient Trivy exclusions.
    command = ["docker", "run", "--rm", "--volume", f"{directory}:/evidence:ro",
               "--volume", f"{ROOT / '.trivyignore'}:/policy:ro", _TRIVY_CONTAINER_IMAGE,
               "image", "--input", "/evidence/inspection.tar", "--scanners", "vuln,secret,license",
               "--ignorefile", "/policy", "--ignore-unfixed", "--severity", "CRITICAL",
               "--exit-code", "1", "--format", "json"]
    run(command, directory / "trivy-policy.json", env=public_environment())
    # Retain fixed/unfixed accounting and reject secrets of every severity.
    arguments = command[:command.index("--ignorefile")] + ["--ignorefile", "/dev/null", "--ignore-unfixed=false",
                 "--severity", "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL", "--exit-code", "0", "--format", "json"]
    run(arguments, directory / "trivy-all.json", env=public_environment())
    payload = json.loads((directory / "trivy-all.json").read_bytes())
    metadata = payload.get("Metadata", {})
    W.require(payload.get("SchemaVersion") == 2
              and payload.get("ArtifactType") == "container_image"
              and metadata.get("ImageID") == graph["image_config_digest"]
              and metadata.get("DiffIDs") == [layer["diff_id"] for layer in graph["layers"]], "trivy_image_identity_required")
    # Trivy omits Results when no analyzers emit result rows. This scratch
    # image deliberately has no installed package database; the separate
    # selected-file SPDX scan still must identify every locked base package.
    results = payload.get("Results", [])
    W.require(isinstance(results, list), "trivy_results_required")
    for result in results:
        W.require(not result.get("Secrets"), "trivy_secret_findings")
        for finding in result.get("Vulnerabilities") or []:
            W.require(finding["Severity"] != "CRITICAL" or not finding.get("FixedVersion"), "fixed_critical_vulnerability")


def _selected_base(directory, digest, graph):
    from npa.deploy.ncore_selected_sbom import scan_archive

    scan_archive(directory / "rootfs.tar", directory / "selected-base",
                 image_digest=digest, platform_digest=graph["image_manifest_digest"],
                 config_digest=graph["image_config_digest"],
                 trivy_command=["docker", "run", "--rm", _TRIVY_CONTAINER_IMAGE])
