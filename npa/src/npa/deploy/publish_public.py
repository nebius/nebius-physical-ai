"""Promote validated public GHCR development images to supported releases.

This tool is license-guarded: it only ever copies tools reported by
``images.publicly_publishable_tools()`` and hard-refuses anything in
``images.RESTRICTED_PUBLICATION_TOOLS`` as defence in depth around that selector.

The Isaac tools fetch Isaac Sim / Isaac Lab at first run under the operator's
own EULA acceptance. Cosmos3 serving is a zero-payload runtime bootstrap, while
SONIC MuJoCo is rebuilt independently without a vendor-container parent. Both
are promoted only from their recorded exact GPU-accepted digests.

Example (dry run first, then execute):

    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai --dry-run
    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai

Development and release tags share one public package. The copy path preflights every
immutable ``dev-<full-git-sha>`` source, skips release tags whose manifest digest already
matches, and verifies anonymous pullability and digest parity afterward. A stale
credential therefore fails before anything is written and unchanged tags are not recopied.

``--verify-parity`` is the read-only drift check for automation: it runs the same source
preflight and then requires every target tag to resolve to the exact same OCI digest. This
is deliberately stronger than ``--verify-public``, because an anonymously pullable tag can
still serve stale bytes.

``--verify-accepted-releases`` is the historical release-byte check. It resolves every
recorded release tag anonymously and compares it directly with the accepted manifest's
``published_digest``. It does not depend on retention of the development tag.

``--skip-missing`` publishes the images that exist when some pin refers to an image nobody
has built yet. The plan comes from the packaging contract (what the repo BUILDS) while the
registry holds what was pushed, so that gap is routine for a young tool; a denial is never
skipped (see ``classify_preflight_failure``).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from npa.deploy import images
from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    is_publicly_redistributable,
    public_container_registry,
    publicly_publishable_tools,
    restricted_image_names,
)


@dataclass(frozen=True)
class PublishItem:
    tool: str
    source_ref: str
    target_ref: str


def _mark_copy_phase_complete() -> None:
    """Tell GitHub Actions that every planned copy operation completed.

    The publish command can still exit non-zero after this point when anonymous
    verification fails. Keeping that state separate from the process exit status
    distinguishes a pre-copy refusal from a failed final public verification.
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as output:
        output.write("copy_phase_completed=true\n")


def build_publish_plan(
    *,
    target_registry: str,
    development_git_sha: str | None = None,
) -> list[PublishItem]:
    """Return the (source -> target) copy plan for the public image subset.

    Raises ``ValueError`` if an Omniverse-restricted tool ever leaks into the
    plan (defense in depth around the license boundary).
    """
    if not target_registry.strip():
        raise ValueError("target_registry is required")
    target_registry = images._ghcr_namespace(
        target_registry, channel="public release"
    )
    target = target_registry.rstrip("/")
    default_source_sha = _development_git_sha(development_git_sha)

    plan: list[PublishItem] = []
    for tool in publicly_publishable_tools():
        # Licence eligibility and evidence are separate gates, and a tool can
        # pass the first while having no built artifact to have checked. Those
        # are excluded from the plan rather than failed during preflight,
        # because the plan is meant to read as "what we would hand to the
        # public" — and something that does not exist is not that.
        if tool in images.PUBLICATION_QUARANTINE_TOOLS:
            continue
        # Read the restricted set through the module, never a from-import: a
        # defence-in-depth check that holds a stale copy of the thing it is defending is
        # worse than no check at all. (`from ... import RESTRICTED_PUBLICATION_TOOLS` binds
        # the value at import time, so this guard and publicly_publishable_tools() could
        # disagree - which is exactly what a test caught once the set stopped being empty.)
        if (
            not is_publicly_redistributable(tool)
            or tool in images.RESTRICTED_PUBLICATION_TOOLS
        ):
            raise ValueError(
                f"refusing to publish restricted tool {tool!r} to a public registry"
            )
        source_sha = (
            images.accepted_publication_development_sha(tool)
            or default_source_sha
        )
        source_ref = images.development_image_for_tool(
            tool, registry=target, git_sha=source_sha
        )
        image_name = CONTAINER_IMAGE_NAMES[tool]
        release_tag = images.public_release_tag_for_tool(tool)
        plan.append(
            PublishItem(
                tool=tool,
                source_ref=source_ref,
                target_ref=f"{target}/{image_name}:{release_tag}",
            )
        )
    return plan


def _development_git_sha(explicit: str | None = None) -> str:
    """Resolve the exact source commit used for every public development tag."""
    value = str(explicit or os.environ.get("NPA_DEVELOPMENT_SHA") or "").strip()
    if not value:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if completed.returncode == 0:
            value = completed.stdout.strip()
    # development_tag performs strict full-SHA validation.
    images.development_tag(value)
    return value.lower()


# --------------------------------------------------------------------------------------
# Source preflight
#
# `crane auth login` writes a config file and exits 0 for ANY token -- it never contacts
# the registry -- so a stale GHCR token looks like a successful login and only surfaces
# as a failed copy. And `crane copy` is a per-image subprocess with no
# transaction around the set, so the Nth image failing leaves N-1 packages already
# created. Reading every source manifest first turns both of those into one fast,
# read-only failure before anything is written.
# --------------------------------------------------------------------------------------

_PREFLIGHT_TIMEOUT_SECONDS = 60
_TRIVY_CONTAINER_IMAGE = (
    "docker.io/aquasec/trivy@"
    "sha256:cffe3f5161a47a6823fbd23d985795b3ed72a4c806da4c4df16266c02accdd6f"
)


def _repository(ref: str) -> str:
    without_digest = ref.split("@", 1)[0]
    slash = without_digest.rfind("/")
    colon = without_digest.rfind(":")
    return without_digest[:colon] if colon > slash else without_digest


def _crane_json(args: list[str]) -> dict[str, Any]:
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError(
            "crane not found on PATH; install go-containerregistry crane"
        )
    completed = subprocess.run(
        [crane, *args], capture_output=True, text=True, check=False
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"crane {' '.join(args)} failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"crane {' '.join(args)} did not return a JSON object")
    return payload


def _crane_blob_json(repository: str, digest: str) -> dict[str, Any]:
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError(
            "crane not found on PATH; install go-containerregistry crane"
        )
    completed = subprocess.run(
        [crane, "blob", f"{repository}@{digest}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or b"").decode(errors="replace")
        raise RuntimeError(detail.strip() or f"cannot read attestation blob {digest}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"attestation blob {digest} is not a JSON object")
    return payload


def _github_attestation_predicates(
    *, repository: str, digest: str
) -> set[str]:
    """Return structurally valid GitHub attestations bound to one OCI digest.

    Build-time actions create Sigstore bundles in GitHub's attestation store and
    GHCR referrers while leaving the subject as an ordinary single-platform
    manifest. The publication gate checks the bundle envelope, transparency-log
    material, exact subject, and predicate payload instead of assuming the image
    itself was rewritten into an attestation-bearing OCI index.
    """

    match = re.fullmatch(r"sha256:([0-9a-f]{64})", digest)
    if match is None:
        raise RuntimeError("attestation subject digest is invalid")
    if repository not in {
        "ghcr.io/nebius/nebius-physical-ai/npa-ltx2",
        "ghcr.io/nebius/nebius-physical-ai/npa-wan2-2",
    }:
        raise RuntimeError("attestation lookup is limited to official image repositories")
    url = (
        "https://api.github.com/repos/nebius/nebius-physical-ai/attestations/"
        + digest
    )
    request = urllib.request.Request(  # noqa: S310 - fixed GitHub API origin
        url,
        headers={"Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed GitHub API origin
            request, timeout=_PREFLIGHT_TIMEOUT_SECONDS
        ) as response:
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read exact-digest GitHub attestations: {exc}") from exc
    attestations = payload.get("attestations") if isinstance(payload, dict) else None
    if not isinstance(attestations, list) or not attestations:
        raise RuntimeError("exact digest has no GitHub attestations")
    predicates: set[str] = set()
    for record in attestations:
        bundle = record.get("bundle") if isinstance(record, dict) else None
        envelope = bundle.get("dsseEnvelope") if isinstance(bundle, dict) else None
        material = bundle.get("verificationMaterial") if isinstance(bundle, dict) else None
        signatures = envelope.get("signatures") if isinstance(envelope, dict) else None
        certificate = material.get("certificate") if isinstance(material, dict) else None
        tlog_entries = material.get("tlogEntries") if isinstance(material, dict) else None
        if (
            not isinstance(signatures, list)
            or not signatures
            or not all(isinstance(item, dict) and item.get("sig") for item in signatures)
            or not isinstance(certificate, dict)
            or not certificate.get("rawBytes")
            or not isinstance(tlog_entries, list)
            or not tlog_entries
            or envelope.get("payloadType") != "application/vnd.in-toto+json"
        ):
            raise RuntimeError("GitHub attestation bundle lacks signed transparency material")
        encoded = str(envelope.get("payload") or "")
        try:
            statement = json.loads(base64.b64decode(encoded, validate=True))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("GitHub attestation carries an invalid DSSE payload") from exc
        subjects = statement.get("subject") if isinstance(statement, dict) else None
        if not isinstance(subjects, list) or not any(
            isinstance(subject, dict)
            and subject.get("name") == repository
            and (subject.get("digest") or {}).get("sha256") == match.group(1)
            for subject in subjects
        ):
            raise RuntimeError("GitHub attestation is not bound to the exact image digest")
        predicate_type = str(statement.get("predicateType") or "")
        predicate = statement.get("predicate")
        if predicate_type == "https://spdx.dev/Document/v2.3":
            if not isinstance(predicate, dict) or not predicate.get("packages"):
                raise RuntimeError("SPDX attestation contains no package inventory")
        elif predicate_type == "https://slsa.dev/provenance/v1":
            if not isinstance(predicate, dict) or not predicate.get("buildDefinition"):
                raise RuntimeError("SLSA attestation contains no build definition")
        predicates.add(predicate_type)
    return predicates


def _trivy_command() -> list[str]:
    """Return a host Trivy or an exact-digest official container invocation."""
    trivy = shutil.which("trivy")
    if trivy:
        return [trivy]

    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError(
            "trivy not found on PATH and docker is unavailable; "
            "install either for the Wan publication gate"
        )

    command = [docker, "run", "--rm"]
    docker_config = Path(os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
    if (docker_config / "config.json").is_file():
        command.extend(["--volume", f"{docker_config.resolve()}:/root/.docker:ro"])
    command.append(_TRIVY_CONTAINER_IMAGE)
    return command


def _scan_trivy_exact_digest(image_ref: str, *, subject: str) -> dict[str, int]:
    """Rerun Trivy against immutable source bytes immediately before copy."""

    completed = subprocess.run(
        [
            *_trivy_command(),
            "image",
            "--platform",
            "linux/amd64",
            "--scanners",
            "vuln,secret",
            "--severity",
            "CRITICAL",
            "--format",
            "json",
            "--quiet",
            "--exit-code",
            "0",
            image_ref,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"{subject} exact-digest Trivy scan failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("Results"), list):
        raise RuntimeError(f"{subject} exact-digest Trivy scan returned invalid JSON")
    vulnerabilities: list[dict[str, Any]] = []
    secrets: list[dict[str, Any]] = []
    for result in payload["Results"]:
        if not isinstance(result, dict):
            raise RuntimeError("Wan exact-digest Trivy result entry is invalid")
        vulnerabilities.extend(
            finding
            for finding in (result.get("Vulnerabilities") or [])
            if isinstance(finding, dict)
            and str(finding.get("Severity") or "").upper() == "CRITICAL"
        )
        secrets.extend(
            finding
            for finding in (result.get("Secrets") or [])
            if isinstance(finding, dict)
        )
    fixed = [
        item for item in vulnerabilities if str(item.get("FixedVersion") or "").strip()
    ]
    if fixed:
        raise RuntimeError(
            f"{subject} exact-digest Trivy scan found {len(fixed)} fixed CRITICAL vulnerabilities"
        )
    if secrets:
        raise RuntimeError(
            f"{subject} exact-digest Trivy scan found {len(secrets)} secret findings"
        )
    return {
        "critical_total": len(vulnerabilities),
        "critical_with_fix": len(fixed),
        "critical_unfixed": len(vulnerabilities) - len(fixed),
        "secrets": len(secrets),
    }


def _scan_wan_trivy_exact_digest(image_ref: str) -> dict[str, int]:
    """Rerun Trivy against the immutable Wan source bytes immediately before copy."""

    return _scan_trivy_exact_digest(image_ref, subject="Wan")


def _scan_content_agents_trivy_exact_digest(image_ref: str) -> dict[str, int]:
    """Rerun Trivy against immutable Content Agents bytes before public copy."""

    return _scan_trivy_exact_digest(image_ref, subject="Content Agents")


def verify_validated_publication(item: PublishItem) -> tuple[bool, str]:
    """Refuse to publish a tool that has no built, validated artifact yet.

    Licence eligibility and evidence are separate gates. A tool can be correctly
    classified `redistribution: public` and still have nothing whose bytes were
    ever scanned or whose capabilities were ever run on a GPU. Publishing that
    would put out an unearned claim, so it is refused by name here rather than
    left to fail incidentally when the tag turns out not to exist.
    """

    if item.tool not in images.PUBLICATION_QUARANTINE_TOOLS:
        return True, "not applicable"
    return False, (
        f"{item.tool} has no accepted image: it has not been built, payload "
        "scanned, or GPU validated. Publication is blocked until that evidence "
        "exists and the tool leaves images.PUBLICATION_QUARANTINE_TOOLS."
    )


def verify_gpu_accepted_publication_source(item: PublishItem) -> tuple[bool, str]:
    """Bind rebuilt GPU surfaces to the exact accepted public manifest."""

    accepted = images.GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS.get(item.tool)
    if accepted is None:
        return True, "not applicable"
    ok, digest = _crane_digest(item.source_ref)
    if not ok:
        return False, digest
    if digest != accepted:
        return False, (
            f"source digest {digest} is not the GPU-accepted digest {accepted}"
        )
    return True, f"exact GPU-accepted digest {digest}"


def verify_wan_publication_source(item: PublishItem) -> tuple[bool, str]:
    """Bind Wan publication to exact clean bytes plus SPDX/SLSA attestations."""

    if item.tool != "wan2-2":
        return True, "not applicable"
    try:
        digest_ok, digest = _crane_digest(item.source_ref)
        if not digest_ok:
            raise RuntimeError(digest)
        accepted = images.wan_accepted_image_manifest()
        accepted_digest = str(accepted.get("oci_digest") or "")
        if digest != accepted_digest:
            raise RuntimeError(
                "Wan source digest is not the immutable GPU-accepted digest "
                f"{accepted_digest}"
            )
        manifest = _crane_json(["manifest", item.source_ref])
        if manifest.get("manifests") is not None:
            raise RuntimeError("Wan source must be one scanned platform manifest")
        if not isinstance(manifest.get("config"), dict) or not isinstance(
            manifest.get("layers"), list
        ):
            raise RuntimeError("Wan source is not a complete container manifest")
        proof = accepted.get("single_gpu_proof")
        if not isinstance(proof, dict):
            raise RuntimeError("Wan accepted manifest has no single-GPU proof")
        if proof.get("gpu_count") != 1 or proof.get("observed_image_digest") != digest:
            raise RuntimeError("Wan single-GPU proof is not bound to the accepted digest")
        if set(proof.get("capabilities_exercised") or ()) != {
            "wan2.2_ti2v_5b_text_to_video",
            "wan2.2_decoded_mp4_validation",
        } or proof.get("deferred") != []:
            raise RuntimeError("Wan single-GPU proof does not cover the release capability")
        for key in (
            "artifact_sha256",
            "mp4_sha256",
            "rrd_sha256",
            "rrd_manifest_sha256",
            "runtime_inventory_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(proof.get(key) or "")) is None:
                raise RuntimeError(f"Wan single-GPU proof has no valid {key}")
        runtime_hash = str(accepted.get("runtime_requirements_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", runtime_hash) is None:
            raise RuntimeError("Wan accepted runtime requirements hash is invalid")
        for identity_name in ("source", "model", "tokenizer"):
            identity = accepted.get(identity_name)
            if not isinstance(identity, dict) or not identity.get("revision"):
                raise RuntimeError(
                    f"Wan accepted manifest has no pinned {identity_name} revision"
                )
        payload_scan = accepted.get("payload_scan")
        if (
            not isinstance(payload_scan, dict)
            or int(payload_scan.get("archives_scanned") or 0) <= 1
            or payload_scan.get("findings") != 0
        ):
            raise RuntimeError("Wan accepted payload-scan proof is invalid")
        vulnerability_scan = accepted.get("vulnerability_scan")
        if not isinstance(vulnerability_scan, dict):
            raise RuntimeError("Wan accepted manifest has no vulnerability scan")
        accepted_total = vulnerability_scan.get("critical_total")
        accepted_unfixed = vulnerability_scan.get("critical_unfixed")
        if (
            not isinstance(accepted_total, int)
            or accepted_total < 0
            or accepted_unfixed != accepted_total
        ):
            raise RuntimeError(
                "Wan accepted vulnerability totals are missing or inconsistent"
            )
        if vulnerability_scan.get("critical_with_fix") != 0:
            raise RuntimeError("Wan accepted image has fixed CRITICAL vulnerabilities")
        if vulnerability_scan.get("secrets") != 0:
            raise RuntimeError("Wan accepted vulnerability/secret scan is invalid")
        repository = _repository(item.source_ref)
        config = _crane_json(["config", f"{repository}@{digest}"])
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise RuntimeError("Wan accepted image is not a linux/amd64 image")
        required_predicates = set(
            (accepted.get("attestations") or {}).get("required_predicates") or ()
        )
        observed_predicates = _github_attestation_predicates(
            repository=repository, digest=digest
        )
        if not required_predicates or not required_predicates.issubset(
            observed_predicates
        ):
            raise RuntimeError("Wan exact digest lacks required SPDX/SLSA attestations")

        scan_script = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "scan_image_wan_payload.py"
        )
        digest_ref = f"{repository}@{digest}"
        scan = subprocess.run(
            [sys.executable, str(scan_script), digest_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if scan.returncode:
            detail = (scan.stderr or scan.stdout or "").strip()
            raise RuntimeError(detail or "Wan exact-digest payload scan failed")
        scan_result = json.loads(scan.stdout)
        if scan_result.get("status") != "pass" or scan_result.get("findings"):
            raise RuntimeError("Wan exact-digest payload scan did not pass cleanly")
        live_vulnerability_scan = _scan_wan_trivy_exact_digest(digest_ref)
        for field in ("critical_total", "critical_with_fix", "critical_unfixed", "secrets"):
            if live_vulnerability_scan[field] != vulnerability_scan.get(field):
                raise RuntimeError(
                    "Wan live Trivy result differs from the accepted manifest: "
                    f"{field} recorded {vulnerability_scan.get(field)!r}, "
                    f"live {live_vulnerability_scan[field]!r}"
                )
        return (
            True,
            f"exact accepted digest {digest}; payload and live Trivy clean; "
            f"SPDX+SLSA bound; residual unfixed CRITICAL findings disclosed: "
            f"{live_vulnerability_scan['critical_total']}",
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return False, str(exc)


def verify_ltx_publication_source(item: PublishItem) -> tuple[bool, str]:
    """Bind LTX publication to exact zero-payload bytes and real GPU evidence."""

    if item.tool != "ltx2":
        return True, "not applicable"
    try:
        digest_ok, digest = _crane_digest(item.source_ref)
        if not digest_ok:
            raise RuntimeError(digest)
        accepted = images.ltx2_accepted_image_manifest()
        accepted_digest = str(accepted.get("oci_digest") or "")
        if digest != accepted_digest:
            raise RuntimeError(
                "LTX source digest is not the immutable GPU-accepted digest "
                f"{accepted_digest}"
            )
        proof = accepted.get("gpu_proof")
        if not isinstance(proof, dict):
            raise RuntimeError("LTX accepted manifest has no GPU proof")
        if proof.get("gpu_count") != 1 or proof.get("observed_image_digest") != digest:
            raise RuntimeError("LTX GPU proof is not bound to the accepted digest")
        required_capabilities = {
            "ltx2_5_text_to_video",
            "ltx2_5_decoded_mp4_validation",
        }
        if set(proof.get("capabilities_exercised") or ()) != required_capabilities:
            raise RuntimeError("LTX GPU proof does not cover the release capabilities")
        if proof.get("deferred") != [] or proof.get("source_baked") is not False or proof.get("weights_baked") is not False:
            raise RuntimeError("LTX GPU proof weakens the zero-payload capability claim")
        for key in ("artifact_sha256", "refusal_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(proof.get(key) or "")) is None:
                raise RuntimeError(f"LTX GPU proof has no valid {key}")
        video = proof.get("video")
        if (
            not isinstance(video, dict)
            or re.fullmatch(r"[0-9a-f]{64}", str(video.get("sha256") or "")) is None
            or int(video.get("width") or 0) <= 0
            or int(video.get("height") or 0) <= 0
            or int(video.get("frame_count") or 0) < 24
            or int(video.get("size_bytes") or 0) < 4096
        ):
            raise RuntimeError("LTX GPU video proof is invalid")
        for identity_name, revision_key in (("source", "revision"), ("weights", "resolved_revision")):
            identity = accepted.get(identity_name)
            if (
                not isinstance(identity, dict)
                or re.fullmatch(r"[0-9a-f]{40}", str(identity.get(revision_key) or "")) is None
                or identity.get("delivery") != "operator-entitled-runtime-fetch"
            ):
                raise RuntimeError(f"LTX accepted {identity_name} identity is invalid")
        repository = _repository(item.source_ref)
        required_predicates = set((accepted.get("attestations") or {}).get("required_predicates") or ())
        observed_predicates = _github_attestation_predicates(
            repository=repository, digest=digest
        )
        if not required_predicates or not required_predicates.issubset(observed_predicates):
            raise RuntimeError("LTX exact digest lacks required SPDX/SLSA attestations")
        config = _crane_json(["config", f"{repository}@{digest}"])
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise RuntimeError("LTX accepted image is not a single linux/amd64 image")
        scan_script = Path(__file__).resolve().parents[3] / "scripts" / "scan_image_ltx_payload.py"
        scan = subprocess.run(
            [sys.executable, str(scan_script), f"{repository}@{digest}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if scan.returncode:
            detail = (scan.stderr or scan.stdout or "").strip()
            raise RuntimeError(detail or "LTX exact-digest payload scan failed")
        scan_result = json.loads(scan.stdout)
        if scan_result.get("status") != "pass" or scan_result.get("findings"):
            raise RuntimeError("LTX exact-digest payload scan did not pass cleanly")
        trivy = _scan_wan_trivy_exact_digest(f"{repository}@{digest}")
        return (
            True,
            f"exact accepted digest {digest}; zero-payload and live Trivy clean; "
            f"SPDX+SLSA bound; residual unfixed CRITICAL findings disclosed: "
            f"{trivy['critical_total']}",
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return False, str(exc)


def _scan_content_agents_payload_exact_digest(
    image_ref: str, *, expected_source_sha: str
) -> dict[str, dict[str, int]]:
    """Run both byte scanners against the immutable Content Agents source."""

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    with tempfile.TemporaryDirectory(prefix="npa-content-agents-publication-scan-") as tmp:
        specialized = subprocess.run(
            [
                sys.executable,
                str(scripts / "scan_content_agents_image.py"),
                image_ref,
                "--expected-npa-source-sha",
                expected_source_sha,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if specialized.returncode:
            detail = (specialized.stderr or specialized.stdout or "").strip()
            raise RuntimeError(detail or "Content Agents exact byte scan failed")
        specialized_result = json.loads(specialized.stdout)
        if specialized_result.get("status") != "pass" or specialized_result.get(
            "findings"
        ):
            raise RuntimeError("Content Agents exact byte scan did not pass cleanly")

        general_path = Path(tmp) / "general.json"
        general = subprocess.run(
            [
                sys.executable,
                str(scripts / "scan_image_omniverse_payload.py"),
                image_ref,
                "--json",
                str(general_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if general.returncode:
            detail = (general.stderr or general.stdout or "").strip()
            raise RuntimeError(detail or "Content Agents general payload scan failed")
        general_result = json.loads(general_path.read_text(encoding="utf-8"))
        if (
            general_result.get("verdict") != "clean"
            or not general_result.get("scan_complete")
            or general_result.get("payload_hits")
            or general_result.get("history_hits")
        ):
            raise RuntimeError(
                "Content Agents general exact-digest payload scan did not pass cleanly"
            )

    return {
        "specialized": {
            "archives_scanned": int(specialized_result.get("archives_scanned") or 0),
            "findings": len(specialized_result.get("findings") or []),
        },
        "general": {
            "entries_scanned": int(general_result.get("entries_scanned") or 0),
            "payload_hits": len(general_result.get("payload_hits") or []),
            "history_hits": len(general_result.get("history_hits") or []),
            "weight_shaped_paths": len(
                general_result.get("weight_shaped_paths") or []
            ),
        },
    }


def verify_content_agents_publication_source(item: PublishItem) -> tuple[bool, str]:
    """Bind Content Agents publication to clean bytes and the accepted RTX run."""

    if item.tool != "content-agents":
        return True, "not applicable"
    try:
        accepted = images.content_agents_accepted_image_manifest()
        digest_ok, index_digest = _crane_digest(item.source_ref)
        if not digest_ok:
            raise RuntimeError(index_digest)
        accepted_digest = str(accepted.get("oci_digest") or "")
        if index_digest != accepted_digest:
            raise RuntimeError(
                "Content Agents source digest is not the immutable RTX-accepted digest "
                f"{accepted_digest}"
            )

        index = _crane_json(["manifest", item.source_ref])
        manifests = index.get("manifests")
        if not isinstance(manifests, list):
            raise RuntimeError("Content Agents source tag is not an attested OCI index")
        platforms = [
            entry
            for entry in manifests
            if isinstance(entry, dict)
            and entry.get("platform") == {"architecture": "amd64", "os": "linux"}
        ]
        if len(platforms) != 1:
            raise RuntimeError(
                "Content Agents OCI index requires exactly one linux/amd64 manifest"
            )
        platform_digest = str(platforms[0].get("digest") or "")
        if platform_digest != accepted.get("amd64_manifest"):
            raise RuntimeError(
                "Content Agents linux/amd64 manifest is not the RTX-accepted digest"
            )
        bound_attestations = [
            entry
            for entry in manifests
            if isinstance(entry, dict)
            and (entry.get("annotations") or {}).get("vnd.docker.reference.type")
            == "attestation-manifest"
            and (entry.get("annotations") or {}).get("vnd.docker.reference.digest")
            == platform_digest
        ]
        allowed = {
            platform_digest,
            *(str(entry.get("digest") or "") for entry in bound_attestations),
        }
        if len(bound_attestations) != 1:
            raise RuntimeError(
                "Content Agents linux/amd64 manifest requires exactly one bound "
                "attestation manifest"
            )
        if any(
            not isinstance(entry, dict)
            or str(entry.get("digest") or "") not in allowed
            for entry in manifests
        ):
            raise RuntimeError(
                "Content Agents OCI index contains an unscanned/unattested extra manifest"
            )

        repository = _repository(item.source_ref)
        attestation_manifest = _crane_json(
            ["manifest", f"{repository}@{bound_attestations[0]['digest']}"]
        )
        layers = attestation_manifest.get("layers")
        if not isinstance(layers, list):
            raise RuntimeError("Content Agents attestation manifest has no layers")
        attestation_proof = accepted.get("attestations")
        if (
            not isinstance(attestation_proof, dict)
            or attestation_proof.get("manifest_count") != 1
            or attestation_proof.get("statement_count") != 2
            or attestation_proof.get("spdx") != 1
            or attestation_proof.get("slsa_provenance_v0_2") != 1
            or attestation_proof.get("bound_to_amd64_manifest") is not True
            or len(layers) != 2
        ):
            raise RuntimeError("Content Agents accepted attestation proof is invalid")
        statements: dict[str, dict[str, Any]] = {}
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            predicate_type = str(
                (layer.get("annotations") or {}).get("in-toto.io/predicate-type")
                or ""
            )
            if not predicate_type:
                continue
            statement = _crane_blob_json(repository, str(layer.get("digest") or ""))
            subjects = statement.get("subject") or []
            if not any(
                isinstance(subject, dict)
                and (subject.get("digest") or {}).get("sha256")
                == platform_digest.removeprefix("sha256:")
                for subject in subjects
            ):
                raise RuntimeError(
                    f"Content Agents {predicate_type} attestation is not bound to "
                    f"{platform_digest}"
                )
            if statement.get("predicateType") != predicate_type:
                raise RuntimeError(
                    f"Content Agents {predicate_type} attestation type disagrees"
                )
            statements[predicate_type] = statement
        spdx = statements.get("https://spdx.dev/Document")
        provenance = statements.get("https://slsa.dev/provenance/v0.2")
        if (
            not spdx
            or not provenance
            or set(statements)
            != {
                "https://spdx.dev/Document",
                "https://slsa.dev/provenance/v0.2",
            }
        ):
            raise RuntimeError(
                "Content Agents source requires bound SPDX and SLSA v0.2 attestations"
            )
        if not (spdx.get("predicate") or {}).get("packages"):
            raise RuntimeError("Content Agents SPDX attestation has no package inventory")
        provenance_predicate = provenance.get("predicate") or {}
        if not provenance_predicate.get("buildType") or not provenance_predicate.get(
            "materials"
        ):
            raise RuntimeError(
                "Content Agents SLSA v0.2 provenance has no build type/materials"
            )

        runtime = accepted.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("Content Agents accepted manifest has no runtime proof")
        if runtime.get("version") != "0.3.0.312915" or runtime.get(
            "delivery"
        ) != "anonymous-runtime-fetch-from-nvidia":
            raise RuntimeError("Content Agents accepted OVRTX delivery is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("lock_sha256") or "")) is None:
            raise RuntimeError("Content Agents accepted OVRTX lock hash is invalid")
        if runtime.get("cache_tier") != "configured-filesystem" or runtime.get(
            "reused_render_stages"
        ) != 3:
            raise RuntimeError("Content Agents accepted runtime-cache proof is invalid")

        proof = accepted.get("rtx_proof")
        if not isinstance(proof, dict):
            raise RuntimeError("Content Agents accepted manifest has no RTX proof")
        if proof.get("gpu_count") != 1 or proof.get("gpu_model") != (
            "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        ):
            raise RuntimeError("Content Agents accepted GPU target is invalid")
        if proof.get("observed_image_id_digest") != accepted_digest:
            raise RuntimeError(
                "Content Agents RTX proof did not observe the accepted image digest"
            )
        if proof.get("upstream_validation") != "pass":
            raise RuntimeError("Content Agents upstream validation did not pass")
        for count_name in (
            "material_render_count",
            "physics_render_count",
            "validation_render_count",
        ):
            if int(proof.get(count_name) or 0) <= 0:
                raise RuntimeError(f"Content Agents RTX proof has no {count_name}")
        for artifact_name in ("usd_sha256", "usdz_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(proof.get(artifact_name) or "")) is None:
                raise RuntimeError(f"Content Agents RTX proof has no {artifact_name}")
        rigid = proof.get("rigid_physics")
        if (
            not isinstance(rigid, dict)
            or rigid.get("rigid_body") is not True
            or rigid.get("collision") is not True
            or rigid.get("fixed") is not False
            or float(rigid.get("mass_or_density") or 0) <= 0
            or not 0.1 <= float(rigid.get("friction") or 0) <= 2.0
        ):
            raise RuntimeError("Content Agents accepted rigid-physics proof is invalid")

        specialized = accepted.get("payload_scan")
        general = accepted.get("general_payload_scan")
        vulnerability = accepted.get("vulnerability_scan")
        for name, record in (
            ("payload", specialized),
            ("general payload", general),
            ("vulnerability", vulnerability),
        ):
            if not isinstance(record, dict) or re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("report_sha256") or "")
            ) is None:
                raise RuntimeError(f"Content Agents accepted {name} scan is invalid")
        accepted_payload_counts = {
            "specialized": {
                "archives_scanned": int(specialized.get("archives_scanned") or 0),
                "findings": int(specialized.get("findings") or 0),
            },
            "general": {
                key: int(general.get(key) or 0)
                for key in (
                    "entries_scanned",
                    "payload_hits",
                    "history_hits",
                    "weight_shaped_paths",
                )
            },
        }
        if (
            accepted_payload_counts["specialized"]["archives_scanned"] <= 1
            or accepted_payload_counts["specialized"]["findings"] != 0
            or accepted_payload_counts["general"]["entries_scanned"] <= 0
            or accepted_payload_counts["general"]["payload_hits"] != 0
            or accepted_payload_counts["general"]["history_hits"] != 0
        ):
            raise RuntimeError("Content Agents accepted payload counts are unsafe")
        expected_source_sha = str(accepted.get("implementation_revision") or "")
        if re.fullmatch(r"[0-9a-f]{40}", expected_source_sha) is None:
            raise RuntimeError("Content Agents accepted implementation revision is invalid")
        live_payload_counts = _scan_content_agents_payload_exact_digest(
            f"{repository}@{index_digest}", expected_source_sha=expected_source_sha
        )
        if live_payload_counts != accepted_payload_counts:
            raise RuntimeError(
                "Content Agents live payload scans disagree with accepted counts: "
                f"live={live_payload_counts}, accepted={accepted_payload_counts}"
            )
        live_vulnerability = _scan_content_agents_trivy_exact_digest(
            f"{repository}@{index_digest}"
        )
        accepted_vulnerability = {
            key: int(vulnerability.get(key) or 0)
            for key in ("critical_total", "critical_with_fix", "secrets")
        }
        if live_vulnerability != accepted_vulnerability:
            raise RuntimeError(
                "Content Agents live Trivy result disagrees with accepted counts: "
                f"live={live_vulnerability}, accepted={accepted_vulnerability}"
            )
        return (
            True,
            f"exact accepted digest {index_digest}; two payload scans and live Trivy "
            f"clean; SPDX+SLSA v0.2 bound to {platform_digest}; RTX renders "
            f"{proof['material_render_count']}/{proof['physics_render_count']}/"
            f"{proof['validation_render_count']}",
        )
    except (KeyError, StopIteration, TypeError, ValueError, RuntimeError) as exc:
        return False, str(exc)


def verify_bootstrap_publication_source(item: PublishItem) -> tuple[bool, str]:
    """Require a digest-bound SkyPilot attestation before any public tag write."""

    from npa.orchestration.skypilot.image_bootstrap_contract import (
        CONTRACT_VERSION,
        verify_attestation,
    )

    match = re.search(r"@(sha256:[0-9a-f]{64})$", item.source_ref)
    if match is None:
        return False, "publication source is not pinned by immutable digest"
    digest = match.group(1)
    try:
        config = _crane_json(["config", item.source_ref])
        nested = config.get("config")
        nested = nested if isinstance(nested, dict) else {}
        labels = nested.get("Labels")
        labels = labels if isinstance(labels, dict) else {}
        evidence = verify_attestation(
            image=item.source_ref,
            digest=digest,
            labels=labels,
        )
        if not evidence.ok or evidence.digest != digest:
            raise RuntimeError(
                evidence.detail or "bootstrap attestation is incompatible"
            )
        return True, f"{CONTRACT_VERSION} bound to {digest}"
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return False, str(exc)


def _crane_manifest_readable(
    ref: str, *, timeout: float = _PREFLIGHT_TIMEOUT_SECONDS
) -> tuple[bool, str]:
    """Whether ``ref`` can be read with the ambient registry credentials."""
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError(
            "crane not found on PATH; install go-containerregistry crane"
        )
    try:
        completed = subprocess.run(
            [crane, "manifest", ref],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:g}s"
    if completed.returncode == 0:
        return True, "ok"
    # crane puts the registry's reason on stderr; keep the last line, which is the
    # useful one (UNAUTHORIZED vs MANIFEST_UNKNOWN distinguishes a dead token from a
    # tag that was never pushed, and those need different fixes).
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return False, detail[-1] if detail else f"crane exited {completed.returncode}"


def preflight_sources(plan: list[PublishItem]) -> list[tuple[PublishItem, str]]:
    """Return the plan items whose SOURCE image cannot be read, with the reason."""
    failures: list[tuple[PublishItem, str]] = []
    for item in plan:
        # Defence in depth: build_plan already drops unvalidated tools, so
        # reaching here means something reintroduced one. Refuse for the right
        # reason rather than letting it fail as a missing tag.
        ok, detail = verify_validated_publication(item)
        if not ok:
            detail = f"UNVALIDATED — {detail}"
        if ok:
            ok, detail = _crane_manifest_readable(item.source_ref)
        if ok and item.tool in images.GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS:
            ok, detail = verify_gpu_accepted_publication_source(item)
            detail = f"GPU ACCEPTANCE GATE — {detail}"
        if ok and item.tool in images.SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS:
            ok, detail = verify_bootstrap_publication_source(item)
            detail = f"BOOTSTRAP GATE — {detail}"
        if ok and item.tool == "wan2-2":
            ok, detail = verify_wan_publication_source(item)
            detail = f"WAN GATE — {detail}"
        if ok and item.tool == "content-agents":
            ok, detail = verify_content_agents_publication_source(item)
            detail = f"CONTENT AGENTS GATE — {detail}"
        if ok and item.tool == "ltx2":
            ok, detail = verify_ltx_publication_source(item)
            detail = f"LTX GATE — {detail}"
        print(f"  {item.source_ref}  {'ok' if ok else f'UNREADABLE — {detail}'}")
        if not ok:
            failures.append((item, detail))
    return failures


# The plan is derived from the packaging contract, which states what the repo BUILDS -- an
# intent. The registry holds what someone actually pushed. Those diverge whenever a new tool
# lands (Dockerfile, contract entry and pin merged) before its image is built, and the repo
# has no build-and-push automation, so the divergence is normal rather than exceptional.
#
# The two states need opposite handling, which is why they are classified rather than lumped
# into "unreadable". A never-pushed image is a legitimate state that must not hold the ready
# images hostage; a credential or role problem must NEVER be skipped past, because skipping
# it would silently shrink the published set and look like success.
_MISSING_MARKERS = ("NAME_UNKNOWN", "MANIFEST_UNKNOWN")
_DENIED_MARKERS = ("UNAUTHORIZED", "DENIED", "FORBIDDEN")


def classify_preflight_failure(detail: str) -> str:
    """``"missing"``, ``"denied"`` or ``"other"`` for a preflight failure reason.

    ``missing`` means the registry answered authoritatively that it has no such repository
    (``NAME_UNKNOWN``) or no such tag (``MANIFEST_UNKNOWN``) -- the image was never pushed.
    ``denied`` covers anything that is about the identity, which is never skippable.
    """
    upper = detail.upper()
    # Denial wins over absence: a registry that hides repositories behind authz can answer
    # NAME_UNKNOWN for something that exists but is not visible to this identity, and
    # treating that as "not built yet" would quietly drop a publishable image.
    if any(marker in upper for marker in _DENIED_MARKERS):
        return "denied"
    if any(marker in upper for marker in _MISSING_MARKERS):
        return "missing"
    return "other"


# --------------------------------------------------------------------------------------
# Anonymous pullability
#
# Official development and release tags must already belong to public GHCR packages.
# Without the check below, a registry write could report success while consumers cannot
# pull the result. The unauthenticated HTTP path deliberately ignores ambient credentials.
# --------------------------------------------------------------------------------------

_ANON_TIMEOUT_SECONDS = 30


def _registry_host(ref: str) -> str:
    return ref.split("/", 1)[0]


def anonymous_pull_ok(
    ref: str, *, timeout: float = _ANON_TIMEOUT_SECONDS
) -> tuple[bool, str]:
    """Whether ``ref`` can be pulled with NO credentials at all.

    This is the property that actually matters to an external consumer, and the only one
    that distinguishes "pushed" from "published". Implemented with plain HTTP rather than
    a docker/crane call so it cannot accidentally reuse an ambient login and report a
    private package as public -- the whole point is to check the unauthenticated path.
    """
    host = _registry_host(ref)
    remainder = ref[len(host) + 1 :]
    repository, _, reference = remainder.rpartition(":")
    if not repository:  # digest-style or malformed
        repository, reference = remainder, "latest"

    token = ""
    if host == "ghcr.io":
        # GHCR usually hands an anonymous bearer token to anyone and lets the manifest
        # request decide. But when the package does not exist or is private it can refuse
        # at the token endpoint instead, so a 401/403 here is a verdict about the package,
        # not a transient failure -- report it as such rather than as "could not get a
        # token", which reads like a network problem and invites a pointless retry.
        try:
            url = f"https://ghcr.io/token?scope=repository:{repository}:pull&service=ghcr.io"
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                token = json.loads(response.read()).get("token", "")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, (
                    f"HTTP {exc.code} on the anonymous token request — the package is "
                    f"private or does not exist yet"
                )
            return False, f"token request failed: HTTP {exc.code}"
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            return False, f"token request failed: {exc}"

    request = urllib.request.Request(  # noqa: S310 - https registry API
        f"https://{host}/v2/{repository}/manifests/{reference}",
        method="GET",
        headers={
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status == 200, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code in (401, 403):
            hint = " (package is private — set its visibility to Public in the package settings)"
        return False, f"HTTP {exc.code}{hint}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, f"unreachable: {exc}"


def verify_public(plan: list[PublishItem]) -> list[tuple[PublishItem, str]]:
    """Return the plan items that are NOT anonymously pullable, with the reason."""
    failures: list[tuple[PublishItem, str]] = []
    for item in plan:
        ok, detail = anonymous_pull_ok(item.target_ref)
        status = "public" if ok else f"NOT PUBLIC — {detail}"
        print(f"  {item.target_ref}  {status}")
        if not ok:
            failures.append((item, detail))
    return failures


def anonymous_digest(
    ref: str, *, timeout: float = _ANON_TIMEOUT_SECONDS
) -> tuple[bool, str]:
    """Resolve an OCI manifest digest without consulting ambient credentials."""

    host = _registry_host(ref)
    remainder = ref[len(host) + 1 :]
    repository, _, reference = remainder.rpartition(":")
    if not repository:
        return False, "release reference must use a tag"

    token = ""
    if host == "ghcr.io":
        try:
            url = f"https://ghcr.io/token?scope=repository:{repository}:pull&service=ghcr.io"
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                token = json.loads(response.read()).get("token", "")
        except urllib.error.HTTPError as exc:
            return False, f"anonymous token request failed: HTTP {exc.code}"
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            return False, f"anonymous token request failed: {exc}"

    request = urllib.request.Request(  # noqa: S310 - https registry API
        f"https://{host}/v2/{repository}/manifests/{reference}",
        method="GET",
        headers={
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
            digest = str(response.headers.get("Docker-Content-Digest") or "").strip()
            if not digest:
                digest = "sha256:" + hashlib.sha256(body).hexdigest()
            if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                return False, f"registry returned invalid digest {digest!r}"
            return True, digest
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, f"unreachable: {exc}"


def accepted_release_plan(*, target_registry: str) -> list[PublishItem]:
    """Return exact accepted release claims, independent of development tags."""

    target = images._ghcr_namespace(
        target_registry, channel="accepted public release verification"
    ).rstrip("/")
    releases = images.public_release_manifest()["releases"]
    return [
        PublishItem(
            tool=tool,
            source_ref=(
                f"{target}/{CONTAINER_IMAGE_NAMES[tool]}@{entry['published_digest']}"
            ),
            target_ref=f"{target}/{CONTAINER_IMAGE_NAMES[tool]}:{entry['tag']}",
        )
        for tool, entry in sorted(releases.items())
    ]


def verify_accepted_releases(
    plan: list[PublishItem],
) -> list[tuple[PublishItem, str]]:
    """Compare each anonymous release digest with its recorded published digest."""

    failures: list[tuple[PublishItem, str]] = []
    for item in plan:
        expected = item.source_ref.rpartition("@")[2]
        ok, detail = anonymous_digest(item.target_ref)
        if not ok:
            failure = f"anonymous release digest unreadable — {detail}"
        elif detail != expected:
            failure = f"published digest mismatch — recorded {expected}; live {detail}"
        else:
            print(f"  {item.target_ref}  accepted ({detail})")
            continue
        print(f"  {item.target_ref}  DRIFTED — {failure}")
        failures.append((item, failure))
    return failures


def verify_parity(plan: list[PublishItem]) -> list[tuple[PublishItem, str]]:
    """Return items whose source and target do not resolve to identical OCI bytes.

    Anonymous pullability proves that consumers can fetch a package, but it does not
    prove that the target still serves the image pinned by ``main``. The plan is
    expected to have passed source preflight already, so a source read failure here is
    a blocking registry fault rather than a reason to silently omit the comparison.
    """

    failures: list[tuple[PublishItem, str]] = []
    for item in plan:
        source_ok, source_detail = _crane_digest(item.source_ref)
        if not source_ok:
            detail = f"source digest unreadable — {source_detail}"
        else:
            target_ok, target_detail = _crane_digest(item.target_ref)
            if not target_ok:
                detail = f"target digest unreadable — {target_detail}"
            elif target_detail != source_detail:
                detail = (
                    f"digest mismatch — source {source_detail}; target {target_detail}"
                )
            else:
                print(f"  {item.target_ref}  current ({source_detail})")
                continue
        print(f"  {item.target_ref}  DRIFTED — {detail}")
        failures.append((item, detail))
    return failures


def _crane_digest(
    ref: str, *, timeout: float = _PREFLIGHT_TIMEOUT_SECONDS
) -> tuple[bool, str]:
    """Return ``(True, digest)`` or ``(False, registry error)`` for ``ref``.

    ``crane copy`` is content-addressed, but invoking it for every image still walks and
    negotiates every manifest and layer. Comparing the top-level manifest digest first
    lets repeat workflow runs prove that a tag is already current without writing it
    again. The source preflight remains separate and complete: incrementality must not
    weaken the credential, pin, or licensing gates.
    """
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError(
            "crane not found on PATH; install go-containerregistry crane"
        )
    try:
        completed = subprocess.run(
            [crane, "digest", ref],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:g}s"
    if completed.returncode == 0:
        output = (completed.stdout or "").strip().splitlines()
        if output:
            return True, output[-1]
        return False, "crane returned an empty digest"
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return False, detail[-1] if detail else f"crane exited {completed.returncode}"


def _pin_publication_sources(
    plan: list[PublishItem],
) -> tuple[list[PublishItem], list[tuple[PublishItem, str]]]:
    """Resolve every mutable source once before license/attestation gates.

    All later inspection and copying uses the returned immutable references, so
    moving a source tag cannot swap bytes between validation and publication.
    """

    pinned: list[PublishItem] = []
    failures: list[tuple[PublishItem, str]] = []
    for item in plan:
        ok, detail = _crane_digest(item.source_ref)
        if not ok:
            failures.append((item, detail))
            continue
        if re.fullmatch(r"sha256:[0-9a-f]{64}", detail) is None:
            failures.append((item, f"registry returned invalid digest {detail!r}"))
            continue
        pinned.append(
            replace(item, source_ref=f"{_repository(item.source_ref)}@{detail}")
        )
    return pinned, failures


def _crane_copy(item: PublishItem) -> bool:
    """Copy ``item`` only when the target is absent or has a different digest.

    Returns ``True`` when a copy ran and ``False`` when the exact source digest was
    already present. Only authoritative tag absence may proceed to a copy. Denial,
    transient, and unknown failures stop because official packages must already be public.
    """
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError(
            "crane not found on PATH; install go-containerregistry crane"
        )

    if re.search(r"@sha256:[0-9a-f]{64}$", item.source_ref) is None:
        raise RuntimeError("publication source must be pinned by exact OCI digest")

    source_ok, source_detail = _crane_digest(item.source_ref)
    if not source_ok:
        raise RuntimeError(
            f"could not resolve source digest for {item.source_ref}: {source_detail}"
        )

    target_ok, target_detail = _crane_digest(item.target_ref)
    if target_ok and target_detail == source_detail:
        print(f"Already current; skipping copy: {item.target_ref} ({source_detail})")
        return False
    if target_ok:
        print(
            f"Digest changed; copying {item.target_ref}: "
            f"{target_detail} -> {source_detail}"
        )
    else:
        failure_kind = classify_preflight_failure(target_detail)
        if failure_kind != "missing":
            raise RuntimeError(
                f"could not determine target digest for {item.target_ref}: {target_detail}; "
                "refusing to copy because the official package is not proven public"
            )
        print(
            f"Target absent or unreadable ({target_detail}); copying {item.target_ref}"
        )

    subprocess.run([crane, "copy", item.source_ref, item.target_ref], check=True)
    copied_ok, copied_detail = _crane_digest(item.target_ref)
    if not copied_ok or copied_detail != source_detail:
        raise RuntimeError(
            f"copied target digest does not match source {source_detail}: "
            f"{copied_detail}"
        )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=public_container_registry(),
        help="Target public registry (e.g. ghcr.io/nebius/nebius-physical-ai); "
        "defaults to $NPA_PUBLIC_REGISTRY.",
    )
    parser.add_argument(
        "--development-sha",
        default=None,
        help=(
            "Full Git SHA whose immutable public dev-<sha> image is promoted. "
            "Defaults to $NPA_DEVELOPMENT_SHA, then the checked-out HEAD."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without copying."
    )
    parser.add_argument(
        "--verify-public",
        action="store_true",
        help=(
            "Do not copy. Check that every planned release target is pullable with NO "
            "credentials, and exit non-zero if any is not."
        ),
    )
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help=(
            "Do not copy. Preflight every planned source and compare its immutable OCI "
            "digest with the target tag. Exit non-zero when a target is absent, unreadable, "
            "or serves different bytes. Use --skip-missing to omit source images that have "
            "not been built yet."
        ),
    )
    parser.add_argument(
        "--verify-accepted-releases",
        action="store_true",
        help=(
            "Do not copy. Resolve every recorded release tag anonymously and compare "
            "it directly with its accepted published_digest. This does not require "
            "historical development tags."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Do not copy. Read every SOURCE manifest with the ambient registry credentials "
            "and exit non-zero if any is unreadable. Proves the source token works (crane "
            "auth login does not) and that every pinned tag exists, before anything is "
            "written. The copy path runs this automatically."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Operate on the images that exist, skipping any public development tag that does not exist "
            "yet (NAME_UNKNOWN / MANIFEST_UNKNOWN), and report exactly which were skipped. "
            "The plan comes from the packaging contract, which records what this repo BUILDS, "
            "so a tool that landed before its image was built otherwise blocks every ready "
            "image. A denial is never skipped — that is a credential or role fault."
        ),
    )
    args = parser.parse_args(argv)

    if not (args.target or "").strip():
        parser.error("no target registry; pass --target or set NPA_PUBLIC_REGISTRY")

    if args.verify_accepted_releases:
        expected = accepted_release_plan(target_registry=args.target)
        print(f"Verifying {len(expected)} accepted public release digest(s):")
        failures = verify_accepted_releases(expected)
        if failures:
            print(
                f"\n{len(failures)} of {len(expected)} accepted release(s) drifted.",
                file=sys.stderr,
            )
            return 1
        print(f"\nAll {len(expected)} accepted release digest(s) match anonymously.")
        return 0

    plan = build_publish_plan(
        target_registry=args.target,
        development_git_sha=args.development_sha,
    )
    restricted = restricted_image_names()
    print(f"Publishing {len(plan)} OSS image(s) to {args.target.rstrip('/')}")
    if restricted:
        print(
            "Excluded (bakes a runtime we may not redistribute): "
            + ", ".join(restricted)
        )
    else:
        # Don't print a dangling "Excluded: " with nothing after it, and say why the
        # list is empty — an operator reading this needs to know that the Isaac images
        # being absent from the exclusion list is intended, not an oversight.
        print(
            "Excluded: none — every current workbench image is classified for public "
            "redistribution. Vendor runtimes and gated assets use verified runtime-fetch "
            "boundaries where required; rebuilt standalone images contain only their "
            "recorded redistributable payloads."
        )
    for item in plan:
        print(f"  {item.source_ref}  ->  {item.target_ref}")
    if args.verify_public:
        expected = plan
        if args.skip_missing:
            # Otherwise the checklist would list packages for images that were never copied,
            # linking to settings pages that 404 -- and the run would report "not public"
            # about something nobody tried to publish.
            print("\nPreflighting source images to exclude the ones not built yet:")
            expected = _preflight_or_explain(plan, skip_missing=True)
            if not expected:
                return 1
        print("\nVerifying anonymous (unauthenticated) pullability:")
        failures = verify_public(expected)
        if failures:
            _explain_nonpublic_packages(failures, total=len(expected))
            return 1
        print(f"\nAll {len(expected)} image(s) are publicly pullable.")
        return 0

    if args.verify_parity:
        print("\nPreflighting source images before the digest comparison:")
        expected = _preflight_or_explain(plan, skip_missing=args.skip_missing)
        if not expected:
            return 1
        print("\nComparing source and target OCI digests:")
        failures = verify_parity(expected)
        if failures:
            print(
                f"\n{len(failures)} of {len(expected)} image(s) are not at parity; "
                "run the guarded publisher to copy the exact source digests.",
                file=sys.stderr,
            )
            return 1
        print(f"\nAll {len(expected)} image(s) have exact source/target digest parity.")
        return 0

    if args.preflight:
        print("\nPreflighting source images (authenticated read, nothing written):")
        return 0 if _preflight_or_explain(plan, skip_missing=args.skip_missing) else 1

    if args.dry_run:
        print("(dry run — nothing copied)")
        return 0

    print("\nPreflighting source images (authenticated read, nothing written):")
    publishable = _preflight_or_explain(plan, skip_missing=args.skip_missing)
    if not publishable:
        return 1
    copied = 0
    already_current = 0
    for item in publishable:
        # Existing tests and third-party instrumentation historically returned ``None``
        # from _crane_copy; count every result except the explicit incremental ``False``
        # as a copy for backwards-compatible instrumentation.
        if _crane_copy(item) is False:
            already_current += 1
        else:
            copied += 1
    _mark_copy_phase_complete()
    print(f"\nCopied {copied} image(s).")
    print(f"Skipped {already_current} already-current image(s).")

    # Copying is not publishing, so do not stop here and report success. Verifying
    # inline means the operator learns the real state -- and gets the click-through
    # list for the one step that cannot be automated -- from the command that did
    # the copy, rather than from a separate invocation they have to know to run.
    print("\nVerifying anonymous (unauthenticated) pullability:")
    failures = verify_public(publishable)
    if failures:
        _explain_nonpublic_packages(failures, total=len(publishable), after_copy=True)
        return 1
    print(f"\nAll {len(publishable)} image(s) are publicly pullable.")
    return 0


def _preflight_or_explain(
    plan: list[PublishItem], *, skip_missing: bool = False
) -> list[PublishItem]:
    """Run the source preflight and return the items that are safe to copy.

    Returns an empty list when the run must stop. With ``skip_missing`` the never-pushed
    images are dropped from the returned set instead of failing the run, so a young tool
    whose image has not been built yet cannot block the images that are ready.
    """
    pinned_plan, resolution_failures = _pin_publication_sources(plan)
    failures = resolution_failures + preflight_sources(pinned_plan)
    if not failures:
        return pinned_plan

    by_kind: dict[str, list[tuple[PublishItem, str]]] = {}
    for item, detail in failures:
        by_kind.setdefault(classify_preflight_failure(detail), []).append(
            (item, detail)
        )
    missing = by_kind.get("missing", [])
    blocking = by_kind.get("denied", []) + by_kind.get("other", [])

    if skip_missing and missing and not blocking:
        print(
            f"\nSkipping {len(missing)} of {len(plan)} image(s) that are not in the source "
            "registry yet (--skip-missing). The packaging contract records what this repo\n"
            "builds; these have not been built and pushed, so there is nothing to mirror:",
            file=sys.stderr,
        )
        for item, detail in missing:
            print(f"  {item.source_ref}  ({_missing_reason(detail)})", file=sys.stderr)
        print(
            "Build and push them, then re-run to add them to the mirror. Until then they are\n"
            "absent from the public registry and will fail at pull time for consumers.",
            file=sys.stderr,
        )
        skipped_tools = {item.tool for item, _ in missing}
        return [item for item in pinned_plan if item.tool not in skipped_tools]

    lines = [
        f"\n{len(failures)} of {len(plan)} source image(s) could not be read; nothing was "
        "copied."
    ]
    # Whether EVERY read failed is the diagnostic, not just the registry's error code: one
    # UNAUTHORIZED could be a per-repository grant, but all of them means the credential
    # never resolved to an identity at all ("failed to get profile"), so there is no point
    # looking at roles or at the tags.
    if len(failures) == len(plan) and all(
        classify_preflight_failure(detail) == "denied" for _, detail in failures
    ):
        lines.append(
            "Every read was denied, so this is the credential rather than any single tag or\n"
            "grant — the token did not resolve to an identity.\n"
            "In CI, grant the workflow GITHUB_TOKEN package access to the public\n"
            "development packages.\n"
            "Locally, log in with a GHCR package token and retry:\n"
            "  printf '%s' \"$GHCR_TOKEN\" | crane auth login ghcr.io -u \"$GHCR_USER\" "
            "--password-stdin"
        )
    else:
        if blocking:
            lines.append(
                f"{len(blocking)} image(s) failed for a reason that is NOT absence, so nothing "
                "was skipped:"
            )
            lines.extend(f"  {item.source_ref}  {detail}" for item, detail in blocking)
            lines.append(
                "A denial on some images means the credential works but its identity lacks\n"
                "viewer on those repositories — fix the role, not the token."
            )
        if missing:
            lines.append(
                f"{len(missing)} public development tag(s) do not exist:"
            )
            lines.extend(
                f"  {item.source_ref}  ({_missing_reason(detail)})"
                for item, detail in missing
            )
            lines.append(
                "Build and push those images, or correct their pins. To publish the rest now\n"
                "and add these once they are built, re-run with --skip-missing (or the\n"
                "workflow's skip_missing input)."
            )
    print("\n".join(lines), file=sys.stderr)
    return []


def _missing_reason(detail: str) -> str:
    """Which kind of absence, keeping the registry's own code so the line stays greppable.

    The two need different fixes: no repository at all means the image was never built,
    while a missing tag means something was pushed but the pin points elsewhere.
    """
    if "NAME_UNKNOWN" in detail.upper():
        return "NAME_UNKNOWN — no such repository; this image has never been pushed"
    if "MANIFEST_UNKNOWN" in detail.upper():
        return "MANIFEST_UNKNOWN — repository exists but not this tag; the pin points at an unpushed build"
    return detail


def _explain_nonpublic_packages(
    failures: list[tuple[PublishItem, str]], *, total: int, after_copy: bool = False
) -> None:
    lead = (
        "The release-tag copy completed, but public verification failed."
        if after_copy
        else "Public verification failed."
    )
    print(
        f"\n{len(failures)} of {total} image(s) are NOT publicly pullable.\n"
        f"{lead} Official development and release tags must be anonymously pullable.\n"
        "Stop publication and investigate package visibility or registry availability;\n"
        "do not treat an authenticated pull as public evidence.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
