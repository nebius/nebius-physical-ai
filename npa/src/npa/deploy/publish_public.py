"""Mirror the OSS-redistributable workbench images to a public registry.

Nebius Container Registry does not support anonymous/public pulls, so "public
exposure" of the workbench means mirroring the publicly-redistributable image
subset to a public-capable registry (e.g. GHCR ``ghcr.io/<org>/<repo>``).

This tool is license-guarded: it only ever copies tools reported by
``images.publicly_publishable_tools()`` and hard-refuses anything in
``images.OMNIVERSE_RESTRICTED_TOOLS`` as defence in depth around that selector.

The Isaac tools were re-architected to fetch Isaac Sim / Isaac Lab at first run
under the operator's own EULA acceptance and are publishable. The separately
contracted ``cosmos3-serving`` image remains build-your-own because its pinned
vendor base has distribution conditions that anonymous GHCR does not establish.

Example (dry run first, then execute):

    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai --dry-run
    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai

The copy path preflights the source registry first, skips targets whose manifest digest
already matches the source, and verifies the result after. A stale credential therefore
fails before anything is written, unchanged images are not recopied, and a copy cannot
report success while nothing is publicly pullable. Making the packages public is the one
step that cannot be automated (see ``package_settings_url``); the verification prints a
click-through list.

``--verify-parity`` is the read-only drift check for automation: it runs the same source
preflight and then requires every target tag to resolve to the exact same OCI digest. This
is deliberately stronger than ``--verify-public``, because an anonymously pullable tag can
still serve stale bytes.

``--describe-credential`` reads a source-registry credential on stdin and reports its
expiry offline, because the credential is what actually breaks: a Nebius access token
lives 12 hours, so anything stored in CI must be a static key issued for
CONTAINER_REGISTRY instead (see ``describe_credential``).

``--skip-missing`` publishes the images that exist when some pin refers to an image nobody
has built yet. The plan comes from the packaging contract (what the repo BUILDS) while the
registry holds what was pushed, so that gap is routine for a young tool; a denial is never
skipped (see ``classify_preflight_failure``).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from npa.deploy import images
from npa.deploy.images import (
    container_image_for_tool,
    is_publicly_redistributable,
    omniverse_restricted_image_names,
    primary_container_registry,
    public_container_registry,
    publicly_publishable_tools,
)


@dataclass(frozen=True)
class PublishItem:
    tool: str
    source_ref: str
    target_ref: str


def _mark_copy_phase_complete() -> None:
    """Tell GitHub Actions that every planned copy operation completed.

    The publish command can still exit non-zero after this point when GHCR created a
    private package and anonymous verification fails. Keeping that state separate from
    the process exit status prevents a pre-copy failure from producing irreversible
    package-visibility instructions.
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as output:
        output.write("copy_phase_completed=true\n")


def build_publish_plan(
    *,
    target_registry: str,
    source_registry: str | None = None,
) -> list[PublishItem]:
    """Return the (source -> target) copy plan for the public image subset.

    Raises ``ValueError`` if an Omniverse-restricted tool ever leaks into the
    plan (defense in depth around the license boundary).
    """
    if not target_registry.strip():
        raise ValueError("target_registry is required")
    source_registry = source_registry or primary_container_registry()
    target = target_registry.rstrip("/")

    plan: list[PublishItem] = []
    for tool in publicly_publishable_tools():
        # Licence eligibility and evidence are separate gates, and a tool can
        # pass the first while having no built artifact to have checked. Those
        # are excluded from the plan rather than failed during preflight,
        # because the plan is meant to read as "what we would hand to the
        # public" — and something that does not exist is not that.
        if tool in images.UNVALIDATED_PUBLICATION_TOOLS:
            continue
        # Read the restricted set through the module, never a from-import: a
        # defence-in-depth check that holds a stale copy of the thing it is defending is
        # worse than no check at all. (`from ... import OMNIVERSE_RESTRICTED_TOOLS` binds
        # the value at import time, so this guard and publicly_publishable_tools() could
        # disagree - which is exactly what a test caught once the set stopped being empty.)
        if (
            not is_publicly_redistributable(tool)
            or tool in images.OMNIVERSE_RESTRICTED_TOOLS
        ):
            raise ValueError(
                f"refusing to publish restricted (Omniverse Kit) tool {tool!r} to a public registry"
            )
        source_ref = container_image_for_tool(
            tool,
            registry=source_registry,
            tag=images.public_mirror_tag_for_tool(tool),
        )
        image = source_ref.rsplit("/", 1)[-1]  # npa-<tool>:<tag>
        plan.append(
            PublishItem(
                tool=tool, source_ref=source_ref, target_ref=f"{target}/{image}"
            )
        )
    return plan


def filter_publish_plan(
    plan: list[PublishItem], selected_tools: list[str]
) -> list[PublishItem]:
    """Narrow a guarded plan without permitting arbitrary image references."""

    if not selected_tools:
        return plan
    requested = {value.strip() for value in selected_tools if value.strip()}
    available = {item.tool for item in plan}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            "selected tool is not in the eligible public plan: " + ", ".join(unknown)
        )
    return [item for item in plan if item.tool in requested]


# --------------------------------------------------------------------------------------
# Source preflight
#
# `crane auth login` writes a config file and exits 0 for ANY token -- it never contacts
# the registry -- so a stale NEBIUS_CR_TOKEN looks like a successful login and only
# surfaces as a failed copy. And `crane copy` is a per-image subprocess with no
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


def _scan_wan_trivy_exact_digest(image_ref: str) -> dict[str, int]:
    """Rerun Trivy against the immutable Wan source bytes immediately before copy."""

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
        raise RuntimeError(detail or "Wan exact-digest Trivy scan failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("Results"), list):
        raise RuntimeError("Wan exact-digest Trivy scan returned invalid JSON")
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
            f"Wan exact-digest Trivy scan found {len(fixed)} fixed CRITICAL vulnerabilities"
        )
    if secrets:
        raise RuntimeError(
            f"Wan exact-digest Trivy scan found {len(secrets)} secret findings"
        )
    return {
        "critical_total": len(vulnerabilities),
        "critical_with_fix": len(fixed),
        "secrets": len(secrets),
    }


def verify_validated_publication(item: PublishItem) -> tuple[bool, str]:
    """Refuse to publish a tool that has no built, validated artifact yet.

    Licence eligibility and evidence are separate gates. A tool can be correctly
    classified `redistribution: public` and still have nothing whose bytes were
    ever scanned or whose capabilities were ever run on a GPU. Publishing that
    would put out an unearned claim, so it is refused by name here rather than
    left to fail incidentally when the tag turns out not to exist.
    """

    if item.tool not in images.UNVALIDATED_PUBLICATION_TOOLS:
        return True, "not applicable"
    return False, (
        f"{item.tool} has no accepted image: it has not been built, payload "
        "scanned, or GPU validated. Publication is blocked until that evidence "
        "exists and the tool leaves images.UNVALIDATED_PUBLICATION_TOOLS."
    )


def verify_wan_publication_source(item: PublishItem) -> tuple[bool, str]:
    """Bind Wan publication to exact clean bytes plus SPDX/SLSA attestations."""

    if item.tool != "wan2-2":
        return True, "not applicable"
    try:
        digest_ok, index_digest = _crane_digest(item.source_ref)
        if not digest_ok:
            raise RuntimeError(index_digest)
        accepted = images.wan_accepted_image_manifest()
        accepted_digest = str(accepted.get("oci_digest") or "")
        if index_digest != accepted_digest:
            raise RuntimeError(
                "Wan source digest is not the immutable GPU-accepted digest "
                f"{accepted_digest}"
            )
        index = _crane_json(["manifest", item.source_ref])
        manifests = index.get("manifests")
        if not isinstance(manifests, list):
            raise RuntimeError("Wan source tag is not an attested OCI index")
        platform = next(
            (
                entry
                for entry in manifests
                if isinstance(entry, dict)
                and entry.get("platform") == {"architecture": "amd64", "os": "linux"}
            ),
            None,
        )
        if platform is None:
            raise RuntimeError("Wan OCI index has no linux/amd64 platform manifest")
        platform_digest = str(platform.get("digest") or "")
        if platform_digest != accepted.get("amd64_manifest"):
            raise RuntimeError(
                "Wan linux/amd64 manifest is not the GPU-accepted platform digest"
            )
        for proof_name, gpu_count in (
            ("single_gpu_proof", 1),
            ("distributed_proof", 4),
        ):
            proof = accepted.get(proof_name)
            if not isinstance(proof, dict):
                raise RuntimeError(f"Wan accepted manifest has no {proof_name}")
            if proof.get("gpu_count") != gpu_count:
                raise RuntimeError(f"Wan {proof_name} GPU count is invalid")
            if proof.get("observed_image_id_digest") != accepted_digest:
                raise RuntimeError(
                    f"Wan {proof_name} did not observe the accepted image digest"
                )
            for key in ("run_id", "mp4_sha256", "rrd_sha256", "rrd_manifest_sha256"):
                if not proof.get(key):
                    raise RuntimeError(f"Wan {proof_name} has no {key}")
        runtime_hash = str(accepted.get("runtime_requirements_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", runtime_hash) is None:
            raise RuntimeError("Wan accepted runtime requirements hash is invalid")
        for identity_name in ("source", "model", "tokenizer"):
            identity = accepted.get(identity_name)
            if not isinstance(identity, dict) or not identity.get("revision"):
                raise RuntimeError(
                    f"Wan accepted manifest has no pinned {identity_name} revision"
                )
        runtime_acceptance = accepted.get("runtime_acceptance")
        if (
            not isinstance(runtime_acceptance, dict)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(runtime_acceptance.get("manifest_sha256") or "")
            )
            is None
        ):
            raise RuntimeError("Wan accepted runtime proof hash is invalid")
        payload_scan = accepted.get("payload_scan")
        if (
            not isinstance(payload_scan, dict)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(payload_scan.get("report_sha256") or "")
            )
            is None
            or int(payload_scan.get("archives_scanned") or 0) <= 1
            or payload_scan.get("findings") != 0
        ):
            raise RuntimeError("Wan accepted payload-scan proof is invalid")
        vulnerability_scan = accepted.get("vulnerability_scan")
        if not isinstance(vulnerability_scan, dict):
            raise RuntimeError("Wan accepted manifest has no vulnerability scan")
        if vulnerability_scan.get("critical_with_fix") != 0:
            raise RuntimeError("Wan accepted image has fixed CRITICAL vulnerabilities")
        if (
            vulnerability_scan.get("secrets") != 0
            or re.fullmatch(
                r"[0-9a-f]{64}", str(vulnerability_scan.get("report_sha256") or "")
            )
            is None
        ):
            raise RuntimeError("Wan accepted vulnerability/secret scan is invalid")
        bound_attestations = [
            entry
            for entry in manifests
            if isinstance(entry, dict)
            and (entry.get("annotations") or {}).get("vnd.docker.reference.type")
            == "attestation-manifest"
            and (entry.get("annotations") or {}).get("vnd.docker.reference.digest")
            == platform_digest
        ]
        allowed_manifest_digests = {
            platform_digest,
            *(str(entry.get("digest") or "") for entry in bound_attestations),
        }
        unexpected_manifests = [
            entry
            for entry in manifests
            if not isinstance(entry, dict)
            or str(entry.get("digest") or "") not in allowed_manifest_digests
        ]
        if len(bound_attestations) != 1:
            raise RuntimeError(
                "Wan linux/amd64 manifest requires exactly one bound attestation manifest"
            )
        if unexpected_manifests:
            raise RuntimeError(
                "Wan OCI index contains an unscanned/unattested extra manifest"
            )
        attestation = bound_attestations[0]
        repository = _repository(item.source_ref)
        attestation_manifest = _crane_json(
            ["manifest", f"{repository}@{attestation['digest']}"]
        )
        layers = attestation_manifest.get("layers")
        if not isinstance(layers, list):
            raise RuntimeError("Wan attestation manifest has no layers")
        statements: dict[str, dict[str, Any]] = {}
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            predicate_type = str(
                (layer.get("annotations") or {}).get("in-toto.io/predicate-type") or ""
            )
            if predicate_type:
                statement = _crane_blob_json(repository, str(layer.get("digest") or ""))
                subjects = statement.get("subject") or []
                if not any(
                    isinstance(subject, dict)
                    and (subject.get("digest") or {}).get("sha256")
                    == platform_digest.removeprefix("sha256:")
                    for subject in subjects
                ):
                    raise RuntimeError(
                        f"Wan {predicate_type} attestation is not bound to {platform_digest}"
                    )
                if statement.get("predicateType") != predicate_type:
                    raise RuntimeError(
                        f"Wan {predicate_type} attestation type disagrees"
                    )
                statements[predicate_type] = statement
        spdx = statements.get("https://spdx.dev/Document")
        provenance = statements.get("https://slsa.dev/provenance/v1")
        if not spdx or not provenance:
            raise RuntimeError(
                "Wan source requires bound SPDX and SLSA v1 attestations"
            )
        if not (spdx.get("predicate") or {}).get("packages"):
            raise RuntimeError("Wan SPDX attestation contains no package inventory")
        if not (provenance.get("predicate") or {}).get("buildDefinition"):
            raise RuntimeError("Wan SLSA provenance contains no build definition")

        scan_script = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "scan_image_wan_payload.py"
        )
        digest_ref = f"{repository}@{index_digest}"
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
        accepted_vulnerability_scan = {
            key: int(vulnerability_scan.get(key) or 0)
            for key in ("critical_total", "critical_with_fix", "secrets")
        }
        if live_vulnerability_scan != accepted_vulnerability_scan:
            raise RuntimeError(
                "Wan exact-digest live Trivy result disagrees with the accepted "
                f"disclosure: live={live_vulnerability_scan}, "
                f"accepted={accepted_vulnerability_scan}"
            )
        return (
            True,
            f"exact accepted digest {index_digest}; payload and live Trivy clean; "
            f"SPDX+SLSA bound "
            f"to {platform_digest}; residual unfixed CRITICAL findings disclosed: "
            f"{live_vulnerability_scan['critical_total']}",
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
            raise RuntimeError(evidence.detail or "bootstrap attestation is incompatible")
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
        if ok and item.tool in images.SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS:
            ok, detail = verify_bootstrap_publication_source(item)
            detail = f"BOOTSTRAP GATE — {detail}"
        if ok and item.tool == "wan2-2":
            ok, detail = verify_wan_publication_source(item)
            detail = f"WAN GATE — {detail}"
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
# Source credential inspection
#
# The source registry credential is the one thing about a publish that expires on its own,
# and the two credentials Nebius accepts for `docker login -u iam` are wildly different in
# that respect: an access token from `nebius iam get-access-token` lives 12 HOURS, while a
# static key issued for CONTAINER_REGISTRY lives 6 months by default. A manual-dispatch
# workflow holding a stored access token is therefore expired essentially always -- which is
# exactly how run #1 failed, with 23 identical UNAUTHORIZED lines after a two-minute sweep.
#
# An access token is a JWT, so its expiry is readable offline. Checking it before the sweep
# turns that into an immediate, unambiguous verdict, and -- more importantly -- names the
# remedy that does not expire again next week.
# --------------------------------------------------------------------------------------


def _decode_jwt_expiry(token: str) -> int | None:
    """The ``exp`` claim of a JWT-shaped bearer token, or ``None``.

    Best-effort and deliberately non-verifying: the goal is to read the expiry the issuer
    already put in the token, not to validate it. A static key is an opaque string rather
    than a JWT, so returning ``None`` is the normal answer for the credential we actually
    want in CI, and must never be reported as a problem.
    """
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)  # base64url in a JWT is unpadded
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    expiry = claims.get("exp") if isinstance(claims, dict) else None
    return expiry if isinstance(expiry, int) else None


def describe_credential(token: str, *, now: float | None = None) -> tuple[bool, str]:
    """Whether ``token`` is usable, and a one-line verdict that never echoes the secret.

    ``False`` is returned only when the credential is *provably* dead — a JWT whose ``exp``
    has passed. Anything unreadable is reported as usable, because this check exists to
    convert one specific recurring failure into a fast, precise message, not to become a
    second gate that can wrongly refuse a working credential.
    """
    now = time.time() if now is None else now
    token = token.strip()
    if not token:
        return False, "the credential is empty"

    expiry = _decode_jwt_expiry(token)
    if expiry is None:
        return True, (
            "the credential has no readable expiry, which is expected for a static key "
            "(`nebius iam static-key issue --service=CONTAINER_REGISTRY`)"
        )
    remaining = expiry - now
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expiry))
    if remaining <= 0:
        return False, (
            f"the credential EXPIRED at {stamp} ({_approx_duration(-remaining)} ago). A "
            "Nebius access token lives 12 hours, so a stored one is dead by the next "
            "dispatch; issue a long-lived static key instead:\n"
            "  nebius iam static-key issue --account-service-account-id=<sa-id> "
            "--service=CONTAINER_REGISTRY"
        )
    return True, (
        f"the credential is an access token valid until {stamp} "
        f"({_approx_duration(remaining)} left). It will not survive to the next dispatch — "
        "a static key issued for CONTAINER_REGISTRY lasts 6 months."
    )


def _approx_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
        if seconds >= size:
            return f"{seconds / size:.0f}{unit}"
    return f"{seconds:.0f}s"


# --------------------------------------------------------------------------------------
# Anonymous pullability
#
# Pushing to GHCR is NOT the same as publishing. A newly created container package is
# PRIVATE, and a package linked to a repository inherits that repository's access
# *permissions* but explicitly NOT its visibility -- so even a public repo yields private
# packages. Worse, GitHub exposes no REST API to change visibility for ORGANISATION-owned
# packages: it is a manual step in the package's settings UI, and it is one-way (a public
# package cannot be made private again).
#
# Without the check below, `publish_public` copies every image, exits 0, and reports
# success while nothing is actually publicly pullable -- a silent false success on the one
# action in this repo that cannot be undone. So the copy path in main() runs this
# verification inline and fails on it.
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
                    "digest mismatch — "
                    f"source {source_detail}; target {target_detail}"
                )
            else:
                print(f"  {item.target_ref}  current ({source_detail})")
                continue
        print(f"  {item.target_ref}  DRIFTED — {detail}")
        failures.append((item, detail))
    return failures


def ghcr_owner_and_package(target_ref: str) -> tuple[str, str] | None:
    """Split a GHCR reference into its owner and package name.

    GHCR nests the package under the repository: ``ghcr.io/<owner>/<repo>/<image>`` is
    owner ``<owner>`` and package ``<repo>/<image>``. Returns ``None`` for any other
    registry, which has its own naming and visibility model.
    """
    host, _, remainder = target_ref.partition("/")
    if host != "ghcr.io" or not remainder:
        return None
    path = remainder.rpartition(":")[0] or remainder
    owner, _, package = path.partition("/")
    if not owner or not package:
        return None
    return owner, package


def package_settings_url(target_ref: str, *, owner_type: str = "orgs") -> str | None:
    """Deep link to the GHCR package settings page that owns ``target_ref``.

    The visibility flip is the one step of a publish that cannot be automated -- GitHub
    has no REST endpoint for it on organisation-owned packages -- so the least we can do
    is not make someone hunt for each package in a list. The settings path
    percent-encodes the slash in the nested package name; a raw slash 404s.
    """
    parsed = ghcr_owner_and_package(target_ref)
    if parsed is None:
        return None
    owner, package = parsed
    return (
        f"https://github.com/{owner_type}/{owner}/packages/container/"
        f"{urllib.parse.quote(package, safe='')}/settings"
    )


def visibility_checklist(failures: list[tuple[PublishItem, str]]) -> str:
    """A markdown checklist of the packages still needing a manual visibility flip."""
    lines = []
    for item, _ in failures:
        parsed = ghcr_owner_and_package(item.target_ref)
        url = package_settings_url(item.target_ref)
        # Label with the package name as the settings page shows it, so the list reads
        # the same as the page it links to.
        lines.append(
            f"- [ ] [{parsed[1]}]({url})"
            if parsed and url
            else f"- [ ] {item.target_ref}"
        )
    return "\n".join(lines)


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


def _pin_wan_publication_sources(
    plan: list[PublishItem],
) -> tuple[list[PublishItem], list[tuple[PublishItem, str]]]:
    """Resolve Wan tags once and return a plan that can only copy those bytes."""

    pinned: list[PublishItem] = []
    failures: list[tuple[PublishItem, str]] = []
    for item in plan:
        if item.tool != "wan2-2":
            pinned.append(item)
            continue
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
        pinned.append(replace(item, source_ref=f"{_repository(item.source_ref)}@{detail}"))
    return pinned, failures


def _crane_copy(item: PublishItem) -> bool:
    """Copy ``item`` only when the target is absent or has a different digest.

    Returns ``True`` when a copy ran and ``False`` when the exact source digest was
    already present. A target denial is allowed to reach ``crane copy`` because a new
    private GHCR package can be unreadable through the pull path while the workflow's
    token is still authorised to create it; the copy remains the authoritative write
    check. Transient or unknown digest failures stop instead of risking an unnecessary
    rewrite.
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
        if failure_kind == "other":
            raise RuntimeError(
                f"could not determine target digest for {item.target_ref}: {target_detail}; "
                "refusing to copy because the target may already be current"
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
        "--source-registry",
        default=None,
        help="Source registry to copy from (defaults to the primary Nebius registry).",
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        help=(
            "Operate only on this eligible tool; repeat for multiple tools. The value "
            "must already be in the guarded public plan and cannot name an arbitrary image."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without copying."
    )
    parser.add_argument(
        "--verify-public",
        action="store_true",
        help=(
            "Do not copy. Check that every planned target is pullable with NO credentials, "
            "and exit non-zero if any is not. Pushing to GHCR leaves packages PRIVATE, and "
            "there is no API to change that for org-owned packages, so this is how a "
            "publish proves it actually published."
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
        "--describe-credential",
        action="store_true",
        help=(
            "Read a source-registry credential on stdin and report whether it is usable, "
            "without contacting any registry and without echoing the secret. Exits non-zero "
            "only when the credential is provably expired. Runs before the preflight so an "
            "expired 12-hour access token fails in a second with the reason, instead of as "
            "a wall of UNAUTHORIZED lines."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Operate on the images that exist, skipping any the source registry does not have "
            "yet (NAME_UNKNOWN / MANIFEST_UNKNOWN), and report exactly which were skipped. "
            "The plan comes from the packaging contract, which records what this repo BUILDS, "
            "so a tool that landed before its image was built otherwise blocks every ready "
            "image. A denial is never skipped — that is a credential or role fault."
        ),
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help=(
            "With --verify-public, also print a markdown checklist of package settings "
            "links for the targets that are not public yet (for a job summary)."
        ),
    )
    args = parser.parse_args(argv)

    # Before the plan: this mode inspects a credential and touches neither registry, so it
    # must not require a target registry it has no use for.
    if args.describe_credential:
        usable, verdict = describe_credential(sys.stdin.read())
        print(
            f"Source registry credential: {verdict}",
            file=sys.stdout if usable else sys.stderr,
        )
        return 0 if usable else 1

    if not (args.target or "").strip():
        parser.error("no target registry; pass --target or set NPA_PUBLIC_REGISTRY")

    try:
        plan = filter_publish_plan(
            build_publish_plan(
                target_registry=args.target, source_registry=args.source_registry
            ),
            args.tool,
        )
    except ValueError as exc:
        parser.error(str(exc))
    restricted = omniverse_restricted_image_names()
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
            "Excluded: none — every workbench image is publicly redistributable. The "
            "Isaac images fetch Isaac Sim / Isaac Lab at first run under the operator's "
            "own EULA acceptance rather than baking it."
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
            _explain_private_packages(failures, total=len(expected))
            if args.checklist:
                print("\n" + visibility_checklist(failures))
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
        _explain_private_packages(failures, total=len(publishable), after_copy=True)
        print("\n" + visibility_checklist(failures))
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
            "In CI, prefer a credential that does not expire between dispatches. An access\n"
            "token from `nebius iam get-access-token` lives 12 HOURS, so a stored one is dead\n"
            "by the next manual run; a static key issued for the registry lasts 6 months:\n"
            "  nebius iam static-key issue --account-service-account-id=<sa-id> "
            "--service=CONTAINER_REGISTRY\n"
            "Locally, re-mint and log in again:\n"
            "  nebius iam get-access-token | crane auth login <registry-host> -u iam "
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
                f"{len(missing)} image(s) are simply not in the source registry:"
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


def _explain_private_packages(
    failures: list[tuple[PublishItem, str]], *, total: int, after_copy: bool = False
) -> None:
    lead = (
        "The copy succeeded, but pushing to GHCR does not publish."
        if after_copy
        else "Pushing to GHCR does not publish."
    )
    print(
        f"\n{len(failures)} of {total} image(s) are NOT publicly pullable.\n"
        f"{lead} A new container package is private, and a\n"
        "package linked to a repository inherits the repository's access permissions\n"
        "but NOT its visibility. GitHub offers no REST API to change visibility for\n"
        "organisation-owned packages, so this is a MANUAL step per package:\n"
        "  Package settings -> Danger Zone -> Change visibility -> Public\n"
        "It is one-time — visibility persists across later pushes to the same package —\n"
        "and irreversible: a public package cannot be made private again.\n"
        "Direct links below (user-owned packages live under /users/ instead of /orgs/).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
