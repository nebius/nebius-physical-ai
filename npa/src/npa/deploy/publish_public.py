"""Mirror the OSS-redistributable workbench images to a public registry.

Nebius Container Registry does not support anonymous/public pulls, so "public
exposure" of the workbench means mirroring the publicly-redistributable image
subset to a public-capable registry (e.g. GHCR ``ghcr.io/<org>/<repo>``).

This tool is license-guarded: it only ever copies tools reported by
``images.publicly_publishable_tools()`` and hard-refuses anything in
``images.OMNIVERSE_RESTRICTED_TOOLS`` as defence in depth around that selector.

That set is currently empty. It used to hold ``isaac-lab``, ``sonic`` and ``groot``
(plus the derived ``sonic-mujoco``), which baked NVIDIA Omniverse Kit; those images
were re-architected to fetch Isaac Sim / Isaac Lab at first run under the operator's
own EULA acceptance, so every workbench image is now publishable. The refusal is
kept, and tested against a synthetic restricted tool, for the next runtime we cannot
ship.

Example (dry run first, then execute):

    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai --dry-run
    python -m npa.deploy.publish_public --target ghcr.io/nebius/nebius-physical-ai

The copy path preflights the source registry first and verifies the result after, so a
stale credential fails before anything is written and a copy cannot report success while
nothing is publicly pullable. Making the packages public is the one step that cannot be
automated (see ``package_settings_url``); the verification prints a click-through list.

``--describe-credential`` reads a source-registry credential on stdin and reports its
expiry offline, because the credential is what actually breaks: a Nebius access token
lives 12 hours, so anything stored in CI must be a static key issued for
CONTAINER_REGISTRY instead (see ``describe_credential``).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

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
        # Read the restricted set through the module, never a from-import: a
        # defence-in-depth check that holds a stale copy of the thing it is defending is
        # worse than no check at all. (`from ... import OMNIVERSE_RESTRICTED_TOOLS` binds
        # the value at import time, so this guard and publicly_publishable_tools() could
        # disagree - which is exactly what a test caught once the set stopped being empty.)
        if not is_publicly_redistributable(tool) or tool in images.OMNIVERSE_RESTRICTED_TOOLS:
            raise ValueError(
                f"refusing to publish restricted (Omniverse Kit) tool {tool!r} to a public registry"
            )
        source_ref = container_image_for_tool(tool, registry=source_registry)
        image = source_ref.rsplit("/", 1)[-1]  # npa-<tool>:<tag>
        plan.append(PublishItem(tool=tool, source_ref=source_ref, target_ref=f"{target}/{image}"))
    return plan


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


def _crane_manifest_readable(ref: str, *, timeout: float = _PREFLIGHT_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Whether ``ref`` can be read with the ambient registry credentials."""
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError("crane not found on PATH; install go-containerregistry crane")
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
        ok, detail = _crane_manifest_readable(item.source_ref)
        print(f"  {item.source_ref}  {'ok' if ok else f'UNREADABLE — {detail}'}")
        if not ok:
            failures.append((item, detail))
    return failures


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


def anonymous_pull_ok(ref: str, *, timeout: float = _ANON_TIMEOUT_SECONDS) -> tuple[bool, str]:
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
        lines.append(f"- [ ] [{parsed[1]}]({url})" if parsed and url else f"- [ ] {item.target_ref}")
    return "\n".join(lines)


def _crane_copy(item: PublishItem) -> None:
    crane = shutil.which("crane")
    if not crane:
        raise RuntimeError("crane not found on PATH; install go-containerregistry crane")
    subprocess.run([crane, "copy", item.source_ref, item.target_ref], check=True)


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
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying.")
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
        print(f"Source registry credential: {verdict}", file=sys.stdout if usable else sys.stderr)
        return 0 if usable else 1

    if not (args.target or "").strip():
        parser.error("no target registry; pass --target or set NPA_PUBLIC_REGISTRY")

    plan = build_publish_plan(target_registry=args.target, source_registry=args.source_registry)
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
        print("\nVerifying anonymous (unauthenticated) pullability:")
        failures = verify_public(plan)
        if failures:
            _explain_private_packages(failures, total=len(plan))
            if args.checklist:
                print("\n" + visibility_checklist(failures))
            return 1
        print(f"\nAll {len(plan)} image(s) are publicly pullable.")
        return 0

    if args.preflight:
        print("\nPreflighting source images (authenticated read, nothing written):")
        return 1 if _preflight_or_explain(plan) else 0

    if args.dry_run:
        print("(dry run — nothing copied)")
        return 0

    print("\nPreflighting source images (authenticated read, nothing written):")
    if _preflight_or_explain(plan):
        return 1
    for item in plan:
        _crane_copy(item)
    print(f"\nCopied {len(plan)} image(s).")

    # Copying is not publishing, so do not stop here and report success. Verifying
    # inline means the operator learns the real state -- and gets the click-through
    # list for the one step that cannot be automated -- from the command that did
    # the copy, rather than from a separate invocation they have to know to run.
    print("\nVerifying anonymous (unauthenticated) pullability:")
    failures = verify_public(plan)
    if failures:
        _explain_private_packages(failures, total=len(plan), after_copy=True)
        print("\n" + visibility_checklist(failures))
        return 1
    print(f"\nAll {len(plan)} image(s) are publicly pullable.")
    return 0


def _preflight_or_explain(plan: list[PublishItem]) -> bool:
    """Run the source preflight; explain and return True if it failed."""
    failures = preflight_sources(plan)
    if not failures:
        return False
    lines = [
        f"\n{len(failures)} of {len(plan)} source image(s) could not be read; nothing was "
        "copied."
    ]
    # Whether EVERY read failed is the diagnostic, not just the registry's error code: one
    # UNAUTHORIZED could be a per-repository grant, but all of them means the credential
    # never resolved to an identity at all ("failed to get profile"), so there is no point
    # looking at roles or at the tags.
    if len(failures) == len(plan) and all("UNAUTHORIZED" in detail for _, detail in failures):
        lines.append(
            "Every read failed with UNAUTHORIZED, so this is the credential rather than any\n"
            "single tag or grant — the token did not resolve to an identity.\n"
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
        lines.append(
            "UNAUTHORIZED on SOME images means the credential works but its identity lacks\n"
            "viewer on those repositories — fix the role, not the token.\n"
            "MANIFEST_UNKNOWN means the pinned tag is not in the source registry — build and\n"
            "push that image, or correct its pin, before publishing."
        )
    print("\n".join(lines), file=sys.stderr)
    return True


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
