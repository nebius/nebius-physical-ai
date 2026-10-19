"""Bind MK8s installations to immutable, qualified recipe releases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


_RELEASE_KEY = "mk8s_recipe_release"
_CONFIGURATION_CONTRACT = {
    "driver": "npa.cluster.gpu_driver:v1",
    "renderer": "npa.cluster_backends.mk8s_render:v1",
    "required_recipe_paths": [
        "k8s-training/main.tf",
        "k8s-training/variables.tf",
        "modules",
    ],
}


class MK8sReleaseError(ValueError):
    """Raise when MK8s source provenance cannot be used safely."""


class LegacyMK8sReleaseRecoveryError(MK8sReleaseError):
    """Raise when an existing install predates persisted recipe provenance."""


@dataclass(frozen=True)
class MK8sRecipeRelease:
    """One immutable MK8s recipe release accepted by this NPA build."""

    release_id: str
    upstream_tag: str
    upstream_commit: str
    bundle_sha256: str
    provider_lock_sha256: str
    configuration_contract_sha256: str

    def persistence(self) -> dict[str, str]:
        """Return the exact installation-side record for this release.

        Args:
            None.

        Returns:
            A JSON-safe immutable release record.

        Raises:
            None.
        """

        return asdict(self)


_MK8S_RELEASES = {
    "mk8s-v2026-10-18": MK8sRecipeRelease(
        release_id="mk8s-v2026-10-18",
        upstream_tag="main-v2026-10-12",
        upstream_commit="fba7148cecd8bd38bccf60ba194ab8f5fdb02194",
        bundle_sha256="d58ebfebc31bcfc6f8491e0f595015de37b1f17bdb68a580dbc6b059d0db07c0",
        provider_lock_sha256="f4d96577faf2f583803b1b1495681f24574730a0a00201db1440633238a40505",
        configuration_contract_sha256="df66bf041ca9a8b7f2cc48497cbfeb7c0eda6a7c4b863e5f7349088abf7f93b7",
    )
}
DEFAULT_MK8S_RELEASE_ID = "mk8s-v2026-10-18"


def default_mk8s_release() -> MK8sRecipeRelease:
    """Return the release selected for a newly created default installation.

    Args:
        None.

    Returns:
        The current immutable MK8s recipe release.

    Raises:
        MK8sReleaseError: If the catalog has no configured default.
    """

    return resolve_mk8s_release(DEFAULT_MK8S_RELEASE_ID)


def resolve_mk8s_release(release_id: str) -> MK8sRecipeRelease:
    """Resolve one visible release ID without accepting an ambient source ref.

    Args:
        release_id: The public, date-versioned MK8s release identifier.

    Returns:
        The catalog entry for the requested release.

    Raises:
        MK8sReleaseError: If the identifier is unknown or empty.
    """

    release = _MK8S_RELEASES.get(release_id)
    if release is None:
        known = ", ".join(sorted(_MK8S_RELEASES))
        raise MK8sReleaseError(
            f"unknown MK8s release {release_id!r}; accepted releases: {known}"
        )
    return release


def recipe_bundle_sha256(recipe_root: Path) -> str:
    """Return a deterministic digest for the complete materialized source bundle.

    Args:
        recipe_root: Root containing the upstream k8s-training and modules trees.

    Returns:
        Lowercase SHA-256 digest of paths and bytes below the source root.

    Raises:
        MK8sReleaseError: If the source root is missing, incomplete, or contains a link.
    """

    root = recipe_root.expanduser().resolve()
    if (
        not (root / "k8s-training" / "variables.tf").is_file()
        or not (root / "modules").is_dir()
    ):
        raise MK8sReleaseError("MK8s recipe root is incomplete")
    digest = hashlib.sha256()
    for entry in sorted(root.rglob("*")):
        if ".git" in entry.parts:
            continue
        if entry.is_symlink():
            raise MK8sReleaseError("MK8s recipe bundle must not contain symbolic links")
        if entry.is_file():
            digest.update(entry.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(entry.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def configuration_contract_sha256() -> str:
    """Return the digest of the NPA component/configuration compatibility contract.

    Args:
        None.

    Returns:
        SHA-256 digest of the canonicalized immutable compatibility contract.

    Raises:
        None.
    """

    encoded = json.dumps(
        _CONFIGURATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provider_lock_path(recipe_root: Path) -> Path:
    """Find the reviewed Terraform provider lock that accompanies a recipe root.

    Args:
        recipe_root: Materialized solutions-library source root.

    Returns:
        The nearest enclosing tracked Terraform provider lock.

    Raises:
        MK8sReleaseError: If no reviewed provider lock accompanies the source.
    """

    root = recipe_root.expanduser().resolve()
    for directory in (root, *root.parents):
        candidate = directory / ".terraform.lock.hcl"
        if candidate.is_file():
            return candidate
    raise MK8sReleaseError(
        "MK8s recipe has no reviewed Terraform provider lock; create a reviewed "
        "replacement release instead of using an unbound source"
    )


def validate_release_source(release: MK8sRecipeRelease, recipe_root: Path) -> None:
    """Reject source bytes that do not exactly match a catalog release.

    Args:
        release: Catalog release selected for an installation.
        recipe_root: Source tree about to be copied into that installation.

    Returns:
        None.

    Raises:
        MK8sReleaseError: If the source, provider lock, or component contract differs.
    """

    actual = recipe_bundle_sha256(recipe_root)
    if actual != release.bundle_sha256:
        raise MK8sReleaseError(
            f"MK8s recipe bundle does not match {release.release_id}; "
            "create a reviewed replacement release instead of changing an install"
        )
    lock = provider_lock_path(recipe_root)
    lock_digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if lock_digest != release.provider_lock_sha256:
        raise MK8sReleaseError(
            f"MK8s provider lock does not match {release.release_id}; create a "
            "reviewed replacement release instead of changing an install"
        )
    if configuration_contract_sha256() != release.configuration_contract_sha256:
        raise MK8sReleaseError(
            f"MK8s component/configuration contract does not match {release.release_id}; "
            "create a reviewed replacement release instead of changing an install"
        )


def select_installation_release(
    install_dir: Path, requested_release_id: str | None = None
) -> MK8sRecipeRelease:
    """Select a release once, retaining an existing install's exact choice.

    Args:
        install_dir: Per-cluster private installation directory.
        requested_release_id: Explicit requested release, or None for the current default.

    Returns:
        The release that may be used for this lifecycle action.

    Raises:
        LegacyMK8sReleaseRecoveryError: If old state lacks recoverable provenance.
        MK8sReleaseError: If persisted bytes conflict with the catalog or request.
    """

    requested = resolve_mk8s_release(requested_release_id or DEFAULT_MK8S_RELEASE_ID)
    sidecar = install_dir / ".npa-fleet-env.json"
    if not sidecar.exists():
        return requested
    try:
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MK8sReleaseError(
            "persisted MK8s installation identity is unreadable"
        ) from exc
    if not isinstance(saved, dict):
        raise MK8sReleaseError("persisted MK8s installation identity is not an object")
    persisted = saved.get(_RELEASE_KEY)
    if persisted is None:
        raise LegacyMK8sReleaseRecoveryError(
            "existing MK8s installation predates recipe-release provenance; "
            "destroy may use its retained materialized recipe, but deploy/reconcile "
            "requires an explicit reviewed recovery or replacement"
        )
    if not isinstance(persisted, dict):
        raise MK8sReleaseError("persisted MK8s recipe-release provenance is malformed")
    release_id = persisted.get("release_id")
    if not isinstance(release_id, str):
        raise MK8sReleaseError("persisted MK8s recipe-release ID is missing")
    selected = resolve_mk8s_release(release_id)
    if persisted != selected.persistence():
        raise MK8sReleaseError(
            "persisted MK8s recipe-release provenance conflicts with the catalog"
        )
    if selected.release_id != requested.release_id:
        raise MK8sReleaseError(
            "existing MK8s installation is pinned to a different release; "
            "use a reviewed migration or replacement rather than an in-place update"
        )
    return selected


def release_from_sidecar(sidecar: dict[str, Any]) -> dict[str, str] | None:
    """Extract validated release provenance for status output.

    Args:
        sidecar: Parsed private installation identity record.

    Returns:
        A safe immutable release record, or None for a legacy installation.

    Raises:
        MK8sReleaseError: If a present release record is malformed or mismatched.
    """

    persisted = sidecar.get(_RELEASE_KEY)
    if persisted is None:
        return None
    if not isinstance(persisted, dict):
        raise MK8sReleaseError("persisted MK8s recipe-release provenance is malformed")
    release_id = persisted.get("release_id")
    if not isinstance(release_id, str):
        raise MK8sReleaseError("persisted MK8s recipe-release ID is missing")
    release = resolve_mk8s_release(release_id)
    if persisted != release.persistence():
        raise MK8sReleaseError(
            "persisted MK8s recipe-release provenance conflicts with the catalog"
        )
    return release.persistence()
