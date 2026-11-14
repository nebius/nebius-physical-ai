"""Materialize and verify the rigid object selected by an interchange profile."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from npa.workflows.sim2real.interchange_contract import (
    InterchangeContractError,
    RigidObjectProfile,
    rigid_object_profile_from_dict,
)
from npa.workflows.sim2real.workflow_io import storage


@dataclass(frozen=True)
class MaterializedObjectAsset:
    """Verified local object bytes plus their durable profile identity."""

    local_path: Path
    profile: RigidObjectProfile

    def receipt(self) -> dict[str, str]:
        """Return durable source fields safe to publish in collection evidence.

        Args:
            None.

        Returns:
            Exact profile and source identities without a pod-local path.

        Raises:
            None.
        """

        return {
            "profile_identity": self.profile.identity,
            "source_uri": self.profile.source_uri,
            "source_sha256": self.profile.source_sha256,
            "bundle_sha256": self.profile.bundle_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_prepared_object(
    preparation: Mapping[str, Any], destination: Path
) -> MaterializedObjectAsset:
    """Download and verify the prepared rigid object's exact source bytes.

    Args:
        preparation: Validated policy-interchange preparation receipt.
        destination: Task-owned directory for the local runtime copy.

    Returns:
        The verified local path and reconstructed rigid-object profile.

    Raises:
        InterchangeContractError: The preparation or downloaded bytes disagree.
    """

    raw = preparation.get("object")
    if not isinstance(raw, Mapping):
        raise InterchangeContractError("preparation lacks a rigid object profile")
    profile = rigid_object_profile_from_dict(raw)
    parsed = urlparse(profile.source_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise InterchangeContractError(
            "prepared object source_uri must be an exact operator-storage s3 object"
        )
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".usd", ".usda", ".usdc"}:
        raise InterchangeContractError("prepared object source must be a USD file")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"object{suffix}"
    storage().download_file(profile.source_uri, str(target))
    if _sha256(target) != profile.source_sha256:
        raise InterchangeContractError(
            "downloaded object bytes differ from the prepared source SHA-256"
        )
    return MaterializedObjectAsset(local_path=target, profile=profile)


def assert_collection_object_binding(
    asset: Mapping[str, Any], profile: RigidObjectProfile
) -> None:
    """Require a collection receipt to name the prepared object source exactly.

    Args:
        asset: The collection's durable object-asset receipt.
        profile: The profile whose bytes were materialized for collection.

    Returns:
        None.

    Raises:
        InterchangeContractError: Any immutable source identity differs.
    """

    expected = {
        "profile_identity": profile.identity,
        "source_uri": profile.source_uri,
        "source_sha256": profile.source_sha256,
        "bundle_sha256": profile.bundle_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": asset.get(key)}
        for key, value in expected.items()
        if asset.get(key) != value
    }
    if mismatches:
        raise InterchangeContractError(
            f"collection object differs from prepared profile: {mismatches}"
        )
