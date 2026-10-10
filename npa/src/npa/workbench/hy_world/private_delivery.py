"""Narrow guards for HY-World's operator-private bootstrap delivery.

This is deliberately not a generic publisher.  It keeps the candidate's
private delivery script from accepting a Docker Hub shorthand/public namespace
or issuing a success receipt when the pushed OCI manifest is not the locally
scanned image configuration.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from npa.deploy.images import is_public_registry
from npa.orchestration.skypilot.registry_preflight import (
    RegistryPreflightError,
    parse_image_reference,
)


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_TAG = "dev-" + "0" * 40


class PrivateDeliveryError(ValueError):
    """Signal that a private delivery destination or identity is unsafe.

    Args:
        None.

    Returns:
        None.

    Raises:
        None. Raised by delivery validation helpers.
    """


def private_registry(registry: str) -> str:
    """Return one private registry namespace, rejecting public/shorthand input.

    Args:
        registry: Candidate ``HOST/NAMESPACE`` delivery destination.

    Returns:
        The validated private registry namespace.

    Raises:
        PrivateDeliveryError: If the destination is malformed, shorthand, or public.
    """

    value = str(registry or "").strip().rstrip("/")
    if (
        not value
        or value != registry
        or any(token in value for token in ("://", "@", "?", "#"))
    ):
        raise PrivateDeliveryError(
            "registry must be a fully-qualified HOST/NAMESPACE without a scheme"
        )
    try:
        reference = parse_image_reference(f"{value}/npa-hy-world:{_SOURCE_TAG}")
    except RegistryPreflightError as exc:
        raise PrivateDeliveryError(
            "registry must start with a fully-qualified host and include a namespace"
        ) from exc
    if reference.registry != value.split("/", 1)[
        0
    ] or not reference.repository.startswith(value.partition("/")[2] + "/"):
        raise PrivateDeliveryError("registry must be a fully-qualified HOST/NAMESPACE")
    if is_public_registry(value):
        raise PrivateDeliveryError(
            "operator-private delivery refuses an anonymous/public registry namespace"
        )
    return value


def _digest(value: object, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(digest):
        raise PrivateDeliveryError(f"{field} must be a sha256 digest")
    return digest


def bind_remote_manifest(
    *, local_config_digest: str, remote_digest: str, manifest: Any
) -> dict[str, object]:
    """Bind an immutable remote manifest to the locally scanned configuration.

    OCI config identity includes the ordered uncompressed layer diff IDs. The
    remote manifest additionally names the compressed layer blobs. A later
    remote scan consumes this exact digest, so both the local pre-push scan and
    remote post-push scan are attributable without trusting a mutable tag.

    Args:
        local_config_digest: OCI config digest read from the locally scanned image.
        remote_digest: Immutable manifest digest returned by the private registry.
        manifest: Parsed remote OCI/Docker schema-v2 manifest.

    Returns:
        The identity-binding record retained with private delivery evidence.

    Raises:
        PrivateDeliveryError: If digests or required manifest descriptors differ.
    """

    local = _digest(local_config_digest, "local config digest")
    remote = _digest(remote_digest, "remote manifest digest")
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2:
        raise PrivateDeliveryError(
            "remote image manifest must be OCI/Docker schemaVersion 2"
        )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise PrivateDeliveryError("remote image manifest has no config descriptor")
    remote_config = _digest(config.get("digest"), "remote config digest")
    if remote_config != local:
        raise PrivateDeliveryError(
            "remote config digest does not match the locally scanned image"
        )
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise PrivateDeliveryError("remote image manifest has no layer descriptors")
    layer_digests = []
    for item in layers:
        if not isinstance(item, dict):
            raise PrivateDeliveryError(
                "remote image manifest has an invalid layer descriptor"
            )
        layer_digests.append(_digest(item.get("digest"), "remote layer digest"))
    return {
        "schema": "npa.hy_world.private_delivery_identity.v1",
        "status": "matched",
        "local_config_digest": local,
        "remote_manifest_digest": remote,
        "remote_config_digest": remote_config,
        "remote_layer_digests": layer_digests,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    check_registry = subcommands.add_parser("private-registry")
    check_registry.add_argument("registry")
    binding = subcommands.add_parser("bind-manifest")
    binding.add_argument("--local-config-digest", required=True)
    binding.add_argument("--remote-digest", required=True)
    binding.add_argument("--manifest", type=Path, required=True)
    binding.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "private-registry":
            print(private_registry(args.registry))
            return 0
        report = bind_remote_manifest(
            local_config_digest=args.local_config_digest,
            remote_digest=args.remote_digest,
            manifest=json.loads(args.manifest.read_text(encoding="utf-8")),
        )
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    except (OSError, json.JSONDecodeError, PrivateDeliveryError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse exits above


if __name__ == "__main__":  # pragma: no cover - shell helper entrypoint
    raise SystemExit(_main())
