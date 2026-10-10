"""Narrow guards for HY-World's operator-private bootstrap delivery.

This is deliberately not a generic publisher.  It keeps the candidate's
private delivery script from accepting a Docker Hub shorthand/public namespace
or issuing a success receipt when the pushed OCI manifest is not the locally
scanned image configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
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


def _safe_member_name(name: object, field: str) -> str:
    """Validate one non-traversing Docker-save member name."""

    if not isinstance(name, str):
        raise PrivateDeliveryError(f"docker-save {field} path is unsafe")
    value = name
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise PrivateDeliveryError(f"docker-save {field} path is unsafe")
    return value


def _single_file_member(archive: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    """Return exactly one regular archive member, rejecting ambiguity."""

    matches = [member for member in archive.getmembers() if member.name == name]
    if len(matches) != 1 or not matches[0].isfile():
        raise PrivateDeliveryError(
            f"docker-save archive must contain one regular {name}"
        )
    return matches[0]


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    """Read one verified regular member from a Docker-save archive."""

    member = _single_file_member(archive, name)
    handle = archive.extractfile(member)
    if handle is None:  # pragma: no cover - regular members are readable
        raise PrivateDeliveryError(f"could not read docker-save member {name}")
    return handle.read()


def _docker_save_config_bytes(archive: tarfile.TarFile) -> bytes:
    """Read the single image configuration named by Docker-save metadata."""

    try:
        manifest = json.loads(_member_bytes(archive, "manifest.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivateDeliveryError(
            "docker-save manifest.json is not valid JSON"
        ) from exc
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise PrivateDeliveryError(
            "docker-save archive must describe exactly one image"
        )
    item = manifest[0]
    if not isinstance(item, dict):
        raise PrivateDeliveryError("docker-save manifest entry is invalid")
    config_name = _safe_member_name(item.get("Config"), "config")
    config_bytes = _member_bytes(archive, config_name)
    try:
        if not isinstance(json.loads(config_bytes), dict):
            raise PrivateDeliveryError("docker-save image config must be a JSON object")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivateDeliveryError(
            "docker-save image config is not valid JSON"
        ) from exc
    return config_bytes


def docker_save_config_digest(archive_path: Path) -> str:
    """Hash the OCI config bytes of exactly one Docker-save image archive.

    Docker's ``image inspect .Id`` is storage-driver dependent: containerd can
    report an image/manifest identity rather than the OCI config descriptor.
    The archive is both the local scanner input and the push input, so its
    declared config member is the only durable local value to compare with the
    registry manifest's config digest.
    """

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            config_bytes = _docker_save_config_bytes(archive)
    except (OSError, tarfile.TarError) as exc:
        raise PrivateDeliveryError("could not read docker-save archive") from exc
    return "sha256:" + hashlib.sha256(config_bytes).hexdigest()


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
    config_digest = subcommands.add_parser("docker-save-config-digest")
    config_digest.add_argument("--docker-save", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "private-registry":
            print(private_registry(args.registry))
            return 0
        if args.command == "docker-save-config-digest":
            print(docker_save_config_digest(args.docker_save))
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
