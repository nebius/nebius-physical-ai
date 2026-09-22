#!/usr/bin/env python3
"""Prove a BUILT image carries no restricted NVIDIA or LeIsaac runtime payload.

This is the check the whole redistribution reclassification rests on. Reading a
Dockerfile is not enough: the claim is about bytes in layers, so this inspects the built
image's filesystem and its layer history.

    npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py \
        ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab:dev-<full-git-sha>

    # or from a local docker save tarball, with no registry access
    docker save npa-isaac-lab:rc1 -o /tmp/img.tar
    npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py --tarball /tmp/img.tar

Saved Docker and OCI archives must contain their complete manifest, config and
layer graph. A successful export command alone does not establish completeness.
The reader keeps metadata in memory and streams layer contents without extraction.

Why it keys on payload signatures rather than the string "isaac"
---------------------------------------------------------------
The re-architected images deliberately keep a ``/isaac-sim/python.sh`` shim, because ~30
call sites in this repo already invoke Isaac through that path and pods override
ENTRYPOINT, so the shim is the only reliable bootstrap trigger. A naive
``tar -tf | grep isaac`` therefore reports a hit on a 40-line shell script of ours, and
"grep found nothing" is not available as a proof strategy.

Instead this looks for the things only a real Kit install produces - ``libcarb``, Kit's
``kernel/``, ``omni.*`` extension directories, ``extscache``, ``.kit`` app files,
``site-packages/isaacsim`` - and pairs that with a short, explicit ALLOWLIST of the paths
we do ship. Anything matching a signature and not on the allowlist fails the scan, so an
unexpected path fails closed rather than being waved through.

The classifier is unit-tested offline against synthetic listings in
npa/tests/docker/test_image_payload_scan.py, so its logic is covered in CI without a
registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# Path signatures that only a real Omniverse Kit / Isaac Sim install produces.
PAYLOAD_SIGNATURES: tuple[tuple[str, str], ...] = (
    (
        r"(?i)(?:^|/)site-packages/imageio_ffmpeg/binaries/ffmpeg[^/]*$",
        "an imageio-ffmpeg wheel-bundled static FFmpeg executable",
    ),
    (r"(?i)site-packages/isaacsim/", "the isaacsim wheel's installed package tree"),
    (r"(?i)site-packages/isaaclab/", "the isaaclab wheel's installed package tree"),
    (
        r"(?i)(?:^|/)site-packages/ovrtx(?:/|-.*\.(?:dist-info|egg-info)/)",
        "the proprietary OVRTX installed package tree",
    ),
    (r"(?i)(?:^|/)\.ovrtx_venv/", "a baked isolated OVRTX runtime"),
    (r"(?i)(?:^|/)libovrtx[^/]*\.so", "an OVRTX native runtime library"),
    (
        r"(?i)(^|/)isaac-sim/(kit|exts|extscache|extsPhysics|apps)/",
        "an Isaac Sim install tree",
    ),
    (r"(?i)(^|/)kit/kernel/", "Omniverse Kit's kernel"),
    (r"(?i)libcarb", "carb, Omniverse Kit's core runtime library"),
    (r"(?i)libomni[a-z0-9_.]*\.so", "an Omniverse Kit shared library"),
    (r"(?i)(^|/)omni\.[a-z0-9_.]+(-[^/]*)?/", "an Omniverse Kit extension directory"),
    (r"(?i)extscache", "Omniverse Kit's extension cache"),
    (r"(?i)\.kit$", "a Kit app configuration file"),
    (r"(?i)omniverse", "an Omniverse-branded path"),
    (r"(?i)isaac.?sim.?assets", "Isaac Sim's bundled assets"),
    (
        r"(?i)(^|/)leisaac-cache/client/[^/]+/",
        "a staged versioned NVIDIA WebRTC browser client",
    ),
    (
        r"(?i)(^|/)omniverse-webrtc-streaming-library-[^/]+\.(tgz|tar\.gz)$",
        "a staged NVIDIA WebRTC browser-client archive",
    ),
    (
        r"(?i)(^|/)leisaac-cache/assets/runtime/.*\.(usd|usda|usdc)$",
        "a staged LeIsaac runtime task asset",
    ),
    (
        r"(?i)(^|/)leisaac/assets/(robots|scenes)/.*\.(usd|usda|usdc)$",
        "a LeIsaac source-tree task asset",
    ),
)

# Gated model weights are a separate licence axis from Omniverse Kit, and the workbench
# rule is the same for both: never baked, always fetched at run time by the operator with
# their own token. gear_sonic's weights sit behind git LFS, so a plain checkout leaves
# ~130-byte pointer stubs; a pointer is a reference the operator resolves, not a weight.
# This scanner only sees a tar listing (names, not contents), so it reports weight-shaped
# paths for a human to eyeball rather than failing on them - the authoritative
# content-based check runs inside the image build, where the bytes are available.
WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
)

# Paths we DO ship that a loose name filter would flag. Deliberately short and exact: an
# unlisted path that matches a signature fails the scan.
ALLOWED_EXACT: frozenset[str] = frozenset(
    {
        "isaac-sim/python.sh",
        "opt/npa/bin/isaac-python",
        "opt/npa/bin/isaac-bootstrap",
    }
)
ALLOWED_PREFIXES: tuple[str, ...] = (
    "opt/npa/docker/workbench/common/",
    "opt/npa/docker/workbench/isaac-lab/",
    "opt/npa/docker/workbench/sonic/",
    "opt/npa/docker/workbench/groot/",
)
# Directory entries that are legitimately present but empty (mount points, workdirs).
ALLOWED_DIRS: frozenset[str] = frozenset(
    {
        "isaac-sim",
        "opt/isaac-cache",
        "opt/isaac-cache/v",
        "workspace/isaaclab",
        "opt/isaac-lab",
    }
)

# Layer commands that would mean Isaac was installed during the build. Kept in step with
# packaging-contract.yaml's omniverse_bake_patterns, but applied to the image's own
# recorded history rather than to a Dockerfile, so it also catches an image built from a
# Dockerfile nobody reviewed.
HISTORY_BAKE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?i)pip[^\n]*install[^\n]*\bisaacsim\b", "a build layer pip-installed isaacsim"),
    (r"(?i)pip[^\n]*install[^\n]*\bisaaclab\b", "a build layer pip-installed isaaclab"),
    (
        r"(?i)(?:pip[^\n]*install[^\n]*(?:\bovrtx\b|pylock\.ovrtx-runtime)|"
        r"render_ovrtx[^\n]*--provision-only)",
        "a build layer installed or provisioned OVRTX",
    ),
    (
        r"(?i)nvcr\.io/nvidia/(isaac-lab|isaac-sim|omniverse)",
        "a layer references an NVIDIA vendor image",
    ),
    (
        r"(?i)\b(isaac-bootstrap|isaac_bootstrap\.sh)\s+(ensure|warm|verify)\b",
        "a build layer ran the runtime bootstrap, materialising Isaac into the image",
    ),
    (
        r"(?i)(OMNI_KIT_ACCEPT_EULA|ISAACSIM_ACCEPT_EULA|PRIVACY_CONSENT)=",
        "a layer bakes EULA acceptance",
    ),
)


def _normalize(member: str) -> str:
    """Normalise a tar member path for matching (no leading ./ or /)."""
    return member.lstrip("./").lstrip("/")


def is_allowed(path: str) -> bool:
    normalized = _normalize(path)
    if normalized in ALLOWED_EXACT or normalized.rstrip("/") in ALLOWED_DIRS:
        return True
    return any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def classify_path(path: str) -> str | None:
    """Return why ``path`` looks like Kit payload, or ``None`` if it is fine."""
    if is_allowed(path):
        return None
    normalized = _normalize(path)
    for pattern, why in PAYLOAD_SIGNATURES:
        if re.search(pattern, normalized):
            return why
    return None


def _history_instructions(command: str) -> str:
    """Strip comment lines from a recorded layer command.

    buildkit records the *whole* RUN, heredocs included, so a Python comment inside an
    inlined script ends up in the image history. That bit for real: sonic's build-time
    check contains the line

        # happens on GPU (isaac-bootstrap verify / the golden eval).

    which made the scanner report "a build layer ran the runtime bootstrap" against an
    image that had done no such thing. A false positive here is not harmless - it would
    block a legitimate reclassification, and the obvious "fix" is to loosen the pattern.
    Same prose-versus-instruction distinction the packaging guard makes on Dockerfiles.
    """
    return "\n".join(
        line for line in command.splitlines() if not line.lstrip().startswith("#")
    )


def classify_history(command: str) -> str | None:
    """Return why a layer command looks like a build-time Isaac install, or ``None``."""
    instructions = _history_instructions(command)
    for pattern, why in HISTORY_BAKE_PATTERNS:
        if re.search(pattern, instructions):
            return why
    return None


@dataclass
class ScanReport:
    image: str
    source: str
    digest: str | None = None
    archive_binding: dict[str, object] | None = None
    entries_scanned: int = 0
    allowlisted_hits: list[str] = field(default_factory=list)
    payload_hits: list[dict[str, str]] = field(default_factory=list)
    history_hits: list[dict[str, str]] = field(default_factory=list)
    weight_shaped_paths: list[str] = field(default_factory=list)
    #: True when only the layer history was inspected. Recorded in the JSON report so a
    #: consumer can never mistake a fast gate result for a full-filesystem proof.
    history_only: bool = False

    @property
    def clean(self) -> bool:
        return not self.payload_hits and not self.history_hits

    def to_dict(self) -> dict:
        return {
            "format": "npa_restricted_payload_scan_v2",
            "image": self.image,
            "source": self.source,
            "digest": self.digest,
            # A local archive can be cited as evidence only when a successful
            # complete-graph verifier bound these exact archive, manifest and config
            # bytes. Registry scans retain their existing digest behavior.
            "archive_binding": self.archive_binding,
            # Recorded so a consumer can never mistake a fast pre-publish gate result for
            # a full-filesystem proof: a history-only "clean" says the build ran no Isaac
            # install, not that the image ships no Isaac bytes.
            "history_only": self.history_only,
            # A report is serialized only after every requested registry/tar stream and
            # history query has completed successfully. Consumers can therefore require
            # this marker instead of treating a partial listing as authoritative.
            "scan_complete": True,
            "entries_scanned": self.entries_scanned,
            "verdict": "clean" if self.clean else "restricted-payload-detected",
            "payload_hits": self.payload_hits,
            "history_hits": self.history_hits,
            "allowlisted_paths_present": sorted(self.allowlisted_hits),
            "weight_shaped_paths": sorted(self.weight_shaped_paths),
        }


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"{tool} not found on PATH")
    return path


def _iter_crane_export(image: str, *, max_attempts: int = 4):
    """Stream the flattened filesystem of a remote image, member by member."""
    crane = _require("crane")
    command = [crane, "export", image, "-"]
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        process = subprocess.Popen(  # noqa: S603 - fixed argv
            command, stdout=subprocess.PIPE
        )
        assert process.stdout is not None
        archive_error: tarfile.TarError | None = None
        try:
            # r|* streams without seeking, so a multi-GB image never lands on disk.
            with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
                for member in archive:
                    yield member.name + ("/" if member.isdir() else "")
        except tarfile.TarError as exc:
            archive_error = exc
        finally:
            process.stdout.close()
            returncode = process.wait()

        if returncode == 0 and archive_error is None:
            return
        if attempt == max_attempts:
            if returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode, command
                ) from archive_error
            assert archive_error is not None
            raise archive_error

        # Registry exports are large enough to encounter transient 429s even after
        # crane's internal retries. Restart the stream from byte zero; scan() ignores
        # duplicate paths yielded by an interrupted attempt.
        time.sleep(min(2 ** (attempt - 1), 30))


class _LayerProbe:
    """Retain only the opening bytes needed to distinguish a layer from JSON."""

    def __init__(self, handle):
        self.handle = handle
        self.prefix: bytearray | None = bytearray()

    def read(self, size=-1):
        chunk = self.handle.read(size)
        if self.prefix is not None:
            self.prefix.extend(chunk)
        return chunk


def _iter_saved_member(handle, name, documents, scanned_layers):
    probe = _LayerProbe(handle)
    try:
        layer = tarfile.open(fileobj=probe, mode="r|*")
    except tarfile.TarError:
        # Docker's content-addressed store gives JSON and layer blobs the same
        # names. Preserve the small failed header probe without seeking the pipe.
        prefix = bytes(probe.prefix)
        if prefix.lstrip().startswith((b"{", b"[")):
            documents[name] = json.loads(prefix + handle.read())
        yield name
        return
    probe.prefix = None
    with layer:
        for entry in layer:
            yield entry.name + ("/" if entry.isdir() else "")
    scanned_layers.add(name)


def _require_saved_config(name, documents):
    if not isinstance(documents.get(name), dict):
        raise RuntimeError(
            f"Incomplete image archive: missing or invalid config {name}"
        )


def _require_saved_layer(name, scanned_layers):
    if name not in scanned_layers:
        raise RuntimeError(
            f"Incomplete image archive: missing or unreadable layer {name}"
        )


def _check_docker_manifest(manifest, documents, scanned_layers):
    if not isinstance(manifest, list) or not manifest:
        raise RuntimeError("Invalid Docker image archive manifest.json")
    for image in manifest:
        if not isinstance(image, dict) or not isinstance(image.get("Config"), str):
            raise RuntimeError("Invalid Docker image archive config reference")
        _require_saved_config(image["Config"], documents)
        layers = image.get("Layers")
        if not isinstance(layers, list) or not all(
            isinstance(item, str) for item in layers
        ):
            raise RuntimeError("Invalid Docker image archive layer references")
        for name in layers:
            _require_saved_layer(name, scanned_layers)


def _saved_descriptor_path(descriptor, sizes):
    if not isinstance(descriptor, dict):
        raise RuntimeError("Invalid OCI image archive descriptor")
    digest = descriptor.get("digest", "")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[a-z0-9]+:[a-f0-9]+", digest) is None
    ):
        raise RuntimeError("Invalid OCI image archive descriptor digest")
    name = "blobs/" + digest.replace(":", "/", 1)
    if name not in sizes:
        raise RuntimeError(
            f"Incomplete image archive: missing referenced member {name}"
        )
    if descriptor.get("size") != sizes[name]:
        raise RuntimeError(f"Incomplete image archive: referenced size mismatch {name}")
    return name


def _check_oci_manifest(name, documents, scanned_layers, sizes, ancestors=()):
    if name in ancestors:
        raise RuntimeError("Invalid OCI image archive: cyclic index reference")
    document = documents.get(name)
    if not isinstance(document, dict) or document.get("schemaVersion") != 2:
        raise RuntimeError(
            f"Incomplete image archive: missing or invalid manifest {name}"
        )
    if "manifests" in document:
        children = document["manifests"]
        if not isinstance(children, list) or not children:
            raise RuntimeError(f"Invalid OCI image archive: empty index {name}")
        for descriptor in children:
            child = _saved_descriptor_path(descriptor, sizes)
            _check_oci_manifest(
                child, documents, scanned_layers, sizes, (*ancestors, name)
            )
        return
    config = _saved_descriptor_path(document.get("config"), sizes)
    _require_saved_config(config, documents)
    layers = document.get("layers")
    if not isinstance(layers, list):
        raise RuntimeError(f"Invalid OCI image archive: missing layer list {name}")
    for descriptor in layers:
        layer = _saved_descriptor_path(descriptor, sizes)
        _require_saved_layer(layer, scanned_layers)


def _check_saved_image(documents, scanned_layers, sizes):
    if "manifest.json" not in documents and "index.json" not in documents:
        raise RuntimeError("Incomplete image archive: no Docker or OCI manifest")
    if "manifest.json" in documents:
        _check_docker_manifest(documents["manifest.json"], documents, scanned_layers)
    if "index.json" in documents:
        layout = documents.get("oci-layout")
        if not isinstance(layout, dict) or layout.get("imageLayoutVersion") != "1.0.0":
            raise RuntimeError(
                "Incomplete image archive: missing or invalid oci-layout"
            )
        _check_oci_manifest("index.json", documents, scanned_layers, sizes)


def _iter_saved_image(fileobj, *, mode: str):
    """Stream layer paths and require complete Docker/OCI references before success."""
    documents, scanned_layers, sizes = {}, set(), {}
    with tarfile.open(fileobj=fileobj, mode=mode) as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if not member.isfile():
                yield name + ("/" if member.isdir() else "")
                continue
            if name in sizes:
                raise RuntimeError(f"Invalid image archive: duplicate member {name}")
            sizes[name] = member.size
            handle = archive.extractfile(member)
            assert handle is not None
            with handle:
                yield from _iter_saved_member(handle, name, documents, scanned_layers)
    _check_saved_image(documents, scanned_layers, sizes)


def _iter_tarball(tarball: Path):
    """Yield member names from a `docker save` tarball, including inside layer blobs."""
    with tarball.open("rb") as handle:
        yield from _iter_saved_image(handle, mode="r")


_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_MEDIA_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
_CONFIG_MEDIA_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
_LAYER_MEDIA_TYPES = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.docker.image.rootfs.diff.tar",
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
}


@dataclass(frozen=True)
class _SavedImageGraph:
    archive_format: str
    config_digest: str
    config_size: int
    diff_ids: tuple[str, ...]
    layer_names: tuple[str, ...]
    manifest_digest: str | None
    manifest_layers: tuple[dict[str, object], ...] | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_name(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Invalid bound image archive member reference")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("Unsafe bound image archive member reference")
    return str(path)


def _descriptor_fields(
    descriptor: object, media_types: set[str]
) -> tuple[str, int, str]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("mediaType") not in media_types
        or not isinstance(descriptor.get("digest"), str)
        or _SHA256_DIGEST.fullmatch(descriptor["digest"]) is None
        or type(descriptor.get("size")) is not int
        or descriptor["size"] < 0
    ):
        raise RuntimeError("Invalid bound image descriptor")
    return descriptor["digest"], descriptor["size"], descriptor["mediaType"]


def _bound_descriptor(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    descriptor: object,
    media_types: set[str],
) -> tuple[str, bytes]:
    digest, size, _ = _descriptor_fields(descriptor, media_types)
    name = "blobs/sha256/" + digest.removeprefix("sha256:")
    member = members.get(name)
    if member is None or not member.isfile() or member.size != size:
        raise RuntimeError("Incomplete bound image descriptor")
    handle = archive.extractfile(member)
    assert handle is not None
    data = handle.read()
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Bound image descriptor digest mismatch")
    return name, data


def _saved_image_graph(tarball: Path) -> _SavedImageGraph:
    """Read the identity graph shared by classic and containerd Docker saves."""
    with tarfile.open(tarball, "r:*") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            name = _bound_name(member.name)
            if name in members:
                raise RuntimeError("Duplicate bound image archive member")
            members[name] = member
        saved_member = members.get("manifest.json")
        if saved_member is None or not saved_member.isfile():
            raise RuntimeError("Bound image archive requires a saved-image manifest")
        saved_handle = archive.extractfile(saved_member)
        assert saved_handle is not None
        saved = json.load(saved_handle)
        if (
            not isinstance(saved, list)
            or len(saved) != 1
            or not isinstance(saved[0], dict)
            or not isinstance(saved[0].get("Config"), str)
            or not isinstance(saved[0].get("Layers"), list)
            or not saved[0]["Layers"]
            or not all(isinstance(item, str) for item in saved[0]["Layers"])
        ):
            raise RuntimeError("Invalid bound saved-image manifest")
        config_name = _bound_name(saved[0]["Config"])
        layer_names = tuple(_bound_name(item) for item in saved[0]["Layers"])
        config_member = members.get(config_name)
        if config_member is None or not config_member.isfile():
            raise RuntimeError("Incomplete bound saved-image config")
        config_handle = archive.extractfile(config_member)
        assert config_handle is not None
        config_bytes = config_handle.read()
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        config_hash = config_digest.removeprefix("sha256:")
        if (
            PurePosixPath(config_name).stem != config_hash
            and config_name != "blobs/sha256/" + config_hash
        ):
            raise RuntimeError("Saved-image config name does not bind its bytes")
        config = json.loads(config_bytes)
        rootfs = config.get("rootfs") if isinstance(config, dict) else None
        diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, dict) else None
        if (
            not isinstance(rootfs, dict)
            or rootfs.get("type") != "layers"
            or not isinstance(diff_ids, list)
            or len(diff_ids) != len(layer_names)
            or not all(
                isinstance(item, str) and _SHA256_DIGEST.fullmatch(item)
                for item in diff_ids
            )
        ):
            raise RuntimeError("Saved-image config has no complete rootfs identity")
        for name in layer_names:
            member = members.get(name)
            if member is None or not member.isfile():
                raise RuntimeError("Incomplete bound saved-image layer population")

        layout = members.get("oci-layout")
        index_member = members.get("index.json")
        if layout is None and index_member is None:
            return _SavedImageGraph(
                archive_format="docker-save-classic",
                config_digest=config_digest,
                config_size=len(config_bytes),
                diff_ids=tuple(diff_ids),
                layer_names=layer_names,
                manifest_digest=None,
                manifest_layers=None,
            )
        if (
            layout is None
            or index_member is None
            or not layout.isfile()
            or not index_member.isfile()
        ):
            raise RuntimeError("Incomplete bound OCI image layout")
        layout_handle = archive.extractfile(layout)
        index_handle = archive.extractfile(index_member)
        assert layout_handle is not None and index_handle is not None
        if json.load(layout_handle) != {"imageLayoutVersion": "1.0.0"}:
            raise RuntimeError("Invalid bound OCI layout")
        index = json.load(index_handle)
        descriptors = index.get("manifests") if isinstance(index, dict) else None
        descriptor = (
            descriptors[0]
            if isinstance(descriptors, list) and len(descriptors) == 1
            else None
        )
        if (
            not isinstance(index, dict)
            or index.get("schemaVersion") != 2
            or index.get("mediaType", "application/vnd.oci.image.index.v1+json")
            != "application/vnd.oci.image.index.v1+json"
            or not isinstance(descriptor, dict)
        ):
            raise RuntimeError("OCI index does not bind one image manifest")
        _, manifest_bytes = _bound_descriptor(
            archive, members, descriptor, _MANIFEST_MEDIA_TYPES
        )
        manifest_digest, _, manifest_media_type = _descriptor_fields(
            descriptor, _MANIFEST_MEDIA_TYPES
        )
        manifest = json.loads(manifest_bytes)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schemaVersion") != 2
            or manifest.get("mediaType") != manifest_media_type
        ):
            raise RuntimeError("Invalid bound image manifest")
        described_config_name, described_config = _bound_descriptor(
            archive, members, manifest.get("config"), _CONFIG_MEDIA_TYPES
        )
        if described_config_name != config_name or described_config != config_bytes:
            raise RuntimeError("OCI and saved-image configs disagree")
        manifest_layers = manifest.get("layers")
        if not isinstance(manifest_layers, list) or len(manifest_layers) != len(
            layer_names
        ):
            raise RuntimeError("OCI and saved-image layer populations disagree")
        for layer_name, layer_descriptor in zip(
            layer_names, manifest_layers, strict=True
        ):
            layer_digest, layer_size, _ = _descriptor_fields(
                layer_descriptor, _LAYER_MEDIA_TYPES
            )
            expected_name = "blobs/sha256/" + layer_digest.removeprefix("sha256:")
            if layer_name != expected_name or members[layer_name].size != layer_size:
                raise RuntimeError("OCI and saved-image layer order disagrees")
        return _SavedImageGraph(
            archive_format="oci-with-saved-manifest",
            config_digest=config_digest,
            config_size=len(config_bytes),
            diff_ids=tuple(diff_ids),
            layer_names=layer_names,
            manifest_digest=manifest_digest,
            manifest_layers=tuple(manifest_layers),
        )


def _exact_manifest_bytes(payload: bytes, expected_digest: str) -> bytes:
    candidates = [payload]
    if payload.endswith(b"\n"):
        candidates.append(payload[:-1])
    matches = [
        candidate
        for candidate in candidates
        if "sha256:" + hashlib.sha256(candidate).hexdigest() == expected_digest
    ]
    if len(matches) != 1:
        raise RuntimeError("Fetched registry manifest digest mismatch")
    return matches[0]


def _fetch_registry_evidence(
    registry_image: str, expected_manifest_digest: str
) -> tuple[bytes, object]:
    repository, separator, reference_digest = registry_image.rpartition("@")
    if (
        not repository
        or separator != "@"
        or reference_digest != expected_manifest_digest
    ):
        raise RuntimeError("Registry evidence requires an exact-digest image reference")
    crane = _require("crane")
    digest = subprocess.run(
        [crane, "digest", registry_image],
        capture_output=True,
        check=False,
    )  # noqa: S603
    if (
        digest.returncode != 0
        or digest.stdout.decode(errors="replace").strip() != expected_manifest_digest
    ):
        raise RuntimeError("Could not independently verify registry manifest digest")
    manifest = subprocess.run(
        [crane, "manifest", registry_image],
        capture_output=True,
        check=False,
    )  # noqa: S603
    if manifest.returncode != 0:
        raise RuntimeError("Could not fetch exact registry manifest evidence")
    manifest_bytes = _exact_manifest_bytes(manifest.stdout, expected_manifest_digest)
    docker = _require("docker")
    inspected = subprocess.run(
        [docker, "image", "inspect", registry_image],
        capture_output=True,
        check=False,
    )  # noqa: S603
    if inspected.returncode != 0:
        raise RuntimeError("Could not inspect the exact pulled registry image")
    return manifest_bytes, json.loads(inspected.stdout)


def _bind_registry_graph(
    graph: _SavedImageGraph,
    manifest_bytes: bytes,
    inspected: object,
    expected_manifest_digest: str,
) -> tuple[int, str]:
    manifest_bytes = _exact_manifest_bytes(manifest_bytes, expected_manifest_digest)
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 2
        or manifest.get("mediaType") not in _MANIFEST_MEDIA_TYPES
    ):
        raise RuntimeError("Invalid exact registry image manifest")
    config_digest, config_size, _ = _descriptor_fields(
        manifest.get("config"), _CONFIG_MEDIA_TYPES
    )
    if config_digest != graph.config_digest or config_size != graph.config_size:
        raise RuntimeError("Registry manifest and saved-image configs disagree")
    registry_layers = manifest.get("layers")
    if not isinstance(registry_layers, list) or len(registry_layers) != len(
        graph.layer_names
    ):
        raise RuntimeError("Registry and saved-image layer populations disagree")
    for descriptor in registry_layers:
        _descriptor_fields(descriptor, _LAYER_MEDIA_TYPES)

    image = (
        inspected[0]
        if isinstance(inspected, list)
        and len(inspected) == 1
        and isinstance(inspected[0], dict)
        else None
    )
    repo_digests = image.get("RepoDigests") if isinstance(image, dict) else None
    rootfs = image.get("RootFS") if isinstance(image, dict) else None
    if (
        not isinstance(image, dict)
        or image.get("Id")
        not in (
            {graph.config_digest, expected_manifest_digest}
            if graph.manifest_digest is not None
            else {graph.config_digest}
        )
        or not isinstance(repo_digests, list)
        or not any(
            isinstance(item, str) and item.endswith("@" + expected_manifest_digest)
            for item in repo_digests
        )
        or not isinstance(rootfs, dict)
        or rootfs.get("Type") != "layers"
        or rootfs.get("Layers") != list(graph.diff_ids)
    ):
        raise RuntimeError("Pulled registry image does not bind the saved-image graph")

    if graph.manifest_digest is not None:
        if (
            graph.manifest_digest != expected_manifest_digest
            or graph.manifest_layers is None
            or [
                _descriptor_fields(item, _LAYER_MEDIA_TYPES)
                for item in graph.manifest_layers
            ]
            != [
                _descriptor_fields(item, _LAYER_MEDIA_TYPES) for item in registry_layers
            ]
        ):
            raise RuntimeError("Registry and OCI archive descriptor graphs disagree")
        layer_binding_kind = "exact-oci-layer-descriptors"
    else:
        # A classic save retains uncompressed layer tar streams and config diff IDs,
        # not registry-compressed descriptors. Docker's successful exact-digest pull
        # binds those descriptor positions to the inspected rootfs diff-ID sequence.
        layer_binding_kind = "exact-pull-config-diff-ids"
    return len(registry_layers), layer_binding_kind


def _verify_archive_binding(
    tarball: Path,
    verification_report: Path,
    *,
    registry_image: str | None = None,
    expected_manifest_digest: str | None = None,
) -> dict[str, object]:
    """Bind a clean local payload result to successful complete-graph evidence."""
    if (registry_image is None) != (expected_manifest_digest is None):
        raise RuntimeError(
            "Registry image and expected manifest digest are inseparable"
        )
    if expected_manifest_digest is not None and (
        _SHA256_DIGEST.fullmatch(expected_manifest_digest) is None
    ):
        raise RuntimeError("Require an exact expected registry manifest digest")
    report_bytes = verification_report.read_bytes()
    report = json.loads(report_bytes)
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "npa.curobo.image-verification.v1"
        or report.get("valid") is not True
        or report.get("findings") != []
    ):
        raise RuntimeError("Require a successful complete image verification report")
    archive_sha256 = report.get("docker_save_sha256")
    manifest_digest = report.get("image_manifest_digest")
    config_digest = report.get("image_config_digest")
    inspected_identity = report.get("expected_image_id")
    verified_diff_ids = report.get("verified_layer_diff_ids")
    layer_count = report.get("layer_count")
    if (
        not isinstance(archive_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
        or not isinstance(config_digest, str)
        or _SHA256_DIGEST.fullmatch(config_digest) is None
        or not isinstance(inspected_identity, str)
        or _SHA256_DIGEST.fullmatch(inspected_identity) is None
        or (
            manifest_digest is not None
            and (
                not isinstance(manifest_digest, str)
                or _SHA256_DIGEST.fullmatch(manifest_digest) is None
            )
        )
        or not isinstance(verified_diff_ids, list)
        or not all(
            isinstance(item, str) and _SHA256_DIGEST.fullmatch(item)
            for item in verified_diff_ids
        )
        or type(layer_count) is not int
    ):
        raise RuntimeError("Complete image report has no exact archive graph binding")
    if _sha256_file(tarball) != archive_sha256:
        raise RuntimeError("Image archive does not match complete verification report")
    graph = _saved_image_graph(tarball)
    if (
        graph.config_digest != config_digest
        or graph.manifest_digest != manifest_digest
        or list(graph.diff_ids) != verified_diff_ids
        or len(graph.layer_names) != layer_count
        or inspected_identity not in {graph.config_digest, graph.manifest_digest}
        or (graph.manifest_digest is None and inspected_identity != graph.config_digest)
    ):
        raise RuntimeError("Image archive does not match complete graph evidence")

    binding: dict[str, object] = {
        "schema_version": "npa.restricted-payload.archive-binding.v2",
        "docker_save_sha256": archive_sha256,
        "archive_format": graph.archive_format,
        "image_config_digest": graph.config_digest,
        "layer_count": len(graph.layer_names),
        "archive_layer_binding_kind": (
            "verified-oci-layer-descriptors"
            if graph.manifest_digest is not None
            else "verified-config-diff-ids"
        ),
        "verification_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    if registry_image is not None:
        assert expected_manifest_digest is not None
        manifest_bytes, registry_inspect = _fetch_registry_evidence(
            registry_image, expected_manifest_digest
        )
        registry_layer_count, layer_binding_kind = _bind_registry_graph(
            graph,
            manifest_bytes,
            registry_inspect,
            expected_manifest_digest,
        )
        binding.update(
            {
                "content_identity_kind": "registry-manifest-digest",
                "content_identity": expected_manifest_digest,
                "registry_manifest_digest": expected_manifest_digest,
                "registry_manifest_evidence_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
                "registry_layer_count": registry_layer_count,
                "registry_layer_binding_kind": layer_binding_kind,
            }
        )
    elif graph.manifest_digest is not None:
        binding.update(
            {
                "content_identity_kind": "oci-manifest-digest",
                "content_identity": graph.manifest_digest,
                "image_manifest_digest": graph.manifest_digest,
            }
        )
    else:
        binding.update(
            {
                "content_identity_kind": "image-config-digest",
                "content_identity": graph.config_digest,
            }
        )
    return binding


def _iter_docker_save(image: str):
    """Stream all local image layers without materialising a second image-sized file."""
    docker = _require("docker")
    command = [docker, "save", image]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)  # noqa: S603
    assert process.stdout is not None
    try:
        yield from _iter_saved_image(process.stdout, mode="r|*")
    finally:
        process.stdout.close()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _local_image_history(image: str) -> list[str]:
    docker = _require("docker")
    command = [
        docker,
        "history",
        "--no-trunc",
        "--format",
        "{{json .CreatedBy}}",
        image,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def _image_history(image: str) -> tuple[list[str], str | None]:
    crane = _require("crane")
    config_command = [crane, "config", image]
    config = subprocess.run(  # noqa: S603 - fixed argv
        config_command, capture_output=True, text=True, check=False
    )
    if config.returncode != 0:
        raise subprocess.CalledProcessError(config.returncode, config_command)
    payload = json.loads(config.stdout)
    commands = [
        entry.get("created_by", "")
        for entry in payload.get("history", [])
        if entry.get("created_by")
    ]
    digest_command = [crane, "digest", "--platform", "linux/amd64", image]
    digest = subprocess.run(  # noqa: S603 - fixed argv
        digest_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if digest.returncode != 0:
        raise subprocess.CalledProcessError(digest.returncode, digest_command)
    digest_value = digest.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest_value) is None:
        raise RuntimeError("crane returned an invalid linux/amd64 image digest")
    return commands, digest_value


def scan(
    image: str | None,
    tarball: Path | None,
    *,
    docker_image: str | None = None,
    verification_report: Path | None = None,
    registry_image: str | None = None,
    expected_manifest_digest: str | None = None,
    max_report: int = 40,
    history_only: bool = False,
) -> ScanReport:
    """Scan an image for Omniverse payload.

    ``history_only`` skips the filesystem walk and inspects only the layer history, which
    needs the config blob (a few KB) instead of streaming the whole image (tens of GB).
    That makes it usable as a pre-publish gate in CI, where streaming 69 GB is not.

    It is strictly weaker: it catches a build that RAN an Isaac install, not a payload
    that arrived some other way (a COPY from a vendor stage, an ADD of a tarball). Use it
    as a fast gate in front of an irreversible action, never as the proof itself -- the
    full scan is what the redistribution claim actually rests on.
    """
    if docker_image is not None:
        report = ScanReport(image=docker_image, source="local-docker-stream")
        entries = () if history_only else _iter_docker_save(docker_image)
        history = _local_image_history(docker_image)
    elif tarball is not None:
        report = ScanReport(image=str(tarball), source="tarball")
        if (registry_image is None) != (expected_manifest_digest is None):
            raise RuntimeError(
                "Registry image and expected manifest digest are inseparable"
            )
        if registry_image is not None and verification_report is None:
            raise RuntimeError("Registry binding requires a graph verification report")
        if verification_report is not None:
            report.archive_binding = _verify_archive_binding(
                tarball,
                verification_report,
                registry_image=registry_image,
                expected_manifest_digest=expected_manifest_digest,
            )
            identity = report.archive_binding["content_identity"]
            assert isinstance(identity, str)
            report.digest = identity
        entries = _iter_tarball(tarball)
        history: list[str] = []
    else:
        assert image is not None
        report = ScanReport(image=image, source="registry")
        history, report.digest = _image_history(image)
        entries = () if history_only else _iter_crane_export(image)
    report.history_only = history_only

    seen_paths: set[str] = set()
    for path in entries:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        report.entries_scanned += 1
        if is_allowed(path):
            report.allowlisted_hits.append(_normalize(path))
            continue
        if (
            path.endswith(WEIGHT_SUFFIXES)
            and len(report.weight_shaped_paths) < max_report
        ):
            report.weight_shaped_paths.append(_normalize(path))
        why = classify_path(path)
        if why and len(report.payload_hits) < max_report:
            report.payload_hits.append({"path": _normalize(path), "why": why})

    for command in history:
        why = classify_history(command)
        if why:
            report.history_hits.append({"command": command.strip()[:400], "why": why})

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", help="Image reference to scan with crane.")
    parser.add_argument(
        "--tarball",
        type=Path,
        help="Scan a `docker save` tarball instead of a registry.",
    )
    parser.add_argument(
        "--docker-image",
        help=(
            "Scan every layer of a local Docker image through a streaming `docker save`; "
            "no image-sized temporary archive is created."
        ),
    )
    parser.add_argument(
        "--verification-report",
        type=Path,
        help=("Successful complete-graph report for this exact local tarball."),
    )
    parser.add_argument(
        "--registry-image",
        help=(
            "Exact digest-qualified registry image whose fetched manifest and pulled "
            "Docker graph must match this tarball. Requires "
            "--expected-manifest-digest and --verification-report."
        ),
    )
    parser.add_argument(
        "--expected-manifest-digest",
        help=(
            "Exact sha256 registry manifest identity to fetch and bind. Requires "
            "--registry-image and --verification-report."
        ),
    )
    parser.add_argument("--json", type=Path, help="Write the JSON report here.")
    parser.add_argument(
        "--history-only",
        action="store_true",
        help=(
            "Inspect only the layer history (config blob, a few KB) instead of streaming "
            "the whole filesystem. Fast enough for a CI pre-publish gate; strictly weaker "
            "than a full scan, so never treat it as the redistribution proof itself."
        ),
    )
    args = parser.parse_args(argv)

    selected = sum(
        bool(value) for value in (args.image, args.tarball, args.docker_image)
    )
    if selected != 1:
        parser.error("pass exactly one image reference, --tarball, or --docker-image")
    if (args.registry_image is None) != (args.expected_manifest_digest is None):
        parser.error(
            "--registry-image and --expected-manifest-digest must be provided together"
        )
    if args.registry_image is not None and args.verification_report is None:
        parser.error("--registry-image requires --verification-report")
    if (
        args.verification_report is not None or args.registry_image is not None
    ) and args.tarball is None:
        parser.error("archive graph binding is supported only with --tarball")

    report = scan(
        args.image,
        args.tarball,
        docker_image=args.docker_image,
        verification_report=args.verification_report,
        registry_image=args.registry_image,
        expected_manifest_digest=args.expected_manifest_digest,
        history_only=args.history_only,
    )
    payload = report.to_dict()

    print(f"image            {payload['image']}")
    if payload["digest"]:
        print(f"digest           {payload['digest']}")
    if report.history_only:
        print("mode             history-only (layer commands; filesystem NOT scanned)")
    print(f"entries scanned  {payload['entries_scanned']}")
    print(
        f"allowlisted      {len(payload['allowlisted_paths_present'])} path(s) we do ship:"
    )
    for path in payload["allowlisted_paths_present"][:20]:
        print(f"                   {path}")
    if report.payload_hits:
        print(
            f"\nRESTRICTED RUNTIME PAYLOAD DETECTED ({len(report.payload_hits)} path(s)):"
        )
        for hit in report.payload_hits:
            print(f"  {hit['path']}\n      -> {hit['why']}")
    if report.history_hits:
        print(
            f"\nBUILD-TIME ISAAC INSTALL DETECTED ({len(report.history_hits)} layer(s)):"
        )
        for hit in report.history_hits:
            print(f"  {hit['why']}\n      {hit['command']}")
    if report.weight_shaped_paths:
        print(
            f"\nFYI - {len(report.weight_shaped_paths)} weight-shaped path(s) present. "
            f"A tar listing has no contents, so these may be git-LFS pointers (fine) or "
            f"real tensors (not fine). The image build checks this by content:"
        )
        for path in report.weight_shaped_paths[:15]:
            print(f"  {path}")
    print(f"\nVERDICT: {payload['verdict']}")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"report written to {args.json}")

    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
