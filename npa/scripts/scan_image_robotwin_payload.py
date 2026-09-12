#!/usr/bin/env python3
"""Fail closed when a RoboTwin bootstrap contains any runtime/vendor payload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, Mapping
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
    parse_http_list,
    parse_keqv_list,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_wan_payload as walker  # noqa: E402


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "robotwin_source",
        re.compile(
            r"(?:^|/)(?:(?:opt|workspace|src)/(?:RoboTwin|robotwin-source)(?:/|$)|"
            r"script/collect_data\.py$|envs/_base_task\.py$|task_config/[^/]+\.ya?ml$)",
            re.I,
        ),
    ),
    (
        "curobo_source_or_runtime",
        re.compile(r"(?:^|/)(?:opt/)?curobo(?:/|$)", re.I),
    ),
    (
        "vendor_python_runtime",
        re.compile(
            r"(?:^|/)site-packages/(?:curobo|sapien|mplib|warp|torch|torchvision|"
            r"nvidia|cv2|h5py)(?:[./_-]|/|$)",
            re.I,
        ),
    ),
    (
        "cuda_or_cudnn_runtime",
        re.compile(
            r"(?:^|/)(?:usr/local/cuda(?:/|$)|[^/]*(?:lib)?cu(?:da|dnn|blas|fft|"
            r"rand|solver|sparse|pti|rtc)[^/]*\.so(?:[./0-9]*|$))",
            re.I,
        ),
    ),
    (
        "robotwin_asset_archive",
        re.compile(r"(?:^|/)(?:embodiments|objects|background_texture)\.zip$", re.I),
    ),
    (
        "robotwin_extracted_asset",
        re.compile(
            r"(?:^|/)(?:assets/)?(?:embodiments|objects)/[^/]+(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_huggingface_cache",
        re.compile(
            r"(?:^|/)datasets--TianxingChen--RoboTwin2\.0(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_runtime_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:huggingface|torch|warp)|robotwin-runtime-cache|"
            r"runtime-ready\.json)(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_generated_output",
        re.compile(
            r"(?:^|/)(?:robotwin-native(?:/|$)|robotwin-smoke\.json$|"
            r"episode_[0-9]+\.(?:hdf5|mp4)$|[^/]+\.(?:hdf5|mp4)$|frames?(?:/|$))",
            re.I,
        ),
    ),
    (
        "credential_or_manager_context",
        re.compile(
            r"(?:^|/)(?:runtime-context\.json|kubeconfig(?:\.ya?ml)?|"
            r"docker/config\.json|\.aws/credentials)$",
            re.I,
        ),
    ),
    (
        "source_control_metadata",
        re.compile(r"(?:^|/)\.(?:git|hg|svn)(?:/|$)", re.I),
    ),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "robotwin_asset_fetch_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:hf_hub_download|(?:hf|huggingface-cli)\s+download|"
            r"(?:curl|wget)\b)[^\n]*(?:TianxingChen/RoboTwin2\.0|"
            r"embodiments\.zip|objects\.zip|background_texture\.zip)",
            re.I | re.S,
        ),
    ),
    (
        "vendor_runtime_installed_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:pip|uv)\s+(?:install|sync)[^\n]*"
            r"(?:curobo|sapien|mplib|warp-lang|torch|nvidia-cuda|nvidia-cudnn)",
            re.I | re.S,
        ),
    ),
    (
        "vendor_source_fetched_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:git\s+clone|curl|wget)[^\n]*"
            r"(?:RoboTwin-Platform/RoboTwin|NVlabs/curobo)",
            re.I | re.S,
        ),
    ),
    (
        "nvidia_or_pytorch_base",
        re.compile(r"\bFROM\s+(?:nvidia/cuda|nvcr\.io/|pytorch/)", re.I),
    ),
)

FORBIDDEN_ELF_DEPENDENCY = re.compile(
    rb"(?:libcuda|libcudnn|libcublas|libcudart|libnvrtc|libtorch)[^\x00]*\.so",
    re.I,
)

SECRET_CONTENT = (
    *walker.SECRET_CONTENT,
    re.compile(
        rb'(?i)"ownership_provenance"\s*:\s*"manager-issued"'
    ),
)
MAX_IMAGE_REFERENCE_BYTES = 2048
MAX_DOCKER_CONFIG_BYTES = 1024 * 1024
MAX_REGISTRY_METADATA_BYTES = 8 * 1024 * 1024
REGISTRY_TIMEOUT_SECONDS = 60
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Do not forward registry authorization to a different redirect origin."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old = urlsplit(request.full_url)
        new = urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            redirected.remove_header("Authorization")
        return redirected


_URL_OPENER = build_opener(_SafeRedirectHandler())


def _read_bounded(stream: BinaryIO, limit: int) -> bytes:
    raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise RuntimeError("registry metadata exceeds the scanner byte limit")
    return raw


def _read_docker_config(path: Path) -> dict[str, Any]:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_DOCKER_CONFIG_BYTES:
            raise RuntimeError("Docker credential configuration is invalid")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = _read_bounded(stream, MAX_DOCKER_CONFIG_BYTES)
    except FileNotFoundError:
        return {}
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Docker credential configuration is invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Docker credential configuration is invalid")
    return payload


def _credential_helper(helper: str, registry: str) -> dict[str, str]:
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", helper) is None:
        raise RuntimeError("Docker credential helper name is invalid")
    completed = subprocess.run(
        [f"docker-credential-{helper}", "get"],
        input=f"{registry}\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Docker credential helper failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Docker credential helper returned invalid data") from exc
    username = str(payload.get("Username") or "") if isinstance(payload, dict) else ""
    secret = str(payload.get("Secret") or "") if isinstance(payload, dict) else ""
    if not secret:
        raise RuntimeError("Docker credential helper returned no credential")
    return {"username": username, "secret": secret}


def _docker_credentials(
    registry: str, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    config_root = str(source.get("DOCKER_CONFIG") or "").strip()
    if config_root:
        config_path = Path(config_root) / "config.json"
    else:
        config_path = Path(str(source.get("HOME") or "")) / ".docker" / "config.json"
    config = _read_docker_config(config_path)
    keys = (registry, f"https://{registry}", f"https://{registry}/v1/")
    helpers = config.get("credHelpers")
    if isinstance(helpers, dict):
        for key in keys:
            helper = helpers.get(key)
            if isinstance(helper, str) and helper:
                return _credential_helper(helper, registry)
    auths = config.get("auths")
    if isinstance(auths, dict):
        for key in keys:
            record = auths.get(key)
            if not isinstance(record, dict):
                continue
            identity_token = str(record.get("identitytoken") or "")
            if identity_token:
                return {"identity_token": identity_token}
            encoded = str(record.get("auth") or "")
            if encoded:
                try:
                    decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
                except (UnicodeDecodeError, ValueError) as exc:
                    raise RuntimeError("Docker registry credential is invalid") from exc
                username, separator, secret = decoded.partition(":")
                if not separator or not secret:
                    raise RuntimeError("Docker registry credential is invalid")
                return {"username": username, "secret": secret}
    store = config.get("credsStore")
    if isinstance(store, str) and store:
        return _credential_helper(store, registry)
    return {}


def _parse_private_image(image: str) -> tuple[str, str, str]:
    reference = image.removeprefix("docker:")
    repository_ref, separator, digest = reference.rpartition("@")
    if not separator or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError("private image reference must use an immutable sha256 digest")
    registry, slash, repository = repository_ref.partition("/")
    if (
        not slash
        or not registry
        or not repository
        or any(character.isspace() for character in reference)
        or any(character in registry for character in "/@?#")
        or re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]{1,5})?", registry, re.I)
        is None
        or re.fullmatch(
            r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*",
            repository,
            re.I,
        )
        is None
    ):
        raise RuntimeError("private image reference is malformed")
    return registry, repository, digest


class _RegistryClient:
    """Minimal pull-only OCI client that never puts a private reference in argv."""

    def __init__(self, registry: str, repository: str) -> None:
        self.registry = registry
        self.repository = repository
        self.base_url = (
            f"https://{registry}/v2/"
            + "/".join(quote(part, safe="._-") for part in repository.split("/"))
        )
        credentials = _docker_credentials(registry)
        self._basic = ""
        self._authorization = ""
        identity_token = credentials.get("identity_token", "")
        if identity_token:
            self._authorization = f"Bearer {identity_token}"
        elif credentials.get("secret"):
            raw = f"{credentials.get('username', '')}:{credentials['secret']}".encode()
            self._basic = "Basic " + base64.b64encode(raw).decode("ascii")
            self._authorization = self._basic

    def _bearer_authorization(self, challenge: str) -> str:
        scheme, separator, parameters = challenge.partition(" ")
        if scheme.lower() != "bearer" or not separator:
            if scheme.lower() == "basic" and self._basic:
                return self._basic
            raise RuntimeError("registry authentication challenge is unsupported")
        try:
            values = parse_keqv_list(parse_http_list(parameters))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("registry authentication challenge is invalid") from exc
        realm = str(values.get("realm") or "")
        parsed_realm = urlsplit(realm)
        if (
            parsed_realm.scheme != "https"
            or not parsed_realm.netloc
            or parsed_realm.username
            or parsed_realm.password
            or parsed_realm.fragment
        ):
            raise RuntimeError("registry token service is not secure")
        query = {
            "service": str(values.get("service") or ""),
            "scope": str(
                values.get("scope") or f"repository:{self.repository}:pull"
            ),
        }
        query = {key: value for key, value in query.items() if value}
        token_url = realm + ("&" if parsed_realm.query else "?") + urlencode(query)
        request = Request(token_url, headers={"Accept": "application/json"})
        if self._basic:
            request.add_header("Authorization", self._basic)
        with _URL_OPENER.open(request, timeout=REGISTRY_TIMEOUT_SECONDS) as response:
            raw = _read_bounded(response, MAX_REGISTRY_METADATA_BYTES)
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("registry token response is invalid") from exc
        token = (
            str(payload.get("token") or payload.get("access_token") or "")
            if isinstance(payload, dict)
            else ""
        )
        if not token:
            raise RuntimeError("registry token response has no token")
        return f"Bearer {token}"

    def open(self, path: str, *, accept: str = "application/octet-stream"):
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(2):
            request = Request(url, headers={"Accept": accept})
            if self._authorization:
                request.add_header("Authorization", self._authorization)
            try:
                return _URL_OPENER.open(request, timeout=REGISTRY_TIMEOUT_SECONDS)
            except HTTPError as exc:
                if exc.code != 401 or attempt:
                    raise RuntimeError("registry request failed") from None
                challenge = str(exc.headers.get("WWW-Authenticate") or "")
                self._authorization = self._bearer_authorization(challenge)
        raise RuntimeError("registry authorization failed")

    def metadata(self, path: str, *, accept: str, digest: str = "") -> bytes:
        with self.open(path, accept=accept) as response:
            raw = _read_bounded(response, MAX_REGISTRY_METADATA_BYTES)
        if digest and hashlib.sha256(raw).hexdigest() != digest.removeprefix("sha256:"):
            raise RuntimeError("registry metadata digest mismatch")
        return raw

    def blob(self, digest: str, destination: Path, *, expected_size: int | None) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise RuntimeError("OCI blob digest is invalid")
        calculated = hashlib.sha256()
        size = 0
        with self.open(f"blobs/{quote(digest, safe=':')}") as response:
            with destination.open("xb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    calculated.update(chunk)
                    stream.write(chunk)
        if calculated.hexdigest() != digest.removeprefix("sha256:"):
            raise RuntimeError("OCI blob digest mismatch")
        if expected_size is not None and size != expected_size:
            raise RuntimeError("OCI blob size mismatch")


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OCI {label} is invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"OCI {label} is invalid")
    return payload


def _descriptor(record: Any, label: str) -> tuple[str, int | None]:
    if not isinstance(record, dict):
        raise RuntimeError(f"OCI {label} descriptor is invalid")
    digest = str(record.get("digest") or "")
    size_value = record.get("size")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"OCI {label} digest is invalid")
    if size_value is not None and (type(size_value) is not int or size_value < 0):
        raise RuntimeError(f"OCI {label} size is invalid")
    return digest, size_value


def _write_flattened_rootfs(layers: list[Path], destination: Path, temp_dir: Path) -> None:
    """Create a whiteout-aware rootfs tar without extracting untrusted paths."""

    entries: dict[str, tuple[str, Path | None, str, int]] = {}
    content_root = temp_dir / "rootfs-content"
    content_root.mkdir(mode=0o700)
    content_index = 0
    for layer in layers:
        with tarfile.open(layer, "r:*") as archive:
            for member in archive:
                path = member.name
                while path.startswith("./"):
                    path = path[2:]
                path = path.rstrip("/")
                parts = path.split("/") if path else []
                if (
                    not path
                    or path.startswith("/")
                    or any(part in {"", ".", ".."} for part in parts)
                    or "\x00" in path
                ):
                    raise RuntimeError("OCI layer contains an unsafe path")
                leaf = parts[-1]
                parent = "/".join(parts[:-1])
                if leaf == ".wh..wh..opq":
                    prefix = f"{parent}/" if parent else ""
                    entries = {
                        name: value
                        for name, value in entries.items()
                        if not name.startswith(prefix) or name == parent
                    }
                    continue
                if leaf.startswith(".wh."):
                    target = "/".join((*parts[:-1], leaf.removeprefix(".wh.")))
                    entries = {
                        name: value
                        for name, value in entries.items()
                        if name != target and not name.startswith(f"{target}/")
                    }
                    continue
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError("OCI layer file is unreadable")
                    content = content_root / f"{content_index:08d}"
                    content_index += 1
                    with extracted, content.open("xb") as stream:
                        shutil.copyfileobj(extracted, stream)
                    entries[path] = ("file", content, "", member.mode & 0o7777)
                elif member.isdir():
                    entries[path] = ("dir", None, "", member.mode & 0o7777)
                elif member.issym() or member.islnk():
                    link = member.linkname
                    if "\x00" in link:
                        raise RuntimeError("OCI layer link is invalid")
                    entries[path] = (
                        "symlink" if member.issym() else "hardlink",
                        None,
                        link,
                        member.mode & 0o7777,
                    )
                else:
                    raise RuntimeError("OCI layer contains an unsupported special file")
    with tarfile.open(destination, "x") as archive:
        for path, (kind, content, link, mode) in sorted(entries.items()):
            info = tarfile.TarInfo(path)
            info.mode = mode
            if kind == "file":
                assert content is not None
                info.size = content.stat().st_size
                with content.open("rb") as stream:
                    archive.addfile(info, stream)
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                info.linkname = link
                archive.addfile(info)


def _private_remote_material(image: str, temp_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    registry, repository, digest = _parse_private_image(image)
    client = _RegistryClient(registry, repository)
    manifest_raw = client.metadata(
        f"manifests/{quote(digest, safe=':')}", accept=MANIFEST_ACCEPT, digest=digest
    )
    manifest = _json_object(manifest_raw, "manifest")
    if isinstance(manifest.get("manifests"), list):
        candidates = [
            record
            for record in manifest["manifests"]
            if isinstance(record, dict)
            and isinstance(record.get("platform"), dict)
            and record["platform"].get("os") == "linux"
            and record["platform"].get("architecture") == "amd64"
            and not record["platform"].get("variant")
        ]
        if len(candidates) != 1:
            raise RuntimeError("OCI index does not select exactly one linux/amd64 image")
        child_digest, child_size = _descriptor(candidates[0], "platform manifest")
        manifest_raw = client.metadata(
            f"manifests/{quote(child_digest, safe=':')}",
            accept=MANIFEST_ACCEPT,
            digest=child_digest,
        )
        if child_size is not None and len(manifest_raw) != child_size:
            raise RuntimeError("OCI platform manifest size mismatch")
        manifest = _json_object(manifest_raw, "platform manifest")
    config_digest, config_size = _descriptor(manifest.get("config"), "config")
    config_path = temp_dir / "config.json"
    client.blob(config_digest, config_path, expected_size=config_size)
    config = _json_object(config_path.read_bytes(), "config")
    layer_records = manifest.get("layers")
    if not isinstance(layer_records, list) or not layer_records:
        raise RuntimeError("OCI manifest contains no layers")
    layers: list[Path] = []
    for index, record in enumerate(layer_records):
        layer_digest, layer_size = _descriptor(record, f"layer {index}")
        layer_path = temp_dir / f"layer-{index:03d}.tar"
        client.blob(layer_digest, layer_path, expected_size=layer_size)
        layers.append(layer_path)
    rootfs = temp_dir / "rootfs.tar"
    _write_flattened_rootfs(layers, rootfs, temp_dir)
    return [rootfs, *layers], config


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    """Scan one tar plus image history under the RoboTwin boundary policy."""

    return scan_tars([rootfs_tar], config)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    """Scan layer/rootfs tars plus image history under the RoboTwin policy."""

    with walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files={},
        audited_libraries={},
        secret_content=SECRET_CONTENT,
        forbidden_elf_dependency=FORBIDDEN_ELF_DEPENDENCY,
    ):
        return walker.scan_tars(tars, config)


docker_save_material = walker.docker_save_material


def _read_image_stdin() -> str:
    """Read one bounded private image reference without placing it in argv."""

    raw = sys.stdin.buffer.read(MAX_IMAGE_REFERENCE_BYTES + 1)
    if len(raw) > MAX_IMAGE_REFERENCE_BYTES:
        raise ValueError("private image reference is too large")
    try:
        image = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("private image reference is not UTF-8") from exc
    if not image or image != image.strip() or "\x00" in image:
        raise ValueError("private image reference is malformed")
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--image-stdin", action="store_true")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    choices = (args.image, args.image_stdin, args.rootfs_tar, args.docker_save)
    if sum(bool(value) for value in choices) != 1:
        parser.error("provide exactly one IMAGE, --rootfs-tar, or --docker-save")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")

    selected_image = args.image
    try:
        if args.image_stdin:
            selected_image = _read_image_stdin()
        with tempfile.TemporaryDirectory(prefix="npa-robotwin-byte-scan-") as tmp:
            if selected_image:
                tars, config = (
                    _private_remote_material(selected_image, Path(tmp))
                    if args.image_stdin
                    else walker.remote_material(selected_image, Path(tmp))
                )
            elif args.docker_save:
                tars, config = docker_save_material(args.docker_save, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = scan_tars(tars, config)
    except Exception as exc:  # noqa: BLE001 - every scan failure is fatal
        error = "private image scan failed" if args.image_stdin else str(exc)
        print(json.dumps({"status": "error", "error": error}, indent=2))
        return 2

    result = {
        "format": "npa_robotwin_image_byte_scan_v1",
        "image": selected_image or (
            "docker-save" if args.docker_save else "offline-rootfs"
        ),
        "status": "pass" if not findings else "fail",
        "archives_scanned": len(tars),
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
