"""Immutable Antioch project staging with archive traversal protection."""

from __future__ import annotations

import hashlib
import gzip
import json
import re
import tarfile
from pathlib import Path

from npa.clients.storage import StorageClient

from .schemas import ProjectManifest
from .storage import join_uri, sha256_file


class AntiochProjectError(RuntimeError):
    pass


_FORBIDDEN = {
    ".env",
    "auth.json",
    "credentials.json",
    "machines.json",
    "id_rsa",
    "id_ed25519",
}


def deterministic_project_id(workflow_run: str, state_id: str) -> str:
    digest = hashlib.sha256(f"{workflow_run}\n{state_id}".encode()).hexdigest()[:20]
    slug = re.sub(r"[^a-z0-9-]+", "-", state_id.lower()).strip("-")[:28] or "run"
    return f"npa-{slug}-{digest}"


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = Path(member.name)
    return (
        bool(member.name)
        and not path.is_absolute()
        and ".." not in path.parts
        and not member.issym()
        and not member.islnk()
        and not member.isdev()
        and (member.isfile() or member.isdir())
    )


def _assert_no_secrets(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered in _FORBIDDEN or lowered.endswith((".pem", ".key", ".p12", ".pfx")):
            raise AntiochProjectError(
                f"project archive contains forbidden credential file: {path.name}"
            )


def _rewrite_project_id(manifest_path: Path, project_id: str) -> None:
    """Change only the top-level manifest id without adding a YAML dependency."""

    text = manifest_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^(id\s*:\s*).*$")
    if not pattern.search(text):
        raise AntiochProjectError("antioch.yaml has no top-level id field")
    manifest_path.write_text(
        pattern.sub(rf"\g<1>{project_id}", text, count=1), encoding="utf-8"
    )


def stage_project(
    client: StorageClient,
    input_path: str,
    destination: Path,
    *,
    project_id: str,
) -> tuple[Path, ProjectManifest, str]:
    """Download, verify, safely extract, and deterministically identify one project."""

    destination.mkdir(parents=True, exist_ok=True)
    manifest_result = client.read_bytes_with_etag(
        join_uri(input_path, "project-manifest.json")
    )
    if manifest_result is None:
        raise AntiochProjectError("project-manifest.json is missing from input_path")
    try:
        raw_manifest = json.loads(manifest_result[0])
        manifest = ProjectManifest.model_validate(raw_manifest)
    except Exception as exc:
        raise AntiochProjectError("project-manifest.json is malformed") from exc
    archive = destination / manifest.archive.name
    client.download_file(join_uri(input_path, manifest.archive.name), str(archive))
    if archive.stat().st_size != manifest.archive.size_bytes:
        raise AntiochProjectError(
            "project archive size does not match its immutable manifest"
        )
    digest = sha256_file(archive)
    if digest != manifest.archive.sha256:
        raise AntiochProjectError(
            "project archive checksum does not match its immutable manifest"
        )
    project = destination / "project"
    project.mkdir()
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if not members or not all(_safe_member(member) for member in members):
                raise AntiochProjectError(
                    "project archive contains an unsafe path or link"
                )
            bundle.extractall(project)
    except (tarfile.TarError, OSError) as exc:
        raise AntiochProjectError(
            "project archive could not be safely extracted"
        ) from exc
    candidates = list(project.rglob("antioch.yaml"))
    if len(candidates) != 1:
        raise AntiochProjectError(
            "project archive must contain exactly one antioch.yaml"
        )
    root = candidates[0].parent
    _assert_no_secrets(root)
    _rewrite_project_id(candidates[0], project_id)
    return root, manifest, digest


def package_project(
    source: Path,
    destination: Path,
    *,
    source_name: str,
    source_revision: str,
    source_license: str,
    source_sha256: str,
    asset_hashes: dict[str, str] | None = None,
) -> ProjectManifest:
    """Create a deterministic, credential-free project package for S3 staging."""

    if not source.is_dir() or not (source / "antioch.yaml").is_file():
        raise AntiochProjectError(
            "project directory must contain antioch.yaml at its root"
        )
    _assert_no_secrets(source)
    members = [
        path
        for path in sorted(source.rglob("*"))
        if not any(
            part in {".git", ".venv", "__pycache__"}
            for part in path.relative_to(source).parts
        )
        and path.suffix != ".pyc"
    ]
    if any(
        path.is_symlink() or not (path.is_file() or path.is_dir()) for path in members
    ):
        raise AntiochProjectError(
            "project directory contains a link or unsupported filesystem object"
        )
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "project.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as bundle,
    ):
        for path in members:
            relative = path.relative_to(source).as_posix()
            info = bundle.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if info.isfile():
                info.mode = 0o644
                with path.open("rb") as stream:
                    bundle.addfile(info, stream)
            else:
                info.mode = 0o755
                bundle.addfile(info)
    manifest = ProjectManifest(
        archive={
            "name": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        source_name=source_name,
        source_revision=source_revision,
        source_license=source_license,
        source_sha256=source_sha256,
        asset_hashes=asset_hashes or {},
    )
    (destination / "project-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
