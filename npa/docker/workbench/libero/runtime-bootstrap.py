#!/usr/bin/env python3
"""Materialize the pinned LIBERO runtime into an operator-owned cache.

The public image contains this fetcher and immutable manifests, not LIBERO,
PyTorch/CUDA wheels, demonstrations, task assets, models, or populated caches.
Download success is never treated as permission: ``ensure`` refuses before the
first cache mutation unless a manager-issued decision is present and hash-bound.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

SCHEMA = "npa.libero.runtime-manifest.v1"
DECISION_SCHEMA = "npa.libero.runtime-use-decision.v1"
COMPLETE_SCHEMA = "npa.libero.runtime-cache.v1"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "cc3556e776aeaca5c73e27be3bdcfea24b97848493eb1eae2883e662fd741c76"
)
DEFAULT_MANIFEST = Path("/opt/npa/libero/runtime-manifest.json")
DEFAULT_CACHE = Path("/workspace/.cache/npa/libero")
ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "files.pythonhosted.org",
        "download-r2.pytorch.org",
        "huggingface.co",
    }
)
EXPECTED_DECISION_BOUNDARIES = frozenset(
    {"source", "runtime_packages", "demonstration", "task_inputs", "language_model"}
)


class BootstrapRefusal(RuntimeError):
    """A fail-closed identity, permission, or boundary refusal."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal(f"cannot read required JSON {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapRefusal(f"required JSON is not an object: {path}")
    return value


def _is_private_regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _validate_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise BootstrapRefusal("runtime manifest must be a regular image file")
    manifest_sha256 = _sha256(path)
    if manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise BootstrapRefusal("runtime manifest bytes differ from the image contract")
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("solution") != "libero":
        raise BootstrapRefusal("runtime manifest identity is invalid")
    source = manifest.get("source") or {}
    if (
        source.get("repository")
        != "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
        or not _is_hex(source.get("revision"), 40)
        or not _is_hex(source.get("tree"), 40)
        or source.get("license") != "MIT"
        or not _is_hex(source.get("license_sha256"), 64)
    ):
        raise BootstrapRefusal("pinned LIBERO source contract is invalid")
    sparse_paths = source.get("sparse_paths")
    if (
        not isinstance(sparse_paths, list)
        or len(sparse_paths) != len(set(sparse_paths))
        or not all(isinstance(item, str) and item for item in sparse_paths)
        or "libero/libero/assets" in sparse_paths
        or source.get("forbidden_paths") != ["libero/libero/assets", ".git"]
    ):
        raise BootstrapRefusal("LIBERO sparse-source boundary is invalid")
    demonstration = manifest.get("demonstration") or {}
    if (
        demonstration.get("repository") != "yifengzhu-hf/LIBERO-datasets"
        or not _is_hex(demonstration.get("revision"), 40)
        or demonstration.get("license") != "CC-BY-4.0"
        or not str(demonstration.get("attribution") or "").strip()
        or not str(demonstration.get("filename") or "").endswith(".hdf5")
        or not _is_hex(demonstration.get("sha256"), 64)
        or not isinstance(demonstration.get("size_bytes"), int)
        or demonstration["size_bytes"] <= 0
    ):
        raise BootstrapRefusal("official demonstration contract is invalid")
    _validate_download_url(str(demonstration.get("url") or ""))
    task = manifest.get("task") or {}
    if (
        task.get("suite") != "libero_spatial"
        or not str(task.get("name") or "").strip()
        or not str(task.get("description") or "").strip()
    ):
        raise BootstrapRefusal("LIBERO task identity is invalid")
    for key in ("bddl", "initial_states", "embedding_source"):
        record = task.get(key) or {}
        if not str(record.get("path") or "").strip() or not _is_hex(
            record.get("sha256"), 64
        ):
            raise BootstrapRefusal(f"LIBERO task identity is invalid: {key}")
    model = manifest.get("language_model") or {}
    model_files = model.get("files")
    if (
        model.get("repository") != "google-bert/bert-base-cased"
        or not _is_hex(model.get("revision"), 40)
        or model.get("license") != "Apache-2.0"
        or not isinstance(model_files, list)
        or not model_files
    ):
        raise BootstrapRefusal("task language-model contract is invalid")
    model_names: set[str] = set()
    for item in model_files:
        if not isinstance(item, dict):
            raise BootstrapRefusal("task language-model file is not an object")
        filename = str(item.get("filename") or "")
        if (
            not filename
            or filename in model_names
            or "/" in filename
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
            or not _is_hex(item.get("sha256"), 64)
        ):
            raise BootstrapRefusal("task language-model file identity is invalid")
        model_names.add(filename)
    artifacts = manifest.get("runtime_artifacts")
    if (
        manifest.get("runtime_artifact_count") != 135
        or not isinstance(artifacts, list)
        or len(artifacts) != manifest["runtime_artifact_count"]
    ):
        raise BootstrapRefusal(
            "runtime artifact inventory must contain exactly 135 items"
        )
    names: set[str] = set()
    filenames: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise BootstrapRefusal("runtime artifact entry is not an object")
        name = str(item.get("name") or "")
        filename = str(item.get("filename") or "")
        if name in names or filename in filenames or not name or not filename:
            raise BootstrapRefusal(
                "runtime artifact names and filenames must be unique"
            )
        names.add(name)
        filenames.add(filename)
        _validate_download_url(str(item.get("url") or ""))
        if not _is_hex(item.get("sha256"), 64):
            raise BootstrapRefusal(f"invalid runtime artifact hash for {name}")
    return manifest, manifest_sha256


def _is_hex(value: object, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS
        or parsed.fragment
    ):
        raise BootstrapRefusal(
            "runtime download URL is outside the immutable allowlist"
        )


def _validate_redirect_url(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or not (
            hostname in ALLOWED_DOWNLOAD_HOSTS
            or hostname.endswith(".hf.co")
            or hostname.endswith(".huggingface.co")
        )
        or parsed.fragment
    ):
        raise BootstrapRefusal("runtime redirect left the download allowlist")
    return parsed


def _open_https_download(
    url: str,
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    current_url = url
    for _ in range(6):
        parsed = _validate_redirect_url(current_url)
        connection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=60
        )
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET", target, headers={"User-Agent": "npa-libero-runtime/1"}
        )
        response = connection.getresponse()
        if response.status not in {301, 302, 303, 307, 308}:
            if response.status != 200:
                status = response.status
                response.close()
                connection.close()
                raise BootstrapRefusal(
                    f"runtime download returned unexpected HTTP status {status}"
                )
            return connection, response
        location = response.getheader("Location")
        response.close()
        connection.close()
        if not location:
            raise BootstrapRefusal("runtime redirect omitted its target")
        current_url = urllib.parse.urljoin(current_url, location)
    raise BootstrapRefusal("runtime download exceeded the redirect limit")


def _validate_cache_root(cache_root: Path, output_dir: Path | None) -> Path:
    if cache_root.is_symlink():
        raise BootstrapRefusal("runtime cache root may not be a symlink")
    resolved = cache_root.resolve(strict=False)
    if not resolved.is_absolute() or resolved == Path("/"):
        raise BootstrapRefusal("runtime cache root must be a narrow absolute path")
    if output_dir is not None:
        output = output_dir.resolve(strict=False)
        if (
            output == resolved
            or output in resolved.parents
            or resolved in output.parents
        ):
            raise BootstrapRefusal("runtime cache and output boundaries overlap")
    return resolved


def _validate_decision(
    path: Path, expected_sha256: str, manifest: dict[str, Any], manifest_sha256: str
) -> tuple[dict[str, Any], str]:
    if not _is_hex(expected_sha256, 64):
        raise BootstrapRefusal("expected decision SHA-256 is required")
    if not _is_private_regular_file(path):
        raise BootstrapRefusal(
            "runtime-use decision must be an owner-only regular file"
        )
    observed = _sha256(path)
    if observed != expected_sha256:
        raise BootstrapRefusal("runtime-use decision hash does not match")
    decision = _load_json(path)
    source = manifest["source"]
    boundaries = decision.get("authorized_boundaries")
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("solution") != "libero"
        or decision.get("decision") != "authorized"
        or decision.get("runtime_fetch_authorized") is not True
        or decision.get("runtime_manifest_sha256") != manifest_sha256
        or decision.get("source_revision") != source["revision"]
        or not isinstance(boundaries, list)
        or frozenset(boundaries) != EXPECTED_DECISION_BOUNDARIES
        or len(boundaries) != len(EXPECTED_DECISION_BOUNDARIES)
        or not str(decision.get("manager_receipt_sha256") or "").startswith("sha256:")
        or not _is_hex(str(decision.get("manager_receipt_sha256"))[7:], 64)
    ):
        raise BootstrapRefusal("runtime-use decision does not bind the exact contract")
    forbidden = {key for key in decision if key.startswith("ACCEPT_")}
    if forbidden:
        raise BootstrapRefusal("invented acceptance proxy is forbidden")
    return decision, observed


def _download_verified(
    destination: Path, *, url: str, sha256: str, size: int | None
) -> None:
    _validate_download_url(url)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    observed_size = 0
    connection: http.client.HTTPSConnection | None = None
    response: http.client.HTTPResponse | None = None
    try:
        connection, response = _open_https_download(url)
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
    if digest.hexdigest() != sha256 or (size is not None and observed_size != size):
        temporary.unlink(missing_ok=True)
        raise BootstrapRefusal(
            "runtime download bytes do not match their immutable identity"
        )
    temporary.replace(destination)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _fetch_source(root: Path, source: dict[str, Any]) -> None:
    destination = root / "source"
    destination.mkdir(mode=0o700)
    _run(["git", "init", "--quiet"], cwd=destination)
    _run(["git", "remote", "add", "origin", source["repository"]], cwd=destination)
    _run(["git", "config", "remote.origin.promisor", "true"], cwd=destination)
    _run(
        ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
        cwd=destination,
    )
    _run(
        [
            "git",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            source["revision"],
        ],
        cwd=destination,
    )
    fetched = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD^{commit}"], cwd=destination, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD^{tree}"], cwd=destination, text=True
    ).strip()
    if fetched != source["revision"] or tree != source["tree"]:
        raise BootstrapRefusal(
            "fetched LIBERO source identity differs from the manifest"
        )
    _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=destination)
    _run(
        ["git", "sparse-checkout", "set", "--no-cone", "--", *source["sparse_paths"]],
        cwd=destination,
    )
    _run(["git", "checkout", "--quiet", "--detach", fetched], cwd=destination)
    if _sha256(destination / source["license_file"]) != source["license_sha256"]:
        raise BootstrapRefusal(
            "LIBERO source license does not match the reviewed MIT file"
        )
    if (destination / "libero" / "libero" / "assets").exists():
        raise BootstrapRefusal("forbidden LIBERO render payload survived sparse fetch")
    shutil.rmtree(destination / ".git")


def _install_runtime(root: Path, artifacts: list[dict[str, Any]]) -> None:
    wheelhouse = root / "downloads"
    wheelhouse.mkdir(mode=0o700)
    for item in artifacts:
        _download_verified(
            wheelhouse / item["filename"],
            url=item["url"],
            sha256=item["sha256"],
            size=None,
        )
    venv = root / "venv"
    _run([sys.executable, "-m", "venv", str(venv)])
    bootstrap_names = {"pip", "setuptools", "wheel"}
    bootstrap = [
        str(wheelhouse / item["filename"])
        for item in artifacts
        if item["name"] in bootstrap_names
    ]
    remaining = [
        str(wheelhouse / item["filename"])
        for item in artifacts
        if item["name"] not in bootstrap_names
    ]
    pip = str(venv / "bin" / "python")
    _run([pip, "-m", "pip", "install", "--no-index", "--no-deps", *bootstrap])
    _run(
        [
            pip,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            *remaining,
        ]
    )
    site_packages = subprocess.check_output(
        [pip, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()
    Path(site_packages, "npa-libero-source.pth").write_text(
        str(root / "source") + "\n", encoding="utf-8"
    )
    shutil.rmtree(wheelhouse)


def _fetch_inputs(root: Path, manifest: dict[str, Any]) -> None:
    demonstration = manifest["demonstration"]
    _download_verified(
        root / "data" / demonstration["filename"],
        url=demonstration["url"],
        sha256=demonstration["sha256"],
        size=int(demonstration["size_bytes"]),
    )
    model = manifest["language_model"]
    model_root = root / "models" / f"bert-base-cased-{model['revision']}"
    for item in model["files"]:
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{item['filename']}?download=true"
        _download_verified(
            model_root / item["filename"],
            url=url,
            sha256=item["sha256"],
            size=int(item["size_bytes"]),
        )
    model_record = {
        "repository": model["repository"],
        "revision": model["revision"],
        "files": {item["filename"]: item["sha256"] for item in model["files"]},
    }
    record_path = model_root / "npa-language-model.json"
    record_path.write_text(
        json.dumps(model_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(record_path, 0o600)


def _validate_task_inputs(root: Path, manifest: dict[str, Any]) -> None:
    source = root / "source"
    task = manifest["task"]
    for key in ("bddl", "initial_states", "embedding_source"):
        record = task[key]
        path = source / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise BootstrapRefusal(
                f"genuine upstream task input failed identity: {key}"
            )


def _complete_record(
    root: Path, manifest: dict[str, Any], manifest_sha256: str, decision_sha256: str
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    record = {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "decision_sha256": decision_sha256,
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_license_sha256": source["license_sha256"],
        "runtime_artifact_count": len(manifest["runtime_artifacts"]),
        "demonstration_sha256": demonstration["sha256"],
        "demonstration_size_bytes": demonstration["size_bytes"],
        "task_bddl_sha256": manifest["task"]["bddl"]["sha256"],
        "task_initial_states_sha256": manifest["task"]["initial_states"]["sha256"],
        "language_model_revision": manifest["language_model"]["revision"],
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
    }
    complete = root / ".complete.json"
    complete.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(complete, 0o600)
    return record


def _validate_complete(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    decision_sha256: str,
) -> dict[str, Any]:
    record = _load_json(root / ".complete.json")
    expected = _complete_record_values(manifest, manifest_sha256, decision_sha256)
    if record != expected:
        raise BootstrapRefusal(
            "existing runtime cache completion record does not match"
        )
    if (root / "source" / "libero" / "libero" / "assets").exists() or (
        root / "source" / ".git"
    ).exists():
        raise BootstrapRefusal(
            "existing runtime cache contains forbidden source payload"
        )
    if not (root / "venv" / "bin" / "python").is_file():
        raise BootstrapRefusal("existing runtime cache is incomplete")
    demonstration = manifest["demonstration"]
    data = root / "data" / demonstration["filename"]
    if (
        not data.is_file()
        or data.stat().st_size != demonstration["size_bytes"]
        or _sha256(data) != demonstration["sha256"]
    ):
        raise BootstrapRefusal("existing demonstration cache does not match")
    model = manifest["language_model"]
    model_root = root / "models" / f"bert-base-cased-{model['revision']}"
    for item in model["files"]:
        path = model_root / item["filename"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise BootstrapRefusal("existing task language-model cache does not match")
    _validate_task_inputs(root, manifest)
    return record


def _complete_record_values(
    manifest: dict[str, Any], manifest_sha256: str, decision_sha256: str
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    return {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "decision_sha256": decision_sha256,
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_license_sha256": source["license_sha256"],
        "runtime_artifact_count": len(manifest["runtime_artifacts"]),
        "demonstration_sha256": demonstration["sha256"],
        "demonstration_size_bytes": demonstration["size_bytes"],
        "task_bddl_sha256": manifest["task"]["bddl"]["sha256"],
        "task_initial_states_sha256": manifest["task"]["initial_states"]["sha256"],
        "language_model_revision": manifest["language_model"]["revision"],
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
    }


def ensure(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest, manifest_sha256 = _validate_manifest(manifest_path)
    decision_path = Path(args.decision)
    _, decision_sha256 = _validate_decision(
        decision_path, args.decision_sha256, manifest, manifest_sha256
    )
    output = Path(args.output_dir) if args.output_dir else None
    cache_root = _validate_cache_root(Path(args.cache_root), output)
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(cache_root, 0o700)
    lock_path = cache_root / ".bootstrap.lock"
    with lock_path.open("a+b") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = cache_root / manifest_sha256
        current = cache_root / "current"
        warm_reuse = final.is_dir()
        if warm_reuse:
            record = _validate_complete(
                final, manifest, manifest_sha256, decision_sha256
            )
        else:
            partial = Path(
                tempfile.mkdtemp(prefix=f".{manifest_sha256}.partial-", dir=cache_root)
            )
            os.chmod(partial, 0o700)
            try:
                _fetch_source(partial, manifest["source"])
                _validate_task_inputs(partial, manifest)
                _install_runtime(partial, manifest["runtime_artifacts"])
                _fetch_inputs(partial, manifest)
                record = _complete_record(
                    partial, manifest, manifest_sha256, decision_sha256
                )
                partial.replace(final)
            except Exception:
                shutil.rmtree(partial, ignore_errors=True)
                raise
        temporary_link = cache_root / f".current-{os.getpid()}"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(final.name)
        temporary_link.replace(current)
    return {**record, "cache_path": str(final), "warm_reuse": warm_reuse}


def status(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha256 = _validate_manifest(Path(args.manifest))
    cache_root = _validate_cache_root(Path(args.cache_root), None)
    final = cache_root / manifest_sha256
    complete = final / ".complete.json"
    return {
        "schema": "npa.libero.runtime-status.v1",
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "materialized": complete.is_file(),
        "source_revision": manifest["source"]["revision"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("ensure", "status"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--decision", default="")
    parser.add_argument("--decision-sha256", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = ensure(args) if args.command == "ensure" else status(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "npa.libero.runtime-bootstrap-result.v1",
                    "status": "refused",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({**payload, "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
