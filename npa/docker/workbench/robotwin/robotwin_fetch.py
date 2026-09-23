"""Customer-side installation and execution for the locked RoboTwin capability.

Imported only after the caller has checked customer authorization. Nothing in
this module downloads, installs, imports a simulator, or writes state on import.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)
import zipfile


class RuntimeFailure(RuntimeError):
    """A fixed category that never incorporates provider errors or credentials."""


def fail(category: str) -> None:
    raise RuntimeFailure(f"ROBOTWIN_RUNTIME_FAILED:{category}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_artifacts(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = lock.get("runtime_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        fail("artifact-lock-empty")
    names = set()
    for item in artifacts:
        if not isinstance(item, dict):
            fail("artifact-lock-invalid")
        filename = item.get("filename", "")
        if not isinstance(filename, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._+%~-]*", filename
        ):
            fail("artifact-filename-invalid")
        if filename in names:
            fail("artifact-filename-duplicate")
        names.add(filename)
        parsed = urlsplit(str(item.get("url", "")))
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            fail("artifact-url-invalid")
        if parsed.hostname not in {
            "files.pythonhosted.org",
            "download.pytorch.org",
            "developer.download.nvidia.com",
            "snapshot.ubuntu.com",
        }:
            fail("artifact-provider-invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            fail("artifact-digest-invalid")
        if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
            fail("artifact-size-invalid")
        if item.get("kind") not in {"deb", "bdist_wheel", "sdist", "cuda-archive"}:
            fail("artifact-kind-invalid")
    kinds = {item["kind"] for item in artifacts}
    if not {"deb", "bdist_wheel", "cuda-archive"} <= kinds:
        fail("artifact-closure-incomplete")
    return artifacts


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            fail("artifact-redirect-invalid")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(item: Mapping[str, Any], destination: Path) -> None:
    """Verify all bytes before making an artifact available to an installer."""
    temporary = destination.with_name(destination.name + ".partial")
    opener = build_opener(ProxyHandler({}), HTTPSHandler(), HTTPSRedirects())
    size = 0
    value = hashlib.sha256()
    try:
        with (
            opener.open(Request(item["url"])) as response,
            temporary.open("xb") as output,
        ):
            if urlsplit(response.geturl()).scheme != "https":
                fail("artifact-redirect-invalid")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > item["size_bytes"]:
                    fail("artifact-size-mismatch")
                value.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != item["size_bytes"] or value.hexdigest() != item["sha256"]:
            fail("artifact-integrity-mismatch")
        temporary.replace(destination)
    except RuntimeFailure:
        temporary.unlink(missing_ok=True)
        raise
    except Exception:
        temporary.unlink(missing_ok=True)
        fail("artifact-fetch-failed")


def _member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        fail("archive-member-invalid")
    return path


def extract_tar(archive: Path, destination: Path) -> None:
    """Extract CUDA's single-root archives without traversal or special files."""
    with tarfile.open(archive) as source:
        members = source.getmembers()
        roots = {
            _member_path(member.name).parts[0] for member in members if member.name
        }
        if len(roots) != 1:
            fail("archive-root-invalid")
        for member in members:
            path = _member_path(member.name)
            relative = Path(*path.parts[1:])
            if not relative.parts:
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    fail("archive-member-collision")
                reader = source.extractfile(member)
                if reader is None:
                    fail("archive-member-unreadable")
                with reader, target.open("wb") as output:
                    shutil.copyfileobj(reader, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            elif member.issym():
                # CUDA library links are sibling links, not arbitrary paths.
                link = _member_path(member.linkname)
                if len(link.parts) != 1:
                    fail("archive-link-invalid")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
            else:
                fail("archive-member-type-invalid")


def extract_assets(archive: Path, destination: Path, expected_root: str) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        paths = {_member_path(member.filename) for member in members}
        sidecars = set()
        for member in members:
            path = _member_path(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode) or (
                stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                fail("asset-archive-member-type-invalid")
            if path.parts and path.parts[0] == "__MACOSX":
                if (
                    len(path.parts) < 2
                    or not path.name.startswith("._")
                    or path.name[2:] in {"", ".", ".."}
                    or stat.S_IFMT(mode) != stat.S_IFREG
                    or member.is_dir()
                    or path in sidecars
                ):
                    fail("asset-metadata-invalid")
                companion = path.relative_to("__MACOSX").with_name(path.name[2:])
                if companion.parts[0] != expected_root or companion not in paths:
                    fail("asset-metadata-companion-invalid")
                # AppleDouble v2 magic/version (RFC 1740); read verifies ZIP CRC.
                if not source.read(member).startswith(
                    b"\x00\x05\x16\x07\x00\x02\x00\x00"
                ):
                    fail("asset-metadata-invalid")
                sidecars.add(path)
                continue
            if not path.parts or path.parts[0] != expected_root:
                fail("asset-archive-root-invalid")
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as reader, target.open("xb") as output:
                    shutil.copyfileobj(reader, output)


def command(
    argv: list[str], *, env: Mapping[str, str], cwd: Path | None = None
) -> None:
    # Keep native compiler/simulator diagnostics only in the customer's private
    # runtime tree. These files never enter the output upload or public logs.
    logs = Path(env["HOME"]).parent / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    sequence = len(list(logs.glob("command-*.log"))) + 1
    log = logs / f"command-{sequence:04d}.log"
    descriptor = os.open(
        log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as output:
        output.write((json.dumps(argv) + "\n").encode())
        output.flush()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=dict(env),
                check=False,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        except OSError:
            fail("command-unavailable")
    if completed.returncode:
        fail(f"command-{sequence:04d}-failed-private-log-retained")


def runtime_environment(root: Path, environ: Mapping[str, str]) -> dict[str, str]:
    env = {
        "HOME": str(root / "home"),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "CUDA_HOME": str(root / "cuda"),
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.7.8",
        "NPA_ROBOTWIN_BOOTSTRAP_DIGEST": environ.get("BYOF_IMAGE", "").rsplit("@", 1)[
            -1
        ],
        "DEBIAN_FRONTEND": "noninteractive",
        "XDG_CACHE_HOME": str(root / "cache"),
        "CUDA_CACHE_PATH": str(root / "cache/cuda"),
        "TORCH_HOME": str(root / "cache/torch"),
        "WARP_CACHE_PATH": str(root / "cache/warp"),
        "MPLCONFIGDIR": str(root / "cache/matplotlib"),
        "TMPDIR": str(root / "tmp"),
    }
    for key in (
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_ICD_FILENAMES",
        "LD_LIBRARY_PATH",
    ):
        if environ.get(key):
            env[key] = environ[key]
    return env


def checkout(
    item: Mapping[str, Any], destination: Path, env: Mapping[str, str]
) -> None:
    revision = item.get("revision") or item["version"]
    command(["git", "init", str(destination)], env=env)
    command(
        ["git", "-C", str(destination), "remote", "add", "origin", item["origin"]],
        env=env,
    )
    command(
        [
            "git",
            "-C",
            str(destination),
            "-c",
            "protocol.file.allow=never",
            "fetch",
            "--depth=1",
            "origin",
            revision,
        ],
        env=env,
    )
    command(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], env=env
    )
    for expression, expected in [
        ("HEAD", revision),
        ("HEAD^{tree}", item["tree_sha1"]),
    ]:
        result = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", expression],
            env=dict(env),
            check=True,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip() != expected:
            fail("source-identity-mismatch")
    if digest(destination / "LICENSE") != item["license_sha256"]:
        fail("source-license-mismatch")


def stage_apt_archives(
    artifacts: list[dict[str, Any]], downloads: Path, cache: Path
) -> list[str]:
    """Expose verified debs under the names APT's offline archive cache expects."""
    cache.mkdir(mode=0o700)
    debs = []
    for item in artifacts:
        if item["kind"] != "deb":
            continue
        architecture = item["filename"].rsplit("_", 1)[-1]
        version = str(item.get("version", "")).replace(":", "%3a")
        filename = f"{item.get('name', '')}_{version}_{architecture}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+%~-]*\.deb", filename):
            fail("apt-cache-filename-invalid")
        source = downloads / item["filename"]
        with source.open("rb") as reader, (cache / filename).open("xb") as output:
            shutil.copyfileobj(reader, output)
        debs.append(str(source))
    return debs


def execute(
    lock: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    runtime_parent: Path = Path("/workspace/robotwin-runtime"),
) -> int:
    """Install exact customer-fetched bytes, then collect and publish real output."""
    artifacts = validate_artifacts(lock)
    run_id = environ.get("NPA_BYOF_RUN_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        fail("run-id-invalid")
    output = Path("/workspace/byof-runs") / run_id
    if environ.get("NPA_SMOKE_OUTPUT_DIR") != str(output) or output.exists():
        fail("output-destination-invalid-or-exists")
    runtime_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (
        runtime_parent.is_symlink()
        or runtime_parent.stat().st_mode & 0o077
        or runtime_parent.stat().st_uid != os.geteuid()
    ):
        fail("runtime-directory-not-owner-only")
    root = Path(tempfile.mkdtemp(prefix=run_id + "-", dir=runtime_parent))
    for name in ("home", "tmp", "cache", "downloads", "cuda"):
        (root / name).mkdir(mode=0o700)
    env = runtime_environment(root, environ)
    print("robotwin: fetching verified runtime artifacts", flush=True)
    for item in artifacts:
        fetch(item, root / "downloads" / item["filename"])
    print("robotwin: installing runtime system dependencies", flush=True)
    apt_cache = root / "apt-cache"
    debs = stage_apt_archives(artifacts, root / "downloads", apt_cache)
    command(
        [
            "sudo",
            "-n",
            "env",
            "DEBIAN_FRONTEND=noninteractive",
            "apt-get",
            "install",
            "-y",
            "--no-download",
            "-o",
            f"Dir::Cache::archives={apt_cache}",
            "--no-install-recommends",
            "--allow-downgrades",
            *debs,
        ],
        env=env,
    )
    for item in artifacts:
        if item["kind"] == "cuda-archive":
            extract_tar(root / "downloads" / item["filename"], root / "cuda")
    command(["/usr/bin/python3", "-m", "venv", str(root / "venv")], env=env)
    python = str(root / "venv/bin/python")
    env["PATH"] = f"{root}/venv/bin:{root}/cuda/bin:" + env["PATH"]
    wheels = [
        str(root / "downloads" / item["filename"])
        for item in artifacts
        if item["kind"] == "bdist_wheel"
    ]
    command(
        [python, "-m", "pip", "install", "--no-deps", "--no-index", *wheels], env=env
    )
    for item in artifacts:
        if item["kind"] == "sdist":
            command(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--no-build-isolation",
                    str(root / "downloads" / item["filename"]),
                ],
                env=env,
            )
    source = root / "RoboTwin"
    for item in lock["sources"]:
        checkout(
            item, source if item["name"] == "RoboTwin" else source / "envs/curobo", env
        )
    print("robotwin: compiling pinned CuRobo for sm_120", flush=True)
    command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-build-isolation",
            str(source / "envs/curobo"),
        ],
        env=env,
    )
    command([python, "-m", "pip", "check"], env=env)
    for asset in lock["assets"]:
        archive = root / "downloads" / asset["name"]
        fetch(
            {
                **asset,
                "url": f"https://huggingface.co/datasets/{asset['repository']}/resolve/{asset['revision']}/{asset['name']}",
            },
            archive,
        )
        extract_assets(archive, source / "assets", asset["name"].removesuffix(".zip"))
    ready = {
        "schema_version": "npa.robotwin.runtime-ready.v1",
        "sources": [s.get("revision", s["version"]) for s in lock["sources"]],
        "artifacts": [
            {"sha256": i["sha256"], "size_bytes": i["size_bytes"]} for i in artifacts
        ],
    }
    marker = root / "ready.json.partial"
    marker.write_text(json.dumps(ready, sort_keys=True) + "\n")
    marker.replace(root / "ready.json")
    output.mkdir(mode=0o700, parents=True)
    print("robotwin: searching and replaying an upstream successful seed", flush=True)
    collector = Path(__file__).with_name("robotwin_collect.py")
    command(
        [
            python,
            str(collector),
            "collect",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        env=env,
        cwd=source,
    )
    upload_env = {
        **env,
        **{
            key: environ[key]
            for key in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_ENDPOINT_URL",
                "NEBIUS_S3_ENDPOINT",
                "NPA_S3_BUCKET",
                "S3_OUTPUT_PREFIX",
            )
            if key in environ
        },
    }
    command([python, str(collector), "upload", "--output", str(output)], env=upload_env)
    print(
        "robotwin: validated native HDF5, decoded video and frames uploaded", flush=True
    )
    return 0
