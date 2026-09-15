#!/usr/bin/env python3
"""Explicit Linux/amd64 scanner bootstrap; never called implicitly by pytest.

All downloads, build caches, logs, notices and binaries stay under an external,
owner-only analysis directory. The trusted checkout supplies source and policy.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

GO_VERSION = "1.27.1"
GO_ARCHIVE_SHA256 = "63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445"
GO_ARCHIVE_URL = f"https://go.dev/dl/go{GO_VERSION}.linux-amd64.tar.gz"
GITLEAKS_VERSION = "v8.28.0"
SOURCE_NAMES = ("main.go", "main_test.go", "go.mod", "go.sum", "build.py",
                "LICENSE-GO", "LICENSE-GITLEAKS", "README.md")
TEST_SOURCE_NAME = "npa/tests/docker/test_image_byte_go_build.py"
GO_NAMES = ("main.go", "main_test.go", "go.mod", "go.sum")


class BuildError(Exception):
    """A fixed diagnostic code; raw subprocess output stays in protected logs."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_uid,
            info.st_gid, info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)


def no_symlinks(path: Path) -> Path:
    """Reject aliasing before resolving any caller supplied root."""
    if ".." in Path(path).parts:
        raise BuildError("path_parent_component")
    path = Path(os.path.abspath(path))
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise BuildError("path_symlink")
    return path


def directory_fd(path: Path, *, create: bool = False) -> int:
    """Resolve every component beneath a held descriptor without following links."""
    path = no_symlinks(path)
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=current)
            os.close(current)
            current = child
        result = current
        current = -1
        return result
    finally:
        if current >= 0:
            os.close(current)


def read_regular(path: Path) -> bytes:
    path = no_symlinks(path)
    parent = directory_fd(path.parent)
    fd = -1
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BuildError("input_not_regular")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read()
        if _signature(before) != _signature(os.fstat(fd)) or len(data) != before.st_size:
            raise BuildError("input_changed")
        if _signature(before) != _signature(os.stat(path.name, dir_fd=parent, follow_symlinks=False)):
            raise BuildError("input_replaced")
        # A rename of a parent must not silently change the named binding even
        # though its held descriptor kept this read on the original inode.
        current_parent = directory_fd(path.parent)
        try:
            left, right = os.fstat(parent), os.fstat(current_parent)
            if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                raise BuildError("input_parent_replaced")
        finally:
            os.close(current_parent)
        return data
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent)


def private_dir(path: Path) -> Path:
    path = no_symlinks(path)
    fd = directory_fd(path, create=True)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise BuildError("directory_not_private")
    finally:
        os.close(fd)
    return path


def write_new(path: Path, data: bytes, mode: int = 0o600) -> None:
    path = no_symlinks(path)
    parent = directory_fd(path.parent)
    fd = -1
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     mode, dir_fd=parent)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
        os.fsync(parent)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def json_stream(data: bytes) -> list[dict]:
    decoder = json.JSONDecoder()
    text = data.decode("utf-8")
    values = []
    while text.strip():
        value, end = decoder.raw_decode(text.lstrip())
        if not isinstance(value, dict):
            raise BuildError("module_receipt_shape")
        values.append(value)
        text = text.lstrip()[end:]
    return values


def extract_toolchain(archive: Path, destination: Path) -> Path:
    """Only the exact pinned archive may reach this strict regular-file extractor."""
    raw = read_regular(archive)
    if digest(raw) != GO_ARCHIVE_SHA256:
        raise BuildError("toolchain_digest")
    # The verified bytes, rather than a reopened pathname, feed extraction.
    import io

    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as source:
        for member in source:
            name = member.name.rstrip("/")
            parts = PurePosixPath(name).parts
            if (not parts or parts[0] != "go" or name.startswith("/") or "\\" in name
                    or any(part in {".", "..", ""} for part in name.split("/"))
                    or name in seen or not (member.isdir() or member.isfile())):
                raise BuildError("toolchain_member")
            seen.add(name)
            target = destination.joinpath(*parts)
            if member.isdir():
                private_dir(target)
            else:
                private_dir(target.parent)
                stream = source.extractfile(member)
                if stream is None:
                    raise BuildError("toolchain_member_read")
                with stream:
                    content = stream.read()
                if len(content) != member.size:
                    raise BuildError("toolchain_member_size")
                write_new(target, content, 0o700 if member.mode & 0o111 else 0o600)
    go = destination / "go/bin/go"
    if not go.is_file() or not os.access(go, os.X_OK):
        raise BuildError("toolchain_executable")
    return go


def isolated_environment(work: Path, config: Path) -> dict[str, str]:
    # In particular, do not inherit credentials, GOFLAGS, HOME overrides, build
    # wrappers, alternate proxies, GOPRIVATE, or user Go configuration.
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "GOTOOLCHAIN": "local", "GOENV": "off", "GOWORK": "off",
           "GOPROXY": "https://proxy.golang.org", "GOSUMDB": "sum.golang.org",
           "GOPRIVATE": "", "GONOPROXY": "", "GONOSUMDB": "",
           "GOFLAGS": "-mod=readonly", "CGO_ENABLED": "0", "GOOS": "linux",
           "GOARCH": "amd64", "NPA_IMAGE_BYTE_TEST_CONFIG": str(config)}
    for key, directory in (("GOPATH", "gopath"), ("GOMODCACHE", "modules"),
                           ("GOCACHE", "cache"), ("GOTMPDIR", "tmp"),
                           ("TMPDIR", "tmp")):
        env[key] = str(private_dir(work / directory))
    return env


class BuildCancelled(BaseException):
    """Raised by the CLI signal boundary after the first cancellation request."""


_CANCELLATION = ContextVar("image_byte_bootstrap_cancellation", default=None)


@contextmanager
def cancellation_scope():
    state = {"interrupted": False, "creating": False, "pending": False}
    token = _CANCELLATION.set(state)

    def cancel(signum, frame):
        if not state["interrupted"]:
            state["interrupted"] = True
            if state["creating"]:
                state["pending"] = True
            else:
                raise BuildCancelled()

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        for signum in previous:
            signal.signal(signum, cancel)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _CANCELLATION.reset(token)


def _owned_group_running(group: int) -> bool:
    """Inspect only membership/state; never serialize process names or arguments."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    observed = False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == group and int(fields[3]) == group:
            observed = True
            if fields[0] != "Z":
                return True
    if not observed:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        raise BuildError("owned_group_inventory_unresolved")
    return False


def _stop_owned(process: subprocess.Popen) -> tuple[bytes, bytes]:
    """Only signal the session created for this command, then reap its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        output = None
    # The leader may exit while a child ignores TERM and has closed its pipes.
    # Finish the entire owned group even in that case.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if output is None:
        output = process.communicate()
    while _owned_group_running(process.pid):
        time.sleep(0.001)
    return output


def run_step(argv: list[str], role: str, work: Path, env: dict, logs: Path,
             input_data: bytes | None = None) -> bytes:
    process = None
    state = _CANCELLATION.get()
    try:
        # Defer a signal until Popen returns its owned process handle. Blocking
        # signals in the OS here would also leave them blocked in the child.
        if state is not None:
            state["creating"] = True
        try:
            process = subprocess.Popen(
                argv, cwd=work, env=env,
                stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        finally:
            if state is not None:
                state["creating"] = False
        if state is not None and state["pending"]:
            raise BuildCancelled()
        stdout, stderr = process.communicate(input_data)
    except BaseException:
        if process is None:
            raise
        stdout, stderr = _stop_owned(process)
        write_new(logs / f"{role}.stdout.log", stdout)
        write_new(logs / f"{role}.stderr.log", stderr)
        write_new(logs / f"{role}.status.json", json_bytes({"exit_code": process.returncode,
                                                           "interrupted": True, "leader_joined": True}))
        raise
    # A successful leader cannot leave background commands running in its
    # session. Trusted build tools normally wait for all of their children.
    descendant = False
    try:
        os.killpg(process.pid, 0)
        descendant = True
    except ProcessLookupError:
        pass
    if descendant:
        _stop_owned(process)
    write_new(logs / f"{role}.stdout.log", stdout)
    write_new(logs / f"{role}.stderr.log", stderr)
    write_new(logs / f"{role}.status.json", json_bytes({"exit_code": process.returncode,
                                                       "leader_joined": True}))
    if descendant:
        raise BuildError(f"step_{role}_unjoined_descendant")
    if process.returncode:
        raise BuildError(f"step_{role}_failed")
    return stdout


def verify_native_tests(raw: bytes, source: bytes) -> dict:
    """A successful go command must also prove complete, unskipped collection."""
    declared = set(re.findall(rb"^func (Test\w+)\(t \*testing\.T\)", source, re.MULTILINE))
    declared_names = {name.decode("ascii") for name in declared}
    rows = json_stream(raw)
    if not declared_names or not rows:
        raise BuildError("native_tests_empty")
    if any(row.get("Action") == "skip" for row in rows):
        raise BuildError("native_tests_skipped")
    if any(row.get("Action") == "fail" for row in rows):
        raise BuildError("native_tests_failed")
    passed = {row["Test"] for row in rows if row.get("Action") == "pass" and row.get("Test")}
    if not declared_names.issubset(passed):
        raise BuildError("native_tests_missing")
    if not any(row.get("Action") == "pass" and not row.get("Test") for row in rows):
        raise BuildError("native_tests_package_incomplete")
    return {"passed": len(passed), "declared": len(declared_names), "skipped": 0, "failed": 0}


def verify_module_set(modules: list[dict], requested: list[str]) -> None:
    identities = [f"{module.get('Path')}@{module.get('Version')}" for module in modules]
    if len(identities) != len(set(identities)) or set(identities) != set(requested):
        raise BuildError("module_response_incomplete")


def _ready(raw: bytes, config_sha: str) -> dict:
    rows = raw.splitlines(keepends=True)
    if len(rows) != 2:
        raise BuildError("helper_handshake_shape")
    ready, summary = (json.loads(row) for row in rows)
    if (ready.get("type") != "ready" or ready.get("version") != GITLEAKS_VERSION[1:]
            or ready.get("protocol") != "whole-file-gitleaks.v1"
            or ready.get("config_sha256") != config_sha
            or ready.get("max_target_megabytes") != 0
            or ready.get("ignore_inline_allow") is not True or ready.get("redact") != 100
            or not isinstance(ready.get("rule_count"), int) or ready["rule_count"] < 217
            or summary != {"type": "summary", "files": 0, "bytes": 0, "findings": 0}):
        raise BuildError("helper_handshake_policy")
    for key in ("policy_before_sha256", "policy_after_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", ready.get(key, "")):
            raise BuildError("helper_handshake_digest")
    return {"value": ready, "raw": rows[0]}


def locked_modules(sums: bytes) -> list[str]:
    """Download only payloads explicitly pinned by this source snapshot."""
    result = []
    seen = set()
    for line in sums.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[2].startswith("h1:"):
            raise BuildError("module_sum_shape")
        name, version, _ = fields
        if version.endswith("/go.mod"):
            continue
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._~+!/-]*", name)
                or not re.fullmatch(r"v[A-Za-z0-9._+!-]+", version)
                or (name, version) in seen):
            raise BuildError("module_sum_shape")
        seen.add((name, version))
        result.append(f"{name}@{version}")
    if not result:
        raise BuildError("module_sum_empty")
    return sorted(result)


def module_notices(modules: list[dict], cache: Path, output: Path, sums: bytes) -> list[dict]:
    """Retain exact notices for the downloaded pinned module closure."""
    records = []
    locked = set(sums.decode().splitlines())
    for module in modules:
        name, version, checksum = (module.get(key) for key in ("Path", "Version", "Sum"))
        if (not all(isinstance(value, str) and value for value in (name, version, checksum))
                or module.get("Error") or module.get("Replace")
                or f"{name} {version} {checksum}" not in locked):
            raise BuildError("module_unlocked")
        folder = no_symlinks(Path(module["Dir"]))
        if not folder.is_relative_to(cache):
            raise BuildError("module_outside_cache")
        notices = []
        for source in sorted(folder.iterdir()):
            if re.fullmatch(r"(?:licen[cs]e|copying|notice|copyright)(?:[._-].*)?",
                            source.name, re.IGNORECASE) and source.is_file():
                data = read_regular(source)
                target = output / f"module-{len(records):03d}-{len(notices):02d}.txt"
                write_new(target, data)
                notices.append({"name": source.name, "path": str(target), "sha256": digest(data)})
        if not notices:
            raise BuildError("module_license_missing")
        records.append({"module": name, "version": version, "sum": checksum,
                        "go_mod_sum": module.get("GoModSum"), "notices": notices})
    if not records or not any(row["module"] == "github.com/zricethezav/gitleaks/v8"
                              and row["version"] == GITLEAKS_VERSION for row in records):
        raise BuildError("detector_module_missing")
    return records


def build(analysis_root: Path, trusted_root: Path, output_dir: Path,
          toolchain_archive: Path | None = None) -> dict:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise BuildError("unsupported_platform")
    trusted = no_symlinks(trusted_root)
    analysis = no_symlinks(analysis_root)
    output = no_symlinks(output_dir)
    if (analysis == trusted or analysis.is_relative_to(trusted)
            or output.is_relative_to(trusted)
            or not output.is_relative_to(analysis) or output == analysis):
        raise BuildError("output_scope")
    source = trusted / "npa/scripts/image_byte_scan/go_helper"
    if source != Path(__file__).resolve().parent:
        raise BuildError("trusted_source_mismatch")
    private_dir(analysis)
    private_dir(output)
    # One preparation per output. A failure retains its log and never publishes
    # the terminal dependency receipt; retry in a fresh output directory.
    write_new(output / "go-build.claim", b"exclusive preparation\n")
    source_paths = {name: source / name for name in SOURCE_NAMES}
    source_paths[TEST_SOURCE_NAME] = trusted / TEST_SOURCE_NAME
    inputs = {name: read_regular(path) for name, path in source_paths.items()}
    config_source = trusted / ".gitleaks.toml"
    config_data = read_regular(config_source)
    work = Path(tempfile.mkdtemp(prefix="go-build-", dir=output))
    stage = private_dir(work / "source")
    logs = private_dir(work / "logs")
    notices = private_dir(output / "licenses-go")
    for name in GO_NAMES:
        write_new(stage / name, inputs[name])
    config = output / "gitleaks-config.toml"
    write_new(config, config_data)
    archive = work / "go.tar.gz"
    if toolchain_archive is None:
        # An explicit bootstrap command is the only download entrypoint.
        with urllib.request.urlopen(GO_ARCHIVE_URL) as response:
            if not response.geturl().startswith("https://"):
                raise BuildError("toolchain_transport")
            write_new(archive, response.read())
    else:
        write_new(archive, read_regular(toolchain_archive))
    go = extract_toolchain(archive, private_dir(work / "toolchain"))
    if read_regular(go.parent.parent / "LICENSE") != inputs["LICENSE-GO"]:
        raise BuildError("toolchain_license_mismatch")
    env = isolated_environment(work, config)
    version = run_step([str(go), "version"], "version", stage, env, logs).decode().strip()
    if version != f"go version go{GO_VERSION} linux/amd64":
        raise BuildError("toolchain_version")
    downloaded = run_step([str(go), "mod", "download", "-json", *locked_modules(inputs["go.sum"])],
                          "download", stage, env, logs)
    run_step([str(go), "mod", "verify"], "verify", stage, env, logs)
    test_output = run_step([str(go), "test", "-count=1", "-json", "./..."], "tests", stage, env, logs)
    native_counts = verify_native_tests(test_output, inputs["main_test.go"])
    binary = output / "whole-file-scanner"
    run_step([str(go), "build", "-trimpath", "-buildvcs=false", "-o", str(binary), "."],
             "build", stage, env, logs)
    binary.chmod(0o700)
    download_records = json_stream(downloaded)
    verify_module_set(download_records, locked_modules(inputs["go.sum"]))
    for module in download_records:
        if module.get("Path") == "github.com/zricethezav/gitleaks/v8":
            if read_regular(Path(module["Dir"]) / "LICENSE") != inputs["LICENSE-GITLEAKS"]:
                raise BuildError("detector_license_mismatch")
    module_records = module_notices(download_records, Path(env["GOMODCACHE"]),
                                    notices, inputs["go.sum"])
    for name in ("LICENSE-GO", "LICENSE-GITLEAKS"):
        write_new(notices / name, inputs[name])
    handshake = run_step([str(binary), "--config", str(config)], "handshake", stage,
                         env, logs, b"")
    parsed = _ready(handshake, digest(config_data))
    ready_path = output / "helper-ready.json"
    write_new(ready_path, parsed["raw"])
    for name, data in inputs.items():
        if read_regular(source_paths[name]) != data:
            raise BuildError("source_changed")
    for name in GO_NAMES:
        if read_regular(stage / name) != inputs[name]:
            raise BuildError("staged_source_changed")
    if read_regular(config_source) != config_data or read_regular(config) != config_data:
        raise BuildError("config_changed")
    receipt = {"schema_version": "npa.image-byte-scan-tools.v1",
               "helper": {"path": str(binary), "sha256": digest(read_regular(binary))},
               "config": {"path": str(config), "sha256": digest(config_data)},
               "ready": {"path": str(ready_path), "sha256": digest(parsed["raw"])},
               "source": {name: digest(data) for name, data in inputs.items()},
               "toolchain": {"version": GO_VERSION, "archive_url": GO_ARCHIVE_URL,
                             "archive_sha256": GO_ARCHIVE_SHA256},
               "modules": module_records,
               "validation": {"native_tests": "passed", "native_counts": native_counts, "module_verification": "passed",
                              "empty_handshake": "passed", "logs": str(logs)}}
    write_new(output / "dependency-receipt.json", json_bytes(receipt))
    return receipt


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise BuildError("arguments_invalid")


def main(argv: list[str] | None = None) -> int:
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--toolchain-archive", type=Path)
    try:
        args = parser.parse_args(argv)
    except BuildError:
        print(json.dumps({"status": "failed", "error": "arguments_invalid"}), file=sys.stderr)
        return 2
    os.umask(0o077)
    error = None
    try:
        with cancellation_scope():
            receipt = build(args.analysis_root, args.trusted_root, args.output_dir,
                            args.toolchain_archive)
    except BuildCancelled:
        print(json.dumps({"status": "cancelled", "error": "bootstrap_cancelled"}), file=sys.stderr)
        return 130
    except BuildError as exc:
        error = str(exc)
    except (OSError, ValueError, KeyError, tarfile.TarError, EOFError, UnicodeError):
        error = "bootstrap_io_or_schema"
    if error:
        print(json.dumps({"status": "failed", "error": error}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "helper_sha256": receipt["helper"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
