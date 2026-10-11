"""Obtain human IAM tokens through an explicitly selected Nebius CLI profile."""

from __future__ import annotations

import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from npa.clients.nebius_auth import strip_ambient_token_env
from npa.clients.nebius_vm_auth import VmAuthError, parse_auth_transcript

from .errors import AuthenticationError

_OUTPUT_LIMIT = 131072
_REFRESH_SECONDS = 35


@dataclass(frozen=True)
class NebiusProfile:
    """Expose profile selection metadata without credential material.

    Args:
        name: Explicit Nebius CLI profile name.
        auth_type: Authentication mechanism reported by the CLI.
        active: Whether the CLI marks this profile as its default.
    Returns:
        Immutable profile metadata.
    Raises:
        None.
    """

    name: str
    auth_type: str
    active: bool = False


def _environment() -> dict[str, str]:
    environment = strip_ambient_token_env()
    environment.pop("NPA_NEBIUS_IAM_TOKEN", None)
    return environment


def _prefix(profile: str = "", config_file: str | Path | None = None) -> list[str]:
    command = ["nebius", "--no-check-update", "--no-progress", "--color=false"]
    command.extend(["--debug=false", "--insecure=false"])
    if profile:
        command.extend(["--profile", _profile_name(profile)])
    if config_file is not None:
        command.extend(["--config", str(Path(config_file).expanduser())])
    return command


def _profile_name(value: str) -> str:
    if (
        not value
        or value.startswith("-")
        or len(value) > 256
        or any(
            character.isspace() or not character.isprintable() for character in value
        )
    ):
        raise AuthenticationError(
            "Select a valid Nebius CLI profile with --nebius-profile"
        )
    return value


def _configuration_output(
    command: list[str], *, allow_unconfigured: bool = False
) -> str:
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=_environment(),
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        raise AuthenticationError(
            "Install the official Nebius CLI before npa login"
        ) from None
    except (OSError, subprocess.SubprocessError):
        raise AuthenticationError(
            "Nebius CLI profile configuration could not be read"
        ) from None
    if allow_unconfigured and _unconfigured_profiles(result):
        return ""
    if result.returncode:
        raise AuthenticationError(
            "Nebius CLI profile was not found or could not be read"
        )
    return result.stdout.strip()


def _unconfigured_profiles(result: subprocess.CompletedProcess) -> bool:
    if result.returncode != 4 or result.stdout.strip():
        return False
    first_line = result.stderr.splitlines()[0] if result.stderr else ""
    return first_line == "Error: config: get profile: no profile configured" or (
        first_line.startswith("Error: missing configuration: ")
        and first_line.endswith(": no such file or directory")
    )


def _auth_type(profile: str, config_file: str | Path | None = None) -> str:
    value = _configuration_output(
        [*_prefix(profile, config_file), "config", "get", "auth-type"]
    )
    return value if value in {"federation", "service account"} else "unknown"


def list_profiles(*, config_file: str | Path | None = None) -> list[NebiusProfile]:
    """List existing CLI profiles without changing the CLI default.

    Args:
        config_file: Optional isolated CLI configuration; default uses Nebius settings.
    Returns:
        Profile names, authentication mechanisms, and default markers.
    Raises:
        AuthenticationError: The CLI is missing or metadata cannot be read.
    """
    listing = _configuration_output(
        [*_prefix(config_file=config_file), "profile", "list"],
        allow_unconfigured=True,
    )
    profiles = []
    for line in listing.splitlines():
        active = line.endswith((" [default]", " [active]"))
        name = _profile_name(line.removesuffix(" [default]").removesuffix(" [active]"))
        profiles.append(NebiusProfile(name, _auth_type(name, config_file), active))
    return profiles


def _configuration_destination(config_file: str | Path) -> Path:
    path = Path(config_file).expanduser().absolute()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise AuthenticationError(
            "Nebius configuration directory could not be created"
        ) from None
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_mode & 0o077
        or parent.st_uid != os.getuid()
    ):
        raise AuthenticationError(
            "Nebius configuration requires an owned private directory"
        )
    if path.exists() or path.is_symlink():
        raise AuthenticationError("Choose a new private Nebius configuration file")
    return path


def _initialize_profile(profile: str, path: Path) -> None:
    _configuration_output(
        [
            *_prefix(config_file=path),
            "profile",
            "create",
            profile,
            "--skip-auth",
            "--endpoint",
            "api.nebius.cloud",
            "--federation-endpoint",
            "auth.nebius.com",
        ]
    )
    os.chmod(path, 0o600)
    if _auth_type(profile, path) != "federation":
        raise AuthenticationError("Nebius CLI did not create a human sign-in profile")


def create_human_profile(profile: str, config_file: str | Path) -> None:
    """Create an isolated human profile without changing existing CLI settings.

    Args:
        profile: Name for the new human profile.
        config_file: New configuration file inside an owned private directory.
    Returns:
        None; the official CLI owns the new configuration.
    Raises:
        AuthenticationError: The destination is unsafe or profile creation fails.
    """
    _profile_name(profile)
    path = _configuration_destination(config_file)
    descriptor, name = tempfile.mkstemp(prefix=".nebius-login-", dir=path.parent)
    os.close(descriptor)
    staging = Path(name)
    try:
        _initialize_profile(profile, staging)
        os.link(staging, path)
    except OSError:
        raise AuthenticationError(
            "Nebius profile configuration could not be saved"
        ) from None
    finally:
        staging.unlink(missing_ok=True)


@dataclass
class _Capture:
    interactive: bool
    ssh_host: str
    output: TextIO
    token: bytearray = field(default_factory=bytearray)
    transcript: bytearray = field(default_factory=bytearray)
    browser_url: str = ""


def _relay_authentication(capture: _Capture) -> None:
    try:
        instructions = parse_auth_transcript(
            capture.transcript.decode("utf-8", errors="replace"),
            ssh_host=capture.ssh_host or "localhost",
        )
    except VmAuthError:
        return
    if not capture.interactive:
        raise AuthenticationError(
            "Nebius browser sign-in is required; run npa login again"
        )
    if capture.browser_url == instructions.browser_url:
        return
    capture.browser_url = instructions.browser_url
    if capture.ssh_host:
        print("On your laptop, open this callback tunnel first:", file=capture.output)
        print(instructions.ssh_command, file=capture.output)
    print("Finish Nebius sign-in in your browser:", file=capture.output)
    print(instructions.browser_url, file=capture.output, flush=True)


def _read_ready(selector: selectors.BaseSelector, capture: _Capture) -> None:
    for key, _ in selector.select(timeout=0.1):
        chunk = os.read(key.fileobj.fileno(), 65536)
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        target = capture.token if key.data == "stdout" else capture.transcript
        target.extend(chunk)
        if len(target) > _OUTPUT_LIMIT:
            raise AuthenticationError(
                "Nebius CLI returned unexpected authentication output"
            )
        if key.data == "stderr":
            _relay_authentication(capture)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _collect_token(process: subprocess.Popen, capture: _Capture) -> str:
    deadline = None if capture.interactive else time.monotonic() + _REFRESH_SECONDS
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            if deadline is not None and time.monotonic() >= deadline:
                raise AuthenticationError(
                    "Nebius token refresh timed out; run npa login again"
                )
            _read_ready(selector, capture)
    if process.wait():
        raise AuthenticationError(
            "Nebius sign-in did not complete; run npa login again"
        )
    token = capture.token.decode("utf-8", errors="replace").strip()
    if not token or any(not 33 <= ord(character) <= 126 for character in token):
        raise AuthenticationError("Nebius CLI did not return a valid access token")
    return token


def _start_cli(command: list[str]) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_environment(),
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise AuthenticationError(
            "Nebius CLI could not start; check its installation"
        ) from None


def _run_token(
    profile: str,
    *,
    interactive: bool,
    no_browser: bool,
    ssh_host: str,
    output: TextIO | None,
    config_file: str | Path | None,
) -> str:
    _profile_name(profile)
    if ssh_host.startswith("-") or any(
        character.isspace() or not character.isprintable() for character in ssh_host
    ):
        raise AuthenticationError("Use an SSH host alias or user@host for --ssh-host")
    if _auth_type(profile, config_file) != "federation":
        raise AuthenticationError(
            "Select a human Nebius federation profile for npa login"
        )
    command = _prefix(profile, config_file)
    if no_browser or not interactive or ssh_host:
        command.append("--no-browser")
    if not interactive:
        command.extend(["--auth-timeout", "30s"])
    process = _start_cli([*command, "iam", "get-access-token"])
    try:
        return _collect_token(
            process, _Capture(interactive, ssh_host, output or sys.stderr)
        )
    finally:
        _stop_process(process)
        process.stdout.close()
        process.stderr.close()


def login_token(
    profile: str,
    *,
    no_browser: bool = False,
    ssh_host: str = "",
    output: TextIO | None = None,
    config_file: str | Path | None = None,
) -> str:
    """Sign in using the official CLI and return the token only in memory.

    Args:
        profile: Existing human CLI profile, selected without changing defaults.
        no_browser: Print the browser URL instead of opening a local browser.
        ssh_host: Optional SSH destination for remote callback instructions.
        output: Destination for safe sign-in instructions; defaults to stderr.
        config_file: Optional isolated CLI configuration created for Workbench.
    Returns:
        A fresh access token; callers must never print or persist it.
    Raises:
        AuthenticationError: Profile selection or official CLI sign-in failed.
    """
    return _run_token(
        profile,
        interactive=True,
        no_browser=no_browser,
        ssh_host=ssh_host,
        output=output,
        config_file=config_file,
    )


def fresh_token(profile: str, *, config_file: str | Path | None = None) -> str:
    """Refresh a saved human session without starting an interactive login.

    Args:
        profile: Explicit human CLI profile selected during npa login.
        config_file: Optional isolated CLI configuration created for Workbench.
    Returns:
        A fresh access token kept only in memory.
    Raises:
        AuthenticationError: Browser sign-in is needed or token refresh failed.
    """
    return _run_token(
        profile,
        interactive=False,
        no_browser=True,
        ssh_host="",
        output=None,
        config_file=config_file,
    )
