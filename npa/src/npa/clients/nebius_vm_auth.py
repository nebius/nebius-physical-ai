"""Safe no-browser Nebius CLI profile authentication for remote operator VMs."""

from __future__ import annotations

import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Mapping, TextIO
from urllib.parse import parse_qsl, unquote, urlsplit

from npa.clients.nebius_auth import strip_ambient_token_env


class VmAuthError(RuntimeError):
    """The remote profile flow could not complete safely."""


_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_JWT_RE = re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+")
_TOKEN_LINE_RE = re.compile(
    r"(?im)^(\s*(?:access[_ -]?token|iam[_ -]?token|authorization)\s*[:=]\s*).+$"
)
_OPAQUE_LINE_RE = re.compile(r"(?m)^(\s*)[A-Za-z0-9._~-]{40,}(\s*)$")
_SAFE_BROWSER_SUFFIXES = (".nebius.com", ".nebius.cloud")
_CALLBACK_KEYS = frozenset({"redirect_uri", "redirect_url", "callback", "callback_url", "return_url"})
_SECRET_QUERY_KEYS = frozenset({"access_token", "id_token", "token", "refresh_token"})


@dataclass(frozen=True)
class AuthInstructions:
    """Public instructions derived from one CLI authentication transcript."""

    browser_url: str
    callback_port: int
    ssh_command: str


@dataclass(frozen=True)
class ProfileVerification:
    """Secret-free proof that a profile identifies and can mint an IAM token."""

    profile: str
    identity_verified: bool
    iam_token_minted: bool


def _clean_url(raw: str) -> str:
    return raw.rstrip(".,;)]}\x1b")


def _loopback_port(raw: str) -> int | None:
    """Return a safe loopback callback port, otherwise ``None``."""

    try:
        parsed = urlsplit(unquote(raw))
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme != "http" or host not in {"localhost", "127.0.0.1", "::1"}:
        return None
    return port if port is not None and 1 <= port <= 65535 else None


def _nested_callback_ports(url: str) -> set[int]:
    ports: set[int] = set()
    direct = _loopback_port(url)
    if direct is not None:
        ports.add(direct)
    try:
        query = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    except ValueError:
        return ports
    for key, value in query:
        if key.lower() in _CALLBACK_KEYS:
            port = _loopback_port(value)
            if port is not None:
                ports.add(port)
    return ports


def _safe_browser_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        query_keys = {key.lower() for key, _ in parse_qsl(parsed.query)}
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and (host == "nebius.com" or host.endswith(_SAFE_BROWSER_SUFFIXES))
        and not (query_keys & _SECRET_QUERY_KEYS)
    )


def ssh_localhost_forward(
    port: int, *, ssh_host: str, ssh_user: str = "", identity_file: str = ""
) -> str:
    """Render an exact local-machine SSH forward for one verified callback port."""

    if not 1 <= int(port) <= 65535:
        raise VmAuthError("callback port is outside the valid TCP range")
    host = str(ssh_host or "").strip()
    user = str(ssh_user or "").strip()
    if not host or any(char.isspace() for char in host + user):
        raise VmAuthError("a safe SSH host and optional user are required")
    destination = f"{user}@{host}" if user else host
    command = ["ssh", "-N", "-L", f"{port}:127.0.0.1:{port}"]
    if identity_file:
        command.extend(["-i", str(identity_file)])
    command.append(destination)
    return shlex.join(command)


def parse_auth_transcript(
    transcript: str,
    *,
    ssh_host: str,
    ssh_user: str = "",
    identity_file: str = "",
) -> AuthInstructions:
    """Extract one official browser URL and one unambiguous loopback callback."""

    clean = _ANSI_RE.sub("", str(transcript or ""))
    urls = [_clean_url(match.group(0)) for match in _URL_RE.finditer(clean)]
    browser_urls = [url for url in urls if _safe_browser_url(url)]
    ports: set[int] = set()
    for url in urls:
        ports.update(_nested_callback_ports(url))
    if len(browser_urls) != 1:
        raise VmAuthError("Nebius CLI output did not contain exactly one safe browser URL")
    if len(ports) != 1:
        raise VmAuthError("Nebius CLI output did not contain one unambiguous loopback callback")
    port = ports.pop()
    return AuthInstructions(
        browser_url=browser_urls[0],
        callback_port=port,
        ssh_command=ssh_localhost_forward(
            port,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            identity_file=identity_file,
        ),
    )


def redact_auth_output(text: str) -> str:
    """Remove token-shaped material and URLs from relay/log-safe status text."""

    value = _JWT_RE.sub("[REDACTED]", str(text or ""))
    value = _TOKEN_LINE_RE.sub(r"\1[REDACTED]", value)
    value = _OPAQUE_LINE_RE.sub(r"\1[REDACTED]\2", value)
    return _URL_RE.sub("<authentication-url>", value)


def verify_profile(
    profile: str = "",
    *,
    nebius_cli: str = "nebius",
    env: Mapping[str, str] | None = None,
    timeout: int = 30,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ProfileVerification:
    """Verify identity and IAM minting while discarding both command outputs."""

    clean_env = strip_ambient_token_env(env)
    prefix = [nebius_cli, *( ["--profile", profile] if profile else [])]
    common = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": timeout,
        "check": False,
        "env": clean_env,
    }
    try:
        identity = runner([*prefix, "iam", "whoami"], **common)
        minted = runner([*prefix, "iam", "get-access-token"], **common)
    except (OSError, subprocess.SubprocessError):
        return ProfileVerification(profile, False, False)
    return ProfileVerification(
        profile,
        getattr(identity, "returncode", 1) == 0,
        getattr(minted, "returncode", 1) == 0,
    )


def run_vm_profile_auth(
    *,
    ssh_host: str,
    ssh_user: str = "",
    identity_file: str = "",
    profile: str = "",
    auth_timeout_seconds: int = 900,
    nebius_cli: str = "nebius",
    output: TextIO | None = None,
) -> ProfileVerification:
    """Run the real no-browser CLI flow, relay safely, then verify the profile."""

    output = output or sys.stdout
    if not shutil.which(nebius_cli):
        raise VmAuthError("Nebius CLI is not installed")
    initial = verify_profile(profile, nebius_cli=nebius_cli)
    if initial.identity_verified and initial.iam_token_minted:
        output.write("Nebius CLI profile is already authenticated; identity and IAM minting verified.\n")
        return initial

    command = [
        nebius_cli,
        "--no-browser",
        "--auth-timeout",
        f"{int(auth_timeout_seconds)}s",
        "--no-check-update",
        "profile",
        "create",
    ]
    if profile:
        command.append(profile)
    script_bin = shutil.which("script")
    if not script_bin:
        raise VmAuthError("the `script` PTY helper is required for interactive CLI authentication")
    # Nebius profile creation is an interactive terminal UI. ``script`` provides
    # a PTY while writing its transcript only to /dev/null; NPA sanitizes the
    # relayed stream and retains no authentication transcript.
    command = [script_bin, "-qefc", shlex.join(command), "/dev/null"]
    env = strip_ambient_token_env(os.environ)
    process = subprocess.Popen(
        command,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    transcript = ""
    instructions: AuthInstructions | None = None
    deadline = time.monotonic() + int(auth_timeout_seconds)
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.send_signal(signal.SIGINT)
                raise VmAuthError("Nebius CLI authentication timed out and was cancelled")
            readable, _, _ = select.select([process.stdout], [], [], 0.2)
            if not readable:
                continue
            chunk = process.stdout.readline()
            if not chunk:
                continue
            transcript = (transcript + chunk)[-131072:]
            output.write(redact_auth_output(chunk))
            output.flush()
            if instructions is None:
                try:
                    instructions = parse_auth_transcript(
                        transcript,
                        ssh_host=ssh_host,
                        ssh_user=ssh_user,
                        identity_file=identity_file,
                    )
                except VmAuthError:
                    pass
                else:
                    output.write(f"Open locally: {instructions.browser_url}\n")
                    output.write(f"In another local terminal: {instructions.ssh_command}\n")
                    output.flush()
        remainder = process.stdout.read() or ""
        transcript = (transcript + remainder)[-131072:]
        if remainder:
            output.write(redact_auth_output(remainder))
    except KeyboardInterrupt as exc:
        process.send_signal(signal.SIGINT)
        raise VmAuthError("Nebius CLI authentication was cancelled") from exc
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()

    if process.returncode != 0:
        raise VmAuthError("Nebius CLI profile creation did not complete")
    if instructions is None:
        # A successful interactive federation flow must have advertised a safe
        # callback. Do not declare success from malformed or unexpected output.
        raise VmAuthError("profile completed without a verifiable safe loopback callback")
    result = verify_profile(profile, nebius_cli=nebius_cli)
    if not (result.identity_verified and result.iam_token_minted):
        raise VmAuthError("profile callback completed, but identity or IAM minting verification failed")
    output.write("CLI profile callback completed; identity and subsequent IAM token minting verified.\n")
    return result
