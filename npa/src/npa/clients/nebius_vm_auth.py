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
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"
)
_TOKEN_LINE_RE = re.compile(
    r"(?im)^(\s*(?:access[_ -]?token|iam[_ -]?token|authorization)\s*[:=]\s*).+$"
)
_OPAQUE_LINE_RE = re.compile(r"(?m)^(\s*)[A-Za-z0-9._~-]{40,}(\s*)$")
_SAFE_BROWSER_SUFFIXES = (".nebius.com", ".nebius.cloud")
_CALLBACK_KEYS = frozenset(
    {"redirect_uri", "redirect_url", "callback", "callback_url", "return_url"}
)
_SECRET_QUERY_KEYS = frozenset({"access_token", "id_token", "token", "refresh_token"})
_PTY_COLUMNS = 4096
_MAX_TRANSCRIPT_BYTES = 131072


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
    # Interactive terminal UIs commonly redraw the same instruction (often with
    # carriage returns). Identical safe URLs are one candidate; distinct URLs
    # remain ambiguous and fail closed below.
    browser_urls = list(dict.fromkeys(url for url in urls if _safe_browser_url(url)))
    ports: set[int] = set()
    for url in urls:
        ports.update(_nested_callback_ports(url))
    if len(browser_urls) != 1:
        raise VmAuthError(
            "Nebius CLI output did not contain exactly one safe browser URL"
        )
    if len(ports) != 1:
        raise VmAuthError(
            "Nebius CLI output did not contain one unambiguous loopback callback"
        )
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


def _wait_readable(stream: object, timeout: float) -> bool:
    readable, _, _ = select.select([stream], [], [], max(0.0, timeout))
    return bool(readable)


def _read_chunk(stream: object) -> bytes:
    try:
        return os.read(stream.fileno(), 65536)  # type: ignore[attr-defined]
    except BlockingIOError:
        return b""


def _signal_process_group(
    process: subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    """Best-effort signal for both ``script`` and its interactive CLI child."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, OSError):
        try:
            process.send_signal(sig)
        except OSError:
            pass


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Best-effort bounded cleanup for the PTY wrapper and its child."""

    if process.poll() is None:
        _signal_process_group(process, signal.SIGINT)
    try:
        process.wait(timeout=2)
        return
    except (subprocess.TimeoutExpired, TypeError):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, OSError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=2)
    except (subprocess.TimeoutExpired, TypeError):
        pass


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
    prefix = [nebius_cli, *(["--profile", profile] if profile else [])]
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
    _clock: Callable[[], float] = time.monotonic,
    _wait_for_output: Callable[[object, float], bool] = _wait_readable,
    _read_output: Callable[[object], bytes] = _read_chunk,
) -> ProfileVerification:
    """Run the real no-browser CLI flow, relay safely, then verify the profile."""

    output = output or sys.stdout
    if not shutil.which(nebius_cli):
        raise VmAuthError("Nebius CLI is not installed")
    initial = verify_profile(profile, nebius_cli=nebius_cli)
    if initial.identity_verified and initial.iam_token_minted:
        output.write(
            "Nebius CLI profile is already authenticated; identity and IAM minting verified.\n"
        )
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
    auth_probe = [
        nebius_cli,
        *(["--profile", profile] if profile else []),
        "--no-browser",
        "--auth-timeout",
        f"{int(auth_timeout_seconds)}s",
        "--no-check-update",
        "iam",
        "whoami",
    ]
    script_bin = shutil.which("script")
    if not script_bin:
        raise VmAuthError(
            "the `script` PTY helper is required for interactive CLI authentication"
        )
    # Nebius profile creation is an interactive terminal UI. ``script`` provides
    # a PTY while writing its transcript only to /dev/null. Set a deliberately
    # wide PTY before exec so long OAuth URLs (including late redirect_uri query
    # parameters) are not terminal-wrapped at common 80/100/120-column widths.
    # Current CLI releases create the profile before OAuth and start federation
    # on the first authenticated call. Keep both operations in the same PTY so
    # releases that authenticate during create and releases that authenticate on
    # first use both expose the safe callback to the same parser/deadline.
    pty_command = (
        f"stty cols {_PTY_COLUMNS} rows 24; "
        f"{shlex.join(command)} && exec {shlex.join(auth_probe)}"
    )
    command = [script_bin, "-qefc", pty_command, "/dev/null"]
    env = strip_ambient_token_env(os.environ)
    process = subprocess.Popen(
        command,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=env,
        start_new_session=True,
    )
    assert process.stdout is not None
    try:
        os.set_blocking(process.stdout.fileno(), False)
    except (AttributeError, OSError):
        pass
    transcript_bytes = bytearray()
    instructions: AuthInstructions | None = None
    deadline = _clock() + int(auth_timeout_seconds)
    try:
        while True:
            now = _clock()
            if process.poll() is None and now >= deadline:
                _signal_process_group(process, signal.SIGINT)
                raise VmAuthError(
                    "Nebius CLI authentication timed out and was cancelled"
                )
            readable = _wait_for_output(
                process.stdout, min(0.2, max(0.0, deadline - now))
            )
            if not readable:
                if process.poll() is not None:
                    break
                continue
            chunk = _read_output(process.stdout)
            if not chunk:
                if process.poll() is not None:
                    break
                continue
            transcript_bytes.extend(chunk)
            del transcript_bytes[:-_MAX_TRANSCRIPT_BYTES]
            transcript = transcript_bytes.decode("utf-8", errors="replace")
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
                    output.write(
                        f"In another local terminal: {instructions.ssh_command}\n"
                    )
                    output.flush()
    except KeyboardInterrupt as exc:
        _signal_process_group(process, signal.SIGINT)
        raise VmAuthError("Nebius CLI authentication was cancelled") from exc
    finally:
        _stop_process(process)

    if process.returncode != 0:
        raise VmAuthError("Nebius CLI profile creation did not complete")
    if instructions is None:
        # A successful interactive federation flow must have advertised a safe
        # callback. Do not declare success from malformed or unexpected output.
        raise VmAuthError(
            "profile completed without a verifiable safe loopback callback"
        )
    result = verify_profile(profile, nebius_cli=nebius_cli)
    if not (result.identity_verified and result.iam_token_minted):
        raise VmAuthError(
            "profile callback completed, but identity or IAM minting verification failed"
        )
    output.write(
        "CLI profile callback completed; identity and subsequent IAM token minting verified.\n"
    )
    return result
