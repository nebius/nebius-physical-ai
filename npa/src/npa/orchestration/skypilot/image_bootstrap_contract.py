"""Digest-bound SkyPilot 0.12.2 worker-image bootstrap verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


CONTRACT_VERSION = "skypilot-0.12.2-v1"
ATTESTATION_LABEL = "org.nebius.npa.skypilot-bootstrap-contract"
FIRST_PARTY_REPOSITORY_PREFIX = "npa-"
PROBE_TIMEOUT_SECONDS = 180


class ImageBootstrapContractError(RuntimeError):
    """The selected immutable image cannot satisfy SkyPilot bootstrap."""


@dataclass(frozen=True)
class SkyPilotImageBootstrapContract:
    version: str = CONTRACT_VERSION
    permits_root_or_passwordless_sudo: bool = True
    required_commands: tuple[str, ...] = (
        "sh",
        "sudo",
        "sshd",
        "rsync",
        "service",
    )
    writable_locations: tuple[str, ...] = ("/tmp", "$HOME")
    requires_argument_forwarding: bool = True


@dataclass(frozen=True)
class ImageContractEvidence:
    image: str
    digest: str
    contract_version: str
    state: str
    source: str
    checks: tuple[str, ...] = ()
    cleanup: str = "not_applicable"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "compatible"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def immutable_image_reference(image: str, digest: str) -> str:
    base = str(image).removeprefix("docker:").split("@", 1)[0]
    if "/" not in base:
        raise ImageBootstrapContractError("image must have a registry-qualified repository")
    tail = base.rsplit("/", 1)[-1]
    if ":" in tail:
        base = base.rsplit(":", 1)[0]
    normalized = str(digest or "").strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise ImageBootstrapContractError(
            "registry did not return an immutable sha256 image digest"
        )
    return f"{base}@{normalized}"


def is_first_party_image(image: str) -> bool:
    repository = str(image).removeprefix("docker:").split("@", 1)[0]
    repository = repository.rsplit(":", 1)[0]
    return repository.rsplit("/", 1)[-1].startswith(FIRST_PARTY_REPOSITORY_PREFIX)


def verify_attestation(
    *, image: str, digest: str, labels: Mapping[str, object]
) -> ImageContractEvidence:
    """Validate machine-readable metadata from the selected digest's config."""

    immutable = immutable_image_reference(image, digest)
    declared = str(labels.get(ATTESTATION_LABEL) or "").strip()
    if declared != CONTRACT_VERSION:
        return ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="incompatible",
            source="oci_attestation",
            detail=(
                "missing bootstrap-contract attestation"
                if not declared
                else f"attestation version mismatch: {declared}"
            ),
        )
    return ImageContractEvidence(
        image=immutable,
        digest=digest,
        contract_version=CONTRACT_VERSION,
        state="compatible",
        source="oci_attestation",
        checks=("digest_bound", "attestation_version"),
    )


def probe_name(digest: str) -> str:
    suffix = hashlib.sha256(f"{digest}\0{CONTRACT_VERSION}".encode()).hexdigest()[:16]
    return f"npa-sky-image-probe-{suffix}"


Runner = Callable[[list[str], Mapping[str, str]], subprocess.CompletedProcess[str]]


def _run(argv: list[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def probe_image_capabilities(
    *,
    image: str,
    digest: str,
    context: str,
    kubeconfig: str = "",
    runner: Runner = _run,
) -> ImageContractEvidence:
    """Run and exactly clean one bounded capability pod for an unattested image."""

    immutable = immutable_image_reference(image, digest)
    if not str(context or "").strip():
        raise ImageBootstrapContractError("an exact Kubernetes context is required")
    name = probe_name(digest)
    env = dict(os.environ)
    if kubeconfig:
        env["KUBECONFIG"] = kubeconfig
    script = (
        "set -eu; "
        "test -w /tmp; test -w \"$HOME\"; "
        "command -v rsync; command -v service; "
        "(command -v sshd || test -x /usr/sbin/sshd); "
        "if [ \"$(id -u)\" != 0 ]; then command -v sudo; sudo -n true; fi; "
        "test \"$(/bin/sh -c 'printf %s forwarded' sentinel)\" = forwarded"
    )
    common = ["kubectl", "--context", context]
    create = runner(
        [
            *common,
            "run",
            name,
            "--restart=Never",
            f"--image={immutable}",
            "--labels=npa.nebius.com/owned=true,npa.nebius.com/purpose=sky-image-probe",
            "--",
            "/bin/sh",
            "-c",
            script,
        ],
        env,
    )
    primary = "" if create.returncode == 0 else (create.stderr or create.stdout).strip()
    wait = None
    if create.returncode == 0:
        wait = runner(
            [
                *common,
                "wait",
                f"pod/{name}",
                "--for=jsonpath={.status.phase}=Succeeded",
                f"--timeout={PROBE_TIMEOUT_SECONDS}s",
            ],
            env,
        )
        if wait.returncode != 0:
            primary = (wait.stderr or wait.stdout).strip()
    delete = runner(
        [*common, "delete", "pod", name, "--ignore-not-found=true", "--wait=true"],
        env,
    )
    cleanup = "verified" if delete.returncode == 0 else "failed"
    if delete.returncode != 0:
        return ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="indeterminate",
            source="ephemeral_capability_probe",
            cleanup=cleanup,
            detail="exact probe cleanup could not be verified",
        )
    if primary:
        return ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="incompatible" if create.returncode == 0 else "indeterminate",
            source="ephemeral_capability_probe",
            cleanup=cleanup,
            detail=primary[:500],
        )
    return ImageContractEvidence(
        image=immutable,
        digest=digest,
        contract_version=CONTRACT_VERSION,
        state="compatible",
        source="ephemeral_capability_probe",
        checks=(
            "effective_user",
            "passwordless_sudo_or_root",
            "openssh_server",
            "rsync",
            "service_init",
            "writable_locations",
            "entrypoint_argument_forwarding",
        ),
        cleanup=cleanup,
    )


def cache_key(digest: str) -> str:
    return f"{digest}|{CONTRACT_VERSION}"


def load_cached_evidence(path: Path, digest: str) -> ImageContractEvidence | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        item = payload.get(cache_key(digest))
        if not isinstance(item, dict):
            return None
        if isinstance(item.get("checks"), list):
            item = {**item, "checks": tuple(str(value) for value in item["checks"])}
        evidence = ImageContractEvidence(**item)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return evidence if evidence.ok and evidence.digest == digest else None


def store_cached_evidence(path: Path, evidence: ImageContractEvidence) -> None:
    if not evidence.ok or evidence.cleanup == "failed":
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload[cache_key(evidence.digest)] = evidence.to_dict()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
