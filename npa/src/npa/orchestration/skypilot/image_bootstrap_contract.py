"""Digest-bound SkyPilot 0.12.2 worker-image bootstrap verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
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


@dataclass(frozen=True)
class OCIReference:
    registry: str
    repository: str
    tag: str = ""
    digest: str = ""

    @property
    def name(self) -> str:
        return f"{self.registry}/{self.repository}"


def parse_oci_reference(image: str) -> OCIReference:
    """Parse one registry-qualified OCI reference without confusing ports and tags."""

    raw = str(image or "").strip()
    if raw.startswith("docker:"):
        raw = raw[len("docker:") :]
    if not raw or any(character.isspace() for character in raw) or "://" in raw:
        raise ImageBootstrapContractError("image is not a valid OCI reference")
    if raw.count("@") > 1:
        raise ImageBootstrapContractError("image has multiple digest separators")
    named, separator, digest = raw.partition("@")
    if separator and not digest:
        raise ImageBootstrapContractError("image digest is empty")
    if digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ImageBootstrapContractError("image digest is not a valid sha256 digest")
    if "/" not in named:
        raise ImageBootstrapContractError("image must have a registry-qualified repository")
    registry, path = named.split("/", 1)
    if not registry or not path or path.startswith("/") or path.endswith("/"):
        raise ImageBootstrapContractError("image has an invalid registry or repository")
    if registry.startswith("["):
        close = registry.find("]")
        if close < 2 or registry[close + 1 :] not in {""} and not re.fullmatch(
            r":[0-9]+", registry[close + 1 :]
        ):
            raise ImageBootstrapContractError("image has an invalid IPv6 registry authority")
    elif registry.count(":") > 1 or (
        ":" in registry and not registry.rsplit(":", 1)[1].isdigit()
    ):
        raise ImageBootstrapContractError("image has an invalid registry authority")
    final = path.rsplit("/", 1)[-1]
    tag = ""
    if ":" in final:
        final_name, tag = final.rsplit(":", 1)
        if not final_name or not tag:
            raise ImageBootstrapContractError("image has an invalid tag")
        path = f"{path.rsplit('/', 1)[0]}/{final_name}" if "/" in path else final_name
    if not all(part and part not in {".", ".."} for part in path.split("/")):
        raise ImageBootstrapContractError("image has an invalid repository path")
    return OCIReference(registry=registry, repository=path, tag=tag, digest=digest)


def immutable_image_reference(image: str, digest: str) -> str:
    parsed = parse_oci_reference(image)
    normalized = str(digest or "").strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise ImageBootstrapContractError(
            "registry did not return an immutable sha256 image digest"
        )
    if parsed.digest and parsed.digest != normalized:
        raise ImageBootstrapContractError(
            "image reference digest conflicts with the registry-resolved digest"
        )
    return f"{parsed.name}@{normalized}"


def is_first_party_image(image: str) -> bool:
    try:
        parsed = parse_oci_reference(image)
    except ImageBootstrapContractError:
        return False
    return parsed.repository.rsplit("/", 1)[-1].startswith(
        FIRST_PARTY_REPOSITORY_PREFIX
    )


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


def probe_name(digest: str, nonce: str = "") -> str:
    """Return a per-invocation name while retaining digest correlation."""

    correlation = hashlib.sha256(
        f"{digest}\0{CONTRACT_VERSION}".encode()
    ).hexdigest()[:10]
    unique = str(nonce or secrets.token_hex(8)).lower()
    if not re.fullmatch(r"[0-9a-f]{8,32}", unique):
        raise ImageBootstrapContractError("probe nonce must be 8-32 hexadecimal characters")
    return f"npa-sky-image-probe-{correlation}-{unique}"[:63].rstrip("-")


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
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(8),
) -> ImageContractEvidence:
    """Run and exactly clean one bounded capability pod for an unattested image."""

    immutable = immutable_image_reference(image, digest)
    if not str(context or "").strip():
        raise ImageBootstrapContractError("an exact Kubernetes context is required")
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
    name = ""
    probe_id = ""
    create: subprocess.CompletedProcess[str] | None = None
    for _attempt in range(3):
        probe_id = str(nonce_factory()).lower()
        name = probe_name(digest, probe_id)
        labels = (
            "npa.nebius.com/owned=true,"
            "npa.nebius.com/purpose=sky-image-probe,"
            f"npa.nebius.com/probe-id={probe_id}"
        )
        try:
            create = runner(
                [
                    *common,
                    "run",
                    name,
                    "--restart=Never",
                    f"--image={immutable}",
                    f"--labels={labels}",
                    "--",
                    "/bin/sh",
                    "-c",
                    script,
                ],
                env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _probe_evidence(
                immutable,
                digest,
                state="indeterminate",
                cleanup="not_applicable",
                detail=f"probe creation failed: {type(exc).__name__}: {exc}",
            )
        if create.returncode == 0:
            break
        detail = (create.stderr or create.stdout).strip()
        if "alreadyexists" not in detail.replace(" ", "").lower():
            return _probe_evidence(
                immutable,
                digest,
                state="indeterminate",
                cleanup="not_applicable",
                detail=detail[:500],
            )
    if create is None or create.returncode != 0:
        return _probe_evidence(
            immutable,
            digest,
            state="indeterminate",
            cleanup="not_applicable",
            detail="probe name collisions exhausted the bounded retry",
        )

    identity, identity_error = _read_owned_probe_identity(
        common=common,
        name=name,
        probe_id=probe_id,
        immutable=immutable,
        runner=runner,
        env=env,
    )
    if identity is None:
        return _probe_evidence(
            immutable,
            digest,
            state="indeterminate",
            cleanup="refused_identity_mismatch",
            detail=identity_error,
        )

    primary = ""
    interrupted: BaseException | None = None
    try:
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
    except BaseException as exc:  # cleanup must run even on operator interruption
        interrupted = exc
        primary = f"probe wait interrupted: {type(exc).__name__}"

    current, current_error = _read_owned_probe_identity(
        common=common,
        name=name,
        probe_id=probe_id,
        immutable=immutable,
        runner=runner,
        env=env,
        expected_uid=identity,
    )
    if current is None:
        cleanup = "refused_identity_mismatch"
        cleanup_detail = current_error
    else:
        try:
            delete = runner(
                [
                    *common,
                    "delete",
                    "pod",
                    name,
                    "--ignore-not-found=true",
                    "--wait=true",
                ],
                env,
            )
            cleanup = "verified" if delete.returncode == 0 else "failed"
            cleanup_detail = (delete.stderr or delete.stdout).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup = "failed"
            cleanup_detail = f"{type(exc).__name__}: {exc}"
    if interrupted is not None:
        if cleanup != "verified":
            note = f"probe cleanup={cleanup}: {cleanup_detail[:300]}"
            add_note = getattr(interrupted, "add_note", None)
            if callable(add_note):
                add_note(note)
            else:  # Python 3.10 compatibility; preserve the primary exception.
                setattr(interrupted, "__npa_cleanup_note__", note)
        raise interrupted
    if cleanup != "verified":
        return ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="indeterminate",
            source="ephemeral_capability_probe",
            cleanup=cleanup,
            detail=f"exact probe cleanup could not be verified: {cleanup_detail[:400]}",
        )
    if primary:
        return ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="incompatible",
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


def _probe_evidence(
    immutable: str,
    digest: str,
    *,
    state: str,
    cleanup: str,
    detail: str,
) -> ImageContractEvidence:
    return ImageContractEvidence(
        image=immutable,
        digest=digest,
        contract_version=CONTRACT_VERSION,
        state=state,
        source="ephemeral_capability_probe",
        cleanup=cleanup,
        detail=str(detail or "")[:500],
    )


def _read_owned_probe_identity(
    *,
    common: list[str],
    name: str,
    probe_id: str,
    immutable: str,
    runner: Runner,
    env: Mapping[str, str],
    expected_uid: str = "",
) -> tuple[str | None, str]:
    """Read and verify the immutable identity of exactly this caller's pod."""

    try:
        result = runner([*common, "get", "pod", name, "-o", "json"], env)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"probe identity read failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return None, "probe identity could not be read after creation"
    try:
        payload = json.loads(result.stdout)
        metadata = payload["metadata"]
        labels = metadata["labels"]
        uid = str(metadata["uid"])
        containers = payload["spec"]["containers"]
        actual_image = str(containers[0]["image"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return None, f"probe identity response is invalid: {type(exc).__name__}"
    if (
        str(metadata.get("name") or "") != name
        or str(labels.get("npa.nebius.com/owned") or "") != "true"
        or str(labels.get("npa.nebius.com/purpose") or "") != "sky-image-probe"
        or str(labels.get("npa.nebius.com/probe-id") or "") != probe_id
        or not uid
        or actual_image != immutable
        or (expected_uid and uid != expected_uid)
    ):
        return None, "probe ownership or immutable pod identity did not match this caller"
    return uid, ""


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
