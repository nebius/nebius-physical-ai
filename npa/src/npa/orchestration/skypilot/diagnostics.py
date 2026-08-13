"""Classify SkyPilot output that would otherwise fail silently or retry forever.

SkyPilot surfaces several failures as an endlessly retrying controller loop or as
a job that sits in ``PENDING``. Both look identical to an operator watching a
quiet terminal, so this module turns the handful of signatures NPA has actually
hit into a named diagnosis with a one-line remedy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SkyPilotDiagnosis:
    """A recognized SkyPilot failure mode and how to get out of it."""

    code: str
    summary: str
    remedy: str

    def render(self) -> str:
        return f"{self.summary}\nSuggested action: {self.remedy}"


_KUBERNETES_CLIENT_POD_CONFIG = SkyPilotDiagnosis(
    code="kubernetes_client_pod_config",
    summary=(
        "SkyPilot rejected every pod_config because its kubernetes client is too new "
        "(client 36+ renamed the generated openapi type names, so SkyPilot tries to "
        "import a module named 'kubernetes.client.models.dict[str, str]'). The "
        "managed-jobs controller retries this forever, so submit appears to hang."
    ),
    remedy=(
        "rerun `npa skypilot bootstrap` to re-pin the client, then resubmit "
        "(bootstrap installs 'kubernetes>=20.0.0,!=32.0.0,<36')."
    ),
)

_REGISTRY_PULL_FORBIDDEN = SkyPilotDiagnosis(
    code="registry_pull_forbidden",
    summary=(
        "A worker pod could not pull its container image: the registry answered "
        "403 Forbidden. The job stays in PENDING/ImagePullBackOff instead of failing, "
        "because Kubernetes retries image pulls indefinitely."
    ),
    remedy=(
        "run `npa workbench workflow preflight-images <spec.yaml>` to reproduce the pull "
        "with the exact credentials the run injects, then grant the run's service account "
        "pull access to that registry (a readable tag list is not enough)."
    ),
)

_IMAGE_PULL_BACKOFF = SkyPilotDiagnosis(
    code="image_pull_backoff",
    summary=(
        "A worker pod is stuck in ImagePullBackOff/ErrImagePull, so the job stays "
        "PENDING rather than failing."
    ),
    remedy=(
        "run `npa workbench workflow preflight-images <spec.yaml>` to check the image "
        "reference and pull credentials from the target cluster's point of view."
    ),
)

_ACCELERATOR_UNSATISFIABLE = SkyPilotDiagnosis(
    code="accelerator_unsatisfiable",
    summary=(
        "SkyPilot could not satisfy the requested accelerator on this cluster. "
        "Kubernetes clusters advertise the GPU product string from their node labels, "
        "which rarely matches the short marketing name used in workflow specs."
    ),
    remedy=(
        "run `npa workbench workflow gpus --cluster <name>` to print the accelerator names "
        "this cluster actually advertises, then export "
        "NPA_WORKFLOW_GPU_ACCELERATOR=<exact-name>:<count-per-node>."
    ),
)


def _matches_kubernetes_client_pod_config(text: str) -> bool:
    lowered = text.lower()
    if "kubernetes.client.models" not in lowered:
        return False
    return "invalid pod_config" in lowered or "no module named" in lowered


def _matches_registry_pull_forbidden(text: str) -> bool:
    lowered = text.lower()
    if "403" not in lowered and "forbidden" not in lowered:
        return False
    return any(
        needle in lowered
        for needle in ("pull", "image", "manifest", "registry", "denied")
    )


def _matches_image_pull_backoff(text: str) -> bool:
    lowered = text.lower()
    return "imagepullbackoff" in lowered or "errimagepull" in lowered


_ACCELERATOR_PATTERN = re.compile(
    r"(no .{0,40}(gpu|accelerator)|accelerator[s]? .{0,40}(not (available|found|supported))"
    r"|failed_prechecks|no resources satisfy)",
    re.IGNORECASE,
)


def _matches_accelerator_unsatisfiable(text: str) -> bool:
    return bool(_ACCELERATOR_PATTERN.search(text))


_MATCHERS: tuple[tuple[SkyPilotDiagnosis, object], ...] = (
    (_KUBERNETES_CLIENT_POD_CONFIG, _matches_kubernetes_client_pod_config),
    (_REGISTRY_PULL_FORBIDDEN, _matches_registry_pull_forbidden),
    (_IMAGE_PULL_BACKOFF, _matches_image_pull_backoff),
    (_ACCELERATOR_UNSATISFIABLE, _matches_accelerator_unsatisfiable),
)


def diagnose_skypilot_output(text: str) -> SkyPilotDiagnosis | None:
    """Return the first recognized failure mode in ``text``, if any.

    Matchers are ordered most-specific first so that, for example, a pod_config
    failure is not reported as a generic accelerator problem.
    """

    if not text or not text.strip():
        return None
    for diagnosis, matches in _MATCHERS:
        if matches(text):  # type: ignore[operator]
            return diagnosis
    return None


def diagnose_all(lines: Iterable[str]) -> list[SkyPilotDiagnosis]:
    """Return every distinct diagnosis found across ``lines``, in first-seen order."""

    seen: dict[str, SkyPilotDiagnosis] = {}
    for line in lines:
        diagnosis = diagnose_skypilot_output(line)
        if diagnosis is not None and diagnosis.code not in seen:
            seen[diagnosis.code] = diagnosis
    return list(seen.values())
