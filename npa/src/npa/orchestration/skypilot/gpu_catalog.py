"""Nebius VM GPU catalog discovery for SkyPilot-backed launches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
import subprocess
import time

from npa.orchestration.skypilot._bin import SkyBin, resolve_sky_bin

DEFAULT_CATALOG_DISCOVERY_ATTEMPTS = 3
DEFAULT_CATALOG_DISCOVERY_BACKOFF_SECONDS = 1.0
DEFAULT_CATALOG_DISCOVERY_TIMEOUT_SECONDS = 120


class NebiusGpuCatalogError(RuntimeError):
    """Raised when the live SkyPilot Nebius GPU catalog cannot be discovered."""


class InvalidNebiusGpuRequestError(ValueError):
    """Raised when no requested Nebius VM GPU candidate is valid."""


@dataclass(frozen=True)
class AcceleratorRequest:
    """A quantity-aware SkyPilot accelerator request."""

    name: str
    quantity: int

    @property
    def spec(self) -> str:
        """Return the SkyPilot accelerator spec."""

        return f"{self.name}:{self.quantity}"


@dataclass(frozen=True)
class NebiusGpuCatalog:
    """Accepted Nebius VM accelerators and quantities from `sky show-gpus`."""

    quantities_by_accelerator: dict[str, frozenset[int]]
    raw_output: str = ""

    def canonicalize(self, request: AcceleratorRequest) -> AcceleratorRequest | None:
        """Return a catalog-cased request when name and quantity are supported."""

        for name, quantities in self.quantities_by_accelerator.items():
            if name.casefold() != request.name.casefold():
                continue
            if request.quantity not in quantities:
                return None
            return AcceleratorRequest(name=name, quantity=request.quantity)
        return None

    def format_available(self) -> str:
        """Return a compact human-readable accelerator catalog."""

        entries = []
        for name in sorted(self.quantities_by_accelerator, key=str.casefold):
            quantities = ", ".join(str(quantity) for quantity in sorted(self.quantities_by_accelerator[name]))
            entries.append(f"{name}: {quantities}")
        return "; ".join(entries) if entries else "none"


@dataclass(frozen=True)
class NebiusGpuResolution:
    """Validated Nebius VM GPU preferences."""

    selected: str
    accelerators: tuple[str, ...]
    rejected: tuple[str, ...]
    catalog: NebiusGpuCatalog


def parse_nebius_gpu_catalog(output: str) -> NebiusGpuCatalog:
    """Parse `sky show-gpus --cloud nebius` output into accepted quantities."""

    quantities_by_accelerator: dict[str, frozenset[int]] = {}
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line or "AVAILABLE_QUANTITIES" in line or line.startswith(("WARNING:", "Hint:", "The --")):
            continue
        columns = re.split(r"\s{2,}", line)
        if len(columns) < 2:
            continue
        name = columns[0].strip()
        quantities = frozenset(int(value) for value in re.findall(r"\d+", columns[-1]))
        if not name or not quantities:
            continue
        quantities_by_accelerator[name] = quantities
    return NebiusGpuCatalog(quantities_by_accelerator=quantities_by_accelerator, raw_output=output)


def discover_nebius_gpu_catalog(
    *,
    sky_bin: SkyBin = None,
    attempts: int = DEFAULT_CATALOG_DISCOVERY_ATTEMPTS,
    backoff_seconds: float = DEFAULT_CATALOG_DISCOVERY_BACKOFF_SECONDS,
    timeout: int = DEFAULT_CATALOG_DISCOVERY_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> NebiusGpuCatalog:
    """Query SkyPilot for the live Nebius VM GPU catalog with retries."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    sky_executable = str(resolve_sky_bin(sky_bin))
    cmd = [sky_executable, "show-gpus", "--cloud", "nebius"]
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
        else:
            output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            if result.returncode == 0:
                catalog = parse_nebius_gpu_catalog(output)
                if catalog.quantities_by_accelerator:
                    return catalog
                errors.append(f"attempt {attempt}: SkyPilot returned an empty Nebius GPU catalog")
            else:
                detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
                errors.append(f"attempt {attempt}: {detail}")
        if attempt < attempts:
            sleep(backoff_seconds * attempt)

    detail = "; ".join(errors[-attempts:])
    raise NebiusGpuCatalogError(
        "Unable to discover Nebius VM GPU catalog via `sky show-gpus --cloud nebius` "
        f"after {attempts} attempt(s): {detail}"
    )


def resolve_nebius_gpu_preferences(
    gpu: str = "",
    gpu_failover: str = "",
    *,
    catalog: NebiusGpuCatalog | None = None,
    sky_bin: SkyBin = None,
    attempts: int = DEFAULT_CATALOG_DISCOVERY_ATTEMPTS,
    backoff_seconds: float = DEFAULT_CATALOG_DISCOVERY_BACKOFF_SECONDS,
    timeout: int = DEFAULT_CATALOG_DISCOVERY_TIMEOUT_SECONDS,
) -> NebiusGpuResolution:
    """Return only Nebius VM GPU candidates accepted by the live catalog."""

    resolved_catalog = catalog or discover_nebius_gpu_catalog(
        sky_bin=sky_bin,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        timeout=timeout,
    )
    valid: list[str] = []
    rejected: list[str] = []
    for raw_candidate in _candidate_tokens(gpu, gpu_failover):
        try:
            request = parse_accelerator_request(raw_candidate)
        except ValueError as exc:
            rejected.append(f"{raw_candidate}: {exc}")
            continue
        canonical = resolved_catalog.canonicalize(request)
        if canonical is None:
            rejected.append(f"{request.spec}: not in Nebius VM catalog")
            continue
        if canonical.spec not in valid:
            valid.append(canonical.spec)

    if valid:
        return NebiusGpuResolution(
            selected=valid[0],
            accelerators=tuple(valid),
            rejected=tuple(rejected),
            catalog=resolved_catalog,
        )

    requested = ", ".join(_candidate_tokens(gpu, gpu_failover)) or "none"
    raise InvalidNebiusGpuRequestError(
        "No requested Nebius VM GPU candidate is valid. "
        f"Requested: {requested}. "
        f"Currently valid Nebius VM accelerators: {resolved_catalog.format_available()}."
    )


def parse_accelerator_request(candidate: str) -> AcceleratorRequest:
    """Parse a SkyPilot accelerator token, defaulting bare names to quantity 1."""

    value = str(candidate or "").strip()
    if not value:
        raise ValueError("accelerator must not be empty")
    if ":" in value:
        name, quantity_text = value.rsplit(":", 1)
    else:
        name, quantity_text = value, "1"
    name = name.strip()
    quantity_text = quantity_text.strip()
    if not name:
        raise ValueError("accelerator name must not be empty")
    if not quantity_text.isdigit():
        raise ValueError("accelerator quantity must be a positive integer")
    quantity = int(quantity_text)
    if quantity <= 0:
        raise ValueError("accelerator quantity must be positive")
    return AcceleratorRequest(name=name, quantity=quantity)


def _candidate_tokens(gpu: str = "", gpu_failover: str = "") -> list[str]:
    tokens: list[str] = []
    for raw in (gpu, gpu_failover):
        tokens.extend(token.strip() for token in str(raw or "").split(",") if token.strip())
    return tokens


def resolve_kubernetes_gpu_preferences(*_: object, **__: object) -> None:
    """Extension point for managed-Kubernetes GPU resolution.

    RTX PRO 6000 (`gpu-rtx6000`, 96 GB) is not a SkyPilot Nebius VM catalog
    accelerator. It belongs to the managed-Kubernetes path, currently in
    us-central1, where GPUs are scheduled through node labels and
    `nvidia.com/gpu` rather than Nebius VM accelerator strings. That path is
    intentionally not implemented here.
    """

    return None
