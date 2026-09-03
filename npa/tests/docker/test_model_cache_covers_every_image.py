"""Every image's weight cache must be one the durable cache can redirect.

The cache works the same way for every image precisely because
``MODEL_CACHE_LAYOUT`` names every variable a workbench runtime reads. That is a
list someone has to remember to extend, and forgetting is invisible: the image
keeps working, and quietly re-downloads its weights on every run while the stage
next to it hits the cache. So the Dockerfiles are the source of truth here, and a
new cache-shaped variable has to be either redirected or explicitly excused.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from npa.workbench.model_cache import MODEL_CACHE_ENV_NAMES

DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker" / "workbench"

# Anything whose name looks like it could point at downloaded model bytes.
_CACHEISH = r"[A-Z][A-Z0-9_]*(?:CACHE|_HOME|MODEL_DIR|MODELS|WEIGHTS)[A-Z0-9_]*"
CANDIDATE = re.compile(rf"^\s*(?:ENV\s+)?({_CACHEISH})=(\S+)", re.M)
# `ENV NAME value`, the other spelling Docker accepts.
CANDIDATE_SPACE_FORM = re.compile(rf"^\s*ENV\s+({_CACHEISH})\s+(\S+)", re.M)

# Excused, with the reason each one is not a weight cache. Keep this specific:
# a bare "it is fine" entry is how the next real cache gets missed.
EXCUSED: dict[str, str] = {
    # Build-time package manager switches, not runtime caches.
    "PIP_NO_CACHE_DIR": "pip build flag",
    "UV_NO_CACHE": "uv build flag",
    "UV_CACHE_DIR": "wheel cache, not model weights",
    "PIP_CACHE_DIR": "wheel cache, not model weights",
    # Where the tool is installed, not where its weights land.
    "COSMOS_HOME": "install root",
    "GROOT_HOME": "install root",
    "SONIC_HOME": "install root",
    "NPA_GENESIS_HOME": "install root",
    "CUDA_HOME": "toolkit root",
    "JAVA_HOME": "toolkit root",
    "PYTHONHOME": "interpreter root",
    # Application state and viewer config, not runtime-fetched weights.
    "FIFTYONE_HOME": "dataset app state",
    "XDG_CONFIG_HOME": "viewer config",
    "XDG_DATA_HOME": "viewer state",
    # Generic XDG cache root (fontconfig, matplotlib, uv). Every library that
    # downloads weights is pointed at an explicit variable above, so nothing
    # model-sized reaches its fallback here.
    "XDG_CACHE_HOME": "generic cache root, not a weight fallback in practice",
    # Runtime dependency closures, deliberately a separate volume: they are
    # verified wheel sets warmed once and consumed read-only, not weights that
    # accumulate. See docs/workbench/model-weight-cache.md.
    "NPA_ISAAC_CACHE_DIR": "Isaac wheel closure (warm-isaac-cache.yaml)",
    "NPA_LTX_RUNTIME_CACHE": "CUDA wheel closure",
    "NPA_WAN_RUNTIME_CACHE": "CUDA wheel closure",
    # Per-tool data mounts that the VM deploy already bind-mounts from the host,
    # so they outlive the container by their own mechanism.
    "COSMOS_DATA_HOME": "host-mounted data dir (deploy_cosmos)",
    "COSMOS_MODEL_DIR": "host-mounted data dir (deploy_cosmos)",
    "GROOT_MODEL_DIR": "host-mounted data dir (deploy_groot)",
}


def _declared_cache_variables() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    # Dockerfile.* too: an image can split its build across several, and a cache
    # declared in the one this did not read would be exactly as invisible.
    for dockerfile in sorted(DOCKER_DIR.glob("*/Dockerfile*")):
        text = dockerfile.read_text(encoding="utf-8")
        for pattern in (CANDIDATE, CANDIDATE_SPACE_FORM):
            for name, _value in pattern.findall(text):
                found.setdefault(name, set()).add(dockerfile.parent.name)
    return found


def test_every_cache_variable_an_image_bakes_is_redirected_or_excused() -> None:
    unaccounted = {
        name: sorted(images)
        for name, images in _declared_cache_variables().items()
        if name not in MODEL_CACHE_ENV_NAMES and name not in EXCUSED
    }

    assert not unaccounted, (
        "these images bake a cache-shaped variable the durable model cache does not "
        f"redirect: {unaccounted}. Add it to MODEL_CACHE_LAYOUT so its downloads land "
        "in the shared cache, or excuse it in EXCUSED with the reason it is not "
        "runtime-fetched model weights."
    )


def test_the_excuse_list_does_not_rot() -> None:
    """An excuse for a variable no image sets any more is a stale claim."""

    declared = set(_declared_cache_variables())
    # Excuses for variables set outside the Dockerfiles (toolkit roots inherited
    # from base images) are legitimate; only flag ones this repo used to declare.
    stale = {
        name
        for name in EXCUSED
        if name.startswith(("NPA_", "COSMOS_", "GROOT_", "SONIC_", "FIFTYONE_"))
        and name not in declared
    }

    assert not stale, f"excused variables no image declares any more: {sorted(stale)}"


@pytest.mark.parametrize("name", sorted(MODEL_CACHE_ENV_NAMES))
def test_no_variable_is_both_redirected_and_excused(name: str) -> None:
    assert name not in EXCUSED, f"{name} is redirected; the excuse contradicts it"


# Pod-local volumes are how the original bug looked in a manifest: a cache mount
# that dies with the pod. Each one still in the tree has to say why it is not
# runtime-fetched model weights.
EXCUSED_EMPTY_DIRS = {
    "rrd-data": "Rerun recordings written by the run, not downloaded weights",
    "fiftyone-data": "dataset app state",
    "openpi-cache": "fallback when no durable cache is configured; redirected when one is",
    "leisaac-cache": "fallback when no durable cache is configured; redirected when one is",
    "isaac-cache": "Isaac wheel closure; warm-isaac-cache.yaml is its shared volume",
    "tmp": "scratch space",
    "shm": "/dev/shm, sized for the renderer",
    "npa-sudo-shim": "standard init-container sudo shim (tiny script), never a weights cache",
    "dshm": "/dev/shm sized for the NRE renderer/reconstruction (medium Memory)",
}

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "npa"


def test_no_new_pod_local_cache_volume_appears_unnoticed() -> None:
    named = re.compile(r'\{\s*"name":\s*"([\w.-]+)",\s*\n?\s*"emptyDir"', re.M)
    found: dict[str, str] = {}
    for source in SRC_ROOT.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        if '"emptyDir"' not in text:
            continue
        for name in named.findall(text):
            found[name] = source.relative_to(SRC_ROOT).as_posix()
        if not named.findall(text):
            found[f"<unnamed in {source.name}>"] = source.relative_to(SRC_ROOT).as_posix()

    unaccounted = {
        name: where for name, where in found.items() if name not in EXCUSED_EMPTY_DIRS
    }

    assert not unaccounted, (
        f"pod-local volumes with no stated purpose: {unaccounted}. If one holds "
        "runtime-fetched weights, point it at the durable cache instead; if not, "
        "excuse it in EXCUSED_EMPTY_DIRS with the reason."
    )
