"""Locate an NVIDIA Cosmos Evaluator checkout so upstream code can be imported.

The upstream project (https://github.com/nvidia-cosmos/cosmos-evaluator,
Apache-2.0) is built with Bazel and publishes no wheel, so it can only be
imported from a source tree on ``sys.path``. This module finds such a tree —
``NPA_COSMOS_EVALUATOR_SRC`` first, then the conventional image locations — and
puts it on ``sys.path`` for the check modules.

Nothing here downloads code. When no checkout is present the check modules fall
back to the in-repo port of the same published algorithm, so a stage never
silently degrades to a stub and never fetches code at run time.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from pathlib import Path

_log = logging.getLogger(__name__)

UPSTREAM_REPO = "https://github.com/nvidia-cosmos/cosmos-evaluator"
UPSTREAM_LICENSE = "Apache-2.0"
SRC_ENV = "NPA_COSMOS_EVALUATOR_SRC"

# Conventional locations a workbench image can bake an upstream checkout into.
DEFAULT_SRC_CANDIDATES = (
    "/opt/cosmos-evaluator",
    "/opt/nvidia/cosmos-evaluator",
)


class CosmosEvaluatorError(RuntimeError):
    """Raised when a Cosmos Evaluator request cannot be satisfied."""


class CosmosEvaluatorStorageError(CosmosEvaluatorError):
    """Raised when object storage could not answer, as opposed to having no object.

    A variant that is simply absent is a legitimate skip and scores nothing. A
    credential, endpoint, or transport failure is not: every clip would skip for a
    reason that has nothing to do with the clips, and the run would look like a
    batch that genuinely scored zero. Keeping the two apart lets the report say
    which one happened.
    """


def upstream_source_dir(*, environ: dict[str, str] | None = None) -> Path | None:
    """Return a Cosmos Evaluator checkout root, or ``None`` when there is none.

    A directory counts as a checkout when it holds the upstream ``checks``
    package, which is the import root for every checker.
    """

    env = os.environ if environ is None else environ
    explicit = str(env.get(SRC_ENV, "") or "").strip()
    candidates = [explicit] if explicit else list(DEFAULT_SRC_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "checks" / "__init__.py").is_file() or (root / "checks").is_dir():
            return root
    return None


def ensure_upstream_importable(*, environ: dict[str, str] | None = None) -> Path | None:
    """Put an upstream checkout on ``sys.path`` and return its root.

    Returns ``None`` when no checkout is available; callers then use the in-repo
    port instead of raising.
    """

    root = upstream_source_dir(environ=environ)
    if root is None:
        return None
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    _install_runfiles_alias()
    return root


def _install_runfiles_alias() -> None:
    """Expose ``python.runfiles`` so upstream's config loader imports.

    Upstream resolves its bundled ``cosmos_evaluator.yaml`` through Bazel
    runfiles, importing ``python.runfiles`` — the module path Bazel's
    ``rules_python`` provides inside a build. The published ``bazel-runfiles``
    wheel installs the same code as a top-level ``runfiles`` module, so aliasing
    it under the Bazel path lets the checkers import outside Bazel. Callers pass
    an explicit ``config_dir``, so the runfiles lookup is never exercised.
    """

    if "python.runfiles" in sys.modules:
        return
    try:
        runfiles = importlib.import_module("runfiles")
    except ImportError:
        return
    package = sys.modules.get("python")
    if package is None:
        package = types.ModuleType("python")
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["python"] = package
    sys.modules["python.runfiles"] = runfiles
    setattr(package, "runfiles", runfiles)
