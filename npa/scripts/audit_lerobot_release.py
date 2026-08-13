#!/usr/bin/env python
"""Check a LeRobot release against the exact surface this repo binds to.

Every LeRobot release note says "some import paths have changed". That sentence
cannot be actioned by reading it: the only question that matters here is whether
*the symbols, CLI entry points, flags, and dependency bounds this repo actually
uses* still hold in the candidate release. This answers that mechanically, from
the published wheel, before anyone builds a multi-GB CUDA image.

    npa/.venv/bin/python npa/scripts/audit_lerobot_release.py 0.6.1
    npa/.venv/bin/python npa/scripts/audit_lerobot_release.py 0.6.1 --baseline 0.5.1
    npa/.venv/bin/python npa/scripts/audit_lerobot_release.py 0.6.1 --json

Wheels are cached under ``--cache-dir`` (default ``.tmp-lerobot-wheels``), so a
second run is offline. ``--offline`` refuses to reach PyPI at all and fails when
the wheel is not already cached.

What it does and does not prove
-------------------------------
It reads the wheel's Python source with ``ast``; it never imports LeRobot, so it
runs on any machine with no CUDA, no torch, and no GPU. That makes it a fast
*pre-flight*: a PASS means the upgrade is not blocked by a renamed module,
removed symbol, dropped parameter, renamed CLI flag, changed dataset
``CODEBASE_VERSION``, or a torch/torchvision/diffusers bound that contradicts a
pin this repo forces. It cannot prove runtime behaviour. A real GPU train+eval
smoke (``npa.smoke.test_lerobot_functional``) remains the merge gate.

The surface below is a reviewed list with call-site provenance, not a scrape.
When a new call site starts using LeRobot, add it here too.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / ".tmp-lerobot-wheels"
PYPI_JSON = "https://pypi.org/pypi/lerobot/{version}/json"

# ── The surface this repo binds to ──────────────────────────────────────────
# (module, symbol, call-site provenance). symbol=None checks the module only.
IMPORT_SURFACE: tuple[tuple[str, str | None, str], ...] = (
    ("lerobot.datasets.lerobot_dataset", "LeRobotDataset", "npa/workflows/lerobot_dataset.py"),
    ("lerobot.datasets", "LeRobotDataset", "npa/setup/install_lerobot.sh"),
    ("lerobot.datasets.factory", "make_dataset", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.datasets.sampler", "EpisodeAwareSampler", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.configs.policies", "PreTrainedConfig", "npa/server/app.py, npa/genesis/eval_student.py"),
    ("lerobot.configs.train", "TrainPipelineConfig", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.configs.default", "DatasetConfig", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.configs.default", "EvalConfig", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.configs.default", "WandBConfig", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.policies.factory", "make_policy", "npa/server/app.py"),
    ("lerobot.policies.factory", "make_pre_post_processors", "npa/server/app.py, npa/genesis/eval_student.py"),
    ("lerobot.policies.utils", "prepare_observation_for_inference", "npa/server/app.py"),
    ("lerobot.policies.act.modeling_act", "ACTPolicy", "npa/genesis/eval_student.py"),
    ("lerobot.policies.act.configuration_act", "ACTConfig", "npa/smoke/test_lerobot_env.py"),
    ("lerobot.policies.diffusion.modeling_diffusion", "DiffusionPolicy", "npa/genesis/eval_student.py"),
    ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy", "npa/genesis/eval_student.py"),
    ("lerobot.optim.factory", "make_optimizer_and_scheduler", "research/lerobot-deploy/training/profile_train.py"),
    ("lerobot.utils.device_utils", "get_safe_torch_device", "npa/server/app.py"),
    ("lerobot.envs.configs", "EnvConfig", "npa/server/app.py"),
)

# Parameters this repo passes by keyword. Upstream may append parameters freely;
# it may not remove one of these without breaking a call site.
CALLABLE_PARAMS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("lerobot/policies/factory.py", "make_policy", ("cfg", "env_cfg", "ds_meta"), "npa/server/app.py"),
    (
        "lerobot/policies/factory.py",
        "make_pre_post_processors",
        ("policy_cfg", "pretrained_path"),
        "npa/server/app.py",
    ),
    (
        "lerobot/policies/utils.py",
        "prepare_observation_for_inference",
        ("observation", "device", "task", "robot_type"),
        "npa/server/app.py",
    ),
    (
        "lerobot/datasets/lerobot_dataset.py",
        "LeRobotDataset.__init__",
        ("repo_id", "root", "episodes", "revision"),
        "npa/workflows/lerobot_dataset.py",
    ),
    ("lerobot/optim/factory.py", "make_optimizer_and_scheduler", ("cfg", "policy"), "profile_train.py"),
)

# Console scripts this repo shells out to.
REQUIRED_ENTRY_POINTS = ("lerobot-train", "lerobot-eval")

# Policy types this repo trains or evaluates, and the module that implements
# each. LeRobot 0.6.x enforces optional dependencies at *construction* time
# (``require_package`` inside ``__init__``), so a policy whose extra is missing
# still imports cleanly and only fails once ``make_policy`` runs. Checking the
# import surface alone cannot see that; this maps each policy to its gates.
POLICY_MODULES: tuple[tuple[str, str, str], ...] = (
    ("act", "lerobot/policies/act/modeling_act.py", "npa/genesis/eval_student.py"),
    (
        "diffusion",
        "lerobot/policies/diffusion/modeling_diffusion.py",
        "npa/genesis/eval_student.py, golden evals",
    ),
    ("smolvla", "lerobot/policies/smolvla/modeling_smolvla.py", "npa/genesis/eval_student.py"),
)

# Dataset layout version our adapters write; a bump here means every
# npa/adapter/*_lerobot.py writer needs migrating.
EXPECTED_CODEBASE_VERSION = "v3.0"

# Versions the B300 Dockerfile force-installs regardless of LeRobot version.
# The per-version pins are not hardcoded here: they are read from this repo's
# own manifest, so this doubles as a check that the manifest is not lying.
B300_IMAGE_PINS: dict[str, str] = {
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "torchcodec": "0.8.1",
}
MANIFEST_PIN_KEYS = {
    "torch_pin": "torch",
    "torchvision_pin": "torchvision",
    "diffusers_pin": "diffusers",
}


@dataclass
class Report:
    """Collected check results for one candidate version."""

    version: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, *, warn: bool = False) -> None:
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.checks.append({"check": name, "status": status, "detail": detail})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "FAIL"]


# ── Wheel acquisition ───────────────────────────────────────────────────────


def fetch_wheel(version: str, cache_dir: Path, *, offline: bool) -> Path:
    """Return an unpacked wheel tree for ``version``, downloading if needed."""

    target = cache_dir / f"lerobot-{version}"
    if (target / "lerobot" / "__init__.py").exists():
        return target
    if offline:
        raise SystemExit(f"--offline set but no cached wheel for {version} at {target}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(PYPI_JSON.format(version=version), timeout=60) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not reach PyPI for lerobot {version}: {exc}") from exc

    url = next((u["url"] for u in payload["urls"] if u["packagetype"] == "bdist_wheel"), None)
    if url is None:
        raise SystemExit(f"lerobot {version} publishes no wheel")

    archive = cache_dir / f"lerobot-{version}.whl"
    with urllib.request.urlopen(url, timeout=300) as resp, archive.open("wb") as handle:
        shutil.copyfileobj(resp, handle)
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    return target


# ── Static inspection helpers ───────────────────────────────────────────────


def _module_file(root: Path, dotted: str) -> Path | None:
    rel = dotted.replace(".", "/")
    for candidate in (root / f"{rel}.py", root / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"))


def _top_level_names(path: Path) -> set[str]:
    """Names a module defines or re-exports (import-from counts as re-export)."""

    names: set[str] = set()
    try:
        tree = _parse(path)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def _function_params(root: Path, relpath: str, dotted: str) -> set[str] | None:
    """Return parameter names for ``dotted`` (``Class.method`` supported)."""

    path = root / relpath
    if not path.exists():
        return None
    try:
        tree = _parse(path)
    except SyntaxError:
        return None

    def collect(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        args = fn.args
        found = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        if args.kwarg:  # **kwargs absorbs anything we pass
            found.add("**")
        return found

    if "." in dotted:
        cls_name, method = dotted.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method:
                        return collect(sub)
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == dotted:
            return collect(node)
    return None


def _dataclass_fields(root: Path, relpath: str, cls_name: str) -> set[str]:
    """Return annotated attribute names on a class (draccus config fields)."""

    path = root / relpath
    if not path.exists():
        return set()
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return {
                sub.target.id
                for sub in node.body
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name)
            }
    return set()


def _dist_info(root: Path) -> Path:
    return next(root.glob("lerobot-*.dist-info"))


def _declared_bounds(root: Path) -> dict[str, str]:
    """First declared specifier per dependency we care about."""

    bounds: dict[str, str] = {}
    for line in (_dist_info(root) / "METADATA").read_text(encoding="utf-8").splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        req = Requirement(line.split(":", 1)[1].strip())
        if req.name in {"torch", "torchvision", "diffusers", "torchcodec"}:
            bounds.setdefault(req.name, str(req.specifier))
    return bounds


def _extra_requirements(root: Path) -> tuple[dict[str, list[Requirement]], list[Requirement]]:
    """Split ``Requires-Dist`` into per-extra requirements and unconditional ones."""

    per_extra: dict[str, list[Requirement]] = {}
    base: list[Requirement] = []
    for line in (_dist_info(root) / "METADATA").read_text(encoding="utf-8").splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        req = Requirement(line.split(":", 1)[1].strip())
        extras = re.findall(r'extra\s*==\s*[\'"]([^\'"]+)[\'"]', str(req.marker or ""))
        if extras:
            for extra in extras:
                per_extra.setdefault(extra, []).append(req)
        else:
            base.append(req)
    return per_extra, base


def _extra_closure(root: Path, extras: Iterable[str]) -> set[str]:
    """Distributions installed by ``pip install lerobot[<extras>]``.

    Extras reference each other (``training`` -> ``dataset``, ``diffusion`` ->
    ``diffusers-dep``), so this walks the self-referential graph to a fixpoint.
    """

    per_extra, base = _extra_requirements(root)
    provided = {canonicalize_name(req.name) for req in base}
    queue = list(extras)
    seen: set[str] = set()
    while queue:
        extra = queue.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for req in per_extra.get(extra, []):
            if canonicalize_name(req.name) == "lerobot":
                queue.extend(req.extras)
            else:
                provided.add(canonicalize_name(req.name))
    return provided


def _require_package_gates(root: Path, relpath: str) -> list[tuple[str, str]]:
    """``(package, extra)`` pairs a module demands at construction time."""

    path = root / relpath
    if not path.exists():
        return []
    gates: list[tuple[str, str]] = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "require_package" or not node.args:
            continue
        pkg = node.args[0]
        if not isinstance(pkg, ast.Constant) or not isinstance(pkg.value, str):
            continue
        extra = next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "extra"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ),
            "",
        )
        if (pkg.value, extra) not in gates:
            gates.append((pkg.value, extra))
    return gates


def _entry_points(root: Path) -> set[str]:
    path = _dist_info(root) / "entry_points.txt"
    if not path.exists():
        return set()
    return {
        line.split("=", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("[")
    }


def _requires_python(root: Path) -> str:
    for line in (_dist_info(root) / "METADATA").read_text(encoding="utf-8").splitlines():
        if line.startswith("Requires-Python:"):
            return line.split(":", 1)[1].strip()
    return ""


# ── Checks ──────────────────────────────────────────────────────────────────


def check_imports(root: Path, report: Report) -> None:
    missing = []
    for module, symbol, provenance in IMPORT_SURFACE:
        path = _module_file(root, module)
        if path is None:
            missing.append(f"{module} (module gone) <- {provenance}")
        elif symbol and symbol not in _top_level_names(path):
            missing.append(f"{module}:{symbol} (symbol gone) <- {provenance}")
    total = len(IMPORT_SURFACE)
    report.add(
        "import-surface",
        not missing,
        f"{total - len(missing)}/{total} bindings resolve"
        + ("; missing: " + ", ".join(missing) if missing else ""),
    )


def check_signatures(root: Path, report: Report) -> None:
    problems = []
    for relpath, dotted, params, provenance in CALLABLE_PARAMS:
        actual = _function_params(root, relpath, dotted)
        if actual is None:
            problems.append(f"{dotted} not found in {relpath} <- {provenance}")
            continue
        if "**" in actual:
            continue
        dropped = sorted(p for p in params if p not in actual)
        if dropped:
            problems.append(f"{dotted} dropped {dropped} <- {provenance}")
    report.add(
        "callable-parameters",
        not problems,
        "all keyword parameters still accepted" if not problems else "; ".join(problems),
    )


def check_entry_points(root: Path, report: Report) -> None:
    available = _entry_points(root)
    missing = [ep for ep in REQUIRED_ENTRY_POINTS if ep not in available]
    extra = sorted(available - set(REQUIRED_ENTRY_POINTS))
    report.add(
        "console-entry-points",
        not missing,
        f"required present ({', '.join(REQUIRED_ENTRY_POINTS)})" if not missing
        else f"missing {missing}",
    )
    report.add("available-entry-points", True, ", ".join(extra) or "none")


def check_policy_extras(root: Path, report: Report, manifest_entry: dict[str, Any] | None) -> None:
    """Do the manifest's extras actually let every policy we use be constructed?"""

    if manifest_entry is None:
        report.add(
            "policy-extras",
            True,
            "version not in lerobot_version_manifest.json -- no extras declared yet",
        )
        return

    declared = [e.strip() for e in str(manifest_entry.get("pip_extras", "")).split(",") if e.strip()]
    provided = _extra_closure(root, declared)

    problems: list[str] = []
    satisfied: list[str] = []
    for policy, relpath, provenance in POLICY_MODULES:
        missing = [
            f"{pkg} (add extra '{extra}')"
            for pkg, extra in _require_package_gates(root, relpath)
            if canonicalize_name(pkg) not in provided
        ]
        if missing:
            problems.append(f"--policy.type={policy} needs {', '.join(missing)} <- {provenance}")
        else:
            satisfied.append(policy)

    report.add(
        "policy-extras",
        not problems,
        f"lerobot[{','.join(declared)}] constructs {satisfied or 'no policies'}"
        if not problems
        else "; ".join(problems)
        + " -- these gates fire in __init__, so the policy imports and only"
        " fails when make_policy() runs",
    )


def check_dataset_format(root: Path, report: Report) -> None:
    path = root / "lerobot" / "datasets" / "dataset_metadata.py"
    found = ""
    if path.exists():
        for node in _parse(path).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CODEBASE_VERSION" for t in node.targets
            ):
                if isinstance(node.value, ast.Constant):
                    found = str(node.value.value)
    ok = found == EXPECTED_CODEBASE_VERSION
    report.add(
        "dataset-codebase-version",
        ok,
        f"{found or 'not found'} (adapters write {EXPECTED_CODEBASE_VERSION})"
        + ("" if ok else " -- npa/adapter/*_lerobot.py writers need migrating"),
    )


def check_cli_flags(root: Path, report: Report, manifest_entry: dict[str, Any] | None) -> None:
    train_fields = _dataclass_fields(root, "lerobot/configs/train.py", "TrainPipelineConfig")
    policy_fields = _dataclass_fields(root, "lerobot/configs/policies.py", "PreTrainedConfig")

    present = sorted(f for f in ("eval_freq", "env_eval_freq") if f in train_fields)
    report.add(
        "train-env-eval-flag",
        bool(present),
        f"TrainPipelineConfig exposes {present}" if present else "neither eval_freq nor env_eval_freq found",
    )
    report.add(
        "eval-checkpoint-flag",
        "pretrained_path" in policy_fields,
        "PreTrainedConfig.pretrained_path present (--policy.pretrained_path)"
        if "pretrained_path" in policy_fields
        else "PreTrainedConfig.pretrained_path missing",
    )

    parser = root / "lerobot" / "configs" / "parser.py"
    has_path_key = parser.exists() and 'PATH_KEY = "path"' in parser.read_text(encoding="utf-8")
    report.add(
        "policy-path-alias",
        has_path_key,
        'parser.PATH_KEY == "path" (--policy.path)' if has_path_key else "PATH_KEY alias changed",
    )

    if manifest_entry:
        declared = str(manifest_entry.get("train_env_eval_flag", ""))
        ok = declared in train_fields
        report.add(
            "manifest-agrees-with-wheel",
            ok,
            f"manifest declares train_env_eval_flag={declared!r}, wheel {'has' if ok else 'LACKS'} it",
        )


def _pin_floor(spec: str) -> str:
    """Lowest version a pin like ``torch==2.12.1`` or ``diffusers>=0.38.0`` allows."""

    return spec.split("==")[-1].split(">=")[-1].strip()


def _conflicts(pins: dict[str, str], bounds: dict[str, str]) -> list[str]:
    problems = []
    for package, pinned in pins.items():
        declared = bounds.get(package)
        if declared is None:
            continue
        if Version(pinned) not in Requirement(f"{package}{declared}").specifier:
            problems.append(f"{package}=={pinned} violates {declared}")
    return problems


def check_dependency_bounds(
    root: Path, report: Report, manifest_entry: dict[str, Any] | None
) -> None:
    bounds = _declared_bounds(root)
    report.add("requires-python", True, _requires_python(root) or "unspecified")
    report.add("declared-bounds", True, ", ".join(f"{k}{v}" for k, v in sorted(bounds.items())))

    problems = _conflicts(B300_IMAGE_PINS, bounds)
    report.add(
        "b300-image torch stack",
        not problems,
        "torch 2.9.0 / torchvision 0.24.0 / torchcodec 0.8.1 satisfy declared bounds"
        if not problems
        else "; ".join(problems),
    )

    if manifest_entry is None:
        report.add(
            "manifest torch pins",
            True,
            "version not in lerobot_version_manifest.json -- nothing pinned yet",
            warn=True,
        )
        return

    pins = {
        package: _pin_floor(str(manifest_entry[key]))
        for key, package in MANIFEST_PIN_KEYS.items()
        if manifest_entry.get(key)
    }
    if not pins:
        report.add(
            "manifest torch pins",
            True,
            "manifest forces no torch pins for this version (upstream resolver decides)",
        )
        return

    problems = _conflicts(pins, bounds)
    report.add(
        "manifest torch pins",
        not problems,
        f"{pins} satisfy declared bounds" if not problems
        else "; ".join(problems) + " -- these are force-installed after `pip install lerobot`,"
        " so pip does not re-resolve and the image ships broken",
    )


def load_manifest_entry(version: str) -> dict[str, Any] | None:
    path = REPO_ROOT / "npa" / "src" / "npa" / "deploy" / "lerobot_version_manifest.json"
    if not path.exists():
        return None
    versions = json.loads(path.read_text(encoding="utf-8")).get("versions", {})
    entry = versions.get(version)
    return entry if isinstance(entry, dict) else None


def audit(version: str, cache_dir: Path, *, offline: bool) -> Report:
    root = fetch_wheel(version, cache_dir, offline=offline)
    manifest_entry = load_manifest_entry(version)
    report = Report(version=version)
    check_imports(root, report)
    check_signatures(root, report)
    check_entry_points(root, report)
    check_dataset_format(root, report)
    check_policy_extras(root, report, manifest_entry)
    check_cli_flags(root, report, manifest_entry)
    check_dependency_bounds(root, report, manifest_entry)
    return report


def render(report: Report) -> None:
    print(f"\nLeRobot {report.version}")
    print("-" * 78)
    for check in report.checks:
        print(f"  [{check['status']:<4}] {check['check']}: {check['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="candidate LeRobot version, e.g. 0.6.1")
    parser.add_argument("--baseline", help="also audit this version for comparison")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true", help="use cached wheels only")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    reports = [audit(args.version, args.cache_dir, offline=args.offline)]
    if args.baseline:
        reports.append(audit(args.baseline, args.cache_dir, offline=args.offline))

    if args.json:
        print(json.dumps([{"version": r.version, "checks": r.checks} for r in reports], indent=2))
    else:
        for report in reports:
            render(report)

    candidate = reports[0]
    sys.stdout.flush()
    if candidate.failed:
        print(
            f"\nFAIL: {len(candidate.failed)} blocking finding(s) for lerobot {candidate.version}.",
            file=sys.stderr,
        )
        return 1
    print(f"\nPASS: lerobot {candidate.version} satisfies this repo's static LeRobot surface.")
    print("Static pre-flight only -- a real GPU train+eval smoke is still the merge gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
