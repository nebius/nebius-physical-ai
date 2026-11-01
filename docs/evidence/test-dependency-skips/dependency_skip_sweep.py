"""Measure which skipped tests actually need the dependency they are skipped for.

A review lane found two tests in this change switched off everywhere by an
`importorskip` in a fixture they shared, for a module neither of them used. The
fix was one line. The question worth answering is how far the pattern reaches,
and that cannot be answered by reading code: whether a test needs a module is a
property of what it executes, not of what its file imports.

So this measures it. Two runs of the same test file:

`absent`
    The module is made genuinely missing -- `ModuleNotFoundError` from a meta-path
    finder, not a stub raising `ImportError`, because `importorskip` propagates
    `ImportError` on purpose so a broken install is not masked, and skips only on
    `ModuleNotFoundError`. This is what CI sees.

`poison`
    The module imports, and every attribute access on it raises. `importorskip`
    is satisfied, so the tests run; any test that *touches* the module fails on
    the first attribute it reads.

A test that skips under `absent` and passes under `poison` did not touch the module it
was skipped for through any path that raises. That is the finding. A test that fails
under `poison` needs it, and its skip is correct.

"Through any path that raises" rather than "never touched it", because the poison
raises `AssertionError`: a test that reads the module inside `except Exception`
swallows the complaint and passes. That is the one caveat here that would make a
number an overcount rather than a lower bound, so every row reports the broad
handlers in its file, and a test constructs one to show the limit is real.

The classification is deliberately one-sided. Poisoning a module that other
libraries import during collection can break collection for reasons that have
nothing to do with the test's own needs, so only a *pass* under poison counts as
evidence. The result is a lower bound on how many tests are switched off
unnecessarily, which is the safe direction for a claim of this kind.

Run from the repository root:
    python3 docs/evidence/open3d/harness/dependency_skip_sweep.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("not inside a repository checkout")


REPO = repository_root()
TESTS = REPO / "npa/tests"
PYTHON = REPO / "npa/.venv/bin/python"

OUT = Path(
    os.environ.get(
        "NPA_SWEEP_OUT",
        str(Path(__file__).resolve().parent / "dependency-skip-sweep.json"),
    )
)

PLUGIN = '''
import sys
from types import ModuleType

TARGET = {target!r}
MODE = {mode!r}


class _Blocker:
    """Absence as the import system reports it, so importorskip skips rather than errors."""

    def find_spec(self, name, path=None, target=None):
        if name == TARGET or name.startswith(TARGET + "."):
            raise ModuleNotFoundError("No module named %r" % name, name=name)
        return None


class _Poison(ModuleType):
    """Importable, and unusable. Reading any attribute is the test admitting it needs this."""

    def __getattr__(self, name):
        raise AssertionError("touched %s.%s" % (self.__name__, name))


def pytest_configure(config):
    for name in [m for m in list(sys.modules) if m == TARGET or m.startswith(TARGET + ".")]:
        del sys.modules[name]
    if MODE == "absent":
        sys.meta_path.insert(0, _Blocker())
        return

    # Poison the leaf only, and import real parents. Stubbing the parents too gave
    # this sweep a visible artifact: poisoning `pxr.UsdPhysics` replaced `pxr` with a
    # stub whose `__path__` is empty, so an unrelated fixture's
    # `importorskip("pxr.Usd")` started failing and three tests moved from passing to
    # skipped. A negative result from a measurement instrument is the instrument
    # talking about itself.
    parts = TARGET.split(".")
    if len(parts) > 1:
        import importlib

        try:
            parent = importlib.import_module(".".join(parts[:-1]))
        except ImportError:
            # Without a real parent there is no way to make the leaf importable but
            # unusable without stubbing the parent as well, which is the artifact
            # above. Say so instead of measuring something else.
            raise RuntimeError("NPA_SWEEP_PARENT_UNAVAILABLE") from None
    else:
        parent = None
    poison = _Poison(TARGET)
    poison.__path__ = []
    sys.modules[TARGET] = poison
    if parent is not None:
        setattr(parent, parts[-1], poison)
'''

OUTCOME = re.compile(r"(\d+) (passed|failed|skipped|error|errors)")
#: `-v` names every test and its outcome, which `-rA` does not: its SKIPPED lines
#: carry a file and line rather than a node identifier, so they cannot be intersected
#: with anything.
VERBOSE_OUTCOME = re.compile(
    r"^(\S+::\S+) (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)", re.MULTILINE
)


def fixture_level_sites() -> list[tuple[Path, str, list[str]]]:
    """Every `importorskip` inside a fixture, which is as broad as the file's widest need."""
    sites: list[tuple[Path, str, list[str]]] = []
    for path in sorted(TESTS.rglob("test_*.py")) + sorted(TESTS.rglob("conftest.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any("fixture" in ast.unparse(d) for d in node.decorator_list):
                continue
            modules = [
                call.args[0].value
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "importorskip"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ]
            if modules:
                sites.append((path, node.name, modules))
    return sites


def run(path: Path, target: str, mode: str) -> dict[str, int | list[str]]:
    with tempfile.TemporaryDirectory() as plugin_dir:
        name = "npa_sweep_plugin"
        (Path(plugin_dir) / f"{name}.py").write_text(
            PLUGIN.format(target=target, mode=mode)
        )
        environment = dict(os.environ, PYTHONPATH=plugin_dir)
        completed = subprocess.run(
            [
                str(PYTHON),
                "-m",
                "pytest",
                str(path if not path.is_relative_to(REPO) else path.relative_to(REPO)),
                "-v",
                "--tb=no",
                "-p",
                name,
                "-p",
                "no:cacheprovider",
                "--timeout=120",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
    tail = completed.stdout[-6000:]
    counts = {kind: 0 for kind in ("passed", "failed", "skipped", "error")}
    for number, kind in OUTCOME.findall(tail):
        counts["error" if kind.startswith("error") else kind] += int(number)
    by_outcome: dict[str, list[str]] = {}
    for node, outcome in VERBOSE_OUTCOME.findall(completed.stdout):
        by_outcome.setdefault(outcome.lower(), []).append(node)
    return {
        **counts,
        # Per test, because the criterion is about individual tests and a count is
        # not. A delta over counts cannot tell a test that moved from skipped into
        # passing from one that moved from *failed* into passing, and it nets a rise
        # against a drop to zero. Both of those were found in this sweep's published
        # numbers by a reviewer who took this reading and I had not.
        "nodes": {
            outcome: sorted(nodes) for outcome, nodes in sorted(by_outcome.items())
        },
        "parent_unavailable": "NPA_SWEEP_PARENT_UNAVAILABLE"
        in (completed.stdout + completed.stderr),
    }


OBSERVED = Path(__file__).resolve().parent / "ci-observed-packages.json"


def ci_installs(module: str) -> bool | None:
    """Whether CI has the module, read from a real CI job rather than inferred.

    Inference was tried first and was wrong. `npa/ci/requirements.txt` pins torch,
    which reads as though CI installs it, but that file is passed with `-c` and
    constrains versions without installing anything. The extras CI does install
    carry no torch at all. A real job has `torch==2.14.0+cpu`, arriving
    transitively, and the inference was one step from publishing the opposite.

    So the oracle is `ci-observed-packages.json`, which names the job it was read
    from. `None` means the snapshot does not cover this module, which is reported
    rather than guessed.
    """
    snapshot = json.loads(OBSERVED.read_text())
    root = module.split(".")[0]
    distribution = snapshot["import_name_to_distribution"].get(root)
    if distribution is None:
        return None
    if distribution in snapshot["observed_installed"]:
        return True
    if distribution in snapshot["observed_absent"]:
        return False
    return None


def broad_except_sites(path: Path) -> list[int]:
    """Line numbers of handlers that would swallow the poison's AssertionError.

    This is the one limit of the instrument that errs in the unsafe direction. A test
    that calls into the module inside `except Exception` catches the poison's
    complaint and passes, and would be recorded as a test that does not need the
    module. A reviewer demonstrated it with a purpose-built control rather than
    arguing it, and then checked that no flagged file contains such a handler, so no
    published number was affected.

    Measured per file and published beside each row, so the claim is checkable at the
    row level instead of resting on a one-time audit of three files.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    swallowing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        names: list[str] = []
        if caught is None:
            names = ["bare"]
        elif isinstance(caught, ast.Name):
            names = [caught.id]
        elif isinstance(caught, ast.Tuple):
            names = [e.id for e in caught.elts if isinstance(e, ast.Name)]
        if {"bare", "Exception", "BaseException"} & set(names):
            swallowing.append(node.lineno)
    return swallowing


def importable(module: str) -> bool:
    """Whether this environment has the module, which decides if a finding is live.

    A site that would switch off tests for a module the test runner already has is
    latently wrong rather than actively wrong: nothing is being skipped today, and
    the tests go quiet the moment the dependency is dropped. Both are worth fixing;
    only one is costing coverage now, and conflating them would overstate this.
    """
    probe = subprocess.run(
        [
            str(PYTHON),
            "-c",
            f"import importlib.util as u; raise SystemExit(0 if u.find_spec({module!r}) else 1)",
        ],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def narrative(measured: dict[str, int]) -> dict[str, str]:
    """The record's prose, as a function of its numbers.

    Kept here so a test can hand the committed record back to this function and
    require the two to agree. A sentence maintained by hand beside a generated
    number is the thing that goes stale, and a membership check against the figures
    the record already holds will wave the stale version through.
    """
    live = measured["live"]
    return {
        "what_this_found": (
            f"{measured['fixtures']} fixtures in this tree call importorskip, which "
            f"makes the skip as broad as the widest dependency of anything in their "
            f"file rather than what the tests beneath them use. Of the "
            f"{measured['measurable']} fixture-and-module pairs this environment can "
            f"decide, {measured['findings']} switch off tests that never touch the "
            f"module through any path that raises, {measured['tests']} tests in total."
        ),
        "what_it_costs_today": (
            f"{live} of those {measured['findings']} are costing coverage in CI now. "
            "CI installs the modules the others are gated on, so those tests run "
            "today and go quiet the moment one of those dependencies moves."
            if live == 0
            else f"{live} of those {measured['findings']} are costing coverage in CI "
            "now, because CI does not install the module they are gated on."
        ),
        "where_this_came_from": (
            "A review lane found two tests switched off everywhere by this pattern in "
            "PR 1011: the fixture they shared asked for Open3D to satisfy a "
            "module-scope import that only the harness's main() needed, and CI "
            "installs no Open3D. One of the two was the test validating that change's "
            "own headline reading. Blocking the import the way real absence does, "
            "that file went from 20 passed and 2 skipped to 23 passed. That site is "
            "fixed in its own branch; this measures how far the shape reaches."
        ),
        "what_this_does_not_establish": (
            "Not that the tests are good, only that they run. Not that a module is "
            "unused by the file -- several of these fixtures sit beside tests that do "
            "need the dependency, which is why the measurement is per test rather than "
            "per file. And not a complete census: a pair this environment cannot "
            "decide is reported as undecided, not as clean."
        ),
    }


def classify(absent: dict, poisoned: dict) -> dict:
    """Decide one row from the two runs, by test identity rather than by a count.

    Separate from `main` so the two ways a count delta gets this wrong can be driven
    directly by a test, instead of only through a three-minute sweep of the tree.
    """
    skipped_when_absent = set(absent["nodes"].get("skipped", []))
    passing_when_poisoned = set(poisoned["nodes"].get("passed", []))
    passing_when_absent = set(absent["nodes"].get("passed", []))

    # The criterion, applied to the tests it is about. A reviewer reproduced this
    # sweep's counts exactly and then took the one reading it had not: the single test
    # behind its `torch` finding was FAILED under absence and PASSED under poison. It
    # never skipped, so no over-broad skip ever switched it off, and a count delta had
    # counted it anyway.
    switched_off = sorted(skipped_when_absent & passing_when_poisoned)

    # The same delta can also hide a finding, by netting a rise against a drop. That
    # reviewer checked ten of the thirteen rows with the capacity for it and found
    # none, so it was real in principle and had cost nothing in practice. Both
    # directions are reported per row now rather than left to be rechecked by hand.
    rose = passing_when_poisoned - passing_when_absent
    return {
        "tests_that_pass_without_it": len(switched_off),
        "which_tests_pass_without_it": switched_off,
        "what_a_count_delta_would_have_said": int(poisoned["passed"])
        - int(absent["passed"]),
        "rose_into_passing_without_having_skipped": sorted(rose - skipped_when_absent),
        "dropped_out_of_passing_under_poison": sorted(
            passing_when_absent - passing_when_poisoned
        ),
    }


def main() -> int:
    sites = fixture_level_sites()
    rows = []
    for path, fixture, modules in sites:
        for target in modules:
            here = importable(target)
            in_ci = ci_installs(target)
            absent = run(path, target, "absent")
            if not absent["skipped"]:
                # The module is installed in this environment, or the fixture is
                # unused. Either way this environment cannot tell us anything about
                # whether the skip was necessary, and saying so is better than
                # reporting a zero that reads like a clean result.
                rows.append(
                    {
                        "file": str(path.relative_to(REPO)),
                        "fixture": fixture,
                        "module": target,
                        "module_present_in_this_environment": here,
                        "ci_installs_it": in_ci,
                        "verdict": "not_measurable_here",
                        "why": "nothing skips for this module in this environment",
                        "absent": absent,
                    }
                )
                continue
            poisoned = run(path, target, "poison")
            if poisoned["parent_unavailable"]:
                rows.append(
                    {
                        "file": str(path.relative_to(REPO)),
                        "fixture": fixture,
                        "module": target,
                        "module_present_in_this_environment": here,
                        "ci_installs_it": in_ci,
                        "verdict": "not_measurable_here",
                        "why": (
                            "the parent package is not installed, so the leaf cannot be "
                            "made importable-but-unusable without stubbing the parent, "
                            "which perturbs unrelated fixtures in the same file"
                        ),
                        "absent": absent,
                    }
                )
                print(
                    f"{path.relative_to(TESTS)}::{fixture} [{target}] "
                    f"not measurable: parent package absent",
                    flush=True,
                )
                continue
            verdict = classify(absent, poisoned)
            gained = verdict["tests_that_pass_without_it"]

            rows.append(
                {
                    "file": str(path.relative_to(REPO)),
                    "fixture": fixture,
                    "module": target,
                    "module_present_in_this_environment": here,
                    "ci_installs_it": in_ci,
                    "verdict": (
                        "tests_switched_off_unnecessarily"
                        if gained > 0
                        else "skip_is_justified"
                    ),
                    "live_or_latent": (
                        "latent: CI installs this module, so nothing is skipped today "
                        "and these tests go quiet the moment the dependency is dropped"
                        if in_ci
                        else "live: CI does not install this module, so these tests are "
                        "being skipped in CI now"
                    ),
                    "handlers_that_could_swallow_the_poison": broad_except_sites(path),
                    **verdict,
                    "absent": absent,
                    "poisoned": poisoned,
                }
            )
            delta = verdict["what_a_count_delta_would_have_said"]
            print(
                f"{path.relative_to(TESTS)}::{fixture} [{target}] "
                f"absent {absent['passed']}p/{absent['skipped']}s -> "
                f"poison {poisoned['passed']}p/{poisoned['skipped']}s/"
                f"{poisoned['failed']}f  switched off {gained}"
                + (f"  (a count delta would say {delta})" if delta != gained else ""),
                flush=True,
            )

    measurable = [r for r in rows if r["verdict"] != "not_measurable_here"]
    switched_off = [
        r for r in measurable if r["verdict"] == "tests_switched_off_unnecessarily"
    ]
    measured = {
        "fixtures": len(sites),
        "pairs": len(rows),
        "measurable": len(measurable),
        "findings": len(switched_off),
        "tests": sum(int(r["tests_that_pass_without_it"]) for r in switched_off),
        "live": sum(1 for r in switched_off if not r["ci_installs_it"]),
    }
    record = {
        **narrative(measured),
        "what_this_measures": (
            "Whether a test skipped for a missing module actually uses that module. A "
            "test that skips when the module is absent and passes when the module is "
            "importable but unusable did not touch it through any path that raises, so "
            "the skip was switching off a test that had nothing to do with the "
            "dependency."
        ),
        "why_not_provably_never_touched_it": (
            "An earlier version of this file said 'provably never touched it'. That is "
            "stronger than the instrument can support, and a reviewer demonstrated it "
            "rather than arguing it: the poison raises AssertionError, so a test that "
            "calls into the module inside `except Exception` swallows the complaint and "
            "passes. They built a control that does exactly that and it passed under "
            "poison. No published number was affected, because no flagged file contains "
            "such a handler -- which is now measured per row as "
            "handlers_that_could_swallow_the_poison rather than audited once by hand. "
            "This is the only caveat here that would make a number an overcount; the "
            "others all make it a lower bound."
        ),
        "why_test_identity_and_not_a_count_delta": (
            "The first version of this sweep classified a row by "
            "poisoned_passed - absent_passed. That counts any test moving into passing, "
            "while the criterion is narrower: a test that *skipped* and then passed. A "
            "reviewer reproduced the counts exactly, took per-test identities, and found "
            "the single test behind the torch finding was FAILED under absence and "
            "PASSED under poison -- never skipped, so never switched off by a skip. The "
            "same delta can also hide a finding by netting a rise against a drop; they "
            "measured ten of the thirteen rows with that capacity and found none, so it "
            "was real in principle and had cost nothing in practice. Both are now "
            "measured by intersecting sets of test identities, and every row publishes "
            "what the delta would have said so the two can be compared."
        ),
        "why_a_pass_under_poison_is_the_only_evidence_counted": (
            "Poisoning a module that other libraries import during collection can break "
            "collection for reasons unrelated to the test, so a failure is not evidence "
            "of need. Only a pass is counted, which makes every number here a lower bound."
        ),
        "fixtures_with_an_importorskip": len(sites),
        "fixture_and_module_pairs": len(rows),
        "pairs_measurable_in_this_environment": len(measurable),
        "pairs_with_tests_switched_off_unnecessarily": len(switched_off),
        "of_those_live_in_ci": sum(1 for r in switched_off if not r["ci_installs_it"]),
        "tests_switched_off_in_ci_now": sum(
            int(r["tests_that_pass_without_it"])
            for r in switched_off
            if not r["ci_installs_it"]
        ),
        "how_to_reproduce_this": {
            "command": (
                "npa/.venv/bin/python "
                "docs/evidence/test-dependency-skips/dependency_skip_sweep.py"
            ),
            "needs_a_real_checkout": (
                "The sweep refuses to run outside a git checkout, because it resolves the "
                "repository root by walking up for .git and its findings are per-file. An "
                "exported archive has no repository marker, so the evidence cannot be "
                "regenerated from one. A reviewer hit this as eight errors and reported it "
                "as their own plumbing rather than a defect here; recorded so the next one "
                "does not have to discover it."
            ),
            "needs_the_interpreter_it_invokes": (
                "It shells out to npa/.venv/bin/python, so that path must exist. Link the "
                "environment *directory* if you are relocating it -- linking the interpreter "
                "binary alone corrupts prefix resolution and the run fails with seven "
                "unrelated-looking test failures. That is the same reviewer's finding."
            ),
            "runtime": "about three minutes, two pytest runs per fixture-and-module pair",
        },
        "how_live_was_determined": {
            "oracle": "ci-observed-packages.json",
            "which_is": (
                "the uv install list of a named real CI job, not an inference from "
                "pyproject extras and not this environment's contents"
            ),
            "job": json.loads(OBSERVED.read_text())["source"],
            "why_not_inferred": (
                "Inference said CI had no torch, because torch appears in no extra CI "
                "installs. CI has torch 2.14.0+cpu transitively. Publishing the "
                "inference would have called a latent finding live."
            ),
        },
        "tests_switched_off_unnecessarily": sum(
            int(r["tests_that_pass_without_it"]) for r in switched_off
        ),
        "ownership": (
            "Every row below is in another lane's tree. This change measures and reports "
            "them and edits none of them. The one site that was costing coverage in CI "
            "was in my own branch and is fixed there, not here."
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"\n{len(sites)} fixture-level sites, {len(measurable)} measurable here, "
        f"{len(switched_off)} with tests switched off unnecessarily "
        f"({record['tests_switched_off_unnecessarily']} tests)"
    )
    print(f"wrote {OUT.relative_to(REPO) if OUT.is_relative_to(REPO) else OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
