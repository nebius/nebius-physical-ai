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

`bypass`
    The module remains genuinely absent, but the fixture's own `importorskip`
    returns an unusable sentinel instead of skipping. A downstream `import` still
    raises `ModuleNotFoundError`, so import-only dependencies are detected too.

A test that skips under `absent` and passes under `bypass` did not need the module.
A test that fails or errors under `bypass` does need it. Tests that still skip for
another dependency are undecided rather than reported as justified.

The classification is deliberately per-test and conservative. A gated test that
passes is unnecessary, one that fails or errors requires the dependency, and one
that still skips is undecided. Collection failures and unexecuted tests are never
reported as evidence that a skip was justified.

Run from the repository root with the environment under test:
    ${PYTHON} docs/evidence/test-dependency-skips/dependency_skip_sweep.py
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


def repository_root(start: Path | None = None) -> Path:
    """Walk up for a repository marker, refusing to guess when there is none.

    ``start`` exists so the refusal can be asserted as behaviour. Its one real caller
    below passes nothing. Review's guard for this used to grep the function's own source
    for ``.git`` and ``raise``, which cannot tell a real behaviour from a caveat in
    prose -- an added ``return`` above the raise keeps both words and silently resolves
    a wrong root, and that mutant passed.
    """
    for candidate in (start or Path(__file__).resolve()).parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("not inside a repository checkout")


PYTHON = Path(sys.executable)

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
    """Fixture sentinel: unusable, but deliberately absent from sys.modules."""

    def __getattr__(self, name):
        raise AssertionError("touched %s.%s" % (self.__name__, name))


def pytest_configure(config):
    for name in [m for m in list(sys.modules) if m == TARGET or m.startswith(TARGET + ".")]:
        del sys.modules[name]
    sys.meta_path.insert(0, _Blocker())
    if MODE == "absent":
        return
    if MODE != "bypass":
        raise RuntimeError("unknown sweep mode")

    import pytest

    original = pytest.importorskip
    sentinel = _Poison(TARGET)

    def importorskip(name, *args, **kwargs):
        if name == TARGET:
            return sentinel
        return original(name, *args, **kwargs)

    pytest.importorskip = importorskip
'''

#: `-v` names every test and its outcome, which `-rA` does not: its SKIPPED lines
#: carry a file and line rather than a node identifier, so they cannot be intersected
#: with anything.
VERBOSE_OUTCOME = re.compile(
    r"^(\S+::\S+) (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)", re.MULTILINE
)


def _is_fixture_decorator(node: ast.expr) -> bool:
    """Recognize @pytest.fixture/@fixture, not arbitrary text containing fixture."""
    target = node.func if isinstance(node, ast.Call) else node
    return (isinstance(target, ast.Name) and target.id == "fixture") or (
        isinstance(target, ast.Attribute)
        and target.attr == "fixture"
        and isinstance(target.value, ast.Name)
        and target.value.id == "pytest"
    )


def fixture_level_sites() -> list[tuple[Path, str, list[str]]]:
    """Every `importorskip` inside a fixture, which is as broad as the file's widest need."""
    sites: list[tuple[Path, str, list[str]]] = []
    tests = repository_root() / "npa/tests"
    for path in sorted(tests.rglob("test_*.py")) + sorted(tests.rglob("conftest.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(_is_fixture_decorator(d) for d in node.decorator_list):
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
    repo = repository_root()
    with tempfile.TemporaryDirectory() as plugin_dir:
        name = "npa_sweep_plugin"
        (Path(plugin_dir) / f"{name}.py").write_text(
            PLUGIN.format(target=target, mode=mode)
        )
        existing_pythonpath = os.environ.get("PYTHONPATH")
        environment = dict(
            os.environ,
            PYTHONPATH=(
                plugin_dir
                if not existing_pythonpath
                else plugin_dir + os.pathsep + existing_pythonpath
            ),
        )
        completed = subprocess.run(
            [
                str(PYTHON),
                "-m",
                "pytest",
                str(path if not path.is_relative_to(repo) else path.relative_to(repo)),
                "-v",
                "--tb=no",
                "-p",
                name,
                "-p",
                "no:cacheprovider",
                "--timeout=120",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
    by_outcome: dict[str, list[str]] = {}
    for node, outcome in VERBOSE_OUTCOME.findall(completed.stdout):
        by_outcome.setdefault(outcome.lower(), []).append(node)
    return {
        **{
            outcome: len(by_outcome.get(outcome, []))
            for outcome in ("passed", "failed", "skipped", "error")
        },
        # Per test, because the criterion is about individual tests and a count is
        # not. A delta over counts cannot tell a test that moved from skipped into
        # passing from one that moved from *failed* into passing, and it nets a rise
        # against a drop to zero. Both of those were found in this sweep's published
        # numbers by a reviewer who took this reading and I had not.
        "nodes": {
            outcome: sorted(nodes) for outcome, nodes in sorted(by_outcome.items())
        },
        "returncode": completed.returncode,
    }


OBSERVED = Path(__file__).resolve().parent / "ci-observed-packages.json"


def ci_installs(module: str) -> bool | None:
    """Whether CI has the module, read from a real CI job rather than inferred.

    The named job is retained because environment contents change. Current CI also
    explicitly installs the sonic and genesis-test extras and asserts that torch and
    rsl_rl import; the snapshot maps import names to the actual distribution names.

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
    """Line numbers of handlers that could swallow the sentinel's AssertionError.

    This is the one limit of the instrument that errs in the unsafe direction. A test
    that reads the fixture sentinel inside `except Exception` catches the sentinel's
    complaint and passes, and would be recorded as a test that does not need the
    module. A reviewer demonstrated it with a purpose-built control rather than
    arguing it. This scans only the test file, not arbitrary downstream call graphs,
    so the limitation is retained beside each row rather than claimed away.

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
        cwd=repository_root(),
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
            f"module through the measured path, {measured['tests']} tests in total. "
            f"Another {measured['undecided']} pairs are explicitly not fully measured."
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
            "Not that the tests are good, only that the measured path runs. Not that a module is "
            "unused by the file -- several of these fixtures sit beside tests that do "
            "need the dependency, which is why the measurement is per test rather than "
            "per file. And not a complete census: a pair this environment cannot "
            "decide is reported as undecided, not as clean."
        ),
    }


def classify(absent: dict, bypassed: dict) -> dict:
    """Decide one row from the two runs, by test identity rather than by a count.

    Separate from `main` so the two ways a count delta gets this wrong can be driven
    directly by a test, instead of only through a three-minute sweep of the tree.
    """
    skipped_when_absent = set(absent["nodes"].get("skipped", []))
    passing_when_bypassed = set(bypassed["nodes"].get("passed", []))
    failing_when_bypassed = set(bypassed["nodes"].get("failed", [])) | set(
        bypassed["nodes"].get("error", [])
    )
    skipped_when_bypassed = set(bypassed["nodes"].get("skipped", []))
    passing_when_absent = set(absent["nodes"].get("passed", []))

    # The criterion, applied to the tests it is about. A reviewer reproduced this
    # sweep's counts exactly and then took the one reading it had not: the single test
    # behind its `torch` finding was FAILED under absence and PASSED under the old
    # poison method. It
    # never skipped, so no over-broad skip ever switched it off, and a count delta had
    # counted it anyway.
    switched_off = sorted(skipped_when_absent & passing_when_bypassed)
    dependency_required = sorted(skipped_when_absent & failing_when_bypassed)
    undecided = sorted(skipped_when_absent & skipped_when_bypassed)
    exercised = sorted(set(switched_off) | set(dependency_required))

    # The same delta can also hide a finding, by netting a rise against a drop. That
    # reviewer checked ten of the thirteen rows with the capacity for it and found
    # none, so it was real in principle and had cost nothing in practice. Both
    # directions are reported per row now rather than left to be rechecked by hand.
    rose = passing_when_bypassed - passing_when_absent
    return {
        "tests_that_pass_without_it": len(switched_off),
        "which_tests_pass_without_it": switched_off,
        "tests_that_require_it": len(dependency_required),
        "which_tests_require_it": dependency_required,
        "gated_tests_exercised": len(exercised),
        "which_gated_tests_remain_undecided": undecided,
        "what_a_count_delta_would_have_said": int(bypassed["passed"])
        - int(absent["passed"]),
        "rose_into_passing_without_having_skipped": sorted(rose - skipped_when_absent),
        "dropped_out_of_passing_under_bypass": sorted(
            passing_when_absent - passing_when_bypassed
        ),
    }


def main() -> int:
    repo = repository_root()
    tests = repo / "npa/tests"
    sites = fixture_level_sites()
    rows = []
    for path, fixture, modules in sites:
        for target in modules:
            here = importable(target)
            in_ci = ci_installs(target)
            if path.name == "conftest.py":
                rows.append(
                    {
                        "file": str(path.relative_to(repo)),
                        "fixture": fixture,
                        "module": target,
                        "module_present_in_this_environment": here,
                        "ci_installs_it": in_ci,
                        "verdict": "not_measured_conftest",
                        "why": (
                            "A conftest.py is not a test target. This sweep does not "
                            "claim a dependency verdict until fixture consumers are "
                            "resolved and run explicitly."
                        ),
                    }
                )
                continue
            absent = run(path, target, "absent")
            if not absent["skipped"]:
                # The module is installed in this environment, or the fixture is
                # unused. Either way this environment cannot tell us anything about
                # whether the skip was necessary, and saying so is better than
                # reporting a zero that reads like a clean result.
                rows.append(
                    {
                        "file": str(path.relative_to(repo)),
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
            bypassed = run(path, target, "bypass")
            verdict = classify(absent, bypassed)
            gained = verdict["tests_that_pass_without_it"]
            required = verdict["tests_that_require_it"]
            undecided = verdict["which_gated_tests_remain_undecided"]
            if verdict["gated_tests_exercised"] == 0:
                row_verdict = "undecided_no_gated_test_executed"
            elif undecided:
                row_verdict = "partially_measured"
            elif gained and required:
                row_verdict = "mixed_dependency_need"
            elif gained:
                row_verdict = "tests_switched_off_unnecessarily"
            else:
                row_verdict = "dependency_required"

            if in_ci is True:
                ci_status = "latent: CI installs this module"
            elif in_ci is False:
                ci_status = "live: CI does not install this module"
            else:
                ci_status = "unknown: the CI snapshot does not map this module"

            rows.append(
                {
                    "file": str(path.relative_to(repo)),
                    "fixture": fixture,
                    "module": target,
                    "module_present_in_this_environment": here,
                    "ci_installs_it": in_ci,
                    "verdict": row_verdict,
                    "live_or_latent": ci_status,
                    "test_file_handlers_that_could_swallow_the_sentinel": (
                        broad_except_sites(path)
                    ),
                    **verdict,
                    "absent": absent,
                    "bypassed": bypassed,
                }
            )
            delta = verdict["what_a_count_delta_would_have_said"]
            print(
                f"{path.relative_to(tests)}::{fixture} [{target}] "
                f"absent {absent['passed']}p/{absent['skipped']}s -> "
                f"bypass {bypassed['passed']}p/{bypassed['skipped']}s/"
                f"{bypassed['failed'] + bypassed['error']}f/e  "
                f"unnecessary {gained}, required {required}, undecided {len(undecided)}"
                + (f"  (a count delta would say {delta})" if delta != gained else ""),
                flush=True,
            )

    fully_decided_verdicts = {
        "dependency_required",
        "mixed_dependency_need",
        "tests_switched_off_unnecessarily",
    }
    measurable = [r for r in rows if r["verdict"] in fully_decided_verdicts]
    undecided_rows = [r for r in rows if r["verdict"] not in fully_decided_verdicts]
    switched_off = [r for r in rows if r.get("tests_that_pass_without_it", 0) > 0]
    measured = {
        "fixtures": len(sites),
        "pairs": len(rows),
        "measurable": len(measurable),
        "findings": len(switched_off),
        "tests": sum(int(r["tests_that_pass_without_it"]) for r in switched_off),
        "live": sum(1 for r in switched_off if r["ci_installs_it"] is False),
        "undecided": len(undecided_rows),
    }
    record = {
        **narrative(measured),
        "what_this_measures": (
            "Whether a test skipped for a missing module runs when only its fixture's "
            "importorskip is bypassed while the module remains genuinely absent. "
            "Downstream imports still fail, so import-only dependencies are classified "
            "as required rather than as unnecessary skips."
        ),
        "sentinel_overcount_limit": (
            "An earlier version of this file said 'provably never touched it'. That is "
            "stronger than the instrument can support, and a reviewer demonstrated it "
            "rather than arguing it: the fixture sentinel raises AssertionError, so a "
            "test that reads it inside `except Exception` can swallow the complaint and "
            "pass. The row records broad handlers found in the test file, but this is "
            "not a downstream call-graph analysis. Findings therefore establish the "
            "observed bypass behaviour, not that no downstream code touched a value."
        ),
        "why_test_identity_and_not_a_count_delta": (
            "The first version of this sweep classified a row by "
            "bypassed_passed - absent_passed. That counts any test moving into passing, "
            "while the criterion is narrower: a test that *skipped* and then passed. A "
            "reviewer reproduced the counts exactly, took per-test identities, and found "
            "the single test behind the torch finding was FAILED under absence and "
            "PASSED under the old poison method -- never skipped, so never switched off "
            "by a skip. The "
            "same delta can also hide a finding by netting a rise against a drop; they "
            "measured ten of the thirteen rows with that capacity and found none, so it "
            "was real in principle and had cost nothing in practice. Both are now "
            "measured by intersecting sets of test identities, and every row publishes "
            "what the delta would have said so the two can be compared."
        ),
        "how_bypass_outcomes_are_interpreted": (
            "The target stays genuinely absent. A gated pass shows the fixture skip was "
            "unnecessary; a gated failure or error shows the dependency is required; a "
            "continued skip is undecided. Unexecuted tests and collection-only outcomes "
            "are not classified as justified."
        ),
        "fixtures_with_an_importorskip": len(sites),
        "fixture_and_module_pairs": len(rows),
        "pairs_measurable_in_this_environment": len(measurable),
        "pairs_not_fully_measured": len(undecided_rows),
        "pairs_with_tests_switched_off_unnecessarily": len(switched_off),
        "of_those_live_in_ci": sum(
            1 for r in switched_off if r["ci_installs_it"] is False
        ),
        "tests_switched_off_in_ci_now": sum(
            int(r["tests_that_pass_without_it"])
            for r in switched_off
            if r["ci_installs_it"] is False
        ),
        "how_to_reproduce_this": {
            "command": (
                "${PYTHON} docs/evidence/test-dependency-skips/dependency_skip_sweep.py"
            ),
            "needs_a_real_checkout": (
                "The sweep refuses to run outside a git checkout, because it resolves the "
                "repository root by walking up for .git and its findings are per-file. An "
                "exported archive has no repository marker, so the evidence cannot be "
                "regenerated from one. A reviewer hit this and reported it as their own "
                "plumbing rather than a defect here; recorded so the next one does not "
                "have to discover it. No failure count is quoted because the number "
                "depends on the environment and neither of us can reproduce it."
            ),
            "child_interpreter": "sys.executable",
            "interpreter_contract": (
                "Set PYTHON to the environment under test. Child pytest runs use that "
                "interpreter's sys.executable; no checkout-local virtualenv path is required."
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
                "The named workflow installs sonic and genesis-test extras and asserts "
                "torch and rsl_rl imports. The retained package list supplies the exact "
                "versions, including the rsl-rl-lib distribution behind rsl_rl."
            ),
        },
        "measured_source_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "measurement_environment": {
            "python": sys.version.split()[0],
            "executable_kind": "sys.executable",
            "module_presence": {
                module: importable(module)
                for module in sorted({row["module"] for row in rows})
            },
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
    print(f"wrote {OUT.relative_to(repo) if OUT.is_relative_to(repo) else OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
