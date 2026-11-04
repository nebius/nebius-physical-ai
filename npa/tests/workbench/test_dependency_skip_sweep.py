"""Controls for the instrument that decides whether a skipped test needed its skip.

The claim the sweep makes is a negative one -- *this* test never touched the module
it was skipped for -- so the instrument has to be shown capable of the opposite
verdict before any of its zeroes mean anything. These build two synthetic test
files, identical except that one of them actually uses the dependency, and require
the sweep to tell them apart.

Synthetic rather than pointed at the repository's own tests, because a control that
depends on torch not being installed stops being a control the day someone installs
torch.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs/evidence/test-dependency-skips"
SWEEP = EVIDENCE / "dependency_skip_sweep.py"
RECORD = EVIDENCE / "dependency-skip-sweep.json"
OBSERVED = EVIDENCE / "ci-observed-packages.json"

#: A name no distribution claims, so `absent` mode needs no blocking to be absent.
FAKE = "npa_dependency_that_does_not_exist"

UNUSED = f'''
import pytest


@pytest.fixture
def dependency():
    pytest.importorskip("{FAKE}")
    return "not the module"


def test_first(dependency):
    assert dependency == "not the module"


def test_second(dependency):
    assert len(dependency) == 14
'''

USED = f'''
import pytest


@pytest.fixture
def dependency():
    return pytest.importorskip("{FAKE}")


def test_first(dependency):
    assert dependency.some_attribute == 1
'''


@pytest.fixture(scope="module")
def sweep() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dependency_skip_sweep", SWEEP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_sweep_detects_a_skip_that_was_not_needed(sweep, tmp_path: Path) -> None:
    """Two tests gated on a module neither of them uses. Both should be reported."""
    path = tmp_path / "test_unused_dependency.py"
    path.write_text(UNUSED)

    absent = sweep.run(path, FAKE, "absent")
    bypassed = sweep.run(path, FAKE, "bypass")

    assert absent["skipped"] == 2 and absent["passed"] == 0, absent
    assert bypassed["passed"] == 2 and bypassed["skipped"] == 0, bypassed
    assert sweep.classify(absent, bypassed)["tests_that_pass_without_it"] == 2


def test_the_sweep_leaves_a_skip_that_was_needed_alone(sweep, tmp_path: Path) -> None:
    """The same shape, except the test reads an attribute. That is a real dependency.

    Without this case the instrument could be a counter of skips rather than a
    discriminator, and every zero it reports would be unfalsifiable.
    """
    path = tmp_path / "test_used_dependency.py"
    path.write_text(USED)

    absent = sweep.run(path, FAKE, "absent")
    bypassed = sweep.run(path, FAKE, "bypass")

    assert absent["skipped"] == 1 and absent["passed"] == 0, absent
    assert bypassed["passed"] == 0, bypassed
    assert bypassed["failed"] + bypassed["error"] >= 1, bypassed
    assert sweep.classify(absent, bypassed)["tests_that_require_it"] == 1


def test_bypassing_a_dotted_module_does_not_disturb_its_siblings(
    sweep, tmp_path: Path
) -> None:
    """The artifact that produced a negative result, pinned so it cannot return.

    The first version stubbed every parent of a dotted target. Replacing
    `pxr.UsdPhysics` therefore replaced `pxr` with a stub whose `__path__` was
    empty, an unrelated fixture's `importorskip("pxr.Usd")` began failing, and three
    passing tests were reported as newly skipped. A measurement instrument returning
    a negative number is the instrument describing itself.
    """
    path = tmp_path / "test_sibling_submodule.py"
    path.write_text(
        """
import pytest


@pytest.fixture
def sibling():
    return pytest.importorskip("json.decoder")


def test_the_sibling_still_works(sibling):
    assert sibling.JSONDecoder is not None
"""
    )

    bypassed = sweep.run(path, "json.encoder", "bypass")

    assert bypassed["passed"] == 1, bypassed
    assert bypassed["skipped"] == 0, bypassed


def test_an_import_only_dependency_is_detected(sweep, tmp_path: Path) -> None:
    """Bypassing the fixture must not make downstream imports magically succeed."""
    (tmp_path / "consumer.py").write_text(f"import {FAKE}\n")
    path = tmp_path / "test_import_only.py"
    path.write_text(
        f'''
import importlib
import pytest


@pytest.fixture
def dependency(monkeypatch):
    pytest.importorskip("{FAKE}")
    monkeypatch.syspath_prepend({str(tmp_path)!r})
    return importlib.import_module("consumer")


def test_consumer_imports(dependency):
    assert dependency is not None
'''
    )

    absent = sweep.run(path, FAKE, "absent")
    bypassed = sweep.run(path, FAKE, "bypass")
    verdict = sweep.classify(absent, bypassed)

    assert absent["skipped"] == 1, absent
    assert bypassed["error"] + bypassed["failed"] == 1, bypassed
    assert verdict["tests_that_require_it"] == 1
    assert verdict["tests_that_pass_without_it"] == 0


def test_fixture_detection_uses_decorator_structure_not_substrings(sweep) -> None:
    """A helper whose name merely contains 'fixture' is not a pytest fixture."""
    tree = ast.parse(
        """
@pytest.fixture
def real(): ...

@fixture()
def imported(): ...

@not_a_fixture
def false_positive(): ...
"""
    )
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    detected = {
        node.name
        for node in functions
        if any(sweep._is_fixture_decorator(item) for item in node.decorator_list)
    }
    assert detected == {"real", "imported"}


def test_the_prose_is_reproducible_from_the_measurements_it_describes(
    sweep, record
) -> None:
    """Regenerate the sentences, do not check that their figures appear somewhere.

    A sibling change learned this the hard way: a membership check against the
    numbers a record already holds passes a sentence that is false about the run,
    because the figures in the false sentence are all figures from somewhere else in
    the same record.
    """
    measured = {
        "fixtures": record["fixtures_with_an_importorskip"],
        "pairs": record["fixture_and_module_pairs"],
        "measurable": record["pairs_measurable_in_this_environment"],
        "findings": record["pairs_with_tests_switched_off_unnecessarily"],
        "tests": record["tests_switched_off_unnecessarily"],
        "live": record["of_those_live_in_ci"],
        "undecided": record["pairs_not_fully_measured"],
    }
    regenerated = sweep.narrative(measured)
    for block in sorted(regenerated):
        assert record[block] == regenerated[block], block


def test_the_published_findings_are_recomputed_from_the_rows(record) -> None:
    """The summary is derived, so derive it again rather than reading it."""
    rows = record["rows"]
    decided = {
        "dependency_required",
        "mixed_dependency_need",
        "tests_switched_off_unnecessarily",
    }
    measurable = [r for r in rows if r["verdict"] in decided]
    unmeasured = [r for r in rows if r["verdict"] not in decided]
    switched_off = [r for r in rows if r.get("tests_that_pass_without_it", 0) > 0]

    assert record["fixture_and_module_pairs"] == len(rows)
    assert record["pairs_measurable_in_this_environment"] == len(measurable)
    assert record["pairs_not_fully_measured"] == len(unmeasured)
    assert record["pairs_with_tests_switched_off_unnecessarily"] == len(switched_off)
    assert record["tests_switched_off_unnecessarily"] == sum(
        r["tests_that_pass_without_it"] for r in switched_off
    )
    assert record["of_those_live_in_ci"] == sum(
        1 for r in switched_off if r["ci_installs_it"] is False
    )
    assert record["tests_switched_off_in_ci_now"] == sum(
        r["tests_that_pass_without_it"]
        for r in switched_off
        if r["ci_installs_it"] is False
    )


def test_every_finding_is_the_intersection_of_skipped_and_bypassed_passes(record):
    """Use test identities; an aggregate count delta is not the criterion."""
    for row in record["rows"]:
        if not row.get("tests_that_pass_without_it"):
            continue
        expected = sorted(
            set(row["absent"]["nodes"].get("skipped", []))
            & set(row["bypassed"]["nodes"].get("passed", []))
        )
        assert row["which_tests_pass_without_it"] == expected, row["file"]
        assert row["tests_that_pass_without_it"] == len(expected), row["file"]


def test_whether_ci_has_a_module_comes_from_a_named_job(record) -> None:
    """Not from this environment, and not from the constraints file.

    `npa/ci/requirements.txt` pins torch, and CI passes it with `-c`, which
    constrains a version without installing anything. No extra CI installs carries
    torch. A real CI job has it anyway, transitively. An inference from the manifests
    was about to publish the torch finding as costing coverage today.
    """
    snapshot = json.loads(OBSERVED.read_text())
    source = snapshot["source"]
    for field in ("workflow", "job", "run_id", "job_id", "retrieved_utc"):
        assert source.get(field), field

    assert record["how_live_was_determined"]["job"] == source
    assert "torch" in snapshot["import_name_to_distribution"]
    assert snapshot["observed_installed"]["torch"].startswith("2.")
    assert snapshot["import_name_to_distribution"]["rsl_rl"] == "rsl-rl-lib"
    assert snapshot["observed_installed"]["rsl-rl-lib"] == "5.5.1"
    assert "open3d" in snapshot["observed_absent"]


def test_the_sweep_reports_sites_it_cannot_measure_rather_than_passing_them(
    record,
) -> None:
    """A site this environment cannot decide must not be counted as decided either way."""
    unmeasured = [
        r
        for r in record["rows"]
        if r["verdict"]
        in {
            "not_measurable_here",
            "not_measured_conftest",
            "partially_measured",
            "undecided_no_gated_test_executed",
        }
    ]
    assert unmeasured, "the sweep claims every site was measurable, which is unlikely"
    for row in unmeasured:
        assert row["why"], row["module"]
        if row["verdict"] in {"not_measurable_here", "not_measured_conftest"}:
            assert "tests_that_pass_without_it" not in row, row["module"]


#: A real gated finding beside a test that imports the target without using the fixture.
#: True absence must keep the latter failed in both modes; the old module-level poison
#: made it pass and corrupted the count.
FAILED_THEN_PASSED = f'''
import pytest


@pytest.fixture
def dependency():
    pytest.importorskip("{FAKE}")
    return "not the module"


def test_gated(dependency):
    assert dependency == "not the module"


def test_not_gated_at_all():
    import {FAKE} as module

    assert module is not None
'''

#: A gated finding beside a test that is expected to pass only while the module is
#: absent. Both modes keep it absent, so its outcome must not change.
RISE_CANCELS_DROP = f'''
import pytest


@pytest.fixture
def dependency():
    pytest.importorskip("{FAKE}")
    return "not the module"


def test_gated(dependency):
    assert dependency == "not the module"


def test_passes_only_while_the_module_is_missing():
    with pytest.raises(ModuleNotFoundError):
        import {FAKE}  # noqa: F401
'''

#: Touches the fixture sentinel and swallows its complaint. This remains a documented
#: limitation of the observed-behaviour claim.
SWALLOWS_THE_SENTINEL = f'''
import pytest


@pytest.fixture
def dependency():
    return pytest.importorskip("{FAKE}")


def test_reads_an_attribute_but_catches_everything(dependency):
    try:
        dependency.some_attribute
    except Exception:
        pass
'''


def test_a_test_that_failed_rather_than_skipped_is_not_a_finding(
    sweep, tmp_path: Path
) -> None:
    """The false positive, reproduced and then excluded.

    A reviewer reproduced this sweep's counts exactly and then took the reading it had
    not: the single test behind its `torch` finding was FAILED under absence and PASSED
    under the old poison mode. Never skipped, so no over-broad skip had switched it off.
    Keeping the target genuinely absent now leaves that unrelated test failed.

    Both numbers are asserted here, because the point is not that the new one is right
    but that the two disagree on a case that occurs in this repository.
    """
    path = tmp_path / "test_failed_then_passed.py"
    path.write_text(FAILED_THEN_PASSED)

    absent = sweep.run(path, FAKE, "absent")
    bypassed = sweep.run(path, FAKE, "bypass")
    verdict = sweep.classify(absent, bypassed)

    assert absent["skipped"] == 1 and absent["failed"] == 1, absent
    assert bypassed["passed"] == 1 and bypassed["failed"] == 1, bypassed
    assert verdict["what_a_count_delta_would_have_said"] == 1, verdict
    assert verdict["tests_that_pass_without_it"] == 1, verdict
    assert [
        node.split("::")[-1] for node in verdict["which_tests_pass_without_it"]
    ] == ["test_gated"]
    assert verdict["rose_into_passing_without_having_skipped"] == []


def test_true_absence_keeps_unrelated_outcomes_stable(sweep, tmp_path: Path) -> None:
    """Bypass changes only the gated test, not ordinary import semantics."""
    path = tmp_path / "test_rise_cancels_drop.py"
    path.write_text(RISE_CANCELS_DROP)

    absent = sweep.run(path, FAKE, "absent")
    bypassed = sweep.run(path, FAKE, "bypass")
    verdict = sweep.classify(absent, bypassed)

    assert verdict["what_a_count_delta_would_have_said"] == 1, verdict
    assert verdict["tests_that_pass_without_it"] == 1, verdict
    assert [
        node.split("::")[-1] for node in verdict["which_tests_pass_without_it"]
    ] == ["test_gated"]
    assert verdict["dropped_out_of_passing_under_bypass"] == []


def test_a_broad_handler_can_swallow_the_sentinel_and_is_reported(
    sweep, tmp_path: Path
) -> None:
    """The instrument's one overcounting limit, demonstrated rather than asserted.

    The sentinel raises `AssertionError` on attribute access, so a test that reads the
    module inside `except Exception` passes anyway and is counted as not needing it.
    That is why the record says these tests "did not touch it through any path that
    raises" and not "provably never touched it".

    No published row contains such a handler, which is a measurement rather than an
    audit: `broad_except_sites` runs on every file the sweep visits, and this asserts
    it finds the one here.
    """
    path = tmp_path / "test_swallows_the_sentinel.py"
    path.write_text(SWALLOWS_THE_SENTINEL)

    bypassed = sweep.run(path, FAKE, "bypass")
    absent = sweep.run(path, FAKE, "absent")

    assert absent["skipped"] == 1, absent
    assert bypassed["passed"] == 1, bypassed
    assert sweep.classify(absent, bypassed)["tests_that_pass_without_it"] == 1

    swallowing = sweep.broad_except_sites(path)
    assert len(swallowing) == 1, swallowing
    source = path.read_text().splitlines()
    assert source[swallowing[0] - 1].strip() == "except Exception:"


def test_published_findings_expose_test_file_sentinel_handlers(
    record,
) -> None:
    """The limit above, applied to the catalogue, by measurement rather than by eye."""
    flagged = [
        row for row in record["rows"] if row.get("tests_that_pass_without_it", 0) > 0
    ]
    assert flagged, "no findings to check"
    for row in flagged:
        assert "test_file_handlers_that_could_swallow_the_sentinel" in row, row["file"]


def test_the_documented_reproduction_matches_what_the_sweep_actually_runs(
    sweep, record
) -> None:
    """A reproduction note that drifts from the code is worse than none at all.

    A reviewer running this sweep from an exported archive hit errors from the missing
    repository marker, then more from a harness missing the interpreter it shells out
    to, and reported both as their own plumbing rather than as defects here. The record
    now says so, and this keeps the saying true: both paths in that note are derived
    from the module rather than retyped beside it.
    """
    how = record["how_to_reproduce_this"]

    assert str(SWEEP.relative_to(REPO)) in how["command"]
    assert how["command"].startswith("${PYTHON} ")
    assert how["child_interpreter"] == "sys.executable"
    assert sweep.PYTHON == Path(sweep.sys.executable)


def test_the_sweep_refuses_a_tree_that_is_not_a_checkout(sweep) -> None:
    """Exercise the refusal the note promises, rather than grepping for its spelling.

    This assertion used to read the resolver's source for `.git` and `raise`, under a
    comment claiming to distinguish a real behaviour from a caveat in prose -- which a
    substring search is precisely unable to do. Review mutated the resolver the way an
    ordinary refactor would, adding a `return` above the raise so it falls back to the
    file's own directory, and it passed: both words were still there while the sweep
    silently accepted a non-checkout and resolved a wrong root. Deleting the body
    outright did fail, so the guard caught vandalism and missed the realistic edit.

    Calling it is the whole fix. Every mutant that keeps the contract survives this and
    every mutant that breaks it fails, including theirs.
    """
    assert sweep.repository_root(REPO / "npa/tests/workbench/anything.py") == REPO

    with pytest.raises(RuntimeError):
        sweep.repository_root(Path("/__npa_not_a_checkout__/nowhere/deep"))
