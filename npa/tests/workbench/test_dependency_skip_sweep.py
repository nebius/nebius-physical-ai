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
    poisoned = sweep.run(path, FAKE, "poison")

    assert absent["skipped"] == 2 and absent["passed"] == 0, absent
    assert poisoned["passed"] == 2 and poisoned["skipped"] == 0, poisoned
    assert poisoned["passed"] - absent["passed"] == 2


def test_the_sweep_leaves_a_skip_that_was_needed_alone(sweep, tmp_path: Path) -> None:
    """The same shape, except the test reads an attribute. That is a real dependency.

    Without this case the instrument could be a counter of skips rather than a
    discriminator, and every zero it reports would be unfalsifiable.
    """
    path = tmp_path / "test_used_dependency.py"
    path.write_text(USED)

    absent = sweep.run(path, FAKE, "absent")
    poisoned = sweep.run(path, FAKE, "poison")

    assert absent["skipped"] == 1 and absent["passed"] == 0, absent
    assert poisoned["passed"] == 0, poisoned
    assert poisoned["failed"] + poisoned["error"] >= 1, poisoned
    assert poisoned["passed"] - absent["passed"] == 0


def test_poisoning_a_dotted_module_does_not_disturb_its_siblings(
    sweep, tmp_path: Path
) -> None:
    """The artifact that produced a negative result, pinned so it cannot return.

    The first version stubbed every parent of a dotted target. Poisoning
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

    poisoned = sweep.run(path, "json.encoder", "poison")

    assert poisoned["passed"] == 1, poisoned
    assert poisoned["skipped"] == 0, poisoned


def test_a_target_whose_parent_is_missing_is_reported_not_measured(
    sweep, tmp_path: Path
) -> None:
    """There is no way to make a leaf usable-but-poisoned without a real parent.

    Stubbing the parent is the artifact above, so the sweep says it cannot measure
    this instead of measuring something else and calling it the same thing.
    """
    path = tmp_path / "test_absent_parent.py"
    path.write_text(UNUSED.replace(FAKE, f"{FAKE}.child"))

    poisoned = sweep.run(path, f"{FAKE}.child", "poison")

    assert poisoned["parent_unavailable"] is True, poisoned


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
    }
    regenerated = sweep.narrative(measured)
    for block in sorted(regenerated):
        assert record[block] == regenerated[block], block


def test_the_published_findings_are_recomputed_from_the_rows(record) -> None:
    """The summary is derived, so derive it again rather than reading it."""
    rows = record["rows"]
    measurable = [r for r in rows if r["verdict"] != "not_measurable_here"]
    switched_off = [
        r for r in measurable if r["verdict"] == "tests_switched_off_unnecessarily"
    ]

    assert record["fixture_and_module_pairs"] == len(rows)
    assert record["pairs_measurable_in_this_environment"] == len(measurable)
    assert record["pairs_with_tests_switched_off_unnecessarily"] == len(switched_off)
    assert record["tests_switched_off_unnecessarily"] == sum(
        r["tests_that_pass_without_it"] for r in switched_off
    )
    assert record["of_those_live_in_ci"] == sum(
        1 for r in switched_off if not r["ci_installs_it"]
    )
    assert record["tests_switched_off_in_ci_now"] == sum(
        r["tests_that_pass_without_it"] for r in switched_off if not r["ci_installs_it"]
    )


def test_every_finding_counts_only_tests_that_passed_without_the_module(record) -> None:
    """The one-sided rule, enforced rather than described.

    A failure under poison is not evidence of need -- poisoning a module that other
    libraries import during collection can break collection for its own reasons --
    so only a pass is counted. That makes every number here a lower bound, and it is
    worth a test because the tempting stronger reading is right there in the data.
    """
    for row in record["rows"]:
        if row["verdict"] != "tests_switched_off_unnecessarily":
            continue
        gained = row["poisoned"]["passed"] - row["absent"]["passed"]
        assert row["tests_that_pass_without_it"] == gained, row["file"]
        assert gained > 0, row["file"]


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
    assert "open3d" in snapshot["observed_absent"]


def test_the_sweep_reports_sites_it_cannot_measure_rather_than_passing_them(
    record,
) -> None:
    """A site this environment cannot decide must not be counted as decided either way."""
    unmeasured = [r for r in record["rows"] if r["verdict"] == "not_measurable_here"]
    assert unmeasured, "the sweep claims every site was measurable, which is unlikely"
    for row in unmeasured:
        assert row["why"], row["module"]
        assert "tests_that_pass_without_it" not in row, row["module"]


#: A test that skips when the module is absent and passes when it is poisoned -- a real
#: finding -- beside a second test that is not gated at all and whose outcome tracks
#: whether the module imports. That second one FAILS under absence and PASSES under
#: poison without ever skipping, which is the shape that produced a false positive.
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

#: The other direction: one test rises into passing and one drops out of it, so the two
#: cancel and a count delta reads zero over a row that has a real finding in it.
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

#: Touches the module, and swallows the poison's complaint. Passes under poison anyway.
SWALLOWS_THE_POISON = f'''
import pytest


@pytest.fixture
def dependency():
    pytest.importorskip("{FAKE}")
    return "not the module"


def test_reads_an_attribute_but_catches_everything(dependency):
    import {FAKE} as module

    try:
        module.some_attribute
    except Exception:
        pass
'''


def test_a_test_that_failed_rather_than_skipped_is_not_a_finding(
    sweep, tmp_path: Path
) -> None:
    """The false positive, reproduced and then excluded.

    A reviewer reproduced this sweep's counts exactly and then took the reading it had
    not: the single test behind its `torch` finding was FAILED under absence and PASSED
    under poison. Never skipped, so no over-broad skip had switched it off -- and
    `poisoned.passed - absent.passed` counted it regardless. One of the three published
    findings was that one test.

    Both numbers are asserted here, because the point is not that the new one is right
    but that the two disagree on a case that occurs in this repository.
    """
    path = tmp_path / "test_failed_then_passed.py"
    path.write_text(FAILED_THEN_PASSED)

    absent = sweep.run(path, FAKE, "absent")
    poisoned = sweep.run(path, FAKE, "poison")
    verdict = sweep.classify(absent, poisoned)

    assert absent["skipped"] == 1 and absent["failed"] == 1, absent
    assert poisoned["passed"] == 2, poisoned
    assert verdict["what_a_count_delta_would_have_said"] == 2, verdict
    assert verdict["tests_that_pass_without_it"] == 1, verdict
    assert [
        node.split("::")[-1] for node in verdict["which_tests_pass_without_it"]
    ] == ["test_gated"]
    assert [
        node.split("::")[-1]
        for node in verdict["rose_into_passing_without_having_skipped"]
    ] == ["test_not_gated_at_all"]


def test_a_rise_cancelling_a_drop_does_not_hide_the_finding(
    sweep, tmp_path: Path
) -> None:
    """The other way a count delta is wrong: it nets out, and the row reads clean.

    The reviewer who found the false positive raised this as the matching risk and
    checked ten of the thirteen rows that had the capacity for it, finding none. So it
    was real in principle and had cost nothing in practice. Here it is constructed, to
    keep it that way: the delta is zero while a test was genuinely switched off.
    """
    path = tmp_path / "test_rise_cancels_drop.py"
    path.write_text(RISE_CANCELS_DROP)

    absent = sweep.run(path, FAKE, "absent")
    poisoned = sweep.run(path, FAKE, "poison")
    verdict = sweep.classify(absent, poisoned)

    assert verdict["what_a_count_delta_would_have_said"] == 0, verdict
    assert verdict["tests_that_pass_without_it"] == 1, verdict
    assert [
        node.split("::")[-1] for node in verdict["which_tests_pass_without_it"]
    ] == ["test_gated"]
    assert [
        node.split("::")[-1] for node in verdict["dropped_out_of_passing_under_poison"]
    ] == ["test_passes_only_while_the_module_is_missing"]


def test_a_broad_handler_can_swallow_the_poison_and_is_reported(
    sweep, tmp_path: Path
) -> None:
    """The instrument's one overcounting limit, demonstrated rather than asserted.

    The poison raises `AssertionError` on attribute access, so a test that reads the
    module inside `except Exception` passes anyway and is counted as not needing it.
    That is why the record says these tests "did not touch it through any path that
    raises" and not "provably never touched it".

    No published row contains such a handler, which is a measurement rather than an
    audit: `broad_except_sites` runs on every file the sweep visits, and this asserts
    it finds the one here.
    """
    path = tmp_path / "test_swallows_the_poison.py"
    path.write_text(SWALLOWS_THE_POISON)

    poisoned = sweep.run(path, FAKE, "poison")
    absent = sweep.run(path, FAKE, "absent")

    assert absent["skipped"] == 1, absent
    assert poisoned["passed"] == 1, poisoned
    assert sweep.classify(absent, poisoned)["tests_that_pass_without_it"] == 1

    swallowing = sweep.broad_except_sites(path)
    assert len(swallowing) == 1, swallowing
    source = path.read_text().splitlines()
    assert source[swallowing[0] - 1].strip() == "except Exception:"


def test_no_published_finding_rests_on_a_file_that_could_swallow_the_poison(
    record,
) -> None:
    """The limit above, applied to the catalogue, by measurement rather than by eye."""
    flagged = [
        row
        for row in record["rows"]
        if row["verdict"] == "tests_switched_off_unnecessarily"
    ]
    assert flagged, "no findings to check"
    for row in flagged:
        assert row["handlers_that_could_swallow_the_poison"] == [], row["file"]
