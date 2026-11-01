"""Guards for the Poisson reproducibility finding.

The finding is a negative one: figures this lane published to sixteen significant digits
do not survive a second run of the harness that wrote them. These tests pin the finding,
the cause, and the two remedies that fail -- and one of them pins a mistake I made while
measuring it, because that mistake is the reason the record's wording is derived from the
numbers instead of written beside them.

Reads the record only. No Open3D, so this runs wherever the suite runs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "docs" / "evidence" / "open3d" / "poisson-figure-reproducibility.json"
HARNESS = (
    REPO
    / "docs"
    / "evidence"
    / "open3d"
    / "harness"
    / "poisson_figure_reproducibility.py"
)

#: The threshold the audited record's conclusion rests on.
MINIMUM_VISIBLE_AREA = 0.5


@pytest.fixture
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_finding_is_that_no_condition_reproduces(record) -> None:
    """The negative result, stated for every condition rather than in general."""
    conditions = record["cause"]["conditions"]
    assert set(conditions) == {"default", "seeded_before_each_call", "thread_limited"}
    for name, row in conditions.items():
        assert row["repeats"] >= 4, f"{name} has too few repeats to say anything"
        assert row["surface_area"]["identical"] is False, (
            f"{name} reproduced exactly, which would retract the finding -- if that is "
            f"real the record needs rewriting, not this test relaxing"
        )
        assert row["surface_area"]["relative_spread"] > 0.0


def test_the_cause_is_positions_and_not_a_different_mesh(record) -> None:
    """Stable topology is what makes "same mesh, moved vertices" a measurement.

    Without it the finding would be consistent with Poisson producing a genuinely
    different reconstruction each time, which would be a far larger claim.
    """
    for name, row in record["cause"]["conditions"].items():
        assert row["topology_is_stable"] is True, name
        assert row["vertices"] > 0 and row["triangles"] > 0, name


def test_seeding_is_not_offered_as_a_remedy(record) -> None:
    """The seeded condition has to be no better, and the record has to say so.

    There is a real sampling seed in `audit_view_occlusion.py`, and it does bind what it
    claims to bind. The trap is extending that to Poisson, whose variation is not drawn
    from Open3D's generator at all.
    """
    conditions = record["cause"]["conditions"]
    assert conditions["seeded_before_each_call"]["seeded_before_each_call"] is True
    assert conditions["seeded_before_each_call"]["surface_area"]["identical"] is False
    assert "Does not help" in record["what_the_remedies_do"]["seeding"]


def test_the_thread_limiting_verdict_agrees_with_its_own_numbers(record) -> None:
    """This test exists because of two mistakes, and both are worth keeping visible.

    A first probe used two repeats and put single-threading two orders of magnitude ahead
    of the default. At four repeats it is within a small factor, and the spreads
    themselves move between runs - in one run the limited condition is the worse of the
    two. "Two orders of magnitude" was about to go into the record as a finding.

    The second mistake was the experiment itself: it set `OMP_NUM_THREADS=1`, and a
    reviewer showed this build links no OpenMP runtime and exports no OpenMP symbols, so
    the variable changed nothing and the measurement under it tested nothing. It now uses
    Open3D's own `set_max_threads`, whose effect is readable, and the record carries the
    reading.

    So the verdict is derived from the measured spreads rather than written next to them,
    and this checks the derivation still holds.
    """
    conditions = record["cause"]["conditions"]
    default = conditions["default"]["surface_area"]["relative_spread"]
    single = conditions["thread_limited"]["surface_area"]["relative_spread"]
    verdict = record["what_the_remedies_do"]["limiting_threads"]

    assert "Does not fix it" in verdict
    assert f"{default:.2e}" in verdict, verdict
    assert f"{single:.2e}" in verdict, verdict
    assert f"factor of {default / single:.1f}" in verdict, verdict
    # The direction has to match the numbers, because a factor below one reads as an
    # improvement to anyone skimming.
    assert ("smaller" if single < default else "larger") in verdict, verdict

    engaged = conditions["thread_limited"]
    assert engaged["the_knob_engaged"] is True, (
        "the thread limit did not take, so this condition measures nothing -- which is "
        "exactly the defect that made the previous OMP_NUM_THREADS experiment void"
    )
    assert engaged["max_threads_observed_during"] == 1
    assert engaged["max_threads_before"] != 1
    assert f"({engaged['max_threads_before']} threads to 1)" in verdict, verdict


def test_the_void_experiment_is_recorded_as_void(record) -> None:
    """A reviewer found the earlier experiment could not have worked; that is evidence.

    The lesson is cheap to state and was expensive to learn: check that the knob turned
    before believing the result. So the build facts that made `OMP_NUM_THREADS` inert are
    recorded, not just the replacement.
    """
    void = record["cause"]["why_an_earlier_experiment_here_was_void"]
    assert void["openmp_runtime_linked"] is False
    assert void["openmp_symbols_present"] is False
    assert void["so_omp_num_threads_is_not_a_knob_here"] is True
    assert void["binary"], void


def test_the_conclusion_is_not_in_danger_and_the_record_shows_the_arithmetic(
    record,
) -> None:
    """A reproducibility finding that did change a conclusion would be a different PR.

    The point of publishing the margin beside the drift is that a reader can see which
    kind of finding this is without taking my word for it.
    """
    unaffected = record["does_this_change_any_conclusion"]
    margin = unaffected["margin_against_the_threshold"]
    observed = unaffected["observed_difference"]

    published = record["the_published_figures"]["values"][
        "visible_with_only_its_own_self_occlusion"
    ]
    assert margin == pytest.approx(published - MINIMUM_VISIBLE_AREA, abs=1e-12)
    assert observed > 0.0
    assert unaffected["ratio"] == pytest.approx(margin / observed, rel=1e-9)
    # Three orders is the weakest form of "the drift cannot reach the threshold".
    assert unaffected["ratio"] > 1000.0, unaffected


#: The figures this record is about. Scoped to known non-reproducible quantities rather
#: than written as a style rule about floats.
NON_REPRODUCIBLE_LITERALS = ("0.7848810005123376", "0.3521503113541809")


def _scanned_files() -> list[Path]:
    """Every Python file in the repository.

    This went through three scopes, each widened by a reviewer finding a pin outside it.
    `npa/tests` missed the 36 `test_*.py` files elsewhere, including `npa/src/npa/smoke/`,
    which holds the golden-eval modules that run in containers. `test_*.py` repo-wide plus
    the harness directory still missed a pin in any file not named that way, which they
    demonstrated.

    So the scope is now every `.py` file, which is the only scope with no boundary to
    evade. It costs about a second: discovery is 3700 paths, and a file is parsed only if
    it contains one of the figures, which two do.
    """
    return [
        path
        for path in REPO.rglob("*.py")
        if not any(
            part in {".venv", "node_modules", ".git", "__pycache__"}
            for part in path.parts
        )
    ]


def _tolerance_floor() -> float:
    """The deviation below which a tolerance is exact equality with extra steps.

    A reviewer showed the first version rejected exactly one value: `abs=0`. `abs=1e-300`
    is the same pin spelled differently, and so is any tolerance smaller than the movement
    the figure actually has.

    They suggested the default condition's `absolute_spread`, and that figure is in the
    record -- but it is a spread of surface *area*, around 157 units, while these figures
    are fractions near 0.78. Using it would have compared a fraction's tolerance against
    an area's spread. The right quantity is the movement observed in the published
    figures themselves, which is also in the record, and it keeps the disclosed honest
    example (`abs=1e-4`) on the correct side.
    """
    record = json.loads(RECORD.read_text())
    return record["the_published_figures"]["absolute_difference_on_the_headline"]


def _exact_pins(tree: ast.AST, literals: tuple[str, ...]) -> list[tuple[int, str]]:
    """Exact comparisons against these figures, including through a named constant.

    An AST pass rather than a line scan, because extracting a literal to a name is the
    most ordinary refactor there is and it removes a line-based guard without touching
    its logic. A reviewer demonstrated exactly that::

        EXPECTED = 0.7848810005123376
        assert value == EXPECTED        # the `==` line no longer holds the literal

    So names bound to one of these figures are resolved first, and comparisons against
    those names count as pins. `approx(..., abs=0)` is caught too: it is exact equality
    wearing the word that buys the exemption.
    """
    values = {float(literal) for literal in literals}
    floor = _tolerance_floor()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        holds_a_figure = False
        if isinstance(node.value, ast.Constant):
            holds_a_figure = (
                isinstance(node.value.value, float) and node.value.value in values
            )
        elif isinstance(node.value, (ast.Tuple, ast.List, ast.Set)):
            # A name bound to a container holding a figure is the nearest neighbour of
            # the extract-a-constant refactor, and a reviewer supplied this case with
            # both halves of the fix after their first attempt at it did not work.
            holds_a_figure = any(
                isinstance(element, ast.Constant)
                and isinstance(element.value, float)
                and element.value in values
                for element in node.value.elts
            )
        if holds_a_figure:
            bound.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )

    def names_a_figure(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, float) and node.value in values
        if isinstance(node, ast.Name):
            return node.id in bound
        return False

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            exact = any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
            operands = [node.left, *node.comparators]
            if exact and any(names_a_figure(operand) for operand in operands):
                found.append((node.lineno, "exact comparison"))
            # Membership in a literal container is exact equality spelled differently,
            # which is the form a reviewer reached for once `==` was covered.
            member = any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
            containers = [
                element
                for operand in node.comparators
                if isinstance(operand, (ast.Tuple, ast.List, ast.Set))
                for element in operand.elts
            ]
            # Both halves are needed: binding the name is not enough unless membership
            # also looks at bare names, not only at literal containers.
            if member and (
                any(names_a_figure(element) for element in containers)
                or any(names_a_figure(operand) for operand in node.comparators)
            ):
                found.append((node.lineno, "membership in a container of figures"))
        if isinstance(node, ast.Call):
            callee = node.func
            name = (
                callee.attr
                if isinstance(callee, ast.Attribute)
                else getattr(callee, "id", "")
            )
            if name != "approx" or not any(names_a_figure(arg) for arg in node.args):
                continue
            for keyword in node.keywords:
                if keyword.arg not in {"abs", "rel"} or not isinstance(
                    keyword.value, ast.Constant
                ):
                    continue
                tolerance = keyword.value.value
                if not isinstance(tolerance, (int, float)) or isinstance(
                    tolerance, bool
                ):
                    continue
                reference = floor if keyword.arg == "abs" else floor / max(values)
                if tolerance < reference:
                    found.append(
                        (
                            node.lineno,
                            f"approx with {keyword.arg}={tolerance!r}, below the "
                            f"{reference:.3e} these figures actually move",
                        )
                    )
    return found


def test_the_published_figures_are_never_pinned_by_exact_equality() -> None:
    """The guard that outlives the finding.

    A figure that does not reproduce must not be asserted exactly. Such a test passes on
    the machine that wrote the record and fails on the next, and because it looks like a
    tightening rather than a mistake, the likely repair is regenerating the record to
    match -- which buries the finding this PR exists to record.

    A reviewer attacked the first version and found six ways through. All six are closed:
    the extract-a-constant refactor, `approx(..., abs=0)`, membership in a literal
    container, test modules outside `npa/tests`, the harness directory, and -- the last
    one, which was still open at the previous head -- a pin in a file not named
    `test_*.py`. Two classes they raised afterwards are closed too: a name bound to a
    container of figures, and a tolerance that is tiny rather than zero.

    What is still not reached: a figure arrived at through an attribute, a dict value, or
    a function return. Only direct literals, names bound to them, and names bound to
    containers of them are resolved.
    """
    offenders: list[str] = []
    for path in _scanned_files():
        text = path.read_text()
        if not any(literal in text for literal in NON_REPRODUCIBLE_LITERALS):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for lineno, kind in _exact_pins(tree, NON_REPRODUCIBLE_LITERALS):
            offenders.append(f"{path.relative_to(REPO)}:{lineno}: {kind}")
    assert not offenders, "a non-reproducible figure is pinned exactly:\n" + "\n".join(
        offenders
    )


def test_the_harness_and_record_are_committed_together(record) -> None:
    """A record whose harness is absent cannot be re-measured by a reviewer."""
    assert HARNESS.exists()
    assert record["environment"]["open3d"].startswith("0.20")
    assert record["environment"]["input_points"] > 0
    assert record["environment"]["poisson_depth"] >= 8
