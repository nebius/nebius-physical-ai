"""Nothing on `main` reads any of the 41 Open3D evidence records. These read them.

A reviewer went looking and found that no test, script, or CI workflow opens a single one. The
only places their names appear are docstring citations in `test_open3d.py`, prose in
`skills/tools/open3d/SKILL.md`, and comments in `runner.py` -- all mentions, and a mention
guards nothing. They demonstrated it rather than inferring it: corrupting eight records in
place, every boolean flipped and every number negated, left all 120 executing tests passing.
That reproduces here on `band-validity-sweep.json`.

It matters because several are `capability-record-*.json`, and those assert hardware, image
digests, artifact hashes, exit codes and acceptance checks. A capability record that can drift
away from reality with no gate noticing is a weaker artifact than it looks.

What these tests deliberately do **not** do is assert each record's contents against a
hardcoded copy. That would double every figure and make the test the thing that drifts. Each
check below derives what it expects from something outside the record: the bytes of a file the
record names, a figure some other file cites, or an invariant the field's own name implies.

Coverage is uneven, and rather than leave a reader to assume otherwise, the figures live in
`docs/evidence/open3d/evidence-record-guards.json` and the last two tests recompute them from
the catalogue. At the time of writing, of 42 records: 40 carry quantities whose range
or sign their own names imply, 7 are pinned by a precise figure cited in code or prose, 2 name
a committed file with its digest, 40 are covered by at least one of those, and **2 by
structural checks only** -- and those two contain no numbers at all, which is a floor rather
than an oversight.

Those numbers are deliberately not repeated as literals in the assertions below. Duplicating
them here and in the record would give them somewhere to drift apart, which is the defect this
file exists to answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"

#: Files that cite Open3D figures in prose or code. A figure here that resolves to exactly one
#: record binds that record, which is how the binding is derived rather than declared.
CITING_SOURCES = (
    REPO / "npa" / "src" / "npa" / "workbench" / "open3d" / "runner.py",
    REPO / "skills" / "tools" / "open3d" / "SKILL.md",
    EVIDENCE / "README.md",
    REPO / "npa" / "tests" / "workbench" / "test_open3d.py",
)

#: Field names whose value is a proportion and so cannot leave [0, 1]. Deliberately excludes
#: `ratio`, `multiple` and `_over_` names: `spacing_over_voxel` legitimately reaches 5.449, and
#: a rule that fired on it would be switched off within a week.
#:
#: `_rate$`, `^recall_of` and `^precision_` were added after measuring: they are what the
#: downstream-consumer records call their occupancy numbers, and without them the records
#: carrying this lane's actual consumer evidence had no invariant at all.
BOUNDED_NAME = re.compile(
    r"(fraction|_share$|share_of|^share|probability|rate_for|coverage$"
    r"|_rate$|^recall_of|^precision_)",
    re.I,
)

#: A difference of two proportions, which lives in [-1, 1]. `recall_delta` is genuinely
#: negative at -0.0016, so folding it into the rule above would have failed on correct data --
#: which is how I found it.
DELTA_NAME = re.compile(r"^recall_delta$|^precision_delta$|_rate_delta$", re.I)

#: Field names that count things. `_cells$` is here *after* the proportion test deliberately:
#: `recall_of_observed_cells` ends in `_cells` and is a rate, not a count, so a rule checked in
#: the other order would reject a correct 0.96.
COUNT_NAME = re.compile(
    r"(^samples$|_count$|^count$|triangles|vertices|observations|^n_|frames?_logged"
    r"|_cells$|_rows$|^reads$|_emitted$|_scanned$|^occurrences|^leaves_)",
    re.I,
)

#: Magnitudes that cannot be negative without being nonsense: a volume, a byte size, a spatial
#: extent, an aspect. None is bounded above, so only the sign is checkable.
NON_NEGATIVE_NAME = re.compile(r"(_m3$|^size$|_span$|_aspect$)", re.I)

#: Fit ratios in the viewer capability record are keyed by human view names -- "Observed scan
#: only" -- so no field-name rule can reach them. Their *path* can: a fit ratio is required
#: extent over available extent, and zero or negative is impossible.
FIT_PATH = re.compile(r"fit_measured|_fit_|/fit$|fit_ratio", re.I)

#: Four or more decimals, so a figure precise enough that its appearance in two files is a
#: citation rather than a coincidence.
PRECISE_FIGURE = re.compile(r"(?<![\w.])0\.\d{4,}(?![\w.])")


def records() -> list[Path]:
    found = sorted(EVIDENCE.rglob("*.json"))
    assert found, f"no evidence records under {EVIDENCE}"
    return found


def record_key(path: Path) -> str:
    """Identify a record by its path below the evidence root, never by its basename.

    `ordinary-window-audit.json` exists twice, once at the top level and once under
    `viewer-ui/`, byte for byte identical. A basename-keyed index of the catalogue therefore
    holds 40 entries for 41 files and silently drops one, which is how the first version of
    this file came to assert a count of 41 against a set of 40.
    """
    return path.relative_to(EVIDENCE).as_posix()


def scalars(node: object, path: str = "") -> list[tuple[str, object]]:
    """Every non-string scalar with the path that reaches it."""
    if isinstance(node, dict):
        return [
            leaf
            for key, value in node.items()
            for leaf in scalars(value, f"{path}/{key}")
        ]
    if isinstance(node, list):
        return [
            leaf
            for index, value in enumerate(node)
            for leaf in scalars(value, f"{path}[{index}]")
        ]
    return [] if isinstance(node, str) else [(path, node)]


def numbers_in(path: Path) -> set[float]:
    return {
        float(value)
        for _, value in scalars(json.loads(path.read_text()))
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def field_name(path: str) -> str:
    return path.rsplit("/", 1)[-1].split("[")[0]


@pytest.fixture(scope="module")
def loaded() -> dict[Path, object]:
    return {path: json.loads(path.read_text()) for path in records()}


def test_every_record_parses_and_holds_something(loaded: dict) -> None:
    """The cheapest check there is, and the one that covers all 41.

    A record regenerated into an empty object or truncated mid-write is the failure mode this
    catches, and nothing caught it before.
    """
    empty = [record_key(path) for path, data in loaded.items() if not data]
    assert empty == [], empty


def test_no_record_carries_a_non_finite_number(loaded: dict) -> None:
    """`json.dumps` writes `NaN` and `Infinity` happily, and `json.loads` reads them back.

    So a measurement that divided by zero can reach a committed record and be read out again
    without any parser objecting.
    """
    offenders = [
        f"{record_key(path)}{where} = {value}"
        for path, data in loaded.items()
        for where, value in scalars(data)
        if isinstance(value, float) and not math.isfinite(value)
    ]
    assert offenders == [], offenders


def invariant_for(path: str) -> str | None:
    """Which invariant, if any, a recorded quantity is subject to, from how it is named.

    The order is load-bearing. `recall_delta` is a signed difference and
    `recall_of_observed_cells` is a rate whose name ends in `_cells`, so testing the delta rule
    first and the proportion rule before the count rule is what keeps both correct. I found
    both by measuring candidate rules against the committed catalogue rather than by reading
    the names, and each one rejected correct data on the first attempt.
    """
    name = field_name(path)
    if DELTA_NAME.search(name):
        return "delta"
    if BOUNDED_NAME.search(name):
        return "proportion"
    if COUNT_NAME.search(name):
        return "count"
    if NON_NEGATIVE_NAME.search(name):
        return "non_negative"
    if FIT_PATH.search(path):
        return "fit_ratio"
    return None


def satisfies(kind: str, value: float) -> bool:
    """Whether a value is consistent with the invariant its name implies."""
    if kind == "proportion":
        # -0.0 is not a share of anything. It only arises from negating a zero, which is
        # exactly how the one record that resisted the reviewer's corruption resisted it:
        # `fraction_beyond_one_voxel = 0.0` negates to a value still inside [0, 1].
        return 0.0 <= value <= 1.0 and not (
            value == 0.0 and math.copysign(1.0, value) < 0
        )
    if kind == "delta":
        return -1.0 <= value <= 1.0
    if kind == "count":
        return value >= 0 and value == int(value)
    if kind == "fit_ratio":
        return value > 0
    return value >= 0


def classified_quantities(
    data: object, record: str | None = None
) -> list[tuple[str, str, float]]:
    """Each quantity in a record that some invariant applies to, with its path and kind.

    One exclusion, and it is not a convenience. The coverage block of
    `evidence-record-guards.json` is a summary *of this measurement*, and its field names
    describe the invariant classes -- so `values_checked_by_invariant.count` is itself matched
    as a count and `fit_ratio` as a fit ratio. Counting them makes the published total a fixed
    point that moves every time it is written: I measured 1093, wrote it down, and the act of
    writing it made the answer 1095. The block is held to its arithmetic by
    `test_the_published_coverage_figures_are_the_measured_ones`, which is a stronger check
    than a range rule anyway.
    """
    summary_block = record == record_key(COVERAGE_RECORD)
    found = []
    for where, value in scalars(data):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if summary_block and where.startswith("/coverage/"):
            continue
        kind = invariant_for(where)
        if kind:
            found.append((where, kind, float(value)))
    return found


def invariant_violations(data: object, record: str | None = None) -> list[str]:
    """Every recorded quantity that contradicts the invariant its own name implies."""
    return [
        f"{kind}: {where} = {value}"
        for where, kind, value in classified_quantities(data, record)
        if not satisfies(kind, value)
    ]


def test_every_recorded_quantity_satisfies_the_invariant_its_name_implies(
    loaded: dict,
) -> None:
    """1093 values across 40 records, checked against what their own names mean.

    This is what catches the corruption the reviewer demonstrated on records that no citation
    pins: negating a proportion puts it outside [0, 1], negating a count makes it negative,
    negating a fit ratio makes it impossible. The rules carry domain meaning rather than
    bytes, which is why they are worth more than a digest manifest over the same files -- a
    manifest would also fire on every legitimate regeneration.
    """
    offenders = [
        f"{record_key(path)} {violation}"
        for path, data in loaded.items()
        for violation in invariant_violations(data, record_key(path))
    ]
    assert offenders == [], offenders


def test_the_invariants_actually_detect_the_demonstrated_corruption() -> None:
    """A guard nobody has seen fail is a guard nobody has seen.

    This applies the reviewer's exact corruption -- every boolean flipped, every number negated
    -- to each record in turn and requires an invariant to object.

    It now objects for every record the invariants reach. It did not at first: with only the
    proportion and count rules, `complete-capture-threshold-sweep.json` survived, because its
    one bounded field is `fraction_beyond_one_voxel = 0.0` and negating zero leaves a value
    inside [0, 1]. Immunity by way of a zero rather than by way of the record being sound,
    which is why `satisfies` rejects a negative zero share.
    """

    def wreck(node: object) -> object:
        if isinstance(node, dict):
            return {key: wreck(value) for key, value in node.items()}
        if isinstance(node, list):
            return [wreck(value) for value in node]
        if isinstance(node, bool):
            return not node
        if isinstance(node, (int, float)):
            return -node
        return node

    reached = {
        record_key(path)
        for path in records()
        if classified_quantities(json.loads(path.read_text()), record_key(path))
    }
    caught = {
        record_key(path)
        for path in records()
        if invariant_violations(wreck(json.loads(path.read_text())), record_key(path))
    }

    assert reached - caught == set(), sorted(reached - caught)
    assert len(caught) >= 36, (
        len(caught),
        "the invariants reach fewer records than measured",
    )


def test_the_one_duplicated_filename_stays_a_duplicate_rather_than_a_divergence() -> (
    None
):
    """`ordinary-window-audit.json` is committed twice, and `README.md` cites it ambiguously.

    Two copies with one name are harmless while they are identical and actively confusing the
    moment they are not: the README's "measured in `ordinary-window-audit.json`" would then
    point at two different measurements. This does not delete either copy -- both are committed
    evidence and the ambiguous reference does not say which is meant -- but it does hold them
    equal, so a later edit to one has to confront the other.
    """
    duplicated = [
        path for path in records() if path.name == "ordinary-window-audit.json"
    ]
    assert len(duplicated) == 2, [record_key(path) for path in duplicated]
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in duplicated}
    assert len(digests) == 1, {
        record_key(path): path.read_text()[:80] for path in duplicated
    }

    others = [
        path.name for path in records() if path.name != "ordinary-window-audit.json"
    ]
    assert len(others) == len(set(others)), (
        "a second filename is now duplicated; name it here"
    )


def test_every_digest_a_record_pairs_with_a_committed_file_matches_its_bytes(
    loaded: dict,
) -> None:
    """Where a record names a file it can be checked against, check it.

    Only pairs whose file is actually committed are checked. Several records deliberately name
    artifacts that are not -- one says so in the field itself, "retained in the lane evidence
    pack [...] and deliberately not committed" -- and a guard that failed on those would be
    demanding that scan data be committed to the repository.
    """
    checked = 0
    mismatches = []
    for path, data in loaded.items():

        def walk(node: object) -> None:
            nonlocal checked
            if isinstance(node, dict):
                named = next(
                    (node[k] for k in ("path", "file") if isinstance(node.get(k), str)),
                    None,
                )
                digest = node.get("sha256")
                if named and isinstance(digest, str):
                    target = EVIDENCE / named
                    if target.is_file():
                        checked += 1
                        actual = hashlib.sha256(target.read_bytes()).hexdigest()
                        if actual != digest.removeprefix("sha256:"):
                            mismatches.append(
                                f"{record_key(path)}: {named} -> {actual[:12]} != {digest[:12]}"
                            )
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(data)

    assert mismatches == [], mismatches
    assert checked >= 4, (
        checked,
        "fewer digest pairs resolve than before; a file may have moved",
    )


def citation_bindings() -> list[tuple[str, str, str]]:
    """Precise figures in code or prose that resolve to exactly one record.

    The binding is derived here rather than taken from the docstring that cites a record by
    name, because those citations are looser than they look: three figures in `test_open3d.py`
    docstrings that name a record are not in the record they name -- one is another lane's
    number, one is a descriptor. Deriving the binding means a figure from elsewhere simply
    falls out of scope instead of producing a false failure.
    """
    table = {record_key(path): numbers_in(path) for path in records()}
    bindings = []
    for source in CITING_SOURCES:
        if not source.is_file():
            continue
        for literal in sorted(set(PRECISE_FIGURE.findall(source.read_text()))):
            decimals = len(literal.split(".")[1])
            value = float(literal)
            holders = [
                name
                for name, values in table.items()
                if any(round(held, decimals) == value for held in values)
            ]
            if len(holders) == 1:
                bindings.append((source.name, literal, holders[0]))
    return bindings


def test_figures_quoted_in_code_and_prose_are_still_in_the_record_that_holds_them() -> (
    None
):
    """The silent-decay class, in its prose form.

    #1004 was a literal duplicated between a product and four harnesses that drifted when the
    product renamed it. This is the same shape: a figure measured once, then written into a
    docstring, a skill page and a source comment. Regenerate the record with different numbers
    and the prose keeps the old ones, saying nothing.

    This is the only check here that catches a believable edit. Re-measuring one fraction in
    `poisson-procedure-and-overcall-audit.json` by 4 percent, from 0.009108 to 0.009473, leaves
    a finite number inside [0, 1] that every other test in this file accepts; `test_open3d.py`
    goes on quoting 0.00911. Only this test fails.
    """
    bindings = citation_bindings()
    assert len(bindings) >= 29, (
        len(bindings),
        "fewer figures resolve to exactly one record than when this was written; either a "
        "record changed or a citation was edited, and both are worth a look",
    )

    table = {record_key(path): numbers_in(path) for path in records()}
    broken = []
    for source, literal, record in bindings:
        decimals = len(literal.split(".")[1])
        value = float(literal)
        if not any(round(held, decimals) == value for held in table[record]):
            broken.append(f"{source} quotes {literal}, absent from {record}")
    assert broken == [], broken


#: The record that publishes this file's coverage. It is the one record here whose own figures
#: a test recomputes, which is why it counts as covered below.
COVERAGE_RECORD = EVIDENCE / "evidence-record-guards.json"


def covered_records() -> tuple[set[str], dict[str, int]]:
    """Which records something here actually checks, and the tallies that go with it.

    A record counts as covered when one of four things checks it: a range or sign invariant on
    a field whose name implies one, a digest paired with committed bytes, a precise figure
    quoted elsewhere, or -- for the coverage record alone -- a test that recomputes its
    contents. That last clause is why it is listed as covered rather than taking credit for
    itself.
    """
    with_invariant: set[str] = set()
    with_digest: set[str] = set()
    digest_pairs = 0
    by_kind: dict[str, int] = {}

    for path in records():
        data = json.loads(path.read_text())
        for _, kind, _value in classified_quantities(data, record_key(path)):
            by_kind[kind] = by_kind.get(kind, 0) + 1
            with_invariant.add(record_key(path))

        def walk(node: object, this: Path = path) -> None:
            nonlocal digest_pairs
            if isinstance(node, dict):
                named = next(
                    (node[k] for k in ("path", "file") if isinstance(node.get(k), str)),
                    None,
                )
                if named and isinstance(node.get("sha256"), str):
                    with_digest.add(record_key(this))
                    if (EVIDENCE / named).is_file():
                        digest_pairs += 1
                for value in node.values():
                    walk(value, this)
            elif isinstance(node, list):
                for value in node:
                    walk(value, this)

        walk(data)

    bindings = citation_bindings()
    with_citation = {record for _, _, record in bindings}
    covered = (
        with_invariant | with_digest | with_citation | {record_key(COVERAGE_RECORD)}
    )

    return covered, {
        "records_total": len(records()),
        "records_with_range_or_sign_invariants": len(with_invariant),
        "records_pinned_by_a_precise_figure_cited_elsewhere": len(with_citation),
        "figure_bindings_derived": len(bindings),
        "records_naming_a_committed_file_with_its_digest": len(with_digest),
        "digest_pairs_that_resolve_to_committed_bytes": digest_pairs,
        "records_with_at_least_one_meaning_bearing_check": len(covered),
        "records_with_structural_checks_only": len(records()) - len(covered),
        "values_checked_by_invariant": dict(sorted(by_kind.items())),
        "values_checked_in_total": sum(by_kind.values()),
    }


def test_the_published_coverage_figures_are_the_measured_ones() -> None:
    """The coverage claim is recomputed rather than restated.

    Writing these ten numbers into both the record and the test would give them somewhere to
    drift apart, which is the defect this whole file answers. So the record publishes them and
    this recomputes them from the catalogue. Editing the record without re-measuring fails
    here, and so does adding a record without extending coverage.
    """
    published = json.loads(COVERAGE_RECORD.read_text())["coverage"]
    _, measured = covered_records()

    assert set(published) == set(measured), {
        "only_published": sorted(set(published) - set(measured)),
        "only_measured": sorted(set(measured) - set(published)),
    }
    disagreements = {
        field: {"published": published[field], "measured": measured[field]}
        for field in measured
        if published[field] != measured[field]
    }
    assert disagreements == {}, disagreements


def test_the_unguarded_remainder_is_named_and_not_just_counted() -> None:
    """The records checked for parsing and little else should be nameable, not just counted.

    A coverage figure on its own invites "two thirds is most of it". Naming them makes it
    concrete that the remainder still includes a capability record and several viewer-UI
    audits, and it means a newly added record cannot join that remainder unnoticed.
    """
    covered, measured = covered_records()
    remainder = sorted({record_key(path) for path in records()} - covered)

    assert len(remainder) == measured["records_with_structural_checks_only"], remainder
    published = json.loads(COVERAGE_RECORD.read_text())["what_is_still_unguarded"]
    assert published["structural_only_records"] == len(remainder), (
        published["structural_only_records"],
        remainder,
    )
