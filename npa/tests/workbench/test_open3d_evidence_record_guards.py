"""Apply catalogue-wide semantic checks to the Open3D evidence records.

The existing reference auditor binds committed files to their recorded digests. These tests
add checks for finite values, name-implied numeric invariants, cited figures, and duplicate
records. They intentionally avoid a frozen catalogue size or a second digest implementation:
new evidence can join without editing this module, and file/hash semantics stay owned by the
reference auditor.

It matters because several are `capability-record-*.json`, and those assert hardware, image
digests, artifact hashes, exit codes and acceptance checks. A capability record that can drift
away from reality with no gate noticing is a weaker artifact than it looks.

What these tests deliberately do **not** do is assert each record's contents against a
hardcoded copy. That would double every figure and make the test the thing that drifts. Each
check below derives what it expects from something outside the record: the bytes of a file the
record names, a figure some other file cites, or an invariant the field's own name implies.

Coverage remains intentionally partial: a range check cannot detect an in-range but incorrect
measurement, and these tests do not recreate the geometry that produced a record.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
REFERENCE_AUDITOR = EVIDENCE / "harness" / "audit_evidence_references.py"

#: Field names whose value is a proportion and so cannot leave [0, 1]. Deliberately excludes
#: `ratio`, `multiple` and `_over_` names: `spacing_over_voxel` legitimately reaches 5.449, and
#: a rule that fired on it would be switched off within a week.
#:
#: `_rate$`, `^recall_of` and `^precision_` were added after measuring: they are what the
#: downstream-consumer records call their occupancy numbers, and without them the records
#: carrying this lane's actual consumer evidence had no invariant at all.
BOUNDED_NAME = re.compile(
    r"(fraction|_share$|share_of|^share|probability|rate_for|coverage$"
    r"|(?<!minority)_rate$|^recall_of|^precision_)",
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
    r"(^samples$|_count$|^count$|(?:^|_)(?:triangles|vertices|observations)$|^n_|frames?_logged"
    r"|_cells$|_rows$|^reads$|_emitted$|_scanned$|^occurrences|^leaves_)",
    re.I,
)

#: Magnitudes that cannot be negative without being nonsense: a volume, a byte size, a spatial
#: extent, an aspect. None is bounded above, so only the sign is checkable.
NON_NEGATIVE_NAME = re.compile(r"(_m3$|^size$|_span$|_aspect$)", re.I)

#: Fit ratios in the viewer capability record are keyed by human view names -- "Observed scan
#: only" -- so no field-name rule can reach them. Their *path* can: a fit ratio is required
#: extent over available extent, and zero or negative is impossible.
FIT_PATH = re.compile(r"fit_measured|/fit$|fit_ratio", re.I)


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

    The optional record argument is retained for the mutation harness API. No record receives
    a snapshot-specific exclusion.
    """
    found = []
    for where, value in scalars(data):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
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
    """Check quantities whose own names carry an unambiguous range or sign contract.

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


def test_duplicate_filenames_do_not_diverge() -> None:
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
    assert len(duplicated) in {1, 2}, [record_key(path) for path in duplicated]
    if len(duplicated) == 2:
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


def load_reference_auditor():
    """Load the catalogue's canonical file-reference and digest implementation."""
    spec = importlib.util.spec_from_file_location(
        "open3d_reference_auditor", REFERENCE_AUDITOR
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_reference_auditor_accepts_every_record(loaded: dict) -> None:
    """Where a record names a file it can be checked against, check it.

    Only pairs whose file is actually committed are checked. Several records deliberately name
    artifacts that are not -- one says so in the field itself, "retained in the lane evidence
    pack [...] and deliberately not committed" -- and a guard that failed on those would be
    demanding that scan data be committed to the repository.
    """
    auditor = load_reference_auditor()
    checked = 0
    mismatches = []
    for path, data in loaded.items():
        run_dir = (
            data.get("artifacts_committed_under") if isinstance(data, dict) else None
        )
        bound, _unbound = auditor.bound_pairs(data)
        for _where, named, claimed, _declared in bound:
            target = auditor.resolve(named, run_dir)
            if target is None:
                continue
            checked += 1
            actual = auditor.digest(target)
            if actual != claimed:
                mismatches.append(
                    f"{record_key(path)}: {named} -> {actual[:12]} != {claimed[:12]}"
                )

    assert mismatches == [], mismatches
    assert checked >= 4, (
        checked,
        "fewer digest pairs resolve than before; a file may have moved",
    )


SWEEP = REPO / "docs/evidence/open3d/harness/believable_edit_sweep.py"
MUTATIONS = REPO / "docs/evidence/open3d/harness/evidence_guard_mutations.py"


def load_sweep():
    spec = importlib.util.spec_from_file_location("believable_edit_sweep", SWEEP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mutations():
    spec = importlib.util.spec_from_file_location("evidence_guard_mutations", MUTATIONS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sweep_refuses_to_read_a_run_that_never_happened(tmp_path: Path) -> None:
    """Zero passed is not zero failed, in the file that did not know it.

    `run_guards` searched for "N failed" and returned 0 when it was absent, so a clean
    run and a run that never started produced the same value. A reviewer's first
    attempt at this sweep reported 0 caught and 40 unnoticed in both modes -- which
    reads as a devastating result about every guard in this file, and was a broken
    interpreter on their side. The pre-flight consumes the same value, so it would
    also have gone on to edit 42 records on a false premise.

    The two mutation harnesses in the same directory already require a populated pass
    count. This file was the outlier.
    """
    sweep = load_sweep()
    no_pytest = tmp_path / "python"
    no_pytest.write_text("#!/bin/sh\necho 'No module named pytest' >&2\nexit 4\n")
    no_pytest.chmod(0o755)
    sweep.PYTHON = no_pytest

    with pytest.raises(RuntimeError) as raised:
        sweep.run_guards()

    assert "never happened" in str(raised.value)


def test_the_sweep_restores_a_record_when_the_guard_run_raises(tmp_path: Path) -> None:
    """An interruption must not leave a believable-looking mutation on disk.

    The write, run, restore sequence had no `try`/`finally`, so a failure anywhere in
    the loop left one record edited. What makes that worth a test rather than a
    shrug is that in type-respecting mode the residue is legitimate-looking -- counts
    are still counts and proportions still sit in range -- so it is not something a
    reader would catch by opening the file.
    """
    sweep = load_sweep()
    record = tmp_path / "capability-record-under-test.json"
    before = json.dumps({"count": 7, "fraction": 0.25}, indent=2) + "\n"
    record.write_text(before)

    calls = {"n": 0}

    def exploding_guards() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return 0  # the pre-flight, which must pass for the loop to be reached
        raise KeyboardInterrupt("interrupted mid-sweep")

    class OneRecord:
        @staticmethod
        def records() -> list[Path]:
            return [record]

        @staticmethod
        def record_key(path: Path) -> str:
            return path.name

        @staticmethod
        def invariant_for(path: str) -> str | None:
            return "count" if path.endswith("count") else None

    sweep.run_guards = exploding_guards
    sweep.load_guards = lambda: OneRecord()
    with pytest.raises(KeyboardInterrupt):
        sweep.main()

    assert record.read_text() == before, "the record was left mutated"


def test_mutation_harness_refuses_a_run_without_executed_tests(tmp_path: Path) -> None:
    mutations = load_mutations()
    no_pytest = tmp_path / "python"
    no_pytest.write_text("#!/bin/sh\necho 'collection failed' >&2\nexit 4\n")
    no_pytest.chmod(0o755)
    mutations.PYTHON = no_pytest

    with pytest.raises(RuntimeError, match="did not complete a test run"):
        mutations.failing_tests()


def test_mutation_harness_rejects_noop_and_restores_after_failure(
    tmp_path: Path,
) -> None:
    mutations = load_mutations()
    record = tmp_path / "record.json"
    original = '{\n  "count": 7\n}\n'
    record.write_text(original)

    with pytest.raises(RuntimeError, match="mutation did not change"):
        mutations.run_mutation("no-op", record, lambda data: data)
    assert record.read_text() == original

    def interrupted_run() -> list[str]:
        raise KeyboardInterrupt()

    mutations.failing_tests = interrupted_run
    with pytest.raises(KeyboardInterrupt):
        mutations.run_mutation("interrupted", record, lambda data: {"count": 8})
    assert record.read_text() == original


def test_no_record_restates_a_length_it_cannot_see() -> None:
    """The defect this record exists to catch, found by a reviewer inside the record.

    `mutations_the_guard_was_tested_against.count` said 9 against the 13 results beside
    it, and `verification.guard_tests` said "9 passed" against 12 tests collected.
    Nothing read either -- the guard binds `len(results)` -- so both drifted. A number
    with no reader is exactly what this pull request argues against.

    Both are deleted rather than given a second reader. This stops that shape coming
    back, and only that shape: a count whose *name* says which collection it counts must
    agree with it.

    The first version of this test compared any `*_count` against every sibling list,
    and it immediately failed on `capability-record.json`, where
    `published_object_count` is 17 and the sibling `rrd_entities` holds 6. Those are
    unrelated quantities and the record is correct. Counting something that is not a
    sibling list is ordinary, so a guard that forbids it is wrong -- the narrow rule
    below still catches `count` beside `results`, which is what drifted.
    """
    offenders: list[str] = []
    for record in records():
        payload = json.loads(record.read_text())

        def walk(node: object, path: str) -> None:
            if not isinstance(node, dict):
                if isinstance(node, list):
                    for index, item in enumerate(node):
                        walk(item, f"{path}[{index}]")
                return
            for key, value in node.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    # Only where the name says which collection it counts: a bare
                    # `count` beside `results`, or `<name>_count` beside `<name>`.
                    named = (
                        key[: -len("_count")]
                        if key.endswith("_count")
                        else ("results" if key == "count" else None)
                    )
                    sibling = node.get(named) if named else None
                    if isinstance(sibling, list) and value != len(sibling):
                        offenders.append(
                            f"{record.name}:{path}.{key} is {value} while the "
                            f"{named} it names holds {len(sibling)}"
                        )
                walk(value, f"{path}.{key}")

        walk(payload, "")

    assert not offenders, "\n".join(offenders)
