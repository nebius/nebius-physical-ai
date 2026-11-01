# Claim: the Open3D evidence catalogue gets a reader

## Scope

`docs/evidence/open3d/**/*.json` and a new test module that reads it. No product source, no
existing evidence record, and no change to `test_open3d.py`.

## The gap

Independent review found that nothing on `main` opens any of the 41 committed Open3D evidence
records. Their names appear in `test_open3d.py` docstrings, `skills/tools/open3d/SKILL.md` and
comments in `runner.py` — all mentions. They demonstrated it rather than inferring it:
corrupting eight records in place, every boolean flipped and every number negated, left all 120
executing tests passing. I reproduced that independently on `band-validity-sweep.json` before
writing anything.

It matters most for the `capability-record-*.json` files, which assert hardware, image digests,
artifact hashes, exit codes and acceptance checks.

## Acceptance, frozen before building

1. The reviewer's exact corruption, applied to a record, fails at least one test.
2. A *believable* edit — one cited fraction re-measured by a few percent, still finite, still
   inside [0, 1] — fails at least one test. This is the one that distinguishes a real guard
   from a parse check.
3. No check asserts a record's contents against a copy of those contents held in the test.
   Every expectation is derived from something outside the record.
4. No false alarm on the catalogue as committed.
5. The coverage the guard provides is published as measured numbers that a test recomputes, and
   a record that nothing checks cannot be added without those numbers being brought up to date.

## What I publish if it fails

The measured coverage and the mutations that got through, including the case where the only
honest answer is that a large part of the catalogue is not guardable without either a digest
manifest or a schema that the records do not currently share.

## Result

All five met. The mutations, their count, and the coverage figures are published in
`docs/evidence/open3d/evidence-record-guards.json`, recomputed by
`test_the_published_coverage_figures_are_the_measured_ones`, and reproducible with
`npa/.venv/bin/python docs/evidence/open3d/harness/evidence_guard_mutations.py`.

Coverage is **40 of 42** records with a meaning-bearing check and **2** with structural checks
only. That remainder is named by a test, not just counted, and the record states plainly that
this does not make the catalogue self-verifying.

Criterion 2 is met as written — a cited fraction re-measured by a few percent fails a test —
and the honest generalisation is narrower than the criterion sounds. Measured across the
catalogue, an edit that keeps every value legitimate for its own type is caught in **18 of the
40** records that contain numbers: **7** resist because a figure in them is quoted outside the
record, and **11** resist through another cross-check. A range invariant cannot detect an
in-range value, so the rest are a floor of the mechanism and not an oversight.
`what_a_believable_edit_actually_resists` in the record carries the measurement, including a
migration defect this sweep found and closed.

### Three figures in this section were wrong, and nothing read them

The previous version of this section said *"Seven mutations applied, seven caught"* and
*"Coverage is 29 of 42 records with a meaning-bearing check and 13 with structural checks
only"*. All three numbers were wrong, and all three understated the work:

| claimed here | actual |
|---|---|
| 7 mutations | **13** |
| 29 records with a meaning-bearing check | **40** |
| 13 with structural checks only | **2** |

Twenty-nine is the number of *figure bindings*, not records; those 29 bindings fall across 7
records. The prose then derived 42 − 29 = 13 for the remainder. Review found the coverage
errors by running `covered_records()` on this head; I found the stale mutation count while
fixing them, after adding two mutations and not revisiting this file.

This is the finding of this very claim, one level up. The gap was evidence published where
nothing reads it; the fix shipped a reader for the records and then restated its result in
prose that nothing read, and the restatement was wrong. So the restatement now has a reader
too: `test_the_claim_document_agrees_with_the_measured_coverage` recomputes these figures and
compares them against this file. The numbers above are not maintained by hand any more.

## Two things found while measuring, neither fixed here

- `ordinary-window-audit.json` is committed twice, byte-identical, and `README.md` cites it by
  bare filename. Held equal by a test rather than deleted; which copy should go is review's.
- Three figures in `test_open3d.py` docstrings are not in the record the docstring names. One
  is a descriptor, one is attributed in the prose itself to another lane. Recorded, not edited:
  several open PRs in this lane touch that file and a prose-only change would conflict for no
  functional gain.

## Not in scope

Deleting either duplicate, rewording `test_open3d.py`, and a digest manifest over all 42
records. The last is a different decision — it reports every edit, including legitimate
regeneration — and belongs to review if they want acknowledge-on-change.
