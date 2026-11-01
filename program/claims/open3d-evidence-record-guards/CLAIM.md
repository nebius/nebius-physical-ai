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

All five met. Seven mutations applied, seven caught, recorded in
`docs/evidence/open3d/evidence-record-guards.json` and reproducible with
`npa/.venv/bin/python docs/evidence/open3d/harness/evidence_guard_mutations.py`.

Coverage is 29 of 42 records with a meaning-bearing check and 13 with structural checks only.
That remainder is named by a test, not just counted, and the record states plainly that this
does not make the catalogue self-verifying.

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
