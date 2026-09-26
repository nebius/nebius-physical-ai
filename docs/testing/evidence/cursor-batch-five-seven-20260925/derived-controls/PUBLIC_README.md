# Derived replay controls

This harness exercises production workflow replay using a private copy of a completed GPU run's verified artifact readback. Its fault controls are **derived integration evidence, not native GPU events**. Each control writes only to its own local copy; provider and network callbacks must remain unused.

Nine controls cover unchanged completed replay, changed/missing workflow/source/image identity, and absent/corrupt copied outputs. Identity failures must occur before the output checker. Artifact failures use a strict injected hash/readback checker that raises when bytes cannot be verified; they do not claim the native boolean missing-output recovery policy forbids resubmission.

The separate synthetic workflow uses two real catalog toolRefs to prove effective image mappings through the production renderer. It verifies distinct images, stable mappings and identity after map reorder, and changed mappings and identity after selector swaps. It does not run inference or GPUs.

Run the author checks with the candidate's repository virtualenv and `PYTHONPATH` pointing to its `npa/src` and this harness directory:

```text
npa/.venv/bin/python -m pytest test_author_controls.py -q
```

Run the renderer control using an exact, clean candidate revision:

```text
npa/.venv/bin/python derived_controls.py --candidate CANDIDATE --source-sha REVISION --output NEW_PRIVATE_OUTPUT
```

Adding `--bundle PRIVATE_BUNDLE` executes the nine copied-evidence controls. That private bundle must bind a complete object readback inventory, byte hashes, resolved workflow, staged source, exact effective image-selection options and completed runtime state. The harness rejects absent or inconsistent evidence. The live bundle is deliberately excluded from this generic packet.

`public-report.json` records the revision, source-module hashes, artifact-inventory hash, typed outcomes, zero-call counters and synthetic selector result. `selector-fixture.readiness.json` distinguishes validated planning from unperformed live prerequisites. Author tests use explicitly synthetic completed records; they are separate from the executed controls over real copied evidence.
