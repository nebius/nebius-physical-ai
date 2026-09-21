# BEHAVIOR Comet profile qualification

This reusable GPU smoke loads a private, fully verified Comet checkpoint twice in separate processes and compares the first action for one exact synthetic observation for task 1, picking up trash. The private workflow wrapper supplies checkpoint, source, runtime, and storage locations; this directory contains no model weights or infrastructure identifiers.

Stage the public `comet_policy.py`, `comet_server.py`, and both checkpoint inventory JSON files beside these scripts. Invoke `qualify_comet.py` with `--profile comet12` or `--profile comet50`, the pinned CPython 3.11 runtime, the clean Comet source checkout, an extracted checkpoint, its full-readback fetch receipt, the frozen observation NPZ, `--expected-observation-sha256`, and a new output directory. A private wrapper may also pass `--expected-action-sha256` when an earlier qualification produced an exact action on the same observation bytes.

Both profiles must produce byte-identical finite floating 23-action results across two fresh processes. Equality is over raw bytes, so signed-zero drift is rejected. The reusable core does not embed a campaign observation or expected action. The private Comet12 wrapper binds the earlier 3abb observation and first-action identities; Comet50 has no historical action to compare.

The receipt is loader and adapter evidence. It does not run an evaluator or simulator, measure Q or success, reproduce an author benchmark, or establish compliance with the official single-24GB policy GPU limit.
