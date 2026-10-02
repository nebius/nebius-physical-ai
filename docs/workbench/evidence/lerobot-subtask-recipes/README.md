# Historical recording recipe

The adjacent SHA-256-named Python file is an **archived recipe, not runtime code**.
It preserves `npa/src/npa/fiftyone_lerobot_subtasks.py` from commit
`efec5e8b6af10f430170bd945325f8a54c2fc3a1` byte for byte.

Both the [synthetic recording](../lerobot-subtasks/README.md) and the
[real SO100 recording](../lerobot-video-subtasks/README.md) bind this exact source
digest in their original manifests and RRD provenance. Their MP4s, RRDs, manifests,
and capture receipts remain unchanged. Current exporter fixes must not relabel
those older screen recordings as new runs.

Historical evidence tests check the archived file against the original digest;
other unchanged recipe files are checked in place. Fresh producer tests execute
the current implementation. The separate native FiftyOne integration test
exercises current import, export, episode identity, and re-import behavior.

The `0e489dd…` recipe preserves the original recording producer from commit
`b1d1ee592e7b0e47bb328de77e7aa8be12f48608`. The current producer uses the
Rerun 0.38 streaming decoder; the archived producer and its original artifacts
remain evidence for the recorded 0.31.4 run.
