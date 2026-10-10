# RTX configuration search receipts

These are exploratory measurements, separate from the final repeated comparison
in the parent directory. Failed candidates are retained deliberately.

- `m*-n*-t*-s*-r*.json`: all 40 configurations, each tested on six representative
  shapes. Names encode tile M/N, threads, stages and Q-in-registers. Reports
  include resource rejection, numerical failure, eager timings, CUDA graph
  timings and kernel identities. A configuration with any numerical failure was
  excluded from selection.
- `confirm-*.json`: four additional containers checking the selected register
  and cross-attention configurations on all 12 shapes.
- `backward-*.json`: separate FA2/native-FA4/upstream-proposal comparison. The
  proposal was not adopted; its exact revision and mounted source hashes are
  recorded in the candidate reports and `backward-provenance.json`.
- `*.py.txt`: verbatim research workers retained to bind their recorded source
  hashes. They are archival evidence, not supported application APIs. Remove
  the final `.txt` when reproducing in an isolated experimental environment.
  The supported helper and qualification worker live under `npa/`.

The sweep worker copies the pinned upstream forward function into a private
namespace and compile cache before changing constructor settings. It does not
mutate the global FA4 dispatcher. Each configuration ran in its own container.
Provide the JSON `configuration` from the desired report to `sweep.py` with
`--configuration` and an `--output` path. Use `--all-cases` for the confirmation
reports, and put `direct_attention.py` beside the worker for direct candidates.
The supporting benchmark files were from `dc649703e5e18fcc147fd930da752b8adb3f180d`;
all source hashes are included. Final confirmation workers in the parent
reports instead use `b002a53e32d7d149219ec45a11f96ec7d8b42904`.

The initial grid used five warmups and five blocks of 25 eager iterations. CUDA
graph timing captured ten operations and replayed the graph ten times per
sample, with five samples. These short exploratory eager timings were sensitive
to launch/clock transients and are not pooled with the final 1,000-warmup,
200-iteration measurements. Graph times help separate device work from Python
launch overhead; they do not predict uncaptured full-model latency.
