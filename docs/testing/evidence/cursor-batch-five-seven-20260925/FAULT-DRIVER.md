# Controlled driver-exit experiment

`gpu-fault-driver.py` is the exact harness used for the successful native GPU
recovery experiments. Its hash is recorded in `gpu-proof.json`. It runs outside
the candidate source tree and does not change the production checkout.

The harness requires an already authorized, isolated NPA/SkyPilot environment,
the matching frozen candidate installed in its own environment, and the
`gpu-workload` workflow configured with `post_output_verification=until-release`.
Use a fresh run and empty storage prefix for each experiment. Keep credentials
and the configuration below in operator-controlled private storage.

Set `NPA_PROOF_FAULT_CONFIG` to an absolute path to a private JSON file containing:

| Field | Value |
| --- | --- |
| `mode` | `blocked-live`, `after-cancellation-event`, or `after-runtime-marker` |
| `run_id` | Exact explicitly selected workflow run ID |
| `source_sha` | Candidate full Git commit SHA |
| `payload_sha` | Executed workload archive SHA-256 |
| `staged_source_sha256` | Content-addressed staged candidate fingerprint |
| `candidate_root` | Absolute candidate checkout path |
| `run_prefix_uri` | Exact unsigned S3 prefix for this run's workload artifacts |
| `receipt` | New absolute receipt path beneath an owned mode-0700 directory |

Invoke the harness with the candidate environment's interpreter and the same
arguments used for ordinary `npa workbench workflow submit`, including the
explicit matching `--run-id` or `--resume-run` and `--stage-src`. Follow the
[workload instructions](gpu-workload/README.md) for the pinned image, payload,
normal credential preflight, isolated state, configuration and unlimited
observation settings. The harness validates the clean candidate commit, imported
runtime path, exact run/source/payload identities and actual post-output progress
before arming. It saves an exclusive mode-0600 receipt and fsyncs before exit.

The selected exit codes are 93 after a blocked-live marker, 91 after the real
immutable cancellation event but before the mutable reuse marker, and 92 after
that mutable marker. Recover with the ordinary unhooked candidate CLI and the
same exact run, source, workflow, image and isolated state. Verify actual queue
and pod transitions, persisted records, independent CPU results and output byte
equality; the exit code alone is not acceptance evidence.

In the two cancellation-crash modes, the corrected harness explicitly invokes
production supervision on a real RUNNING observation and injects only
`PROVIDER_INTERRUPTION` as a supported
transient reason. The provider did not report that reason. Exact cancellation,
terminal provider status, durable record writes and subsequent plain resume are
real. Native polling normally invokes supervision while pending/retrying. This
exercise validates a selected recovery path and its two persistence boundaries;
it does not prove a naturally occurring provider outage or optimizer resume.

The initial experiment without reason injection correctly adopted the healthy
RUNNING job and did not reach exit 91. It was preserved as a failed harness
experiment and cancelled; it is not counted as a successful boundary test.

`gpu-fault-tests/test_fault_driver.py` contains 110 deterministic author controls
for the harness, including malformed/binding-mismatched receipts, namespace and
identity checks, failed cancellation, healthy-job adoption and explicit reason
injection. They use synthetic records and are not physical GPU evidence.

Preserve all observations before cancelling or destroying only the resources
owned by the experiment. Do not publish the private configuration, receipts,
operational logs, storage URIs or raw checkpoints that contain live run bindings.
