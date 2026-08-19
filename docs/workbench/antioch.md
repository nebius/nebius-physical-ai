# Antioch Workbench integration

The Antioch adapter is a CPU-only control-plane service. It stages immutable
projects from S3, submits scenarios or suites with Antioch's supported structured
CLI, reconciles retries, collects checks/logs/results/Rerun files, and optionally
publishes a strict LeRobotDataset v3 for offline policy training. Antioch executes
simulation on its managed infrastructure; this image contains no simulator.

## Authentication and runtime boundary

Install Antioch's CLI normally and authenticate once as the operator. Confirm the
existing session without printing identity data:

```bash
npa workbench antioch health --output json
```

Do not copy the Antioch config into an image. For Kubernetes, create a secret from
the existing config out of band and mount it read-only with `deploy
--antioch-config-secret`; create a separate secret whose `token` key protects the
adapter HTTP API. The deploy command prints secret *names*, never values.
Provide S3 credentials through a pre-created `--s3-credentials-secret`, or omit
that option when the pod workload identity supplies S3 access. The secret uses
the ordinary `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
optional `AWS_SESSION_TOKEN` keys.

The public adapter pins `antioch-sim==0.3.47` and its reviewed SHA-256. On first
use it fetches the wheel directly from the vendor's PyPI delivery into
`NPA_ANTIOCH_RUNTIME_CACHE`, verifies it, and installs it in that writable volume.
`NPA_ANTIOCH_RUNTIME_OFFLINE=1` fails closed when the cache is cold. Neither the
wheel nor runtime cache belongs in the adapter image. The operator's direct
delivery and use remain subject to the operator's Antioch/NVIDIA terms.

Today the CLI session is personal OAuth stored in Antioch's config directory.
That is suitable for a human-operated smoke, but not a production unattended
identity. Production deployment should use an Antioch service identity when the
vendor exposes one; until then, token expiry requires an operator to refresh the
mounted session. The adapter never initiates interactive login.

## Immutable input and S3 output

`--input-path` names a prefix containing:

```text
project-manifest.json
project.tar.gz
```

The manifest uses schema `npa.antioch.project.v1` and records archive name, size,
SHA-256, source name/revision/license/digest, and asset hashes. Extraction rejects links,
traversal, device nodes, credentials, key files, and projects without exactly one
`antioch.yaml`. The adapter rewrites only its project id to a deterministic value
derived from workflow run and state identities.

`--output-path` must be a run-scoped S3 prefix. Durable state is under `_control/`.
Collected bytes are under `artifacts/<scenario-run>/`, normalized training data
under `dataset/`, and the versioned manifest at `manifests/v1.json`. `_SUCCESS.json`
is created immutably only after every preceding write and checksum check succeeds.
Consumers must gate on that marker. Never reuse one output prefix across unrelated
workflow runs.

## Operations and recovery

```bash
npa workbench antioch submit --input-path s3://BUCKET/input \
  --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --suite SUITE --output json
npa workbench antioch status --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --output json
npa workbench antioch resume --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --output json
npa workbench antioch cancel --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --output json
```

`run` is submit, monitor, and collect. A conditional S3 claim and deterministic
Antioch project id ensure a pod retry reconnects rather than creating another
billable suite. `reconcile` repairs the submission-to-state crash window. `resume`
does not rerun terminal work unless `--rerun-terminal` is explicit. HTTP 429 and
5xx failures are retryable; authentication failures, malformed JSON, conflicting
identity, invalid artifacts, and schema failures are terminal. Cancel is
idempotent. Cancel test work before releasing any machine it used.

The sanitized operation record contains the vendor run id. Open that run in the
Antioch Mission Control console using the authenticated account; never paste a
signed console URL into logs, manifests, issues, or pull requests. If a supported
CLI response supplies a non-signed console URL, the adapter may expose its
redacted form. It does not construct undocumented Rome URLs.

## Policy data contract

Arbitrary logs or telemetry are not training data. Every collected `.npz` episode
must carry arrays `observation_state`, `observation_image_workspace`,
`observation_image_wrist`, `action`, `reward`, `terminated`, `truncated`,
`timestamp`, plus JSON `provenance`. Lengths must agree; timestamps must increase;
observations/actions must be finite; and exactly one of terminated/truncated must
be true on the final frame only. The action width must match `action_schema`.
The pinned LeRobot ACT path currently requires at least two physically meaningful
action channels; collection fails closed rather than padding or duplicating a
single-channel control.

The `npa.antioch.episode.v1` provenance includes scenario, case, seed, parameters,
engine and SDK versions, source SHA-256, asset hashes, observation/action schemas,
and FPS. Incompatible episodes fail collection, leaving no completion marker.
Validated episodes are converted by the real NPA LeRobot v3 adapter, with
`meta/antioch-provenance.json` retaining provenance. This supports static offline
imitation training. It does **not** turn the export into an online PPO/RSL-RL
environment.

The executable example
`npa/workflows/workbench/npa-workflows/antioch-offline-policy-train.yaml` follows
collection with real LeRobot ACT training and publishes a genuine checkpoint.

For online Franka camera control with OpenPI pi0.5-DROID, use the separate
[RTX Isaac-to-B200 OpenPI bridge](antioch-openpi-franka.md). It shares native
Isaac code with the Antioch scenario wrapper but does not confuse a
credential-free Kubernetes proof with an Antioch-hosted run.

## Security and cleanup

The adapter filters sensitive keys and bearer/JWT/signed-URL forms from CLI errors
and log objects. It never emits environment dumps, identity fields, config files,
tokens, or customer metadata. Use only synthetic/public projects for validation.
After a smoke, cancel only the run ids created for that smoke, then release only
the associated project machine if one was allocated; queued managed execution
normally requires no persistent operator machine.
