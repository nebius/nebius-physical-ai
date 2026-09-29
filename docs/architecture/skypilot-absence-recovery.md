# Recover a failed submit after native resources are absent

A failed managed submit can retain a project lease after its controller has
already been removed. `workflow reconcile-controller` needs a live, exclusively
bound controller. `workflow reconcile-absent` covers a different case: original
producing records identify the removed resources, and fresh Kubernetes reads
prove that no matching resources remain. It does not declare a workload
successful or invent an original Pod UID.

This implementation depends on the journal locking and compare-and-swap helpers
introduced by PR #709. It does **not** accept that PR's MK8s provisioning evidence
as proof of a Sky submission. The CLI base and workload source remain separate.

```bash
npa workbench workflow reconcile-absent --evidence-file /private/evidence.json
npa workbench workflow reconcile-absent --evidence-file /private/evidence.json --apply
```

The first command performs fresh reads and retains a private audit without
releasing the lease. `--apply` repeats the verification under the original project
and operation execution locks, preserves original journal/lease bytes, then
commits the bound transition. A crash between journal finalization and lease
release can be resumed with the identical manifest. Neither form deletes cloud
resources, changes credentials, or submits a replacement workflow. A new run ID
or isolated controller directory is not a workaround for an unresolved lease.
Crash recovery rechecks the retained raw response hashes before releasing the
bound lease; a summary without those response bytes is insufficient.

## Required original evidence

The private manifest uses schema `npa.sky.absence.v1`. Each file reference is an
absolute `path` plus its exact `sha256`. Keep all infrastructure values and raw
responses outside the repository.

- `operation_id` and `original_journal`: a preserved original failed
  `npa workbench workflow submit` generation.
- `producer_transcript` and `producer_line`: the original completed tool-call
  record containing the supported submit command, its explicit isolated root,
  project, Kubernetes context, run identity and output path. The bounded reader
  accepts the retained `shellToolCall` JSONL transport; it never executes the old
  command. A newly written owner summary is insufficient.
- `producer_response` and `submission_ledger`: original agreeing structured NPA
  launch records, including the ambiguity's complete native ID set and task
  profiles. Only their human-readable `operator_remedy` prose may differ.
- `isolated_root`, `native_wrappers`, `native_configs`, `native_state`: original
  Sky files under that root, including each implicated managed ID and the
  controller's original native configuration. If a nonempty WAL exists, also pin
  it as `native_wal`. SQLite reads private copies of both files, preserving the
  original database, WAL and shared-memory state. An earlier implicated ID is
  not relabeled as newly created.
- `sky_version` and `naming_source`: exact reviewed SkyPilot 0.12.2 source file
  hashes. Ordinary non-pool Kubernetes task/controller names preserve the full
  isolated user suffix through truncation. This allows complete inventory
  coverage even if temporary native DAG files are gone. Unsupported clouds,
  pools, metadata overrides or configuration shapes refuse recovery.
- `cluster_registration`, `original_kubeconfig`, `reader_kubeconfig`: original
  target and an existing supported explicit key-profile reader for the same
  context, namespace, HTTPS endpoint and CA. The original auth helper is not run.
  An empty legacy registration endpoint is allowed only because the fresh
  provider cluster response must match the original kubeconfig endpoint and CA;
  a contradictory nonempty registration endpoint is rejected.
- `authority`: the existing `profile`, pinned `config`, `key` and provider
  `binary`. This selector requires the pinned config's basename to be
  `config.yaml`: the supported token exec uses that default filename under
  `NEBIUS_CONFIG_DIR`, while the provider GET passes the same file explicitly.
  Alternate filenames are rejected so the two reads cannot select different
  authority. The profile's default project is not compute ownership: the fresh
  exact cluster read must independently match the original project, endpoint and
  CA. No IAM or profile change is made.

## Issued attempt with no accepted managed ID

The separate `npa.sky.zero-id-absence.v1` schema covers one original issued
attempt whose terminal response and ledger agree on `terminal_failure`, empty
managed ID, `existence=absent`, and `verified_absent_no_retry`. It refuses a
successful, accepted, unknown, repeated, or contradictory launch. The original
journal must remain failed and recovery-required with no recorded resources.

Pin the original `producer_freeze` (argv, source/tree and runner hash),
`producer_process` (failed exit, completion time and output hashes),
`producer_response`, `producer_stderr`, `producer_runner`, `producer_workflow`,
`producer_environment`, and `producer_modules`. The reviewed source module pins
explain the original reconciliation behavior; the original Git object must
match the recorded tree and module bytes. No retained runner is executed.
The original argv and environment bind the root, context, operation and ledger.
`native_submission_config` must be the original restricted Kubernetes config.
Its local API endpoint must match the pinned same-root `original_api_daemon`
port on `127.0.0.1`. Global Pod configuration can contain only named pull
Secrets; metadata, name, and other Pod overrides are rejected.
Both `empty_native_databases` (`state.db` and `spot_jobs.db`) must still be empty
of controller and managed-job identities. Pin any nonempty WAL in
`empty_native_wals` under its database filename; reads use private copies and
reject changed or unbound WAL bytes.

The two retained absence decisions have different origins. The first comes
from the original healthy `controller_absent` probe. Only the second invokes
the exact-name queue reconciliation. Historical queue stdout was not retained
by this producer; the report explicitly records that limit. Neither record is
relabeled as two raw queue calls or as proof of current cloud absence. Fresh
provider identity and complete metadata reads, using the same pinned naming
and reader authority contracts above, are still mandatory before the unchanged
original-generation compare-and-swap transition.

## Meaning of success

The collector retains raw exact controller Pod/Service responses and every page
of the full Pod/Service metadata inventories. Any matching, recreated,
terminating, ambiguously foreign or incompletely described resource refuses
recovery. So do permission errors, transport errors, incomplete/repeated pages,
changed evidence, live owner processes and changed journal/lease generations.
Unrelated resource identities are preserved; shared Kubernetes bootstrap
resources are not silently claimed as owned.

`reconciled-absent` means the local failed operation is finalized as absent and
its lease is released. `historical_workload_outcome` remains `unknown`; original
errors and events remain in the journal. Reads are separately timed observations,
not an atomic cloud snapshot. Local locks serialize cooperating NPA writers,
not external Kubernetes actors. A replacement workflow still needs its own
health, target, image pull, storage, observed Pod and final cleanup evidence.

## Read-only live regression

The opt-in `npa/tests/e2e/test_workflow_absence_preview_live.py` calls only the
preview path. Set `NPA_SKY_ABSENCE_PREVIEW_LIVE_CONFIG` to a private JSON file with
`source_revision`, `evidence_file`, `evidence_sha256` and `operation_id`, and run
it with `NPA_INTEGRATION_E2E=1`. It requires the exact clean source, checks the
manifest hash, performs the fresh reads, and verifies that the journal remains
unchanged. It does not apply the lease transition. Without the explicit private
configuration it skips; a skipped test is not live absence evidence.
