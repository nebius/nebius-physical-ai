Provider and absence recovery proof, October 1

The reports distinguish actual native Terraform provider deadlines from process
cleanup deadlines. These controls make no GPU or VLM workload claim.

Final owned-infrastructure cleanup is verified in
`provider-final-cleanup.public.json`: supported destroy and the committed cleanup
test passed, and ten fresh same-authority reads confirmed the owned cluster, node
group, two application releases, node instance and managed boot disk are absent.
Earlier receipts retain their original pending-cleanup status as history.

Final-source controls use commit `400606df0a904a93af297ef6dd3c0246155c4635`:
`final-source-binding.public.json` verifies byte-identical production and native
guardrail files after the subsequent documentation-only commit.

- `provider-native-deadline-final.public.json`: the current provider module
  configured a successful authenticated project data-source read. Explicit
  operator deadlines remained unchanged and produced a native typed
  `DeadlineExceeded` failure. These read-only controls declared zero resources.
- `provider-timeout-final.public.json`: the actual Nebius CLI process timed out,
  was joined, left isolated synthetic journal and lease bytes intact, and released
  both recovery locks. A successful authority read preceded the timeout control.
- `absence-process-final.public.json`: 23 actual process controls passed on Linux
  and macOS; 191 affected tests passed. The report identifies producer commits
  and the final formatting-only change. Removing the ownership guard makes both
  real external-reaper regressions fail. These checks do not synchronize with an
  arbitrary competing reaper that acts after the immediate ownership check.
- `native-sky-contract-final.public.json`: all three committed guardrail tests
  passed using the actual pinned SkyPilot 0.12.2 wheel, including all nine native
  file hashes and the shared required-version constant.

Earlier evidence is retained with its original source attribution:

- `provider-native-deadline.public.json`: exact recipe-installed native provider,
  authenticated project data-source read and typed request-deadline failure.
  No resources are declared or changed. Two earlier harness corrections occurred
  before provider RPCs and remain recorded.
- `provider-timeout-live.public.json`: actual Nebius CLI process was started and
  forcibly timed out. It was joined; production recovery locks were reacquired;
  isolated synthetic local journal and lease bytes remained identical. This is
  process cleanup proof, not a slow backend RPC claim.
- `native-sky-contract.public.json`: all nine actual SkyPilot wheel source files
  match the reviewed naming hashes; exact validator module hashes are retained.
- `absence-mutations.public.json`: an actual local read deadline control detects
  removing the timeout, and upgraded global Sky pins reject old naming evidence.

The fresh supported CPU provision and committed live readback test passed.
`provider-lifecycle-live.public.json` binds the source, materialized deadlines,
actual Ready CPU node and private retained input hashes. That lifecycle was
produced by the report's earlier commit, not the final-source controls above.
The same node subsequently supported the separate private image pull test and
was destroyed after that test's workload cleanup completed.
Combined-tree tests, hosted CI and final head bindings are completed by the root
reviewer separately; these reports do not claim they have passed.

The owned node's private-image pull investigation is separate from NPA's read
deadlines. `node-watchdog-diagnostic.public.json` binds the actual node logs to
both failed large-image attempts: containerd canceled each pull after its
five-minute progress watchdog expired. This establishes the cancellation source;
it does not prove the precise upstream progress-accounting race described in
[containerd issue 13909](https://github.com/containerd/containerd/issues/13909).

`node-watchdog-adjustment.public.json` records an explicit adaptation on that
single ephemeral node. A task-owned drop-in changes only the behavior setting
`image_pull_progress_timeout` from `5m0s` to `0s`, which the native parser accepted.
Independent reads verified the effective configuration, active runtime,
CRI readiness and Kubernetes readiness after restart. Temporary debug resources
were deleted with UID preconditions and verified absent. The original main
configuration remained byte-identical. The resulting transfer-mode attempt
stalled. `node-watchdog-zero-stall.public.json` records its idle metrics, source
analysis and verified namespace cleanup. In the reviewed containerd 2.3.3
[transfer path](https://github.com/containerd/containerd/blob/v2.3.3/internal/cri/server/images/image_pull.go#L919-L942),
zero disables the goroutine that also consumes an unbuffered progress channel;
the callback can then block. No native goroutine dump was collected.

`node-local-pull-adjustment.public.json` records the second scoped adaptation:
retain `0s` and enable `use_local_image_pull`, using the actual daemon's
version-compatible configuration. The local pull implementation avoids that
transfer progress channel. Independent readback and runtime health passed;
temporary debug resources were again deleted and verified absent. A refused
legacy-key attempt was rolled back before restart and remains in private evidence.

Any later successful pull uses the adapted local runtime and may reuse partial
cached layers from the failed attempts;
it is not proof that the stock runtime default succeeds or that every layer was
downloaded again. Native logs now confirm the successful local-mode pull in
`node-local-pull-native-success.public.json`; the root report supplies the
committed probe result. The node and managed boot disk are verified absent, so
the temporary runtime override has no surviving node to restore.
