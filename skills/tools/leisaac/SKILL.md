---
name: leisaac
description: Use when building, launching, operating, reviewing, or live-testing the NPA LeIsaac browser teleoperation tool, its secure agent-relay transport, immutable LeRobot datasets, custom bundles, or PAIDF export/materialization.
---

# LeIsaac browser teleoperation

Use this skill for the real LightwheelAI/LeIsaac workbench service and its
authenticated NPA agent UI. Read `docs/workbench/leisaac-teleoperation.md` and
`docs/workbench/guides/leisaac-transport-latency.md` before changing runtime,
transport, recorder, or UI behavior.

Also load these skills when their surfaces are involved:

- `third-party-eula-preflight`, `gpu-selection`, and `nebius-infra` before a
  live launch;
- `npa-agent` and `agent-fresh-operate` for agent deployment or live UI tests;
- `workbench-tool`, `testing-conventions`, `real-components`, and
  `solution-licensing` for implementation, validation, or packaging changes;
- `physical-ai-data-factory` for PAIDF export or materialization.

## Ground truth and capability boundary

- Source is LeIsaac `0.4.0` at
  `1651c321e9b0c1bb54233211fc7b3cd70d8373d5` with exact, build-gated NPA
  packaging and observability patches.
- The supported real tasks are `LeIsaac-SO101-PickOrange-v0` and
  `LeIsaac-SO101-LiftCube-v0`. Browser operation uses upstream
  `SO101Keyboard`; one session controls exactly one environment.
- Rendering requires RT cores. The current launcher supports only the named
  RTX PRO 6000 Kubernetes pool. Do not route this workload to H100/H200, and
  do not advertise L40S until this exact image and launch path are validated.
- Isaac Sim/Lab, NVIDIA's browser client, SO-101, and scene assets are
  runtime-fetched only after the shared Isaac preflight. The single public
  input defaults `ACCEPT_EULA=Y`; `Y`, `YES`, `1`, and `TRUE` normalize to `Y`,
  while empty, `N`, `NO`, `0`, and `FALSE` explicitly opt out before download
  and other values are invalid. The runtime derives
  `OMNI_KIT_ACCEPT_EULA=YES` internally. Do not add duplicate public consent
  variables or prompts. Keep `PRIVACY_CONSENT` and telemetry off by default.
- This is a lifecycle-bearing interactive service, not a finite
  `npa.workflow` step. Launch and destroy it with the workbench CLI.

## Security and transport decision

Use `agent-relay`, the CLI default, for production. It keeps Kubernetes on a
private ClusterIP and carries status, controls, frames, and signaling through
the agent's authenticated HTTPS origin and an authenticated WSS backhaul. TURN
uses a digest-pinned coturn sidecar, session-scoped credentials, bounded relay
ports, and operator CIDR restrictions. The browser never receives the raw
session nonce or agent credentials.

`public-load-balancer` is an unsupported historical transport. Its S3 manifest
cannot securely deliver the browser credential, because publication strips the
session nonce and there is no provenance-bound reinjection path. Launch rejects
it before EULA or infrastructure mutation. Retain only status/destroy handling
needed to diagnose and clean up already-existing historical resources; never
describe that mode as usable, TLS, or secure.

All control transports terminate in the same ordered runtime ledger and
controller lease. WebSocket, data-channel, and HTTP polling fallbacks must
enforce identical ownership: a second authenticated browser cannot drive keys,
direct actions, modes, or orbit while another browser owns the lease. Preserve
exact-key bounded messages, idempotent sequence acknowledgements, disconnect
release, and bounded replay semantics.

All same-run mutators (launch, destroy, and the lifecycle proof) use a renewed
Kubernetes Secret as an exclusion lock. Preflight requires `get`, `create`,
`update`, and `delete` on Secrets in the selected namespace before mutation.
Custom cleanup-only roles must include those verbs; never bypass the lock to
make teardown proceed concurrently with launch or credential rotation.

## Operate

Inspect supported tasks before launch:

```bash
npa/.venv/bin/npa workbench leisaac list-tasks --output json
```

Launch a digest-pinned image. Isaac acceptance defaults on for this route;
`ACCEPT_EULA=N` is the explicit pre-download opt-out:

```bash
npa/.venv/bin/npa workbench leisaac launch \
  --run-id <run-id> \
  --image <registry>/npa-leisaac@sha256:<digest> \
  --context <rtxpro-kubernetes-context> \
  --namespace <namespace> \
  --source-range <operator-public-ip>/32 \
  --agent-project <project-alias> \
  --agent-name <agent-name> \
  --task LeIsaac-SO101-LiftCube-v0 \
  --environment-id <stable-environment-id> \
  --environment-index 0 --seed 42 --num-envs 1 \
  --output-path s3://<bucket>/datasets/<dataset> \
  --manifest-prefix s3://<bucket>/checkpoints
```

The omitted `--transport` deliberately defaults to `agent-relay`. The image
must be `repository@sha256:digest`; source ranges must be explicit restricted
CIDRs. `--output-path` is the dataset prefix. `--manifest-prefix` is the
capability prefix or exact `.../reports/leisaac-session.json` leaf.
LeIsaac's required node affinity accepts either GPU Feature Discovery's exact
RTX PRO 6000 product label or Nebius's managed-node `RTX6000` provider label;
this preserves RT-core routing when managed drivers intentionally disable GPU
Operator operands.

Inspect and clean up only the selected run:

```bash
npa/.venv/bin/npa workbench leisaac status \
  --run-id <run-id> --context <context> --namespace <namespace>
npa/.venv/bin/npa workbench leisaac destroy \
  --run-id <run-id> --context <context> --namespace <namespace>
```

Destroy removes transient Kubernetes, relay, TURN, recorder-secret, and exact
NPA-managed ingress resources. It preserves immutable S3 manifests, episodes,
versions, and evidence. If launch fails, preserve the original error while
reporting every cleanup failure and partial ingress progress.

## Artifacts and real-component guarantees

- Capability: `<manifest-prefix>/<run-id>/reports/leisaac-session.json`, or the
  exact leaf supplied by the operator. It is write-once evidence, while agent
  resolution uses bounded freshness so a same-ID relaunch cannot retain a stale
  nonce or endpoint.
- Dataset: immutable episode commits, version trees, Parquet records, raw JPEG
  evidence, and H.264 faststart MP4s under `--output-path`. `latest.json` is
  only a bounded-retry monotonic pointer; it is not the dataset record.
- Custom robot/scene/device bundles are canonical, hash-addressed, bounded, and
  runtime revalidated. Uploaded Python is parsed but never executed.
- PAIDF export must invoke the real Cosmos Transfer tool. Materialization must
  fail closed unless a real input-conditioned result preserves frame/timestamp
  alignment; never turn a manifest stub into a claimed augmentation.

## Packaging and licensing

`npa-leisaac` is a public runtime-fetch image only when the built digest proves
that Isaac/Omniverse payloads, the NVIDIA client, task assets, credentials, and
caches are absent. Follow `npa/docker/workbench/leisaac/REDISTRIBUTION.md`,
`THIRD_PARTY_NOTICES.md`, and `npa/docker/workbench/packaging-contract.yaml`.
Validate the built image itself with
`npa/scripts/scan_image_omniverse_payload.py`; Dockerfile inspection is not
evidence.

The host-side `imageio-ffmpeg==0.6.0` dependency belongs only to the `leisaac`
extra, never `full` or core. Its wheel-bundled FFmpeg is excluded from the
container; the image uses Debian FFmpeg. Keep coturn, pygame, aiortc, FFmpeg,
and every other installed dependency represented accurately in notices.

## Live agent UI verification

Deploy or bootstrap the agent from the branch under test using `npa-agent` and
`agent-fresh-operate`. Use only the HTTPS customer URL and owner-only auth file;
never print credentials. At minimum verify authenticated `/api/health`, the
LeIsaac tab/panel and unavailable state, status no-store behavior, secure
transport labeling, and backend authorization. With an explicitly accepted,
live LeIsaac run, also exercise connect/reconnect, controller contention,
motion and orbit, recorder transitions, finalized episode playback/ranges, and
the live Cypress suite:

```bash
cd npa/tests/browser
npm run cy:live-leisaac
```

Set the required `NPA_AGENT_BASE_URL`, `NPA_AGENT_USER`,
`NPA_AGENT_PASSWORD`, `NPA_LEISAAC_RUN_ID`, and `NPA_AGENT_TASK` environment
values without committing or logging secrets. Keep the session available when
the operator requested continuing access; otherwise use the scoped `destroy`
command and record cleanup status.

When replacing only the agent for an explicitly selected existing run, retain
the exact `.../reports/leisaac-session.json` URI before agent teardown and use
`npa workbench leisaac reconnect-agent` after the fresh agent is healthy. This
command must preserve the existing Deployment, image, task, dataset, and
run-scoped EULA values; it may rotate only the relay Secret/nonce, exact agent
ingress, and that exact manifest. Never substitute `launch`, create another
workload, or use broad bucket discovery for this recovery.

When an operator explicitly requests relay restart and expiry proof against an
existing run, use
`npa/scripts/verify_leisaac_relay_lifecycle_live.py`. It is a mutating,
same-Deployment check: it holds and safely releases a control across restart,
verifies browser-facing control/video recovery, fires bounded credential
expiry, resumes the same controller lease to prove its held keys were released
before any pod restart, writes a new immutable capability generation for the
rotated session credential, rejects the stale credential, and restores
recorder-idle service.
It temporarily scales that same Deployment to zero so current-nonce acceptance
and stale-nonce denial are isolated from the single-active-backhaul rule. The
original capability and immutable dataset objects remain unchanged. Never use
it to infer EULA consent, launch a replacement workload, or test an unrelated
run. Store its secret-free evidence outside Git with owner-only permissions.

## Validation

Use the repository venv, never bare Python:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_leisaac.py \
  npa/tests/workbench/test_leisaac_cli.py \
  npa/tests/workbench/test_leisaac_dataset.py \
  npa/tests/workbench/test_leisaac_agent_relay.py \
  npa/tests/workbench/test_leisaac_paidf.py \
  npa/tests/cli/test_agent_leisaac.py \
  npa/tests/cli/test_agent_leisaac_bundles.py \
  npa/tests/cli/test_agent_leisaac_episodes.py \
  npa/tests/cli/test_leisaac_datachannel.py \
  npa/tests/cli/test_leisaac_transport.py -q
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_hygiene_guards.py \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_packaging_contract.py -q
cd npa/tests/browser && npm run cy:mock
```

Also run Ruff, mypy, docs generation, the broad non-E2E suite, and image build
plus full-filesystem payload scan when image contents change. Do not weaken
content scans, upstream patch gates, exact input validation, escaped UI sinks,
immutable publication, real-component checks, or the terminator guard.
