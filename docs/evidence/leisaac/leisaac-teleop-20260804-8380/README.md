# LeIsaac live validation: `leisaac-teleop-20260804-8380`

Status: **incomplete — not successful teleoperation evidence**

This record preserves the final live result for Codex run
`20260804T015543Z-8380f904`. It must not be cited as proof that browser
teleoperation completed.

## Immutable inputs

- NPA image source commit: `0ff94dced1212c8d0ad98c33cb142be244f37cb0`
- Image digest: `sha256:306e8887c6834b79284d10dc9110f4aade19413a7be2954ac19743cec27ee35f`
- Upstream: LightwheelAI/LeIsaac `0.4.0` at
  `1651c321e9b0c1bb54233211fc7b3cd70d8373d5`
- Task: `LeIsaac-SO101-PickOrange-v0`
- Teleoperation device: upstream `SO101Keyboard`
- Isaac Sim: `5.1.0.0`
- Isaac Lab: `2.3.2.post1`
- GPU target and observed device: one NVIDIA RTX PRO 6000 Blackwell Server
  Edition
- Configured artifact target:
  `s3://<agent-artifact-bucket>/checkpoints/leisaac-teleop-20260804-8380/reports/leisaac-session.json`

The image build fetched the Apache-2.0 LeIsaac source at the pinned commit. It
did not bake NVIDIA Isaac runtime bytes, NVIDIA browser-client bytes, the SO101
asset, the kitchen asset, or EULA acceptance. The operator supplied both EULA
acceptance variables at launch, and those licensed artifacts were downloaded
into the pod's runtime caches.

## What live validation proved

- The approved build/push path produced the digest above. The mandatory image
  payload scan inspected 30 entries, allowlisted zero, and returned `VERDICT:
  clean`.
- The Kubernetes deployment requested and limited the main container to one
  NVIDIA GPU and selected an RTX PRO 6000 Blackwell node. `nvidia-smi` observed
  driver `580.95.05`, 97,887 MiB GPU memory, and the Isaac Python process using
  1,152 MiB.
- The real task reached `Completed setting up the environment` and printed the
  upstream PickOrange keyboard controls. The relay sidecar was ready.
- Agent bootstrap reused the existing VM and resolved its public endpoint from
  provider state. From the operator host, `https://<agent-public-ip>/healthz`
  returned 200 without credentials; `/` and `/api/health` returned 401 without
  credentials and 200 with basic authentication. TLS 1.3 negotiated. TCP ports
  8787, 8080, 49100, and 47998 were not publicly reachable.
- Before cleanup, the selected-run API returned `available: false` with reason
  `LeIsaac service health returned HTTP 503`, which kept the conditional tab
  unavailable as designed.

## Unresolved live blocker

The main LeIsaac container never became ready. Both `/status` and `/healthz`
continued to return HTTP 503. After roughly 16 minutes, the configured liveness
probe restarted the main container once while preserving its cache. The
restarted Isaac log again reported:

```text
[INFO][AppLauncher]: Using device: cpu
```

It enumerated the RTX PRO GPU for Vulkan rendering and completed environment
setup, but readiness remained false. The final pod state was relay sidecar
ready, main container unready, with one main-container restart. Because the
launcher publishes the run manifest only after live attestation, this final
attempt did not produce a new ready-session artifact at the configured target.

Consequently, end-to-end signaling, decoded browser video, keyboard-event
attestation through the public endpoint, and successful public-UI
teleoperation were **not completed**.

## Screenshots

No successful live screenshots were captured or committed. Mock-browser images
and partial simulator output are deliberately not represented as live evidence.

## Deterministic validation already completed

- Full Python suite: 5,744 passed, 292 skipped, 1 expected pass; 143 warnings.
- Focused LeIsaac Python suite: 34 passed.
- Focused Ruff check: passed.
- Browser UI suite on the unchanged frontend: 52 passed, including both
  conditional LeIsaac-tab tests. A later concurrent rerun passed the two
  LeIsaac tests but produced one unrelated Rerun MediaStream capture timing
  failure (51/52 overall); the failed diagnostic screenshot was not committed.
- Trivy completed after increasing its scanner timeout: zero secret findings,
  zero misconfigurations, and 23 critical findings inherited from the pinned
  Isaac Lab base (`linux-libc-dev` and an Nsight Compute Go binary). This is a
  recorded residual image risk, not a clean vulnerability result.

## Cleanup

`npa workbench leisaac destroy` removed the task deployment, services, relay
secret, relay/TURN units, and matching NPA-managed UDP ingress. The pre-existing
agent VM and durable object storage were preserved. After cleanup, the agent's
public `/healthz` still returned 200.
