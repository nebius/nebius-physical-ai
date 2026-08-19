# Release and recovery

## Classify built bytes independently

Apply `skills/atomic/solution-licensing/SKILL.md`,
`skills/workflows/contribute-workbench-image/SKILL.md`, and
`skills/atomic/protect-nebius-infra-details/SKILL.md`.

- Public Antioch-owned layers may contain approved repository integration source
  and redistributable dependencies. Exclude auth/config, cookies, tokens, SSH
  host/user identities, `.antioch/` locks/state, detached logs, customer data,
  private assets, model weights/caches, and live infrastructure values.
- A public OpenPI server contains bootstrap/source/dependencies only. Fetch
  weights and tokenizer at runtime into the verified cache.
- An Isaac/Omniverse image is public only when registry-byte scans prove the
  clean runtime-fetch design contains no Isaac, proprietary Isaac Lab wheel,
  Omniverse Kit, driver userspace, or other restricted NVIDIA bytes. Antioch
  approval cannot relicense NVIDIA content. Keep restricted or ambiguous images
  private.

## Validate and publish

1. Prove the selected build environment is explicitly mutable. Never infer
   build authority from access to an operator/developer VM. When a host is
   read-only, use the trusted registry's or authorized Kubernetes-native build
   path. Build with an immutable additive tag and exact source revision, then
   independently verify the manifest digest, config user, labels, history, and
   pullability.
2. Scan the digest—not the Dockerfile—with the repository Antioch, OpenPI, and
   Omniverse payload scanners. Produce an SBOM and vulnerability/secret report.
3. Run the real capability smoke on the intended GPUs: B200 `sm_100` checkpoint
   load and finite `[15,8]` inference; RTX rendering/cameras, private transport,
   and safe target application.
4. Run the publisher preflight/dry run. Publish only byte-proven eligible images
   to the established GHCR namespace. Verify digest equality, anonymous pull,
   and public package visibility. Never put a private source registry reference
   or cloud identifier in public evidence.

## Recover interrupted operations

- **Crane copy/push:** inspect the process and destination manifest by the exact
  intended tag. If no process remains, compare source/destination digests. Resume
  the exact additive tag only when absent or digest-equal; never overwrite a
  conflicting tag or publish based on local cache alone.
- **Payload scan:** an interrupted scan has no valid verdict. Discard only its
  exact incomplete temporary report/tar and rerun against the registry digest.
- **Kubernetes:** query exact named Deployment/Job/Service/NetworkPolicy objects
  and pod/container states. Preserve a healthy needed workload; delete only the
  exact failed run objects before replacement. Never delete a namespace or
  shared PVC until ownership and requested retention are proven.
- **Port-forward/connector:** process existence is not readiness. Verify the
  exact process belongs to this run, policy health, Antioch service/API state,
  and an application request. Terminate only exact stale PIDs; restart with
  bounded backoff.
- **Antioch run:** query exact scenario/suite state or reconcile by invocation,
  project, authored name, caller, and narrow time window. Never resubmit while a
  unique nonterminal candidate exists. Cancel the exact run, stop exact project
  services, then release the exact task-owned machine.

Store raw registry, cluster, and Antioch evidence only in access-controlled
external files. Public summaries may contain intended public image names/tags
and digests, generic GPU classes, check outcomes, shapes, and sanitized failure
classes—never private registry paths, tenant/project/cluster IDs, endpoints,
machine IDs, usernames, or secret values.
