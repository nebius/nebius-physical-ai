# Contributing a Containerized Solution

Use this checklist to add an open-source solution to NPA. The contributor
supplies reproducible code and real test evidence. The Nebius Physical AI team
builds, scans, pushes, and, when approved, publishes the official image.

Contributors do **not** need official registry credentials.

## Agent contract

Collect before editing:

- upstream repository and immutable commit/tag;
- solution name and upstream capability to prove;
- required CPU/GPU architecture and count;
- real input, command, and expected output artifact; and
- authoritative licenses for source, baked runtime, weights, and data.

Stop and report instead of proceeding when:

- the public-image build requires a credential;
- any baked component's redistribution rights are unclear;
- gated weights, data, SDKs, or proprietary assets would enter an image layer;
- a released public tag would be reused; or
- the claimed capability cannot be tested on its required hardware.

Never weaken a license guard, expose registry credentials to fork CI, or commit
tenant, project, registry, bucket, key, IP, or secret values.

## 1. Choose the smallest integration

| Path | Add |
| --- | --- |
| BYOF candidate | Pinned install, real `solution-smoke`, and `byof-<solution>.yaml` |
| Solution workflow | Candidate/curated image plus an `npa.workflow` spec |
| First-class Workbench tool | Dockerfile, image pin/contract, CLI/SDK/workflow wiring, tests, docs, and skill |

Use `docs/architecture/oss-onboarding-ladder.md` for promotion criteria.

## 2. Implement a reproducible container

For a BYOF candidate:

1. Add `npa/workflows/workbench/npa-workflows/byof-<solution>.yaml`.
2. Use `workload: solution-smoke` with the real upstream capability command.
3. Pin the source/install and write a named JSON result under
   `$NPA_SMOKE_OUTPUT_DIR` containing `solution`, `capability`, and a real proof
   value.
4. Verify request construction without pushing or launching infrastructure:

```bash
npa/.venv/bin/npa workbench byof run \
  --repo-url <public-repository-url> \
  --repo-ref <immutable-ref> \
  --base-profile ubuntu \
  --workload solution-smoke \
  --build-command '<pinned-install-command>' \
  --smoke-command '<real-capability-command>' \
  --solution-name <solution> \
  --capability-name <capability> \
  --smoke-artifact-name <solution>_<capability>.json \
  --skip-push --skip-run --dry-run --output json
```

For a first-class image, add or update:

- `npa/docker/workbench/<tool>/Dockerfile` and preferably `build.sh`;
- `npa/docker/workbench/packaging-contract.yaml`;
- `npa/src/npa/deploy/images.py` and its version source;
- real CLI/SDK/workflow resolution and focused tests; and
- relevant documentation and root skill/index entry.

Use a new additive tag, a digest-pinned base when resolvable, a non-root final
user, and no baked credentials, gated content, or generated artifacts.

## 3. Classify redistribution

Classify these independently:

1. upstream source;
2. everything baked into the runtime image; and
3. weights, datasets, and other assets.

Set `redistribution: public` only when every shipped byte may be handed to third
parties. Otherwise use `restricted` or stop for human review. Keep gated content
as an operator-authorized runtime fetch when legally valid. Record the decision
in `npa/docker/workbench/packaging-contract.yaml`.

## 4. Prove the real capability

The PR must contain live evidence matching every claim:

- CPU claim: run the functional workload on a real CPU.
- GPU claim: run it on a compatible real GPU and record model, architecture,
  count, driver/runtime, command, exit status, and output artifact.
- CPU and GPU claim: run both.
- Multi-GPU claim: prove the requested device count and created mesh/workers.

Imports, `--help`, container start, and `torch.cuda.is_available()` are not
capability tests. Use the smallest real inference, training step, simulation,
render, conversion, query, or service request. Mark untested capabilities
`deferred`; do not call them accepted.

Include this compact evidence block in the PR:

```text
Upstream: <repo>@<immutable-ref>
Capability: <upstream capability name>
Image: <public development tag/digest, if any>
Hardware: <CPU or GPU model/architecture/count>
Command: <exact command, secrets removed>
Result: <exit status and key measured output>
Artifact: <path/type/size/hash or inspected metrics>
Limits: <deferred paths and residual risks>
```

## 5. Validate and open the PR

```bash
python3 -m venv npa/.venv
npa/.venv/bin/pip install -e "npa[dev]"
npa/.venv/bin/python -m pytest \
  npa/tests/docker/ npa/tests/deploy/ \
  npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_byof_solution_smokes.py -q
```

Also run focused solution tests and validate/plan each new workflow. Open the
PR against `main` with the evidence block, base image/digest, approximate size,
target architecture, packaging tier, redistribution class, license sources,
build-time credential statement, and known risks.

## Maintainer handoff

After review, the Nebius Physical AI team will:

1. run all pre-publication gates, rebuild the exact trusted commit, and push
   `dev-<full-git-sha>` to the public image package in official GHCR;
2. scan the pushed bytes for secrets, vulnerabilities, restricted payloads,
   weights, caches, and non-root/runtime-contract violations;
3. pull that digest through the real NPA path and repeat the required CPU/GPU
   capability tests; and
4. after explicit authorization, promote only the validated public-development
   digest to the supported release tag and verify anonymous digest identity.

Fork CI never publishes official images. Public development bytes are
irreversibly disclosed even if a failed tag is later deleted.

Related policy:

- `skills/workflows/contribute-workbench-image/SKILL.md`
- `skills/atomic/solution-licensing/SKILL.md`
- `docs/workbench/container-packaging.md`
