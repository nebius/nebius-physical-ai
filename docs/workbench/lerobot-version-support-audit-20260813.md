# LeRobot version support audit and proposal — 2026-08-13

Upstream LeRobot is at **0.6.1** (released 2026-08-03). This workbench defaults
to **0.5.1** (2026-04-07) and offers **0.6.0** (2026-07-06) as a selectable
alternative. This audit answers whether the workbench should support 0.6.1, what
it would cost, and what is already wrong independent of that decision.

Every claim below was checked against the published wheels rather than the
release notes, and is reproducible with the pre-flight added alongside this
document:

```bash
npa/.venv/bin/python npa/scripts/audit_lerobot_release.py 0.6.1 --baseline 0.5.1
```

The conclusions were then **executed** against a real LeRobot 0.6.1 install on
the dev VM and against the shipped `npa-lerobot:0.6.0` image on an H100. That
step is what turned up D6, a defect no amount of static reading would have
found — see [Executed validation](#executed-validation).

## Recommendation

**Adopt 0.6.1 as an additional supported version, and make it the default once
the GPU gates pass. Do not treat this as a risky upgrade — treat the current
0.5.1 default as the risk.**

The upgrade itself is nearly free. All 19 LeRobot bindings this repo holds
resolve unchanged in 0.6.1; the sole breaking rename in 0.6.1
(`lerobot.types` → `lerobot.lerobot_types`) is not referenced anywhere in this
repo; the dataset format is unchanged; and the CLI contract is identical to
0.6.0's, which `version_compat.py` already models. The scaffolding built by
[#191](https://github.com/nebius/nebius-physical-ai/pull/191) means adding a
third version is mostly manifest data, not new code.

The more consequential findings are separate from the upgrade. **Two published
images install a torch/torchvision/diffusers stack that upstream 0.5.1 declares
incompatible**, and one of them — `npa-lerobot-policy:0.1.1` — is a current pin,
not a superseded tag. And running the existing `npa-lerobot:0.6.0` image on an
H100 showed **it cannot construct a Diffusion Policy at all** (D6): 0.6.0 moved
`diffusers` behind an extra we never requested. That one is fixed in this PR.

## Scope: what "supporting LeRobot" actually spans

"Support the latest LeRobot" is not one decision. LeRobot enters this repo
through five independent surfaces, each with its own failure mode and its own
upgrade cost:

| Surface | How it binds | Breaks when |
|---|---|---|
| Python API | 19 `from lerobot...` bindings across the policy server, Genesis student eval, workflow dataset staging, and the research profiler | a module is renamed or a symbol removed |
| CLI subprocess | `lerobot-train` / `lerobot-eval` argv built in four places | an entry point or draccus flag is renamed |
| Dataset format | three hand-rolled writers emit `meta/info.json` + `data/chunk-*` + `videos/` directly | `CODEBASE_VERSION` bumps or `info.json` gains required keys |
| Container images | four image families pin LeRobot at build time | Python or torch bounds move |
| Orchestration | three `workbench.lerobot.*` catalog toolRefs and the sim2real staged engine | the image they route to changes shape |

The Python API and CLI surfaces are the ones release notes talk about. The
dataset and container surfaces are the ones that actually cost effort here.

## Where LeRobot is pinned today

| Image | LeRobot | Python | Torch stack | Status |
|---|---|---|---|---|
| `npa-lerobot:cuda13-b300-0.5.1-…` (**default**) | 0.5.1 from git `1396b9f` | 3.12 | torch 2.9.0 / tv 0.24.0 / torchcodec 0.8.1 cu130 | clean |
| `npa-lerobot:0.5.1` (CUDA 12 semver) | 0.5.1 | 3.12 | torch 2.12.1 / tv 0.27.1 / diffusers ≥0.38 | **conflicting** |
| `npa-lerobot:0.6.0` (CUDA 12 semver) | 0.6.0 | 3.12 | resolver-chosen | clean |
| `npa-lerobot-policy:0.1.1` | 0.5.1 | 3.12 | torch 2.12.1 / tv 0.27.1 / diffusers ≥0.38 | **conflicting** |
| `npa-genesis:…0.4.6…` | 0.4.4 | 3.10 / 3.11 | torch 2.6.0 / tv 0.21.0 | clean, but frozen |
| `npa-lerobot-vlm-rl:cuda13-b300-0.1.1-…` | inherits Genesis 0.4.4 | 3.11 | inherits | clean, but frozen |

Version knowledge is duplicated across `npa/pyproject.toml`,
`lerobot_version_manifest.json`, `deploy/images.py`, `smoke/_versions.py`,
`smoke/golden_evals.yaml`, `terraform/variables.tf`, `cloud_init.yaml.tpl`,
`setup/install_lerobot.sh`, `lerobot/build.sh`, `lerobot/Dockerfile`, and four
skill/doc files. [#191](https://github.com/nebius/nebius-physical-ai/pull/191)
touched 26 files to add one version.

## What 0.6.1 changes

0.6.1 is a consolidation release. The single declared breaking change is the
`lerobot.types` → `lerobot.lerobot_types` module rename. The bulk of the diff is
internal refactoring — shared VLA components across pi0/pi0.5/eo1/pi0_fast/
smolvla, policy components resolved by convention, Transformers subclassing
instead of vendoring for wall-x and x-vla — plus HF Storage Bucket streaming
(`repo_type="bucket"`), which is why `huggingface-hub` moves to ≥1.6.0.

For context on the jump from the current default, 0.6.0 carried the larger
break: extras became mandatory for dataset/training deps, `eval_freq` became
`env_eval_freq`, `sac` became `gaussian_actor`, `--dataset.vcodec` became
`--dataset.rgb_encoder.vcodec`, and GR00T N1.5 became N1.7.

### Compatibility findings

Checked against the 0.5.1, 0.6.0, and 0.6.1 wheels:

- **Import surface: 19/19 bindings resolve in 0.6.1**, unchanged all the way
  back from 0.5.1. `lerobot.types` is not imported anywhere in this repo — the
  only reference is an archived work log — so 0.6.1's one breaking change is a
  no-op here.
- **Signatures are additive only.** `make_pre_post_processors` gained
  `pretrained_revision` and `EpisodeAwareSampler` gained `seed` /
  `absolute_to_relative_idx` (both in 0.6.0); `LeRobotDataset.__init__` gained
  keyword-only `token` in 0.6.1. Every parameter this repo passes by keyword
  still exists.
- **CLI contract is identical to 0.6.0.** `lerobot-train` and `lerobot-eval`
  entry points, `TrainPipelineConfig.env_eval_freq`,
  `PreTrainedConfig.pretrained_path`, and the draccus `PATH_KEY = "path"` alias
  behind `--policy.path` are all unchanged. The existing 0.6.0 manifest entry is
  therefore correct for 0.6.1 as-is.
- **Dataset format is unchanged.** `CODEBASE_VERSION` is still `v3.0` in 0.6.1,
  so `sim_to_lerobot.py`, `isaac_lab_lerobot.py`, and `groot.py` need no
  migration. Better, 0.6.x replaced the untyped `info.json` dict with a
  `DatasetInfo` dataclass whose `from_dict` explicitly *ignores unknown keys for
  forward compatibility*, and the field set exactly covers what our writers emit.
- **None of 0.6.0's other breaks apply.** This repo never passes
  `--dataset.vcodec`, never uses the `sac` policy type, and never writes
  per-frame `subtask_index`.
- **The validated B300 torch stack already satisfies 0.6.1.** torch 2.9.0 /
  torchvision 0.24.0 / torchcodec 0.8.1 sit inside 0.6.1's declared
  `torch<2.12,>=2.7`, `torchvision<0.27,>=0.22`, `torchcodec<0.12,>=0.3`. The
  Blackwell Dockerfile needs no torch change to build 0.6.1.

Two new entry points arrive in 0.6.x that this repo does not yet use:
`lerobot-rollout` and `lerobot-annotate`. `workbench.lerobot.policy_rollout`
currently implements rollouts through `lerobot-eval`, and the Physical AI Data
Factory blueprint has its own annotate stage. Both are worth a look later;
neither is upgrade-blocking.

## Defects in the current state

These exist today and are not caused by upgrading. They are the reason the
recommendation is "move forward" rather than "stay put".

**D1 — Two images force a dependency stack upstream 0.5.1 declares
incompatible.** `npa/docker/workbench/lerobot/Dockerfile` (non-0.6 branch) and
`npa/docker/workbench/lerobot-policy/Dockerfile` both run
`pip install --upgrade torch==2.12.1 torchvision==0.27.1 "diffusers>=0.38.0"`
*after* installing `lerobot==0.5.1`, which declares `torch<2.11.0`,
`torchvision<0.26.0`, and `diffusers<0.36.0`. All three violate. Because the
upgrade is a second pip invocation, pip does not re-resolve and the build
succeeds, so the breakage ships silently — this is the mechanical root cause of
the "0.5.1 fails LeRobot training at step 0 with a torch/torchcodec ABI
mismatch" note already in `CHANGELOG.md`. `npa-lerobot:0.5.1` is documented as
superseded, but **`npa-lerobot-policy:0.1.1` is a current pin** in
`[tool.npa.supported-tools]`.

**D2 — The manifest's torch pins are dead data.** `torch_pin`,
`torchvision_pin`, and `diffusers_pin` for 0.5.1 are read only by
`torch_install_pins()`, which has no production caller — the sole references are
its own unit test. The pins that actually execute are hardcoded in the two
Dockerfiles. A contract that nothing enforces will be trusted by the next author
and is already inconsistent with the VM installer, which pins neither
(`install_lerobot.sh` installs unpinned `torch torchvision` from the cu124
index).

**D3 — 0.6.0 has no hardware-validated image.** The 0.5.1 manifest entry carries
an `image_tag` and an `image_digest`; the 0.6.0 entry carries neither, so
`resolve_lerobot_image_tag("0.6.0")` falls through to the bare `0.6.0` semver
tag built from the CUDA 12 Dockerfile. Selecting `--lerobot-version 0.6.0`
therefore silently opts out of the Blackwell-validated image family that the
default uses. That bare tag does exist in the registry and runs — it is what D6
was reproduced on — it simply has no recorded digest and no hardware gate, which
is why a defect of D6's size could sit in it unnoticed.

**D4 — The stated reason for pinning the default at 0.5.1 is stale.**
`skills/tools/lerobot/SKILL.md` and `skills/workflows/byof-onboard/SKILL.md`
justify the 0.5.1 default as "keep for GR00T N1.5 / sim2real policy image
parity". But `npa-groot` clones NVIDIA's Isaac-GR00T directly and already ships
**N1.7** — it does not consume LeRobot's `groot` extra at all. Only the sim2real
parity half of that justification survives.

**D6 — `npa-lerobot:0.6.0` cannot run Diffusion Policy.** *(found by running the
image; fixed in this PR)* 0.6.0 moved `diffusers` and `transformers` out of the
base install and behind extras, and enforces them with `require_package()`
**inside each policy's `__init__`**. Our extras string was never updated, so
`lerobot[training,evaluation,pusht,libero]` installs no `diffusers`, the module
still imports cleanly, and the failure lands at
`make_policy()` — that is, at run time, on the real training command:

```
lerobot/policies/diffusion/modeling_diffusion.py:77 in __init__
    require_package("diffusers", extra="diffusion")
ImportError: 'diffusers' is required but not installed.
             Install it with: pip install 'lerobot[diffusion]'
```

0.5.1 is unaffected because `diffusers` is a *core* dependency there, which is
also why the default B300 image — built from a genuine 0.5.1 commit — was never
exposed. The blast radius is anyone who opts into `--lerobot-version 0.6.0`,
plus anyone who would have promoted 0.6.x to default. Two details corroborate
that the extras set has been drifting: the Dockerfiles hand-install `num2words`,
which is precisely what the `smolvla` extra supplies, and `libero` is the only
reason `transformers` is present at all (transitively, by accident).

Fixed here by requesting the extras the policies gate on
(`…,libero,diffusion,smolvla`) in both the manifest and the Dockerfile, and by
teaching the pre-flight to resolve the extras graph and cross-check it against
the `require_package` gates it finds in each policy module.

**D5 — The sim2real stack is hard-capped at LeRobot 0.4.4 by Python, not by
choice.** LeRobot has required **Python ≥3.12 since 0.5.0**. `npa-base` builds a
Python **3.11** venv and the CUDA 12 Genesis image builds on Ubuntu 22.04's
Python **3.10**. Genesis and the `npa-lerobot-vlm-rl` trainer derived from it
therefore cannot move past 0.4.4 (Feb 2026) without an interpreter change. This
is the one genuinely invasive item in this audit, and nothing in the docs
currently records it as a constraint.

## Proposal

### P1 — Add 0.6.1 as a supported version

Cheapest possible change, because 0.6.1's CLI contract equals 0.6.0's. The
manifest entry is the 0.6.0 entry plus an image tag:

```json
"0.6.1": {
  "pip_extras": "training,evaluation,pusht,libero,diffusion,smolvla",
  "train_env_eval_flag": "env_eval_freq",
  "eval_checkpoint_flag": "policy.path",
  "policy_eval_checkpoint_flag": "policy.pretrained_path",
  "torch_pin": null,
  "torchvision_pin": null,
  "diffusers_pin": null
}
```

Beyond the manifest, a version addition touches the same surface #191 did:
`pyproject.toml`, `deploy/images.py`, `lerobot/Dockerfile` (extend the `0.6.0`
branch condition to all 0.6.x), `lerobot/build.sh`, `terraform/variables.tf`,
`cloud_init.yaml.tpl`, `setup/install_lerobot.sh`, `smoke/golden_evals.yaml`, and
the four skill/doc files. `smoke/_versions.py` needs no change: its
`startswith("0.6")` test already yields `env_eval_freq` for 0.6.1.

The extras string carries the D6 fix forward; without `diffusion` and `smolvla`
a 0.6.1 entry would inherit the same broken policy coverage 0.6.0 shipped with.

Note that 0.6.1 tightens `diffusers` to `<0.40.0,>=0.38.0` — the opposite
direction from 0.5.1's `<0.36.0`. Any image that pins diffusers must pin it
per-LeRobot-version, not globally.

### P2 — Fix D1 before, not with, the upgrade

The conflicting force-upgrade should be removed from
`lerobot-policy/Dockerfile` and from the non-0.6 branch of `lerobot/Dockerfile`,
and `lerobot-policy/requirements.txt` moved to a LeRobot version whose declared
bounds match the torch stack actually wanted. This is worth doing on its own
merits: `npa-lerobot-policy:0.1.1` is a live pin serving the BYO-policy path, and
today it ships a dependency set upstream says will not work. Rebuilding it on
0.6.1 resolves the conflict and the ABI mismatch in one move, since 0.6.1 is the
first release whose `diffusers>=0.38.0` floor matches what the image wants.

### P3 — Delete or wire up the manifest torch pins (D2)

Either give `torch_install_pins()` a real caller (have the Dockerfiles read the
manifest rather than hardcode) or drop the three keys. The first is better — it
makes `audit_lerobot_release.py` a genuine gate on the image build. The second is
acceptable. Leaving unenforced pins in a file named "manifest" is not.

### P4 — Give the non-default version a validated image (D3)

Whichever version is *not* the default still needs an `image_tag` and
`image_digest` in the manifest, produced by the same
`validate_blackwell_image.sh` path as the default. Otherwise `--lerobot-version`
quietly switches users onto a different, less-validated image family.

### P5 — Record the Python 3.12 wall for the sim2real stack (D5)

Genesis and `npa-lerobot-vlm-rl` cannot follow LeRobot forward while their
interpreters are 3.10/3.11. Two options, neither cheap:

- Add a dedicated Python 3.12 `/opt/lerobot/venv` to the Genesis image, exactly
  as `Dockerfile.b300` already does for `npa-lerobot`, leaving Genesis itself on
  its own interpreter. This is the precedent-following option and keeps Genesis
  and RSL-RL untouched.
- Move `npa-base` to Python 3.12 and rebuild the dependent image family. Larger
  blast radius across every tool built on `npa-base`.

Until one of those lands, the docs should state plainly that the sim2real
trainer runs LeRobot 0.4.4 and that this is an interpreter constraint, so nobody
reads the `npa-lerobot` version selector and assumes the sim2real path follows.

### P6 — Keep the default at 0.5.1 until the gates pass, then move it

Promoting the default is the one change with real blast radius: it moves golden
evals, the sim2real policy image, and every `npa workbench lerobot` invocation
that does not pass `--lerobot-version`. Sequence it as add-then-promote, and
when promoting, correct the stale GR00T N1.5 rationale in the two skill files
(D4) rather than copying it forward.

## Executed validation

Static pre-flight is triage, not a merge gate, so the audit's conclusions were
executed against a real install. This is what found D6.

**Real LeRobot 0.6.1, dev VM, Python 3.12.13, torch 2.9.0, CPU.** Installed
`lerobot[training,evaluation,pusht]==0.6.1` into a clean venv and ran the
contract rather than reading it:

| Claim | Result |
|---|---|
| 19/19 bindings resolve | all 19 **import** for real, including `DiffusionPolicy` and `SmolVLAPolicy` |
| `env_eval_freq` replaces `eval_freq` | `lerobot-train --help` offers `--env_eval_freq`, and no longer `--eval_freq` |
| `--policy.path` still works | accepted — parsing reaches a Hub lookup, while a bogus `--policy.x` is rejected as unrecognized. It is absent from `--help` because draccus resolves the path alias before argument parsing, which is easy to misread as a break |
| adapters need no migration | a dataset written by `npa/adapter/sim_to_lerobot.py` loads under 0.6.1 `LeRobotDataset`, with video frames decoded and the expected `observation.state` / `action` / `observation.images.*` keys |
| the workbench train path works | `lerobot-train --policy.type=act` runs real steps on that dataset and checkpoints (exit 0) |

**The gap between "imports" and "works".** Every policy imported, so a smoke
test built on imports would have passed. Constructing them is where 0.6.x
actually enforces its extras, and that is the failure the workbench would hit:

```
--policy.type=act        exit 0   trains and checkpoints
--policy.type=diffusion  exit 1   ImportError in make_policy(): 'diffusers' is required
--policy.type=diffusion  exit 0   trains and checkpoints, after adding lerobot[diffusion]
```

### Executed validation — GPU

The above is a hand-built CPU venv, so it proves the *contract* but not the
*artifact*. Pulling the shipped `npa-lerobot:0.6.0` image onto an H100 80GB
(driver 580.126.09, CUDA 13.0) confirms the defect is in the image itself:

```
lerobot 0.6.0 | torch 2.11.0+cu130 | cuda available: True | NVIDIA H100 80GB HBM3
diffusers in image:    0        <- absent
transformers in image: 1        <- present, but only transitively via [libero]

A. DiffusionPolicy as shipped  -> ImportError: 'diffusers' is required but not installed
B. ACTPolicy as shipped        -> ValueError: You must provide at least one image ...
C. DiffusionPolicy + diffusers -> ValueError: You must provide at least one image ...
```

B is the control: `ValueError` means construction got past dependency checks and
reached ordinary config validation of the deliberately empty config used here.
C returning *the same* error as B is the point — once the extra is present,
Diffusion behaves exactly like ACT, so the missing extra was the entire defect.

Incidentally the image ships `torch 2.11.0+cu130`, which already sits inside
0.6.1's declared `torch<2.12,>=2.7`.

The pre-flight now models this: it resolves the manifest's extras through the
self-referential extra graph (`diffusion` → `diffusers-dep` → `diffusers`) and
cross-checks the result against the `require_package` gates it finds in each
policy module. Run against the pre-fix manifest it reports:

```
LeRobot 0.6.0
  [FAIL] policy-extras: --policy.type=diffusion needs diffusers (add extra 'diffusion')
LeRobot 0.5.1
  [PASS] policy-extras: lerobot[pusht,libero] constructs ['act', 'diffusion', 'smolvla']
```

## Remaining gates before 0.6.1 becomes default

The above covers the API, CLI, dataset, and extras contract. What is left is
image-level and needs a build, matching the precedent set by #191:

1. Build `npa-lerobot:0.6.1` from the CUDA 12 Dockerfile and the B300 variant.
2. On real GPU: `python -m npa.smoke.test_lerobot_env` (4/4) and
   `python -m npa.smoke.test_lerobot_functional` (5/5, 50-step PushT train +
   eval) with `NPA_LEROBOT_VERSION=0.6.1`.
3. `validate_blackwell_image.sh` for the B300 tag, recording `published_digest`
   into `blackwell-dc-images.json` and the manifest.
4. A `workbench.lerobot.policy_train` workflow run end-to-end, since that stage
   installs npa into `/opt/lerobot/venv`.

One caveat the CPU rig surfaced and could not settle: in a CPU-only venv,
`torchcodec` 0.11.1 failed to load against `torch 2.9.0+cpu` and LeRobot fell
back to PyAV. That is a property of the ad-hoc CPU rig, not of the workbench
images, which pin torchcodec 0.8.1 against cu130 torch — but it is the same
torch/torchcodec ABI coupling that D1 describes, so step 2 should confirm
torchcodec loads rather than silently falling back.

## What this audit does not cover

It reads wheel source with `ast` and never imports LeRobot, so it says nothing
about runtime behaviour, kernel compatibility, throughput, or checkpoint
portability. It does not evaluate whether pre-0.6 public checkpoints such as
`lerobot/diffusion_pusht` migrate cleanly to the 0.6 processor format — a known
issue already noted in `CHANGELOG.md` and unchanged by 0.6.1. It does not assess
the new `lerobot-rollout` and `lerobot-annotate` entry points beyond noting they
exist. And it makes no claim about registry bytes; image-level redistribution
and payload review remain governed by
`skills/atomic/solution-licensing/SKILL.md`.
