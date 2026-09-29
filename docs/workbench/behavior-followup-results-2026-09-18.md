# BEHAVIOR development follow-up — September 18, 2026

The balanced three-task fine-tune did not improve the published RLC policy.
Across the complete matched panel, stock scored **Q=0.404444 with 5/30
successes**, compared with **Q=0.227778 with 2/30** for the fine-tune. The
candidate improved four cases, tied eleven, and worsened fifteen. This result
does not support replacing stock with the balanced checkpoint.

This follow-up evaluates a checkpoint preserved from the
[September 17 experiment](behavior-experiment-results-2026-09-17.md). No new
optimizer updates were performed. See the [challenge guide](behavior-challenge.md)
for evaluation rules and [policy research](behavior-policy-research.md) for the
training rationale.

Including the separately qualified Meta results below, this window collected
**76 completed rollouts and 512,716 simulator steps**. Across both experiment
windows, that is **156 completed rollouts and 1,010,824 steps** on the same
30 distinct development task/instance pairs, plus the earlier 18,000 optimizer
updates. Repeated rollouts do not expand task coverage.

## Protocol and scope

Both policies ran tasks 0 (`turning_on_radio`), 1 (`picking_up_trash`), and 22
(`putting_shoes_on_rack`) on development instances 311–320. These are the same
30 task/instance pairs used in the earlier experiment, not an unseen test set.
Every policy/case combination received one rollout with the unchanged
BEHAVIOR-1K v3.9.2 evaluator, official robot and observation wrapper, and default
task timeout. Policies received only the three permitted RGB images and robot
proprioception. No evaluator state entered inference.

All six cells completed 10/10 cases. Collectors verified the original evaluator
JSON and MP4 hashes and decoded every video. The analysis required the exact
case list, checkpoint identity, source revisions, and linked training receipt;
it rejected incomplete cells. The 60 rollouts executed **442,179 simulator
steps**. Q is the official goal-completion score, including partial credit.
The [per-case metrics CSV](behavior-followup-cases-2026-09-18.csv) records each
completed rollout's policy, task, instance, Q, success, simulator steps, and
cell completion status.

This follow-up ran no reporting cases and created no submission ZIP. The
earlier official-baseline reporting run remains separately recorded at Q=0.00;
reporting data has therefore been used previously, although this follow-up uses
only development cases. A three-task development mean is not the full
1,000-case challenge score.

## Complete stock-versus-balanced comparison

| Policy | Task | Mean Q | Successes | Simulator steps |
| --- | --- | ---: | ---: | ---: |
| Published RLC checkpoint 2 | Radio | 0.300000 | 3/10 | 28,916 |
| Balanced fine-tune | Radio | 0.200000 | 2/10 | 28,928 |
| Published RLC checkpoint 2 | Trash | 0.433333 | 1/10 | 76,353 |
| Balanced fine-tune | Trash | 0.233333 | 0/10 | 79,020 |
| Published RLC checkpoint 2 | Shoes | 0.480000 | 1/10 | 113,062 |
| Balanced fine-tune | Shoes | 0.250000 | 0/10 | 115,900 |

| Candidate minus stock | Higher Q | Equal Q | Lower Q | Mean Q change |
| --- | ---: | ---: | ---: | ---: |
| Radio | 1 | 7 | 2 | -0.100000 |
| Trash | 2 | 3 | 5 | -0.200000 |
| Shoes | 1 | 1 | 8 | -0.230000 |
| Complete panel | 4 | 11 | 15 | -0.176667 |

The fine-tune also used 5,517 more simulator steps and had three fewer
successes. These are descriptive results from one rollout per policy and case;
they do not estimate significance or performance on the other 97 tasks.

## Repeating the stock control

| Stock measurement | Equal-task mean Q | Successes | Simulator steps |
| --- | ---: | ---: | ---: |
| Earlier September 17 window | 0.427778 | 7/30 | 208,775 |
| This follow-up | 0.404444 | 5/30 | 218,331 |

The repeated stock policy improved eight cases, tied twelve, and worsened ten.
Radio and trash kept the same task means; shoes changed from Q=0.55 to Q=0.48.
Individual successful cases also changed. The checkpoint, cases, policy recipe,
and evaluator match where verified, but Workbench revisions and workflow/runtime
artifact bytes differ. The audit therefore cannot attribute the differences
solely to simulator or policy randomness. Both measurements remain visible, and
the fine-tune comparison uses this follow-up's stock control.

## Meta100 evaluation

The separately frozen public Meta100 step-139999 checkpoint completed ten radio
cases and six trash cases. Each comparison below uses exactly the same cases
from stock, including only instances 311–316 for trash.

| Policy | Task and evaluated instances | Mean Q | Successes | Simulator steps |
| --- | --- | ---: | ---: | ---: |
| Stock | Radio, 311–320 | 0.300000 | 3/10 | 28,916 |
| Meta100 | Radio, 311–320 | 0.200000 | 2/10 | 29,489 |
| Stock | Trash, 311–316 | 0.555556 | 0/6 | 47,412 |
| Meta100 | Trash, 311–316 | 0.555556 | 2/6 | 41,048 |

On radio, Meta improved two cases, tied five, and worsened three. On the trash
prefix, it improved two, tied three, and worsened one: average Q stayed equal,
despite two more full successes and 6,364 fewer simulator steps. This small
prefix does not establish an improvement on the task. Shoes remains unscored,
and no complete three-task Meta mean is reported.

After the radio evaluator finished, its workflow failed while creating an input
manifest upload snapshot beside a read-only source file. The original summary,
attempt ledger, policy provenance, JSON results, and videos were already in
storage, but the final input manifest was absent. The strict complete-cell
collector correctly refused this incomplete publication.

A separate collector preserved a stable snapshot of all 84 original objects,
verified their hashes, and fully decoded all ten videos. It explicitly records
the exact frozen input manifest as **collector-supplied**, rather than an
original uploaded artifact. Its evidence class remains `partial_cutoff_prefix`
even though the evaluator completed 10/10; it makes no strict complete-cell or
reporting claim. The CSV identifies these rows as
`evaluator_complete_workflow_failed`. The workflow failure was not retried, and
no missing or interrupted case was assigned a zero score.

The trash coordinator stopped at its shutdown deadline, at 06:10:27 UTC. Its
seventh attempt, instance 317, remained unfinished. After controller removal,
the same supplemental collector verified a stable snapshot, decoded all six
completed videos, and compared only those six instances with stock. The CSV
marks these rows `cutoff_prefix`; stock and balanced rows marked `complete`
come from the strict ten-case collections. Missing Meta cases are omitted,
including the interrupted attempt. The trash manifest provenance is also
explicitly collector-supplied.

### Serving and startup repairs

The first task-1 startup failed
because the extracted author repository belonged to root while Git ran as the
worker user. It failed before the GPU inference check or any evaluator case.
The corrected bootstrap assigns the two restored runtime directories to the
worker; the checkpoint, normalization, policy code, cases, and evaluator remain
fixed. Workbench resumed that failed run under the same run ID. The subsequent
three workflows passed the frozen runtime, package, checkpoint, and source
checks, then stopped without uploading evaluation artifacts. The initial
diagnosis incorrectly inferred that the Gemma acceptance flag was missing by
looking only at the rendered YAML's `envs` field. The flag was present in
SkyPilot's separate `secrets` field and survived task deserialization. This
does not establish an environment-propagation bug.

Retained worker receipts identified the actual later failure: the restored
runtime lacked Chex, so policy imports failed before weights loaded. Installing
the exact Chex and Toolz versions from the author's lockfile passed a CPU import
check and full checkpoint construction. The next GPU attempt restored the
weights and reached server readiness, then its smoke client closed the
connection during the first action's JAX compilation. That client used a
20-second heartbeat timeout; the unchanged
[official client](https://github.com/StanfordVL/BEHAVIOR-1K/blob/b1979916ec1549b10a4e65e630bc6504a9af1b00/OmniGibson/omnigibson/eval/utils/network_utils.py#L88-L95)
explicitly uses 300 seconds. The corrected smoke client adopts its 60-second ping
interval and 300-second timeout while retaining the two finite-action and reset
checks.

The private recovery launcher also misinterpreted Workbench's retry count as an
attempt ordinal, allowing an extra automatic startup attempt. The corrected launcher
allows one new attempt per invocation and records the ordinal separately. Failed
startup attempts remain in the evidence; no scored case was selected for retry.

The repaired GPU check passed: full checkpoint restore, server readiness, and
two finite 23-element actions across a reset. The synthetic observations
contained only 61-element proprioception and the three permitted RGB images.
The policy process peaked at **7,326 MiB on a 97,887 MiB device** during this
short check. This is a serving measurement, not a rollout score or certification
on a 24 GB GPU.

The coordinator then stopped before evaluation because the private package builder
had excluded a required nested `MANIFEST.json`. Its expected bytes already existed
in the prepared source directory; the corrected package includes that exact
file. Readback of the retained shoe run directory found no evaluation-start
marker, claim, or attempts. Changing that workflow after a declared output exists
requires a new run, which was not added within this window's cleanup constraints.
Radio and trash each passed their own GPU checks with the corrected package and
started the official development evaluator. Their case lists remain fixed at
instances 311–320, with no retry after evaluation began.

A diagnostic reviewed the first chronological Meta radio case, instance 311,
against its stock control. Meta received Q=0 after 3,225 steps; stock received
Q=1 after 2,009. Four evenly spaced frames from each fully decoded video showed
Meta approaching the table without visible radio contact in the samples, while
stock visibly reached the radio. The samples showed no gross joint or camera
failure, but cannot validate action-channel mapping or establish a cause. This
single-case diagnostic is separate from the aggregate evaluation.

A second diagnostic reviewed Meta's first trash success, instance 314, against
stock: Q=1 after 5,494 steps versus Q=2/3 after 7,902. Both videos decoded
fully. Four fixed-position samples showed coherent navigation and manipulation;
the final Meta sample showed a can at or inside the receptacle opening, while
stock's receptacle lay on its side with a can outside. This example was selected
because Meta succeeded, so it is not representative evidence or a causal
explanation of the policy difference.

A source audit found no concrete mismatch in the frozen adapter's state indices,
camera order, numeric task conditioning, normalization, or action/controller
order. The checkpoint evidence does not bind its training-time task metadata,
and no retained test compares the original author entrypoint with this adapter
on the same real observation. These checks support the serving contract without
establishing exact reproduction of the author's training run.

## Shutdown and resource verification

The authorized window began at 19:10:26 UTC on September 17, with a hard end
at 07:10:26 UTC on September 18. Cleanup was scheduled one hour before that
end. The trash coordinator independently stopped evaluation at 06:10:27 UTC;
subsequent cleanup delays did not extend its scoring window.

The first cleanup prestart failed because the operator disk was full. After
preserving test logs and verifying that the owned test processes had exited,
removing only their temporary directory allowed cleanup to start at 06:15:47.
The frozen cleanup commands then exposed a second environment problem: their
system Python lacked PyYAML, although the virtualenv precheck had passed. A
temporary, hash-bound dependency-path file let those same commands use the
existing virtualenv packages. It was removed after cleanup.

All **25 cleanup checkpoints** verified their expected terminal state: nine
runs, nine controllers, six volumes, and one fleet. Provider inventory verified
zero campaign compute instances, disks, filesystems, Kubernetes clusters, and
IP allocations at **06:32:43 UTC**, about 38 minutes before the hard end. An
independent provider read confirmed those five zero counts at 06:33:57 UTC.
The project, reserved capacity, artifact storage, and checkpoints were retained.

The disk-exhausted Linux test attempt is invalid validation evidence: its raw
log is retained, but it produced no reliable final counts. It is not reported
as a suite pass.

## Reproducibility evidence

Exact operational evidence remains in access-controlled artifacts. These hashes
identify the inputs and completed comparison without publishing infrastructure
details:

| Artifact | SHA-256 |
| --- | --- |
| Stock checkpoint archive | `9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea` |
| Balanced checkpoint archive | `ec5ef7ba824acb44cd87fdb12e535fb9dc10faa01c4cdcfd763e5fd441d6cdfb` |
| Stock input manifest | `5aa0257428cf3d9e58f2095fd846631b776200ed1d56148b421ad4c317e95ad6` |
| Balanced input manifest | `5f1d66f21e76538c4467f2baa26e0594b2e3448df2081d42e26b2dc53d68d8f7` |
| Complete six-cell analysis | `c5fe52de93f6dff3335eefad5008ad190a1b909b897cd00b3a309c8b269834c0` |
| Stock repeat audit | `4f16c284ff5ca0b80f53b5d653d1699cbdf6699c30b20542dbd3955159b13f92` |
| Meta checkpoint archive | `33f592ac8fc543664da90470762e92616692c61d375610b8e608e3b4ef2e5362` |
| Meta frozen input manifest | `3a2bdb026ec6f29e54f60e5b8d5bb90b493a3278f9487877ad367b7f139c0513` |
| Meta radio supplemental collection | `01aa77607010a49047e60f85963fe1b8541335014af965c19ae79cf51c40fc66` |
| Meta trash prefix collection | `77613e4db541fc975f187648c9fd50ebb51df6fcd89f16bbd2cca226de8391ff` |
| Public per-case CSV | `db4dafd99bd841e55f3c0cfefcf305f5b8eb483074be57e36741ff39c10cfca7` |
| Provider-zero receipt | `94de1f73797817d88eaefea39ee1934c32eb3f59e22a68568eb7032e7e3bbcf6` |
| Temporary dependency-path removal | `a4b0d839f89a002dd6e59fe12e0a08302b4db556683a46237f4feb5e7139e027` |

The initial six workflows first stopped on a coordinator digest mismatch before
any evaluator case. Correcting that digest and passing all six workflow
validation and planning checks allowed native same-ID resumes. The failed
startup evidence remains retained; no scored case was selected or retried.
