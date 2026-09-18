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

## Meta100 measurement in progress

The separately frozen public Meta100 step-139999 checkpoint is undergoing
startup validation for the same three-task panel. Its first task-1 startup failed
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
does not establish an environment-propagation bug. Worker-local smoke receipts
and logs are needed to identify where startup failed; absent uploaded artifacts
alone do not establish that no GPU work occurred.

No Meta100 rollout result or 24 GB serving claim is established by that startup
correction. Its completed measurements will be recorded separately from the
stock-versus-balanced result above.

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

The initial six workflows first stopped on a coordinator digest mismatch before
any evaluator case. Correcting that digest and passing all six workflow
validation and planning checks allowed native same-ID resumes. The failed
startup evidence remains retained; no scored case was selected or retried.
