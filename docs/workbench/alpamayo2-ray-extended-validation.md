# Extended Alpamayo Ray GPU validation

On September 12, 2026, a dedicated Ray cluster completed **340 real Alpamayo 2
Super inferences on two RTX PRO 6000 GPUs on separate physical hosts**. NVIDIA
telemetry recorded approximately **2 hours 20 minutes between first and last
GPU activity per worker**. All requested cases produced verified artifacts.

This extends the [earlier 20-case workflow validation](alpamayo2-ray-validation.md).
The implementation and reusable templates are described in the
[Alpamayo Ray guide](alpamayo2-super.md#ray-experiments).

## Experiment and Ray execution

The experiment followed the baseline, threshold selection and paired refinement
graph from the base templates:

| Stage | Cases | Configuration |
|---|---:|---|
| Baseline | 100 | Five manifest samples, seeds 42–61, 10 diffusion steps |
| Broad refinement | 200 | Same samples/seeds, 20 and 30 steps; selection threshold 0.0 m |
| Exploratory hard-case follow-up | 40 | Sample 2, same seeds, 50 and 100 steps; selected by the baseline's default 2.0 m gate |

The two compute applications completed through native Ray Jobs. Ray Core
scheduled two GPU actors concurrently in each stage, with one inference in
flight per actor. The final State API snapshot recorded 340 finished inference
tasks, six actor initializations, 11 finished CPU reduction tasks and 858
finished CPU telemetry tasks. Three Ray nodes ran on two physical GPU hosts;
the CPU head shared a physical host with one worker.

This extended run exercised the shared application SDK through native Ray Jobs.
It did not exercise standard SkyPilot workflow submission, KubeRay, or the
templates' default B200 allocation. The earlier run separately exercised the
workflow executor in Kubernetes Jobs. Ray token authentication was enabled;
an unauthenticated Jobs request returned HTTP 401.

The runtime fetched the gated dataset and model after access preflight passed.
The upstream source revision was
`beb2977d9a7e9d66837d4a3ad5144ff59de37519`, model revision
`00554695e729a6ff0b6281fd2c81b18d06e33dbe`, and dataset revision
`b719eea7f0a63619ef51ec7f54178af0937ef050`. The source archive SHA-256 was
`e29d3fdda10c6dd8c622fdd327740706e064720fb4fca583bd6a2e28910d5be6`.
Runtime, sweep and reduction module hashes matched the local checkout on all
three Ray nodes. Separate driver provenance also matched for the follow-up.

## Actual GPU use

| Measurement | Worker 1 | Worker 2 |
|---|---:|---:|
| First-to-last activity span | 2 h 19 m 12 s | 2 h 20 m 00 s |
| Mean utilization during that span | 9.73% | 10.44% |
| Peak utilization | 100% | 100% |
| Peak memory | 71,447 MiB | 71,447 MiB |
| Peak board power | 587.41 W | 614.47 W |
| Retained one-second samples | 9,278 | 9,276 |

These spans include idle intervals, preprocessing, weight reloads and rendering.
They are not continuous busy GPU time. The utilization-weighted estimates were
0.226 and 0.243 GPU-hours, respectively. Initial model downloading is excluded
from the activity span. Sample files contain no detected sampling gaps.
GPU process evidence matched the actual upstream inference commands.

The current adapter starts an upstream subprocess for each case. Downloaded
model files are cached, but model weights are not resident between cases.
The repeated memory allocation pattern and low mean utilization expose this
performance limitation; adding Ray workers alone does not remove it. A future
resident-model adapter needs separate correctness and throughput qualification.

Each worker used approximately 67 GiB of local model cache, with adequate local
disk space remaining. Unlike the earlier run, this study required no manual
model-blob relocation. Initial storage attempts encountered block-storage quota
and mounted-filesystem capacity constraints; the final inference cache used
local worker storage. No GPU worker restarted during the study.

## Measured trajectory errors

Values are means of the emitted minimum average displacement error (ADE), in
meters, over 20 matched seeds per scenario. Lower values are better.

| Manifest sample | 10 steps | 20 steps | 30 steps | 50 steps | 100 steps |
|---|---:|---:|---:|---:|---:|
| 0 | 1.2088 | 1.2333 | 1.2458 | — | — |
| 1 | 1.5129 | 1.6214 | 1.6645 | — | — |
| 2 | 2.1673 | 2.3257 | 2.3813 | 2.4331 | 2.4690 |
| 3 | 1.1925 | 1.2408 | 1.2581 | — | — |
| 4 | 1.1366 | 1.1795 | 1.1970 | — | — |

More steps increased mean ADE for all five clips. Only sample 2 exceeded the
default baseline selection threshold, and its 50/100-step follow-up also
increased mean error. The follow-up was exploratory, selected after observing
the baseline. These five clips do not establish general model quality or
driving safety.

Independent verification decoded all 340 case JSON/PNG outputs, checked report
hashes, exact case coverage, clip identity, pinned revisions, projection shape,
request settings and the executed sample/seed/step arguments. It recomputed
17 per-setting summaries and 240 paired comparisons from downloaded
measurements. All 12 overlapping cases from the earlier experiment reproduced
ADE/FDE exactly. Individual metrics were checked for consistency across
artifacts; raw trajectory coordinates were not emitted, so their geometry
was not independently recomputed.

## Bugs fixed and validation

The branch adds Ray sweep/refinement templates and fixes manifest fallback,
artifact validation, empty download results and renderer dependency formatting.
The extended work also fixed two failures found through testing:

- Agent planning truncated a complete tool catalog at the generic observation
  size limit. Catalog observations now preserve the complete result, with a
  regression that checks what the next planner receives.
- SkyPilot startup could mistake a temporarily undiscoverable, still-running
  child process for an exited server. Startup now retains the owned subprocess
  handle and waits while that process is alive. Regression cases reproduce
  one and three missed discovery scans and preserve failure on actual exit.

| Check | Result |
|---|---|
| Full Linux `make test` target, two pytest workers | 20,104 passed; 123 skipped; 1 XPASS |
| Package-only coverage | 75.98%; 60% gate passed |
| Guardrails after startup fix | 2,828 passed |
| Focused startup regressions | Failed before fix; 49 related tests passed after fix |
| Independent live artifact tests | Two passed for each of three reports |
| Ruff, generated CLI docs drift, diff confidentiality and secret scans | Passed |

The Linux environment included the repository development/adapter/SONIC
dependencies, official CPU PyTorch, and CLI prerequisites. Source hashes for
the tested fixes matched the checkout. The full target excludes separately
marked live/GPU/E2E tests; this is not a claim that every browser, scanner
comparison or CI job ran. Earlier macOS failures remain recorded in the
initial report; the complete Linux target now passes.

## Recording and operational handoff

Private evidence includes the actual browser viewport captured while observing
Ray State API data, Jobs logs, measured results and NVIDIA telemetry. It is a
recording of the monitoring browser, not the whole desktop. The startup segment
lasts 18 m 52.6 s; the main segment lasts 2 h 00 m 37.6 s, separated by a
31.25-second restart gap. Original-speed MP4 and a labeled 90× timelapse are
provided alongside the original WebM files.

The main capture stopped when a monitor reload failed. Its unfinalized WebM
was recovered by remuxing recorded packets without re-encoding; the original
bytes were retained. The recovered duration comes from packet timestamps,
not an invented completion timestamp. It shows the 300-case completion but
does not cover completion of the final 40 cases. A separate, explicitly
labeled post-run recording shows the final 340-case state.

Monitor refresh gaps over 30 seconds were approximately 45, 232, 627 and 46
seconds. The largest followed expiry of the operator's Kubernetes login;
telemetry observation resumed through CPU Ray tasks. The underlying one-second
GPU files continued without gaps. The monitor graphs display recent peaks;
the table above uses the complete telemetry files for averages.

Both compute applications succeeded. Inference actors exited, both GPUs had
zero remaining compute processes, and the telemetry application and samplers
stopped. Owned CPU test pods were removed. After operator authentication was
renewed, cleanup verified resource ownership and removed all three Ray pods,
both unused cache PVCs and their namespace. Both owned persistent volumes were
reclaimed, releasing the workload's GPU reservations. All four shared cluster
nodes remained Ready; the shared cluster, filesystem and bucket were preserved.
Private evidence retains the ownership checks and cleanup receipt.

Exact resource identifiers, credentials, recordings and gated camera artifacts
remain outside Git in private operator evidence.
