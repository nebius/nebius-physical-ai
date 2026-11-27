# Eight actions per query produced five fewer successes across 150 paired states

The update-1,000 checkpoint scored **101/150 (67.3%)** with 16 actions per policy query and **96/150 (64.0%)** with eight. Both arms ran fresh episodes on the same 150 task/state pairs. A gain means only horizon 8 succeeded. A regression means only horizon 16 succeeded.

| Task | Horizon 16 | Horizon 8 | Gains | Regressions | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black bowl into drawer, then close it | 35/50 | 40/50 | 11 | 6 | +5 |
| Mug on plate and pudding to its right | 30/50 | 15/50 | 2 | 17 | -15 |
| Mug into microwave, then close it | 36/50 | 41/50 | 13 | 8 | +5 |
| **All three tasks** | **101/150** | **96/150** | **26** | **31** | **-5** |

Both horizons succeeded on 70 pairs and timed out on 23. To advance to a full-suite evaluation, horizon 8 needed both 133/150 successes and 21 net gains. It reached neither threshold. Horizon 16 remains the full-suite setting. The earlier full-suite evaluation scored 429/500. Its 112 successes on these same 150 initial states came from earlier episodes. [Full-suite evaluation](../cosmos3-wam-rtx-libero-full500/README.md)

## CPU replay located the largest loss at pudding placement

Ordered CPU replay reproduced the native outcome and exact step count of all 300 A/B episodes. Both finger pads touched the pudding in all 17 mug-and-pudding regressions at horizon 8. In 16, the mug was on the plate at timeout and the pudding never reached its target. In the other (state 9), the pudding reached its target and the mug was off the plate at timeout. The pudding reached its target at least once in 31/50 horizon-16 episodes and 17/50 horizon-8 episodes. It rose at least 2 cm above its starting height during two-pad contact in only 3/50 per arm. This lift measure alone could not distinguish successful placement.

In 10 of the 11 drawer gains, horizon 16 ended with the bowl inside and the closure predicate false. In five of the six regressions, horizon 8 ended that way. On microwave, the failing arm never inserted the mug in 10 of the 13 gains and seven of the eight regressions. [Per-state outcomes, steps and replay measurements](paired-trials.csv)

## Evaluation settings

Both arms used seed 42, task IDs 3, 6 and 9, and all 50 native initial states in order for each task. Each task and horizon had a fresh policy server on one RTX PRO 6000 Blackwell Server Edition GPU and one simulator environment. The server returned 16 actions per query with 30 denoising steps, guidance 1.0 and 20 fps. The native client used `--action_horizon 16` or `--action_horizon 8` to select how many actions to execute before querying again. Both arms used the 520-step cap and completed with zero simulator or inference errors. The GPU rollout times summed across workers were 4.59 hours at horizon 16 and 7.81 hours at horizon 8, excluding server startup and checkpoint verification. [Pinned evaluator](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/simulation/libero/closed_loop_eval.py) · [Checkpoint and source revisions](../cosmos3-wam-rtx-libero-paired/results.json)

In the CSV, `state` is the native initial-state index within a task. The `h16_` and `h8_` columns belong to the fresh arms. `prior_full_run_success` gives the earlier full-suite outcome for that key. Blank task-specific fields do not apply. The replay restored each native initial state and applied the saved environment actions in order, including the ten settling commands. Replays added zero scored episodes. [CPU diagnosis of the earlier full-suite timeouts](../cosmos3-wam-rtx-libero-full500/cpu-replay.md)
