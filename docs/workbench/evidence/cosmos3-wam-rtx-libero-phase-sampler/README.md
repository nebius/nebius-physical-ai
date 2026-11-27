# Doubling drawer and pudding phase windows yielded two fewer successes on 150 matched states

At update 1,100, the control solved 113 of 150 trials (75.3%) and the phase sampler solved 111 (74.0%). Both Cosmos 3 Nano policies ran fresh episodes on the same 50 initial states for each of three LIBERO-10 tasks. A gain is a state where only the 2× sampler succeeded. A regression is a state where only the 1× control succeeded.

| Task | 1× control | 2× sampler | Gains | Regressions | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3. Black bowl into bottom drawer, then close it | 38/50 | 35/50 | 9 | 12 | -3 |
| 6. White mug on plate, chocolate pudding to its right | 39/50 | 38/50 | 4 | 5 | -1 |
| 9. Mug into microwave, then close it | 36/50 | 38/50 | 12 | 10 | +2 |
| **All three tasks** | **113/150** | **111/150** | **25** | **27** | **-2** |

Both policies succeeded on 86 pairs and timed out on 12. The difference was -1.3 percentage points (two-sided exact McNemar p = 0.890). This single-seed comparison found no improvement from doubling the target windows. The [429/500 full-suite benchmark](../cosmos3-wam-rtx-libero-full500/README.md) measured the update-1,000 checkpoint on all ten tasks. Neither update-1,100 checkpoint received a full-suite evaluation.

## Drawer closure and pudding placement in the CPU replays

- **Drawer.** In all 12 regressions, the sampler ended with the bowl inside the drawer and the simulator's closure predicate false. The control ended the same way on eight of the nine gains. Across all 50 states, this outcome rose from 11 with the control to 15 with the sampler.
- **Pudding.** The pudding reached the target region at least once on 39 control trials and 38 sampler trials. In all five regressions, both finger pads contacted the pudding during the sampler episode, but it never reached the target. Four of the five ended with the mug on the plate. The control also contacted the pudding without placing it on all four gains.

One ordered CPU replay per native episode (150 per arm) reproduced all 300 outcomes and exact step counts. They add zero scored trials. The CSV gives one row per task and initial state, both native outcomes and step counts, and the replay predicates. Columns prefixed with `control_` describe the 1× arm and `weighted_` the 2× arm. Empty predicate cells belong to a different task. [All 150 paired outcomes](paired-trials.csv)

## Training and evaluation settings

Both arms resumed the same seed-42 checkpoint at update 1,000 and trained for 100 optimizer updates on eight RTX PRO 6000 Blackwell Server Edition GPUs. Both used all ten training tasks and 16-action windows. The control sampled each eligible window once per logical epoch. The sampler repeated windows wholly within either labeled phase once more. The drawer phase started at bowl release and the pudding phase at mug release. Across one complete logical epoch, the control enumerated 94,250 distinct windows. The sampler repeated 5,930 of them once, for 100,180 window occurrences. These counts describe a complete epoch. The runs used a fixed budget of 100 updates. The checkpoint contained no data-loader cursor, so both arms restarted the seeded sampling epoch.

Each checkpoint ran at action horizon 16 and seed 42 on native initial states 0 through 49 for each task. Each task and checkpoint used a fresh policy server and one simulator environment with a 520-step limit. All 300 native episodes completed without a simulator or inference error. The [earlier horizon comparison](../cosmos3-wam-rtx-libero-horizon/README.md) used the same states and scored the update-1,000 checkpoint at 101/150 with horizon 16.
