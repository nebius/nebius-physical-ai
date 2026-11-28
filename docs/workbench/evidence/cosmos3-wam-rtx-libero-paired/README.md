# Cosmos 3 Nano on LIBERO-10 at updates 500 and 1,000

One training run used eight RTX PRO 6000 Blackwell Server Edition GPUs, seed 42, and a global batch of 2,048. A second host reloaded each checkpoint and scored it on the same ten tasks. The table reports successful episodes out of valid trials. Each task/state pair contributed one episode per checkpoint.

| Initial state indices | Update 500 | Update 1,000 | Fail to pass | Pass to fail |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6/10 | 8/10 | 3 | 1 |
| 1 and 2 | 9/20 | 17/20 | 8 | 0 |

On state 0, the later checkpoint gained the mug-and-pudding, two-moka-pot, and microwave tasks. It lost the bowl-in-drawer task. The additional-state evaluation used three parallel workers per checkpoint and also repeated state 0 as a control. Those repeats scored 5/10 and 8/10, respectively. Only indices 1 and 2 contribute to the second row. Both comparisons cover one training seed and a subset of the 50 native initial states per task. The model manifest matched at transfer and scoring for each checkpoint. [Per-episode outcomes, steps, source revisions and checkpoint verification](results.json)

## Recorded replays of the microwave task

[Play the side-by-side video (MP4, 26 seconds)](microwave-recorded-replays.mp4). The robot must place the yellow and white mug in the microwave and close the door. Both panels start from the same state at seed 42 and play at 20 frames per second. The right panel holds its final frame when its episode ends.

| Task 9, initial state 0 | Update 500 | Update 1,000 |
| --- | --- | --- |
| Original scored episode | Timeout at 520 steps | Success in 249 steps |
| Fresh recorded replay | Timeout at 520 steps | Success in 244 steps |

The benchmark counts use the original scored episodes. The recordings illustrate one task and add zero trials to either cohort. [Simulation and model sources](NOTICE.md)
