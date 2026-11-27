# Cosmos 3 Nano on all 500 LIBERO-10 trials at update 1,000

The checkpoint succeeded on **429 of 500 trials (85.8%)**, 21 successes short of the 450 required for 90%. All 71 failures reached the native 520-step limit. There were zero simulator or inference errors.

| LIBERO-10 task | Successes / valid trials | Success rate |
| --- | ---: | ---: |
| 0. Alphabet soup and tomato sauce into basket | 48/50 | 96% |
| 1. Cream cheese and butter into basket | 40/50 | 80% |
| 2. Turn on stove and place moka pot on it | 46/50 | 92% |
| 3. Black bowl into bottom drawer, then close it | 39/50 | 78% |
| 4. White mug on left plate, yellow-and-white mug on right plate | 49/50 | 98% |
| 5. Book into back compartment of caddy | 46/50 | 92% |
| 6. White mug on plate, chocolate pudding to its right | 35/50 | 70% |
| 7. Alphabet soup and cream cheese into basket | 46/50 | 92% |
| 8. Both moka pots on stove | 42/50 | 84% |
| 9. Yellow-and-white mug into microwave, then close it | 38/50 | 76% |

Each task ran once on each of its 50 native initial states, indices 0 through 49. All episodes were newly scored, including states 0 through 2. The paired comparison used the same update-1,000 checkpoint. Its earlier scored episodes and the recorded microwave replays contribute zero trials to this full evaluation. [Paired cohorts and recorded replays](../cosmos3-wam-rtx-libero-paired/README.md)

The run used one training seed (42) and evaluation seed 42. Each of eight workers used one RTX PRO 6000 Blackwell Server Edition GPU for policy inference and one simulator environment. Every replica loaded the same update-1,000 checkpoint, with 30 denoising steps, guidance 1.0 and 20 fps. Its model manifest SHA-256 was `9e6323ea6f53cdaca450fa8d8fa22e5ddcd67da3242aca658a53839302f1c662`. [Training configuration, source revisions and checkpoint hashes](../cosmos3-wam-rtx-libero-paired/results.json)

The trial file gives the success flag and simulator step count for each task and initial state. A value of 1 means success. Every row with a 0 in this run reached 520 simulator steps. [Download all 500 trial outcomes and step counts](trials.csv)
