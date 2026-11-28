# CPU replay of 38 timeouts in the full-suite evaluation

Tasks 3, 6 and 9 accounted for 38 of the 71 timeouts in the 429/500 evaluation. Replaying all 50 initial states for each of those three tasks reproduced the 112 successes at their exact native step and all 38 timeouts at step 520. These are 150 of the original 500 trials. CPU replay added zero trials to the benchmark. [Full-suite score and trial records](README.md)

| Task | Original timeouts | Native goal conditions during replay |
| --- | ---: | --- |
| Mug and pudding | 15 | The mug reached the plate in all 15. It was on the plate at timeout in 11, including two that lost and regained placement. The other four ended with the mug off the plate. The pudding never reached its target in any of the 15. |
| Microwave | 12 | Nine never inserted the mug or closed the door. One closed the door without inserting the mug. Two inserted the mug but never closed the door. |
| Bottom drawer | 11 | All 11 ended with the bowl inside. The drawer never registered closed. The median time remaining after final bowl placement was 382 steps. The 39 successful episodes took a median of 79 steps from final placement to closure. |

In 10 of the 11 mug-and-pudding timeouts that ended with the mug on the plate, both pads touched the pudding and its maximum rise above its initial height was at most 1.55 cm. State 12 recorded no two-pad contact and an anomalous 13.24 m simulated rise. That state is excluded from the lift interpretation and included in the timeout count. [Per-goal records for all 38 timeouts](timeout-goals.csv)

The replay used MuJoCo 2.3.7 and robosuite 1.4.1 with rendering disabled. For each task, it restored states 0 through 49 in native order and executed the saved seven-dimensional commands directly, including the ten settling commands. It sampled both LIBERO goal predicates after each step. A contact pass in the same order reproduced every goal transition on all 38 timeouts. An exploratory pass that skipped successful episodes changed one drawer outcome and was excluded.

The CSV has two rows per timeout, one for each goal. A blank `first_true` means the goal never registered. A blank `last_loss` means it never changed from true to false. Each height belongs to the named object. For the pudding lift calculation, select states whose mug row has `terminal_true=True` and read `max_rise_cm` from the pudding row for the same state. Heights are in centimetres. Two-pad contact means both finger pads touched the object on the same simulator step. [Pinned LIBERO task definitions](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files/libero_10)
