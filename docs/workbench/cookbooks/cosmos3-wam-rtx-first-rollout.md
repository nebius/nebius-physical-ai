# A first scored Cosmos 3 Nano rollout on RTX

This route trains a two-update LIBERO policy on one eight-GPU RTX PRO 6000 Blackwell Server Edition host, reloads it, then scores it on a second host with one RTX GPU. Training qualification verifies two optimizer updates and a native checkpoint reload. The separate scoring step produces ten episodes, one per task. An episode that reaches the 520-step limit is a valid failure.

## Start from the public recipe

Use the full commit ID of the version you intend to reproduce. For the draft, copy the head commit from PR #1101. Set `WORKBENCH_REV` to that ID and use the same value on both hosts. Set `WAM_SHARED_ROOT` to the absolute private input directory on each host and `WAM_RUN_ROOT` to its absolute output directory.

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
git fetch origin refs/pull/1101/head  # only needed for an unmerged draft
git switch --detach "${WORKBENCH_REV:?set the full Workbench commit ID}"
test "$(git rev-parse HEAD)" = "$WORKBENCH_REV"
export WAM_RECIPE="$PWD/npa/workflows/workbench/cosmos3-wam-slurm"
```

1. **Prepare inputs.** Choose private absolute paths for the shared inputs and the fresh run directories. Follow [the pinned Linux installation](cosmos3-wam-slurm.md#2-install-the-pinned-native-training-environment) and [the two-host RTX staging commands](../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md#stage-the-two-single-host-runtimes). Use an account with access to the [pinned guardrail](../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md#native-vm-and-evaluation-additions). Verify the policy import on both hosts and the two-camera simulation probe on the evaluation host. A gated download or failed decoder probe needs fixing before training.
2. **Budget scratch.** The measured training inputs, converted base and one full checkpoint total about 247 GB before environments, caches and temporary files. Each extra full checkpoint needs about 177 GB. Allow at least 122 GB for a direct model receive or 243 GB if the receiving host holds a relay mirror and a second verified copy. Check the free space on the *actual output filesystem*. A RAM-backed checkpoint survives only while that filesystem exists.
3. **Train and reload.** On the training host use the [RTX smoke commands](../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md#rtx-two-update-smoke), including the IPC probe for a detached job. Use eager mode with the qualified driver 580.173.02 and PyTorch 2.10.0+cu130. Plan one host with eight ranks, two updates and seed 42, then run `qualify.py`. Use the fresh run only when `qualification/qualification.json` reports `passed`. Save that receipt and its `source_training_report_sha256` separately from the model transfer.
4. **Receive and score.** Use [the receiver and evaluator commands](../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md#verify-the-model-copy-before-timing-libero) on the second host. Pass the trusted digest to the receiver and require its transfer receipt when scoring. The receiver writes `native-training.json` and `transfer.json` to `EVAL_RUN`. Pass `--source-training-report "$EVAL_RUN/native-training.json" --require-transfer-receipt` to the evaluator. Use one worker, one environment, seed 42 and one trial per task. Inspect `quality.json` and the worker's native task summaries. If you already have authorized access to a qualified update-2 source run and its trusted digest, start at this step after preparing the evaluation host. The public evidence contains measurements and hashes. Model weights require a separately accessible checkpoint.

After scoring, run this on the evaluation host with `EVAL_OUTPUT` set to the output directory passed to `evaluate.py`. Each row gives the task ID, initial-state index, native success flag and episode step count.

```bash
jq -r '.task_results[] | .task_id as $task | .episode_results[] |
  [$task, .episode, .success, .steps] | @tsv' "$EVAL_OUTPUT/quality.json"
```

## Measured runtime for the first rollout

A clean run from prepared inputs passed the training supervisor on its first attempt. All eight ranks exited successfully after update 2. The native reload verified the full checkpoint. Seed 42 then scored 0/10 on the second host, one initial state per task, with zero simulator or inference errors.

| Operation | Wall time | Resources |
| --- | ---: | --- |
| Training qualification, including save, hashing and reload | 9.3 min | 8 GPUs reserved, 1.24 allocated GPU-hours |
| Model-only network relay | 5.5 min | 121.41 GB |
| Receiver copy and verification | 3.5 min | Separate scoring copy |
| Checkpoint verification, server startup and ten scored rollouts | 31.5 min | 1 GPU assigned, 0.53 assigned GPU-hours |

The checkpoint files were on RAM-backed scratch. The 34.88-second save is specific to that storage. In the [earlier recovered run](../evidence/cosmos3-wam-rtx-update2.json), saving a comparable checkpoint took 596.85 seconds. The relay retry created a missing destination scratch directory before any model bytes were sent. The times above cover the successful operations. [Clean-run receipt and per-task results](../evidence/cosmos3-wam-rtx-clean-update2.json)

## Inspect a demonstration before reserving training GPUs

The [worked reference](../evidence/cosmos3-wam-data/README.md) shows both camera views, timestamps, raw actions and native normalized values. The reference loader expects 20 Hz, front and wrist views, eight state values and seven action values per frame. For this dataset the actions are native-frame translation and axis-angle deltas plus one gripper command. The loader converts rotation to six values and applies the reference quantile statistics to the resulting ten-dimensional action.

With the staged framework interpreter, run the data-only preparation on CPU to download and check the reference dataset, then inspect episode 0. Run the same command with your local LeRobot v3 dataset and its episode number. Set `LOCAL_DATASET` to its absolute directory and `LOCAL_EPISODE` to an episode index present in its metadata. Use a fresh output directory under an existing parent.

```bash
export WAM_PYTHON="$WAM_SHARED_ROOT/framework/.venv/bin/python"
"$WAM_PYTHON" "$WAM_RECIPE/prepare.py" --shared-root "$WAM_SHARED_ROOT" --data-only
"$WAM_PYTHON" "$WAM_RECIPE/inspect_episode.py" \
  --dataset "$WAM_SHARED_ROOT/data/libero_10" --episode 0 --output-dir reference-window
"$WAM_PYTHON" "$WAM_RECIPE/inspect_episode.py" \
  --dataset "$LOCAL_DATASET" --episode "$LOCAL_EPISODE" --output-dir local-window
```

Open `front.png` and `wrist.png`. In `window.json`, compare the 17 state timestamps and 16 raw actions to your controller log. The inspector checks decoded presentation times against each camera's episode offset in its shared video file. It also shifts expected times by 25 ms and checks that the misalignment is detected. For the pinned dataset, [the native exporter](../evidence/cosmos3-wam-data/README.md) produces the converted and normalized actions for comparison.

Use your controller specification to identify units, the action frame, absolute versus relative motion and which gripper value means open. Matching array sizes and timestamps cannot establish those conventions. If your data differs, document the camera mapping or action conversion and identify matching normalization statistics and an evaluator before training. Keep local recordings and inspection outputs private.

For an independent walkthrough, have an engineer new to Cosmos record the checkout revision, one native rollout outcome and step count, and one local episode's mapping or incompatibility. Record where they needed help and have them retry the revised instruction.
