# Resume an interrupted CLIP Job

The [advanced CLIP application](../../npa/workflows/workbench/ray-clip-development/README.md)
can resume a partial driver result directory through another native Ray Job.
Keep the same application, worker, validation and canonical UDF bytes, model
snapshot, runtime, inputs, batch size and output path. Use a new submission ID.
The original Job must be terminal before another Job writes that directory.

After the original Job has committed multiple shards, stop that exact Job and
verify its terminal status:

```bash
ray job stop --address http://127.0.0.1:8265 "$ORIGINAL_JOB"
ray job status --address http://127.0.0.1:8265 "$ORIGINAL_JOB"
ray job logs --address http://127.0.0.1:8265 "$ORIGINAL_JOB"
```

Require `STOPPED`. Keep the full partial directory, including `execution.json`
and `shards/`. `RAY_CLIP_CHECKPOINT` logs identify committed later shards;
inspect the retained files after stop to establish which commits survived.
An in-flight inference can finish without its output being committed.

From the unchanged application source directory, submit the same application
options. For example, if the interrupted Job used two actors, 8,192 records,
32 records per batch and the path below:

```bash
ray job submit --address http://127.0.0.1:8265 \
  --submission-id "$RESUMED_JOB" --working-dir . -- \
  python application.py --actors 2 --records 8192 --batch-size 32 \
  --output-path /tmp/ray-clip-results/interrupted
ray job status --address http://127.0.0.1:8265 "$RESUMED_JOB"
ray job logs --address http://127.0.0.1:8265 "$RESUMED_JOB"
```

Every shard is preprocessed again to verify its input identity. Before
scheduling later GPU calls, the application validates all their existing commit
identities and Parquet hashes. Matching committed shards schedule no inference
and retain their original bytes, actor identity and timing. A missing commit
marker means unfinished work and is recomputed. An incompatible identity,
malformed commit marker, missing committed Parquet file or mismatched hash fails
the Job; it does not silently replace committed data.

Uncached shards are assigned by their scheduling order, so sparse shard indices
cannot put two first-wave tasks behind the same serial actor. The observation
barrier expects only actors receiving uncached work. With zero or one remaining
shard, completion requires no multi-actor overlap; the report records zero or
one participant and `concurrent_actor_inference_observed: false`. A wave with
multiple participating actors still requires actual observed overlap.

Read `report.json` together with the shard commits and native logs:

| Field or artifact | Meaning |
| --- | --- |
| `inferred_shards` | Shard indices inferred and committed by this invocation. |
| `reused_checkpoint_shards` | Valid committed indices retained from earlier work. |
| `final_actors[].inference_calls` | Completed calls by this invocation's surviving actors. |
| `inference_actor_seconds_sum` | Inference duration summed over newly committed shards. |
| `retained_checkpoint_inference_actor_seconds_sum` | Historical duration of reused shards, excluded from current inference time. |
| `preprocessing_task_seconds_sum` | Current preprocessing, including validation of reused inputs. |
| `shards/*/commit.json` | Original producer and inference timings; reuse never rewrites them. |
| `RAY_CLIP_INFERENCE` logs | Actual completed CUDA calls and their record IDs, independently comparable with the report. |

Worker monotonic clocks are local to their processes. Coordinator events show
the current first wave only; retained worker intervals do not demonstrate
overlap in the resumed Job. Model loading still occurs in newly created actors,
including when all shard outputs are cached. This is verified application
checkpoint reuse, not automatic Ray exactly-once execution or driver disk
durability. Final aggregation checks every record ID and persisted vector.
With `--recovery-check`, the deliberately killed first actor's call remains in
its shard receipt and native log but is absent from surviving actor counters.

A directory with an existing successful `report.json` remains protected against
another writer. Resume is for interrupted partial results. Preserve and hash
the completed outputs through the guide's download path before cleanup; worker
storage disappears with its infrastructure.

## Live regression

`npa/tests/e2e/test_ray_clip_checkpoint_live.py` uses the native Jobs SDK to stop
after multiple commits, resume unchanged work, reopen Parquet and Lance vectors,
and compare each call in the actor logs with checkpoint and actor counters. It
also exercises sparse, single, zero, uncommitted and corrupt checkpoint cases.
It provisions no infrastructure and builds no image.

Prepare an isolated Ray 2.58 environment with two compatible CUDA GPUs, the
guide's exact image and public model revision, an authenticated SSH path to its
driver, and the matching Ray client installed in the repository venv. Supply a
private JSON config through `NPA_RAY_CLIP_CHECKPOINT_LIVE_CONFIG`:

- `address`: loopback Ray Jobs address reached through the authenticated tunnel.
- `ssh`: SSH argv ending in the owned driver's destination; use verified host keys.
- `remote_python`: argv for the driver's prepared Python, optionally through
  `docker exec` when the driver runs in a container.
- `remote_root`: absolute, separately owned driver output root outside source.
- `evidence_dir`: owner-only local directory outside the checkout.

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_ray_clip_checkpoint_live.py -q
```

Retain the native status/log receipts and downloaded result trees privately.
The test stops its nonterminal Jobs on failure. The operator must preserve
evidence, cancel the exact hosting workload and remove only proven owned
infrastructure, then independently verify provider absence.
