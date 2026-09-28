# Run the public scan-to-policy demo

The [sample workflow](../../../workflows/testing/rgbd-scan-to-policy-demo.yaml)
downloads the complete public TUM `fr3/long_office_household` RGB-D sequence,
reconstructs its measured surfaces, packages matching Isaac collision geometry,
measures valid navigation resets, trains a policy, evaluates held-out goals, and
writes an offline HTML report. There is no manual capture conversion, case JSON,
checkpoint preparation, or cross-branch assembly.

Use an existing configured Workbench project with writable object storage and
an RTX PRO 6000 Kubernetes execution target:

```bash
npa workbench workflow demo run real-to-sim --project '<project-alias>' \
  --infra 'k8s/<rtx-context>'
```

The shared demo launcher performs normal workflow preflight and stages this
checkout. The underlying workflow can also be submitted directly:

```bash
npa workbench workflow validate-spec workflows/testing/rgbd-scan-to-policy-demo.yaml
npa workbench workflow plan-spec workflows/testing/rgbd-scan-to-policy-demo.yaml \
  --run-id preview
npa workbench workflow submit workflows/testing/rgbd-scan-to-policy-demo.yaml \
  --run-id '<unique-run-id>' --project '<project-alias>' --runtime \
  --infra 'k8s/<rtx-context>' --stage-src --var 'bucket=<your-bucket>' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

The configured project, credentials, GPU capacity, and applicable Isaac runtime
access must be ready. The workflow fetches the public sample itself; it requires
no Hugging Face token. Its CPU stages use the digest-pinned SONIC image's
Open3D/SciPy stack. USD assembly adds `usd-core==26.8` into an isolated temporary
site directory. Native physics, training, and rendering use the pinned Isaac
image. Source overlay staging supplies this checkout's adapters.

## What it runs

1. Verify the downloaded archive against SHA-256
   `c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c`,
   reject unsafe archive members, and prepare all 2,485 associated RGB-D frames.
2. Integrate 1,984 frames with real Open3D TSDF; check the 501 excluded frames
   against measured-depth quality gates before publishing the surface.
3. Assemble the colored measured triangle mesh and exact static colliders into
   a self-contained USDZ. No inferred floor or filled unknown region is added.
4. Derive a connected floor-support grid and distinct start/goal pairs from that
   mesh. Each selected reset's rotated footprint is raycast against the actual
   scan. Preserve separate training and evaluation cases.
5. Open the scene in native Isaac and validate the scan's measured-depth PhysX
   probes. Bind the scene, cases, image, source and controls into one recipe.
6. Train with 4,000 concurrent robots and 500 PPO iterations, then reload the
   exact checkpoint and evaluate 4,000 held-out resets for 300 control steps.
7. Publish the actual rollout frames, optional video, measurements and HTML report. The unchanged
   80% success gate determines success. A losing candidate remains a failed run;
   its completed evaluation is still available in the report.

`num_envs`, `iterations`, and `episode_steps` are explicit workflow configuration
values, overridable with `--var`. Smaller values are experiments, not equivalent
to the full reference qualification. Changing `num_envs` still requires enough
measured, unique reset pairs; the code will not duplicate cases to fill a cohort.
The standalone sample SDK can reuse a cached archive with
`navigation_sample.prepare_sample(output_path, archive_path=...)`; its SHA-256
is checked again. Neither download nor conversion silently samples fewer frames.

## View the result

```bash
npa workbench workflow demo view real-to-sim '<run-id>' --project '<project-alias>'
```

The report is at `reports/index.html` beneath the run prefix. That single file
embeds actual scored-episode thumbnails and measured evidence; it works offline
without downloading the full dataset or installing a video encoder. The focal
robot preview ends at its scored terminal step. Later observer motion remains
in the original recordings and is excluded from the scored preview. Playback
is a sampled-frame slideshow, and the cohort success rate covers every held-out
robot, not only the pictured one.

`reports/summary.json` is the machine-readable result. The report verifies the
actual checkpoint bytes, sealed recipe and held-out case identities, derived
success rate, configured threshold and carried reconstruction lineage before
presenting the result. Incomplete native evaluations receive an explicit failure
report without fabricated media or success measurements. Separate run prefixes
retain the complete capture, reconstruction, measured reset support, native
physics, trained weights, isolation controls and evaluation evidence.

The range-based navigation policy is evaluated on held-out goals in its
reconstructed training scene. A passing result does not establish unseen-site
transfer, four-camera policy learning, or physical-robot performance. Use the
[generic calibrated capture workflow](rgbd-scan-to-isaac.md) for other sensor
captures; the measured sample reset preset intentionally rejects another scan.

The prior component GPU evidence is described in that guide. A new combined
workflow's qualification is recorded separately in its
[readiness record](../../../workflows/testing/rgbd-scan-to-policy-demo.readiness.json);
validating or planning the workflow does not establish native execution success.

## Sample attribution

The workflow downloads data at runtime under
[TUM's CC BY 4.0 attribution terms](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers,
*A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
It retains `ATTRIBUTION.txt` and the attribution in every capture/report.
The preparation associates registered depth within 20 ms and interpolates
camera poses within 50 ms brackets. The original RGB index selects every fifth
frame for validation; exclusions do not shift that split. Calibration and depth
scale follow the [official TUM file format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
