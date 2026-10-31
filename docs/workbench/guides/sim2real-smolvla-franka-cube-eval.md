# SmolVLA Franka/cube experimental evaluation

This workflow evaluates an existing, qualified native SmolVLA checkpoint on the
Franka fixed-cube surface-placement simulation task. It is intentionally smaller
than the adaptation workflow: it does not collect demonstrations, export a
dataset, train, reload training output, qualify a new object or robot, or deploy
to a physical robot.

The supported entry point is the standard workflow CLI:

```bash
npa workbench workflow validate-spec workflows/testing/sim2real-smolvla-franka-cube-eval.yaml
npa workbench workflow plan-spec workflows/testing/sim2real-smolvla-franka-cube-eval.yaml \
  --run-id <run-id> --check-render \
  --var bucket=<bucket> \
  --var source_sha=<consumer-source-sha> \
  --var isaac_image=<exact-image-digest> \
  --var isaac_image_source_sha=<baked-source-sha> \
  --var runtime_image=<exact-image-digest> \
  --var runtime_image_source_sha=<baked-source-sha> \
  --var profile_uri=<qualified-profile-uri> \
  --var gold_collection_uri=<sealed-gold-uri> \
  --var adapted_checkpoint_uri=<selected-checkpoint-uri> \
  --var adapted_policy_adapter_uri=<checkpoint-adapter-uri>
```

After the same plan and normal health/access preflights pass, submit the same
spec with `npa workbench workflow submit ... --runtime`. The workflow creates or
adopts only its exact run-owned caches, authentication and service-controller
RBAC, starts the selected checkpoint service, evaluates 8 sealed scenarios in
each of three modes, emits native evidence, and deletes its exact service and
runtime prerequisites.

The policy receives workspace and moving-wrist images plus measured robot state.
Every control uses a fresh native prediction and explicit uniform temporal
ensembling. The three conditions are fresh wrist, frozen wrist, and the legacy
`pixel_channel_shuffle` identifier, which means a spatial-and-channel
pixel-value shuffle—not an RGB-only or temporal shuffle. The final standalone
HTML embeds synchronized consumed workspace/wrist views and independent
goal/release witnesses; the lossless policy-input array and per-query hashes are
authoritative.

Historical accepted evidence for the original c7 execution path is 5/8 fresh,
0/8 frozen, and 0/8 shuffled placements. Those measurements remain bound to
their original source, images and workflow. They motivate this scoped surface
but are not relabelled as a run of this new spec. The predeclared positive causal
wrist criterion was not met, so this workflow makes no causal-benefit claim.

This is an experimental fixed-cube simulation workflow. It does not establish a
sixteen-stage adaptation clean start, interrupted-training adoption, arbitrary
assets or embodiments, active view recovery, or physical-robot readiness.
