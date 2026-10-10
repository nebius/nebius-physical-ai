# Public policy training and deployed simulation proof

[One workflow YAML](../../../workflows/testing/policy-public-training.yaml)
executes public SmolVLA continued training and deployment through the standard
Workbench runtime. Every stage is discoverable in the tool catalog; the agent can
validate, plan, submit, inspect artifacts, resume and cancel using the existing
workflow surfaces. No customer data or private training scripts are needed.

Run on a Linux operator with Workbench installed, an authorized Nebius managed
Kubernetes context, configured private S3 storage and two available RTX PRO 6000
GPU workers. Training uses one GPU with 16 CPUs and 64 GiB; deployment uses two
separate GPU workers with 4 CPUs and 24 GiB each. CPU stages use 4 CPUs and 16 GiB.
The immutable published images are checked through the standard image preflight.
Public HF snapshots are revision-pinned and fetched without an HF credential.

If the existing cluster was provisioned outside this Workbench configuration,
first adopt its verified provider identity with `npa cluster kubeconfig
--cluster-name <provider-cluster-name> --project <project> --context <context>`.
This writes local state without provisioning another cluster. Use the resulting
kubeconfig for the checks and submission below.

```bash
npa workbench health preflight --checks s3,nebius --project "<project>" --json
npa skypilot verify --cluster "<context>" --kubeconfig "<kubeconfig>"
npa workbench workflow validate-spec workflows/testing/policy-public-training.yaml --json
npa workbench workflow submit workflows/testing/policy-public-training.yaml \
  --project "<project>" --infra "k8s/<context>" --runtime --stage-src \
  --max-wait-seconds 0 --image-bootstrap-timeout-seconds 0 \
  --var "bucket=<private-bucket>" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Select the exact kubeconfig with `KUBECONFIG` before submission. Secret names
resolve from the selected project's configuration or the process environment;
never put their values in YAML. The runtime persists each wave, provider job
identity, gate decision and artifact under the run prefix. For interruption,
use the same checkout/source selection, project/context, storage and isolated
runtime state with `--resume --resume-run <run-id>`. Training additionally saves
complete native model, optimizer, scheduler, RNG and sampler recovery state to
S3 during execution. Failed gates resume their preceding native candidate and
add another epoch increment. There is no implicit iteration or cost cap. When
submitting in a foreground terminal, stop its driver with Ctrl-C first, then
reconcile cloud cancellation with
`npa workbench workflow cancel <run-id> --project "<project>" --json`.
Wait for the exact recorded jobs to be terminal before teardown; an independently
running foreground driver can otherwise advance while cancellation is querying
the current stage.

The recipe in YAML controls epochs, batch size, seed, workers, evaluation trials,
deployment state offset and the success threshold. Change it before starting a
new run. The deployment set must follow both promotion sets and fit LIBERO's 50
initial states; overlapping or wrapping offsets are rejected. The standard
runtime workflow identity is sealed into the recipe and final proof.

The canonical recipe uses 40 specialist epochs and reserves deployment states
30–39 before training. Promotion evaluations use states 0–9 and 10–19. An earlier
10-epoch pilot passed both promotion gates but scored 5/10 on deployment states
20–29; the unchanged 70% requirement correctly rejected it. That result informed
the training change, so those earlier deployment states are not used as the
revised recipe's independent final check.

The split targets 90/5/5 within each task's unique trajectory groups, rounding
each holdout upward so small tasks retain both reserved sets. Actual episode
counts appear in the report; rounding can reduce the overall training fraction.
The selected public model is `HuggingFaceVLA/smolvla_libero`, with a frozen
SmolVLM backbone and a trainable action expert. The final specialist checkpoint
is served, rather than the original downloaded weights. Simulator scenes and
objects come from the revision-pinned `lerobot/libero-assets` snapshot.
The demonstration partitions are excluded from this run's optimizer updates.
Prior exposure by the downloaded pretrained checkpoint is not ruled out;
these results qualify the workflow and deployment, not novel-data generalization.

| Stage | Real computation and proof |
| --- | --- |
| Prepare | Fetch pinned `lerobot/libero`; audit trajectories and decode every episode preview. |
| Curate | Real FiftyOne measures brightness/sharpness and selects episodes; quality flags and the complete measurements are retained. This is preview-frame quality, without an object detector or an in-house review claim. |
| Split | Trajectory hashes group duplicates within each task; selected episodes become train, validation and test partitions. Training normalization is fitted using training frames only. |
| Pretrain | Native LeRobot SmolVLA via torchrun performs continued multi-task training. Actual optimizer loss, gradient norms, changed weights and durable checkpoints are retained. |
| Evaluate / first gate | Measure policy loss over all validation demonstration frames and native simulation completion on initial states starting at zero. Failure returns to pretraining. |
| Fine-tune | Start from the first promoted checkpoint; train only the chosen task's training episodes. |
| Evaluate / second gate | Measure loss over the reserved task test frames and native simulation completion on the next reset set. Failure returns to fine-tuning. |
| Export | Require both promotion records and export the exact final specialist with portable model/tokenizer dependencies. |
| Serve | One GPU runs authenticated HTTP SmolVLA inference; a distinct GPU worker renders native LIBERO while applying returned actions in MuJoCo CPU physics. The recipe reserves a separate reset set, disjoint from both promotion evaluations. |
| Report | Reconcile every served/applied action, fully decode every episode video and create `reports/demo.html`, `demo.mp4`, `proof.json` and checksums. Deployment qualification failure preserves the measured proof and fails the workflow. |

The HTML embeds footage, all deployment trials including failures, actual action
vectors, measured responses, fresh model chunk calls, native optimizer curves
and stage identities. It opens offline with no external libraries or network
requests. Only HTML, MP4 and sanitized proof JSON are public-safe; raw worker
artifacts, S3 locations, logs and the authenticated rendezvous remain private.

Collect the private run's `reports/` and `serving/{server,client}/` directories
through the configured artifact transport, then verify the same-run evidence:

```bash
NPA_INTEGRATION_E2E=1 NPA_POLICY_PUBLIC_RESULTS="<collected-run-directory>" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_policy_public_live.py -q
```

This implements the two training/gate loops and deployed benchmark-container
pattern as a public executable baseline. It does not qualify in-house foundation
pretraining, weighted customer data soup, physical robot transfer or concurrent
client batching. It uses managed Kubernetes and torchrun, **not Slurm**. The
separate [Slurm workflow](policy-training-slurm.md) supports operator training
scripts and Soperator access. The SDG pipeline remains separate.

Runtime configuration: `NPA_WORKFLOW_SHA256` is supplied by the renderer;
`NPA_VLA_RECOVERY_URI`, `NPA_VLA_RECIPE_SHA256` and
`NPA_VLA_SELECTION_SHA256` are supplied by the training stage, not operator
settings. GPU stages require CUDA and NVIDIA EGL. Native package installation
and stage errors retain private diagnostics under the stage's diagnostics prefix.

## Recorded live qualification

[Sanitized qualification record](../../../workflows/testing/evidence/policy-public-training-qualification.json) binds the workflow bytes, runtime source commit, changed checkpoints, both worker roles and exported media hashes. The complete standard runtime recorded 3727 pretrain updates, 2431 finetune updates; promotion results were finetune 7/10, pretrain 7/10. The independent two-GPU deployment completed 9/10 trials and reconciled all 947 served/applied actions. All three live artifact tests passed. Offline browser validation recorded zero network requests and JavaScript errors, with working video, chapter selection and mobile layout. Retry iterations actually observed: 0.

For an isolated local SkyPilot API, use its supported RSA service-account profile shape: `auth-type: service account`, `service-account-id`, `public-key-id`, and `private-key-file-path`. Normal derived token-cache renewal is supported with that stable key binding. Keep credential values outside the recipe and verify the selected operator profile before submission.
