# Present the workbench demo on a **new** agent UI

Use a dedicated agent deployment — **not** an existing shared `rtxpro/agent`
instance.

| Field | Value |
|-------|-------|
| Agent | `$PROJECT_ALIAS` / `workbench-demo` |
| Public URL | `$AGENT_PUBLIC_URL` (from `npa agent status`) |
| Rerun | `$AGENT_PUBLIC_URL/rerun/` |
| Auth file | `~/.npa/agents/$PROJECT_ALIAS/workbench-demo/auth.env` |
| Demo pack (S3) | `s3://$BUCKET/checkpoints/sim2real-b/demo-workbench-ui/reports/` |
| **Run ID (Artifacts / Rerun)** | `demo-workbench-ui` |

Story: **G1 sim → GR00T overlay → multi-env closed-loop → world model → curation**.

The default Rerun recording is **Unitree G1 + GR00T predictions** (cyan skeleton /
orange predictions). It is **not** the stock Franka `/world/franka` scene.

In the agent UI, select run **`demo-workbench-ui`**, then open
`groot-predictions-overlay.rrd` (or the default `sim2real.rrd` alias).

---

## 0) Resolve URLs and sign in (once)

On the operator/dev VM:

```bash
export PROJECT_ALIAS=rtxpro
export AGENT_NAME=workbench-demo
export NPA_NEBIUS_PROFILE="${NPA_NEBIUS_PROFILE:-npa-mk8s}"

source ~/.npa/agents/$PROJECT_ALIAS/$AGENT_NAME/auth.env
# AGENT_USER / AGENT_PASSWORD come from auth.env — do not commit them

eval "$(npa/.venv/bin/npa agent status --project "$PROJECT_ALIAS" --name "$AGENT_NAME" \
  | awk '/^public_url:/{print "export AGENT_PUBLIC_URL="$2}
         /^rerun_url:/{print "export AGENT_RERUN_URL="$2}')"
echo "AGENT_PUBLIC_URL=$AGENT_PUBLIC_URL"
```

In the browser:

1. Open `$AGENT_PUBLIC_URL/healthz` and accept the self-signed cert if prompted.
2. Open `$AGENT_PUBLIC_URL/login-help.html` and sign in with `AGENT_USER` /
   `AGENT_PASSWORD`.
3. Land on `$AGENT_PUBLIC_URL/`.

Keep this tab as the **only** demo UI. Do not reuse an older agent deployment.

---

## 1) Open with infra (30s)

**Terminal (dev VM):**

```bash
# RTX PRO cluster (Isaac / RT-core)
export KUBECONFIG=~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count

# 8× H200 demo host (start if STOPPED) — use the instance id from your live run
export NPA_NEBIUS_PROFILE=npa-mk8s
nebius profile activate npa-mk8s
# nebius compute instance get --id "$H200_INSTANCE_ID" | grep -E "name:|state:"
# if STOPPED: nebius compute instance start --id "$H200_INSTANCE_ID"
# then: ssh -i "$NPA_SSH_KEY" ubuntu@"$H200_VM_IP" 'nvidia-smi -L'
```

Talk track: “One 8× H200 host for Cosmos/GR00T; RTX PRO mk8s for Isaac/RT-core.”

---

## 2) Non-stock sim → policy (main visual)

### 2a. G1 Isaac trajectory + GR00T overlay (Rerun) — **open this first**

**Run ID:** `demo-workbench-ui`

In the UI: **Artifacts / Runs** → `demo-workbench-ui` →
`groot-predictions-overlay.rrd` (or `sim2real.rrd`).

Or from the operator VM:

```bash
source ~/.npa/agents/$PROJECT_ALIAS/$AGENT_NAME/auth.env
: "${AGENT_PUBLIC_URL:?set via npa agent status}"
: "${BUCKET:?set your artifact bucket}"

curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -H "Content-Type: application/json" \
  -d "{\"run_id\":\"demo-workbench-ui\",\"s3_uri\":\"s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/groot-predictions-overlay.rrd\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

**Rerun URL rule:** always pass an **absolute** `https://…/rerun/recordings/….rrd`
in the `?url=` query. Path-only values like `/rerun/recordings/sim2real.rrd` are
parsed by the Rerun web viewer as host `rerun` and fail under HTTPS.

```bash
# Encode an absolute recording URL for the viewer
REC="${AGENT_PUBLIC_URL%/}/rerun/recordings/sim2real.rrd"
OPEN="${AGENT_PUBLIC_URL%/}/rerun/?url=$(python3 -c 'import os,urllib.parse; print(urllib.parse.quote(os.environ["REC"], safe=""))' REC="$REC")&hide_welcome_screen=1&camera=workspace"
echo "$OPEN"
```

Or from the agent home page, use **Reload Rerun data** / **Open full Rerun**.

What you should see (not Franka):

- `/world/skeleton/*` — cyan G1 joints/bones + hip/knee/shoulder angle series
- `/world/predictions/*` — orange GR00T predicted trajectory overlaid

Optional: Isaac-only G1 skeleton (no predictions):

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -H "Content-Type: application/json" \
  -d "{\"s3_uri\":\"s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/isaac-lab-trajectory.rrd\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

MP4 companions (Artifacts / preview URL):

- `isaac-lab-trajectory.mp4`
- `groot-predictions-overlay.mp4`

### 2b. Multi-env closed-loop Rerun (rich camera / critique pack)

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -H "Content-Type: application/json" \
  -d "{\"s3_uri\":\"s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/sim2real-closedloop-multienv.rrd\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

Entities include `/heldout/camera/env-*`, `/heldout/scores`, `/signal/reward*`,
`/critique`. After loading, open the returned `rerun_iframe_url` (or **Reload
Rerun data**).

Also available: `sim2real-e2e-main.rrd`, `kinova-heldout-env06.mp4`.

### 2c. Stock Franka (optional contrast only)

Kept aside as `sim2real-franka-stock.rrd` / `isaac-franka-working-sim.mp4` —
do **not** lead with these.

Talk track: “G1 locomotion skeleton → GR00T action overlay → closed-loop
multi-env eval cameras.”

---

## 3) World model / Cosmos (1–2 min)

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -H "Content-Type: application/json" \
  -d "{\"s3_uri\":\"s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/cosmos-augmented-a.mp4\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

Play the preview URL in the agent UI (or Artifacts → `cosmos-augmented-a.mp4`).

Optional live infer (only if the H200 host is RUNNING and Cosmos is healthy):

```bash
npa/.venv/bin/npa workbench cosmos -p "$PROJECT_ALIAS" -n "$COSMOS_ALIAS" status
# prefer staged mp4 above if live infer is unavailable
```

---

## 4) Curation (1 min)

FiftyOne may be unavailable (registry image pull). Prefer the alt GR00T overlay
vs primary, or the second Cosmos clip:

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -H "Content-Type: application/json" \
  -d "{\"s3_uri\":\"s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/groot-predictions-overlay-alt.rrd\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"

curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -H "Content-Type: application/json" \
  -d "{\"s3_uri\":\"s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/cosmos-augmented-b.mp4\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

Talk track: “Rank / pick the better synthetic or predicted trajectory before
training.”

---

## 5) Close (30s)

```bash
set -a; source ~/.npa/live-e2e.env; set +a
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 ls \
  "s3://${BUCKET}/checkpoints/sim2real-b/demo-workbench-ui/reports/"
```

In the agent chat panel, ask: “what artifacts can I view?” — grounded reply
should list runs under `checkpoints/sim2real-b`.

Talk track: “Same HTTPS agent UI an LLM drives — Rerun, MP4, S3 artifacts.”

---

## Room tab checklist (this UI only)

| Beat | Open |
|------|------|
| Login | `$AGENT_PUBLIC_URL/login-help.html` |
| Home | `$AGENT_PUBLIC_URL/` |
| **Run ID** | `demo-workbench-ui` |
| **G1 + GR00T Rerun** | Artifacts → `demo-workbench-ui` → `groot-predictions-overlay.rrd` (absolute `?url=https://…/recordings/…`) |
| Multi-env closed-loop | same run → `sim2real-closedloop-multienv.rrd` |
| Policy / Cosmos / curation | Artifacts → `demo-workbench-ui` → mp4 / alt overlay |
| Infra | `kubectl get nodes …` + `nvidia-smi` on H200 |

Do **not** open an older agent URL (for example a previous `rtxpro/agent`
deployment).

---

## Pack contents (non-stock)

| Artifact | Role |
|----------|------|
| `groot-predictions-overlay.rrd` | **Default** — G1 skeleton + GR00T predictions |
| `isaac-lab-trajectory.rrd` / `.mp4` | G1 Isaac-only trajectory |
| `groot-predictions-overlay.mp4` | Overlay video companion |
| `groot-predictions-overlay-alt.rrd` | Curation alt overlay |
| `sim2real-closedloop-multienv.rrd` | Multi-env cameras / scores / critique |
| `sim2real-e2e-main.rrd` | Large e2e held-out camera pack |
| `kinova-heldout-env06.mp4` | Held-out camera sequence |
| `sim2real-franka-stock.rrd` | Old Franka scene (contrast only) |

---

## Recreate this agent later

```bash
cd ~/nebius-physical-ai
export NPA_NEBIUS_PROFILE=npa-mk8s NPA_SSH_KEY=~/.ssh/id_ed25519
# PROJECT_ID / TENANT_ID / REGION from your Nebius project — do not hardcode
nebius profile activate npa-mk8s
npa/.venv/bin/npa agent fresh-setup \
  --project rtxpro --name workbench-demo \
  --project-id "$PROJECT_ID" \
  --tenant-id "$TENANT_ID" \
  --region "$REGION"
# if TF refresh fails after VM create, register IP/instance_id then:
npa/.venv/bin/npa agent bootstrap --project rtxpro --name workbench-demo \
  --ssh-key "$NPA_SSH_KEY" --refresh-credentials
```
