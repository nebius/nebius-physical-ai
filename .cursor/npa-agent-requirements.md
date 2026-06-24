# NPA Agent — Requirements Addendum (read every loop iteration)

**MANDATORY:** Apply these requirements on top of the base `npa agent` loop goal. Do not regress existing deliverables.

---

## 1. Simulation visualization (Rerun timeline)

### What exists in NPA today

| Layer | Source | Behavior |
| --- | --- | --- |
| **Post-run `.rrd`** | Stage 14 `run_finalize` → `_run_sim2real_viz_stage` | `sim2real_viz.py` emits `reports/sim2real.rrd` from rollout frames, VLM critiques, held-out scores |
| **Hosted viewer (workbench)** | `sim2real_rerun_serve.py` | K8s LoadBalancer serves static `.rrd` from S3; nginx basic auth; gRPC 9876 — **not** agent primary iframe (see Rerun architecture); agent uses VM-co-located viewer |
| **Rollout cameras** | Policy rollouts / held-out eval | Multi-cam frames logged under `actions/train/<rollout_id>/frames/` (workspace, wrist) and held-out render episodes |
| **Agent stub** | `npa/src/npa/cli/agent.py` | Co-located uvicorn rerun stub + iframe at `/rerun/` — **must become real Rerun web viewer**, not placeholder HTML |

Rerun in this repo is **batch-first**: `.rrd` is written after stages complete. Live streaming during an active sim run is a **new agent-side bridge**, not yet in Sim2Real engine.

### Rerun architecture (HYBRID — not a pivot)

**Decision:** Keep agent sim viz **hybrid** with the workbench Rerun stack. Do **not** make the K8s Sim2Real rerun serve LoadBalancer the primary `/rerun/` iframe.

| Concern | Choice |
| --- | --- |
| **Primary viewer** | VM-co-located Rerun on the agent VM, same **nginx origin** as the agent UI (`/rerun/` iframe + basic auth). |
| **Runtime** | Prefer workbench **`npa-rerun-viewer`** Docker image (alias **`npa-sim2real-rerun-viewer`** until renamed) **or** `rerun-sdk>=0.32` on the VM. |
| **Shared code** | Factor shared helpers from `sim2real_rerun_serve.py` (serve command, nginx config, CORS, S3 sync) so agent deploy and workbench K8s serve share one implementation surface. |
| **K8s rerun serve pod** | **Not** primary iframe — avoids region, lifecycle, and auth split (cluster LB vs VM nginx). Remains the workbench operator path for post-run `.rrd` on cluster. |
| **v2 (optional)** | Agent backend proxies **gRPC** to in-cluster serve (`DEFAULT_GRPC_PORT=9876`) for live rollouts; v1 static `.rrd` polling stays default and fallback. |

**Image vs orchestration:** The viewer image is **generic** (rerun-sdk CLI / web viewer only). **Sim2Real-specific** wiring — S3 `.rrd` sync, nginx basic auth, K8s LoadBalancer lifecycle — lives in orchestration (`sim2real_rerun_serve.py`, agent deploy), not in the container image name or contents.

### Agent UI requirement — live sim run visualization

1. **Embedded Rerun panel** (`/rerun/` iframe, same nginx basic-auth edge):
   - **v1 (required):** Poll agent backend for the active run's latest `.rrd` URI (local cache or S3 proxy). Reload iframe when a newer recording lands (Stage 7+ rollouts, Stage 14 finalize). Show run-id + stage badge overlay.
   - **v2 (optional):** Backend proxies gRPC to in-cluster Rerun serve (`DEFAULT_GRPC_PORT=9876` from `sim2real_rerun_serve.py`) for live rollouts — **not** a separate iframe origin. Degrade to v1 polling when gRPC unavailable.
2. **Backend routes** (FastAPI on agent VM, port e.g. 8787):
   - `GET /api/sim-viz/status` → `{ run_id, stage, rrd_uri, rrd_updated_at, live_grpc_url?, mode: "static"|"live" }`
   - `GET /api/sim-viz/rrd` → proxy/download latest `.rrd` for authenticated clients (or signed redirect to S3)
3. **Agent chat integration:** When user asks to "watch the sim", agent calls `GET /api/sim-viz/status` and surfaces the iframe URL + current stage. After workflow submit, agent polls until `rrd_uri` is non-empty.
4. **CLI:** `deploy` / `status --json` include `sim_viz_url` (public URL to Rerun iframe path). `verify-live` checks it returns 200 behind basic auth.

### NPA artifact paths (for viz wiring)

```
s3://<bucket>/<prefix>/<run-id>/reports/sim2real.rrd          # Stage 14 canonical
s3://<bucket>/<prefix>/<run-id>/actions/train/<rollout>/frames/  # per-step camera PNGs
s3://<bucket>/<prefix>/<run-id>/stage_02_assets/consumed_scene_spec.json
```

Reuse `npa.workflows.sim2real_viz.emit_sim2real_rerun` helpers; no GPU imports in agent unit tests.

---

## 2. Camera angle inspector

### What cameras are in NPA

- **Schema:** `CameraSpec` in `npa/src/npa/genesis/scene_assets.py` — `name`, `placement` (`stock_workspace` \| `stock_ee_mounted` \| `custom`), `pos`, `look_at`, `fov`, `resolution` (H×W).
- **Stock defaults:** `DEFAULT_CAMERA_NAMES = ("workspace", "wrist")`; stock placements in `DEFAULT_CAMERA_STOCK` (`sim2real_assets.py`).
- **Custom input:** `cameras` block inside `SceneSpec` JSON **or** standalone `cameras.json` at `CAMERAS_URI` / `NPA_SIM2REAL_CAMERAS_URI` (merged by `merge_standalone_cameras_uri`).
- **Downstream use:** Stage 2 → envgen `embodiment` block; Genesis `FrankaPickPlaceEnv` and Isaac held-out eval read resolved cameras from consumed scene spec.
- **Rerun multi-cam:** `sim2real_viz.py` logs rollout frames per camera entity (`rollouts/iter_XX/<id>/cameras/<name>`).

There is **no standalone viewport API** on Genesis/Isaac workbench services today; camera config flows through Stage 2 JSON only.

### Agent UI requirement — camera angle inspector

1. **Panel section** (inside Sim Assets tab or dedicated **Cameras** sub-tab):
   - List cameras from active selection / Stage 2 consumed spec: name, placement, pos, look_at, fov, resolution.
   - **Select / highlight** one camera; show schematic top-down frustum (2D SVG from pos + look_at + fov) for v1.
   - **Feed preview (v1):** When an active run exists, show latest frame thumbnail for selected camera from rollout artifact tree (proxy via agent backend).
   - **Switch multi-cam view:** Toggle which camera feeds appear in Rerun blueprint sidebar (pass `?camera=workspace` to viz status or client-side Rerun entity filter).
2. **Backend routes:**
   - `GET /api/sim-assets/cameras` → `{ cameras: [{ name, placement, pos, look_at, fov, resolution, preview_url? }] }`
   - `PUT /api/sim-assets/cameras/selection` → `{ selected: ["workspace"] }` (session state for UI + viz filter)
3. **Editing (v1):** Allow editing `pos`, `look_at`, `fov` in UI → writes draft `cameras.json` to agent scratch S3 prefix → updates selection URIs (see §3). Stock placements remain read-only unless user switches to `custom`.
4. **Examples:** Follow `docs/workbench/guides/sim2real-customer-assets.md` camera templates (`cameras-custom.json.example`, `scene-spec-full.json.example`).

---

## 3. Sim asset panel — browse, inspect, **specify/select**

### What sim assets are in this repo

NPA **sim assets** = Sim2Real **Stage 2** scene / robot / camera configuration — distinct from Stage 14 Rerun timeline visualization.

| Artifact | Path | Content |
| --- | --- | --- |
| Scene | `stage_02_assets/consumed_scene_spec.json` | Stock tabletop or BYO `SceneSpec` (meshes, fixtures, cameras) |
| Robot | `stage_02_assets/consumed_robot_spec.json` | Stock Franka or BYO `RobotSpec` (URDF, preset) |
| Manifest | `stage_02_assets/assets_manifest.json` | Provenance / status record |

**Schemas:** `SceneSpec` (`scene_assets.py`), `RobotSpec` (`robot_assets.py`).

**Config URIs (operator / workflow):**

| Env var | Purpose |
| --- | --- |
| `ASSETS_URI` / `NPA_SIM2REAL_ASSETS_URI` | BYO mesh directory or single mesh |
| `SCENE_SPEC_URI` / `NPA_SIM2REAL_SCENE_SPEC_URI` | Full SceneSpec JSON on S3 |
| `CAMERAS_URI` / `NPA_SIM2REAL_CAMERAS_URI` | Standalone cameras JSON |
| `ROBOT_SPEC_URI` / `NPA_SIM2REAL_ROBOT_SPEC_URI` | Robot spec + URDF pointer |
| `ROBOT_PRESET` / `NPA_SIM2REAL_ROBOT_PRESET` | `franka` \| `ur5e` \| `flexiv` … |
| `NPA_SIM2REAL_SIM_BACKEND` | `isaac` (default) \| `genesis` |

**S3 layout for customer assets:** `s3://<bucket>/customer-assets/<task-id>/` — scene meshes (OBJ/STL/GLB/URDF), `scene-spec.json`, `robot-spec.json`, `cameras.json`. See `docs/workbench/guides/sim2real-customer-assets.md`.

**Not in scope v1:** USD/Omniverse viewer (`docs/architecture/partner-skills-roadmap.md`) — do not block on it.

### Agent UI requirement — sim asset panel

1. **Tab or split panel:** **Sim Assets** alongside embedded **Rerun** iframe (see layout below).
2. **Co-located service** on agent VM (same nginx basic-auth edge as Rerun):
   - Route: `/assets/` proxied to local FastAPI (e.g. port 9091).
   - **Browse:** Tree/list of known asset roots (configured `customer-assets/` prefix + stock presets).
   - **Inspect:** Render active manifest — scene objects (name, role, asset_source, uri, sha256), robot preset/URDF, cameras summary.
   - **Specify/select (required):** User picks scene, robot, props, cameras before or during a run:
     - Scene mode: `stock` \| `byo_mesh` \| `scene_spec`
     - Robot mode: `stock_franka` \| `preset:<name>` \| `byo`
     - Camera mode: `stock` \| `custom` (+ URI or inline edit)
     - Props: checkbox list of static/manipuland objects from SceneSpec
   - Selection persists in agent session → `POST /api/sim-assets/selection`
3. **Backend JSON API:**
   - `GET /api/sim-assets` → `{ scene_spec, robot_spec, assets_manifest, selection, resolved_uris }`
   - `GET /api/sim-assets/catalog` → browsable entries under allowed S3 prefixes (names + URIs only; no secrets)
   - `POST /api/sim-assets/selection` → body `{ scene_spec_uri?, assets_uri?, robot_spec_uri?, cameras_uri?, robot_preset?, sim_backend?, props?: string[] }` → validates via `scene_assets.parse_*` / `robot_assets.parse_*` (no GPU); uploads drafts to scratch prefix when user edits inline JSON
   - `GET /api/sim-assets/selection` → current selection for UI + agent LLM context
4. Reuse `npa.workflows.sim2real_assets` (`run_assets_stage`, `merge_standalone_cameras_uri`, `resolve_stage_cameras`) where practical.

### How the agent invokes workbench with selected assets

The agent **does not** call Genesis/Isaac APIs directly for asset binding. It passes URIs through workflow config / Sim2Real submit env:

**Path A — Sim2Real staged run (primary):**

```yaml
# Agent backend builds env block from GET /api/sim-assets/selection
NPA_SIM2REAL_SCENE_SPEC_URI: "{{ selection.scene_spec_uri }}"
NPA_SIM2REAL_ASSETS_URI: "{{ selection.assets_uri }}"
NPA_SIM2REAL_CAMERAS_URI: "{{ selection.cameras_uri }}"
NPA_SIM2REAL_ROBOT_SPEC_URI: "{{ selection.robot_spec_uri }}"
NPA_SIM2REAL_ROBOT_PRESET: "{{ selection.robot_preset }}"
NPA_SIM2REAL_SIM_BACKEND: "{{ selection.sim_backend | default('isaac') }}"
```

Submit via existing runbook / `npa workflows sim2real` CLI (see `npa/workflows/workbench/sim2real/runbook.yaml`). Stage 2 materializes consumed specs; downstream envgen + rollouts inherit cameras and embodiment.

**Path B — npa.workflow toolRef (agent tool catalog):**

Agent composes or selects a workflow spec with `config` keys matching the same URIs, then `npa workbench workflow run-spec`. Relevant toolRefs: `workbench.sim2real_envgen.raw_shard`, `workbench.sim2real.policy_rollouts`, `workbench.sim2real.heldout_eval`, `workbench.sim2real.finalize` (`npa/src/npa/orchestration/npa_workflow/catalog.py`).

**Path C — Single-tool smoke (Genesis / Isaac):**

For quick sim without full loop: `npa workbench genesis …` or `npa workbench isaac-lab train|eval` with `--input-path` pointing at consumed scene artifacts. Asset URIs must still be resolved through Stage 2 or equivalent JSON on S3.

**Agent chat contract:** Before submit, agent confirms selection summary (scene, robot, cameras, backend). After submit, agent links `sim_viz_url` + polls `/api/sim-viz/status`.

### UI layout (minimum)

```
┌──────────────────────────────────────────────────────────────┐
│ NPA Agent chat + workbench actions                           │
├───────────────────────┬──────────────────────────────────────┤
│ Sim Assets            │ Rerun (live / latest .rrd)           │
│ ├─ Browse catalog     │ /rerun/ iframe                       │
│ ├─ Scene / Robot /    │ stage badge + run-id overlay         │
│ │  Props selectors    │                                      │
│ └─ Cameras inspector  │                                      │
│    (list + frustum +  │                                      │
│     feed preview)     │                                      │
└───────────────────────┴──────────────────────────────────────┘
```

Tabs acceptable if split layout is harder; all three surfaces (assets, cameras, rerun) must be reachable without leaving the authenticated UI.

---

## 4. CLI + verify-live gates

### CLI surface (extend existing `npa agent`)

| Command | New fields |
| --- | --- |
| `deploy` / `status --json` | `sim_viz_url`, `sim_assets_url`, `cameras_api_url` (may share same host, different paths) |
| `verify-live` | All checks below must pass |

### verify-live checklist (complete gate)

Exit 0 **only** when **all** true:

1. VM non-localhost public IP in `us-central1`
2. UI `200` behind basic auth
3. **Sim viz:** `sim_viz_url` (Rerun iframe `/rerun/`) returns `200` behind basic auth
4. **Sim assets:** `GET {sim_assets_url}/api/sim-assets` returns `200` + JSON with `scene_spec` and `robot_spec` keys
5. **Cameras API:** `GET {sim_assets_url}/api/sim-assets/cameras` returns `200` + JSON with `cameras` array (≥1 entry for stock defaults)
6. **Asset selection path:** `POST {sim_assets_url}/api/sim-assets/selection` with stock preset body → `200`; `GET .../selection` reflects posted URIs; agent `POST /api/workflows/sim2real/submit` (or equivalent) accepts selection and returns `run_id` (mock in unit tests; live e2e may use dry-run / plan-only if full GPU submit is too heavy)
7. Agent toolRef catalog ≥19 entries (`GET /api/tools`)
8. `pytest npa/tests/cli/test_agent.py` pass
9. `pytest npa/tests/e2e/test_agent_live.py` pass (includes sim-viz, cameras, asset-selection HTTP checks)

### Tests

- Unit: mock HTTP + S3 in `npa/tests/cli/test_agent.py` — sim-viz status, cameras list, selection POST/GET, workflow payload builder
- E2E: `npa/tests/e2e/test_agent_live.py` — live HTTP checks with basic auth (no credential leaks; use env fixtures)

---

## 5. Unchanged base requirements (do not drop)

- VM public IP in `us-central1` (not localhost-only)
- HTTP basic auth at nginx edge (htpasswd pattern from sim2real rerun serve)
- Full workbench **toolRef catalog** (≥19 entries) via `npa.workflow` `TOOL_CATALOG`
- Real **Rerun web viewer** at `/rerun/` on agent VM (`npa-rerun-viewer` / legacy `npa-sim2real-rerun-viewer` image or `rerun-sdk>=0.32`; replace stub when implementing §1)
- `npa agent deploy|status|destroy|verify-live` registered in `npa/src/npa/cli/main.py`
- No credential leaks; use `redact_value` helpers
- Commit when tests pass; keep diff focused

---

## 6. Install note

If `npa agent` is missing from the venv entrypoint, run editable install:

```bash
cd npa && ../npa/.venv/bin/pip install -e .
```

Or set `PYTHONPATH=npa/src` in the loop success command.

---

## 7. Implementation priority (for loop iterations)

1. Editable install + real deploy so `verify-live` base gate passes
2. Replace rerun stub with static `.rrd` viewer + `/api/sim-viz/status` (v1 polling)
3. Sim assets panel + `GET/POST /api/sim-assets/selection` wired to stock defaults
4. Cameras inspector API + UI frustum list
5. Workflow submit with selection → Stage 2 URI env block
6. Optional gRPC proxy to cluster serve for live rollouts (v2); VM-co-located viewer remains primary
