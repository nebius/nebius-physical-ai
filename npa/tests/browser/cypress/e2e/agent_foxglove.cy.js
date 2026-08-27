/**
 * Embedded Foxglove viewer coverage for the NPA agent UI.
 *
 * The SDK under test is the real `@foxglove/embed` npm package (served from
 * /foxglove/sdk exactly as the agent VM serves it). The Foxglove application
 * itself is a licensed product that cannot run in CI, so the iframe points at a
 * protocol-accurate stand-in that implements the documented postMessage
 * handshake and records every command the UI sends.
 */

import { NON_STOCK_RUN_ID, resolveLiveAgentConfig } from "../support/e2e";
import { SceneUpdate as OfficialSceneUpdate } from "@foxglove/schemas/jsonschema";

// The SDK requires an absolute embed URL (`new URL(src)`), exactly like a real
// Foxglove deployment; the agent backend enforces the same rule.
const MOCK_EMBED_SRC = `${Cypress.config("baseUrl")}/mock-foxglove-app/`;
const MCAP_URL = "/foxglove/data/tok-session.mcap";

const RICH_TOPICS = {
  "/camera": "foxglove.CompressedImage",
  "/camera/side": "foxglove.CompressedImage",
  "/camera/workspace": "foxglove.CompressedImage",
  "/robot/diagnostic_scene": "foxglove.SceneUpdate",
  "/robot/diagnostic_pose": "foxglove.PoseInFrame",
  "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
  "/robot/diagnostic_joint_states": "foxglove.JointStates",
  "/actuators/commands": "npa.ActuatorCommands",
  "/run/state": "npa.RunState",
  "/metrics/execution": "npa.RunMetrics.execution",
  "/log": "foxglove.Log",
};

function richLayout() {
  const panel = (panelType, title, config) => ({
    type: "panel", panelType, title, config, version: 1,
  });
  const split = (direction, items) => ({
    type: "split",
    direction,
    items: items.map(([proportion, content]) => ({ proportion, content })),
  });
  const cameraPanel = (topic, label, index) => panel(
    "Image",
    `${label} camera — preserved source RGB`,
    {
      imageMode: { imageTopic: topic, imageSchemaName: "foxglove.CompressedImage" },
      synchronize: true,
      syncedTopics: { [topic]: true },
      npaCamera: { index, label, topic, sourceFidelity: "source-rgb-only" },
    },
  );
  const cameraTabs = {
    type: "tabs",
    selectedTabIndex: 0,
    tabs: [
      { title: "Primary (/camera)", content: cameraPanel("/camera", "Primary", 0) },
      { title: "Side (/camera/side)", content: cameraPanel("/camera/side", "Side", 1) },
      { title: "Workspace (/camera/workspace)", content: cameraPanel("/camera/workspace", "Workspace", 2) },
    ],
  };
  return {
    version: 1,
    content: split("column", [
      [0.72, split("row", [
        [0.58, panel("ThreeDee", "Robot motion and end-effector trajectory", {
          fixedFrame: "npa_action_space",
          topics: {
            "/robot/diagnostic_scene": { visible: true },
            "/robot/diagnostic_pose": { visible: true },
            "/robot/diagnostic_trajectory": { visible: true },
          },
        })],
        [0.42, cameraTabs],
      ])],
      [0.28, split("row", [
        [0.45, panel("Plot", "Execution performance", { paths: [{ value: "/metrics/execution.reward" }] })],
        [0.30, panel("StateTransitions", "Run phase", { paths: [{ value: "/run/state.phase" }] })],
        [0.25, panel("Log", "Run events", { topicToRender: "/log" })],
      ])],
    ]),
  };
}

function foxgloveConfig(overrides) {
  return Object.assign(
    {
      available: true,
      reason: "",
      enabled: true,
      sdk_url: "/foxglove/sdk/index.js",
      host_module_url: "/foxglove/app/npa-foxglove-host.js",
      sdk_version: "0.58.0",
      sdk_ready: true,
      embed_src: MOCK_EMBED_SRC,
      viewer_backend: "foxglove-sdk",
      viewer_backends: ["foxglove-sdk", "self-hosted"],
      self_hosted_ready: false,
      self_hosted_url: "",
      org_slug: "acme-robotics",
      color_scheme: "dark",
      layout_storage_key: "npa-agent-foxglove-robot-motion-v3",
      cloud_import_timeout_seconds: 300,
      layout: richLayout(),
      visualization: {
        contract: "npa.foxglove.robot-motion.v3",
        fixed_frame: "npa_action_space",
        fidelity: "Action-derived diagnostic schematic; not calibrated robot/world kinematics.",
        topics: RICH_TOPICS,
        checked: true,
      },
      live_url: "",
      data_source: { type: "remote-file", urls: [MCAP_URL] },
      run_id: "mock-run",
      artifact_run_ref: "npa1_mock_run",
      artifact_key: "mock-run/reports/session.mcap",
      artifact_uri: "s3://mock/mock-run/reports/session.mcap",
      project_id: "project-local",
      resource_bucket: "mock",
      bucket: "mock",
      resolved_prefix: "",
      artifact_sha256: "a".repeat(64),
      selected_artifact: {
        run_id: "mock-run",
        run_ref: "npa1_mock_run",
        key: "mock-run/reports/session.mcap",
        s3_uri: "s3://mock/mock-run/reports/session.mcap",
        resource_bucket: "mock",
        bucket: "mock",
        project_id: "project-local",
        resolved_prefix: "",
        sha256: "a".repeat(64),
      },
      recording_url: MCAP_URL,
      updated_at: "2026-07-30T00:00:00+00:00",
    },
    overrides || {}
  );
}

function stubFoxgloveApis(configOverrides) {
  const config = foxgloveConfig(configOverrides);
  cy.intercept("GET", "/api/foxglove/config", (request) => {
    request.reply({ statusCode: 200, body: config });
  }).as("foxgloveConfig");
  cy.intercept("GET", "/api/foxglove/status", (request) => {
    request.reply({ statusCode: 200, body: {
      available: config.available,
      reason: config.reason,
      sdk_version: config.sdk_version,
      embed_src: config.embed_src,
      org_slug: config.org_slug,
      viewer_backend: config.viewer_backend,
      self_hosted_ready: config.self_hosted_ready,
      self_hosted_url: config.self_hosted_url,
      foxglove_ready: Boolean(config.data_source),
      run_id: config.run_id,
      artifact_run_ref: config.artifact_run_ref,
      artifact_key: config.artifact_key,
      artifact_uri: config.artifact_uri,
      artifact_render: "foxglove",
      project_id: config.project_id,
      resource_bucket: config.resource_bucket,
      bucket: config.bucket,
      resolved_prefix: config.resolved_prefix,
      artifact_sha256: config.artifact_sha256,
      selected_artifact: config.selected_artifact,
      recording_url: config.recording_url,
      data_source_type: config.data_source ? config.data_source.type : "",
      data_source: config.data_source,
    }});
  }).as("foxgloveStatus");
  return config;
}

function mockAppFrame() {
  return cy.get("#viewerPaneFoxglove iframe", { timeout: 20000 });
}

// Re-query the iframe on every retry: chaining through contentDocument can
// detach the subject while the SDK is still wiring the frame up.
function expectMockAppState(state) {
  return mockAppFrame().should(($frame) => {
    const doc = $frame[0].contentDocument;
    expect(doc, "embedded app document").to.exist;
    const el = doc.querySelector("[data-testid=mock-foxglove-state]");
    expect(el, "embedded app state element").to.exist;
    expect(el.getAttribute("data-state"), "embedded viewer handshake state").to.eq(state);
  });
}

function assertFoxgloveControlsUnobstructed() {
  cy.get("#viewerPaneFoxglove iframe").then(($frame) => {
    const iframe = $frame[0].getBoundingClientRect();
    const controls = {
      left: iframe.left,
      right: iframe.right,
      top: iframe.bottom - Math.min(80, iframe.height),
      bottom: iframe.bottom,
    };
    const overlaps = (rect) =>
      Math.max(rect.left, controls.left) < Math.min(rect.right, controls.right) &&
      Math.max(rect.top, controls.top) < Math.min(rect.bottom, controls.bottom);
    for (const selector of ["#foxgloveStatus", "#statusBar", "#chatDrawerToggle"]) {
      const element = Cypress.$(selector)[0];
      expect(element, `${selector} exists for geometry proof`).to.exist;
      const style = getComputedStyle(element);
      if (style.display !== "none" && style.visibility !== "hidden") {
        expect(overlaps(element.getBoundingClientRect()), `${selector} clears playback controls`)
          .to.eq(false);
      }
    }
    expect(getComputedStyle(Cypress.$("#foxgloveStatus")[0]).position).to.eq("static");
    expect(getComputedStyle(Cypress.$("#statusBar")[0]).position).to.eq("static");
  });
}

function assertSingleFoxgloveWebAction(options = {}) {
  const enabled = options.enabled !== false;
  const visible = options.visible !== false;
  const action = cy.get('[data-testid="open-foxglove-web"]')
    .should("have.length", 1)
    .and("have.prop", "tagName", "BUTTON")
    .and("have.attr", "type", "button")
    .and("have.attr", "aria-describedby", "foxgloveExportNote");
  action.should(visible ? "be.visible" : "not.be.visible");
  cy.get("#foxgloveOpenWeb").should("have.text", "Open in Foxglove");
  cy.get("#foxgloveOpenWeb").should(enabled ? "be.enabled" : "be.disabled");
  cy.get("#foxgloveOpenDesktop").should("not.exist");
  cy.contains(/Foxglove Desktop/i).should("not.exist");
  cy.get("#renderModeLichtblick").should("have.length", 1);
}

function activateFoxglovePane() {
  cy.get("#renderModeFoxglove").click();
  cy.get("#viewerPaneFoxglove")
    .should("have.attr", "aria-hidden", "false")
    .and("have.class", "is-active-viewer");
}

function foxgloveExportResponse(runId, overrides = {}) {
  const canonicalHash = String(overrides.sha256 || "a".repeat(64));
  const layoutId = String(overrides.layoutId === undefined ? "layout_rich_v1" : overrides.layoutId);
  const seek = "2026-08-10T12:00:00.250000000Z";
  const recordingUrl = `https://agent.example/foxglove/data/${runId}.mcap`;
  const params = new URLSearchParams();
  params.append("ds", "remote-file");
  params.append("ds.url", recordingUrl);
  params.append("time", seek);
  if (layoutId) params.set("layoutId", layoutId);
  const webUrl = `https://app.foxglove.dev/~/view?${params.toString()}`;
  return {
    ok: true,
    converted: Boolean(overrides.converted),
    run_id: runId,
    artifact_key: `${runId}/reports/sim2real.mcap`,
    sim_viz: {
      run_id: runId,
      artifact_key: `${runId}/reports/sim2real.mcap`,
      artifact_render: "mcap",
      canonical_mcap_s3_uri: `s3://mock/${runId}/reports/sim2real.mcap`,
      canonical_mcap_sha256: canonicalHash,
      canonical_mcap_source: overrides.converted ? "converted" : "native-reused",
      canonical_mcap_provenance: {
        start_time_ns: 1786363200000000000,
        end_time_ns: 1786363209937500000,
        rich_run: {
          engine_provenance: {
            engine: "NVIDIA Isaac Sim + Isaac Lab via LeIsaac",
            task: "LeIsaac-SO101-LiftCube-v0",
          },
          timestamp_semantics: "episode-relative at source-recorded 16 FPS",
          trajectory_semantics: "real observation state; not world geometry",
          limitations: ["No calibrated depth or world-frame object pose."],
        },
      },
      transport_state: "published-local-cache",
      lichtblick_ready: true,
    },
    export: {
      available: true,
      recording_url: recordingUrl,
      download_url: recordingUrl,
      web_url: webUrl,
      data_source: "remote-file",
      web_open_mode: "remote-file",
      layout_id: layoutId,
      layout: layoutId
        ? { available: true, layout_id: layoutId, reused: true, reason: "" }
        : { available: false, layout_id: "", reused: false, reason: "Layout API unavailable." },
      layout_note: layoutId
        ? "Foxglove Web was opened with the canonical shared NPA layout."
        : "Layout API unavailable; select a saved layout after signing in.",
      canonical_s3_uri: `s3://mock/${runId}/reports/sim2real.mcap`,
      sha256: canonicalHash,
      provenance: {
        start_time_ns: 1786363200000000000,
        end_time_ns: 1786363209937500000,
        schemas: RICH_TOPICS,
        numeric_paths: {
          "/metrics/execution": ["reward", "object_lift_m", "object_goal_distance_m"],
          "/run/state": ["progress", "step"],
        },
      },
    },
  };
}

function exactArtifactExportResponse(runId, runRef, key, s3Uri) {
  const response = foxgloveExportResponse(runId);
  const selectedArtifact = {
    run_id: runId,
    run_ref: runRef,
    key,
    s3_uri: s3Uri,
    bucket: "mock",
    resource_bucket: "mock",
    project_id: "project-local",
    resolved_prefix: "",
    sha256: "b".repeat(64),
    size_bytes: 16384,
    recording_url: response.export.recording_url,
  };
  response.artifact_key = key;
  response.selected_artifact = selectedArtifact;
  response.sim_viz.artifact_key = key;
  response.sim_viz.artifact_run_ref = runRef;
  response.export.selected_artifact = selectedArtifact;
  response.export.sha256 = selectedArtifact.sha256;
  return response;
}

function applyExactArtifactConfig(config, response) {
  const selected = response.selected_artifact;
  Object.assign(config, {
    run_id: selected.run_id,
    artifact_run_ref: selected.run_ref,
    artifact_key: selected.key,
    artifact_uri: selected.s3_uri,
    project_id: selected.project_id,
    resource_bucket: selected.resource_bucket,
    bucket: selected.bucket,
    resolved_prefix: selected.resolved_prefix,
    artifact_sha256: selected.sha256,
    selected_artifact: { ...selected },
    recording_url: response.export.recording_url,
    data_source: { type: "remote-file", urls: [response.export.recording_url] },
  });
  return config;
}

function assertRichOfficialUrl(webUrl, expectedRecordingUrl = null) {
  const parsed = new URL(webUrl);
  expect(parsed.origin).to.eq("https://app.foxglove.dev");
  expect(parsed.pathname).to.eq("/~/view");
  expect(parsed.searchParams.get("ds")).to.eq("remote-file");
  expect(parsed.searchParams.getAll("ds.url")).to.have.length(1);
  if (expectedRecordingUrl) {
    expect(parsed.searchParams.get("ds.url")).to.eq(expectedRecordingUrl);
  }
  expect(parsed.searchParams.get("layoutId")).to.eq("layout_rich_v1");
  expect(parsed.searchParams.get("time")).to.eq("2026-08-10T12:00:00.250000000Z");
  expect(parsed.searchParams.get("openIn")).to.eq(null);
  expect(parsed.searchParams.get("ds.recordingId")).to.eq(null);
  expect(webUrl).not.to.match(/token|password|authorization/i);
}

describe("NPA agent UI — embedded Foxglove viewer", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  });

  it("pins the authoritative Foxglove SceneUpdate array contract", () => {
    expect(OfficialSceneUpdate.title).to.eq("foxglove.SceneUpdate");
    expect(OfficialSceneUpdate.properties.deletions.items.title).to.eq(
      "foxglove.SceneEntityDeletion",
    );
    const entity = OfficialSceneUpdate.properties.entities.items.properties;
    expect({
      metadata: entity.metadata.items.title,
      arrows: entity.arrows.items.title,
      cubes: entity.cubes.items.title,
      spheres: entity.spheres.items.title,
      cylinders: entity.cylinders.items.title,
      lines: entity.lines.items.title,
      triangles: entity.triangles.items.title,
      texts: entity.texts.items.title,
      models: entity.models.items.title,
    }).to.deep.eq({
      metadata: "foxglove.KeyValuePair",
      arrows: "foxglove.ArrowPrimitive",
      cubes: "foxglove.CubePrimitive",
      spheres: "foxglove.SpherePrimitive",
      cylinders: "foxglove.CylinderPrimitive",
      lines: "foxglove.LinePrimitive",
      triangles: "foxglove.TriangleListPrimitive",
      texts: "foxglove.TextPrimitive",
      models: "foxglove.ModelPrimitive",
    });
  });

  [
    { name: "desktop", width: 1440, height: 1000 },
    { name: "mobile-safe", width: 390, height: 844 },
  ].forEach((viewport) => {
    it(`keeps the exact visible viewer order and nonzero panes at ${viewport.name} size`, () => {
      cy.viewport(viewport.width, viewport.height);
      stubFoxgloveApis();
      cy.get("#tabRerun").should("have.text", "View").click();
      cy.get(".render-mode-tabs .render-mode-tab").then(($tabs) => {
        expect([...$tabs].map((tab) => tab.textContent.trim())).to.deep.eq([
          "View", "Foxglove", "Lichtblick", "Video", "Image", "Data",
        ]);
      });
      [
        ["View", "#viewerPaneRerun"],
        ["Foxglove", "#viewerPaneFoxglove"],
        ["Lichtblick", "#viewerPaneLichtblick"],
      ].forEach(([label, pane]) => {
        cy.contains(".render-mode-tabs .render-mode-tab", new RegExp(`^${label}$`))
          .scrollIntoView()
          .click();
        cy.get(pane)
          .should("have.attr", "aria-hidden", "false")
          .and("have.class", "is-active-viewer")
          .should(($pane) => {
            const rect = $pane[0].getBoundingClientRect();
            expect(rect.width, `${label} pane width`).to.be.greaterThan(0);
            expect(rect.height, `${label} pane height`).to.be.greaterThan(0);
          });
      });
      activateFoxglovePane();
      cy.get("#foxgloveOpenWeb")
        .scrollIntoView()
        .should("be.visible")
        .and("have.text", "Open in Foxglove");
    });
  });

  [
    { name: "desktop", width: 1440, height: 1000 },
    { name: "mobile-safe", width: 390, height: 844 },
  ].forEach((viewport) => {
    it(`opens the exact discovered MCAP card in Foxglove at ${viewport.name} size`, () => {
      cy.viewport(viewport.width, viewport.height);
      const config = stubFoxgloveApis();
      const runRef = "npa1_mock_non_stock";
      const key = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
      const s3Uri = `s3://mock/${key}`;
      const exported = exactArtifactExportResponse(
        NON_STOCK_RUN_ID,
        runRef,
        key,
        s3Uri,
      );
      cy.intercept("POST", "/api/foxglove/export", (request) => {
        expect(request.body).to.deep.eq({
          run_id: NON_STOCK_RUN_ID,
          run_ref: runRef,
          key,
          resource_bucket: "mock",
          project_id: "project-local",
          resolved_prefix: "",
          s3_uri: s3Uri,
        });
        applyExactArtifactConfig(config, exported);
        request.reply({ delay: 3000, statusCode: 200, body: exported });
      }).as("exactArtifactExport");
      cy.window().then((win) => {
        cy.stub(win, "open").as(`exactArtifactWindowOpen-${viewport.name}`);
      });

      cy.get("#tabRerun").click();
      cy.location("href").as(`exactArtifactTopUrl-${viewport.name}`);
      cy.get("#artifactRefreshRuns").click();
      cy.wait("@artifactRuns");
      cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
      cy.wait("@nonStockArtifactList");
      cy.get(`.artifact-card:has([data-key="${key}"])`)
        .as("exactArtifactCard")
        .scrollIntoView()
        .should("be.visible");
      cy.get("@exactArtifactCard").find(".artifact-card-actions .btn").then(($buttons) => {
        expect([...$buttons].map((button) => button.textContent.trim())).to.deep.eq([
          "View in Foxglove",
          "View in Lichtblick",
          "Download",
        ]);
      });
      cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
        .should("be.visible")
        .and("be.enabled")
        .and("have.attr", "aria-disabled", "false")
        .click();
      cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
        .should("have.attr", "aria-busy", "true")
        .and("be.disabled");
      cy.get("#renderModeFoxglove")
        .should("have.attr", "aria-selected", "true")
        .and("have.class", "is-active");
      cy.get("#viewerPaneFoxglove")
        .should("have.attr", "aria-hidden", "false")
        .should(($pane) => {
          const rect = $pane[0].getBoundingClientRect();
          expect(rect.width, "embedded Foxglove pane width").to.be.greaterThan(0);
          expect(rect.height, "embedded Foxglove pane height").to.be.greaterThan(0);
        });
      // The SDK iframe mounts and is visible while the deliberately delayed
      // backend request is still in flight; playback binding happens later.
      mockAppFrame()
        .should("be.visible")
        .and("have.attr", "src")
        .and("include", MOCK_EMBED_SRC);
      cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
        .should("have.attr", "aria-busy", "true")
        .and("be.disabled");
      cy.wait("@exactArtifactExport");
      expectMockAppState("ready");
      mockAppFrame().should(($frame) => {
        const rect = $frame[0].getBoundingClientRect();
        expect(rect.width, "embedded Foxglove iframe width").to.be.greaterThan(0);
        expect(rect.height, "embedded Foxglove iframe height").to.be.greaterThan(0);
        const messages = $frame[0].contentWindow.__mockFoxgloveReceived || [];
        const source = messages.find((message) => message?.type === "set-data-source");
        expect(source, "exact setDataSource command").to.exist;
        expect(source.payload).to.deep.eq({
          type: "remote-file",
          urls: [exported.export.recording_url],
        });
        const layout = messages.find((message) => message?.type === "select-layout");
        expect(layout, "canonical selectLayout command").to.exist;
        expect(layout.payload.storageKey).to.eq("npa-agent-foxglove-robot-motion-v3");
        expect(layout.payload.layout).to.deep.eq(config.layout);
        expect(layout.payload.force).to.eq(undefined);
      });
      cy.get(`@exactArtifactWindowOpen-${viewport.name}`).should("not.have.been.called");
      cy.get(`@exactArtifactTopUrl-${viewport.name}`).then((topUrl) => {
        cy.location("href").should("eq", topUrl);
      });
      cy.get("#viewerPaneFoxglove")
        .should("have.attr", "data-run-id", NON_STOCK_RUN_ID)
        .and("have.attr", "data-run-ref", runRef)
        .and("have.attr", "data-artifact-key", key)
        .and("have.attr", "data-project-id", "project-local")
        .and("have.attr", "data-resource-bucket", "mock")
        .and("have.attr", "data-resolved-prefix", "")
        .and("have.attr", "data-sha256", exported.export.sha256)
        .and("have.attr", "data-recording-url", exported.export.recording_url);
      cy.get("#foxgloveHost")
        .should("have.attr", "data-sdk-ready", "true")
        .and("have.attr", "data-set-data-source-count", "1")
        .and("have.attr", "data-data-source-type", "remote-file")
        .and("have.attr", "data-data-source-url", exported.export.recording_url)
        .and(
          "have.attr",
          "data-layout-storage-key",
          "npa-agent-foxglove-robot-motion-v3",
        );
      cy.get("#foxgloveExportNote")
        .should("have.attr", "data-state", "success")
        .and("contain.text", "Embedded source is the selected MCAP");
      cy.get("#foxgloveOpenWeb")
        .should("have.text", "Open in Foxglove")
        .and("be.visible");
      assertFoxgloveControlsUnobstructed();
      cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
        .should("have.attr", "aria-busy", "false")
        .and("be.enabled");
      mockAppFrame().its("0.contentWindow.__mockFoxgloveTopicErrors").should(
        "deep.equal",
        [],
      );
    });
  });

  it("reuses the SDK iframe, data source, and layout for an unchanged exact MCAP", () => {
    const config = stubFoxgloveApis();
    const runRef = "npa1_mock_non_stock";
    const key = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
    const s3Uri = `s3://mock/${key}`;
    const exported = exactArtifactExportResponse(NON_STOCK_RUN_ID, runRef, key, s3Uri);
    let exports = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      exports += 1;
      applyExactArtifactConfig(config, exported);
      exported.cache_reused = exports > 1;
      exported.foxglove = config;
      request.reply({ statusCode: 200, body: exported });
    }).as("reusedArtifactExport");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");

    let firstFrame;
    const clickExactArtifact = () => {
      cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
        .scrollIntoView()
        .should("be.enabled")
        .click();
      cy.wait("@reusedArtifactExport");
      expectMockAppState("ready");
    };
    clickExactArtifact();
    mockAppFrame().then(($frame) => {
      firstFrame = $frame[0];
    });
    cy.get("#foxgloveHost")
      .should("have.attr", "data-set-data-source-count", "1")
      .and("have.attr", "data-layout-storage-key", "npa-agent-foxglove-robot-motion-v3")
      .and("have.attr", "data-layout-select-count", "1");

    clickExactArtifact();
    mockAppFrame().should(($frame) => {
      expect($frame[0], "same official SDK iframe").to.equal(firstFrame);
    });
    cy.get("#foxgloveHost")
      .should("have.attr", "data-set-data-source-count", "1")
      .and("have.attr", "data-layout-select-count", "1")
      .and("not.have.class", "is-switching");
    cy.then(() => expect(exports, "one request per deliberate click").to.eq(2));
    assertFoxgloveControlsUnobstructed();
  });

  it("keeps the exact selected MCAP open when its card rerenders during preparation", () => {
    const config = stubFoxgloveApis();
    const runRef = "npa1_mock_non_stock";
    const key = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
    const s3Uri = `s3://mock/${key}`;
    const exported = exactArtifactExportResponse(
      NON_STOCK_RUN_ID,
      runRef,
      key,
      s3Uri,
    );
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      applyExactArtifactConfig(config, exported);
      request.reply({ delay: 900, statusCode: 200, body: exported });
    }).as("rerenderedCardExport");
    cy.window().then((win) => {
      cy.stub(win, "open").as("rerenderedCardWindowOpen");
    });

    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
      .should("be.enabled")
      .click();
    // A normal same-run inventory refresh replaces the card DOM node while
    // the immutable run/ref/key export remains in flight.
    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@nonStockArtifactList");
    cy.wait("@rerenderedCardExport");
    expectMockAppState("ready");
    cy.get("#viewerPaneFoxglove")
      .should("have.attr", "aria-hidden", "false")
      .and("have.attr", "data-run-id", NON_STOCK_RUN_ID)
      .and("have.attr", "data-artifact-key", key)
      .and("have.attr", "data-sha256", exported.export.sha256);
    cy.get("@rerenderedCardWindowOpen").should("not.have.been.called");
    cy.get("#foxgloveExportNote")
      .should("have.attr", "data-state", "success")
      .and("contain.text", "Embedded source is the selected MCAP");
  });

  it("shows an actionable embedded preparation error and retries the exact card", () => {
    const config = stubFoxgloveApis();
    const runRef = "npa1_mock_non_stock";
    const key = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
    const s3Uri = `s3://mock/${key}`;
    const exported = exactArtifactExportResponse(NON_STOCK_RUN_ID, runRef, key, s3Uri);
    let exports = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      exports += 1;
      if (exports === 1) {
        request.reply({ statusCode: 503, body: { detail: "selected MCAP transport unavailable" } });
        return;
      }
      applyExactArtifactConfig(config, exported);
      request.reply({ statusCode: 200, body: exported });
    }).as("retryArtifactExport");
    cy.window().then((win) => cy.stub(win, "open").as("retryArtifactWindowOpen"));
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
      .scrollIntoView()
      .should("be.enabled")
      .click();
    cy.wait("@retryArtifactExport");
    cy.get("#foxgloveStatus")
      .should("have.class", "is-error")
      .and("contain.text", "selected MCAP transport unavailable");
    cy.get("#foxgloveMessage")
      .should("not.have.attr", "hidden");
    cy.get("#foxgloveMessage")
      .should("contain.text", "Could not load this MCAP")
      .and("contain.text", "verify that the object is a valid MCAP");
    cy.get("#foxgloveArtifactRetry").should("be.visible").click();
    cy.wait("@retryArtifactExport");
    expectMockAppState("ready");
    cy.get("#viewerPaneFoxglove")
      .should("have.attr", "data-run-id", NON_STOCK_RUN_ID)
      .and("have.attr", "data-artifact-key", key)
      .and("have.attr", "data-sha256", exported.export.sha256);
    cy.get("#foxgloveMessage").should("have.attr", "hidden");
    cy.get("@retryArtifactWindowOpen").should("not.have.been.called");
    cy.then(() => expect(exports, "one failed request and one exact retry").to.eq(2));
  });

  it("lets a rapid second MCAP click win without binding the stale first source", () => {
    const config = stubFoxgloveApis();
    const runRef = "npa1_mock_non_stock";
    const canonicalKey = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
    const nativeKey = `${NON_STOCK_RUN_ID}/recordings/native-single-camera.mcap`;
    const canonical = exactArtifactExportResponse(
      NON_STOCK_RUN_ID,
      runRef,
      canonicalKey,
      `s3://mock/${canonicalKey}`,
    );
    const native = exactArtifactExportResponse(
      NON_STOCK_RUN_ID,
      runRef,
      nativeKey,
      `s3://mock/${nativeKey}`,
    );
    native.selected_artifact.sha256 = "c".repeat(64);
    native.export.selected_artifact.sha256 = native.selected_artifact.sha256;
    native.export.sha256 = native.selected_artifact.sha256;
    native.export.recording_url = "https://agent.example/foxglove/data/native-single-camera.mcap";
    native.export.download_url = native.export.recording_url;
    native.selected_artifact.recording_url = native.export.recording_url;
    let requests = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      requests += 1;
      const response = request.body.key === nativeKey ? native : canonical;
      applyExactArtifactConfig(config, response);
      if (response === native) {
        config.layout = {};
        config.layout_storage_key = "npa-agent-foxglove-source-default";
        config.visualization = { checked: false };
      }
      request.reply({
        delay: response === native ? 100 : 900,
        statusCode: 200,
        body: response,
      });
    }).as("rapidArtifactExport");
    cy.window().then((win) => cy.stub(win, "open").as("rapidArtifactWindowOpen"));

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${canonicalKey}"]`)
      .should("be.enabled")
      .click();
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${nativeKey}"]`)
      .should("be.enabled")
      .click();
    cy.wait("@rapidArtifactExport");
    expectMockAppState("ready");
    cy.get("#viewerPaneFoxglove")
      .should("have.attr", "data-artifact-key", nativeKey)
      .and("have.attr", "data-sha256", native.selected_artifact.sha256)
      .and("have.attr", "data-recording-url", native.export.recording_url);
    mockAppFrame().should(($frame) => {
      const messages = $frame[0].contentWindow.__mockFoxgloveReceived || [];
      const sourceCommands = messages.filter((message) => message?.type === "set-data-source");
      expect(sourceCommands, "only the latest generation binds a source").to.have.length(1);
      expect(sourceCommands[0].payload.urls).to.deep.eq([native.export.recording_url]);
      expect(
        messages.filter((message) => message?.type === "select-layout"),
        "generic native MCAP does not inherit the canonical layout",
      ).to.have.length(0);
    });
    cy.get("@rapidArtifactWindowOpen").should("not.have.been.called");
    cy.then(() => expect(requests, "both rapid exact selections reached the backend").to.eq(2));
  });

  it("uses the SDK command-ready contract when a hosted sign-in withholds ready", () => {
    const config = stubFoxgloveApis({ embed_src: `${MOCK_EMBED_SRC}?complete=0` });
    const runRef = "npa1_mock_non_stock";
    const key = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
    const s3Uri = `s3://mock/${key}`;
    const exported = exactArtifactExportResponse(NON_STOCK_RUN_ID, runRef, key, s3Uri);
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      applyExactArtifactConfig(config, exported);
      request.reply({ statusCode: 200, body: exported });
    }).as("signInArtifactExport");
    cy.window().then((win) => cy.stub(win, "open").as("signInArtifactWindowOpen"));

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
      .should("be.enabled")
      .click();
    cy.wait("@signInArtifactExport");
    cy.get("#foxgloveHost")
      .should("have.attr", "data-sdk-ready", "true")
      .and("have.attr", "data-set-data-source-count", "1")
      .and("have.attr", "data-data-source-url", exported.export.recording_url);
    mockAppFrame().its("0.contentDocument.body").should("contain.text", "usable-with-sign-in");
    cy.get("#foxgloveStatus").should("contain.text", "exact selected MCAP sent");
    cy.get("@signInArtifactWindowOpen").should("not.have.been.called");
  });

  it("queues the exact source without claiming ready when sign-in prevents a handshake", () => {
    const config = stubFoxgloveApis({ embed_src: `${MOCK_EMBED_SRC}?handshake=0` });
    const runRef = "npa1_mock_non_stock";
    const key = `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`;
    const s3Uri = `s3://mock/${key}`;
    const exported = exactArtifactExportResponse(NON_STOCK_RUN_ID, runRef, key, s3Uri);
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      applyExactArtifactConfig(config, exported);
      request.reply({ statusCode: 200, body: exported });
    }).as("unsignedArtifactExport");
    cy.window().then((win) => cy.stub(win, "open").as("unsignedArtifactWindowOpen"));

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
      .should("be.enabled")
      .click();
    cy.wait("@unsignedArtifactExport");
    cy.get("#foxgloveHost").should("not.have.attr", "data-sdk-ready");
    cy.get("#foxgloveHost")
      .should("have.attr", "data-set-data-source-count", "1")
      .and("have.attr", "data-data-source-url", exported.export.recording_url)
      .and("have.attr", "data-layout-storage-key", "npa-agent-foxglove-robot-motion-v3");
    mockAppFrame().its("0.contentDocument.body").should("contain.text", "sign-in-required");
    cy.get("#foxgloveStatus")
      .should("have.class", "is-warning")
      .and("contain.text", "queued in the official Foxglove SDK")
      .and("contain.text", "not marked ready");
    cy.get(`button[data-action="open-foxglove-artifact"][data-key="${key}"]`)
      .should("have.attr", "aria-busy", "false")
      .and("be.enabled");
    cy.get("#foxgloveOpenWeb").should("have.text", "Open in Foxglove").and("be.enabled");
    cy.get("@unsignedArtifactWindowOpen").should("not.have.been.called");
  });

  it("mounts the real @foxglove/embed SDK and completes the viewer handshake", () => {
    stubFoxgloveApis();
    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    // The SDK creates the iframe (not our UI): assert its shape.
    mockAppFrame()
      .should("have.attr", "src")
      .and("include", MOCK_EMBED_SRC);
    mockAppFrame().should("have.attr", "title", "Foxglove");
    mockAppFrame().should("have.attr", "allow").and("include", "keyboard-map");

    // The embedded app reports ready only after the documented handshake.
    expectMockAppState("ready");
    cy.get("#foxgloveStatus", { timeout: 20000 })
      .should("have.class", "is-ready")
      .and("contain.text", "Foxglove viewer ready");
    cy.get("#viewerPaneFoxglove").should("have.class", "is-active-viewer");
    cy.get("#foxgloveMessage").should("have.attr", "hidden");
  });

  it("sends the configured org, layout and MCAP data source to the viewer", () => {
    const config = stubFoxgloveApis();
    expect(config.visualization.topics).to.deep.eq(RICH_TOPICS);
    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    mockAppFrame()
      .its("0.contentWindow")
      .then((win) => {
        return Cypress.Promise.resolve()
          .then(() => new Cypress.Promise((resolve) => setTimeout(resolve, 500)))
          .then(() => win.__mockFoxgloveReceived || []);
      })
      .then((messages) => {
        const ack = messages.find((m) => m && m.type === "handshake-ack");
        expect(ack, "handshake-ack was sent by the SDK").to.exist;
        expect(ack.payload.orgSlug).to.eq("acme-robotics");
        expect(ack.payload.initialLayoutParams.storageKey).to.eq(
          "npa-agent-foxglove-robot-motion-v3",
        );
        expect(ack.payload.initialLayoutParams.force).to.eq(undefined);
        const layout = ack.payload.initialLayoutParams.layout;
        expect(layout.version).to.eq(1);
        const collectPanels = (node) => node.type === "panel"
          ? [node]
          : [
            ...(node.items || []).flatMap((item) => collectPanels(item.content)),
            ...(node.tabs || []).flatMap((item) => collectPanels(item.content)),
          ];
        const panels = collectPanels(layout.content);
        expect(panels.map((panel) => panel.panelType)).to.deep.eq([
          "ThreeDee", "Image", "Image", "Image", "Plot", "StateTransitions", "Log",
        ]);
        expect(
          panels.filter((panel) => panel.panelType === "Image")
            .map((panel) => panel.config.imageMode.imageTopic),
        ).to.deep.eq(["/camera", "/camera/side", "/camera/workspace"]);
        const cameraTabs = layout.content.items[0].content.items[1].content;
        expect(cameraTabs.type).to.eq("tabs");
        expect(cameraTabs.tabs.map((tab) => tab.title)).to.deep.eq([
          "Primary (/camera)", "Side (/camera/side)", "Workspace (/camera/workspace)",
        ]);
        expect(panels[0].config.fixedFrame).to.eq("npa_action_space");
        expect(Object.keys(panels[0].config.topics)).to.deep.eq([
          "/robot/diagnostic_scene",
          "/robot/diagnostic_pose",
          "/robot/diagnostic_trajectory",
        ]);
        expect(panels.map((panel) => panel.panelType)).not.to.include("UserScript");
        const source = ack.payload.initialDataSource;
        expect(source, "initial data source").to.exist;
        expect(source.type).to.eq("remote-file");
        // Data-source URLs must be absolute: the viewer fetches them cross-origin.
        expect(source.urls[0]).to.eq(`${window.location.origin}${MCAP_URL}`);
      });
    cy.get("#foxgloveVisualizationSummary")
      .should("be.visible")
      .and("contain.text", "robot + trajectory 3D")
      .and("contain.text", "not calibrated robot/world kinematics");
  });

  it("mounts promptly, then prepares and remounts an unchecked run with the rich layout", () => {
    const rich = foxgloveConfig();
    const unchecked = foxgloveConfig({
      layout_storage_key: "npa-agent-foxglove-robot-motion-v3-source-default",
      layout: {},
      visualization: { checked: false },
    });
    let configReads = 0;
    let prepareRequests = 0;
    cy.intercept("GET", "/api/foxglove/config", (request) => {
      configReads += 1;
      request.reply({ statusCode: 200, body: configReads === 1 ? unchecked : rich });
    }).as("preparationConfig");
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      prepareRequests += 1;
      expect(request.body).to.deep.eq({ run_id: "mock-run" });
      request.reply({
        statusCode: 200,
        body: foxgloveExportResponse("mock-run", { converted: true }),
      });
    }).as("automaticPreparation");

    cy.get("#tabRerun").click();
    cy.contains(".render-mode-tab", /^Foxglove$/).click();
    cy.get("#viewerPaneFoxglove iframe").should("be.visible");
    cy.wait("@automaticPreparation");
    expectMockAppState("ready");
    cy.get("#foxgloveVisualizationSummary")
      .should("contain.text", "robot + trajectory 3D")
      .and("have.attr", "data-state", "ready");
    cy.get("#foxgloveHost iframe").should(($iframe) => {
      const messages = $iframe[0].contentWindow.__mockFoxgloveReceived || [];
      const ack = messages.find((message) => message && message.type === "handshake-ack");
      expect(ack, "prepared rich-viewer handshake").to.exist;
      expect(ack.payload.initialLayoutParams.storageKey).to.eq(
        "npa-agent-foxglove-robot-motion-v3",
      );
      expect(ack.payload.initialLayoutParams.layout.version).to.eq(1);
    });
    cy.contains(".render-mode-tab", /^View$/).click();
    cy.contains(".render-mode-tab", /^Foxglove$/).click();
    cy.then(() => {
      expect(configReads, "unchecked then regenerated config").to.be.at.least(2);
      expect(prepareRequests, "one automatic canonical preparation").to.eq(1);
    });
  });

  it("downloads MCAP and opens the exact safe remote-file deep link", () => {
    const config = stubFoxgloveApis();
    const exportedResponse = foxgloveExportResponse("mock-run");
    const webUrl = exportedResponse.export.web_url;
    const canonicalHash = "a".repeat(64);
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      request.reply({ statusCode: 200, body: {
        ok: true,
        converted: false,
        run_id: "mock-run",
        artifact_key: config.artifact_key,
        sim_viz: {
          run_id: "mock-run",
          artifact_key: config.artifact_key,
          artifact_render: "mcap",
          canonical_mcap_s3_uri: "s3://mock/mock-run/reports/sim2real.mcap",
          canonical_mcap_sha256: canonicalHash,
          canonical_mcap_source: "native-reused",
          canonical_mcap_provenance: foxgloveExportResponse("mock-run").sim_viz
            .canonical_mcap_provenance,
          transport_state: "published-local-cache",
          lichtblick_ready: true,
          lichtblick_iframe_url: "/lichtblick/?ds=remote-file&ds.url=%2Flichtblick%2Frecordings%2Fsim2real.mcap",
        },
        export: request.body.open_web ? {
          available: true,
          recording_url: exportedResponse.export.recording_url,
          download_url: exportedResponse.export.download_url,
          web_url: webUrl,
          data_source: "remote-file",
          web_open_mode: "remote-file",
        } : {
          available: true,
          recording_url: `${window.location.origin}${MCAP_URL}`,
          download_url: `${window.location.origin}${MCAP_URL}`,
        },
      }});
    }).as("foxgloveExport");
    cy.window().then((win) => {
      cy.stub(win.HTMLAnchorElement.prototype, "click").as("downloadClick");
      const replace = cy.stub().as("foxgloveNavigate");
      cy.stub(win, "open").returns({ opener: null, location: { replace }, close: cy.stub() });
    });

    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");
    cy.get("#downloadMcap").click();
    cy.wait("@foxgloveExport");

    cy.get("@downloadClick").should("have.been.calledOnce");
    cy.get("#renderedDataSummary").should("contain.text", "Persistent S3 canonical");
    cy.get("#renderedDataSummary").should("contain.text", canonicalHash);
    cy.get("#renderedDataSummary").should("contain.text", "Ephemeral transport");
    cy.get("#foxgloveOpenWeb").should("have.text", "Open in Foxglove").click();
    cy.wait("@foxgloveExport").its("request.body.open_web").should("eq", true);
    cy.get("@foxgloveNavigate").should("have.been.calledWith", webUrl);
    cy.then(() => assertRichOfficialUrl(webUrl, exportedResponse.export.recording_url));
    cy.get("#foxgloveExportNote").should("contain.text", "remote-file source");
    cy.get("#renderedDataSummary")
      .should("contain.text", "NVIDIA Isaac Sim + Isaac Lab via LeIsaac")
      .and("contain.text", "not world geometry")
      .and("contain.text", "No calibrated depth");
    cy.get("#foxgloveOpenDesktop").should("not.exist");
    cy.contains("Open in Foxglove Desktop").should("not.exist");
    cy.get("#renderModeLichtblick").should("have.length", 1);
    cy.get("#openFullLichtblick").should("not.exist");
  });

  it("keeps exactly one common Foxglove Web action through discovery, export, refresh, tabs, and run switches", () => {
    stubFoxgloveApis();
    const exports = [];
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      const runId = String(request.body.run_id || "");
      const openWeb = request.body.open_web === true;
      exports.push({ runId, openWeb });
      if (openWeb) request.alias = "commonFoxgloveWebExport";
      request.reply({ statusCode: 200, body: foxgloveExportResponse(runId, {
        converted: runId === "mock-run",
        reused: runId !== "mock-run",
      }) });
    }).as("commonFoxgloveExport");
    cy.window().then((win) => {
      cy.stub(win, "open").callsFake(() => ({
        opener: null,
        location: { replace: cy.stub() },
        close: cy.stub(),
      }));
    });

    cy.get("#tabRerun").click();
    assertSingleFoxgloveWebAction({ visible: false });
    ["View", "Foxglove", "Lichtblick", "Video", "Image", "Data"].forEach((label) => {
      cy.contains(".render-mode-tabs .render-mode-tab", new RegExp(`^${label}$`)).click();
      assertSingleFoxgloveWebAction({ visible: label === "Foxglove" });
    });

    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    assertSingleFoxgloveWebAction({ visible: false });
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
    assertSingleFoxgloveWebAction({ visible: false });

    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@commonFoxgloveWebExport").its("request.body").should("deep.include", {
      run_id: "non-stock-customer-run",
      open_web: true,
    });
    cy.get("#foxgloveOpenWeb").should("have.attr", "aria-busy", "false");
    assertSingleFoxgloveWebAction();

    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    assertSingleFoxgloveWebAction();
    cy.get("#runIdInput").clear().type("mock-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "mock-run");
    assertSingleFoxgloveWebAction();
    cy.then(() => {
      expect(exports.filter((item) => item.openWeb).map((item) => item.runId)).to.deep.eq([
        "non-stock-customer-run",
      ]);
    });
  });

  it("uses conversion for a source-only run and opens each selected canonical recording", () => {
    stubFoxgloveApis();
    let exportCount = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      exportCount += 1;
      const runId = String(request.body.run_id || "");
      request.reply({ statusCode: 200, body: foxgloveExportResponse(runId, {
        converted: exportCount === 1,
      }) });
    }).as("pathExport");
    cy.window().then((win) => {
      const replace = cy.stub().as("pathNavigate");
      cy.stub(win, "open").callsFake(() => ({ opener: null, location: { replace }, close: cy.stub() }));
    });
    cy.get("#tabRerun").click();
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@pathExport").its("request.body.run_id").should("eq", "mock-run");
    cy.get("#foxgloveExportNote").should("contain.text", "remote-file source");
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@pathExport").its("request.body.run_id").should("eq", "non-stock-customer-run");
    cy.get("#foxgloveExportNote").should("contain.text", "remote-file source");
    cy.get("@pathNavigate").should("have.been.calledTwice");
    cy.get("@pathNavigate").then((navigate) => {
      navigate.getCalls().forEach((call) => assertRichOfficialUrl(call.args[0]));
    });
  });

  it("opens with documented rich layout/time parameters and degrades honestly when layouts are unavailable", () => {
    stubFoxgloveApis();
    let requestCount = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      requestCount += 1;
      request.reply({
        statusCode: 200,
        body: foxgloveExportResponse("mock-run", {
          reused: true,
          layoutId: requestCount === 1 ? "layout_rich_v1" : "",
        }),
      });
    }).as("layoutExport");
    cy.window().then((win) => {
      const replace = cy.stub().as("layoutNavigate");
      cy.stub(win, "open").callsFake(() => ({
        opener: null,
        location: { replace },
        close: cy.stub(),
      }));
    });
    cy.get("#tabRerun").click();
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@layoutExport").then(({ response }) => {
      assertRichOfficialUrl(
        response.body.export.web_url,
        "https://agent.example/foxglove/data/mock-run.mcap",
      );
    });
    cy.get("@layoutNavigate").should("have.been.calledOnce");
    cy.get("#foxgloveOpenWeb").should("not.be.disabled").click();
    cy.wait("@layoutExport").then(({ response }) => {
      const parsed = new URL(response.body.export.web_url);
      expect(parsed.searchParams.get("layoutId")).to.eq(null);
      expect(parsed.searchParams.get("time")).to.eq("2026-08-10T12:00:00.250000000Z");
      expect(response.body.export.layout_id).to.eq("");
    });
    cy.get("@layoutNavigate").should("have.been.calledTwice");
  });

  it("shows a disabled loading state and suppresses double-click export", () => {
    stubFoxgloveApis();
    let requests = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      requests += 1;
      request.reply({
        delay: 700,
        statusCode: 200,
        body: foxgloveExportResponse(String(request.body.run_id || "mock-run"), { reused: true }),
      });
    }).as("slowExport");
    cy.window().then((win) => {
      cy.stub(win, "open").returns({ opener: null, location: { replace: cy.stub() }, close: cy.stub() });
    });
    cy.get("#tabRerun").click();
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.get("#foxgloveOpenWeb")
      .should("be.disabled")
      .and("have.attr", "aria-busy", "true")
      .and("contain.text", "Opening Foxglove");
    cy.get("#foxgloveOpenWeb").click({ force: true });
    cy.wait("@slowExport");
    cy.get("#foxgloveOpenWeb").should("be.enabled").and("have.attr", "aria-busy", "false");
    cy.then(() => expect(requests, "one export request").to.eq(1));
  });

  it("does not abandon a valid slow canonical MCAP export", () => {
    stubFoxgloveApis();
    const exported = foxgloveExportResponse("mock-run", { reused: true });
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      request.reply({
        delay: 13000,
        statusCode: 200,
        body: exported,
      });
    }).as("slowMcapExport");
    cy.window().then((win) => {
      const replace = cy.stub().as("slowCloudNavigate");
      cy.stub(win, "open").returns({
        opener: null,
        location: { replace },
        close: cy.stub(),
      });
    });

    cy.get("#tabRerun").click();
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@slowMcapExport", { timeout: 20000 });
    cy.get("@slowCloudNavigate").should(
      "have.been.calledOnceWith",
      exported.export.web_url,
    );
    cy.get("#foxgloveExportNote").should(
      "contain.text",
      "remote-file source",
    );
  });

  it("bounds Cloud import requests beyond the server deadline without capping local conversion", () => {
    cy.window().then((win) => {
      const requestTimeoutMs = win.__NPA_AGENT_TEST__.requestTimeoutMs;
      expect(requestTimeoutMs("/api/foxglove/export", {
        body: JSON.stringify({ cloud_import: true }),
      })).to.eq(360000);
      expect(requestTimeoutMs("/api/foxglove/export", {
        body: JSON.stringify({ cloud_import: true }),
      }, { cloud_import_timeout_seconds: 425.5 })).to.eq(485500);
      expect(requestTimeoutMs("/api/foxglove/export", {
        body: JSON.stringify({ run_id: "mock-run" }),
      })).to.eq(0);
      expect(requestTimeoutMs("/api/foxglove/convert-run", {})).to.eq(12000);
    });
  });

  it("keeps the action stable and actionable after backend and popup failures", () => {
    stubFoxgloveApis();
    let exportRequests = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      exportRequests += 1;
      request.reply({ statusCode: 503, body: { detail: "Foxglove quota exceeded; free space or choose another project" } });
    }).as("failedExport");
    cy.window().then((win) => {
      cy.stub(win, "open").onFirstCall().returns({ opener: null, location: { replace: cy.stub() }, close: cy.stub() }).onSecondCall().returns(null);
    });
    cy.get("#tabRerun").click();
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@failedExport");
    assertSingleFoxgloveWebAction();
    cy.get("#foxgloveExportNote").should("contain.text", "quota exceeded").and("have.attr", "data-state", "error");
    cy.get("#foxgloveOpenWeb").click();
    cy.get("#foxgloveExportNote").should("contain.text", "blocked").and("contain.text", "Allow popups");
    cy.then(() => expect(exportRequests, "popup-blocked click starts no upload").to.eq(1));
    assertSingleFoxgloveWebAction();
  });

  it("discards a stale export response after a run switch without opening or overwriting the current run", () => {
    stubFoxgloveApis();
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      request.reply({
        delay: 900,
        statusCode: 200,
        body: foxgloveExportResponse(String(request.body.run_id || "mock-run"), { reused: true }),
      });
    }).as("staleExport");
    cy.window().then((win) => {
      const replace = cy.stub().as("staleNavigate");
      const close = cy.stub().as("stalePopupClose");
      cy.stub(win, "open").returns({ opener: null, location: { replace }, close });
    });
    cy.get("#tabRerun").click();
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").click();
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
    activateFoxglovePane();
    cy.get("#foxgloveOpenWeb").should("be.enabled").and("have.attr", "aria-busy", "false");
    cy.wait("@staleExport");
    cy.get("@staleNavigate").should("not.have.been.called");
    cy.get("@stalePopupClose").should("have.been.called");
    cy.get("#foxgloveExportNote").should("contain.text", "active run changed");
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
    assertSingleFoxgloveWebAction();
  });

  it("disables but does not hide the action when there is no active exportable run", () => {
    cy.intercept("GET", "/api/session", {
      statusCode: 200,
      body: {
        selection: {},
        sim_viz: { run_id: "", active_run_id: "", stage: "idle", rerun_ready: false },
        latest_submit: {},
        camera_selection: ["workspace"],
        chat_history: [],
      },
    }).as("emptySession");
    cy.intercept("GET", "/api/sim-viz/status*", {
      statusCode: 200,
      body: { run_id: "", active_run_id: "", stage: "idle", rerun_ready: false },
    }).as("emptySimViz");
    cy.visit("/");
    cy.wait("@emptySession");
    cy.get("#tabRerun").click();
    activateFoxglovePane();
    assertSingleFoxgloveWebAction({ enabled: false });
    cy.get("#foxgloveOpenWeb").should("have.attr", "aria-busy", "false");
    cy.get("#foxgloveExportNote").should("contain.text", "Load a run").and("have.attr", "data-state", "idle");
  });

  it("maps both live Cypress env naming styles and rejects partial intent", () => {
    expect(resolveLiveAgentConfig({
      agentBaseUrl: "https://agent.example",
      agentUser: "user",
      agentPassword: "pass",
    })).to.deep.eq({ baseUrl: "https://agent.example", username: "user", password: "pass" });
    expect(resolveLiveAgentConfig({
      NPA_AGENT_BASE_URL: "https://agent.example",
      NPA_AGENT_USER: "user",
      NPA_AGENT_PASSWORD: "pass",
    })).to.deep.eq({ baseUrl: "https://agent.example", username: "user", password: "pass" });
    expect(() => resolveLiveAgentConfig({ agentBaseUrl: "https://agent.example" })).to.throw(
      "configuration is incomplete"
    );
  });

  it("keeps the Rerun viewer mounted while Foxglove is active", () => {
    stubFoxgloveApis();
    cy.get("#tabRerun").click();
    cy.get("#rerunFrame").should("have.attr", "src");
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    cy.get("#viewerPaneRerun").should("have.class", "is-inactive-viewer");
    // Rerun's wasm iframe must survive the switch (no remount cost on return).
    cy.get("#rerunFrame").should("have.attr", "src");

    cy.get("#renderModeRerun").click();
    cy.get("#viewerPaneRerun").should("have.class", "is-active-viewer");
    cy.get("#viewerPaneFoxglove").should("have.class", "is-inactive-viewer");
    // The Foxglove iframe is kept alive too, so re-entry is instant.
    cy.get("#viewerPaneFoxglove iframe").should("exist");
  });

  it("surfaces viewer errors instead of silently showing an empty pane", () => {
    stubFoxgloveApis({ embed_src: `${MOCK_EMBED_SRC}?error=1` });
    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    cy.get("#foxgloveStatus", { timeout: 20000 })
      .should("have.class", "is-error")
      .and("contain.text", "mock viewer failed to open the data source");
  });

  it("explains an unconfigured viewer and mounts no iframe", () => {
    stubFoxgloveApis({
      available: false,
      sdk_ready: false,
      reason: "Foxglove SDK assets are not installed on this agent VM (/opt/npa-agent/foxglove/sdk).",
      data_source: null,
    });
    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    // `not.have.attr` yields an undefined subject, so assert separately.
    cy.get("#foxgloveMessage").should("not.have.attr", "hidden");
    cy.get("#foxgloveMessage").should("contain.text", "Foxglove viewer unavailable");
    cy.get("#foxgloveMessage").should("contain.text", "not installed");
    cy.get("#foxgloveStatus").should("have.class", "is-error");
    cy.get("#viewerPaneFoxglove iframe").should("not.exist");
  });

  it("does not load the Foxglove SDK until the operator opens the tab", () => {
    stubFoxgloveApis();
    // Boot completed in beforeEach; nothing Foxglove-related should have run.
    cy.get("@foxgloveConfig.all").should("have.length", 0);
    cy.get("#viewerPaneFoxglove iframe").should("not.exist");

    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");
    cy.get("#viewerPaneFoxglove iframe").should("exist");
  });

  it("renders the recording with the self-hosted OSS viewer when no Foxglove app is configured", () => {
    // The point of the fallback: an operator with no Foxglove account still sees
    // the recording, in-page, instead of a config screen.
    stubFoxgloveApis({
      viewer_backend: "self-hosted",
      embed_src: "",
      self_hosted_ready: true,
      self_hosted_url: `/lichtblick/?ds=remote-file&ds.url=${encodeURIComponent(MCAP_URL)}`,
    });
    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    cy.get("#viewerPaneFoxglove iframe", { timeout: 20000 })
      .should("have.attr", "src")
      .and("include", "/lichtblick/?ds=remote-file");
    // The recording must be pinned to this page's origin: the in-page viewer
    // fetches it same-origin (that path grants no CORS).
    cy.get("#viewerPaneFoxglove iframe").should(($frame) => {
      const src = new URL($frame.attr("src"), window.location.origin);
      const ds = new URL(String(src.searchParams.get("ds.url")), window.location.origin);
      expect(ds.origin).to.eq(window.location.origin);
      expect(ds.pathname).to.eq(MCAP_URL);
    });
    cy.get("#foxgloveStatus", { timeout: 20000 })
      .should("have.class", "is-ready")
      .and("contain.text", "Self-hosted");
    cy.get("#foxgloveMessage").should("have.attr", "hidden");
    // The mock viewer echoes the data source it was handed.
    cy.get("#viewerPaneFoxglove iframe").should(($frame) => {
      const doc = $frame[0].contentDocument;
      expect(doc, "self-hosted viewer document").to.exist;
      expect(String(doc.body.textContent)).to.include("sim2real.mcap".slice(0, 4));
    });
  });

  it("points at the OSS viewer when nothing can render the official app", () => {
    stubFoxgloveApis({
      available: false,
      viewer_backend: "",
      embed_src: "",
      sdk_ready: false,
      self_hosted_ready: true,
      reason: "No Foxglove embed source is configured (NPA_FOXGLOVE_EMBED_SRC).",
      data_source: null,
    });
    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");

    cy.get("#foxgloveMessage").should("not.have.attr", "hidden");
    cy.get("#foxgloveMessage").should("contain.text", "self-hosted");
    cy.get("#foxgloveMessage").should("contain.text", "Lichtblick");
    cy.get("#viewerPaneFoxglove iframe").should("not.exist");
  });

  it("describes the viewer from state and never claims a captured frame", () => {
    stubFoxgloveApis();
    let chatBody = null;
    cy.intercept("POST", "/api/chat", (req) => {
      chatBody = req.body;
      req.reply({
        statusCode: 200,
        body: { reply: "State-level description of the Foxglove viewer.", grounded: false },
      });
    }).as("describeChat");

    cy.get("#tabRerun").click();
    cy.get("#renderModeFoxglove").click();
    cy.wait("@foxgloveConfig");
    cy.get("#foxgloveStatus", { timeout: 20000 }).should("have.class", "is-ready");

    cy.get("#describeVisual").click();
    cy.wait("@describeChat", { timeout: 30000 });
    cy.then(() => {
      expect(chatBody, "chat request body").to.exist;
      expect(chatBody.visual_context.kind).to.eq("foxglove");
      expect(chatBody.visual_context.has_image).to.eq(false);
      expect(chatBody.visual_context.frame_quality).to.eq("cross-origin-embed");
      const text = String(chatBody.messages[chatBody.messages.length - 1].content);
      expect(text).to.include("visual_kind: `foxglove`");
      expect(text).to.include("cross-origin iframe");
      expect(text).to.include("remote-file");
    });
  });
});
