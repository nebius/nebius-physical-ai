/**
 * Embedded Foxglove viewer coverage for the NPA agent UI.
 *
 * The SDK under test is the real `@foxglove/embed` npm package (served from
 * /foxglove/sdk exactly as the agent VM serves it). The Foxglove application
 * itself is a licensed product that cannot run in CI, so the iframe points at a
 * protocol-accurate stand-in that implements the documented postMessage
 * handshake and records every command the UI sends.
 */

import { resolveLiveAgentConfig } from "../support/e2e";

// The SDK requires an absolute embed URL (`new URL(src)`), exactly like a real
// Foxglove deployment; the agent backend enforces the same rule.
const MOCK_EMBED_SRC = `${Cypress.config("baseUrl")}/mock-foxglove-app/`;
const MCAP_URL = "/foxglove/data/tok-session.mcap";

const RICH_TOPICS = {
  "/camera": "foxglove.CompressedImage",
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
  return {
    version: 1,
    content: split("column", [
      [0.72, split("row", [
        [0.68, panel("ThreeDee", "Robot motion and end-effector trajectory", {
          fixedFrame: "npa_action_space",
          topics: {
            "/robot/diagnostic_scene": { visible: true },
            "/robot/diagnostic_pose": { visible: true },
            "/robot/diagnostic_trajectory": { visible: true },
          },
        })],
        [0.32, panel("Image", "Primary camera", { imageMode: { imageTopic: "/camera" } })],
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
      layout_storage_key: "npa-agent-foxglove-robot-motion-v2",
      layout: richLayout(),
      visualization: {
        contract: "npa.foxglove.robot-motion.v2",
        fixed_frame: "npa_action_space",
        fidelity: "Action-derived diagnostic schematic; not calibrated robot/world kinematics.",
        topics: RICH_TOPICS,
        checked: true,
      },
      live_url: "",
      data_source: { type: "remote-file", urls: [MCAP_URL] },
      run_id: "mock-run",
      artifact_key: "mock-run/reports/session.mcap",
      recording_url: MCAP_URL,
      updated_at: "2026-07-30T00:00:00+00:00",
    },
    overrides || {}
  );
}

function stubFoxgloveApis(configOverrides) {
  const config = foxgloveConfig(configOverrides);
  cy.intercept("GET", "/api/foxglove/config", {
    statusCode: 200,
    body: config,
  }).as("foxgloveConfig");
  cy.intercept("GET", "/api/foxglove/status", {
    statusCode: 200,
    body: {
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
      artifact_key: config.artifact_key,
      artifact_render: "foxglove",
      recording_url: config.recording_url,
      data_source_type: config.data_source ? config.data_source.type : "",
      data_source: config.data_source,
    },
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

function assertSingleFoxgloveWebAction(options = {}) {
  const enabled = options.enabled !== false;
  cy.get('[data-testid="open-foxglove-web"]')
    .should("have.length", 1)
    .and("be.visible")
    .and("have.prop", "tagName", "BUTTON")
    .and("have.attr", "type", "button")
    .and("have.attr", "aria-describedby", "foxgloveExportNote");
  cy.contains("button", "View in Foxglove").should("have.length", 1);
  cy.get("#foxgloveOpenWeb").should(enabled ? "be.enabled" : "be.disabled");
  cy.get("#foxgloveOpenDesktop").should("not.exist");
  cy.contains(/Foxglove Desktop/i).should("not.exist");
  cy.get("#renderModeLichtblick").should("have.length", 1);
}

function foxgloveExportResponse(runId, overrides = {}) {
  const canonicalHash = String(overrides.sha256 || "a".repeat(64));
  const layoutId = String(overrides.layoutId === undefined ? "layout_rich_v1" : overrides.layoutId);
  const seek = "2026-08-10T12:00:00.250000000Z";
  const recordingUrl = `${window.location.origin}/foxglove/data/${runId}.mcap`;
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
      cy.get("#foxgloveOpenWeb")
        .scrollIntoView()
        .should("be.visible")
        .and("have.text", "View in Foxglove");
    });
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
          "npa-agent-foxglove-robot-motion-v2",
        );
        expect(ack.payload.initialLayoutParams.force).to.eq(undefined);
        const layout = ack.payload.initialLayoutParams.layout;
        expect(layout.version).to.eq(1);
        const collectPanels = (node) => node.type === "panel"
          ? [node]
          : (node.items || []).flatMap((item) => collectPanels(item.content));
        const panels = collectPanels(layout.content);
        expect(panels.map((panel) => panel.panelType)).to.deep.eq([
          "ThreeDee", "Image", "Plot", "StateTransitions", "Log",
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

  it("prepares an unchecked selected run once before mounting the rich viewer", () => {
    const rich = foxgloveConfig();
    const unchecked = foxgloveConfig({
      layout_storage_key: "npa-agent-foxglove-robot-motion-v2-source-default",
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
    cy.wait("@automaticPreparation");
    expectMockAppState("ready");
    cy.get("#foxgloveVisualizationSummary")
      .should("contain.text", "robot + trajectory 3D")
      .and("have.attr", "data-state", "ready");
    mockAppFrame().its("0.contentWindow").then((win) => {
      const messages = win.__mockFoxgloveReceived || [];
      const ack = messages.find((message) => message && message.type === "handshake-ack");
      expect(ack.payload.initialLayoutParams.storageKey).to.eq(
        "npa-agent-foxglove-robot-motion-v2",
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
    cy.get("#foxgloveOpenWeb").should("have.text", "View in Foxglove").click();
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
      exports.push(runId);
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
    assertSingleFoxgloveWebAction();
    ["View", "Foxglove", "Lichtblick", "Video", "Image", "Data"].forEach((label) => {
      cy.contains(".render-mode-tabs .render-mode-tab", new RegExp(`^${label}$`)).click();
      assertSingleFoxgloveWebAction();
    });

    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    assertSingleFoxgloveWebAction();
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
    assertSingleFoxgloveWebAction();

    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@commonFoxgloveExport").its("request.body").should("deep.include", {
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
    cy.then(() => expect(exports).to.deep.eq(["non-stock-customer-run"]));
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
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@pathExport").its("request.body.run_id").should("eq", "mock-run");
    cy.get("#foxgloveExportNote").should("contain.text", "remote-file source");
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
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
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@layoutExport").then(({ response }) => {
      assertRichOfficialUrl(
        response.body.export.web_url,
        `${window.location.origin}/foxglove/data/mock-run.mcap`,
      );
    });
    cy.get("#foxgloveOpenWeb").click();
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
    cy.get("#foxgloveOpenWeb").click();
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#simRunId").should("contain.text", "non-stock-customer-run");
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
