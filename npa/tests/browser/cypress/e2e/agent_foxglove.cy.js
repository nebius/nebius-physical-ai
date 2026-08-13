/**
 * Embedded Foxglove viewer coverage for the NPA agent UI.
 *
 * The SDK under test is the real `@foxglove/embed` npm package (served from
 * /foxglove/sdk exactly as the agent VM serves it). The Foxglove application
 * itself is a licensed product that cannot run in CI, so the iframe points at a
 * protocol-accurate stand-in that implements the documented postMessage
 * handshake and records every command the UI sends.
 */

// The SDK requires an absolute embed URL (`new URL(src)`), exactly like a real
// Foxglove deployment; the agent backend enforces the same rule.
const MOCK_EMBED_SRC = `${Cypress.config("baseUrl")}/mock-foxglove-app/`;
const MCAP_URL = "/foxglove/data/tok-session.mcap";

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
      layout_storage_key: "npa-agent-foxglove",
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
  cy.contains("button", "Open in Foxglove Web").should("have.length", 1);
  cy.get("#foxgloveOpenWeb").should(enabled ? "be.enabled" : "be.disabled");
  cy.get("#foxgloveOpenDesktop").should("not.exist");
  cy.contains(/Foxglove Desktop/i).should("not.exist");
  cy.get("#renderModeLichtblick").should("have.length", 1);
}

function foxgloveExportResponse(runId, overrides = {}) {
  const canonicalHash = String(overrides.sha256 || "a".repeat(64));
  const recordingId = String(overrides.recordingId || `rec_${runId.replace(/[^A-Za-z0-9]/g, "_")}`);
  const layoutId = String(overrides.layoutId === undefined ? "layout_rich_v1" : overrides.layoutId);
  const start = "2026-08-10T12:00:00.000000000Z";
  const end = "2026-08-10T12:00:09.937500000Z";
  const seek = "2026-08-10T12:00:00.250000000Z";
  const params = new URLSearchParams({
    ds: "foxglove-stream",
    "ds.recordingId": recordingId,
    "ds.start": start,
    "ds.end": end,
    time: seek,
  });
  if (layoutId) params.set("layoutId", layoutId);
  const webUrl = `https://app.foxglove.dev/~/view?${params.toString()}`;
  const cloud = {
    recording_id: recordingId,
    recording_key: `npa-${canonicalHash}`,
    import_status: "complete",
    reused: Boolean(overrides.reused),
    layout: {
      layout_id: layoutId,
      available: Boolean(layoutId),
      created: Boolean(overrides.layoutCreated),
      updated: false,
      reused: !overrides.layoutCreated && Boolean(layoutId),
      reason: layoutId ? "" : "Layout API is unavailable on this plan",
    },
  };
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
      foxglove_cloud: cloud,
      lichtblick_ready: true,
    },
    export: {
      available: true,
      recording_url: `${window.location.origin}/foxglove/data/${runId}.mcap`,
      download_url: `${window.location.origin}/foxglove/data/${runId}.mcap`,
      web_url: webUrl,
      data_source: "foxglove-stream",
      canonical_s3_uri: `s3://mock/${runId}/reports/sim2real.mcap`,
      sha256: canonicalHash,
      cloud,
      provenance: {
        start_time_ns: 1786363200000000000,
        end_time_ns: 1786363209937500000,
        schemas: {
          "/camera/overview": "foxglove.CompressedImage",
          "/camera/workspace": "foxglove.CompressedImage",
          "/trajectory": "foxglove.PointCloud",
          "/tf": "foxglove.FrameTransform",
          "/metrics/execution": "npa.metrics.execution",
          "/log": "foxglove.Log",
        },
        numeric_paths: {
          "/metrics/execution": ["reward", "progress", "state_norm"],
        },
      },
    },
  };
}

function assertRichOfficialUrl(webUrl, recordingId = null) {
  const parsed = new URL(webUrl);
  expect(parsed.origin).to.eq("https://app.foxglove.dev");
  expect(parsed.pathname).to.eq("/~/view");
  expect(parsed.searchParams.get("ds")).to.eq("foxglove-stream");
  if (recordingId) {
    expect(parsed.searchParams.get("ds.recordingId")).to.eq(recordingId);
  }
  expect(parsed.searchParams.get("layoutId")).to.eq("layout_rich_v1");
  expect(parsed.searchParams.get("ds.start")).to.eq("2026-08-10T12:00:00.000000000Z");
  expect(parsed.searchParams.get("ds.end")).to.eq("2026-08-10T12:00:09.937500000Z");
  expect(parsed.searchParams.get("time")).to.eq("2026-08-10T12:00:00.250000000Z");
  expect(parsed.searchParams.get("openIn")).to.eq(null);
  expect(webUrl).not.to.match(/token|password|authorization/i);
}

describe("NPA agent UI — embedded Foxglove viewer", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
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
    stubFoxgloveApis();
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
        expect(ack.payload.initialLayoutParams.storageKey).to.eq("npa-agent-foxglove");
        const source = ack.payload.initialDataSource;
        expect(source, "initial data source").to.exist;
        expect(source.type).to.eq("remote-file");
        // Data-source URLs must be absolute: the viewer fetches them cross-origin.
        expect(source.urls[0]).to.eq(`${window.location.origin}${MCAP_URL}`);
      });
  });

  it("downloads MCAP and exposes one web-only indexed-recording action", () => {
    const config = stubFoxgloveApis();
    const webUrl = foxgloveExportResponse("mock-run", {
      recordingId: "rec_mock123",
      reused: true,
    }).export.web_url;
    const canonicalHash = "a".repeat(64);
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      const cloud = request.body.open_web
        ? {
            recording_id: "rec_mock123",
            recording_key: `npa-${canonicalHash}`,
            import_status: "complete",
            reused: true,
            layout: {
              layout_id: "layout_rich_v1",
              available: true,
              created: false,
              updated: false,
              reused: true,
              reason: "",
            },
          }
        : {};
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
          foxglove_cloud: cloud,
          lichtblick_ready: true,
          lichtblick_iframe_url: "/lichtblick/?ds=remote-file&ds.url=%2Flichtblick%2Frecordings%2Fsim2real.mcap",
        },
        export: request.body.open_web ? {
          available: true,
          recording_url: `${window.location.origin}${MCAP_URL}`,
          download_url: `${window.location.origin}${MCAP_URL}`,
          web_url: webUrl,
          data_source: "foxglove-stream",
          cloud,
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
    cy.get("#foxgloveOpenWeb").should("have.text", "Open in Foxglove Web").click();
    cy.wait("@foxgloveExport").its("request.body.open_web").should("eq", true);
    cy.get("@foxgloveNavigate").should("have.been.calledWith", webUrl);
    cy.then(() => assertRichOfficialUrl(webUrl, "rec_mock123"));
    cy.get("#foxgloveExportNote").should("contain.text", "Reused the unchanged indexed");
    cy.get("#renderedDataSummary").should("contain.text", "Foxglove Cloud: complete (reused)");
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
    ["Rerun", "Lichtblick", "Foxglove", "Video", "Image", "Data"].forEach((mode) => {
      cy.get(`#renderMode${mode}`).click();
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

  it("uses conversion for a source-only run and reuses a canonical recording for the next run", () => {
    stubFoxgloveApis();
    let exportCount = 0;
    cy.intercept("POST", "/api/foxglove/export", (request) => {
      exportCount += 1;
      const runId = String(request.body.run_id || "");
      request.reply({ statusCode: 200, body: foxgloveExportResponse(runId, {
        converted: exportCount === 1,
        reused: exportCount === 2,
        recordingId: exportCount === 1 ? "rec_converted" : "rec_reused",
      }) });
    }).as("pathExport");
    cy.window().then((win) => {
      const replace = cy.stub().as("pathNavigate");
      cy.stub(win, "open").callsFake(() => ({ opener: null, location: { replace }, close: cy.stub() }));
    });
    cy.get("#tabRerun").click();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@pathExport").its("request.body.run_id").should("eq", "mock-run");
    cy.get("#foxgloveExportNote").should("contain.text", "Uploaded and indexed");
    cy.get("#runIdInput").clear().type("non-stock-customer-run");
    cy.get("#loadRunData").click();
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@pathExport").its("request.body.run_id").should("eq", "non-stock-customer-run");
    cy.get("#foxgloveExportNote").should("contain.text", "Reused the unchanged indexed");
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
      assertRichOfficialUrl(response.body.export.web_url, "rec_mock_run");
      expect(response.body.export.cloud.layout.reused).to.eq(true);
    });
    cy.get("#foxgloveOpenWeb").click();
    cy.wait("@layoutExport").then(({ response }) => {
      const parsed = new URL(response.body.export.web_url);
      expect(parsed.searchParams.get("layoutId")).to.eq(null);
      expect(parsed.searchParams.get("time")).to.eq("2026-08-10T12:00:00.250000000Z");
      expect(response.body.export.cloud.layout.available).to.eq(false);
      expect(response.body.export.cloud.layout.reason).to.contain("unavailable");
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
      .and("contain.text", "Opening Foxglove Web");
    cy.get("#foxgloveOpenWeb").click({ force: true });
    cy.wait("@slowExport");
    cy.get("#foxgloveOpenWeb").should("be.enabled").and("have.attr", "aria-busy", "false");
    cy.then(() => expect(requests, "one export request").to.eq(1));
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
