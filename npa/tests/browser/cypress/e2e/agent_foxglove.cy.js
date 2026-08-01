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
