import { currentLiveAgentConfig } from "../support/e2e";

function liveEnabled() {
  return String(Cypress.env("NPA_AGENT_CYPRESS_LIVE") || "") === "1";
}

function liveAgentRequest(path, options = {}) {
  const { baseUrl, username, password } = currentLiveAgentConfig();
  return cy.request({
    url: `${String(baseUrl).replace(/\/$/, "")}${path}`,
    auth: { username, password },
    log: false,
    timeout: 180000,
    ...options,
  });
}

function configuredRunId() {
  return String(
    Cypress.env("NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID") ||
      Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") ||
      "",
  ).trim();
}

function discoverVerificationRun() {
  const wanted = configuredRunId();
  if (wanted) {
    expect(wanted, "configured Foxglove verification run id").to.match(
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/,
    );
  }
  const query = wanted ? `?q=${encodeURIComponent(wanted)}&limit=50` : "?limit=50";
  return liveAgentRequest(`/api/artifacts/runs${query}`).then((response) => {
    expect(response.status).to.eq(200);
    const runs = Array.isArray(response.body.runs) ? response.body.runs : [];
    const matches = wanted
      ? runs.filter((item) => String(item.run_id || "") === wanted)
      : runs;
    expect(matches.length, "a real discovered run for Foxglove verification").to.be.greaterThan(0);
    if (wanted) {
      expect(matches, "one exact source-qualified verification run").to.have.length(1);
    }
    const run = matches[0];
    expect(String(run.run_id || ""), "discovered run id").to.match(
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/,
    );
    expect(run.run_ref, "source-qualified run reference").to.match(/^npa1_/);
    return run;
  });
}

function assertRemoteFileLink(exported) {
  const parsed = new URL(exported.web_url);
  expect(parsed.origin).to.eq("https://app.foxglove.dev");
  expect(parsed.pathname).to.eq("/~/view");
  expect(parsed.searchParams.get("ds")).to.eq("remote-file");
  expect(parsed.searchParams.getAll("ds.url")).to.deep.eq([exported.recording_url]);
  expect(parsed.searchParams.get("ds.recordingId")).to.eq(null);
  expect(parsed.searchParams.get("openIn")).to.eq(null);
  expect(exported.data_source).to.eq("remote-file");
  expect(exported.web_open_mode).to.eq("remote-file");
  expect(exported.recording_url).to.match(/^https:\/\//);
  expect(exported.web_url).not.to.match(/authorization|basic|password|api.?token/i);
}

describe("NPA agent official Foxglove embed against live infrastructure", () => {
  before(function () {
    if (!liveEnabled()) {
      this.skip();
      return;
    }
    // Fail closed before any request when the opt-in gate is set incompletely.
    currentLiveAgentConfig();
  });

  it("mounts the official SDK and opens the selected public MCAP remote-file link", () => {
    discoverVerificationRun().then((run) => {
      const requestBody = {
        run_id: String(run.run_id),
        run_ref: String(run.run_ref),
        open_web: true,
      };
      liveAgentRequest("/api/foxglove/export", {
        method: "POST",
        body: requestBody,
      }).then((prepared) => {
        expect(prepared.status).to.eq(200);
        expect(prepared.body.ok).to.eq(true);
        expect(prepared.body.run_id).to.eq(run.run_id);
        const exported = prepared.body.export;
        assertRemoteFileLink(exported);
        expect(exported.canonical_s3_uri).to.match(
          /^s3:\/\/.+\/reports\/sim2real\.mcap$/,
        );
        expect(exported.sha256).to.match(/^[a-f0-9]{64}$/);
        expect(Number(exported.size_bytes || prepared.body.size_bytes)).to.be.greaterThan(8);

        cy.request({ url: exported.recording_url, method: "HEAD", log: false }).then((head) => {
          expect(head.status).to.eq(200);
          expect(head.headers["accept-ranges"]).to.eq("bytes");
          expect(head.headers).not.to.have.property("content-encoding", "gzip");
        });
        cy.request({
          url: exported.recording_url,
          method: "OPTIONS",
          log: false,
          headers: {
            Origin: "https://embed.foxglove.dev",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Range",
          },
        }).then((options) => {
          expect([200, 204]).to.include(options.status);
          expect(options.headers["access-control-allow-origin"]).to.eq("*");
          expect(options.headers["access-control-allow-headers"]).to.include("Range");
        });
        cy.request({
          url: exported.recording_url,
          log: false,
          headers: { Origin: "https://embed.foxglove.dev", Range: "bytes=0-7" },
          encoding: "binary",
        }).then((range) => {
          expect(range.status).to.eq(206);
          expect(range.body).to.eq("\x89MCAP0\r\n");
          expect(range.headers["content-range"]).to.match(/^bytes 0-7\//);
        });

        liveAgentRequest("/api/foxglove/config").then((response) => {
          expect(response.status).to.eq(200);
          expect(response.body.available).to.eq(true);
          expect(response.body.viewer_backend).to.eq("foxglove-sdk");
          expect(response.body.sdk_ready).to.eq(true);
          expect(response.body.embed_src).to.match(/^https:\/\//);
          expect(response.body.requires_account_note).to.match(/sign in/i);
          expect(response.body.data_source).to.deep.include({ type: "remote-file" });
          expect(response.body.data_source.urls).to.deep.eq([exported.recording_url]);
        });
        liveAgentRequest("/api/foxglove/status").then((response) => {
          expect(response.status).to.eq(200);
          expect(response.body.viewer_backend).to.eq("foxglove-sdk");
          expect(response.body.run_id).to.eq(run.run_id);
          expect(response.body.recording_url).to.eq(exported.recording_url);
        });

        cy.visitLiveAgent();
        cy.get("#tabRerun").click();
        cy.get("#renderModeFoxglove").click();
        cy.get("#viewerPaneFoxglove").should("have.class", "is-active-viewer");
        cy.get("#viewerPaneFoxglove iframe", { timeout: 30000 })
          .should("have.attr", "src")
          .and("match", /^https:\/\/embed\.foxglove\.dev\//);
        cy.get("#foxgloveStatus", { timeout: 30000 }).should(($status) => {
          expect($status, "no SDK or hosted-viewer error").not.to.have.class("is-error");
          expect($status.text()).to.match(/Connecting to|Foxglove viewer ready/);
        });

        cy.intercept("POST", "/api/foxglove/export").as("liveFoxgloveOpen");
        cy.window().then((win) => {
          const replace = cy.stub().as("foxgloveNavigate");
          cy.stub(win, "open").as("foxglovePopup").callsFake(() => ({
            opener: null,
            location: { replace },
            close: cy.stub(),
          }));
        });
        cy.get("#renderedDataSummary").should("contain.text", run.run_id);
        cy.get('[data-testid="open-foxglove-web"]')
          .should("be.visible")
          .and("be.enabled")
          .click();
        cy.get("@foxglovePopup").should("have.been.calledOnceWith", "about:blank", "_blank");
        let openedWebUrl = "";
        cy.wait("@liveFoxgloveOpen", { timeout: 180000 }).then(({ request, response }) => {
          expect(request.body).to.deep.include(requestBody);
          expect(response.statusCode).to.eq(200);
          assertRemoteFileLink(response.body.export);
          expect(response.body.export.sha256).to.eq(exported.sha256);
          openedWebUrl = response.body.export.web_url;
        });
        // cy.wait() resolves as soon as the response completes; allow the
        // application's promise continuation to perform the safe navigation.
        cy.get("@foxgloveNavigate", { timeout: 30000 }).should((navigate) => {
          expect(navigate.callCount, "one popup navigation").to.eq(1);
          expect(navigate.firstCall.args, "exact response deep link").to.deep.eq([
            openedWebUrl,
          ]);
        });
        cy.get("#foxgloveExportNote")
          .should("have.attr", "data-state", "success")
          .and("contain.text", "remote-file source");
        cy.get('[data-testid="open-foxglove-web"]')
          .should("have.attr", "aria-busy", "false")
          .and("be.enabled");
      });
    });
  });
});
