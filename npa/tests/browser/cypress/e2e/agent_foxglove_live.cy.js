import { currentLiveAgentConfig } from "../support/e2e";

function liveAgentRequest(path, options = {}) {
  const { baseUrl, username, password } = currentLiveAgentConfig();
  return cy.request({
    url: `${String(baseUrl).replace(/\/$/, "")}${path}`,
    auth: { username, password },
    // Tenant-wide source discovery and first-time Cloud indexing can each take
    // longer than Cypress's 30-second request default on a cold live agent.
    timeout: 180000,
    ...options,
  });
}

function verificationRunId() {
  const value = String(
    Cypress.env("NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID") ||
      Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") ||
      Cypress.env("NPA_AGENT_RUN_ID") ||
      "",
  ).trim();
  expect(value, "explicit Foxglove verification run id").to.match(
    /^[A-Za-z0-9][A-Za-z0-9._-]*$/,
  );
  return value;
}

function resolveVerificationRun(runId) {
  return liveAgentRequest(
    `/api/artifacts/runs?q=${encodeURIComponent(runId)}&limit=50`,
  ).then((response) => {
    expect(response.status).to.eq(200);
    const matches = (response.body.runs || []).filter(
      (item) => String(item.run_id || "") === runId,
    );
    expect(
      matches,
      "one exact source-qualified verification run",
    ).to.have.length(1);
    expect(matches[0].run_ref, "source-qualified run reference").to.match(
      /^npa1_/,
    );
    return matches[0];
  });
}

function assertOfficialRecordingLink(exported) {
  const parsed = new URL(exported.web_url);
  expect(parsed.origin).to.eq("https://app.foxglove.dev");
  expect(parsed.pathname).to.eq("/~/view");
  expect(parsed.searchParams.get("ds")).to.eq("foxglove-stream");
  expect(parsed.searchParams.get("ds.recordingId")).to.eq(
    exported.cloud.recording_id,
  );
  expect(exported.cloud.layout.available).to.eq(true);
  expect(exported.cloud.layout.layout_id).to.match(/^[A-Za-z0-9_-]+$/);
  expect(parsed.searchParams.get("layoutId")).to.eq(
    exported.cloud.layout.layout_id,
  );
  expect(parsed.searchParams.get("ds.start")).to.match(
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$/,
  );
  expect(parsed.searchParams.get("ds.end")).to.match(
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$/,
  );
  expect(parsed.searchParams.get("time")).to.match(
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$/,
  );
  expect(parsed.searchParams.get("openIn")).to.eq(null);
  expect(decodeURIComponent(parsed.toString())).not.to.include(
    exported.recording_url,
  );
  expect(JSON.stringify(exported)).not.to.match(
    /authorization|basic|password|api.?token/i,
  );
}

describe("NPA agent official Foxglove Web against live infra", () => {
  it("uploads once, reuses the indexed recording, and drives the real UI control", () => {
    const runId = verificationRunId();
    resolveVerificationRun(runId).then((run) => {
      const runRef = run.run_ref;
      liveAgentRequest("/api/foxglove/export", {
        method: "POST",
        body: { run_id: runId, run_ref: runRef, open_web: true },
      }).then((first) => {
        expect(first.status).to.eq(200);
        expect(first.body.ok).to.eq(true);
        expect(first.body.run_id).to.eq(runId);
        if (Cypress.env("NPA_AGENT_CYPRESS_FOXGLOVE_REQUIRE_CONVERSION")) {
          expect(first.body.converted).to.eq(true);
        } else {
          // In the broad runner Lichtblick may already have published this run's
          // MCAP. The focused run requires and proves the first conversion.
          expect(first.body.converted).to.be.a("boolean");
        }
        expect(first.body.export.cloud.import_status).to.eq("complete");
        expect(first.body.export.canonical_s3_uri).to.match(
          /^s3:\/\/.+\/reports\/sim2real\.mcap$/,
        );
        expect(first.body.export.sha256).to.match(/^[a-f0-9]{64}$/);
        expect(first.body.export.cloud.recording_key).to.eq(
          `npa-${first.body.export.sha256}`,
        );
        expect(first.body.export.provenance.duration_s).to.be.greaterThan(5);
        expect(first.body.export.provenance.schemas).to.include({
          "/trajectory": "foxglove.PointCloud",
          "/tf": "foxglove.FrameTransform",
          "/log": "foxglove.Log",
        });
        expect(
          Object.values(first.body.export.provenance.schemas).filter(
            (schema) => schema === "foxglove.CompressedImage",
          ),
          "two synchronized camera schemas",
        ).to.have.length.at.least(2);
        assertOfficialRecordingLink(first.body.export);
        const source = first.body.export.recording_url;
        const size = first.body.size_bytes;

        cy.request({ url: source, method: "HEAD" }).then((head) => {
          expect(head.status).to.eq(200);
          expect(head.headers["accept-ranges"]).to.eq("bytes");
          expect(Number(head.headers["content-length"])).to.eq(size);
        });
        cy.request({
          url: source,
          method: "OPTIONS",
          headers: {
            Origin: "https://app.foxglove.dev",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Range",
          },
        }).then((options) => {
          expect([200, 204]).to.include(options.status);
          expect(options.headers["access-control-allow-origin"]).to.eq("*");
          expect(options.headers["access-control-allow-headers"]).to.include(
            "Range",
          );
        });
        cy.request({
          url: source,
          headers: { Range: "bytes=0-7" },
          encoding: "binary",
        }).then((range) => {
          expect(range.status).to.eq(206);
          expect(range.body).to.have.length(8);
          expect(range.body).to.eq("\x89MCAP0\r\n");
          expect(range.headers["content-range"]).to.match(/^bytes 0-7\//);
        });
        cy.request({
          url: source,
          method: "HEAD",
          headers: { "Accept-Encoding": "gzip" },
        }).then((gzip) => {
          expect(gzip.status).to.eq(200);
          expect(gzip.headers).not.to.have.property("content-encoding", "gzip");
        });

        liveAgentRequest("/api/foxglove/export", {
          method: "POST",
          body: { run_id: runId, run_ref: runRef, open_web: true },
        }).then((second) => {
          expect(second.body.converted).to.eq(false);
          expect(second.body.export.sha256).to.eq(first.body.export.sha256);
          expect(second.body.export.canonical_s3_uri).to.eq(
            first.body.export.canonical_s3_uri,
          );
          expect(second.body.export.web_url).to.eq(first.body.export.web_url);
          expect(second.body.export.cloud.reused).to.eq(true);
          expect(second.body.export.cloud.layout.reused).to.eq(true);
          expect(second.body.export.cloud.layout.created).to.eq(false);
          expect(second.body.export.cloud.layout.updated).to.eq(false);
        });

        liveAgentRequest(
          `/api/artifacts/run/${encodeURIComponent(runRef)}`,
        ).then((listed) => {
          expect(listed.status).to.eq(200);
          const canonical = listed.body.artifacts.find((item) =>
            String(item.key || "").endsWith("/reports/sim2real.mcap"),
          );
          expect(canonical, "canonical MCAP is immediately discoverable").to
            .exist;
          expect(canonical.s3_uri).to.eq(first.body.export.canonical_s3_uri);
          expect(Number(canonical.size)).to.eq(size);
        });
        // The byte-for-byte S3/download/public identity gate runs in Python.
        // Cypress intentionally uses metadata plus bounded range reads here:
        // materializing a multi-megabyte binary body in Electron can starve its
        // socket and screenshot event loops without adding browser coverage.
        expect(first.body.export.provenance.sha256).to.eq(
          first.body.export.sha256,
        );
        liveAgentRequest("/api/sim-viz/load-artifact", {
          method: "POST",
          body: {
            run_id: runId,
            run_ref: runRef,
            key: first.body.artifact_key,
          },
        }).then((loaded) => {
          expect(loaded.status).to.eq(200);
          expect(loaded.body.ok).to.eq(true);
          expect(loaded.body.sim_viz.run_id).to.eq(runId);
        });

        // Load the UI only after the API reuse contract is established. Initial UI
        // artifact hydration writes viewer state, so overlapping it with the first
        // export would turn this into a browser-startup race rather than an export
        // idempotency test.
        cy.visitLiveAgent();
        cy.get("meta[name='npa-ui-version']")
          .should("have.attr", "content")
          .and("match", /^\d+$/);
        // Page boot hydrates the tenant run index in the background. Let that
        // read-only S3 scan finish before exercising the export control so the
        // proof measures the button path, not two competing cold discoveries.
        cy.get("#artifactDiscoverStatus", { timeout: 300000 }).should(
          "contain.text",
          "Runs (latest first)",
        );
        // Exercise the deployed @foxglove/embed path itself. The hosted viewer
        // is cross-origin (and may require the customer's Foxglove sign-in), so
        // this verifies the supported SDK iframe + connection-state contract,
        // not pixels inside the hosted application.
        cy.get("#tabRerun").click();
        cy.get("#renderModeFoxglove").click();
        cy.get("#viewerPaneFoxglove").should("have.class", "is-active-viewer");
        cy.get("#viewerPaneFoxglove iframe", { timeout: 30000 })
          .should("have.attr", "src")
          .and("match", /^https:\/\/embed\.foxglove\.dev\//);
        cy.get("#foxgloveStatus", { timeout: 30000 }).should(($status) => {
          expect($status, "no deployed SDK error state").not.to.have.class("is-error");
          expect($status.text()).to.match(/Connecting to|Foxglove viewer ready/);
        });
        cy.intercept("POST", "/api/foxglove/export").as("liveFoxgloveUiExport");
        cy.window().then((win) => {
          const replace = cy.stub().as("foxgloveNavigate");
          cy.stub(win, "open").callsFake(() => ({
            opener: null,
            location: { replace },
            close: cy.stub(),
          }));
        });
        cy.get("#renderedDataSummary").should("contain.text", runId);
        cy.get('[data-testid="open-foxglove-web"]')
          .should("have.length", 1)
          .and("be.visible")
          .and("be.enabled")
          .and("have.text", "Open in Foxglove Web")
          .click();
        cy.wait("@liveFoxgloveUiExport", { timeout: 180000 }).then(({ request, response }) => {
          expect(request.body).to.deep.include({
            run_id: runId,
            run_ref: runRef,
            open_web: true,
          });
          expect(response.statusCode).to.eq(200);
          expect(response.body.export.web_url).to.eq(first.body.export.web_url);
          expect(response.body.export.cloud.recording_id).to.eq(
            first.body.export.cloud.recording_id,
          );
          expect(response.body.export.cloud.reused).to.eq(true);
        });
        cy.get("@foxgloveNavigate").should(
          "have.been.calledOnceWith",
          first.body.export.web_url,
        );
        cy.get("#foxgloveExportNote").should(
          "contain.text",
          "Reused the unchanged indexed",
        );
        cy.get("#renderModeData").click();
        cy.get('[data-testid="open-foxglove-web"]')
          .should("have.length", 1)
          .and("be.visible")
          .and("be.enabled")
          .click();
        cy.wait("@liveFoxgloveUiExport", { timeout: 180000 }).then(({ request, response }) => {
          expect(request.body).to.deep.include({
            run_id: runId,
            run_ref: runRef,
            open_web: true,
          });
          expect(response.statusCode).to.eq(200);
          expect(response.body.export.web_url).to.eq(first.body.export.web_url);
          expect(response.body.export.cloud.recording_id).to.eq(
            first.body.export.cloud.recording_id,
          );
          expect(response.body.export.cloud.reused).to.eq(true);
        });
        cy.get("@foxgloveNavigate").should("have.been.calledTwice");
        cy.get("@foxgloveNavigate").should(
          "always.have.been.calledWith",
          first.body.export.web_url,
        );
        cy.get('[data-testid="open-foxglove-web"]')
          .should("have.length", 1)
          .and("be.visible")
          .and("have.attr", "aria-busy", "false");
        cy.get("#foxgloveOpenDesktop").should("not.exist");
        cy.contains("Open in Foxglove Desktop").should("not.exist");
        cy.get("#renderModeLichtblick").should("have.length", 1);
        cy.get("#openFullLichtblick").should("not.exist");
        cy.get("#renderModeRerun").should("exist");

      });
    });
  });
});
