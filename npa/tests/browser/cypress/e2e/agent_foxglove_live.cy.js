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
  expect(exported.layout, "server-side shared layout result").to.deep.include({
    available: true,
  });
  expect(exported.layout.layout_id).to.match(/^[A-Za-z0-9_-]+$/);
  expect(parsed.searchParams.get("layoutId")).to.eq(exported.layout.layout_id);
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
        expect(exported.provenance.visualization_contract).to.eq(
          "npa.foxglove.robot-motion.v2",
        );
        expect(exported.provenance.visualization_fixed_frame).to.eq("npa_action_space");
        expect(exported.provenance.visualization_fidelity).to.match(/not calibrated/i);
        expect(exported.provenance.schemas).to.deep.include({
          "/camera": "foxglove.CompressedImage",
          "/robot/diagnostic_scene": "foxglove.SceneUpdate",
          "/robot/diagnostic_pose": "foxglove.PoseInFrame",
          "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
          "/robot/diagnostic_joint_states": "foxglove.JointStates",
          "/actuators/commands": "npa.ActuatorCommands",
          "/run/state": "npa.RunState",
          "/log": "foxglove.Log",
        });
        expect(
          Object.values(exported.provenance.schemas),
          "numeric metrics schema",
        ).to.include("npa.RunMetrics.execution");

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
          expect(response.body.layout.version).to.eq(1);
          expect(response.body.visualization).to.deep.include({
            contract: "npa.foxglove.robot-motion.v2",
            fixed_frame: "npa_action_space",
          });
          expect(response.body.layout_storage_key).to.eq(
            "npa-agent-foxglove-robot-motion-v2",
          );
        });
        liveAgentRequest("/api/foxglove/status").then((response) => {
          expect(response.status).to.eq(200);
          expect(response.body.viewer_backend).to.eq("foxglove-sdk");
          expect(response.body.run_id).to.eq(run.run_id);
          expect(response.body.recording_url).to.eq(exported.recording_url);
        });

        cy.visitLiveAgent();
        cy.get("#tabRerun").should("have.text", "View").click();
        // A clean browser has no persisted active run. Exercise the visible
        // run selector exactly as an operator must before asserting viewer
        // state; the API export above intentionally does not mutate browser
        // selection state.
        cy.get("#artifactPrefix").clear().type(String(run.run_id)).type("{enter}");
        cy.get("#runIdSelect", { timeout: 180000 }).find("option").should(($options) => {
          expect(
            [...$options].map((option) => option.value),
            "source-qualified discovered run option",
          ).to.include(String(run.run_ref));
        });
        cy.get("#runIdInput").clear().type(String(run.run_id));
        cy.get("#loadRunData").should("be.visible").and("be.enabled").click();
        cy.get("#loadRunData", { timeout: 180000 })
          .should("have.attr", "aria-busy", "false")
          .and("be.enabled");
        cy.get("#simRunId", { timeout: 180000 }).should("contain.text", run.run_id);
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
            .should(($pane) => {
              const rect = $pane[0].getBoundingClientRect();
              expect(rect.width, `${label} live pane width`).to.be.greaterThan(0);
              expect(rect.height, `${label} live pane height`).to.be.greaterThan(0);
            });
        });
        cy.get("#renderModeFoxglove").click();
        cy.get("#viewerPaneFoxglove").should("have.class", "is-active-viewer");
        cy.get("#viewerPaneFoxglove iframe", { timeout: 180000 })
          .should("have.attr", "src")
          .and("match", /^https:\/\/embed\.foxglove\.dev\//);
        cy.get("#foxgloveStatus", { timeout: 30000 }).should(($status) => {
          expect($status, "no SDK or hosted-viewer error").not.to.have.class("is-error");
          expect($status.text()).to.match(/Connecting to|Foxglove viewer ready|awaiting browser sign-in/);
        });
        cy.get("#foxgloveVisualizationSummary")
          .should("be.visible")
          .and("contain.text", "robot + trajectory 3D")
          .and("contain.text", "not calibrated robot/world kinematics");
        cy.get('[data-testid="open-foxglove-web"]')
          .should("be.visible")
          .and("be.enabled")
          .and("have.text", "View in Foxglove");

        // This Cypress task launches a separate clean Chromium profile and uses
        // the real button click plus Playwright's browser page event. It never
        // stubs window.open and returns only non-secret contract/evidence facts.
        cy.task(
          "verifyFoxgloveHostedNavigation",
          { runId: String(run.run_id), runRef: String(run.run_ref) },
          // The task performs several independently strict live waits in a
          // second clean profile. Its outer budget must cover their sum rather
          // than terminating before the individual assertions can report.
          { log: false, timeout: 600000 },
        ).then((result) => {
          expect(result.runId).to.eq(run.run_id);
          expect(result.labels).to.deep.eq([
            "View", "Foxglove", "Lichtblick", "Video", "Image", "Data",
          ]);
          for (const label of ["View", "Foxglove", "Lichtblick"]) {
            expect(result.paneGeometry[label].width).to.be.greaterThan(0);
            expect(result.paneGeometry[label].height).to.be.greaterThan(0);
          }
          expect(result.officialContract).to.deep.include({
            requestMatchedResponse: true,
            sourceType: "remote-file",
            oneAbsoluteHttpsMcap: true,
            encodedExactlyOnce: true,
            layoutIdPresent: true,
          });
          expect(result.officialContract.responseStatus).to.be.within(200, 399);
          expect(result.hostedSurface.finalOrigin).to.eq("https://app.foxglove.dev");
          expect(result.hostedSurface.pixels.nonblank).to.eq(true);
          expect(result.evidence.desktop).to.match(/live-agent-desktop-after\.png$/);
          expect(result.evidence.mobile).to.match(/live-agent-mobile-after\.png$/);
          expect(result.evidence.hosted).to.match(/live-hosted-foxglove-after\.png$/);
        });
      });
    });
  });
});
