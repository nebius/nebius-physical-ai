function liveAgentRequest(path, options = {}) {
  const baseUrl = Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl");
  const username = Cypress.env("NPA_AGENT_USER");
  const password = Cypress.env("NPA_AGENT_PASSWORD");
  return cy.request({
    url: `${String(baseUrl || "").replace(/\/$/, "")}${path}`,
    auth: { username, password },
    ...options,
  });
}

describe("NPA agent live RRD artifact", () => {
  before(function () {
    if (!Cypress.env("NPA_AGENT_CYPRESS_ARTIFACT_KEY")) {
      this.skip();
    }
  });

  it("loads an explicitly selected RRD and paints it in the real Rerun viewer", () => {
    const runId = String(Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || "");
    const artifactKey = String(Cypress.env("NPA_AGENT_CYPRESS_ARTIFACT_KEY") || "");
    expect(runId, "NPA_AGENT_CYPRESS_RUN_ID").not.to.eq("");
    expect(artifactKey, "NPA_AGENT_CYPRESS_ARTIFACT_KEY").to.match(/\.rrd$/);

    let capabilityPath = "";
    const loadUntilReady = (attempt) => {
      liveAgentRequest("/api/sim-viz/load-artifact", {
        method: "POST",
        body: { run_id: runId, key: artifactKey },
        timeout: 120000,
        failOnStatusCode: false,
      }).then((response) => {
        if (response.status !== 200 && attempt < 4) {
          cy.wait(2000).then(() => loadUntilReady(attempt + 1));
          return;
        }
        expect(response.status, JSON.stringify(response.body)).to.eq(200);
        expect(response.body).to.have.property("ok", true);
        expect(response.body).to.have.property("render", "rerun");
        expect(String(response.body.artifact_uri || "")).to.include(artifactKey);
        capabilityPath = String((response.body.sim_viz && response.body.sim_viz.artifact_preview_url) || "");
        expect(capabilityPath).to.match(/^\/rerun\/recordings\/cap-[A-Za-z0-9_-]{43}\.rrd$/);
      });
    };
    loadUntilReady(0);

    cy.then(() => {
      const baseUrl = String(
        Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl") || "",
      ).replace(/\/$/, "");
      const capabilityUrl = `${baseUrl}${capabilityPath}`;
      cy.request({ url: capabilityUrl, failOnStatusCode: false }).then((response) => {
        expect(response.status).to.eq(200);
        expect(String(response.body || "").length).to.be.greaterThan(0);
      });
      cy.request({ url: `${baseUrl}/rerun/recordings/sim2real.rrd`, failOnStatusCode: false }).then((response) => {
        expect(response.status, "fixed recording path is denied").to.eq(404);
      });
    });

    cy.visitLiveAgent();
    cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
    cy.get("#tabRerun").click();
    cy.get("#rerunFrame", { timeout: 120000 }).should(($frame) => {
      expect(decodeURIComponent(String($frame.attr("src") || ""))).to.include(capabilityPath);
    });

    const waitForRenderedRecording = (attempt) => {
      cy.window().then((win) => {
        const iframe = win.document.getElementById("rerunFrame");
        return win.__NPA_AGENT_TEST__.probeRerunCanvasContent(iframe);
      }).then((rendered) => {
        if (!rendered && attempt < 60) {
          cy.wait(1000).then(() => waitForRenderedRecording(attempt + 1));
          return;
        }
        expect(rendered, "Rerun painted non-blank pixels from the selected RRD").to.eq(true);
        // The first non-blank paint can be Rerun's video decoder spinner. Give the
        // embedded MP4 time to decode before preserving the visual evidence.
        cy.wait(10000);
        cy.get("#rerunFrame").screenshot("wan-r7-rerun-canvas");
      });
    };
    waitForRenderedRecording(0);
  });
});
