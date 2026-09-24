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

function exactSourceQuery(entry, cursor = "") {
  const params = new URLSearchParams({
    project_id: String(entry.project_id || ""),
    resource_bucket: String(entry.bucket || ""),
    resolved_prefix: String(entry.resolved_prefix || ""),
    source_selected: "1",
    limit: "200",
  });
  if (cursor) params.set("cursor", cursor);
  return params.toString();
}

function findArtifactInSource(entry, artifactKey, cursor = "") {
  return liveAgentRequest(
    `/api/artifacts/run/${encodeURIComponent(entry.run_id)}?${exactSourceQuery(entry, cursor)}`,
    { timeout: 120000 },
  ).then((response) => {
    expect(response.status, "exact source artifact inventory").to.eq(200);
    const artifacts = Array.isArray(response.body.artifacts) ? response.body.artifacts : [];
    const artifact = artifacts.find((item) => String(item.key || "") === artifactKey);
    if (artifact) return { entry, artifact };
    const nextCursor = String(response.body.next_cursor || "");
    return nextCursor ? findArtifactInSource(entry, artifactKey, nextCursor) : null;
  });
}

function discoverExactRun(runId, cursor = "", found = []) {
  const params = new URLSearchParams({ q: runId, limit: "100" });
  if (cursor) params.set("cursor", cursor);
  return liveAgentRequest(`/api/artifacts/runs?${params.toString()}`, {
    timeout: 120000,
  }).then((response) => {
    expect(response.status, "exact run discovery").to.eq(200);
    const matches = (response.body.runs || []).filter(
      (entry) => String(entry.run_id || "") === runId,
    );
    const all = [...found, ...matches];
    const nextCursor = String(response.body.next_cursor || "");
    return nextCursor ? discoverExactRun(runId, nextCursor, all) : all;
  });
}

function resolveRrdSource(runId, artifactKey) {
  return discoverExactRun(runId).then((matches) => {
    expect(matches, "one unambiguous server-issued source tuple").to.have.length(1);
    const entry = matches[0];
    expect(String(entry.run_ref || ""), "source-qualified run reference").to.match(/^npa1_/);
    expect(String(entry.project_id || ""), "source project identity").not.to.eq("");
    expect(String(entry.bucket || ""), "source bucket identity").not.to.eq("");
    return findArtifactInSource(entry, artifactKey).then((found) => {
      expect(found, "RRD in exact source inventory").to.exist;
      return found;
    });
  });
}

describe("NPA agent live RRD artifact", () => {
  before(function () {
    if (!Cypress.env("NPA_AGENT_CYPRESS_ARTIFACT_KEY")) {
      this.skip();
    }
  });

  it("finds one durable default from a fresh unscoped UI search twice", () => {
    const runId = String(Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || "");
    expect(runId, "NPA_AGENT_CYPRESS_RUN_ID").not.to.eq("");

    cy.intercept("GET", "/api/artifacts/runs*", (request) => {
      const query = new URL(request.url).searchParams;
      if (query.get("q") === runId) request.alias = "exactDefaultSearch";
    });

    const searchFromFreshPage = () => {
      cy.visitLiveAgent();
      cy.get("#tabRerun", { timeout: 30000 }).click();
      cy.get("#artifactPrefix").clear().type(runId, { delay: 0 });
      cy.wait("@exactDefaultSearch", { timeout: 120000 }).then(({ request, response }) => {
        const query = new URL(request.url).searchParams;
        expect(query.get("project_id"), "no hidden project scope").to.eq(null);
        expect(query.get("resource_bucket"), "no hidden bucket scope").to.eq(null);
        expect(query.get("prefix") || "", "no hidden prefix scope").to.eq("");
        expect(response.statusCode).to.eq(200);
        expect(response.body.count).to.eq(1);
        if (response.body.query_complete === true) {
          expect(response.body.total_runs).to.eq(1);
          expect(response.body.pagination_complete).to.eq(true);
          expect(response.body.truncated).to.eq(false);
        } else {
          expect(response.body.total_runs).to.eq(null);
          expect(response.body.total_runs_scope).to.eq("unavailable");
          expect(response.body.pagination_complete).to.eq(false);
          expect(response.body.truncated).to.eq(true);
          expect(response.body.source_errors).to.be.an("array").and.not.to.be.empty;
        }
        expect(response.body.runs).to.have.length(1);
        expect(response.body.runs[0].run_id).to.eq(runId);
        expect(response.body.runs[0].run_ref).to.match(/^npa1_/);
      });
      cy.get("#runIdSelect option").then(($options) => {
        const matches = [...$options].filter((option) => option.dataset.runId === runId);
        expect(matches, "one unambiguous run option").to.have.length(1);
        expect(matches[0].value, "source-qualified selector").to.match(/^npa1_/);
      });
    };

    searchFromFreshPage();
    cy.clearAllLocalStorage();
    cy.clearAllSessionStorage();
    cy.then(searchFromFreshPage);
  });

  it("loads an explicitly selected RRD and paints it in the real Rerun viewer", () => {
    const runId = String(Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || "");
    const artifactKey = String(Cypress.env("NPA_AGENT_CYPRESS_ARTIFACT_KEY") || "");
    expect(runId, "NPA_AGENT_CYPRESS_RUN_ID").not.to.eq("");
    expect(artifactKey, "NPA_AGENT_CYPRESS_ARTIFACT_KEY").to.match(/\.rrd$/);

    let capabilityPath = "";
    resolveRrdSource(runId, artifactKey).then(({ entry }) => {
      const loadBody = {
        run_id: runId,
        run_ref: String(entry.run_ref || ""),
        key: artifactKey,
        project_id: String(entry.project_id || ""),
        resource_bucket: String(entry.bucket || ""),
        resolved_prefix: String(entry.resolved_prefix || ""),
        source_selected: true,
      };
      const loadUntilReady = (attempt) => {
        return liveAgentRequest("/api/sim-viz/load-artifact", {
          method: "POST",
          body: loadBody,
          timeout: 120000,
          failOnStatusCode: false,
        }).then((response) => {
          if (response.status !== 200 && attempt < 4) {
            return cy.wait(2000).then(() => loadUntilReady(attempt + 1));
          }
          expect(response.status, JSON.stringify(response.body)).to.eq(200);
          expect(response.body).to.have.property("ok", true);
          expect(response.body).to.have.property("render", "rerun");
          expect(String(response.body.artifact_uri || "")).to.include(artifactKey);
          capabilityPath = String((response.body.sim_viz && response.body.sim_viz.artifact_preview_url) || "");
          expect(capabilityPath).to.match(/^\/rerun\/recordings\/cap-[A-Za-z0-9_-]{43}\.rrd$/);
        });
      };
      return loadUntilReady(0);
    });

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
      const source = decodeURIComponent(String($frame.attr("src") || ""));
      expect(
        source.includes(capabilityPath) || source.includes("/api/sim-viz/rrd-blob"),
        "capability recording or authenticated blob fallback",
      ).to.eq(true);
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
