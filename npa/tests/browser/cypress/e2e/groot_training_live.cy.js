const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name)));
}

function runId() {
  return Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID") || "";
}

function agentReq(path, options = {}) {
  const baseUrl = Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl");
  return cy.request({
    url: `${String(baseUrl).replace(/\/$/, "")}${path}`,
    auth: {
      username: Cypress.env("NPA_AGENT_USER"),
      password: Cypress.env("NPA_AGENT_PASSWORD"),
    },
    ...options,
  });
}

function resolveRecording(suffix) {
  const id = runId();
  expect(id, "explicit live GR00T run id").to.not.equal("");
  return agentReq(`/api/artifacts/run/${id}`).then((resp) => {
    expect(resp.status).to.eq(200);
    const artifacts = (resp.body && resp.body.artifacts) || [];
    const artifact = artifacts.find((item) => String(item.key || "").endsWith(suffix));
    expect(artifact, `${suffix} artifact for ${id}`).to.exist;
    return cy.wrap({ id, artifact }, { log: false });
  });
}

function assertViewerDocument(frame, label) {
  const iframe = frame[0];
  const doc = iframe.contentDocument || (iframe.contentWindow && iframe.contentWindow.document);
  expect(doc, `${label} same-origin document`).to.exist;
  expect(doc.readyState, `${label} document loaded`).to.eq("complete");
  const text = String((doc.body && doc.body.innerText) || "");
  expect(text, `${label} has no load failure`).not.to.match(/failed to load|unable to load|application error/i);
  const canvases = [...doc.querySelectorAll("canvas")];
  expect(canvases.length, `${label} rendered canvas count`).to.be.greaterThan(0);
  expect(
    canvases.some((canvas) => canvas.width > 100 && canvas.height > 100),
    `${label} has a substantive render surface`,
  ).to.eq(true);
}

describe("GR00T training RRD and MCAP against the live agent", () => {
  before(function () {
    if (!liveEnvAvailable() || !runId()) this.skip();
  });

  it("loads the exact RRD and renders it in the real Rerun browser viewer", () => {
    resolveRecording("/reports/groot-training.rrd").then(({ id, artifact }) => {
      agentReq("/api/sim-viz/load-artifact", {
        method: "POST",
        body: { run_id: id, key: artifact.key },
      }).then((resp) => {
        expect(resp.status).to.eq(200);
        expect(resp.body.render).to.eq("rerun");
      });
      agentReq(`/api/sim-viz/status?run_id=${encodeURIComponent(id)}`).then((resp) => {
        const viz = resp.body || {};
        expect(viz.run_id).to.eq(id);
        expect(viz.active_run_id).to.eq(id);
        expect(viz.rerun_ready).to.eq(true);
        expect(viz.camera).to.eq("camera");
        expect(viz.preview_entity).to.eq("camera");
        expect(String(viz.visualization_note || "")).to.include("dataset/synthetic-fps");
        expect(String(viz.visualization_note || "")).to.include("not a policy rollout evaluation");
      });
      agentReq(`/api/sim-viz/rrd-blob?run_id=${encodeURIComponent(id)}`, {
        encoding: "binary",
      }).then((resp) => {
        expect(resp.status).to.eq(200);
        expect(resp.body.length).to.eq(artifact.size);
      });
    });

    cy.visitLiveAgent();
    cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");
    cy.get("#tabRerun").click();
    cy.get("#renderModeRerun").click();
    cy.get("#rerunFrame", { timeout: 120000 })
      .should("have.attr", "src")
      .and("include", "/rerun/");
    cy.get("#rerunFrame", { timeout: 120000 }).should(($frame) => {
      assertViewerDocument($frame, "Rerun");
    });
  });

  it("loads MCAP through Foxglove, proves range access, and renders Lichtblick", () => {
    resolveRecording("/reports/groot-training.mcap").then(({ id, artifact }) => {
      agentReq("/api/foxglove/load-artifact", {
        method: "POST",
        body: { run_id: id, key: artifact.key },
      }).then((resp) => {
        expect(resp.status).to.eq(200);
        expect(resp.body.render).to.eq("mcap");
        expect(resp.body.foxglove.available).to.eq(true);
      });
      agentReq("/api/sim-viz/status").then((resp) => {
        const viz = resp.body || {};
        expect(viz.run_id).to.eq(id);
        expect(viz.foxglove_ready).to.eq(true);
        expect(viz.lichtblick_ready).to.eq(true);
        expect(String(viz.visualization_note || "")).to.include("not a policy rollout");
        agentReq(viz.foxglove_url, {
          encoding: "binary",
          headers: { Range: "bytes=0-63" },
        }).then((rangeResp) => {
          expect(rangeResp.status).to.eq(206);
          expect(rangeResp.headers).to.have.property("content-range");
          expect(rangeResp.body.length).to.eq(64);
          expect(rangeResp.body.slice(0, 8), "MCAP magic").to.eq("\u0089MCAP0\r\n");
        });
      });
      agentReq("/lichtblick/recordings/sim2real.mcap", { encoding: "binary" }).then((resp) => {
        expect(resp.status).to.eq(200);
        expect(resp.body.length).to.eq(artifact.size);
        expect(resp.body).to.include("foxglove.CompressedImage");
        expect(resp.body).to.include("/metrics/loss");
        expect(resp.body).to.include("/metrics/world_size");
      });
    });

    cy.intercept("GET", "/lichtblick/recordings/sim2real.mcap*").as("mcapPlayback");
    cy.visitLiveAgent();
    cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#lichtblickFrame", { timeout: 120000 })
      .should("have.attr", "src")
      .and("include", "/lichtblick/");
    cy.wait("@mcapPlayback", { timeout: 120000 }).then((interception) => {
      expect([200, 206]).to.include(interception.response && interception.response.statusCode);
    });
    cy.get("#lichtblickFrame", { timeout: 120000 }).should(($frame) => {
      assertViewerDocument($frame, "Lichtblick");
    });
  });
});
