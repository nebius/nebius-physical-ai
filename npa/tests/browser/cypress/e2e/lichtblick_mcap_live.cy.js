import {
  decodePngStats,
  firstMcapPngPayload,
  mcapCameraTopicCount,
  mcapHasCompressedImage,
  mcapHasFrameTransform,
  mcapHasHeldoutCamera,
  mcapHasPointCloud,
  mcapPointCloudHasRgbaFields,
} from "../support/e2e";

const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];
const MCAP_RECORDING_PATH = "/lichtblick/recordings/sim2real.mcap";

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name) || Cypress.env(name.replace("NPA_AGENT_", "agent"))));
}

function preferredRunId() {
  return Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID") || "";
}

function agentReq(path, options = {}) {
  const baseUrl = Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl");
  const username = Cypress.env("agentUser") || Cypress.env("NPA_AGENT_USER");
  const password = Cypress.env("agentPassword") || Cypress.env("NPA_AGENT_PASSWORD");
  return cy.request({
    url: `${String(baseUrl || "").replace(/\/$/, "")}${path}`,
    auth: { username, password },
    ...options,
  });
}

// Discover the run to view (explicit env override, else the latest viewable run)
// and its MCAP artifact key.
function resolveMcapArtifact() {
  return agentReq("/api/artifacts/runs").then((resp) => {
    expect(resp.status, "artifacts/runs status").to.eq(200);
    const runs = (resp.body && resp.body.runs) || [];
    expect(runs.length, "discovered runs").to.be.greaterThan(0);
    const explicit = preferredRunId();
    const viewable = runs.filter((run) => run.has_viewable);
    const chosen = explicit ? runs.find((run) => run.run_id === explicit) : viewable[0] || runs[0];
    expect(chosen, "a viewable run to inspect").to.exist;
    return agentReq(`/api/artifacts/run/${chosen.run_id}`).then((detail) => {
      const artifacts = (detail.body && detail.body.artifacts) || [];
      const mcap = artifacts.find((item) => String(item.key || "").endsWith(".mcap"));
      expect(mcap, `run ${chosen.run_id} exposes an .mcap artifact`).to.exist;
      return cy.wrap({ runId: chosen.run_id, key: mcap.key, size: mcap.size }, { log: false });
    });
  });
}

function loadMcap(runId, key) {
  return agentReq("/api/sim-viz/load-artifact", {
    method: "POST",
    body: { run_id: runId, key },
  }).then((resp) => {
    expect(resp.status, "load-artifact status").to.eq(200);
    expect(String(resp.body.render || ""), "artifact render hint").to.eq("mcap");
  });
}

describe("Lichtblick MCAP viewer (live system)", () => {
  before(function () {
    if (!liveEnvAvailable()) {
      this.skip();
    }
  });

  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^\d+$/);
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");
  });

  it("discovers a viewable run that exposes an MCAP artifact", () => {
    resolveMcapArtifact().then(({ runId, key }) => {
      expect(runId, "chosen run id").to.be.a("string").and.not.be.empty;
      expect(key, "mcap artifact key").to.match(/\.mcap$/);
    });
  });

  it("loads the run's MCAP and co-serves a substantive recording", () => {
    resolveMcapArtifact().then(({ runId, key }) => {
      loadMcap(runId, key);
      agentReq(MCAP_RECORDING_PATH, { encoding: "binary", failOnStatusCode: false }).then((resp) => {
        expect([200, 206]).to.include(resp.status);
        const body = resp.body || "";
        // The non-substantive stub was ~114KB; a genuine run is multi-MB.
        expect(body.length, "served mcap bytes (not a stub)").to.be.greaterThan(1000000);
        expect(mcapHasCompressedImage(body), "has foxglove.CompressedImage").to.be.true;
        expect(mcapHasHeldoutCamera(body), "has /heldout/camera/ topics").to.be.true;
        expect(mcapCameraTopicCount(body), "camera topic occurrences").to.be.greaterThan(4);
        // GPU-reconstructed 3D point cloud for the 3D panel.
        expect(mcapHasPointCloud(body), "has foxglove.PointCloud on /heldout/points").to.be.true;
        // ...with the full RGBA field set the layout's rgba-fields mode requires
        // (without alpha the panel re-colours the cloud with a fallback colormap).
        expect(mcapPointCloudHasRgbaFields(body), "point cloud has red/green/blue/alpha").to.be
          .true;
        // A coordinate transform so the 3D panel can place the point cloud.
        expect(mcapHasFrameTransform(body), "has foxglove.FrameTransform on /tf").to.be.true;
      });
    });
  });

  it("injects a default layout so the 3D point cloud + camera show without setup", () => {
    // The embedded viewer document must carry the self-hosted default layout that
    // makes /heldout/points visible in the 3D panel and binds /camera to the Image
    // panel — otherwise Lichtblick opens with the point cloud hidden (empty 3D).
    agentReq("/lichtblick/", { failOnStatusCode: false }).then((resp) => {
      expect([200, 304]).to.include(resp.status);
      const html = String(resp.body || "");
      expect(html, "LICHTBLICK_SUITE_DEFAULT_LAYOUT global").to.include(
        "LICHTBLICK_SUITE_DEFAULT_LAYOUT",
      );
      expect(html, "placeholder replaced with a real layout").to.not.include(
        "LICHTBLICK_SUITE_DEFAULT_LAYOUT_PLACEHOLDER",
      );
      expect(html, "3D panel shows the point cloud topic").to.include("/heldout/points");
      expect(html, "Image panel bound to the camera").to.include('"imageTopic":"/camera"');
    });
  });

  it("renders a real (non-stub, non-noise) held-out camera frame", () => {
    resolveMcapArtifact().then(({ runId, key }) => {
      loadMcap(runId, key);
      agentReq(MCAP_RECORDING_PATH, { encoding: "binary", failOnStatusCode: false }).then((resp) => {
        const payload = firstMcapPngPayload(resp.body || "");
        expect(payload, "a PNG CompressedImage payload").to.be.a("string");
        return decodePngStats(payload).then((stats) => {
          expect(stats.width, "frame width (not a 32px stub)").to.be.greaterThan(64);
          expect(stats.height, "frame height (not a 32px stub)").to.be.greaterThan(64);
          // The PNG-row-filter bug produced dark noise (~40); real renders are ~165.
          expect(stats.mean, "frame brightness (not dark noise)").to.be.greaterThan(80);
          expect(stats.mean, "frame brightness (not saturated)").to.be.lessThan(250);
        });
      });
    });
  });

  it("embeds the Lichtblick viewer wired to the co-served recording", () => {
    resolveMcapArtifact().then(({ runId, key }) => {
      loadMcap(runId, key);
    });
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");
    cy.get("#lichtblickFrame")
      .invoke("attr", "src")
      .then((src) => {
        const url = new URL(String(src), "https://placeholder.invalid");
        const ds = decodeURIComponent(url.searchParams.get("ds.url") || "");
        expect(ds).to.include("/lichtblick/recordings/sim2real.mcap");
      });
    cy.get("#openLichtblick").should("be.visible");

    // Backend status surfaces the Lichtblick embed fields.
    agentReq("/api/sim-viz/status").then((resp) => {
      expect(resp.status).to.eq(200);
      const viz = resp.body || {};
      expect(decodeURIComponent(String(viz.lichtblick_iframe_url || ""))).to.include("/lichtblick/");
      expect(viz).to.have.property("lichtblick_ready");
    });
  });
});
