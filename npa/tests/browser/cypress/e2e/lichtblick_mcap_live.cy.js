import {
  mcapCameraTopicCount,
  mcapHasCompressedImage,
  mcapHasFrameTransform,
  mcapHasHeldoutCamera,
  mcapHasPointCloud,
  mcapPointCloudHasRgbaFields,
  currentLiveAgentConfig,
} from "../support/e2e";
const MCAP_RECORDING_PATH = "/lichtblick/recordings/sim2real.mcap";
const LICHTBLICK_COMPOSITE_SCREENSHOT =
  "cypress/screenshots/lichtblick_mcap_live.cy.js/lichtblick-live-camera-composite.png";

function compositeCameraStats(base64Png) {
  return new Cypress.Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      // The injected layout places the Image panel to the right of the Topics
      // sidebar. Sample its interior, excluding toolbars and the timeline. This
      // reads Chromium's composited screenshot rather than the WebGL backing
      // buffer, which is cleared after paint when preserveDrawingBuffer=false.
      // Sample only the first Image panel. The adjacent 3D panel's blue grid is
      // intentionally excluded so it cannot make a waiting/blank camera pass.
      const x = Math.floor(image.width * 0.35);
      const y = Math.floor(image.height * 0.10);
      const width = Math.max(1, Math.floor(image.width * 0.30));
      const height = Math.max(1, Math.floor(image.height * 0.34));
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, x, y, width, height, 0, 0, width, height);
      const pixels = context.getImageData(0, 0, width, height).data;
      let vivid = 0;
      let sum = 0;
      let sumSquared = 0;
      let count = 0;
      for (let offset = 0; offset < pixels.length; offset += 16) {
        const red = pixels[offset];
        const green = pixels[offset + 1];
        const blue = pixels[offset + 2];
        const value = (red + green + blue) / 3;
        const chroma = Math.max(red, green, blue) - Math.min(red, green, blue);
        if (chroma > 40 && Math.max(red, green, blue) > 80) vivid += 1;
        sum += value;
        sumSquared += value * value;
        count += 1;
      }
      const mean = sum / count;
      resolve({ vivid, variance: Math.max(0, sumSquared / count - mean * mean) });
    };
    image.onerror = reject;
    image.src = `data:image/png;base64,${base64Png}`;
  });
}

function preferredRunId() {
  return (
    Cypress.env("NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID") ||
    Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") ||
    Cypress.env("NPA_AGENT_RUN_ID") ||
    ""
  );
}

function agentReq(path, options = {}) {
  const { baseUrl, username, password } = currentLiveAgentConfig();
  return cy.request({
    url: `${String(baseUrl || "").replace(/\/$/, "")}${path}`,
    auth: { username, password },
    ...options,
  });
}

// Discover the run to view (explicit env override, else the latest viewable run)
// and its MCAP artifact key.
function resolveMcapArtifact() {
  return agentReq("/api/artifacts/runs?limit=2000", { timeout: 120000 }).then((resp) => {
    expect(resp.status, "artifacts/runs status").to.eq(200);
    const runs = (resp.body && resp.body.runs) || [];
    expect(runs.length, "discovered runs").to.be.greaterThan(0);
    const explicit = preferredRunId();
    const viewable = runs.filter((run) => run.has_viewable);
    const candidates = [
      ...(explicit ? [runs.find((run) => run.run_id === explicit) || { run_id: explicit }] : []),
      ...viewable,
      ...runs,
    ].filter((run, index, all) => {
      const identity = String(run.run_ref || run.run_id || "");
      return all.findIndex((item) => String(item.run_ref || item.run_id || "") === identity) === index;
    });
    const findSubstantiveMcap = (index) => {
      if (index >= candidates.length) return cy.wrap(null, { log: false });
      const chosen = candidates[index];
      const selector = String(chosen.run_ref || chosen.run_id || "");
      return agentReq(`/api/artifacts/run/${encodeURIComponent(selector)}`, { timeout: 120000 }).then((detail) => {
        const artifacts = (detail.body && detail.body.artifacts) || [];
        const minimumBytes = explicit && chosen.run_id === explicit ? 1024 : 1000000;
        const mcap = artifacts.find(
          (item) => String(item.key || "").endsWith(".mcap") && Number(item.size || 0) > minimumBytes,
        );
        return mcap
          ? { runId: chosen.run_id, runRef: String(chosen.run_ref || ""), key: mcap.key, size: mcap.size }
          : findSubstantiveMcap(index + 1);
      });
    };
    return findSubstantiveMcap(0).then((found) => {
      expect(found, "a substantive canonical MCAP artifact").to.exist;
      return cy.wrap(found, { log: false });
    });
  });
}

function loadMcap(runId, runRef, key) {
  return agentReq("/api/sim-viz/load-artifact", {
    method: "POST",
    body: { run_id: runId, run_ref: runRef, key },
  }).then((resp) => {
    expect(resp.status, "load-artifact status").to.eq(200);
    expect(String(resp.body.render || ""), "artifact render hint").to.eq("mcap");
  });
}

describe("Lichtblick MCAP viewer (live system)", () => {
  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^\d+$/);
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
  });

  it("discovers a viewable run that exposes an MCAP artifact", () => {
    resolveMcapArtifact().then(({ runId, key }) => {
      expect(runId, "chosen run id").to.be.a("string").and.not.be.empty;
      expect(key, "mcap artifact key").to.match(/\.mcap$/);
    });
  });

  it("loads the run's MCAP and co-serves a substantive recording", () => {
    resolveMcapArtifact().then(({ runId, runRef, key }) => {
      loadMcap(runId, runRef, key);
      agentReq(MCAP_RECORDING_PATH, { encoding: "binary", failOnStatusCode: false }).then((resp) => {
        expect([200, 206]).to.include(resp.status);
        const body = resp.body || "";
        expect(body.length, "served mcap bytes (not a stub)").to.be.greaterThan(1024);
        expect(mcapHasCompressedImage(body), "has foxglove.CompressedImage").to.be.true;
        expect(mcapCameraTopicCount(body), "camera topic occurrences").to.be.greaterThan(0);
        // Native representative recordings may carry real 3D data. When they
        // do, prove its RGBA fields and transform survived byte-for-byte reuse;
        // generated exports must not fabricate either merely for the viewer.
        if (mcapHasPointCloud(body)) {
          expect(mcapPointCloudHasRgbaFields(body), "point cloud has red/green/blue/alpha").to.be
            .true;
          expect(mcapHasFrameTransform(body), "point cloud has its real transform").to.be.true;
        }
        if (mcapHasHeldoutCamera(body)) {
          expect(body).to.include("/heldout/camera/");
        }
      });
    });
  });

  it("injects the canonical 3D and real-camera layout", () => {
    agentReq("/lichtblick/", { failOnStatusCode: false }).then((resp) => {
      expect([200, 304]).to.include(resp.status);
      const html = String(resp.body || "");
      expect(html, "LICHTBLICK_SUITE_DEFAULT_LAYOUT global").to.include(
        "LICHTBLICK_SUITE_DEFAULT_LAYOUT",
      );
      expect(html, "placeholder replaced with a real layout").to.not.include(
        "LICHTBLICK_SUITE_DEFAULT_LAYOUT_PLACEHOLDER",
      );
      expect(html, "Image panel bound to the camera").to.include('"imageTopic":"/camera"');
      expect(html, "second synchronized camera panel").to.include(
        '"imageTopic":"/camera/workspace"',
      );
      expect(html, "3D panel bound to the real trajectory").to.include("/trajectory");
      expect(html, "Plot bound to real execution metrics").to.include(
        "/metrics/execution.reward",
      );
    });
  });

  it("renders the canonical real camera in Lichtblick", () => {
    resolveMcapArtifact().then(({ runId, runRef, key }) => {
      loadMcap(runId, runRef, key);
      cy.get("#tabRerun").click();
      cy.get("#renderModeLichtblick").click();
      cy.window({ timeout: 90000 }).then({ timeout: 90000 }, async (win) => {
        const frame = win.document.getElementById("lichtblickFrame");
        const deadline = Date.now() + 60000;
        let viewerText = "";
        while (Date.now() < deadline) {
          const doc = frame && (frame.contentDocument || (frame.contentWindow && frame.contentWindow.document));
          viewerText = String((doc && doc.body && doc.body.innerText) || "");
          const canvas = doc && doc.querySelector("canvas");
          if (
            canvas && canvas.width > 100 && canvas.height > 100
            && viewerText.includes("/camera")
            && !viewerText.includes("Image topic does not exist")
            && !viewerText.includes("Waiting for image messages")
          ) break;
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
        expect(viewerText).not.to.include("No data source");
        expect(viewerText).not.to.include("Image topic does not exist");
        expect(viewerText).not.to.include("Waiting for image messages");
        expect(viewerText, "active real camera topic").to.include("/camera");
      });
      cy.get("#lichtblickFrame").screenshot("lichtblick-live-camera-composite", { overwrite: true });
      cy.readFile(LICHTBLICK_COMPOSITE_SCREENSHOT, "base64").then((base64Png) =>
        compositeCameraStats(base64Png).then((stats) => {
          expect(stats.vivid, "composited camera image has colored scene pixels").to.be.greaterThan(10);
          expect(stats.variance, "composited camera image is not an empty panel").to.be.greaterThan(100);
        }),
      );
    });
  });

  it("embeds the Lichtblick viewer wired to the co-served recording", () => {
    cy.intercept({ method: "GET", pathname: MCAP_RECORDING_PATH }, (request) => {
      if (request.headers.range) request.alias = "lichtblickMcapRange";
    });
    resolveMcapArtifact()
      .then(({ runId, runRef, key }) => loadMcap(runId, runRef, key))
      .then(() => cy.reload());
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");
    cy.get("#lichtblickFrame")
      .invoke("attr", "src")
      .then((src) => {
        const url = new URL(String(src), "https://placeholder.invalid");
        expect(url.pathname).to.include("/lichtblick/");
      });
    cy.wait("@lichtblickMcapRange", { timeout: 60000 }).then(({ request, response }) => {
      expect(request.headers, "Lichtblick recording request headers").to.have.property("range");
      expect(response && response.statusCode, "Lichtblick range response").to.eq(206);
    });
    cy.get("#lichtblickFrame", { timeout: 60000 }).should(($frame) => {
      const text = String($frame[0].contentDocument && $frame[0].contentDocument.body.innerText || "");
      expect(text).not.to.include("No data source");
      expect(text).not.to.include("Image topic does not exist");
    });
    cy.get("#renderModeLichtblick").should("have.length", 1);

    // Backend status surfaces the Lichtblick embed fields.
    agentReq("/api/sim-viz/status").then((resp) => {
      expect(resp.status).to.eq(200);
      const viz = resp.body || {};
      const iframeUrl = new URL(
        String(viz.lichtblick_iframe_url || ""),
        "https://placeholder.invalid",
      );
      expect(iframeUrl.pathname).to.include("/lichtblick/");
      expect(decodeURIComponent(iframeUrl.searchParams.get("ds.url") || "")).to.include(
        "/lichtblick/recordings/sim2real.mcap",
      );
      expect(viz).to.have.property("lichtblick_ready");
    });
  });
});
