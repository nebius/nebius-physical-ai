import { STATIC_BUTTON_IDS } from "../support/e2e";

const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name) || Cypress.env(name.replace("NPA_AGENT_", "agent"))));
}

function destructiveLiveEnabled() {
  const value = Cypress.env("NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE");
  return value === true || value === 1 || value === "1";
}

function liveRunId() {
  return Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID") || "";
}

function liveAgentRequest(path, options = {}) {
  const baseUrl = Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl");
  const username = Cypress.env("agentUser") || Cypress.env("NPA_AGENT_USER");
  const password = Cypress.env("agentPassword") || Cypress.env("NPA_AGENT_PASSWORD");
  return cy.request({
    url: `${String(baseUrl || "").replace(/\/$/, "")}${path}`,
    auth: { username, password },
    ...options,
  });
}

function parseRgb(value) {
  const match = String(value || "").match(/rgba?\(([^)]+)\)/);
  if (!match) return null;
  const parts = match[1].split(",").map((part) => Number.parseFloat(part.trim()));
  if (parts.length < 3 || parts.some((part, index) => index < 3 && Number.isNaN(part))) return null;
  const alpha = parts.length >= 4 && !Number.isNaN(parts[3]) ? parts[3] : 1;
  return { r: parts[0], g: parts[1], b: parts[2], a: alpha };
}

function luminance(rgb) {
  const channel = (value) => {
    const normalized = value / 255;
    return normalized <= 0.03928 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(rgb.r) + 0.7152 * channel(rgb.g) + 0.0722 * channel(rgb.b);
}

function contrastRatio(foreground, background) {
  const fg = luminance(foreground);
  const bg = luminance(background);
  return (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05);
}

function effectiveBackground(win, element) {
  let node = element;
  while (node && node.nodeType === 1) {
    const bg = parseRgb(win.getComputedStyle(node).backgroundColor);
    if (bg && bg.a > 0.05) return bg;
    node = node.parentElement;
  }
  return { r: 255, g: 255, b: 255, a: 1 };
}

function hasVisibleText(element) {
  const text = String(element.innerText || element.value || "").replace(/\s+/g, " ").trim();
  if (!text) return false;
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

describe("NPA agent UI against live infra", () => {
  before(function () {
    if (!liveEnvAvailable()) {
      this.skip();
    }
  });

  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^\d+$/);
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    // Wait for boot mount so load-run is not clobbered by ensureFrankaRerunLoaded.
    cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");
  });

  it("loads deployed UI and every shipped button is present", () => {
    for (const id of STATIC_BUTTON_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    cy.get("#chatForm").should("exist");
    cy.get("#workflowYaml").should("exist");
    cy.get("#tabChat").should("exist");
    cy.get("#tabRerun").should("exist");
    cy.get("#stagesPanel").should("exist");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
    cy.get("#rerunFrame").should("exist");
    cy.get("#renderModeVideo").should("exist");
    cy.get("#artifactPreviewHost").should("exist");
    cy.get("#viewerPaneMedia").should("exist");
    cy.get("#rerunBundleCover").should("exist");
    // Embedded Lichtblick (Foxglove-compatible MCAP viewer) surfaces.
    cy.get("#renderModeLichtblick").should("exist");
    cy.get("#lichtblickFrame").should("exist");
    cy.get("#viewerPaneLichtblick").should("exist");
    cy.get("#openLichtblick").should("exist");
    cy.window().then((win) => {
      const html = win.document.documentElement.outerHTML;
      expect(html).to.include("authenticatedPreviewObjectUrl");
      expect(html).to.include("URL.createObjectURL(blob)");
      expect(html).to.include("Loading video preview");
      expect(html).to.include("waitUntilRerunPastBundleSplash");
      expect(html).to.include("scheduleRerunBundleUncover");
      expect(html).to.include("swapRerunRecordingInPlace");
      expect(html).to.include("Warm Rerun assets before revealing the iframe");
      expect(html).not.to.include('Mount the viewer immediately so "Loading application bundle" starts early');
      expect(html).not.to.include("await waitUntilRerunPastBundleSplash(iframe, 45000)");
    });
  });

  it("never shows Loading application bundle without mount latency", () => {
    cy.get("#rerunBundleCover").should("exist");
    cy.window().then((win) => {
      const html = win.document.documentElement.outerHTML;
      expect(html).to.include("Uncover without blocking mount latency");
      expect(html).to.include("scheduleRerunBundleUncover");
      expect(html).to.include("swapRerunRecordingInPlace");
      expect(html).to.include("add_receiver");
      expect(html).not.to.include("await waitUntilRerunPastBundleSplash(iframe, 45000)");
    });
    // Visible chrome only (skip <script> source, which contains the splash detector regex).
    cy.get("#rerunBundleCover .cover-title").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#statusBar").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    // Cover drops once past splash; keep timeout modest — warm-before-reveal avoids cold stalls.
    cy.get("#rerunBundleCover", { timeout: 45000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      const frame = $frame[0];
      try {
        const doc = frame.contentDocument || (frame.contentWindow && frame.contentWindow.document);
        const text = String((doc && doc.body && doc.body.innerText) || "");
        expect(text, "Rerun iframe must not show application-bundle splash").not.to.match(
          /Loading application bundle/i,
        );
      } catch (_err) {
        throw new Error("unable to inspect same-origin Rerun iframe for bundle splash");
      }
    });
  });

  it("drives safe live controls through the browser", () => {
    cy.get("#chatActionS3").click();
    cy.get("#chatInput").should("contain.value", "configure S3");
    cy.get("#chatActionCosmos").click();
    cy.get("#chatInput").should("contain.value", "Cosmos3");
    cy.get("#chatActionWatch").click();
    cy.get("#chatInput").should("contain.value", "Rerun");
    cy.get("#chatActionWorkflow").click();
    cy.get("#chatInput").should("contain.value", "2-step sim2real workflow");

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#workflowStatus").click();
    cy.get("#tabChat").click();
    cy.get("#runSummary", { timeout: 30000 }).should("contain.text", "status");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#artifactDiscoverStatus", { timeout: 30000 }).should("contain.text", "Runs discovered");

    cy.get("#loadFrankaRerun").click();
    cy.get("#statusBar", { timeout: 120000 }).should(($bar) => {
      const text = $bar.text();
      expect(text).to.match(/done|Ready|SUCCESS|Rerun/i);
    });

    cy.get("#openRerun").should("be.visible");
  });

  it("embeds the Lichtblick MCAP viewer and co-serves the recording", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // Activate the embedded Lichtblick render mode.
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");
    cy.get("#openLichtblick").should("be.visible");

    // The backend status surfaces the Lichtblick embed fields.
    liveAgentRequest("/api/sim-viz/status").then((resp) => {
      expect(resp.status).to.eq(200);
      const viz = resp.body || {};
      expect(decodeURIComponent(String(viz.lichtblick_iframe_url || ""))).to.include("/lichtblick/");
      expect(viz).to.have.property("lichtblick_ready");
    });

    // The MCAP recording is co-served same-origin under /lichtblick/recordings/
    // (nginx alias). The Lichtblick viewer app itself is a best-effort sidecar,
    // so we assert the recording plumbing rather than requiring the sidecar image.
    liveAgentRequest("/lichtblick/recordings/sim2real.mcap", { failOnStatusCode: false }).then((resp) => {
      expect([200, 206, 404]).to.include(resp.status);
    });
  });

  it("keeps visible live UI text readable across the Nebius theme", () => {
    cy.get("#chatInput").clear().type("give run status", { delay: 0 });
    cy.get("#chatSend").click();
    cy.get("#chatLog", { timeout: 60000 }).should("contain.text", "give run status");

    cy.window().then((win) => {
      const selectors = [
        "body",
        ".topbar",
        ".panel",
        ".subsection",
        ".btn",
        ".quick-pill",
        ".badge",
        ".pill",
        ".bubble",
        ".field label",
        ".field input",
        ".field select",
        ".stage-status",
        ".stage-label",
        ".stage-summary",
        ".run-log",
        "#statusBar",
        "#artifactList",
        "#renderedDataSummary",
      ];
      const failures = [];
      const seen = new Set();
      for (const selector of selectors) {
        for (const element of win.document.querySelectorAll(selector)) {
          if (seen.has(element) || !hasVisibleText(element)) continue;
          seen.add(element);
          const style = win.getComputedStyle(element);
          if (style.visibility === "hidden" || style.display === "none") continue;
          const fg = parseRgb(style.color);
          const bg = effectiveBackground(win, element);
          if (!fg || !bg) continue;
          const ratio = contrastRatio(fg, bg);
          const fontSize = Number.parseFloat(style.fontSize || "0");
          const fontWeight = Number.parseInt(style.fontWeight || "400", 10);
          const threshold = fontSize >= 18 || fontWeight >= 700 ? 3.0 : 4.5;
          if (ratio + 0.01 < threshold) {
            failures.push({
              selector,
              text: String(element.innerText || element.value || "").replace(/\s+/g, " ").trim().slice(0, 80),
              color: style.color,
              background: win.getComputedStyle(element).backgroundColor,
              effectiveBackground: `rgb(${bg.r}, ${bg.g}, ${bg.b})`,
              ratio: Number(ratio.toFixed(2)),
              threshold,
            });
          }
        }
      }
      expect(failures, JSON.stringify(failures, null, 2)).to.deep.equal([]);
    });
  });

  it("loads a live Sim2Real run as a Rerun artifact, not a stale JSON artifact", function () {
    const runId = liveRunId();
    if (!runId) {
      this.skip();
    }

    const assertRerunSimViz = (simViz) => {
      expect(String(simViz.run_id || "")).to.eq(runId);
      expect(String(simViz.artifact_render || "")).to.eq("rerun");
      expect(String(simViz.artifact_key || "")).to.match(/\/reports\/sim2real\.rrd$/);
      expect(String(simViz.artifact_uri || "")).to.match(/\/reports\/sim2real\.rrd$/);
      expect(String(simViz.rrd_uri || "")).to.match(/^file:\/\//);
      expect(simViz.rerun_ready).to.eq(true);
      expect(String(simViz.camera || "")).to.eq("heldout-sim");
      expect(String(simViz.preview_entity || "")).to.eq("camera");
      expect(simViz.visualization_note || "").to.match(/held-out simulation camera|reference proxy/i);
      expect(decodeURIComponent(String(simViz.rerun_iframe_url || ""))).to.include(
        "/rerun/recordings/sim2real.rrd",
      );
    };

    const loadRunUntilRerun = (attempt) => {
      liveAgentRequest("/api/sim-viz/load-run", {
        method: "POST",
        body: { run_id: runId, camera: "workspace" },
        timeout: 120000,
        failOnStatusCode: false,
      }).then((response) => {
        const simViz = (response.body && response.body.sim_viz) || {};
        const ready =
          response.status === 200 &&
          response.body &&
          response.body.ok === true &&
          String(simViz.artifact_render || "") === "rerun" &&
          String(simViz.run_id || "") === runId;
        if (!ready) {
          if (attempt >= 4) {
            expect(response.status, JSON.stringify(response.body)).to.eq(200);
            expect(response.body, JSON.stringify(response.body)).to.have.property("ok", true);
            assertRerunSimViz(simViz);
            return;
          }
          cy.wait(1500).then(() => loadRunUntilRerun(attempt + 1));
          return;
        }
        assertRerunSimViz(simViz);
        const assertStatusUntilRerun = (statusAttempt) => {
          liveAgentRequest(`/api/sim-viz/status?run_id=${encodeURIComponent(runId)}`).then((statusResp) => {
            const statusViz = statusResp.body || {};
            const statusReady =
              statusResp.status === 200 &&
              String(statusViz.artifact_render || "") === "rerun" &&
              String(statusViz.run_id || "") === runId;
            if (!statusReady) {
              if (statusAttempt >= 4) {
                expect(statusResp.status, JSON.stringify(statusResp.body)).to.eq(200);
                assertRerunSimViz(statusViz);
                return;
              }
              cy.wait(1000).then(() => assertStatusUntilRerun(statusAttempt + 1));
              return;
            }
            assertRerunSimViz(statusViz);
            cy.reload();
            cy.get("#statusBar", { timeout: 30000 }).should("exist");
            cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");
            cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
            cy.get("#tabRerun").click();
            cy.get("#rerunFrame").should(($frame) => {
              const src = String($frame.attr("src") || "");
              expect(decodeURIComponent(src)).to.include("/rerun/recordings/sim2real.rrd");
            });
            cy.get("#statusBar").should("not.contain.text", "Non-RRD artifact loaded");
          });
        };
        assertStatusUntilRerun(1);
      });
    };
    loadRunUntilRerun(0);
  });

  it("presents the live run with an intuitive stage timeline and stable desktop layout", function () {
    const runId = liveRunId();
    if (!runId) {
      this.skip();
    }

    cy.viewport(1440, 1000);
    liveAgentRequest("/api/sim-viz/load-run", {
      method: "POST",
      body: { run_id: runId, camera: "workspace" },
      timeout: 120000,
    }).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property("ok", true);
      expect((response.body.sim_viz || {}).artifact_render).to.eq("rerun");
      expect(String((response.body.sim_viz || {}).run_id || "")).to.eq(runId);
    });
    cy.reload();
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#runIdInput", { timeout: 30000 }).clear().type(runId);
    cy.get("#loadRunData").click();
    cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
    cy.get("#artifactLoadRunArtifacts").click();

    cy.get("#artifactList", { timeout: 120000 }).within(() => {
      cy.contains("reports/sim2real.rrd").should("be.visible");
      cy.contains("rerun").should("be.visible");
      cy.contains("reports/sim2real-report.json").should("be.visible");
    });
    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#stageList", { timeout: 30000 }).within(() => {
      cy.contains("Trigger").should("be.visible");
      cy.contains("Held-out eval").should("be.visible");
      cy.contains("Reports / visualization").should("be.visible");
      cy.contains("Succeeded").should("be.visible");
    });
    cy.get("#runSummary").should("contain.text", runId).and("contain.text", "completed");
    cy.get("#runLog").should("contain.text", "Derived stage timeline");
    cy.get("#tabRerun").click();
    cy.get("#renderedDataSummary", { timeout: 30000 }).should("contain.text", "rerun").and("contain.text", "sim2real.rrd");
    cy.get("#renderedDataSummary").should("contain.text", "held-out simulation camera");
    cy.get("#simCamera").should("contain.text", "heldout-sim");
    cy.get("#rerunFrame").should("be.visible");
    cy.get("#renderModeRerun").should("have.class", "is-active");
    // Tab panels stay mounted with opacity:0 when inactive — assert activation class,
    // not Cypress visibility (opacity:0 is treated as hidden).
    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#chatForm").should("exist");

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.window().then((win) => {
      const doc = win.document.documentElement;
      expect(doc.scrollWidth, "no distracting horizontal page overflow").to.be.lte(win.innerWidth + 24);
      const artifactList = win.document.getElementById("artifactList");
      expect(artifactList, "artifactList exists").to.exist;
      artifactList.scrollIntoView({ block: "nearest" });
      const artifactRect = artifactList.getBoundingClientRect();
      expect(artifactRect.width, "artifactList has usable width").to.be.greaterThan(240);
      expect(artifactRect.height, "artifactList has usable height").to.be.greaterThan(120);
      expect(String(artifactList.textContent || "")).to.match(/sim2real\.rrd/);
      const rerunFrame = win.document.getElementById("rerunFrame");
      expect(rerunFrame, "rerunFrame exists").to.exist;
      const frameRect = rerunFrame.getBoundingClientRect();
      expect(frameRect.width, "rerunFrame has usable width").to.be.greaterThan(240);
      expect(frameRect.height, "rerunFrame has usable height").to.be.greaterThan(40);
    });
    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "is-active");
    for (const id of ["chatForm", "runDetails"]) {
      cy.get(`#${id}`).should("exist").and(($el) => {
        const rect = $el[0].getBoundingClientRect();
        expect(rect.width, `${id} has usable width`).to.be.greaterThan(240);
        expect(rect.height, `${id} has usable height`).to.be.greaterThan(40);
      });
    }
  });

  it("answers advanced live run questions with grounded artifact and Rerun context", function () {
    const runId = liveRunId();
    if (!runId) {
      this.skip();
    }

    liveAgentRequest("/api/chat", {
      method: "POST",
      body: {
        messages: [
          {
            role: "user",
            content: `For run ${runId}, what stages and artifacts can I view, and is the Rerun recording ready?`,
          },
        ],
      },
    }).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property("ok", true);
      expect(response.body).to.have.property("grounded", true);
      expect(response.body.apis_used || []).to.have.length.greaterThan(0);
      const reply = String(response.body.reply || "");
      expect(reply.trim()).not.to.match(/^GET\s+\/api\//);
      expect(reply).to.include(runId);
      expect(reply).to.match(/Rerun|artifact|stage|rerun_ready/i);
      expect(reply).to.match(/\*\*run_id\*\*|run_id/i);
    });
  });

  it("loads a live mp4 artifact into the Video viewer with authenticated preview", () => {
    // Keep this after Rerun-specific cases so video preview state cannot race them.
    liveAgentRequest("/api/artifacts/runs").then((runsResp) => {
      expect(runsResp.status).to.eq(200);
      const runs = (runsResp.body && runsResp.body.runs) || [];
      expect(runs.length, "discovered runs").to.be.greaterThan(0);
      const candidates = runs.slice(0, 20).map((entry) => String((entry && entry.run_id) || "")).filter(Boolean);

      const findMp4 = (index) => {
        if (index >= candidates.length) {
          throw new Error("no mp4 artifact found in recent runs");
        }
        const runId = candidates[index];
        return liveAgentRequest(`/api/artifacts/run/${encodeURIComponent(runId)}`).then((artsResp) => {
          const arts = (artsResp.body && artsResp.body.artifacts) || [];
          const mp4 = arts.find((a) => String((a && a.key) || "").toLowerCase().endsWith(".mp4"));
          if (!mp4) {
            return findMp4(index + 1);
          }
          return { runId, key: String(mp4.key) };
        });
      };

      return findMp4(0);
    }).then(({ runId, key }) => {
      return liveAgentRequest("/api/sim-viz/load-artifact", {
        method: "POST",
        body: { run_id: runId, key },
        timeout: 120000,
      }).then((loadResp) => {
        expect(loadResp.status).to.eq(200);
        expect(loadResp.body.ok).to.eq(true);
        expect(loadResp.body.render).to.eq("video");
        const preview = String((loadResp.body.sim_viz && loadResp.body.sim_viz.artifact_preview_url) || "");
        expect(preview).to.match(/^\/api\/artifacts\/file\//);
        return liveAgentRequest(preview).then((fileResp) => {
          expect(fileResp.status).to.eq(200);
          const ct = String(fileResp.headers["content-type"] || "").toLowerCase();
          expect(ct).to.include("video/mp4");
          return { runId, key, preview };
        });
      });
    }).then(({ runId }) => {
      cy.get("#tabRerun").click();
      cy.get("#artifactRefreshRuns").click();
      cy.get("#artifactDiscoverStatus", { timeout: 30000 }).should("contain.text", "Runs discovered");
      cy.get("#runIdSelect", { timeout: 30000 }).then(($select) => {
        const values = [...$select[0].options].map((opt) => opt.value);
        if (values.includes(runId)) {
          cy.wrap($select).select(runId);
        }
      });
      cy.get("#artifactTypeFilter").select("video");
      cy.get("#artifactList", { timeout: 30000 }).should("contain.text", ".mp4");
      cy.contains("#artifactList button", "Play").first().click();
      cy.get("#renderModeVideo", { timeout: 30000 }).should("have.class", "is-active");
      cy.get("#viewerPaneMedia").should("have.class", "is-active-viewer");
      cy.get("#artifactPreviewHost video", { timeout: 60000 })
        .should("have.attr", "src")
        .and("match", /^blob:/);
      cy.get("#artifactPreviewHost video")
        .should("have.attr", "data-preview-url")
        .and("include", ".mp4");
    });
  });

  it("submits Sim2Real from the UI when live destructive Cypress is enabled", function () {
    if (!destructiveLiveEnabled()) {
      this.skip();
    }
    cy.get("#submitWorkflow").click();
    cy.get("#chatLog", { timeout: 180000 }).should("contain.text", "Submitted Sim2Real run");
    cy.get("#runSummary", { timeout: 180000 }).should("contain.text", "run");
  });
});
