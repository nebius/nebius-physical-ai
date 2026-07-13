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
    cy.get("#artifactList", { timeout: 30000 }).should("contain.text", "Runs discovered");

    cy.get("#loadFrankaRerun").click();
    cy.get("#statusBar", { timeout: 120000 }).should(($bar) => {
      const text = $bar.text();
      expect(text).to.match(/done|Ready|SUCCESS|Rerun/i);
    });

    cy.get("#openRerun").should("be.visible");
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

    liveAgentRequest("/api/sim-viz/load-run", {
      method: "POST",
      body: { run_id: runId, camera: "workspace" },
    }).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property("ok", true);
      const simViz = response.body.sim_viz || {};
      expect(simViz.run_id).to.eq(runId);
      expect(simViz.artifact_render).to.eq("rerun");
      expect(simViz.artifact_key).to.match(/\/reports\/sim2real\.rrd$/);
      expect(simViz.artifact_uri).to.match(/\/reports\/sim2real\.rrd$/);
      expect(simViz.rrd_uri).to.match(/^file:\/\//);
      expect(simViz.rerun_ready).to.eq(true);
      expect(simViz.camera).to.eq("heldout-sim");
      expect(simViz.preview_entity).to.eq("camera");
      expect(simViz.visualization_note || "").to.match(/held-out simulation camera|reference proxy/i);
      expect(decodeURIComponent(String(simViz.rerun_iframe_url || ""))).to.include(
        "/rerun/recordings/sim2real.rrd",
      );
    });

    liveAgentRequest("/api/sim-viz/status").then((response) => {
      expect(response.status).to.eq(200);
      const simViz = response.body || {};
      expect(simViz.active_run_id || simViz.run_id).to.eq(runId);
      expect(simViz.artifact_render).to.eq("rerun");
      expect(simViz.artifact_key).to.match(/\/reports\/sim2real\.rrd$/);
      expect(simViz.camera).to.eq("heldout-sim");
      expect(simViz.visualization_note || "").to.match(/held-out simulation camera|reference proxy/i);
      expect(decodeURIComponent(String(simViz.rerun_iframe_url || ""))).to.include(
        "/rerun/recordings/sim2real.rrd",
      );
    });

    cy.reload();
    cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
    cy.get("#tabRerun").click();
    cy.get("#rerunFrame").should(($frame) => {
      const src = String($frame.attr("src") || "");
      expect(decodeURIComponent(src)).to.include("/rerun/recordings/sim2real.rrd");
    });
    cy.get("#statusBar").should("not.contain.text", "Non-RRD artifact loaded");
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
    });
    cy.reload();
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#runIdInput", { timeout: 30000 }).clear().type(runId);
    cy.get("#artifactLoadRunArtifacts").click();

    cy.get("#artifactList", { timeout: 120000 }).within(() => {
      cy.contains("reports/sim2real.rrd").should("be.visible");
      cy.contains("render=rerun").should("be.visible");
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
    cy.get("#renderedDataSummary").should("contain.text", "rerun").and("contain.text", "sim2real.rrd");
    cy.get("#renderedDataSummary").should("contain.text", "held-out simulation camera");
    cy.get("#simCamera").should("contain.text", "heldout-sim");
    cy.get("#tabRerun").click();
    cy.get("#rerunFrame").should("be.visible");
    cy.get("#tabChat").click();
    cy.get("#chatForm").should("be.visible");

    cy.window().then((win) => {
      const doc = win.document.documentElement;
      expect(doc.scrollWidth, "no distracting horizontal page overflow").to.be.lte(win.innerWidth + 24);
      win.document.getElementById("tabRerun").click();
      for (const id of ["artifactList", "rerunFrame"]) {
        const el = win.document.getElementById(id);
        expect(el, `${id} exists`).to.exist;
        const rect = el.getBoundingClientRect();
        expect(rect.width, `${id} has usable width`).to.be.greaterThan(240);
        expect(rect.height, `${id} has usable height`).to.be.greaterThan(id === "artifactList" ? 24 : 40);
      }
      win.document.getElementById("tabChat").click();
      for (const id of ["chatForm", "runDetails"]) {
        const el = win.document.getElementById(id);
        expect(el, `${id} exists`).to.exist;
        const rect = el.getBoundingClientRect();
        expect(rect.width, `${id} has usable width`).to.be.greaterThan(240);
        expect(rect.height, `${id} has usable height`).to.be.greaterThan(40);
      }
    });
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

  it("submits Sim2Real from the UI when live destructive Cypress is enabled", function () {
    if (!destructiveLiveEnabled()) {
      this.skip();
    }
    cy.get("#submitWorkflow").click();
    cy.get("#chatLog", { timeout: 180000 }).should("contain.text", "Submitted Sim2Real run");
    cy.get("#runSummary", { timeout: 180000 }).should("contain.text", "run");
  });
});
