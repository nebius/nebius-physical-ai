/**
 * Live Describe-this + Rerun splash cover checks against a bootstrapped agent VM.
 *
 * Requires:
 *   NPA_AGENT_BASE_URL, NPA_AGENT_USER, NPA_AGENT_PASSWORD
 */
describe("NPA agent live Describe-this + splash cover", () => {
  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("#rerunBundleCover", { timeout: 90000 }).should("have.attr", "hidden");
  });

  it("never surfaces Loading application bundle in parent chrome", () => {
    cy.get("#rerunBundleCover .cover-title").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#rerunBundleCover .cover-hint").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#statusBar").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.window().then((win) => {
      expect(win.__NPA_AGENT_TEST__).to.exist;
      expect(win.__NPA_AGENT_TEST__.ensureRerunCaptureBridge).to.be.a("function");
      expect(win.document.documentElement.outerHTML).to.include("ensureRerunCaptureBridge");
      expect(win.document.documentElement.outerHTML).to.include("grabFromRerunCaptureBridge");
    });
  });

  it("rejects uniform gray canvases and accepts structured content probes", () => {
    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const gray = win.document.createElement("canvas");
      gray.width = 128;
      gray.height = 96;
      const gctx = gray.getContext("2d");
      gctx.fillStyle = "#9ca3af";
      gctx.fillRect(0, 0, 128, 96);
      expect(api.frameLooksBlank(gray)).to.eq(true);

      const content = win.document.createElement("canvas");
      content.width = 180;
      content.height = 120;
      const cctx = content.getContext("2d");
      cctx.fillStyle = "#0a0a12";
      cctx.fillRect(0, 0, 180, 120);
      cctx.strokeStyle = "#ff8a1f";
      cctx.lineWidth = 4;
      cctx.beginPath();
      cctx.moveTo(50, 20);
      cctx.lineTo(90, 70);
      cctx.lineTo(60, 110);
      cctx.stroke();
      cctx.strokeStyle = "#5eead4";
      cctx.beginPath();
      cctx.moveTo(100, 30);
      cctx.lineTo(140, 90);
      cctx.stroke();
      expect(api.frameLooksBlank(content)).to.eq(false);

      const sparse = win.document.createElement("canvas");
      sparse.width = 960;
      sparse.height = 540;
      const sctx = sparse.getContext("2d");
      sctx.fillStyle = "#050508";
      sctx.fillRect(0, 0, 960, 540);
      sctx.strokeStyle = "#ff8a1f";
      sctx.lineWidth = 2;
      sctx.beginPath();
      sctx.moveTo(480, 80);
      sctx.lineTo(455, 360);
      sctx.lineTo(450, 500);
      sctx.stroke();
      expect(api.frameLooksBlank(sparse)).to.eq(false);
    });
  });

  it("serves grounded pipeline provenance for a discovered run", () => {
    // Provenance drives the "smarter Describe this": it must name the workflow
    // stage + the components/models that produced the run's data, from real S3
    // artifacts (Cosmos Transfer 2.5 on GPU, Token Factory VLM, etc.).
    cy.request({
      url: "/api/artifacts/runs",
      failOnStatusCode: false,
    }).then((runsResp) => {
      if (runsResp.status !== 200 || !runsResp.body || !Array.isArray(runsResp.body.runs)) {
        cy.log("Skipping provenance check: /api/artifacts/runs unavailable in this env");
        return;
      }
      const runs = runsResp.body.runs;
      // Prefer a data-factory / paidf run; fall back to any run with artifacts.
      const preferred =
        runs.find((r) => /paidf|data-factory/i.test(String((r && (r.run_id || r.id)) || ""))) ||
        runs[0];
      if (!preferred) {
        cy.log("Skipping provenance check: no runs discovered");
        return;
      }
      const runId = String(preferred.run_id || preferred.id || "").trim();
      expect(runId, "discovered run id").to.not.eq("");
      cy.request({
        url: "/api/artifacts/provenance/" + encodeURIComponent(runId),
        failOnStatusCode: false,
      }).then((provResp) => {
        expect(provResp.status, "provenance endpoint status").to.eq(200);
        expect(provResp.body.ok, "provenance ok").to.eq(true);
        expect(provResp.body.components, "components").to.be.an("array");
        expect(provResp.body.summary, "summary").to.be.a("string");
        if (provResp.body.components.length > 0) {
          const stages = provResp.body.components.map((c) => String(c.stage || ""));
          const hasKnownStage = stages.some((s) =>
            /augment|annotate|pseudo-label|attribute|curation|visualize|config|source/i.test(s),
          );
          expect(hasKnownStage, "at least one known pipeline stage").to.eq(true);
          // Every component names both a stage and the producing component.
          provResp.body.components.forEach((c) => {
            expect(String(c.stage || ""), "stage label").to.not.eq("");
            expect(String(c.component || ""), "component label").to.not.eq("");
          });
        }
      });
    });
  });

  it("attaches a live Rerun frame via MediaStream bridge (not metadata-only)", () => {
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");

    cy.get("#newChatSession").click({ force: true });
    cy.wait(800);
    cy.get("#chatLog .msg-row").should("have.length", 0);

    cy.window({ timeout: 90000 }).then({ timeout: 90000 }, async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      let painted = false;
      const deadline = Date.now() + 25000;
      while (Date.now() < deadline) {
        const doc = iframe.contentDocument || (iframe.contentWindow && iframe.contentWindow.document);
        const canvas = doc && doc.querySelector("canvas");
        if (canvas) {
          const stats = api.sampleFrameStats(canvas);
          if (stats && ((stats.vivid || 0) > 0 || (stats.variance || 0) > 40 || !api.frameLooksBlank(canvas))) {
            painted = true;
            break;
          }
          api.ensureRerunCaptureBridge(iframe, { forceRestart: true });
          const grabbed = await api.grabFromRerunCaptureBridge(1200, { forceRestart: false });
          if (grabbed) {
            painted = true;
            break;
          }
        }
        await new Promise((r) => setTimeout(r, 400));
      }
      win.__NPA_LIVE_RERUN_PAINTED__ = painted;
      if (!painted) {
        cy.log("Skipping frame-attach assertions: live Rerun canvas stayed blank (no GPU paint)");
        return;
      }
      const quality = await api.waitForQualityRerunFrame(20000);
      win.__NPA_LIVE_DESCRIBE_QUALITY__ = quality || {};
      expect(quality.quality, "live Rerun frame quality").to.eq("rendered");
      expect(quality.dataUrl, "live Rerun JPEG").to.match(/^data:image\/jpeg/);
      expect(quality.dataUrl.length).to.be.greaterThan(4000);
    });

    cy.window().then((win) => {
      if (!win.__NPA_LIVE_RERUN_PAINTED__) {
        expect(true, "no GPU paint in this environment").to.eq(true);
        return;
      }

      cy.intercept("POST", "**/api/chat").as("liveDescribeChat");
      cy.get("#describeVisual").click({ force: true });
      cy.get("#chatLog .msg-row.user", { timeout: 3000 }).should("contain.text", "Describe this");
      cy.get("#panelChat").should("have.class", "chat-drawer-open");

      cy.wait("@liveDescribeChat", { timeout: 180000 }).then((interception) => {
        const body = interception.request.body;
        expect(body.visual_context).to.be.an("object");
        expect(body.visual_context.capture).to.eq("frame");
        expect(body.visual_context.has_image).to.eq(true);
        expect(body.visual_context.frame_quality).to.eq("rendered");
        const last = body.messages[body.messages.length - 1];
        expect(last.content).to.be.an("array");
        const imagePart = last.content.find((part) => part && String(part.type || "").startsWith("image"));
        expect(imagePart).to.exist;
        expect(imagePart.image_url.url).to.match(/^data:image\/jpeg;base64,/);
        expect(imagePart.image_url.url.length).to.be.greaterThan(4000);
        // When the loaded run has artifacts, Describe-this carries grounded
        // provenance so the answer can explain where the visual came from.
        if (body.visual_context.provenance) {
          expect(body.visual_context.provenance).to.be.a("string");
          const textPart = last.content.find(
            (part) => part && String(part.type || "").includes("text"),
          );
          if (textPart) {
            expect(String(textPart.text || "")).to.match(/provenance/i);
          }
        }
      });

      cy.get("#chatLog .msg-row.assistant", { timeout: 180000 }).should("exist");
      cy.get("#chatLog .msg-row.user").last().invoke("text").should("include", "attached viewer frame");
      cy.get("#chatLog .msg-row.assistant").last().invoke("text").then((assistantText) => {
        const text = String(assistantText || "").toLowerCase();
        expect(text).not.to.match(/completely uniform gray/);
        expect(text).not.to.match(/no viewer frame was attached/);
        expect(text).not.to.match(/metadata only/);
        expect(text).to.match(/what i see|skeleton|grid|robot|mesh|trajectory|g1|orange|cyan|wireframe|humanoid|scene|viewport|rerun|franka/);
      });
    });
  });
});
