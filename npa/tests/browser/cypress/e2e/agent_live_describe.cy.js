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
      expect(win.__NPA_AGENT_TEST__.safeHideRerunBundleCover).to.be.a("function");
      expect(win.document.documentElement.outerHTML).to.include("non-blank canvas");
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
    });
  });

  it("shows Describe this in chat immediately and never claims uniform gray when a frame attaches", () => {
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");

    // Probe live pixels with the async WebGL-safe path before Describe.
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      let ready = false;
      const deadline = Date.now() + 60000;
      while (Date.now() < deadline) {
        // eslint-disable-next-line no-await-in-loop
        ready = await api.probeRerunCanvasContent(iframe);
        if (ready) break;
        // eslint-disable-next-line no-await-in-loop
        const quality = await api.waitForQualityRerunFrame(1500);
        if (quality && quality.quality === "rendered") {
          ready = true;
          break;
        }
        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => setTimeout(r, 500));
      }
      // Continue even if still unavailable — Describe must stay honest (metadata-only).
      win.__NPA_LIVE_DESCRIBE_READY__ = ready;
    });

    cy.intercept("POST", "**/api/chat").as("liveDescribeChat");
    cy.get("#describeVisual").click({ force: true });
    cy.get("#chatLog .msg-row.user", { timeout: 3000 }).should("contain.text", "Describe this");
    cy.get("#panelChat").should("have.class", "chat-drawer-open");

    cy.wait("@liveDescribeChat", { timeout: 180000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context).to.be.an("object");
      if (body.visual_context.capture === "frame") {
        expect(body.visual_context.frame_quality).to.eq("rendered");
        expect(body.visual_context.has_image).to.eq(true);
        const last = body.messages[body.messages.length - 1];
        expect(last.content).to.be.an("array");
        const imagePart = last.content.find((part) => part && String(part.type || "").startsWith("image"));
        expect(imagePart).to.exist;
        expect(imagePart.image_url.url).to.match(/^data:image\/jpeg;base64,/);
        expect(imagePart.image_url.url.length).to.be.greaterThan(4000);
      } else {
        // Honest metadata-only path — never attach a blank/gray buffer.
        expect(body.visual_context.has_image).to.eq(false);
      }
    });

    cy.get("#chatLog .msg-row.assistant", { timeout: 180000 }).should("exist");
    cy.get("#chatLog .msg-row.assistant").should(($el) => {
      const text = $el.text().toLowerCase();
      const userText = Cypress.$("#chatLog .msg-row.user").last().text();
      if (userText.includes("attached viewer frame")) {
        expect(text).not.to.match(/completely uniform gray/);
        expect(text).not.to.match(/frame appears completely uniform/);
        expect(text).to.match(/what i see|skeleton|grid|robot|mesh|camera|trajectory|scene|viewport|rerun/);
      } else {
        expect(text).to.match(/metadata only|no viewer frame|frame unavailable|could not/);
        expect(text).not.to.match(/completely uniform gray/);
      }
    });
  });
});
