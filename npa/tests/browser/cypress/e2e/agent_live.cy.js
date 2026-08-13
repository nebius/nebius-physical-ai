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

function liveArtifactRunId() {
  return Cypress.env("NPA_AGENT_CYPRESS_ARTIFACT_RUN_ID") || "";
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

function artifactSourceQuery(entry) {
  const params = new URLSearchParams();
  const bucket = String((entry && entry.bucket) || "").trim();
  const projectId = String((entry && entry.project_id) || "").trim();
  const resolvedPrefix = String((entry && entry.resolved_prefix) || "").trim();
  if (bucket) params.set("resource_bucket", bucket);
  if (projectId) params.set("project_id", projectId);
  if (resolvedPrefix) params.set("resolved_prefix", resolvedPrefix);
  if (bucket) params.set("source_selected", "1");
  return params.toString();
}

function sourceAwareLoadRunBody(runId, entry) {
  return {
    run_id: runId,
    run_ref: String((entry && entry.run_ref) || ""),
    camera: "workspace",
    resource_bucket: String((entry && entry.bucket) || ""),
    project_id: String((entry && entry.project_id) || ""),
    resolved_prefix: String((entry && entry.resolved_prefix) || ""),
    source_selected: true,
  };
}

function findArtifactInEntry(entry, predicate, cursor = "") {
  const runId = String((entry && entry.run_id) || "");
  const params = new URLSearchParams(artifactSourceQuery(entry));
  if (cursor) params.set("cursor", cursor);
  return liveAgentRequest(
    `/api/artifacts/run/${encodeURIComponent(runId)}?${params.toString()}`,
    { timeout: 120000, failOnStatusCode: false },
  ).then((artifactsResponse) => {
    if (artifactsResponse.status !== 200) return null;
    const payload = artifactsResponse.body || {};
    const artifact = (payload.artifacts || []).find((item) => predicate(item));
    if (artifact) return artifact;
    const nextCursor = String(payload.next_cursor || "");
    return nextCursor ? findArtifactInEntry(entry, predicate, nextCursor) : null;
  });
}

function findLiveRunArtifact(runId, predicate) {
  return liveAgentRequest(
    `/api/artifacts/runs?limit=200&q=${encodeURIComponent(runId)}`,
    { timeout: 120000 },
  ).then((runsResponse) => {
    expect(runsResponse.status).to.eq(200);
    const entries = ((runsResponse.body && runsResponse.body.runs) || []).filter(
      (entry) => String((entry && entry.run_id) || "") === runId,
    );
    expect(entries.length, `artifact sources for ${runId}`).to.be.greaterThan(0);
    const inspect = (index) => {
      if (index >= entries.length) return cy.wrap(null, { log: false });
      const entry = entries[index];
      return findArtifactInEntry(entry, predicate).then((artifact) => {
        return artifact ? { entry, artifact } : inspect(index + 1);
      });
    };
    return inspect(0);
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
    cy.get("#tabMain").should("exist");
    cy.get("#tabRerun").should("exist");
    // Describe-this (viewer capture) button and the artifact Stage filter must ship.
    cy.get("#describeVisual").should("exist");
    cy.get("#artifactStageFilter").should("exist");
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

  it("selects four live project access states without duplicate or stale details", () => {
    liveAgentRequest("/api/access?refresh=true").then((response) => {
      expect(response.status).to.eq(200);
      const access = response.body || {};
      const projects = Array.isArray(access.projects) ? access.projects : [];
      const deploymentId = String(((access.identity || {}).deployment_project_id) || "");
      const deployment = projects.find((project) => String(project.id || "") === deploymentId);
      const foreignWithBucket = projects.find((project) => (
        !project.deployment_project &&
        (project.resources || []).some((resource) => (
          (((resource.capabilities || {}).artifact_discovery || {}).status) === "available"
        ))
      ));
      const noBucket = projects.find((project) => (project.resources || []).length === 0);
      const emptyBucket = projects.find((project) => (
        (project.resources || []).some((resource) => {
          const read = ((resource.capabilities || {}).artifact_read || {});
          return read.status === "unverified" && /empty|no object/i.test(String(read.reason || ""));
        })
      ));
      const representatives = [deployment, foreignWithBucket, noBucket, emptyBucket];
      expect(representatives.every(Boolean), JSON.stringify(representatives)).to.eq(true);
      expect(new Set(representatives.map((project) => project.id)).size).to.eq(4);

      cy.get('label[for="agentAccessProjectSelect"]').should("be.visible");
      cy.get("#agentAccessProjectSelect")
        .should("have.prop", "tagName", "SELECT")
        .and("be.enabled")
        .and("have.value", deploymentId)
        .focus()
        .should("be.focused");
      cy.get("#agentAccessProjectSelect option").should("have.length", projects.length).then(($options) => {
        const optionIds = [...$options].map((option) => option.value);
        expect(new Set(optionIds)).to.deep.equal(new Set(projects.map((project) => String(project.id || ""))));
      });
      cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1);
      cy.get("#agentAccessPanel").should(($panel) => {
        expect($panel.text()).not.to.match(/\bpartial\b/i);
      });

      let previousProject = "";
      representatives.forEach((project) => {
        const projectId = String(project.id || "");
        const projectName = String(project.name || projectId);
        const resources = Array.isArray(project.resources) ? project.resources : [];
        cy.get("#agentAccessProjectSelect").select(projectId);
        cy.get("#agentAccessProjects .access-project-detail")
          .should("have.length", 1)
          .and("have.attr", "data-project-id", projectId)
          .and("contain.text", projectName)
          .and("contain.text", projectId);
        if (previousProject) {
          cy.get("#agentAccessProjects .access-project-detail").should("not.have.attr", "data-project-id", previousProject);
        }
        if (!resources.length) {
          cy.get("#agentAccessProjects").should("contain.text", "No searchable artifact bucket.");
          cy.get("#agentAccessProjects button[data-access-action]").should("not.exist");
        } else {
          resources.forEach((resource) => {
            const bucket = String(resource.name || "");
            cy.get("#agentAccessProjects .access-project-detail").should("contain.text", bucket);
          });
          cy.get("#agentAccessProjects button[data-access-action]").each(($button) => {
            expect($button.attr("data-project-id")).to.eq(projectId);
            expect(resources.map((resource) => String(resource.name || ""))).to.include(
              String($button.attr("data-resource-bucket") || ""),
            );
          });
        }
        previousProject = projectId;
      });

      const emptyProjectId = String(emptyBucket.id || "");
      const emptyReason = String((((emptyBucket.resources || [])[0].capabilities || {}).artifact_read || {}).reason || "");
      cy.get("#agentAccessProjectSelect").select(emptyProjectId);
      cy.get("#agentAccessProjects").should("contain.text", "Read: Unverified").and("contain.text", emptyReason);
      cy.get('#agentAccessProjects button[data-access-action="read"]').should("be.disabled");
      cy.get("#agentAccessRefresh").click();
      cy.get("#agentAccessStatus", { timeout: 60000 }).should("not.contain.text", "Refreshing access");
      cy.get("#agentAccessProjectSelect", { timeout: 60000 }).should("have.value", emptyProjectId);
      cy.reload();
      cy.get("#agentAccessProjectSelect", { timeout: 60000 }).should("have.value", emptyProjectId);
      cy.get("#agentAccessProjects .access-project-detail")
        .should("have.length", 1)
        .and("have.attr", "data-project-id", emptyProjectId);
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
    cy.get("#tabMain").click();
    cy.get("#runSummary", { timeout: 30000 }).should("contain.text", "status");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#artifactDiscoverStatus", { timeout: 30000 }).should("contain.text", "consolidated");

    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#statusBar", { timeout: 120000 }).should(($bar) => {
      const text = $bar.text();
      expect(text).to.match(/done|Ready|SUCCESS|Rerun|Reload/i);
    });

    cy.get("#openRerun").should("be.visible");
  });

  it("loads a configured artifact-only run and clears stale panels on search/switch", function () {
    const runId = liveArtifactRunId();
    if (!runId) {
      this.skip();
    }

    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear().type(`${runId}{enter}`, { delay: 0 });
    cy.get("#artifactList", { timeout: 120000 }).should("contain.text", runId);
    cy.get("#renderModeData", { timeout: 120000 }).should("have.class", "is-active");
    cy.get("#artifactPreviewHost pre", { timeout: 120000 }).should(($pre) => {
      expect(String($pre.text() || "").trim().length, "JSON/text preview is useful").to.be.greaterThan(2);
    });
    cy.get("#tabMain").click();
    cy.get("#runSummary").should("contain.text", runId);
    cy.get("#stageList .stage-item").should("have.length.greaterThan", 0);
    cy.get("#stageList .stage-evidence").should("have.length.greaterThan", 0);
    cy.get("#stageList .stage-status").each(($status) => {
      expect(String($status.text()).trim()).to.match(/^(Observed output|Status unavailable)$/);
    });
    cy.get("#stageList").should("not.contain.text", "Not run");
    cy.get("#stageList").should("not.contain.text", "Succeeded");
    cy.get("#stageList").should("not.contain.text", "Failed");
    cy.get("#stageList .stage-progress").should("contain.text", "execution status unavailable");

    cy.get("#tabRerun").click();
    cy.get("#artifactPrefix").clear().type("no-such-run-for-stale-state-check", { delay: 0 });
    cy.get("#artifactList").should("contain.text", "Select a run");
    cy.get("#runSummary").should("not.contain.text", runId);
    cy.get("#artifactPrefix").clear();
    cy.get("#runIdInput").clear().type("franka-demo{enter}", { delay: 0 });
    cy.get("#simRunId", { timeout: 120000 }).should("contain.text", "franka-demo");
    cy.get("#tabMain").click();
    cy.get("#stageList").should("contain.text", "Local Franka demo");
  });

  it("embeds the Lichtblick MCAP viewer and co-serves the recording", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // A boot-time status refresh can finish just after the first click and restore
    // the artifact's viewer mode. Re-activate until the explicit operator choice
    // wins, then require the pane to stay selected.
    const activateLichtblick = (attempt) => {
      cy.get("#renderModeLichtblick").click();
      cy.wait(500);
      cy.get("#viewerPaneLichtblick").then(($pane) => {
        if ($pane.hasClass("is-active-viewer")) return;
        if (attempt < 3) {
          activateLichtblick(attempt + 1);
          return;
        }
        expect($pane).to.have.class("is-active-viewer");
      });
    };
    activateLichtblick(0);
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

  it("loads the configured run from its exact artifact source, never stale history", function () {
    const runId = liveRunId();
    if (!runId) {
      this.skip();
    }

    const assertExactSource = (response, source) => {
      expect(response.status, JSON.stringify(response.body)).to.eq(200);
      expect(response.body, JSON.stringify(response.body)).to.have.property("ok", true);
      const simViz = response.body.sim_viz || {};
      const preferred = response.body.preferred || {};
      const prefix = String(source.entry.resolved_prefix || "");
      const expectedScope = prefix ? `${prefix}/${runId}/` : `${runId}/`;
      expect(String(simViz.run_id || "")).to.eq(runId);
      expect(String(simViz.bucket || "")).to.eq(String(source.entry.bucket || ""));
      expect(String(simViz.project_id || "")).to.eq(String(source.entry.project_id || ""));
      expect(String(simViz.resolved_prefix || "")).to.eq(prefix);
      expect(
        String(preferred.key || "").startsWith(expectedScope),
        `preferred key belongs to ${expectedScope}`,
      ).to.eq(true);
      expect(String(simViz.artifact_key || "")).to.eq(String(preferred.key || ""));
      expect(String(simViz.artifact_uri || "")).to.eq(String(preferred.s3_uri || ""));
      expect(String(simViz.artifact_render || "")).to.eq(String(preferred.render || ""));
      expect(String(simViz.artifact_key || "")).not.to.include("training-summary.png");
      if (String(simViz.artifact_render || "") === "rerun") {
        expect(String(simViz.artifact_key || "")).to.match(/\/reports\/sim2real\.rrd$/);
        expect(String(simViz.artifact_uri || "")).to.match(/\/reports\/sim2real\.rrd$/);
        expect(String(simViz.rrd_uri || "")).to.match(/^file:\/\//);
        expect(simViz.rerun_ready).to.eq(true);
        expect(String(simViz.camera || "")).to.eq("heldout-sim");
        expect(String(simViz.preview_entity || "")).to.eq("camera");
        expect(simViz.visualization_note || "").to.match(/held-out simulation camera|reference proxy/i);
        expect(decodeURIComponent(String(simViz.rerun_iframe_url || ""))).to.include(
          String(simViz.artifact_preview_url || ""),
        );
        expect(String(simViz.artifact_preview_url || "")).to.match(
          /^\/rerun\/recordings\/cap-[A-Za-z0-9_-]{43}\.rrd$/,
        );
      } else {
        expect(String(simViz.artifact_render || "")).to.eq("video");
        expect(String(simViz.artifact_key || "")).to.match(/\.mp4$/);
        expect(String(simViz.rrd_uri || "")).to.eq("");
        expect(Boolean(simViz.rerun_ready)).to.eq(false);
        expect(String(simViz.artifact_preview_url || "")).to.match(/^\/api\/artifacts\/file\//);
      }
      return simViz;
    };

    findLiveRunArtifact(
      runId,
      (item) => String((item && item.render) || "") === "video",
    ).then((source) => {
      expect(source, `viewable exact source for ${runId}`).not.to.eq(null);
      return liveAgentRequest("/api/sim-viz/load-run", {
        method: "POST",
        body: sourceAwareLoadRunBody(runId, source.entry),
        timeout: 120000,
      }).then((response) => {
        const loaded = assertExactSource(response, source);
        return liveAgentRequest(
          `/api/sim-viz/status?run_id=${encodeURIComponent(runId)}`,
        ).then((statusResponse) => {
          expect(statusResponse.status).to.eq(200);
          expect(String(statusResponse.body.run_id || "")).to.eq(runId);
          expect(String(statusResponse.body.artifact_key || "")).to.eq(
            String(loaded.artifact_key || ""),
          );
          expect(String(statusResponse.body.resolved_prefix || "")).to.eq(
            String(source.entry.resolved_prefix || ""),
          );
          expect(String(statusResponse.body.artifact_render || "")).to.eq(
            String(loaded.artifact_render || ""),
          );
          cy.reload();
          cy.get("#statusBar", { timeout: 30000 }).should("exist");
          cy.get("#rerunBundleCover", { timeout: 60000 }).should("have.attr", "hidden");
          cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
          cy.get("#renderedDataSummary").should("contain.text", String(loaded.artifact_render));
          cy.get("#renderedDataSummary").should("not.contain.text", "training-summary.png");
          cy.get("#tabRerun").click();
          if (String(loaded.artifact_render || "") === "rerun") {
            cy.get("#rerunFrame").should(($frame) => {
              const src = String($frame.attr("src") || "");
              expect(decodeURIComponent(src)).to.include(
                String(statusResponse.body.artifact_preview_url || ""),
              );
            });
            const publicRecordingUrl = `${String(Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl") || "").replace(/\/$/, "")}${statusResponse.body.artifact_preview_url}`;
            cy.request({ url: publicRecordingUrl, failOnStatusCode: false }).then((rrdResp) => {
              expect(rrdResp.status).to.eq(200);
              expect(String(rrdResp.body || "").length).to.be.greaterThan(0);
            });
            cy.get("#statusBar").should("not.contain.text", "Non-RRD artifact loaded");
          } else {
            cy.get("#renderModeVideo").should("have.class", "is-active");
            cy.get("#artifactPreviewHost video", { timeout: 60000 })
              .should("have.attr", "src")
              .and("match", /^blob:/);
          }
        });
      });
    });
  });

  it("presents the live run with an intuitive stage timeline and stable desktop layout", function () {
    const runId = liveRunId();
    if (!runId) {
      this.skip();
    }

    cy.viewport(1440, 1000);
    findLiveRunArtifact(
      runId,
      (item) => String((item && item.render) || "") === "video",
    ).then((source) => {
      expect(source, `viewable exact source for ${runId}`).not.to.eq(null);
      return liveAgentRequest("/api/sim-viz/load-run", {
        method: "POST",
        body: sourceAwareLoadRunBody(runId, source.entry),
        timeout: 120000,
      });
    }).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property("ok", true);
      expect(String((response.body.sim_viz || {}).run_id || "")).to.eq(runId);
      expect(String((response.body.sim_viz || {}).artifact_render || "")).to.eq(
        String((response.body.preferred || {}).render || ""),
      );
    });
    cy.reload();
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
    cy.get("#artifactLoadRunArtifacts").click();

    cy.get("#artifactList", { timeout: 120000 }).should("contain.text", ".mp4");
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#stageList", { timeout: 30000 }).within(() => {
      cy.get(".stage-item").should("have.length.greaterThan", 0);
      cy.get(".stage-evidence").should("have.length.greaterThan", 0);
      cy.get(".stage-status").each(($status) => {
        expect(String($status.text()).trim()).to.match(
          /^(Succeeded|Failed|Running|Skipped|Not run|Pending|Submitted|Observed output|Status unavailable)$/
        );
      });
    });
    cy.get("#runSummary").should("contain.text", runId);
    cy.get("#stageList .stage-progress").invoke("text").should("not.match", /\d+\/\d+ stages succeeded/i);
    cy.get("#runLog").should("contain.text", "does not establish execution success");
    cy.get("#tabRerun").click();
    cy.get("#renderedDataSummary", { timeout: 30000 }).should("contain.text", "video");
    cy.get("#renderModeVideo").should("have.class", "is-active");
    // Tab panels stay mounted with opacity:0 when inactive — assert activation class,
    // not Cypress visibility (opacity:0 is treated as hidden).
    cy.get("#tabMain").click();
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
      expect(String(artifactList.textContent || "")).to.match(/\.mp4/);
      const mediaPane = win.document.getElementById("viewerPaneMedia");
      expect(mediaPane, "media viewer exists").to.exist;
      const mediaRect = mediaPane.getBoundingClientRect();
      expect(mediaRect.width, "media viewer has usable width").to.be.greaterThan(240);
      expect(mediaRect.height, "media viewer has usable height").to.be.greaterThan(40);
    });
    cy.get("#tabMain").click();
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
      const candidates = runs.filter((entry) => String((entry && entry.run_id) || ""));

      const findMp4 = (index) => {
        // mp4 presence is a live-data property, not a code contract (the Video viewer
        // path is covered by the mocked suite), so signal "not found" and skip below
        // instead of hard-failing on data-starved environments.
        if (index >= candidates.length) {
          return cy.wrap(null, { log: false });
        }
        const entry = candidates[index];
        const runId = String(entry.run_id || "");
        return findArtifactInEntry(
          entry,
          (item) => String((item && item.key) || "").toLowerCase().endsWith(".mp4"),
        ).then((mp4) => {
          if (!mp4) {
            return findMp4(index + 1);
          }
            return {
              runId,
              key: String(mp4.key),
              s3Uri: String(mp4.s3_uri || ""),
              entry,
              role: String(mp4.role || "output"),
            };
        });
      };

      return findMp4(0);
    }).then((found) => {
      if (!found) {
        cy.log("no mp4 artifact discoverable in live runs — skipping Video viewer assertions");
        return;
      }
      const { runId, s3Uri, entry, role } = found;
      return liveAgentRequest("/api/sim-viz/load-artifact", {
        method: "POST",
        body: { run_id: runId, run_ref: String(entry.run_ref || ""), s3_uri: s3Uri },
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
        });
      }).then(() => {
        cy.get("#tabRerun").click();
        cy.get("#artifactRefreshRuns").click();
        cy.get("#artifactDiscoverStatus", { timeout: 30000 }).should("contain.text", "consolidated");
        cy.get("#runIdSelect", { timeout: 30000 }).then(($select) => {
          const options = [...$select[0].options];
          const sourceIndex = options.findIndex((option) =>
            String(option.dataset.runId || "") === runId &&
            (!entry.run_ref || option.value === String(entry.run_ref)) &&
            String(option.dataset.bucket || "") === String(entry.bucket || "") &&
            String(option.dataset.resolvedPrefix || "") === String(entry.resolved_prefix || "")
          );
          expect(sourceIndex, `source option for ${runId}`).to.be.at.least(0);
          $select[0].selectedIndex = sourceIndex;
          cy.wrap($select).trigger("change");
        });
        cy.get("#runIdInput", { timeout: 120000 }).should("have.value", runId);
        cy.get("#artifactList", { timeout: 120000 }).should("contain.text", ".mp4");
        // Artifact inventory is role-separated. Follow the discovered artifact's
        // role before applying the video render filter.
        cy.get("#artifactRoleFilter").select(role);
        cy.get("#artifactTypeFilter").select("video");
        cy.get("#artifactList", { timeout: 120000 }).should("contain.text", ".mp4");
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
  });

  it("surfaces a searched run in both pickers via the server (q=) search path", function () {
    // Reproduces the "runs don't show" report against the deployed agent:
    // typing a fragment must render the matching run as an <option> in BOTH the
    // Rerun-tab (#runIdSelect) and Stages-tab (#stagesRunSelect) pickers, driven
    // by the debounced server search (/api/artifacts/runs?q=). When an old run
    // beyond the newest page is configured via NPA_AGENT_CYPRESS_SEARCH_RUN_ID,
    // this proves the union path (server search + client render) end-to-end;
    // otherwise it falls back to a run discovered on the default page.
    liveAgentRequest("/api/artifacts/runs?limit=100").then((resp) => {
      expect(resp.status).to.eq(200);
      const runs = (resp.body && resp.body.runs) || [];
      expect(runs.length, "default page runs").to.be.greaterThan(0);

      const configured = String(
        Cypress.env("NPA_AGENT_CYPRESS_SEARCH_RUN_ID") ||
          Cypress.env("NPA_AGENT_SEARCH_RUN_ID") ||
          "",
      ).trim();

      const resolveTarget = configured
        ? liveAgentRequest(
            `/api/artifacts/runs?limit=100&q=${encodeURIComponent(configured)}`,
          ).then((qr) => {
            const qruns = (qr.body && qr.body.runs) || [];
            const hit =
              qruns.find((r) => String((r && r.run_id) || "").includes(configured)) ||
              qruns[0];
            return String((hit && hit.run_id) || configured).trim();
          })
        : cy.wrap(
            String((runs[runs.length - 1] && runs[runs.length - 1].run_id) || "").trim(),
          );

      resolveTarget.then((targetRunId) => {
        expect(targetRunId, "target run id").to.not.eq("");
        const fragment = configured
          ? configured
          : targetRunId.length > 12
            ? targetRunId.slice(0, 12)
            : targetRunId;

        // Rerun-tab picker: typing the fragment triggers the debounced server
        // search; the matching run must render as an option (Cypress retries the
        // assertion while the 350ms debounce + fetch complete).
        cy.get("#tabRerun").click();
        cy.get("#artifactPrefix", { timeout: 30000 }).clear().type(fragment, { delay: 0 });
        cy.get("#runIdSelect option", { timeout: 30000 }).should(($opts) => {
          const runIds = [...$opts].map((o) => String(o.dataset.runId || "")).filter(Boolean);
          expect(runIds, "Rerun picker surfaces the searched run").to.include(targetRunId);
        });

        // Stages-tab picker: same server-search path must populate the select.
        cy.get("#tabMain").click();
        cy.get("#stagesRunInput", { timeout: 30000 }).clear().type(fragment, { delay: 0 });
        cy.get("#stagesRunSelect option", { timeout: 30000 }).should(($opts) => {
          const runIds = [...$opts].map((o) => String(o.dataset.runId || "")).filter(Boolean);
          expect(runIds, "Stages picker surfaces the searched run").to.include(targetRunId);
        });
      });
    });
  });

  it("selects a live artifact-backed training run and shows outputs without Rerun", function () {
    const runId = String(
      Cypress.env("NPA_AGENT_CYPRESS_TRAINING_RUN_ID") ||
        Cypress.env("NPA_AGENT_TRAINING_RUN_ID") ||
        "",
    ).trim();
    if (!runId) this.skip();

    cy.get("#tabRerun").click();
    cy.get("#artifactPrefix").clear().type(runId, { delay: 0 });
    cy.get("#runIdSelect option", { timeout: 60000 }).should(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      expect(values, "artifact run selector contains exact training run").to.include(runId);
      expect(values, "workflow stages are not run ids").not.to.include("checkpoints");
      expect(values, "workflow stages are not run ids").not.to.include("evidence");
    });

    cy.intercept("POST", "/api/sim-viz/load-run").as("trainingLoadRunLive");
    cy.intercept("GET", `/api/artifacts/run/${runId}*`).as("trainingArtifactsLive");
    cy.get("#runIdInput").clear().type(runId, { delay: 0 });
    cy.get("#loadRunData").click();
    cy.wait("@trainingLoadRunLive", { timeout: 120000 })
      .its("response.statusCode")
      .should("eq", 200);
    cy.wait("@trainingArtifactsLive", { timeout: 120000 })
      .its("response.statusCode")
      .should("eq", 200);

    cy.get("#artifactRoleFilter").should("have.value", "output");
    cy.get("#artifactList .artifact-card[data-role='output']", { timeout: 120000 })
      .its("length")
      .should("be.greaterThan", 0);
    cy.get("#artifactList .artifact-card[data-role='input']").should("not.exist");
    cy.get("#artifactList").should("contain.text", "manifest.json");
    cy.get("#artifactList").should("contain.text", "workflow.yaml");
    cy.get("#artifactList button[data-action='download-artifact']").should("exist");
    cy.get("#simRunId").should("contain.text", runId);
    cy.get("#rerunPlaceholder", { timeout: 30000 })
      .should("have.attr", "data-state", "no-preview-artifacts")
      .and("contain.text", "No RRD/MCAP recording; use the artifacts below");
    cy.get("#renderedDataSummary").should(
      "contain.text",
      "No RRD/MCAP recording; use the artifacts below",
    );
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
