import { currentLiveAgentConfig } from "../support/e2e";

// Measured on the pre-fix deployed warm-repeat path. The regression gate uses
// server phase timings, not click-to-ready internet latency.
const PRE_FIX_WARM_EXPORT_MS = 31750.8;

function liveEnabled() {
  return String(Cypress.env("NPA_AGENT_CYPRESS_LIVE") || "") === "1";
}

function liveAgentRequest(path, options = {}) {
  const { baseUrl, username, password } = currentLiveAgentConfig();
  return cy.request({
    url: `${String(baseUrl).replace(/\/$/, "")}${path}`,
    auth: { username, password },
    log: false,
    // Cypress treats timeout: 0 as an immediate failure. Give real tenant-wide
    // S3 discovery the same outer infrastructure watchdog as the clean-browser
    // task; the server-side query and exact-match assertions still decide pass.
    timeout: 600000,
    ...options,
  });
}

function configuredRunId() {
  return String(
    Cypress.env("NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID") ||
      Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") ||
      "",
  ).trim();
}

function discoverVerificationRun() {
  const wanted = configuredRunId();
  if (wanted) {
    expect(wanted, "configured Foxglove verification run id").to.match(
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/,
    );
  }
  const query = wanted ? `?q=${encodeURIComponent(wanted)}&limit=50` : "?limit=50";
  return liveAgentRequest(`/api/artifacts/runs${query}`).then((response) => {
    expect(response.status).to.eq(200);
    const runs = Array.isArray(response.body.runs) ? response.body.runs : [];
    const matches = wanted
      ? runs.filter((item) => String(item.run_id || "") === wanted)
      : runs;
    expect(matches.length, "a real discovered run for Foxglove verification").to.be.greaterThan(0);
    if (wanted) {
      expect(matches, "one exact source-qualified verification run").to.have.length(1);
    }
    const run = matches[0];
    expect(String(run.run_id || ""), "discovered run id").to.match(
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/,
    );
    expect(run.run_ref, "source-qualified run reference").to.match(/^npa1_/);
    const exactParams = new URLSearchParams({
      limit: "1000",
      resource_bucket: String(run.bucket || ""),
      project_id: String(run.project_id || ""),
      resolved_prefix: String(run.resolved_prefix || ""),
      source_selected: "1",
    });
    return liveAgentRequest(
      `/api/artifacts/run/${encodeURIComponent(String(run.run_ref))}?${exactParams}`,
    ).then((artifactsResponse) => {
      expect(artifactsResponse.status).to.eq(200);
      const artifacts = Array.isArray(artifactsResponse.body.artifacts)
        ? artifactsResponse.body.artifacts
        : [];
      const artifact = artifacts.find((item) =>
        String(item.key || "").endsWith("/reports/sim2real.mcap")
      );
      expect(artifact, "the exact discovered canonical MCAP artifact").to.exist;
      return {
        ...run,
        artifact,
        bucket: String(artifactsResponse.body.bucket || run.bucket || ""),
        project_id: String(artifactsResponse.body.project_id || run.project_id || ""),
        resolved_prefix: String(
          artifactsResponse.body.resolved_prefix || run.resolved_prefix || "",
        ),
      };
    });
  });
}

function assertEmbeddedExport(exported) {
  const parsed = new URL(exported.recording_url);
  expect(exported.available).to.eq(true);
  expect(parsed.protocol).to.eq("https:");
  expect(parsed.pathname).to.match(/\.mcap$/);
  expect(exported.download_url).to.eq(exported.recording_url);
  expect(exported.recording_url).not.to.match(/authorization|basic|password|api.?token/i);
  expect(exported.web_url, "in-page preparation does not create an external destination").to.eq(
    undefined,
  );
}

describe("NPA agent official Foxglove embed against live infrastructure", () => {
  before(function () {
    if (!liveEnabled()) {
      this.skip();
      return;
    }
    // Fail closed before any request when the opt-in gate is set incompletely.
    currentLiveAgentConfig();
  });

  it("binds the selected MCAP in-page and keeps hosted navigation separate", () => {
    discoverVerificationRun().then((run) => {
      const requestBody = {
        run_id: String(run.run_id),
        run_ref: String(run.run_ref),
        key: String(run.artifact.key),
        resource_bucket: String(run.bucket),
        project_id: String(run.project_id),
        resolved_prefix: String(run.resolved_prefix),
        s3_uri: String(run.artifact.s3_uri),
      };
      liveAgentRequest("/api/foxglove/export", {
        method: "POST",
        body: requestBody,
      }).then((prepared) => {
        expect(prepared.status).to.eq(200);
        expect(prepared.body.ok).to.eq(true);
        expect(prepared.body.run_id).to.eq(run.run_id);
        expect(prepared.body.selected_artifact).to.deep.include({
          run_id: run.run_id,
          run_ref: run.run_ref,
          key: run.artifact.key,
          s3_uri: run.artifact.s3_uri,
          resource_bucket: run.bucket,
          project_id: run.project_id,
          resolved_prefix: run.resolved_prefix,
        });
        const exported = prepared.body.export;
        assertEmbeddedExport(exported);
        expect(exported.canonical_s3_uri).to.match(
          /^s3:\/\/.+\/reports\/sim2real\.mcap$/,
        );
        expect(exported.sha256).to.match(/^[a-f0-9]{64}$/);
        expect(Number(exported.size_bytes || prepared.body.size_bytes)).to.be.greaterThan(8);
        expect(exported.provenance.visualization_contract).to.eq(
          "npa.foxglove.robot-motion.v3",
        );
        expect(exported.provenance.visualization_fixed_frame).to.eq("npa_action_space");
        expect(exported.provenance.visualization_fidelity).to.match(/not calibrated/i);
        expect(exported.provenance.schemas).to.deep.include({
          "/camera": "foxglove.CompressedImage",
          "/camera/side": "foxglove.CompressedImage",
          "/camera/workspace": "foxglove.CompressedImage",
          "/robot/diagnostic_scene": "foxglove.SceneUpdate",
          "/robot/diagnostic_pose": "foxglove.PoseInFrame",
          "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
          "/robot/diagnostic_joint_states": "foxglove.JointStates",
          "/actuators/commands": "npa.ActuatorCommands",
          "/run/state": "npa.RunState",
          "/log": "foxglove.Log",
        });
        expect(
          Object.values(exported.provenance.schemas),
          "numeric metrics schema",
        ).to.include("npa.RunMetrics.execution");
        cy.task(
          "validatePublishedMcap",
          { recordingUrl: exported.recording_url, sha256: exported.sha256 },
          { log: false },
        ).then((validation) => {
          expect(validation.sha256).to.eq(exported.sha256);
          expect(validation.camera_counts).to.deep.eq({
            "/camera": 33,
            "/camera/side": 33,
            "/camera/workspace": 33,
          });
          expect(validation.synchronized).to.eq(true);
          expect(validation.distinct_aligned_triplets).to.eq(33);
          expect(validation.scene_message_count).to.eq(32);
          expect(validation.scene_validation_errors).to.deep.eq([]);
          expect(validation.schema_arrays_without_items).to.deep.eq([]);
        });

        cy.request({ url: exported.recording_url, method: "HEAD", log: false }).then((head) => {
          expect(head.status).to.eq(200);
          expect(head.headers["accept-ranges"]).to.eq("bytes");
          expect(head.headers).not.to.have.property("content-encoding", "gzip");
        });
        cy.request({
          url: exported.recording_url,
          method: "OPTIONS",
          log: false,
          headers: {
            Origin: "https://embed.foxglove.dev",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Range",
          },
        }).then((options) => {
          expect([200, 204]).to.include(options.status);
          expect(options.headers["access-control-allow-origin"]).to.eq("*");
          expect(options.headers["access-control-allow-headers"]).to.include("Range");
        });
        cy.request({
          url: exported.recording_url,
          log: false,
          headers: { Origin: "https://embed.foxglove.dev", Range: "bytes=0-7" },
          encoding: "binary",
        }).then((range) => {
          expect(range.status).to.eq(206);
          expect(range.body).to.eq("\x89MCAP0\r\n");
          expect(range.headers["content-range"]).to.match(/^bytes 0-7\//);
        });

        liveAgentRequest("/api/foxglove/config").then((response) => {
          expect(response.status).to.eq(200);
          expect(response.body.available).to.eq(true);
          expect(response.body.viewer_backend).to.eq("foxglove-sdk");
          expect(response.body.sdk_ready).to.eq(true);
          expect(response.body.embed_src).to.match(/^https:\/\//);
          expect(response.body.requires_account_note).to.match(/sign in/i);
          expect(response.body.data_source).to.deep.include({ type: "remote-file" });
          expect(response.body.data_source.urls).to.deep.eq([exported.recording_url]);
          expect(response.body.layout.version).to.eq(1);
          const collectPanels = (node) => node && node.type === "panel"
            ? [node]
            : [
              ...((node && node.items) || []).flatMap((item) => collectPanels(item.content)),
              ...((node && node.tabs) || []).flatMap((item) => collectPanels(item.content)),
            ];
          const panels = collectPanels(response.body.layout.content);
          expect(
            panels.filter((panel) => panel.panelType === "Image")
              .map((panel) => panel.config.imageMode.imageTopic),
          ).to.deep.eq(["/camera", "/camera/side", "/camera/workspace"]);
          expect(response.body.visualization).to.deep.include({
            contract: "npa.foxglove.robot-motion.v3",
            fixed_frame: "npa_action_space",
          });
          expect(response.body.layout_storage_key).to.eq(
            "npa-agent-foxglove-robot-motion-v3",
          );
        });
        liveAgentRequest("/api/foxglove/status").then((response) => {
          expect(response.status).to.eq(200);
          expect(response.body.viewer_backend).to.eq("foxglove-sdk");
          expect(response.body.run_id).to.eq(run.run_id);
          expect(response.body.recording_url).to.eq(exported.recording_url);
        });

        cy.visitLiveAgent();
        cy.get("#tabRerun").should("have.text", "View").click();
        // A clean browser has no persisted active run. Exercise the visible
        // run selector exactly as an operator must before asserting viewer
        // state; the API export above intentionally does not mutate browser
        // selection state.
        cy.get("#artifactPrefix").clear().type(String(run.run_id)).type("{enter}");
        cy.get("#runIdSelect", { timeout: 180000 }).find("option").should(($options) => {
          expect(
            [...$options].map((option) => option.value),
            "source-qualified discovered run option",
          ).to.include(String(run.run_ref));
        });
        cy.get("#runIdInput").clear().type(String(run.run_id));
        cy.get("#loadRunData").should("be.visible").and("be.enabled").click();
        cy.get("#loadRunData", { timeout: 180000 })
          .should("have.attr", "aria-busy", "false")
          .and("be.enabled");
        cy.get("#simRunId", { timeout: 180000 }).should("contain.text", run.run_id);
        cy.get(".render-mode-tabs .render-mode-tab").then(($tabs) => {
          expect([...$tabs].map((tab) => tab.textContent.trim())).to.deep.eq([
            "View", "Foxglove", "Lichtblick", "Video", "Image", "Data",
          ]);
        });
        [
          ["View", "#viewerPaneRerun"],
          ["Foxglove", "#viewerPaneFoxglove"],
          ["Lichtblick", "#viewerPaneLichtblick"],
        ].forEach(([label, pane]) => {
          cy.contains(".render-mode-tabs .render-mode-tab", new RegExp(`^${label}$`))
            .scrollIntoView()
            .click();
          cy.get(pane)
            .should("have.attr", "aria-hidden", "false")
            .should(($pane) => {
              const rect = $pane[0].getBoundingClientRect();
              expect(rect.width, `${label} live pane width`).to.be.greaterThan(0);
              expect(rect.height, `${label} live pane height`).to.be.greaterThan(0);
            });
        });
        cy.get("#renderModeFoxglove").click();
        cy.get("#viewerPaneFoxglove").should("have.class", "is-active-viewer");
        cy.get("#foxgloveStatus", { timeout: 30000 }).should(($status) => {
          expect($status, "no SDK or hosted-viewer error").not.to.have.class("is-error");
          // Cypress's instrumented Electron page can hold a native dynamic
          // import at its loading boundary. Do not mistake that harness state
          // for SDK readiness: verifyFoxgloveEmbeddedArtifact below launches a
          // clean Chromium profile and must prove the actual iframe, queued or
          // ready SDK command, and exact selected data source before this test
          // can pass.
          expect($status.text()).to.match(
            /Loading Foxglove embed SDK|Connecting to|Foxglove viewer ready|awaiting browser sign-in/,
          );
        });
        cy.get("#foxgloveVisualizationSummary")
          .should("be.visible")
          .and("contain.text", "robot + trajectory 3D")
          .and("contain.text", "not calibrated robot/world kinematics");
        cy.get('[data-testid="open-foxglove-web"]')
          .should("be.visible")
          .and("be.enabled")
          .and("have.text", "Open in Foxglove");

        // This task uses a clean Chromium profile and clicks the actual artifact
        // card action. It proves the Agent page stays put while the official SDK
        // receives the exact selected remote-file source, with readiness or the
        // unsigned hosted sign-in state reported truthfully.
        cy.task(
          "verifyFoxgloveEmbeddedArtifact",
          {
            runId: String(run.run_id),
            runRef: String(run.run_ref),
            artifactKey: String(run.artifact.key),
            projectId: String(run.project_id),
            resourceBucket: String(run.bucket),
            resolvedPrefix: String(run.resolved_prefix),
            s3Uri: String(run.artifact.s3_uri),
          },
          // Cypress interprets timeout: 0 as "fail immediately" (unlike
          // Playwright, where it disables the timeout). Keep the existing
          // outer task watchdog; the real S3 and hosted-navigation waits
          // inside the task remain uncapped and report their own assertions.
          { log: false, timeout: 600000 },
        ).then((result) => {
          expect(result.runId).to.eq(run.run_id);
          expect(result.runRef).to.eq(run.run_ref);
          expect(result.artifactKey).to.eq(run.artifact.key);
          expect(result.exactProvenance).to.deep.include({
            projectId: run.project_id,
            resourceBucket: run.bucket,
            resolvedPrefix: run.resolved_prefix,
            sha256: exported.sha256,
          });
          expect(result.navigation).to.deep.include({
            topUrlUnchanged: true,
            pagesBefore: 1,
            pagesAfter: 1,
          });
          expect(result.navigation.newTargets).to.deep.eq([]);
          expect(result.actions.artifact).to.deep.eq([
            "View in Foxglove", "View in Lichtblick", "Download",
          ]);
          expect(result.actions.mobileArtifact).to.deep.eq([
            "View in Foxglove", "View in Lichtblick", "Download",
          ]);
          expect(result.actions.external).to.eq("Open in Foxglove");
          expect(result.embedded).to.deep.include({
            selected: "true",
            paneAriaHidden: "false",
            iframeOrigin: "https://embed.foxglove.dev",
            setDataSourceCount: 1,
            layoutSelectCount: 1,
            layoutStorageKey: "npa-agent-foxglove-robot-motion-v3",
            iframeReused: true,
            controlsUnobstructed: true,
          });
          expect(
            result.embedded.sdkReady === "true" || result.embedded.signInRequired === true,
            "official SDK is ready or the hosted sign-in state is explicit",
          ).to.eq(true);
          for (const geometry of [
            result.embedded.pane,
            result.embedded.mobilePane,
            result.embedded.iframe,
          ]) {
            expect(geometry.width).to.be.greaterThan(0);
            expect(geometry.height).to.be.greaterThan(0);
          }
          expect(result.embedded.sdkRequestCount).to.be.greaterThan(0);
          expect(result.embedded.desktopClearance.unobstructed).to.eq(true);
          expect(result.embedded.mobileClearance.unobstructed).to.eq(true);
          expect(result.validation).to.deep.eq({
            multipleAnglesVerified: true,
            diagnosticSceneVerified: true,
          });
          expect(result.performance.cacheReused).to.eq(true);
          expect(result.performance.repeatCacheReused).to.eq(true);
          expect(result.performance.clickToPaneMs).to.be.at.most(
            result.performance.clickToApiMs,
          );
          expect(result.performance.clickToReadySeconds).to.be.greaterThan(0);
          expect(result.performance.serverTimingsMs.total).to.be.lessThan(
            PRE_FIX_WARM_EXPORT_MS * 0.5,
          );
          expect(result.performance.repeatServerTimingsMs.total).to.be.lessThan(
            PRE_FIX_WARM_EXPORT_MS * 0.5,
          );
          expect(result.embedded.statusText).to.match(
            /exact selected MCAP sent|queued in the official Foxglove SDK/,
          );
          expect(result.evidence.desktop).to.match(/live-agent-desktop-after\.png$/);
          expect(result.evidence.mobile).to.match(/live-agent-mobile-after\.png$/);
          expect(result.evidence.artifactCardDesktop).to.match(
            /live-artifact-card-desktop-after\.png$/,
          );
        });

        // Hosted navigation is still available, but only from the distinctly
        // labeled action inside the selected Foxglove pane.
        cy.task(
          "verifyFoxgloveHostedNavigation",
          {
            runId: String(run.run_id),
            runRef: String(run.run_ref),
            artifactKey: String(run.artifact.key),
            projectId: String(run.project_id),
            resourceBucket: String(run.bucket),
            resolvedPrefix: String(run.resolved_prefix),
          },
          { log: false, timeout: 600000 },
        ).then((result) => {
          expect(result.cardNavigation).to.deep.eq({
            stayedInPage: true,
            pagesBefore: 1,
            pagesAfter: 1,
          });
          expect(result.officialContract).to.deep.include({
            requestMatchedResponse: true,
            sourceType: "remote-file",
            oneAbsoluteHttpsMcap: true,
            encodedExactlyOnce: true,
            layoutIdPresent: true,
            exactTransportCacheReused: true,
            exportRequestCount: 2,
          });
          expect(result.officialContract.serverTimingsMs.total).to.be.lessThan(
            PRE_FIX_WARM_EXPORT_MS * 0.5,
          );
          expect(result.officialContract.responseStatus).to.be.within(200, 399);
          expect(result.hostedSurface.finalOrigin).to.eq("https://app.foxglove.dev");
          expect(result.hostedSurface.pixels.nonblank).to.eq(true);
          expect(result.evidence.hosted).to.match(/live-hosted-foxglove-after\.png$/);
        });
      });
    });
  });
});
