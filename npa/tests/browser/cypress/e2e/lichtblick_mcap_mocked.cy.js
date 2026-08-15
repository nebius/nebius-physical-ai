import {
  NON_STOCK_RUN_ID,
  decodePngStats,
  firstMcapPngPayload,
  mcapCameraTopicCount,
  mcapHasCompressedImage,
  mcapHasFrameTransform,
  mcapHasHeldoutCamera,
  mcapHasPointCloud,
  mcapPointCloudColorFields,
} from "../support/e2e";

const MCAP_RECORDING_PATH = "/lichtblick/recordings/sim2real.mcap";

// Extensive smoke coverage for the Lichtblick MCAP viewer. These run against the
// mock agent server, which co-serves a real (small) MCAP fixture with genuine
// camera streams so "shows nothing substantive" regressions are caught at the
// data + embed layer without needing live GPU infrastructure.
describe("Lichtblick MCAP viewer (mocked smoke)", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.wait("@simAssets");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  });

  it("co-serves a substantive MCAP with real camera streams", () => {
    cy.request({ url: MCAP_RECORDING_PATH, encoding: "binary" }).then((resp) => {
      expect(resp.status).to.eq(200);
      const body = resp.body || "";
      expect(body.length, "mcap byte length").to.be.greaterThan(10000);
      expect(mcapHasCompressedImage(body), "has foxglove.CompressedImage schema").to.be.true;
      expect(mcapHasHeldoutCamera(body), "has /heldout/camera/ topics").to.be.true;
      expect(mcapCameraTopicCount(body), "camera topic occurrences").to.be.greaterThan(3);
    });
  });

  it("includes a GPU 3D point-cloud stream for the 3D panel", () => {
    cy.request({ url: MCAP_RECORDING_PATH, encoding: "binary" }).then((resp) => {
      const body = resp.body || "";
      expect(mcapHasPointCloud(body), "has foxglove.PointCloud on /heldout/points").to.be.true;
      // The default layout colours the cloud with "rgba-fields", which the viewer
      // only offers when all four colour fields exist; without alpha it falls back
      // to a synthetic colormap and the captured RGB is lost.
      expect(
        mcapPointCloudColorFields(body),
        "point cloud declares red/green/blue/alpha",
      ).to.deep.equal(["red", "green", "blue", "alpha"]);
      // Without a transform defining the cloud's frame the 3D panel places nothing.
      expect(mcapHasFrameTransform(body), "has foxglove.FrameTransform on /tf").to.be.true;
    });
  });

  it("decodes a real (non-stub, non-noise) camera frame from the served MCAP", () => {
    cy.request({ url: MCAP_RECORDING_PATH, encoding: "binary" }).then((resp) => {
      const payload = firstMcapPngPayload(resp.body || "");
      expect(payload, "found a PNG CompressedImage payload").to.be.a("string");
      return decodePngStats(payload).then((stats) => {
        expect(stats.width, "frame width (not a 32px stub)").to.be.greaterThan(40);
        expect(stats.height, "frame height (not a 32px stub)").to.be.greaterThan(40);
        expect(stats.mean, "frame brightness (not dark noise)").to.be.greaterThan(60);
        expect(stats.mean, "frame brightness (not saturated)").to.be.lessThan(250);
      });
    });
  });

  it("feeds the embedded viewer the MCAP data source with camera topics detected", () => {
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");
    cy.get("#lichtblickFrame")
      .its("0.contentWindow.__NPA_MOCK_LICHTBLICK__.loaded", { timeout: 15000 })
      .should("eq", true);
    cy.get("#lichtblickFrame").then(($frame) => {
      const stats = $frame[0].contentWindow.__NPA_MOCK_LICHTBLICK__;
      expect(stats.hasCompressedImage, "viewer received CompressedImage frames").to.be.true;
      expect(stats.hasHeldoutCamera, "viewer received held-out camera topics").to.be.true;
      expect(stats.cameraTopicCount, "camera topics seen by viewer").to.be.greaterThan(3);
      expect(stats.mcapBytes, "mcap bytes fetched by viewer").to.be.greaterThan(10000);
    });
  });

  it("uses Lichtblick-appropriate CTA copy (never Rerun) on the Lichtblick tab", () => {
    // Regression for the reported bug: the shared CTA banner said "No run-specific
    // Rerun recording yet" while the Lichtblick tab was active. The banner copy
    // must track the active render mode and never mention Rerun in Lichtblick mode.
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#simvizCta")
      .invoke("text")
      .should((text) => {
        expect(text).to.not.match(/Rerun recording/i);
      });
  });

  it("decodes the iframe ds.url to the co-served recording path", () => {
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#lichtblickFrame")
      .invoke("attr", "src")
      .then((src) => {
        const url = new URL(String(src), "http://127.0.0.1");
        const ds = decodeURIComponent(url.searchParams.get("ds.url") || "");
        expect(ds).to.include("/lichtblick/recordings/sim2real.mcap");
      });
  });

  it("pins a foreign-origin ds.url onto the page origin", () => {
    // /lichtblick/recordings/ is unauthenticated and grants no CORS, so the viewer's
    // fetch must be same-origin. The backend can build ds.url from a configured
    // public origin (NPA_AGENT_PUBLIC_*) that differs from the origin actually being
    // browsed, so the UI rewrites it back onto window.location.origin.
    cy.window().then((win) => {
      const pin = win.__NPA_AGENT_TEST__.pinLichtblickDsToSameOrigin;
      const foreign =
        "/lichtblick/?ds=remote-file&ds.url=" +
        encodeURIComponent("https://agent.example.test/lichtblick/recordings/sim2real.mcap");
      const pinned = pin(foreign);
      const ds = decodeURIComponent(
        new URL(pinned, win.location.origin).searchParams.get("ds.url") || "",
      );
      expect(ds, "recording fetched from the browsed origin").to.eq(
        win.location.origin + "/lichtblick/recordings/sim2real.mcap",
      );
      // A *relative* ds.url must be absolutized too: Lichtblick's remote-file
      // source silently ignores a relative URL (no range request is ever issued
      // and the viewer sits on "No data source"), which was observed live.
      const relative =
        "/lichtblick/?ds=remote-file&ds.url=" +
        encodeURIComponent("/lichtblick/recordings/sim2real.mcap");
      const relativeDs = decodeURIComponent(
        new URL(pin(relative), win.location.origin).searchParams.get("ds.url") || "",
      );
      expect(relativeDs, "relative ds.url is absolutized").to.eq(
        win.location.origin + "/lichtblick/recordings/sim2real.mcap",
      );
      expect(relativeDs).to.match(/^https?:\/\//);
    });
  });

  it("seeds the injected default layout once instead of wiping it on every mount", () => {
    // The wipe exists to evict a layout saved before our default existed. Doing it on
    // every mount would also discard a layout the user arranged inside the embed, so
    // it must happen once per UI version.
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "ds.url");
    // The wipe resolves asynchronously (IndexedDB deleteDatabase), so retry until the
    // seed is recorded rather than racing it.
    cy.window()
      .should((win) => {
        const api = win.__NPA_AGENT_TEST__;
        expect(win.localStorage.getItem(api.LICHTBLICK_LAYOUT_SEED_KEY)).to.be.a("string");
      })
      .then((win) => {
        const api = win.__NPA_AGENT_TEST__;
        const key = api.LICHTBLICK_LAYOUT_SEED_KEY;
        const version = win.document
          .querySelector("meta[name='npa-ui-version']")
          .getAttribute("content");

        expect(win.localStorage.getItem(key), "seeded with the UI version and layout kind").to.eq(
          `${version}:sim2real`,
        );
        expect(api.lichtblickNeedsLayoutSeed(), "already seeded, no further wipe").to.be.false;

        // A UI redeploy bumps the version tag, which re-seeds so a changed default lands.
        win.localStorage.setItem(key, "0");
        expect(api.lichtblickNeedsLayoutSeed(), "stale seed re-seeds").to.be.true;
        api.markLichtblickLayoutSeeded();
        expect(win.localStorage.getItem(key)).to.eq(`${version}:sim2real`);

        // Behavioural check: with the seed in place, mounting a different recording
        // must leave a layout the user arranged in the embed untouched.
        win.localStorage.setItem("studio.profile-data", "user-arranged-layout");
        api.mountLichtblickIframe({
          lichtblick_ready: true,
          lichtblick_iframe_url:
            "/lichtblick/?ds=remote-file&ds.url=" +
            encodeURIComponent("/lichtblick/recordings/another-run.mcap"),
        });
        expect(
          win.localStorage.getItem("studio.profile-data"),
          "user layout survives switching runs",
        ).to.eq("user-arranged-layout");
      });
  });

  it("keeps the latest MCAP source when prewarm and run mounts race layout seeding", () => {
    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      win.localStorage.setItem(api.LICHTBLICK_LAYOUT_SEED_KEY, "stale-version");
      api.mountLichtblickIframe({
        lichtblick_ready: false,
        lichtblick_iframe_url: "/lichtblick/",
      });
      api.mountLichtblickIframe({
        lichtblick_ready: true,
        lichtblick_iframe_url:
          "/lichtblick/?ds=remote-file&ds.url=" +
          encodeURIComponent("/lichtblick/recordings/latest-run.mcap"),
      });
    });
    cy.get("#lichtblickFrame")
      .should("have.attr", "src")
      .and("include", encodeURIComponent("/lichtblick/recordings/latest-run.mcap"));
  });

  it("filters discovered artifacts to the MCAP (Lichtblick) type", () => {
    cy.get("#tabRerun").click();
    cy.get("#artifactTypeFilter").select("mcap");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`);
    cy.get("#artifactList").should("contain.text", "View in Foxglove");
    cy.get("#artifactList").should("contain.text", "View in Lichtblick");
    // Non-MCAP artifacts are hidden by the type filter.
    cy.get("#artifactList").should("not.contain.text", ".rrd");
    cy.get("#artifactList").should("not.contain.text", ".mp4");
  });

  it("loads an MCAP artifact into the Lichtblick pane with an mcap summary", () => {
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.get(`#artifactList button[data-action="load-artifact"][data-key="${NON_STOCK_RUN_ID}/reports/sim2real.mcap"]`).click();
    cy.wait("@loadArtifact");
    cy.get("#renderModeLichtblick").should("have.class", "is-active");
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#renderedDataSummary").should("contain.text", "mcap");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");
  });

  it("keeps both viewers mounted when switching render modes", () => {
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame")
      .its("0.contentWindow.__NPA_MOCK_LICHTBLICK__", { timeout: 15000 })
      .should("exist");
    cy.get("#lichtblickFrame").then(($frame) => {
      const canvas = $frame[0].contentDocument.querySelector('[data-testid="mock-lichtblick-canvas"]');
      expect(canvas.width, "canvas width").to.be.greaterThan(0);
      expect(canvas.height, "canvas height").to.be.greaterThan(0);
    });
    cy.get("#renderModeRerun").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-inactive-viewer");
    cy.get("#lichtblickFrame").should("exist");
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
  });

});
