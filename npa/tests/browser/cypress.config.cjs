const { defineConfig } = require("cypress");
const crypto = require("crypto");
const { spawnSync } = require("child_process");
const fs = require("fs");
const http = require("http");
const https = require("https");
const os = require("os");
const path = require("path");
const { chromium } = require("playwright-core");
const { PNG } = require("pngjs");

const repoRoot = path.resolve(__dirname, "../..");
const agentSourcePath = path.join(repoRoot, "src/npa/cli/agent.py");
const agentUiPath = path.join(repoRoot, "src/npa/cli/agent_ui.html");
const generatedDir = path.join(__dirname, ".generated");
const generatedUiPath = path.join(generatedDir, "agent-ui.html");
// Real @foxglove/embed browser build (devDependency) + the repo's glue module,
// served exactly the way the agent VM serves them (/foxglove/sdk, /foxglove/app).
const foxgloveSdkDir = path.join(__dirname, "node_modules/@foxglove/embed/dist");
const foxgloveHostModulePath = path.join(
  repoRoot,
  "src/npa/cli/assets/foxglove/npa-foxglove-host.js"
);

function extractPythonConstant(source, name, fallback) {
  const re = new RegExp(`^${name}\\s*=\\s*"([^"]*)"`, "m");
  const match = source.match(re);
  return match ? match[1] : fallback;
}

function generateAgentUiHtml() {
  const source = fs.readFileSync(agentSourcePath, "utf8");
  const replacements = {
    AGENT_UI_VERSION: extractPythonConstant(source, "AGENT_UI_VERSION", "dev"),
    DEFAULT_AGENT_USER: extractPythonConstant(source, "DEFAULT_AGENT_USER", "npa"),
    DEFAULT_LLM_MODEL: extractPythonConstant(source, "DEFAULT_LLM_MODEL", "nvidia/Cosmos3-Super-Reasoner"),
  };
  let html;
  if (fs.existsSync(agentUiPath)) {
    // Preferred: UI lives in agent_ui.html (normal braces, no f-string doubling).
    html = fs.readFileSync(agentUiPath, "utf8");
  } else {
    const match = source.match(
      /cat <<'HTML' \| sudo tee \/opt\/npa-agent\/ui\.html >\/dev\/null\n([\s\S]*?)\nHTML/
    );
    if (!match) {
      throw new Error(`Unable to extract NPA agent UI from ${agentSourcePath} or ${agentUiPath}`);
    }
    html = match[1];
    // Legacy inline heredoc lived inside a Python f-string.
    html = html.replaceAll("{{", "{").replaceAll("}}", "}");
    html = html.replace(/\\\\/g, "\\");
  }
  if (html.includes("__NPA_AGENT_UI_HTML__")) {
    throw new Error("UI heredoc is a placeholder; agent_ui.html is required");
  }
  for (const [name, value] of Object.entries(replacements)) {
    html = html.replaceAll(`{${name}}`, value);
  }
  fs.mkdirSync(generatedDir, { recursive: true });
  fs.writeFileSync(generatedUiPath, html, "utf8");
  return html;
}

function startMockServer(port) {
  const html = generateAgentUiHtml();
  const server = http.createServer((req, res) => {
    const url = new URL(req.url || "/", `http://127.0.0.1:${port}`);
    if (url.pathname === "/" || url.pathname === "/ui.html") {
      res.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      });
      res.end(html);
      return;
    }
    if (url.pathname === "/rerun/" || url.pathname === "/rerun") {
      const fixturePath = path.join(__dirname, "cypress/fixtures/mock_rerun.html");
      const mockHtml = fs.readFileSync(fixturePath, "utf8");
      res.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      });
      res.end(mockHtml);
      return;
    }
    if (url.pathname === "/rerun/re_viewer.js") {
      res.writeHead(200, { "content-type": "application/javascript" });
      res.end("window.__NPA_MOCK_RERUN_JS__ = true;");
      return;
    }
    if (url.pathname === "/rerun/re_viewer_bg.wasm") {
      res.writeHead(200, { "content-type": "application/wasm" });
      res.end(Buffer.from([0x00, 0x61, 0x73, 0x6d]));
      return;
    }
    if (/^\/rerun\/recordings\/cap-[A-Za-z0-9_-]{43}\.rrd$/.test(url.pathname)) {
      res.writeHead(200, { "content-type": "application/octet-stream" });
      res.end(Buffer.alloc(128, 1));
      return;
    }
    if (url.pathname === "/rerun/recordings/sim2real.rrd") {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("recording capability required");
      return;
    }
    // Foxglove embed assets: the REAL @foxglove/embed npm build (devDependency),
    // the repo's shared glue module, and a protocol-accurate stand-in for the
    // Foxglove application (the licensed viewer cannot run in CI).
    if (url.pathname.startsWith("/foxglove/sdk/")) {
      const name = path.basename(url.pathname);
      const sdkFile = path.join(foxgloveSdkDir, name);
      if (!fs.existsSync(sdkFile)) {
        res.writeHead(404, { "content-type": "text/plain" });
        res.end(`missing SDK asset: ${name} (run npm install in npa/tests/browser)`);
        return;
      }
      res.writeHead(200, {
        "content-type": "application/javascript; charset=utf-8",
        "cache-control": "no-store",
      });
      res.end(fs.readFileSync(sdkFile));
      return;
    }
    if (url.pathname === "/foxglove/app/npa-foxglove-host.js") {
      res.writeHead(200, {
        "content-type": "application/javascript; charset=utf-8",
        "cache-control": "no-store",
      });
      res.end(fs.readFileSync(foxgloveHostModulePath));
      return;
    }
    if (url.pathname === "/mock-foxglove-app/" || url.pathname === "/mock-foxglove-app") {
      const fixturePath = path.join(__dirname, "cypress/fixtures/mock_foxglove_app.html");
      res.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      });
      res.end(fs.readFileSync(fixturePath, "utf8"));
      return;
    }
    if (url.pathname === "/foxglove/data/" || url.pathname.startsWith("/foxglove/data/")) {
      // MCAP magic + filler; enough for a range-capable static response.
      const body = Buffer.concat([
        Buffer.from([0x89, 0x4d, 0x43, 0x41, 0x50, 0x30, 0x0d, 0x0a]),
        Buffer.alloc(120, 7),
      ]);
      res.writeHead(200, {
        "content-type": "application/octet-stream",
        "accept-ranges": "bytes",
        "access-control-allow-origin": "*",
      });
      res.end(body);
      return;
    }
    if (url.pathname === "/lichtblick/" || url.pathname === "/lichtblick") {
      const fixturePath = path.join(__dirname, "cypress/fixtures/mock_lichtblick.html");
      const mockHtml = fs.readFileSync(fixturePath, "utf8");
      res.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      });
      res.end(mockHtml);
      return;
    }
    if (url.pathname === "/lichtblick/recordings/sim2real.mcap") {
      // Serve a real (small) MCAP fixture with genuine camera topics so smoke
      // tests can assert the viewer is fed substantive data, not a stub.
      const fixtureMcap = path.join(__dirname, "cypress/fixtures/sim2real_sample.mcap");
      res.writeHead(200, {
        "content-type": "application/octet-stream",
        "accept-ranges": "bytes",
      });
      res.end(fs.readFileSync(fixtureMcap));
      return;
    }
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end(`not found: ${url.pathname}`);
  });
  server.listen(port, "127.0.0.1");
  return server;
}

function liveCredentials(config) {
  const read = (name) => String(config.env[name] || "").trim();
  const result = {
    baseUrl: read("agentBaseUrl") || read("NPA_AGENT_BASE_URL"),
    username: read("agentUser") || read("NPA_AGENT_USER"),
    password: read("agentPassword") || read("NPA_AGENT_PASSWORD"),
  };
  if (!result.baseUrl || !result.username || !result.password) {
    throw new Error("real-browser Foxglove verification requires complete owner credentials");
  }
  return result;
}

function screenshotStats(filePath) {
  const png = PNG.sync.read(fs.readFileSync(filePath));
  const colors = new Set();
  let minLuma = 255;
  let maxLuma = 0;
  let opaqueSamples = 0;
  for (let pixel = 0; pixel < png.width * png.height; pixel += 16) {
    const offset = pixel * 4;
    if (png.data[offset + 3] < 220) continue;
    const r = png.data[offset];
    const g = png.data[offset + 1];
    const b = png.data[offset + 2];
    const luma = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
    minLuma = Math.min(minLuma, luma);
    maxLuma = Math.max(maxLuma, luma);
    colors.add(`${r >> 3}:${g >> 3}:${b >> 3}`);
    opaqueSamples += 1;
  }
  return {
    width: png.width,
    height: png.height,
    sampledColors: colors.size,
    opaqueSamples,
    lumaRange: maxLuma - minLuma,
    nonblank: opaqueSamples > 100 && colors.size > 20 && maxLuma - minLuma > 18,
  };
}

async function screenshotLocatorRegion(page, locator, filePath) {
  // Artifact inventory refresh replaces card nodes in place. Resolve its
  // current geometry, then capture that region at page level so evidence does
  // not depend on the same element instance surviving font/layout work inside
  // Playwright's element-screenshot action.
  await locator.waitFor({ state: "visible", timeout: 0 });
  await locator.evaluate((node) => node.scrollIntoView({ block: "center", inline: "nearest" }));
  const box = await locator.boundingBox();
  const viewport = page.viewportSize();
  if (!box || !viewport) throw new Error("artifact-card evidence region is unavailable");
  const x = Math.max(0, box.x);
  const y = Math.max(0, box.y);
  const width = Math.min(viewport.width, box.x + box.width) - x;
  const height = Math.min(viewport.height, box.y + box.height) - y;
  if (width <= 0 || height <= 0) throw new Error("artifact-card evidence region has zero geometry");
  await page.screenshot({ path: filePath, clip: { x, y, width, height } });
  fs.chmodSync(filePath, 0o600);
}

function downloadPublicMcap(url, destination, redirects = 0) {
  if (redirects > 3) return Promise.reject(new Error("public MCAP redirected too many times"));
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https : http;
    const request = client.get(parsed, { rejectUnauthorized: false }, (response) => {
      if ([301, 302, 303, 307, 308].includes(response.statusCode) && response.headers.location) {
        response.resume();
        const next = new URL(response.headers.location, parsed).toString();
        resolve(downloadPublicMcap(next, destination, redirects + 1));
        return;
      }
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`public MCAP download returned HTTP ${response.statusCode}`));
        return;
      }
      const output = fs.createWriteStream(destination, { mode: 0o600 });
      response.pipe(output);
      output.on("finish", () => output.close(resolve));
      output.on("error", reject);
    });
    request.on("error", reject);
  });
}

async function validatePublishedMcap(taskInput) {
  const recordingUrl = String((taskInput && taskInput.recordingUrl) || "");
  const expectedSha = String((taskInput && taskInput.sha256) || "").toLowerCase();
  if (!recordingUrl.match(/^https:\/\//) || !expectedSha.match(/^[a-f0-9]{64}$/)) {
    throw new Error("public MCAP validation requires HTTPS URL and SHA-256");
  }
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "npa-foxglove-live-"));
  fs.chmodSync(tempDir, 0o700);
  const mcapPath = path.join(tempDir, "recording.mcap");
  try {
    await downloadPublicMcap(recordingUrl, mcapPath);
    const actualSha = crypto.createHash("sha256").update(fs.readFileSync(mcapPath)).digest("hex");
    if (actualSha !== expectedSha) throw new Error("public MCAP SHA-256 does not match export");
    const result = spawnSync(
      path.join(repoRoot, ".venv/bin/python"),
      [path.join(repoRoot, "tests/browser/scripts/validate_published_mcap.py"), mcapPath],
      { encoding: "utf8", env: { ...process.env, PYTHONUNBUFFERED: "1" } },
    );
    if (result.status !== 0) {
      throw new Error(`public MCAP validator failed: ${String(result.stderr || "").trim()}`);
    }
    return JSON.parse(result.stdout);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

async function verifyFoxgloveHostedNavigation(config, taskInput) {
  const credentials = liveCredentials(config);
  const executablePath = String(
    process.env.NPA_PLAYWRIGHT_CHROMIUM_EXECUTABLE || "",
  ).trim();
  const evidenceDir = path.resolve(
    String(process.env.NPA_AGENT_CYPRESS_EVIDENCE_DIR || ""),
  );
  const runId = String((taskInput && taskInput.runId) || "").trim();
  const runRef = String((taskInput && taskInput.runRef) || "").trim();
  const artifactKey = String((taskInput && taskInput.artifactKey) || "").trim();
  const expectedProjectId = String((taskInput && taskInput.projectId) || "").trim();
  const expectedResourceBucket = String(
    (taskInput && taskInput.resourceBucket) || "",
  ).trim();
  const expectedResolvedPrefix = String(
    (taskInput && taskInput.resolvedPrefix) || "",
  ).trim();
  if (!executablePath || !fs.existsSync(executablePath)) {
    throw new Error("real-browser Foxglove verification requires a Chromium executable");
  }
  if (!runId.match(/^[A-Za-z0-9][A-Za-z0-9._-]*$/)) {
    throw new Error("real-browser Foxglove verification requires a safe run id");
  }
  if (!runRef.match(/^npa1_[A-Za-z0-9_-]+$/)) {
    throw new Error("real-browser Foxglove verification requires a source-qualified run ref");
  }
  if (!artifactKey.endsWith("/reports/sim2real.mcap")) {
    throw new Error("real-browser Foxglove verification requires the exact canonical MCAP key");
  }
  if (!process.env.NPA_AGENT_CYPRESS_EVIDENCE_DIR || evidenceDir.startsWith(repoRoot)) {
    throw new Error("real-browser evidence directory must be explicit and outside the clone");
  }
  fs.mkdirSync(evidenceDir, { recursive: true, mode: 0o700 });
  fs.chmodSync(evidenceDir, 0o700);
  const progressPath = path.join(evidenceDir, "live-hosted-progress.json");
  const recordProgress = (phase) => {
    fs.writeFileSync(
      progressPath,
      `${JSON.stringify({ phase, at: new Date().toISOString() }, null, 2)}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    fs.chmodSync(progressPath, 0o600);
  };
  recordProgress("browser_launch");

  const browser = await chromium.launch({
    executablePath,
    headless: true,
    args: ["--ignore-certificate-errors", "--disable-gpu-blocklist"],
  });
  const context = await browser.newContext({
    httpCredentials: {
      username: credentials.username,
      password: credentials.password,
    },
    ignoreHTTPSErrors: true,
    serviceWorkers: "block",
    viewport: { width: 1440, height: 1000 },
  });
  const officialRequests = [];
  const officialResponses = [];
  const agentExportRequests = [];
  context.on("request", (request) => {
    try {
      const url = new URL(request.url());
      if (url.origin === "https://app.foxglove.dev") officialRequests.push(request.url());
      if (url.pathname === "/api/foxglove/export" && request.method() === "POST") {
        let body = {};
        try { body = request.postDataJSON() || {}; } catch (_error) { /* keep empty */ }
        agentExportRequests.push({
          key: String(body.key || ""),
          openWeb: body.open_web === true,
        });
      }
    } catch (_error) { /* ignore non-URL requests */ }
  });
  context.on("response", (response) => {
    try {
      const url = new URL(response.url());
      if (url.origin === "https://app.foxglove.dev") {
        officialResponses.push({ url: response.url(), status: response.status() });
      }
    } catch (_error) { /* ignore non-URL responses */ }
  });

  try {
    const page = await context.newPage();
    await page.goto(credentials.baseUrl, { waitUntil: "domcontentloaded" });
    recordProgress("page_loaded");
    await page.locator("#tabRerun").click();
    // The parent live scenario already exercises the visible run selector.
    // Load this server-qualified source directly for the second clean browser
    // so hosted-navigation verification does not repeat a tenant-wide run
    // discovery before it can click the exact artifact card.
    await page.waitForFunction(
      () => typeof window.npaAgentArtifacts?.loadExactSource === "function",
      null,
      { timeout: 0 },
    );
    await page.evaluate((selection) => {
      window.__npaHostedExactSourceLoadError = "";
      window.__npaHostedExactSourceLoadPromise = window.npaAgentArtifacts
        .loadExactSource(selection)
        .catch((error) => {
          window.__npaHostedExactSourceLoadError = String(
            error && error.message || error,
          );
          return false;
        });
      return true;
    }, {
      run_id: runId,
      run_ref: runRef,
      resource_bucket: expectedResourceBucket,
      project_id: expectedProjectId,
      resolved_prefix: expectedResolvedPrefix,
    });
    const exactButton = page.locator(
      `button[data-action="open-foxglove-artifact"][data-key=${JSON.stringify(artifactKey)}]`,
    );
    await exactButton.waitFor({ state: "visible", timeout: 0 });
    await page.waitForFunction(
      (key) => {
        const button = [...document.querySelectorAll(
          "button[data-action='open-foxglove-artifact']",
        )].find((candidate) => candidate.getAttribute("data-key") === key);
        return Boolean(window.__npaHostedExactSourceLoadError) ||
          Boolean(button && !button.disabled && button.getAttribute("aria-disabled") === "false");
      },
      artifactKey,
      { timeout: 0 },
    );
    const exactSourceLoadError = await page.evaluate(
      () => String(window.__npaHostedExactSourceLoadError || ""),
    );
    if (exactSourceLoadError) {
      throw new Error(
        `hosted exact server-qualified artifact source failed: ${exactSourceLoadError}`,
      );
    }
    recordProgress("exact_source_loaded");
    const artifactCard = exactButton.locator(
      "xpath=ancestor::*[contains(concat(' ', normalize-space(@class), ' '), ' artifact-card ')][1]",
    );
    const cardLabels = async () => (await artifactCard.locator(".artifact-card-actions .btn")
      .allTextContents()).map((value) => value.trim());
    const desktopCardLabels = await cardLabels();
    if (desktopCardLabels.join("|") !== "View in Foxglove|View in Lichtblick|Download") {
      throw new Error("desktop MCAP artifact card does not expose all three actions in order");
    }
    const desktopCardEvidence = path.join(
      evidenceDir,
      "live-hosted-preflight-artifact-card-desktop.png",
    );
    await screenshotLocatorRegion(page, artifactCard, desktopCardEvidence);
    const cardUrl = page.url();
    const cardPageCount = context.pages().length;
    const cardExportResponsePromise = page.waitForResponse(
      (response) => response.url().includes("/api/foxglove/export") &&
        response.request().method() === "POST",
      { timeout: 0 },
    );
    await exactButton.click();
    const cardExportResponse = await cardExportResponsePromise;
    if (cardExportResponse.status() !== 200) {
      throw new Error("artifact-card Foxglove export did not return HTTP 200");
    }
    const cardPayload = await cardExportResponse.json();
    recordProgress("card_export_received");
    const cardSource = String(cardPayload.export?.recording_url || "");
    if (
      String(cardPayload.selected_artifact?.run_id || "") !== runId ||
      String(cardPayload.selected_artifact?.key || "") !== artifactKey
    ) {
      throw new Error("artifact-card Foxglove export lost exact run/artifact provenance");
    }
    await page.waitForFunction(
      ({ key, source }) => {
        const pane = document.querySelector("#viewerPaneFoxglove");
        const host = document.querySelector("#foxgloveHost");
        const status = String(document.querySelector("#foxgloveStatus")?.textContent || "");
        const truthfulHostedState = host?.dataset.sdkReady === "true" ||
          /queued in the official Foxglove SDK/i.test(status);
        return pane?.dataset.artifactKey === key && pane?.dataset.recordingUrl === source &&
          truthfulHostedState && host?.dataset.dataSourceUrl === source;
      },
      { key: artifactKey, source: cardSource },
      { timeout: 0 },
    );
    if (page.url() !== cardUrl || context.pages().length !== cardPageCount) {
      throw new Error("artifact-card View in Foxglove created a target or navigated the Agent page");
    }
    recordProgress("card_source_ready");
    if (agentExportRequests.filter(
      ({ key, openWeb }) => key === artifactKey && !openWeb,
    ).length !== 1) {
      throw new Error("artifact-card View in Foxglove did not issue exactly one export request");
    }
    const expectedLabels = ["View", "Foxglove", "Lichtblick", "Video", "Image", "Data"];
    const desktopLabels = await page.locator(".render-mode-tabs .render-mode-tab").allTextContents();
    if (desktopLabels.map((value) => value.trim()).join("|") !== expectedLabels.join("|")) {
      throw new Error("deployed viewer tab order does not match the required labels");
    }
    const paneGeometry = { Foxglove: {} };
    const desktopPaneBox = await page.locator("#viewerPaneFoxglove").boundingBox();
    if (!desktopPaneBox || desktopPaneBox.width <= 0 || desktopPaneBox.height <= 0) {
      throw new Error("deployed Foxglove pane has zero geometry");
    }
    paneGeometry.Foxglove.desktop = {
      width: Math.round(desktopPaneBox.width),
      height: Math.round(desktopPaneBox.height),
    };
    await page.locator("#viewerPaneFoxglove iframe").waitFor({
      state: "visible",
      timeout: 0,
    });
    await page.waitForFunction(
      () => {
        const summary = String(
          document.querySelector("#foxgloveVisualizationSummary")?.textContent || "",
        );
        return summary.includes("robot + trajectory 3D") &&
          summary.includes("not calibrated robot/world kinematics");
      },
      null,
      { timeout: 0 },
    );
    const desktopEvidence = path.join(evidenceDir, "live-hosted-preflight-desktop.png");
    await page.screenshot({ path: desktopEvidence });
    fs.chmodSync(desktopEvidence, 0o600);

    await page.setViewportSize({ width: 390, height: 844 });
    await exactButton.scrollIntoViewIfNeeded();
    const mobileCardLabels = await cardLabels();
    if (mobileCardLabels.join("|") !== "View in Foxglove|View in Lichtblick|Download") {
      throw new Error("mobile MCAP artifact card does not expose all three actions in order");
    }
    const mobileCardBox = await artifactCard.boundingBox();
    if (!mobileCardBox || mobileCardBox.width <= 0 || mobileCardBox.height <= 0) {
      throw new Error("mobile MCAP artifact card has zero geometry");
    }
    const mobileCardEvidence = path.join(
      evidenceDir,
      "live-hosted-preflight-artifact-card-mobile.png",
    );
    await screenshotLocatorRegion(page, artifactCard, mobileCardEvidence);
    const mobileLabels = await page.locator(".render-mode-tabs .render-mode-tab").allTextContents();
    if (mobileLabels.map((value) => value.trim()).join("|") !== expectedLabels.join("|")) {
      throw new Error("mobile deployed viewer tab order does not match the required labels");
    }
    const mobilePaneBox = await page.locator("#viewerPaneFoxglove").boundingBox();
    if (!mobilePaneBox || mobilePaneBox.width <= 0 || mobilePaneBox.height <= 0) {
      throw new Error("mobile deployed Foxglove pane has zero geometry");
    }
    paneGeometry.Foxglove.mobile = {
      width: Math.round(mobilePaneBox.width),
      height: Math.round(mobilePaneBox.height),
    };
    const mobileEvidence = path.join(evidenceDir, "live-hosted-preflight-mobile.png");
    await page.locator("section.rerun-stage").screenshot({ path: mobileEvidence });
    fs.chmodSync(mobileEvidence, 0o600);
    recordProgress("preflight_complete");

    await page.setViewportSize({ width: 1440, height: 1000 });
    const popupPromise = context.waitForEvent("page");
    const exportResponsePromise = page.waitForResponse(
      (response) => {
        if (!response.url().includes("/api/foxglove/export") ||
            response.request().method() !== "POST") return false;
        try {
          const body = response.request().postDataJSON() || {};
          return body.open_web === true && String(body.key || "") === artifactKey;
        } catch (_error) {
          return false;
        }
      },
      { timeout: 0 },
    );
    const externalAction = page.locator("#foxgloveOpenWeb");
    if (String(await externalAction.textContent() || "").trim() !== "Open in Foxglove") {
      throw new Error("Foxglove pane does not separate the hosted navigation action");
    }
    await externalAction.click();
    const popup = await popupPromise;
    recordProgress("popup_opened");
    const exportResponse = await exportResponsePromise;
    recordProgress("web_export_received");
    if (exportResponse.status() !== 200) {
      throw new Error("real-click Foxglove export did not return HTTP 200");
    }
    const exportPayload = await exportResponse.json();
    const selectedExport = exportPayload.selected_artifact || {};
    const requestPayload = exportResponse.request().postDataJSON();
    const exactRunMatched = String(selectedExport.run_id || "") === runId;
    const exactKeyMatched = String(selectedExport.key || "") === artifactKey;
    const requestKeyMatched = String(requestPayload.key || "") === artifactKey;
    const transportCacheReused = exportPayload.cache_reused === true;
    if (
      !exactRunMatched || !exactKeyMatched || !requestKeyMatched ||
      !transportCacheReused
    ) {
      throw new Error(
        "real-click Foxglove export did not reuse the exact selected artifact " +
        `(run=${exactRunMatched}, key=${exactKeyMatched}, request=${requestKeyMatched}, ` +
        `cache=${transportCacheReused})`,
      );
    }
    const expectedWebUrl = String(exportPayload.export?.web_url || "");
    if (!expectedWebUrl.startsWith("https://app.foxglove.dev/~/view?")) {
      throw new Error("real-click Foxglove export did not produce an official remote-file URL");
    }
    await popup.waitForURL(
      (url) => url.origin === "https://app.foxglove.dev",
      // A cross-origin hosted page may keep third-party resources pending and
      // never emit a full load event. The committed top-level navigation is
      // the authoritative real-popup gate; the nonblank screenshot below
      // separately proves that Foxglove rendered its reachable surface.
      { timeout: 0, waitUntil: "commit" },
    );
    recordProgress("popup_committed");
    const exactExportRequestCount = agentExportRequests.filter(
      ({ key }) => key === artifactKey,
    ).length;
    if (exactExportRequestCount !== 2) {
      throw new Error("Open in Foxglove issued an unexpected number of export requests");
    }
    const requestedUrl = officialRequests.find((value) => value === expectedWebUrl) || "";
    if (!requestedUrl) {
      throw new Error("real popup did not request the exact official response URL");
    }
    const parsed = new URL(requestedUrl);
    const dataUrls = parsed.searchParams.getAll("ds.url");
    if (
      parsed.pathname !== "/~/view" ||
      parsed.searchParams.get("ds") !== "remote-file" ||
      dataUrls.length !== 1
    ) {
      throw new Error("real popup did not use the official remote-file contract");
    }
    const recordingUrl = new URL(dataUrls[0]);
    if (recordingUrl.protocol !== "https:" || !recordingUrl.pathname.endsWith(".mcap")) {
      throw new Error("real popup did not carry one absolute HTTPS MCAP URL");
    }
    if (/%25(?:2f|3a)/i.test(parsed.search)) {
      throw new Error("real popup double-encoded the recording URL");
    }
    const hostedEvidence = path.join(evidenceDir, "live-hosted-foxglove-after.png");
    let pixels = { nonblank: false };
    while (!pixels.nonblank) {
      // Navigation commit can precede the cross-origin application's first
      // paint by many seconds. Keep sampling the owner-only evidence path
      // until Foxglove renders a real sign-in/viewer/error surface; a blank
      // placeholder is never accepted as hosted visual evidence.
      await popup.screenshot({ path: hostedEvidence });
      fs.chmodSync(hostedEvidence, 0o600);
      pixels = screenshotStats(hostedEvidence);
      if (!pixels.nonblank) await popup.waitForTimeout(500);
    }
    recordProgress("hosted_surface_ready");
    const response = officialResponses.find((item) => item.url === expectedWebUrl) ||
      officialResponses[0] || { status: 0 };
    const result = {
      runId,
      artifactKey,
      labels: expectedLabels,
      paneGeometry,
      cardNavigation: {
        stayedInPage: true,
        pagesBefore: cardPageCount,
        pagesAfter: context.pages().length - 1,
      },
      artifactCard: {
        desktopLabels: desktopCardLabels,
        mobileLabels: mobileCardLabels,
      },
      officialContract: {
        requestMatchedResponse: true,
        responseStatus: response.status,
        sourceType: parsed.searchParams.get("ds"),
        oneAbsoluteHttpsMcap: true,
        encodedExactlyOnce: true,
        layoutIdPresent: Boolean(parsed.searchParams.get("layoutId")),
        exactTransportCacheReused: true,
        exportRequestCount: exactExportRequestCount,
        serverTimingsMs: exportPayload.timings_ms || {},
      },
      hostedSurface: {
        finalOrigin: new URL(popup.url()).origin,
        finalPath: new URL(popup.url()).pathname,
        pixels,
      },
      evidence: {
        desktop: desktopEvidence,
        mobile: mobileEvidence,
        hosted: hostedEvidence,
        artifactCardDesktop: desktopCardEvidence,
        artifactCardMobile: mobileCardEvidence,
      },
    };
    recordProgress("complete");
    return result;
  } finally {
    await context.close();
    await browser.close();
  }
}

async function foxgloveControlClearance(page) {
  return await page.evaluate(() => {
    const frame = document.querySelector("#viewerPaneFoxglove iframe");
    if (!frame) throw new Error("Foxglove iframe is missing for control-clearance proof");
    const iframe = frame.getBoundingClientRect();
    const controls = {
      left: iframe.left,
      right: iframe.right,
      top: iframe.bottom - Math.min(80, iframe.height),
      bottom: iframe.bottom,
    };
    const overlaps = (rect) =>
      Math.max(rect.left, controls.left) < Math.min(rect.right, controls.right) &&
      Math.max(rect.top, controls.top) < Math.min(rect.bottom, controls.bottom);
    const elements = {};
    for (const selector of ["#foxgloveStatus", "#statusBar", "#chatDrawerToggle"]) {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`${selector} is missing for control-clearance proof`);
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      const visible = style.display !== "none" && style.visibility !== "hidden";
      elements[selector] = {
        position: style.position,
        visible,
        overlapsControls: visible && overlaps(rect),
      };
    }
    if (elements["#foxgloveStatus"].position !== "static" ||
        elements["#statusBar"].position !== "static") {
      throw new Error("Foxglove status surfaces do not participate in normal layout");
    }
    if (Object.values(elements).some((item) => item.overlapsControls)) {
      throw new Error("an Agent UI element overlaps the Foxglove playback-control region");
    }
    return {
      controls: {
        width: Math.round(controls.right - controls.left),
        height: Math.round(controls.bottom - controls.top),
      },
      elements,
      unobstructed: true,
    };
  });
}

async function verifyFoxgloveEmbeddedArtifact(config, taskInput) {
  const credentials = liveCredentials(config);
  const executablePath = String(process.env.NPA_PLAYWRIGHT_CHROMIUM_EXECUTABLE || "").trim();
  const evidenceDir = path.resolve(String(process.env.NPA_AGENT_CYPRESS_EVIDENCE_DIR || ""));
  const runId = String((taskInput && taskInput.runId) || "").trim();
  const runRef = String((taskInput && taskInput.runRef) || "").trim();
  const artifactKey = String((taskInput && taskInput.artifactKey) || "").trim();
  const expectedProjectId = String((taskInput && taskInput.projectId) || "").trim();
  const expectedResourceBucket = String((taskInput && taskInput.resourceBucket) || "").trim();
  const expectedResolvedPrefix = String((taskInput && taskInput.resolvedPrefix) || "").trim();
  const expectedS3Uri = String((taskInput && taskInput.s3Uri) || "").trim();
  if (!executablePath || !fs.existsSync(executablePath)) {
    throw new Error("real-browser embedded Foxglove verification requires Chromium");
  }
  if (!runId.match(/^[A-Za-z0-9][A-Za-z0-9._-]*$/) || !runRef.match(/^npa1_[A-Za-z0-9_-]+$/)) {
    throw new Error("real-browser embedded Foxglove verification requires an exact run selector");
  }
  if (!artifactKey.endsWith(".mcap")) {
    throw new Error("real-browser embedded Foxglove verification requires an MCAP key");
  }
  if (!process.env.NPA_AGENT_CYPRESS_EVIDENCE_DIR || evidenceDir.startsWith(repoRoot)) {
    throw new Error("real-browser evidence directory must be explicit and outside the clone");
  }
  fs.mkdirSync(evidenceDir, { recursive: true, mode: 0o700 });
  fs.chmodSync(evidenceDir, 0o700);
  const progressPath = path.join(evidenceDir, "live-embedded-progress.json");
  const recordProgress = (phase) => {
    fs.writeFileSync(progressPath, `${JSON.stringify({ phase, at: new Date().toISOString() }, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
    });
    fs.chmodSync(progressPath, 0o600);
  };
  recordProgress("browser_launch");

  const browser = await chromium.launch({
    executablePath,
    headless: true,
    args: ["--ignore-certificate-errors", "--disable-gpu-blocklist"],
  });
  const context = await browser.newContext({
    httpCredentials: { username: credentials.username, password: credentials.password },
    ignoreHTTPSErrors: true,
    serviceWorkers: "block",
    viewport: { width: 1440, height: 1000 },
  });
  const networkRequests = [];
  context.on("request", (request) => {
    networkRequests.push({
      url: request.url(),
      method: request.method(),
      range: String(request.headers().range || ""),
    });
  });
  try {
    const page = await context.newPage();
    page.setDefaultTimeout(0);
    page.setDefaultNavigationTimeout(0);
    const newTargets = [];
    context.on("page", (target) => {
      if (target !== page) {
        const url = new URL(target.url());
        newTargets.push({ origin: url.origin, path: url.pathname });
      }
    });
    await page.goto(credentials.baseUrl, { waitUntil: "domcontentloaded", timeout: 0 });
    recordProgress("page_loaded");
    await page.locator("#tabRerun").click();
    await page.waitForFunction(
      () => typeof window.npaAgentArtifacts?.loadExactSource === "function",
      null,
      { timeout: 0 },
    );
    await page.evaluate((selection) => {
      window.__npaExactSourceLoadError = "";
      window.__npaExactSourceLoadPromise = window.npaAgentArtifacts.loadExactSource(selection)
        .catch((error) => {
          window.__npaExactSourceLoadError = String(error && error.message || error);
          return false;
        });
      return true;
    }, {
      run_id: runId,
      run_ref: runRef,
      resource_bucket: expectedResourceBucket,
      project_id: expectedProjectId,
      resolved_prefix: expectedResolvedPrefix,
    });
    recordProgress("artifact_list_requested");
    const exactButton = page.locator(
      `button[data-action="open-foxglove-artifact"][data-key=${JSON.stringify(artifactKey)}]`,
    );
    await exactButton.waitFor({ state: "visible", timeout: 0 });
    await page.waitForFunction(
      (key) => {
        const button = [...document.querySelectorAll("button[data-action='open-foxglove-artifact']")]
          .find((candidate) => candidate.getAttribute("data-key") === key);
        return Boolean(window.__npaExactSourceLoadError) ||
          Boolean(button && !button.disabled && button.getAttribute("aria-disabled") === "false");
      },
      artifactKey,
      { timeout: 0 },
    );
    const exactSourceLoadError = await page.evaluate(
      () => String(window.__npaExactSourceLoadError || ""),
    );
    if (exactSourceLoadError) {
      throw new Error(`exact server-qualified artifact source failed: ${exactSourceLoadError}`);
    }
    recordProgress("artifact_card_ready");
    const artifactCard = exactButton.locator(
      "xpath=ancestor::*[contains(concat(' ', normalize-space(@class), ' '), ' artifact-card ')][1]",
    );
    const labels = (await artifactCard.locator(".artifact-card-actions .btn").allTextContents())
      .map((value) => value.trim());
    if (labels.join("|") !== "View in Foxglove|View in Lichtblick|Download") {
      throw new Error("live MCAP artifact card action order is incorrect");
    }
    const desktopCardEvidence = path.join(evidenceDir, "live-artifact-card-desktop-after.png");
    await screenshotLocatorRegion(page, artifactCard, desktopCardEvidence);

    const beforeUrl = page.url();
    const beforePages = context.pages().length;
    const responsePromise = page.waitForResponse(
      (response) => response.url().includes("/api/foxglove/export") &&
        response.request().method() === "POST",
      { timeout: 0 },
    );
    const clickStarted = Date.now();
    const iframeVisiblePromise = page.locator("#viewerPaneFoxglove iframe")
      .waitFor({ state: "visible", timeout: 0 })
      .then(() => Date.now());
    await exactButton.click();
    recordProgress("artifact_card_clicked");
    await page.waitForFunction(
      () => document.querySelector("#renderModeFoxglove")?.getAttribute("aria-selected") === "true" &&
        document.querySelector("#viewerPaneFoxglove")?.getAttribute("aria-hidden") === "false",
      null,
      { timeout: 0 },
    );
    const paneVisibleAt = Date.now();
    const exportResponse = await responsePromise;
    const apiResponseAt = Date.now();
    recordProgress("export_response_received");
    if (exportResponse.status() !== 200) {
      throw new Error(`live artifact-card export returned HTTP ${exportResponse.status()}`);
    }
    const exportPayload = await exportResponse.json();
    const selected = exportPayload.selected_artifact || exportPayload.export?.selected_artifact || {};
    const recordingUrl = String(exportPayload.export?.recording_url || "");
    const sha256 = String(exportPayload.export?.sha256 || "");
    const requestPayload = exportResponse.request().postDataJSON();
    for (const [label, actual, expected] of [
      ["run", selected.run_id, runId],
      ["run ref", selected.run_ref, runRef],
      ["key", selected.key, artifactKey],
      ["request run", requestPayload.run_id, runId],
      ["request run ref", requestPayload.run_ref, runRef],
      ["request key", requestPayload.key, artifactKey],
      ["request project", requestPayload.project_id, expectedProjectId],
      ["request resource bucket", requestPayload.resource_bucket, expectedResourceBucket],
      ["request resolved prefix", requestPayload.resolved_prefix, expectedResolvedPrefix],
      ["request S3 URI", requestPayload.s3_uri, expectedS3Uri],
      ["selected project", selected.project_id, expectedProjectId],
      ["selected resource bucket", selected.resource_bucket || selected.bucket, expectedResourceBucket],
      ["selected resolved prefix", selected.resolved_prefix, expectedResolvedPrefix],
      ["selected S3 URI", selected.s3_uri, expectedS3Uri],
    ]) {
      if (String(actual || "") !== String(expected || "")) {
        throw new Error(`live embedded Foxglove ${label} mismatch`);
      }
    }
    if (!/^https:\/\//.test(recordingUrl) || !/^[a-f0-9]{64}$/.test(sha256)) {
      throw new Error("live embedded Foxglove response lacks HTTPS URL/SHA identity");
    }
    await page.waitForFunction(
      ({ key, sha, source }) => {
        const pane = document.querySelector("#viewerPaneFoxglove");
        const host = document.querySelector("#foxgloveHost");
        const status = String(document.querySelector("#foxgloveStatus")?.textContent || "");
        const truthfulHostedState = host?.dataset.sdkReady === "true" ||
          /queued in the official Foxglove SDK/i.test(status);
        return pane?.dataset.artifactKey === key && pane?.dataset.sha256 === sha &&
          pane?.dataset.recordingUrl === source && host?.dataset.dataSourceUrl === source &&
          truthfulHostedState &&
          Number(host?.dataset.setDataSourceCount || 0) === 1;
      },
      { key: artifactKey, sha: sha256, source: recordingUrl },
      { timeout: 0 },
    );
    const dataSourceReadyAt = Date.now();
    recordProgress("data_source_ready");
    const iframe = page.locator("#viewerPaneFoxglove iframe");
    await iframe.waitFor({ state: "visible", timeout: 0 });
    const iframeVisibleAt = await iframeVisiblePromise;
    const paneBox = await page.locator("#viewerPaneFoxglove").boundingBox();
    const iframeBox = await iframe.boundingBox();
    if (!paneBox || paneBox.width <= 0 || paneBox.height <= 0 ||
        !iframeBox || iframeBox.width <= 0 || iframeBox.height <= 0) {
      throw new Error("live embedded Foxglove pane or SDK iframe has zero geometry");
    }
    const state = await page.evaluate(async () => {
      const read = async (url) => {
        const response = await fetch(url, { credentials: "include", cache: "no-store" });
        return await response.json();
      };
      return { config: await read("/api/foxglove/config"), status: await read("/api/foxglove/status") };
    });
    for (const payload of [state.config, state.status]) {
      if (String(payload.run_id || "") !== runId ||
          String(payload.artifact_run_ref || "") !== runRef ||
          String(payload.artifact_key || "") !== artifactKey ||
          String(payload.project_id || "") !== String(selected.project_id || "") ||
          String(payload.resource_bucket || "") !== String(selected.resource_bucket || selected.bucket || "") ||
          String(payload.resolved_prefix || "") !== String(selected.resolved_prefix || "") ||
          String(payload.artifact_sha256 || "") !== sha256 ||
          String(payload.recording_url || "") !== recordingUrl) {
        throw new Error("live Foxglove config/status lost exact selected artifact provenance");
      }
    }
    const topics = state.config.visualization?.topics || {};
    const expectedCameraTopics = ["/camera", "/camera/side", "/camera/workspace"];
    if (!expectedCameraTopics.every((topic) => topics[topic] === "foxglove.CompressedImage")) {
      throw new Error("live Foxglove config lost one or more camera angles");
    }
    if (topics["/robot/diagnostic_scene"] !== "foxglove.SceneUpdate") {
      throw new Error("live Foxglove config lost the diagnostic scene");
    }
    const desktopClearance = await foxgloveControlClearance(page);
    const desktopEvidence = path.join(evidenceDir, "live-agent-desktop-after.png");
    await page.screenshot({ path: desktopEvidence, fullPage: false });
    fs.chmodSync(desktopEvidence, 0o600);

    await page.setViewportSize({ width: 390, height: 844 });
    await artifactCard.scrollIntoViewIfNeeded();
    const mobileLabels = (await artifactCard.locator(".artifact-card-actions .btn").allTextContents())
      .map((value) => value.trim());
    const mobileCardEvidence = path.join(evidenceDir, "live-artifact-card-mobile-after.png");
    await screenshotLocatorRegion(page, artifactCard, mobileCardEvidence);
    const mobilePaneEvidence = path.join(evidenceDir, "live-agent-mobile-after.png");
    await page.locator("section.rerun-stage").screenshot({ path: mobilePaneEvidence });
    fs.chmodSync(mobilePaneEvidence, 0o600);
    const mobilePaneBox = await page.locator("#viewerPaneFoxglove").boundingBox();
    if (!mobilePaneBox || mobilePaneBox.width <= 0 || mobilePaneBox.height <= 0) {
      throw new Error("mobile embedded Foxglove pane has zero geometry");
    }
    const mobileClearance = await foxgloveControlClearance(page);

    // Click the identical exact card again. The backend must prove its cached
    // object version while the browser retains the same iframe/source/layout.
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.evaluate(() => {
      const frame = document.querySelector("#viewerPaneFoxglove iframe");
      if (frame) {
        frame.dataset.npaReuseProbe = "same-frame";
        window.__npaFoxgloveReuseFrame = frame;
      }
    });
    const repeatResponsePromise = page.waitForResponse(
      (response) => response.url().includes("/api/foxglove/export") &&
        response.request().method() === "POST",
      { timeout: 0 },
    );
    const repeatStarted = Date.now();
    await exactButton.click();
    const repeatResponse = await repeatResponsePromise;
    const repeatPayload = await repeatResponse.json();
    if (repeatResponse.status() !== 200 || repeatPayload.cache_reused !== true) {
      throw new Error("repeat exact MCAP click did not use the verified backend cache");
    }
    await page.waitForFunction(
      ({ source }) => {
        const host = document.querySelector("#foxgloveHost");
        const frame = document.querySelector("#viewerPaneFoxglove iframe");
        return frame && frame === window.__npaFoxgloveReuseFrame &&
          frame.dataset.npaReuseProbe === "same-frame" &&
          host?.dataset.dataSourceUrl === source &&
          Number(host?.dataset.setDataSourceCount || 0) === 1 &&
          Number(host?.dataset.layoutSelectCount || 0) === 1 &&
          !host.classList.contains("is-switching");
      },
      { source: recordingUrl },
      { timeout: 0 },
    );
    const repeatReadyAt = Date.now();
    const externalLabel = String(await page.locator("#foxgloveOpenWeb").textContent() || "").trim();
    const statusText = String(await page.locator("#foxgloveStatus").textContent() || "").trim();
    const sdkRequestCount = networkRequests.filter(({ url }) => {
      try {
        const origin = new URL(url).origin;
        return origin === "https://embed.foxglove.dev" ||
          url.includes("/foxglove/sdk/") || url.includes("/foxglove/app/");
      } catch (_error) {
        return false;
      }
    }).length;
    const recordingRequestCount = networkRequests.filter(({ url }) => url === recordingUrl).length;
    const recordingRangeRequestCount = networkRequests.filter(
      ({ url, range }) => url === recordingUrl && Boolean(range),
    ).length;
    const sdkReady = await page.locator("#foxgloveHost").getAttribute("data-sdk-ready");
    const signInRequired = sdkReady !== "true" &&
      /queued in the official Foxglove SDK/i.test(statusText);
    const result = {
      runId,
      runRef,
      artifactKey,
      exactProvenance: {
        projectId: String(selected.project_id || ""),
        resourceBucket: String(selected.resource_bucket || selected.bucket || ""),
        resolvedPrefix: String(selected.resolved_prefix || ""),
        sha256,
        recordingUrlSha256: crypto.createHash("sha256").update(recordingUrl).digest("hex"),
      },
      navigation: {
        topUrlUnchanged: page.url() === beforeUrl,
        pagesBefore: beforePages,
        pagesAfter: context.pages().length,
        newTargets,
      },
      embedded: {
        selected: await page.locator("#renderModeFoxglove").getAttribute("aria-selected"),
        paneAriaHidden: await page.locator("#viewerPaneFoxglove").getAttribute("aria-hidden"),
        pane: { width: Math.round(paneBox.width), height: Math.round(paneBox.height) },
        mobilePane: {
          width: Math.round(mobilePaneBox.width),
          height: Math.round(mobilePaneBox.height),
        },
        iframe: { width: Math.round(iframeBox.width), height: Math.round(iframeBox.height) },
        iframeOrigin: new URL(String(await iframe.getAttribute("src"))).origin,
        setDataSourceCount: Number(await page.locator("#foxgloveHost").getAttribute("data-set-data-source-count") || 0),
        layoutSelectCount: Number(await page.locator("#foxgloveHost").getAttribute("data-layout-select-count") || 0),
        sdkReady,
        signInRequired,
        layoutStorageKey: String(await page.locator("#foxgloveHost").getAttribute("data-layout-storage-key") || ""),
        statusText,
        sdkRequestCount,
        recordingRequestCount,
        recordingRangeRequestCount,
        iframeReused: true,
        controlsUnobstructed: true,
        desktopClearance,
        mobileClearance,
      },
      validation: {
        multipleAnglesVerified: true,
        diagnosticSceneVerified: true,
      },
      performance: {
        cacheReused: exportPayload.cache_reused === true,
        serverTimingsMs: exportPayload.timings_ms || {},
        clickToPaneMs: paneVisibleAt - clickStarted,
        clickToApiMs: apiResponseAt - clickStarted,
        clickToIframeMs: iframeVisibleAt - clickStarted,
        clickToDataSourceReadyMs: dataSourceReadyAt - clickStarted,
        clickToReadySeconds: Number(((dataSourceReadyAt - clickStarted) / 1000).toFixed(3)),
        repeatCacheReused: true,
        repeatServerTimingsMs: repeatPayload.timings_ms || {},
        repeatClickToReadyMs: repeatReadyAt - repeatStarted,
      },
      actions: { artifact: labels, mobileArtifact: mobileLabels, external: externalLabel },
      evidence: {
        desktop: desktopEvidence,
        mobile: mobilePaneEvidence,
        artifactCardDesktop: desktopCardEvidence,
        artifactCardMobile: mobileCardEvidence,
      },
    };
    const resultEvidence = path.join(evidenceDir, "live-embedded-result.json");
    fs.writeFileSync(resultEvidence, `${JSON.stringify(result, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
    });
    fs.chmodSync(resultEvidence, 0o600);
    recordProgress("complete");
    return result;
  } finally {
    await context.close();
    await browser.close();
  }
}

module.exports = defineConfig({
  // The production UI exercises WebGL, media streams, and two viewer SDKs.
  // Release spec state eagerly so long mocked runs stay stable on shared CI
  // hosts instead of accumulating renderer memory.
  experimentalMemoryManagement: true,
  numTestsKeptInMemory: 0,
  e2e: {
    baseUrl: process.env.NPA_AGENT_BASE_URL || `http://127.0.0.1:${process.env.NPA_AGENT_CYPRESS_PORT || 47867}`,
    supportFile: "cypress/support/e2e.js",
    specPattern: "cypress/e2e/**/*.cy.js",
    video: false,
    // Live pages are authenticated. Never persist a live screenshot even when
    // the UI or browser fails; mocked runs keep ordinary screenshot diagnostics.
    screenshotOnRunFailure: process.env.NPA_AGENT_CYPRESS_LIVE === "1" ? false : true,
    chromeWebSecurity: false,
    defaultCommandTimeout: 12000,
    requestTimeout: 30000,
    responseTimeout: 30000,
    setupNodeEvents(on, config) {
    on('before:browser:launch', (browser = {}, launchOptions) => {
      if (browser.family === 'chromium') {
        launchOptions.args.push(
          '--ignore-gpu-blocklist',
          '--use-gl=angle',
          '--use-angle=swiftshader-webgl',
          '--enable-webgl',
          '--enable-unsafe-swiftshader',
          '--disable-web-security'
        );
      }
      return launchOptions;
    });
      let server = null;
      if (!process.env.NPA_AGENT_BASE_URL) {
        server = startMockServer(Number(process.env.NPA_AGENT_CYPRESS_PORT || 47867));
      }
      on("task", {
        verifyFoxgloveEmbeddedArtifact(input) {
          return verifyFoxgloveEmbeddedArtifact(config, input);
        },
        verifyFoxgloveHostedNavigation(input) {
          return verifyFoxgloveHostedNavigation(config, input);
        },
        validatePublishedMcap(input) {
          return validatePublishedMcap(input);
        },
      });
      on("after:run", () => {
        if (server) {
          server.close();
        }
      });
      return config;
    },
  },
});
