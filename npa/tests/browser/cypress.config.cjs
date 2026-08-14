const { defineConfig } = require("cypress");
const fs = require("fs");
const http = require("http");
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

async function verifyFoxgloveHostedNavigation(config, taskInput) {
  const credentials = liveCredentials(config);
  const executablePath = String(
    process.env.NPA_PLAYWRIGHT_CHROMIUM_EXECUTABLE || "",
  ).trim();
  const evidenceDir = path.resolve(
    String(process.env.NPA_AGENT_CYPRESS_EVIDENCE_DIR || ""),
  );
  const runId = String((taskInput && taskInput.runId) || "").trim();
  if (!executablePath || !fs.existsSync(executablePath)) {
    throw new Error("real-browser Foxglove verification requires a Chromium executable");
  }
  if (!runId.match(/^[A-Za-z0-9][A-Za-z0-9._-]*$/)) {
    throw new Error("real-browser Foxglove verification requires a safe run id");
  }
  if (!process.env.NPA_AGENT_CYPRESS_EVIDENCE_DIR || evidenceDir.startsWith(repoRoot)) {
    throw new Error("real-browser evidence directory must be explicit and outside the clone");
  }
  fs.mkdirSync(evidenceDir, { recursive: true, mode: 0o700 });
  fs.chmodSync(evidenceDir, 0o700);

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
  context.on("request", (request) => {
    try {
      const url = new URL(request.url());
      if (url.origin === "https://app.foxglove.dev") officialRequests.push(request.url());
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
    await page.locator("#tabRerun").click();
    await page.waitForFunction(
      (expected) => document.querySelector("#simRunId")?.textContent?.includes(expected),
      runId,
    );
    const expectedLabels = ["View", "Foxglove", "Lichtblick", "Video", "Image", "Data"];
    const desktopLabels = await page.locator(".render-mode-tabs .render-mode-tab").allTextContents();
    if (desktopLabels.map((value) => value.trim()).join("|") !== expectedLabels.join("|")) {
      throw new Error("deployed viewer tab order does not match the required labels");
    }
    const viewerTabs = page.locator(".render-mode-tabs");
    const paneGeometry = {};
    for (const [label, pane] of [
      ["View", "#viewerPaneRerun"],
      ["Foxglove", "#viewerPaneFoxglove"],
      ["Lichtblick", "#viewerPaneLichtblick"],
    ]) {
      await viewerTabs.getByRole("tab", { name: label, exact: true }).click();
      const box = await page.locator(pane).boundingBox();
      if (!box || box.width <= 0 || box.height <= 0) {
        throw new Error(`deployed ${label} pane has zero geometry`);
      }
      paneGeometry[label] = { width: Math.round(box.width), height: Math.round(box.height) };
    }
    await viewerTabs.getByRole("tab", { name: "Foxglove", exact: true }).click();
    await page.locator("#viewerPaneFoxglove iframe").waitFor({ state: "visible" });
    const desktopEvidence = path.join(evidenceDir, "live-agent-desktop-after.png");
    await page.screenshot({ path: desktopEvidence });
    fs.chmodSync(desktopEvidence, 0o600);

    await page.setViewportSize({ width: 390, height: 844 });
    const mobileLabels = await page.locator(".render-mode-tabs .render-mode-tab").allTextContents();
    if (mobileLabels.map((value) => value.trim()).join("|") !== expectedLabels.join("|")) {
      throw new Error("mobile deployed viewer tab order does not match the required labels");
    }
    for (const [label, pane] of [
      ["View", "#viewerPaneRerun"],
      ["Foxglove", "#viewerPaneFoxglove"],
      ["Lichtblick", "#viewerPaneLichtblick"],
    ]) {
      await viewerTabs.getByRole("tab", { name: label, exact: true }).click();
      const box = await page.locator(pane).boundingBox();
      if (!box || box.width <= 0 || box.height <= 0) {
        throw new Error(`mobile deployed ${label} pane has zero geometry`);
      }
    }
    await viewerTabs.getByRole("tab", { name: "Foxglove", exact: true }).click();
    const mobileEvidence = path.join(evidenceDir, "live-agent-mobile-after.png");
    await page.screenshot({ path: mobileEvidence });
    fs.chmodSync(mobileEvidence, 0o600);

    await page.setViewportSize({ width: 1440, height: 1000 });
    const popupPromise = context.waitForEvent("page");
    const exportResponsePromise = page.waitForResponse(
      (response) => response.url().includes("/api/foxglove/export") &&
        response.request().method() === "POST",
    );
    await page.getByRole("button", { name: "View in Foxglove", exact: true }).click();
    const popup = await popupPromise;
    const exportResponse = await exportResponsePromise;
    if (exportResponse.status() !== 200) {
      throw new Error("real-click Foxglove export did not return HTTP 200");
    }
    const exportPayload = await exportResponse.json();
    if (String(exportPayload.run_id || "") !== runId) {
      throw new Error("real-click Foxglove export selected the wrong run");
    }
    const expectedWebUrl = String(exportPayload.export?.web_url || "");
    await popup.waitForURL((url) => url.origin === "https://app.foxglove.dev");
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
    await popup.waitForLoadState("domcontentloaded").catch(() => {});
    await popup.waitForTimeout(2500);
    const hostedEvidence = path.join(evidenceDir, "live-hosted-foxglove-after.png");
    await popup.screenshot({ path: hostedEvidence });
    fs.chmodSync(hostedEvidence, 0o600);
    const pixels = screenshotStats(hostedEvidence);
    if (!pixels.nonblank) {
      throw new Error("hosted Foxglove/sign-in/error surface is visually blank");
    }
    const response = officialResponses.find((item) => item.url === expectedWebUrl) ||
      officialResponses[0] || { status: 0 };
    return {
      runId,
      labels: expectedLabels,
      paneGeometry,
      officialContract: {
        requestMatchedResponse: true,
        responseStatus: response.status,
        sourceType: parsed.searchParams.get("ds"),
        oneAbsoluteHttpsMcap: true,
        encodedExactlyOnce: true,
        layoutIdPresent: Boolean(parsed.searchParams.get("layoutId")),
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
      },
    };
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
        verifyFoxgloveHostedNavigation(input) {
          return verifyFoxgloveHostedNavigation(config, input);
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
