const { defineConfig } = require("cypress");
const fs = require("fs");
const http = require("http");
const path = require("path");

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
    if (url.pathname === "/rerun/recordings/sim2real.rrd") {
      res.writeHead(200, { "content-type": "application/octet-stream" });
      res.end(Buffer.alloc(128, 1));
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

module.exports = defineConfig({
  e2e: {
    baseUrl: process.env.NPA_AGENT_BASE_URL || `http://127.0.0.1:${process.env.NPA_AGENT_CYPRESS_PORT || 47867}`,
    supportFile: "cypress/support/e2e.js",
    specPattern: "cypress/e2e/**/*.cy.js",
    video: false,
    screenshotOnRunFailure: true,
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
      on("after:run", () => {
        if (server) {
          server.close();
        }
      });
      return config;
    },
  },
});
