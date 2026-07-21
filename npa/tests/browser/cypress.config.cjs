const { defineConfig } = require("cypress");
const fs = require("fs");
const http = require("http");
const path = require("path");

const repoRoot = path.resolve(__dirname, "../..");
const agentSourcePath = path.join(repoRoot, "src/npa/cli/agent.py");
const agentUiPath = path.join(repoRoot, "src/npa/cli/agent_ui.html");
const generatedDir = path.join(__dirname, ".generated");
const generatedUiPath = path.join(generatedDir, "agent-ui.html");

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
