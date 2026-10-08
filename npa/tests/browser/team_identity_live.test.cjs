// Qualify real identity-provider logins against a running Workbench gateway.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const crypto = require("node:crypto");
const { chromium } = require("playwright-core");

const configuration = process.env.NPA_TEAM_IDENTITY_LIVE_CONFIG;

test(
  "real external accounts authenticate and receive only their workspace grants",
  {
    skip: !configuration,
  },
  async () => {
    const settings = JSON.parse(fs.readFileSync(configuration, "utf8"));
    const args = certificateArguments(settings);
    const browser = await chromium.launch({
      channel: "chrome",
      headless: true,
      args,
    });
    try {
      for (const account of settings.accounts) {
        await verifyAccount(browser, settings, account);
      }
    } finally {
      await browser.close();
    }
  },
);

function certificateArguments(settings) {
  if (!settings.local_certificate) return [];
  assert.equal(new URL(settings.endpoint).hostname, "localhost");
  const certificate = new crypto.X509Certificate(
    fs.readFileSync(settings.local_certificate),
  );
  const key = certificate.publicKey.export({ type: "spki", format: "der" });
  return [
    `--ignore-certificate-errors-spki-list=${crypto.createHash("sha256").update(key).digest("base64")}`,
  ];
}

async function verifyAccount(browser, settings, account) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 960 },
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto(settings.endpoint);
    await page.getByRole("link", { name: "Sign in", exact: true }).click();
    if (account === settings.accounts[0])
      await verifyWrongPassword(page, account);
    await page.locator("#username").fill(account.username);
    await page.locator("#password").fill(account.password);
    await page.locator("#kc-login").click();
    await page.waitForURL(settings.endpoint + "/");
    await page.locator("#signed-in").waitFor({ state: "visible" });
    await verifyAccess(page, context, account);
    if (settings.screenshot && account === settings.accounts[0]) {
      await page.screenshot({ path: settings.screenshot });
    }
    await verifyLogout(page);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
}

async function verifyAccess(page, context, account) {
  const profile = await page.evaluate(async () =>
    (await fetch("/v1/me")).json(),
  );
  assert.equal(profile.subject, account.subject);
  assert.equal(profile.display_name, account.username);
  assert.deepEqual(profile.groups.sort(), account.groups.sort());
  const grants = Object.fromEntries(
    profile.workspaces.map((item) => [item.name, item.role]),
  );
  assert.deepEqual(grants, account.workspaces);
  for (const name of Object.keys(account.workspaces)) {
    const status = await page.evaluate(
      async (workspace) =>
        (await fetch(`/v1/runs?workspace=${encodeURIComponent(workspace)}`))
          .status,
      name,
    );
    assert.equal(status, 200);
  }
  await verifySession(page, context);
}

async function verifyWrongPassword(page, account) {
  await page.locator("#username").fill(account.username);
  await page.locator("#password").fill(crypto.randomBytes(24).toString("hex"));
  await page.locator("#kc-login").click();
  await page
    .getByText(/Invalid username or password/i)
    .waitFor({ state: "visible" });
}

async function verifySession(page, context) {
  const denied = await page.evaluate(
    async () => (await fetch("/v1/runs?workspace=no-grant")).status,
  );
  assert.equal(denied, 403);
  const csrf = await page.evaluate(
    async () => (await fetch("/auth/logout", { method: "POST" })).status,
  );
  assert.equal(csrf, 403);
  const cookie = (await context.cookies()).find(
    (item) => item.name === "__Host-workbench-session",
  );
  assert.ok(cookie.httpOnly && cookie.secure && cookie.sameSite === "Lax");
  assert.equal(
    await page.evaluate(() => localStorage.length + sessionStorage.length),
    0,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
  );
  await page.setViewportSize({ width: 1280, height: 960 });
}

async function verifyLogout(page) {
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await page.locator("#signed-out").waitFor({ state: "visible" });
  const status = await page.evaluate(
    async () => (await fetch("/v1/me")).status,
  );
  assert.equal(status, 401);
}
