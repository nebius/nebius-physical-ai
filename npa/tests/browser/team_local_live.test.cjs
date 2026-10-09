// Verify real local-account browser sessions against an operator-selected HTTPS gateway.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const crypto = require("node:crypto");
const { chromium } = require("playwright-core");
const configuration = process.env.NPA_TEAM_LOCAL_LIVE_CONFIG;

test("real Workbench accounts sign in without an identity provider", {
  skip: !configuration,
}, async () => {
  const settings = JSON.parse(fs.readFileSync(configuration, "utf8"));
  assert.equal(new URL(settings.endpoint).protocol, "https:");
  const args = [];
  if (settings.certificate) {
    const certificate = new crypto.X509Certificate(fs.readFileSync(settings.certificate));
    const key = certificate.publicKey.export({ type: "spki", format: "der" });
    args.push(`--ignore-certificate-errors-spki-list=${crypto.createHash("sha256").update(key).digest("base64")}`);
  }
  const browser = await chromium.launch({ channel: "chrome", headless: true, args });
  try {
    assert.ok(settings.accounts.length >= 2);
    for (const account of settings.accounts) await verifyAccount(browser, settings, account);
  } finally {
    await browser.close();
  }
});

async function verifyAccount(browser, settings, account) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 960 } });
  const page = await context.newPage();
  try {
    await page.goto(settings.endpoint);
    await page.locator("#key-login").waitFor({ state: "visible" });
    assert.equal(await page.locator("#organization-login").isVisible(), false);
    await page.locator("#access-key").fill("invalid-key");
    await page.getByRole("button", { name: "Sign in with access key" }).click();
    await page.getByText(/Sign-in failed/).waitFor();
    await page.locator("#access-key").fill(account.access_key);
    await page.getByRole("button", { name: "Sign in with access key" }).click();
    await page.locator("#signed-in").waitFor({ state: "visible" });
    await verifyAccess(page, context, account);
    if (settings.screenshot && account === settings.accounts[0]) await page.screenshot({ path: settings.screenshot });
    await page.getByRole("button", { name: "Sign out", exact: true }).click();
    await page.locator("#signed-out").waitFor({ state: "visible" });
    assert.equal(await page.evaluate(async () => (await fetch("/v1/me")).status), 401);
  } finally {
    await context.close();
  }
}

async function verifyAccess(page, context, account) {
  const profile = await page.evaluate(async () => (await fetch("/v1/me")).json());
  assert.equal(profile.subject, account.subject);
  assert.equal(profile.display_name, account.name);
  assert.deepEqual(profile.groups.sort(), account.groups.sort());
  assert.deepEqual(Object.fromEntries(profile.workspaces.map(item => [item.name, item.role])), account.workspaces);
  for (const workspace of Object.keys(account.workspaces)) {
    const result = await page.evaluate(async name => {
      const response = await fetch(`/v1/runs?workspace=${encodeURIComponent(name)}`);
      return { status: response.status, body: await response.json() };
    }, workspace);
    assert.equal(result.status, 200);
    for (const run of account.runs || []) assert.ok(result.body.runs.some(item => item.id === run));
  }
  assert.equal(await page.evaluate(async () => (await fetch("/v1/runs?workspace=no-grant")).status), 403);
  assert.equal(await page.evaluate(async () => (await fetch("/auth/logout", { method: "POST" })).status), 403);
  const cookie = (await context.cookies()).find(item => item.name === "__Host-workbench-session");
  assert.ok(cookie.httpOnly && cookie.secure && cookie.sameSite === "Lax");
  assert.equal(await page.evaluate(() => localStorage.length + sessionStorage.length), 0);
  assert.equal(await page.locator("#access-key").inputValue(), "");
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.setViewportSize({ width: 1280, height: 960 });
}
