// Exercise the standalone team example without a server, credentials, or cloud calls.
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { resolve } = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright-core");

const example = pathToFileURL(
  resolve(__dirname, "../../../docs/demos/team-access.html"),
).href;

async function scenario(page, name) {
  await page.click('[data-tab="playground"]');
  await page.click(`[data-scenario="${name}"]`);
}

async function contains(page, selector, expected) {
  assert.match(await page.locator(selector).innerText(), expected);
}

async function happyPath(page) {
  await scenario(page, "workflow");
  await page.click("#submit");
  await page.click("#retry");
  await contains(page, "#run-count", /1 visible/);
  for (let step = 0; step < 4; step++) await page.click("#advance");
  await contains(page, "#run-status", /Complete/);
  await contains(page, "#gpu-metric", /0 \/ 2/);
  for (const action of ["logs", "artifacts"]) {
    await page.click(`#${action}`);
    await contains(page, "#dialog-body", /EXAMPLE/);
    await page.keyboard.press("Escape");
  }
}

async function capacityAndPrivacy(page) {
  await scenario(page, "quota");
  await page.click("#submit");
  await contains(page, "#run-status", /Quota blocked/);
  await contains(page, "#gpu-metric", /2 \/ 2/);
  await page.selectOption("#person", "bob");
  await contains(page, "#run-count", /0 visible/);
  await page.click("#submit");
  await contains(page, "#run-count", /1 visible/);
  await contains(page, "#boundary-name", /Sample runner B/);
  await scenario(page, "privacy");
  await contains(page, "#dialog-body", /404/);
  await page.click("#close-dialog");
  await contains(page, "#run-count", /0 visible/);
}

async function allocationAndRoleChecks(page) {
  await scenario(page, "projects");
  await page.selectOption("#cluster", "west");
  await page.selectOption("#gpus", "2");
  await page.click("#submit");
  await contains(page, "#run-status", /Quota blocked/);
  await page.selectOption("#gpus", "1");
  await page.click("#submit");
  await contains(page, "#gpu-metric", /1 \/ 1/);
  await page.selectOption("#person", "bob");
  await page.selectOption("#workspace", "vision");
  await page.click("#submit");
  await contains(page, "#notice", /403/);
  await scenario(page, "reader");
  await page.click("#submit");
  await contains(page, "#notice", /Reader access cannot submit/);
  await page.selectOption("#person", "dana");
  await page.click("#submit");
  await contains(page, "#notice", /administrator allocation is required/);
  await page.selectOption("#person", "eve");
  await page.click("#submit");
  await contains(page, "#notice", /no current grant/);
}

async function recoveryAndCancellation(page) {
  await scenario(page, "recovery");
  await page.click("#resume");
  await contains(page, "#run-status", /Running/);
  await contains(page, "#run-count", /1 visible/);
  await page.click("#cancel");
  await contains(page, "#run-status", /Cancelling/);
  await contains(page, "#gpu-metric", /1 \/ 2/);
  await page.click("#cancel");
  await contains(page, "#run-status", /Cancelled/);
  await contains(page, "#gpu-metric", /0 \/ 2/);
  await scenario(page, "lost");
  await page.click("#resume");
  await contains(page, "#notice", /do not blindly submit/);
  await contains(page, "#run-status", /Launch uncertain/);
}

async function offboarding(page) {
  await scenario(page, "offboard");
  await page.click('[data-tab="access"]');
  await page.click("#disable-alice");
  await page.click('[data-tab="playground"]');
  await contains(page, "#run-count", /0 visible/);
  await page.click("#submit");
  await contains(page, "#notice", /403/);
  await page.click('[data-tab="access"]');
  await page.click("#operator-stop");
  await page.click("#enable-alice");
  await page.click('[data-tab="playground"]');
  await contains(page, "#runs", /Cancelled/);
}

async function evidenceAndMobile(page) {
  await page.click('[data-tab="evidence"]');
  assert.equal(await page.locator(".receipt").count(), 20);
  assert.equal(await page.locator(".receipt .red").count(), 0);
  await page.setViewportSize({ width: 390, height: 844 });
  for (const section of ["playground", "access", "architecture", "evidence"]) {
    await page.click(`[data-tab="${section}"]`);
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
  }
}

async function simulatedOnly(page) {
  await contains(page, ".topbar", /simulated infrastructure/);
  await contains(page, "#notice", /All interactive outcomes are simulated/);
}

test("team example covers access, quotas, recovery, and offboarding offline", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const errors = [],
    outbound = [];
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1040 },
    });
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (request) => {
      if (/^https?:/.test(request.url())) outbound.push(request.url());
    });
    await page.goto(example);
    await simulatedOnly(page);
    await happyPath(page);
    await capacityAndPrivacy(page);
    await allocationAndRoleChecks(page);
    await recoveryAndCancellation(page);
    await offboarding(page);
    await evidenceAndMobile(page);
    assert.deepEqual(errors, []);
    assert.deepEqual(outbound, []);
  } finally {
    await browser.close();
  }
});
