import { test, expect } from "@playwright/test";

test("workload page separates a measured null result from runnable instructions", async ({ page }) => {
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  await page.goto("/workload.html");
  await expect(page.locator("#evidence-status")).toHaveText("measured null");
  await expect(page.locator("#evidence-headline")).toContainText("GPU checks passed");
  await expect(page.locator("#normalization-value")).toHaveText("Not confirmed");
  await expect(page.locator("#methods-caption")).toContainText("do not establish a speedup");
  await expect(page.locator("#recorded-query")).toBeEnabled();
  await page.locator("#recorded-demo summary").click();
  await page.locator("#recorded-query").selectOption("1");
  await expect(page.locator("#recorded-output")).toContainText("0.554700");
  await expect(page.locator("#recorded-output")).toContainText("fallback_calls: 1");
  await expect(page.locator("#recorded-note")).toContainText("does not execute GPU code");
  await expect(page.locator("#artifact-link")).toBeHidden();
  await page.locator("#consumer").selectOption("vectors");
  await expect(page.locator("#consumer-help")).toContainText("documents and queries");
  await expect(page.locator("#run-code")).toContainText("kd workload search");
  await page.locator("#guard-example").selectOption("identity");
  await expect(page.locator("#guard-decision")).toHaveText("Plan rejected; library fallback");
  await page.locator("#guard-example").selectOption("strides");
  await expect(page.locator("#guard-explanation")).toContainText("input strides directly");
  await page.getByRole("button",{name:"Copy local search command"}).click();
  await expect(page.locator("#copy-feedback")).not.toBeEmpty();
  expect(errors).toEqual([]);
});

test("narrow layouts, keyboard and reduced motion remain usable", async ({ page }) => {
  await page.emulateMedia({reducedMotion:"reduce"});
  for (const width of [320,390,768,1440]) {
    await page.setViewportSize({width,height:1000});
    await page.goto("/workload.html");
    await expect(page.locator("#evidence-status")).toHaveText("measured null");
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  }
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link",{name:"Skip to workload"})).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main")).toBeFocused();
  await page.locator(".code-disclosure summary").filter({hasText:"Use the callable in Python"}).focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#python-code")).toBeVisible();
});

test("null observations can show costs while ratios and artifact promotion remain hidden", async ({ page }) => {
  await page.route("**/data/workload.json",route=>route.fulfill({json:{
    status:"measured_null",headline:"A retained null study",summary:"Fixture: observed costs are descriptive",
    normalization:{speedup:9,ci:[8,10]},application:{speedup:9},artifact_url:"https://example.com/unconfirmed.zip",
    methods:[{name:"Observed method",observed_cost_ms:3.25,cost_ms:999,status:"Criterion failed"}]}}));
  await page.goto("/workload.html");
  await expect(page.locator("#normalization-value")).toHaveText("Not confirmed");
  await expect(page.locator("#application-value")).toHaveText("Not confirmed");
  await expect(page.locator("#artifact-link")).toBeHidden();
  await expect(page.locator("#methods-body")).toContainText("3.250 ms");
  await expect(page.locator("#methods-body")).not.toContainText("999");
});

test("failed evidence fetch recovers and pending data cannot publish a speedup", async ({ page }) => {
  let failing=true;
  await page.route("**/data/workload.json",route=>failing?route.fulfill({status:503,body:"unavailable"}):route.fulfill({json:{
    status:"pending",headline:"Still unconfirmed",summary:"Fixture: numeric fields must stay hidden",normalization:{speedup:9,ci:[8,10]},application:{speedup:9},artifact_url:"https://example.com/unconfirmed.zip"}}));
  await page.goto("/workload.html");
  await expect(page.locator("#evidence-status")).toHaveText("Evidence unavailable");
  await expect(page.locator("#run-code")).toHaveText("kd workload demo");
  failing=false;
  await page.getByRole("button",{name:"Retry loading evidence"}).click();
  await expect(page.locator("#evidence-headline")).toHaveText("Still unconfirmed");
  await expect(page.locator("#normalization-value")).toHaveText("Pending");
  await expect(page.locator("#application-value")).toHaveText("Pending");
  await expect(page.locator("#artifact-link")).toBeHidden();
});
