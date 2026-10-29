/** Hosted-only production check. Never runs GPU code or imports untrusted source. */
import { chromium, expect } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";

const root = new URL("../", import.meta.url);
const version = process.env.EXPECTED_COMMIT || "manual";
const base = "https://nazeeh111.github.io/KERNELDARWIN/";
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const identities = {};
for (const path of ["index.html", "app.js", "refinement.css", "js/comparison.js", "js/commands.js", "js/evidence-model.js", "js/source-comparison.js", "js/line-diff.js"]) {
  const response = await fetch(`${base}${path}?verify=${version}`);
  if (!response.ok) throw Error(`${path}: HTTP ${response.status}`);
  const expected = sha(readFileSync(new URL(`lab/${path}`, root)));
  if (sha(Buffer.from(await response.arrayBuffer())) !== expected) throw Error(`Deployed file differs: ${path}`);
  identities[path] = expected;
}
const browser = await chromium.launch();
try {
  const page = await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:"reduce"});
  const errors=[]; page.on("pageerror",error=>errors.push(error.message));
  await page.goto(`${base}?verify=${version}#overview`);
  await expect(page.locator(".finding h2")).toContainText("latency reduction");
  await page.locator('.comparison-row[data-candidate="cpu-32"]').click();
  await expect(page.locator(".finding h2")).toHaveText("Measured, but not promoted.");
  await page.locator('.nav-link[data-view="code"]').click();
  await expect(page.getByLabel("Select implementation candidate")).toHaveValue("cpu-32");
  await expect(page.locator(".diff-line.add").first()).toBeVisible();
  await page.getByRole("button",{name:"Search views and evidence",exact:true}).click();
  await page.getByRole("searchbox",{name:"Search views, investigations and candidates"}).fill("cpu-128");
  await page.keyboard.press("Enter");
  await expect(page.locator('.comparison-row[data-candidate="cpu-128"]')).toHaveAttribute("aria-pressed","true");
  await page.reload();
  await expect(page.locator(".finding h2")).toContainText("latency reduction");
  await page.evaluate(()=>document.fonts.ready);
  mkdirSync("test-results/visual-review",{recursive:true});
  await page.screenshot({path:"test-results/visual-review/public-overview.png"});
  if (errors.length) throw Error(errors.join("\n"));
  writeFileSync("test-results/visual-review/public-verification.json",JSON.stringify({url:page.url(),expected_commit:version,source_sha256:identities,browser_errors:errors,linked_selection:true,source_diff:true,command_search:true,reload:true},null,2)+"\n");
  console.log(`Production LAB verified against ${version}; linked selection, diff, command search and reload pass.`);
} finally { await browser.close(); }
