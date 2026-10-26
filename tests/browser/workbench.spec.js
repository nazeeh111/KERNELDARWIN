import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

const fixture = () =>
  JSON.parse(
    readFileSync(
      new URL("../../lab/data/control-softmax.json", import.meta.url),
    ),
  );
async function open(page, route = "overview") {
  await page.goto("/#" + route);
  await expect(page.locator("#record-select")).toHaveValue(fixture().id);
}
async function upload(page, record) {
  await page
    .locator("#file")
    .setInputFiles({
      name: "test-record.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(record)),
    });
}

test("candidate selection and timing inspector use matching recorded evidence", async ({
  page,
}) => {
  await open(page);
  await page.locator('.candidate-button[data-candidate="cpu-32"]').click();
  await expect(page.locator(".improvement-value")).toContainText("1.59");
  await expect(
    page.locator('.candidate-button[data-candidate="cpu-32"]'),
  ).toBeFocused();
  await page
    .getByRole("button", { name: "Paired samples", exact: true })
    .click();
  await page.getByRole("slider", { name: "Inspect timing pair" }).fill("12");
  await expect(page.locator(".pair-readout")).toContainText("PAIR 13");
});
test("dormant-view malformed import cannot poison later navigation", async ({
  page,
}) => {
  await open(page, "timeline");
  const bad = fixture();
  bad.id = "invalid-optional-text";
  bad.experiments[0].intervention = {};
  await upload(page, bad);
  await expect(page.locator("#feedback")).toContainText(
    "previous investigation is preserved",
  );
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await expect(page.locator("h1")).toContainText("Softmax");
  await expect(page.locator("#record-select")).toHaveValue(fixture().id);
});
test("implementation and active-record selects preserve keyboard focus", async ({
  page,
}) => {
  await open(page, "code");
  const candidate = page.getByLabel("Select implementation candidate");
  await candidate.focus();
  await candidate.selectOption("cpu-512");
  await expect(candidate).toBeFocused();
  const record = page.getByLabel("Active investigation");
  await record.focus();
  await record.selectOption({ label: "sum / 32 × 31" });
  await expect(record).toBeFocused();
});
test("mobile and collapsed sidebar keep names, containment and resolved reduced motion", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 900, height: 1000 });
  await open(page);
  await expect(
    page.getByRole("link", { name: "Hypotheses", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 320, height: 844 });
  await page
    .getByLabel("Active investigation")
    .selectOption({ label: "layernorm / 256 × 513" });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    320,
  );
  expect(await page.evaluate(() => document.getAnimations().length)).toBe(0);
});
test("large trace imports are bounded and disclose the visible event cap", async ({
  page,
}) => {
  await open(page);
  const record = fixture();
  record.id = "large-trace-test";
  record.trace = {
    status: "recorded",
    events: Array.from({ length: 150000 }, (_, i) => ({
      name: "test-fixture",
      start_us: i * 2,
      duration_us: 1,
    })),
  };
  await upload(page, record);
  await page.getByRole("link", { name: "GPU timeline", exact: true }).click();
  await expect(page.locator(".timeline-row")).toHaveCount(300);
  await expect(page.getByText(/Showing 300 of 150000/)).toBeVisible();
  await expect(page.locator(".timeline-track").first()).toHaveAccessibleName(
    /starts at 0.00 microseconds/,
  );
});
test("branched lineage retains parent identity without inventing baseline edges", async ({
  page,
}) => {
  await open(page);
  const record = fixture();
  record.id = "branched-test";
  record.experiments[1].parent_id = record.experiments[0].id;
  await upload(page, record);
  await page.getByRole("link", { name: "Evolution", exact: true }).click();
  await expect(page.locator(".lineage.grouped")).toBeVisible();
  await expect(
    page.getByText("Children of cpu-32", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".lineage-branches")).toHaveCount(0);
});
test("atlas filters and research budget remain functional", async ({
  page,
}) => {
  await open(page, "atlas");
  await page.getByRole("searchbox").fill("layernorm");
  await expect(page.locator(".atlas-record")).toHaveCount(1);
  await page.getByLabel("Filter evidence").selectOption("cuda_run");
  await expect(page.locator(".atlas-record")).toHaveCount(0);
  await page.getByRole("link", { name: "Research", exact: true }).click();
  await page.getByLabel("Evaluation holdout split").selectOption("shape");
  await page.getByLabel("Evaluation candidate budget").selectOption("2");
  await expect(page.locator(".chart-context")).toContainText(
    "6 holdout groups · 2 candidates",
  );
});
