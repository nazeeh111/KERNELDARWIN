import { test, expect } from "@playwright/test";
import { readFileSync, mkdirSync } from "node:fs";
const fixture = () =>
  JSON.parse(
    readFileSync(
      new URL("../../lab/data/control-softmax.json", import.meta.url),
    ),
  );
async function open(page) {
  await page.goto("/#overview");
  await expect(page.locator("#record-select")).toHaveValue(fixture().id);
}
async function upload(page, record) {
  await page.locator("#file").setInputFiles({
    name: "local.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(record)),
  });
}

test("interval selection links the finding, measured phase, source and reloadable URL", async ({
  page,
}) => {
  await open(page);
  await expect(page.locator(".finding h2")).toContainText("latency reduction");
  await page.locator('.comparison-row[data-candidate="cpu-32"]').click();
  await expect(
    page.locator('.comparison-row[data-candidate="cpu-32"]'),
  ).toBeFocused();
  await expect(page.locator(".finding h2")).toHaveText(
    "Measured, but not promoted.",
  );
  await expect(page.locator(".plot-caption")).toContainText(
    "Discovery measurement",
  );
  await expect(page).toHaveURL(/candidate=cpu-32/);
  await page.getByRole("link", { name: "Implementation", exact: true }).click();
  await expect(page.getByLabel("Select implementation candidate")).toHaveValue(
    "cpu-32",
  );
  await page.reload();
  await expect(page.getByLabel("Select implementation candidate")).toHaveValue(
    "cpu-32",
  );
  await page.goBack();
  await expect(page.locator(".finding h2")).toHaveText(
    "Measured, but not promoted.",
  );
  await page.goBack();
  await expect(page.locator(".finding h2")).toContainText("latency reduction");
});
test("command search contains focus, supports keyboard selection and restores its trigger", async ({
  page,
}) => {
  await open(page);
  const trigger = page.getByRole("button", {
    name: "Search views and evidence",
    exact: true,
  });
  await trigger.click();
  const search = page.getByRole("searchbox", {
    name: "Search views, investigations and candidates",
  });
  await expect(search).toBeFocused();
  await search.fill("no-matches-abcdef");
  await expect(
    page.getByText("No match. Try a view, operation or candidate ID."),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(trigger).toBeFocused();
  await page.keyboard.press("Control+k");
  await search.fill("cpu-32");
  await page.keyboard.press("ArrowDown");
  await expect(page.locator(".command-result").first()).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator(".command-dialog")).not.toBeVisible();
  await expect(
    page.locator('.comparison-row[data-candidate="cpu-32"]'),
  ).toHaveAttribute("aria-pressed", "true");
  await trigger.click();
  await search.fill("hypotheses");
  await page.keyboard.press("Enter");
  await expect(page.locator("h1")).toHaveText("Hypothesis laboratory");
});
test("source diff navigates exact changes and treats imported HTML as text", async ({
  page,
}) => {
  await open(page);
  const record = fixture();
  record.id = "diff-review";
  record.experiments[0].source = {
    baseline: "same\nrepeat\nrepeat\n",
    candidate: "same\nrepeat\n<script>window.bad=true</script>\n",
  };
  await upload(page, record);
  await page.getByRole("link", { name: "Implementation", exact: true }).click();
  await page
    .getByLabel("Select implementation candidate")
    .selectOption(record.experiments[0].id);
  await expect(page.locator(".diff-summary")).toContainText("+1");
  await expect(page.locator(".diff-summary")).toContainText("−1");
  await page.getByRole("button", { name: "Next changed line" }).click();
  await expect(page.locator(".current-change")).toBeFocused();
  await expect(page.locator(".diff-position")).toHaveText(
    "Changed line 1 of 2",
  );
  await page.getByRole("button", { name: "Changes only", exact: true }).click();
  await expect(page.locator(".diff-line.same")).toHaveCount(0);
  expect(await page.evaluate(() => window.bad)).toBeUndefined();
  await expect(page.locator(".diff-line.add code")).toHaveText(
    "<script>window.bad=true</script>",
  );
});
test("raw samples export exact values and local share explains missing record contents", async ({
  page,
}) => {
  await open(page);
  const record = fixture();
  const winner = record.experiments.find((e) => e.id === record.winner_id);
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Raw samples", exact: true }).click();
  const download = await pending;
  const csv = readFileSync(await download.path(), "utf8");
  expect(csv).toContain(
    `1,${winner.confirmation.baseline_us[0]},${winner.confirmation.candidate_us[0]},confirmation`,
  );
  // An imported replacement using a bundled ID is still local content.
  record.title = "Local replacement";
  await upload(page, record);
  await page
    .getByRole("button", { name: "Share selection", exact: true })
    .click();
  await expect(page.locator("#share-note")).toContainText("imported locally");
  await expect(page.locator("#selection-url")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "Share selection", exact: true }),
  ).toBeFocused();
});
test("oversized source remains downloadable without unbounded interactive diff", async ({
  page,
}) => {
  await open(page);
  const record = fixture();
  record.id = "large-source";
  record.experiments[0].source = {
    baseline: "a".repeat(1000001),
    candidate: "b",
  };
  await upload(page, record);
  await page.getByRole("link", { name: "Implementation", exact: true }).click();
  await page
    .getByLabel("Select implementation candidate")
    .selectOption(record.experiments[0].id);
  await expect(
    page.getByText(/sources exceed the interactive comparison limit/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Reference source", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".diff-line")).toHaveCount(0);
});
test("a shared imported record resumes its requested candidate after import", async ({
  page,
}) => {
  const record = fixture();
  record.id = "shared-local-record";
  await page.goto("/#code?record=shared-local-record&candidate=cpu-32");
  await expect(page.locator("#feedback")).toContainText(
    "Shared investigation not found",
  );
  await upload(page, record);
  await expect(page.getByLabel("Select implementation candidate")).toHaveValue(
    "cpu-32",
  );
  await expect(page).toHaveURL(/record=shared-local-record&candidate=cpu-32/);
});
test("change navigation reaches edits beyond the preview window", async ({
  page,
}) => {
  await open(page);
  const record = fixture();
  record.id = "late-source-change";
  const original = Array.from(
    { length: 1700 },
    (_, i) => `unchanged line ${i + 1}`,
  );
  const modified = [...original];
  modified[1649] = "late edit at line 1650";
  record.experiments[0].source = {
    baseline: original.join("\n"),
    candidate: modified.join("\n"),
  };
  await upload(page, record);
  await page.getByRole("link", { name: "Implementation", exact: true }).click();
  await page
    .getByLabel("Select implementation candidate")
    .selectOption(record.experiments[0].id);
  await page.getByRole("button", { name: "Next changed line" }).click();
  await expect(page.locator(".current-change")).toBeFocused();
  await expect(page.locator(".current-change .diff-old")).toHaveText("1650");
  await page.getByRole("button", { name: "Changes only", exact: true }).click();
  await expect(page.locator(".diff-line")).toHaveCount(2);
  await expect(page.locator(".diff-line.add code")).toHaveText(
    "late edit at line 1650",
  );
});
test("all views remain contained at narrow widths and fresh review images are retained", async ({
  page,
}) => {
  test.setTimeout(120000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await open(page);
  const folder = "test-results/visual-review";
  mkdirSync(folder, { recursive: true });
  for (const width of [1440, 1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const view of [
      "overview",
      "timeline",
      "bottlenecks",
      "hypotheses",
      "evolution",
      "code",
      "atlas",
      "research",
    ]) {
      await page.locator(`.nav-link[data-view="${view}"]`).click();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBe(width);
      if ([1440, 390].includes(width) && ["overview", "code"].includes(view))
        await page.screenshot({
          path: `${folder}/${view}-${width}.png`,
          fullPage: true,
        });
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page
    .getByRole("button", { name: "Search views and evidence", exact: true })
    .click();
  await page
    .getByRole("searchbox", {
      name: "Search views, investigations and candidates",
    })
    .fill("cpu");
  await page.screenshot({ path: `${folder}/command-search-1440.png` });
  expect(await page.evaluate(() => document.getAnimations().length)).toBe(0);
  expect(errors).toEqual([]);
});
