/** Capture the actual local workbench. Requires optional npm test dependencies. */
import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const args = process.argv.slice(2);
const get = (name) =>
  args.includes(name) ? args[args.indexOf(name) + 1] : undefined;
if (!args.includes("--output")) throw Error("Choose a new --output directory.");
const output = path.resolve(get("--output"));
const url = get("--url") || "http://127.0.0.1:8766";
if (!["127.0.0.1", "localhost"].includes(new URL(url).hostname))
  throw Error("Capture supports an explicit local preview only.");
await mkdir(output, { recursive: false });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.PW_EXECUTABLE_PATH
    ? { executablePath: process.env.PW_EXECUTABLE_PATH }
    : {}),
});
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    recordVideo: { dir: output, size: { width: 1440, height: 1000 } },
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(url);
  await page.locator(".improvement-value").waitFor();
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(350);
  await page.screenshot({ path: path.join(output, "lab-desktop.png") });
  const start = Date.now();
  const beats = [];
  async function beat(name, action, hold) {
    await action();
    beats.push({ name, elapsed_ms: Date.now() - start });
    await page.waitForTimeout(hold);
  }
  await beat("Measured overview", async () => {}, 1600);
  await beat(
    "Inspect actual paired samples",
    async () => {
      await page
        .getByRole("button", { name: "Paired samples", exact: true })
        .click();
      await page
        .getByRole("slider", { name: "Inspect timing pair" })
        .fill("15");
    },
    2200,
  );
  await beat(
    "Compare an unsuccessful candidate",
    async () => {
      await page.getByRole("link", { name: "Evolution", exact: true }).click();
      await page.locator('.lineage-node[data-candidate="cpu-512"]').click();
    },
    2400,
  );
  await beat(
    "Read the source comparison",
    async () => {
      await page
        .getByRole("link", { name: "Implementation", exact: true })
        .click();
    },
    2000,
  );
  await beat(
    "Inspect held-out policy budgets",
    async () => {
      await page.getByRole("link", { name: "Research", exact: true }).click();
      await page.getByLabel("Evaluation candidate budget").selectOption("2");
    },
    2300,
  );
  await beat(
    "Return to the measured result",
    async () => {
      await page.getByRole("link", { name: "Overview", exact: true }).click();
    },
    1600,
  );
  const video = page.video();
  await context.close();
  const videoPath = await video.path();
  await browser.close();
  await writeFile(
    path.join(output, "capture.json"),
    JSON.stringify(
      {
        source: url,
        viewport: [1440, 1000],
        video: path.basename(videoPath),
        beats,
        errors,
        note: "Actual CPU-control interaction recording. No GPU execution, invented metrics or narration.",
      },
      null,
      2,
    ),
  );
  if (errors.length) throw Error(errors.join("\n"));
  console.log(JSON.stringify({ output, video: videoPath, beats }, null, 2));
} finally {
  await browser.close();
}
