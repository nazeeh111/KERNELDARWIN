import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8768",
    viewport: { width: 1440, height: 1000 },
    launchOptions: process.env.PW_EXECUTABLE_PATH
      ? { executablePath: process.env.PW_EXECUTABLE_PATH }
      : {},
    trace: "retain-on-failure",
  },
  webServer: {
    command: "python3 -m http.server 8768 --bind 127.0.0.1 --directory lab",
    url: "http://127.0.0.1:8768",
    reuseExistingServer: !process.env.CI,
  },
});
