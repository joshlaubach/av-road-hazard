import { defineConfig } from "@playwright/test";

const SERVER_START_MS = 5 * 60 * 1000; // the site is built before it is served

/**
 * Browser tests against the stub API (USE_STUBS=1), in Chromium at a phone width and a desktop
 * width. Both servers are started here and never reused: if port 3000 or 8000 is taken, the run
 * stops instead of testing against whatever is there. e2e/global-setup.ts then checks the stack.
 */
export default defineConfig({
  testDir: "e2e",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000, // the stub still decodes and resizes each uploaded photo
  reporter: [["list"]],
  outputDir: "test-results",
  use: {
    baseURL: "http://localhost:3000",
    colorScheme: "light",
    reducedMotion: "no-preference",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "phone", use: { browserName: "chromium", viewport: { width: 390, height: 844 } } },
    { name: "desktop", use: { browserName: "chromium", viewport: { width: 1100, height: 800 } } },
  ],
  webServer: [
    {
      command: "node e2e/api.mjs",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: false,
      timeout: SERVER_START_MS,
    },
    {
      command: "node e2e/serve.mjs",
      url: "http://localhost:3000/settings",
      reuseExistingServer: false,
      timeout: SERVER_START_MS,
    },
  ],
});
