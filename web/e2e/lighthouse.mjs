// `npm run lighthouse`: start the same stub stack as the browser tests, run Lighthouse (mobile,
// accessibility only) on the three pages with Playwright's Chromium, print the scores, and exit
// non-zero if any is below 0.90. Reports go to web/.lighthouseci/ (git-ignored).
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { API_ORIGIN, WEB_DIR, WEB_ORIGIN, assertStubStack, isPortBusy, run } from "./stack.mjs";

const MIN_SCORE = 0.9;
const READY_MS = 5 * 60 * 1000;
const REPORT_DIR = join(WEB_DIR, ".lighthouseci");
const require = createRequire(import.meta.url);

function start(script) {
  return spawn(process.execPath, [join("e2e", script)], { cwd: WEB_DIR, stdio: "inherit", detached: process.platform !== "win32" });
}

function stop(child) {
  if (child.exitCode !== null || !child.pid) return;
  if (process.platform === "win32") spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  else process.kill(-child.pid);
}

async function waitFor(url, children) {
  const deadline = Date.now() + READY_MS;
  while (Date.now() < deadline) {
    if (children.some((c) => c.exitCode !== null)) throw new Error("A server stopped before it was ready.");
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      // not up yet
    }
    await new Promise((done) => setTimeout(done, 1000));
  }
  throw new Error(`${url} did not answer in time.`);
}

function printScores() {
  const rows = readdirSync(REPORT_DIR)
    .filter((name) => /^lhr-.*\.json$/.test(name))
    .map((name) => JSON.parse(readFileSync(join(REPORT_DIR, name), "utf8")))
    .map((lhr) => ({ url: new URL(lhr.finalDisplayedUrl).pathname, score: lhr.categories.accessibility.score }))
    .sort((a, b) => a.url.localeCompare(b.url));
  for (const { url, score } of rows) console.log(`Accessibility ${url}: ${score}`);
  return rows;
}

async function main() {
  for (const port of [3000, 8000]) {
    if (await isPortBusy(port)) throw new Error(`Port ${port} is already in use; stop that server first.`);
  }
  const { chromium } = await import("@playwright/test");
  rmSync(REPORT_DIR, { recursive: true, force: true });
  mkdirSync(REPORT_DIR, { recursive: true });

  const servers = [start("api.mjs"), start("serve.mjs")];
  try {
    await waitFor(`${API_ORIGIN}/health`, servers);
    await waitFor(`${WEB_ORIGIN}/settings`, servers);
    await assertStubStack();

    process.env.CHROME_PATH = chromium.executablePath();
    const lhci = [require.resolve("@lhci/cli/src/cli.js")];
    const flags = process.env.CI ? ["--collect.settings.chromeFlags=--no-sandbox"] : [];
    await run(process.execPath, [...lhci, "collect", "--config=lighthouserc.json", ...flags], { cwd: WEB_DIR });
    const rows = printScores();
    if (rows.length !== 3) throw new Error(`Expected 3 reports, found ${rows.length}.`);
    await run(process.execPath, [...lhci, "assert", "--config=lighthouserc.json"], { cwd: WEB_DIR });
  } finally {
    servers.forEach(stop);
  }
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  },
);
