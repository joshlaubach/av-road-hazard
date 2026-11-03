// Shared by the Playwright run and the Lighthouse run: how to start the stub stack, and the checks
// that prove the browser can only be talking to it. Plain .mjs so both can import it (see stack.d.mts).
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import net from "node:net";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const WEB_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const REPO_DIR = resolve(WEB_DIR, "..");
export const API_ORIGIN = "http://127.0.0.1:8000";
export const WEB_ORIGIN = "http://localhost:3000";

// Set to the empty string in the API's environment. common/config.py applies only non-empty values,
// so this cuts every outward connection even if the caller's shell or .env sets them.
const NEUTRALIZED = [
  "QDRANT_URL",
  "QDRANT_API_KEY",
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_DAILY_SPEND_CAP_USD",
  "ANTHROPIC_MONTHLY_SPEND_CAP_USD",
  "OLLAMA_HOST",
  "OLLAMA_MODEL",
];

/** The environment for the stub API: the caller's, with USE_STUBS=1 and every outward setting emptied. */
export function stubApiEnv(base = process.env) {
  const env = { ...base, USE_STUBS: "1" };
  for (const name of NEUTRALIZED) env[name] = "";
  return env;
}

/** The Python to run the API with: E2E_PYTHON, else the repo .venv, else python on PATH. */
export function pythonPath(env = process.env) {
  if (env.E2E_PYTHON) return env.E2E_PYTHON;
  const venv =
    process.platform === "win32"
      ? join(REPO_DIR, ".venv", "Scripts", "python.exe")
      : join(REPO_DIR, ".venv", "bin", "python");
  return existsSync(venv) ? venv : "python";
}

/** Fail with a message naming the interpreter if it cannot import uvicorn. */
export function assertUvicorn(python) {
  const probe = spawnSync(python, ["-c", "import uvicorn"], { cwd: REPO_DIR, encoding: "utf8" });
  if (probe.error || probe.status !== 0) {
    throw new Error(
      `The Python interpreter "${python}" cannot import uvicorn. Set E2E_PYTHON to one that can ` +
        "(for example the repository .venv after pip install -r requirements-dev.lock).",
    );
  }
}

/** Run a command to the end with output shown; reject if it fails. */
export function run(command, args, options) {
  return new Promise((done, fail) => {
    const child = spawn(command, args, { stdio: "inherit", ...options });
    child.on("error", fail);
    child.on("exit", (code) => (code === 0 ? done() : fail(new Error(`${command} ${args.join(" ")} exited ${code}`))));
  });
}

/** Run a server in the foreground: forward its exit code, and stop it when this process is told to. */
export function serve(command, args, options) {
  const child = spawn(command, args, { stdio: "inherit", ...options });
  const stop = () => child.kill();
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
  child.on("error", (error) => {
    console.error(error);
    process.exit(1);
  });
  child.on("exit", (code) => process.exit(code ?? 1));
  return child;
}

/** True when something already accepts connections on the port (on either loopback address). */
export function isPortBusy(port) {
  const tryHost = (host) =>
    new Promise((done) => {
      const socket = net.connect({ port, host });
      socket.once("connect", () => (socket.destroy(), done(true)));
      socket.once("error", () => done(false));
    });
  return Promise.all([tryHost("127.0.0.1"), tryHost("::1")]).then((seen) => seen.some(Boolean));
}

async function readJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  return response.json();
}

/**
 * Refuse to go on unless the stack is the local stub one: the API reports stubs on and an in-memory
 * vector store, the site's proxy reaches that same API, and API_URL (if set) is local.
 */
export async function assertStubStack(env = process.env) {
  const apiUrl = env.API_URL;
  if (apiUrl && !/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/.test(apiUrl)) {
    throw new Error(`API_URL is "${apiUrl}", which is not local. Unset it; the browser tests only use the local stub API.`);
  }
  const direct = await readJson(`${API_ORIGIN}/health`);
  if (direct.stubs !== true || direct.qdrant !== "in_memory") {
    throw new Error(
      `The API at ${API_ORIGIN} is not the stub one (stubs=${direct.stubs}, qdrant=${direct.qdrant}). ` +
        "Stopping so no real model, key or remote database is used.",
    );
  }
  const proxied = await readJson(`${WEB_ORIGIN}/api/health`);
  if (JSON.stringify(proxied) !== JSON.stringify(direct)) {
    throw new Error("The site's /api proxy does not reach the local stub API. Stopping.");
  }
}
