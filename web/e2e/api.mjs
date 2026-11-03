// Starts the stub API on port 8000 and stays in the foreground. Used by Playwright and Lighthouse.
import { REPO_DIR, assertUvicorn, pythonPath, serve, stubApiEnv } from "./stack.mjs";

const python = pythonPath();
assertUvicorn(python);
serve(python, ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"], {
  cwd: REPO_DIR,
  env: stubApiEnv(),
});
