// Builds the site and serves it on port 3000, with the API address set to the local stub API for
// both steps (next.config.mjs reads API_URL when the rewrite is built). Any API_URL from the
// caller's environment is ignored.
import { API_ORIGIN, WEB_DIR, run, serve } from "./stack.mjs";

const env = { ...process.env, API_URL: API_ORIGIN };
const next = ["node_modules/next/dist/bin/next"];

await run(process.execPath, [...next, "build"], { cwd: WEB_DIR, env });
serve(process.execPath, [...next, "start", "--port", "3000"], { cwd: WEB_DIR, env });
