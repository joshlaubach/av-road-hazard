# REPORT-C: dependency security fixes
Branch fix/dependencies (from origin/main 3d588b5), commit 050196f, 2026-11-16.

## Python: requirements.lock
- opentelemetry-api 1.45.1 -> 1.45.2. The lock has no sdk/semantic-conventions pins. The uv resolver (--upgrade-package opentelemetry-api==1.45.2) changed only this line, so this is a one-line edit.
- pip-audit before: 1 vulnerability (opentelemetry-api 1.45.1, CVE-2026-126360, fix 1.45.2). After: "No known vulnerabilities found".
- Not edited (outside scope): requirements-dev.lock line 81 also pins opentelemetry-api==1.45.1 and needs the same bump.

## Web: package.json / package-lock.json
- npm audit fix (no --force): lockfile-only transitive bumps (22 -> 11).
- katex 0.16.47 -> 0.18.11 (GHSA-238p-pmpm-9mq7, fixed in >=0.18.2).
- overrides: postcss 8.5.33 (next 15.5.28 pins 8.4.31).
- @lhci/cli 0.16.0 -> 0.16.1 (dev dependency; fixes the basic-ftp/get-uri/pac-proxy-agent/proxy-agent chain). This goes beyond the brief's "postcss and katex only"; review or drop it.
- npm audit (all deps) before: 22 (1 low, 13 moderate, 6 high, 2 critical). After: 3 (1 moderate, 2 critical).
- Skipped: tinypool <=2.1.1 (critical), @vitest/mocker, vitest. The only fix is vitest 3.2.7 -> 5.1.0 (--force, two major versions); vitest 3 needs tinypool ^1.1.1 and the fix is in 2.1.2. Dev/test only. Owner decision.
- @types/katex 0.16.8 is now redundant (katex 0.18 ships its own types); kept.
- Checks: typecheck pass; lint 0 errors / 1 existing warning (e2e/lighthouse.mjs MIN_SCORE, removed on fix/code-cleanups); test 88/88 (15 files); build success.

## configs/hosted_classifier.json (not changed)
Line 2 uses "revision": "main", a moving branch. It needs a commit hash from the owner (UNKNOWN).
