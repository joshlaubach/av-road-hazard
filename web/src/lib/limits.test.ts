import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { RANGE_BANDS, percentFound } from "./limits";

interface Results {
  effective_range: { bands: { band: string; found: number; trials: number }[] };
}

describe("RANGE_BANDS", () => {
  it("matches the effective range in eval/results.json", () => {
    const results = JSON.parse(readFileSync(resolve(__dirname, "../../../eval/results.json"), "utf-8")) as Results;
    const fromFile = results.effective_range.bands.map((b) => [b.band, b.found, b.trials]).sort();
    const here = RANGE_BANDS.map((b) => [b.band, b.found, b.trials]).sort();
    expect(here).toEqual(fromFile);
  });

  it("rounds the found rate to a whole percent", () => {
    expect(RANGE_BANDS.map(percentFound)).toEqual([60, 30, 5]);
  });
});
