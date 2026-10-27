import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { EXAMPLES } from "./examples";

const ROOT = resolve(__dirname, "../../public/examples");

describe("bundled examples", () => {
  it("are exactly the ones offered, each with a display photo and a trace", () => {
    expect(readdirSync(ROOT).sort()).toEqual(EXAMPLES.map((e) => e.id).sort());
    for (const example of EXAMPLES) {
      expect(readdirSync(join(ROOT, example.id)).sort()).toEqual(["display.jpg", "example.json"]);
    }
  });

  it("credit the photo's authors and describe the full-size analysis", () => {
    for (const example of EXAMPLES) {
      const file = JSON.parse(readFileSync(join(ROOT, example.id, "example.json"), "utf-8"));
      expect(file.credit).toContain("Stellenbosch University");
      expect(file.credit).toContain("Nienaber");
      expect(file.analysis.record.image_size).toEqual(file.analyzed_size);
      expect(file.analyzed_size[0]).toBeGreaterThan(file.display_size[0]); // shown smaller, analyzed full size
      expect(file.analysis.trace.tiles.length).toBeGreaterThan(0);
    }
  });

  it("hold nothing from nuScenes or RDD2022, and no file path", () => {
    for (const example of EXAMPLES) {
      const text = readFileSync(join(ROOT, example.id, "example.json"), "utf-8");
      expect(text).not.toMatch(/nuscenes|nuImages|n008-|rdd2022|C:\\|\/Users\//i);
    }
  });

  it("stay small enough to ship", () => {
    for (const example of EXAMPLES) {
      for (const name of ["display.jpg", "example.json"]) {
        const file = join(ROOT, example.id, name);
        expect(existsSync(file)).toBe(true);
        expect(statSync(file).size).toBeLessThan(600 * 1024);
      }
    }
  });
});
