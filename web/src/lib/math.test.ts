import { describe, expect, it } from "vitest";
import type { FusionStep } from "@/api/types";
import { fmt, hazardTex, stepTex } from "./math";

const STEP: FusionStep = {
  name: "D40", merged_from: 3, in_road: true, road_multiplier: 1, severity: "high",
  severity_weight: 1, confidence: 0.8, proximity: 0.546, score: 0.618,
};

describe("stepTex", () => {
  it("plugs the step's own numbers into the score formula", () => {
    const tex = stepTex(STEP);
    expect(tex).toContain("1 \\times 1.00 \\times 0.80 \\times \\left(0.5 + 0.5 \\times 0.55\\right) = 0.62");
  });
});

describe("hazardTex", () => {
  it("takes the highest defect score and shows it times 100", () => {
    const tex = hazardTex([STEP, { ...STEP, score: 0.3 }], 0.618);
    expect(tex).toContain("\\max\\{0.62,\\ 0.30\\}");
    expect(tex).toContain("= 62");
  });
  it("says what happens with no defects", () => {
    expect(hazardTex([], 0)).toContain("no defects");
  });
});

describe("fmt", () => {
  it("keeps fixed digits", () => {
    expect(fmt(1, 0)).toBe("1");
    expect(fmt(0.5)).toBe("0.50");
  });
});
