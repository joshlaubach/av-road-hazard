import { describe, expect, it } from "vitest";
import { GAUGE_STOPS, arcSegments, gaugeColor, needleVector, scoreBand, toDisplayScore } from "./gauge";

describe("toDisplayScore", () => {
  it("shows the stored 0 to 1 score as a whole number 0 to 100", () => {
    expect(toDisplayScore(0.62)).toBe(62);
    expect(toDisplayScore(0.511)).toBe(51);
    expect(toDisplayScore(0)).toBe(0);
    expect(toDisplayScore(1)).toBe(100);
  });
  it("never leaves the range", () => {
    expect(toDisplayScore(-0.5)).toBe(0);
    expect(toDisplayScore(1.7)).toBe(100);
  });
});

describe("needleVector", () => {
  it("points at (-1, 0) for 0, (1, 0) for 100 and straight up for 50", () => {
    const [x0, y0] = needleVector(0);
    const [x100, y100] = needleVector(100);
    const [x50, y50] = needleVector(50);
    expect([x0, Math.abs(y0)]).toEqual([-1, expect.closeTo(0, 9)]);
    expect([x100, Math.abs(y100)]).toEqual([1, expect.closeTo(0, 9)]);
    expect(x50).toBeCloseTo(0, 9);
    expect(y50).toBeCloseTo(1, 9);
  });
  it("is a unit vector everywhere", () => {
    for (const score of [0, 13, 37, 50, 71, 100]) {
      const [x, y] = needleVector(score);
      expect(Math.hypot(x, y)).toBeCloseTo(1, 9);
    }
  });
});

describe("gaugeColor", () => {
  it("is blue, green, yellow, orange, red at the five stops", () => {
    expect(GAUGE_STOPS.map((s) => gaugeColor(s.at))).toEqual(GAUGE_STOPS.map((s) => s.color));
  });
  it("blends between stops and clamps outside the range", () => {
    expect(gaugeColor(-10)).toBe(GAUGE_STOPS[0]?.color);
    expect(gaugeColor(140)).toBe(GAUGE_STOPS[4]?.color);
    const midway = gaugeColor(12.5);
    expect(midway).not.toBe(GAUGE_STOPS[0]?.color);
    expect(midway).not.toBe(GAUGE_STOPS[1]?.color);
  });
});

describe("arcSegments", () => {
  it("makes one closed piece per step, colored by its own score", () => {
    const pieces = arcSegments(20, 100, 100, 90, 60);
    expect(pieces).toHaveLength(20);
    expect(pieces[0]?.path.startsWith("M ")).toBe(true);
    expect(pieces[0]?.path.endsWith("Z")).toBe(true);
    expect(pieces[0]?.color).not.toBe(pieces[19]?.color);
  });
});

describe("scoreBand", () => {
  it("names the five zones of the gauge", () => {
    expect([0, 19, 20, 45, 79, 80, 100].map((s) => scoreBand(s).label)).toEqual([
      "Calm", "Calm", "Low", "Moderate", "High", "Urgent", "Urgent",
    ]);
  });
});
