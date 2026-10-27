import { describe, expect, it } from "vitest";
import type { TileTrace } from "@/api/types";
import { heatColor, heatGrid } from "./heat";

function tile(box: [number, number, number, number], probabilities: Record<string, number> | null): TileTrace {
  return {
    row: 0, col: 0, box, object_cover: 0, road_cover: 1,
    status: probabilities ? "scored" : "skipped_off_road", probabilities, flagged: [],
  };
}

describe("heatGrid", () => {
  it("covers the photo's shape and leaves cells with no scored square empty", () => {
    const grid = heatGrid([tile([0, 0, 50, 50], { D00: 0.9, D10: 0.1, D20: 0.1, D40: 0.2 })], [100, 50], 10);
    expect([grid.cols, grid.rows]).toEqual([10, 5]);
    expect(grid.values[0]).toBeCloseTo(0.9, 5); // top left cell is inside the square
    expect(Number.isNaN(grid.values[9])).toBe(true); // top right is not
  });

  it("takes the highest of overlapping squares, never their sum", () => {
    const a = tile([0, 0, 60, 50], { D00: 0.3, D10: 0, D20: 0, D40: 0 });
    const b = tile([40, 0, 100, 50], { D00: 0.8, D10: 0, D20: 0, D40: 0 });
    const grid = heatGrid([a, b], [100, 50], 10);
    expect(grid.values[0]).toBeCloseTo(0.3, 5);
    expect(grid.values[5]).toBeCloseTo(0.8, 5); // covered by both: the higher one
    expect(grid.values[9]).toBeCloseTo(0.8, 5);
  });

  it("can show one kind of damage on its own", () => {
    const t = tile([0, 0, 100, 50], { D00: 0.2, D10: 0.1, D20: 0.1, D40: 0.9 });
    expect(heatGrid([t], [100, 50], 4, "D00").values[0]).toBeCloseTo(0.2, 5);
    expect(heatGrid([t], [100, 50], 4, "D40").values[0]).toBeCloseTo(0.9, 5);
    expect(heatGrid([t], [100, 50], 4).values[0]).toBeCloseTo(0.9, 5);
  });

  it("ignores skipped squares", () => {
    const grid = heatGrid([tile([0, 0, 100, 50], null)], [100, 50], 4);
    expect(Array.from(grid.values).every(Number.isNaN)).toBe(true);
  });
});

describe("heatColor", () => {
  it("runs from a faint blue to a strong red", () => {
    const low = heatColor(0);
    const high = heatColor(1);
    expect(low.color).toBe("#2f6fdb");
    expect(high.color).toBe("#d7372b");
    expect(low.alpha).toBeLessThan(high.alpha);
  });
  it("clamps outside 0 to 1", () => {
    expect(heatColor(-3)).toEqual(heatColor(0));
    expect(heatColor(7)).toEqual(heatColor(1));
  });
});
