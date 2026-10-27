import type { TileTrace } from "@/api/types";
import { gaugeColor } from "./gauge";

export interface HeatGrid {
  cols: number;
  rows: number;
  /** Row by row, left to right: the highest probability of any scored square over the cell, or NaN where none. */
  values: Float32Array;
}

/** A square's probability: for one kind of damage, or the highest of the four. */
function tileProbability(tile: TileTrace, code: string | null): number | null {
  if (!tile.probabilities) return null;
  if (code) return tile.probabilities[code] ?? null;
  return Math.max(...Object.values(tile.probabilities));
}

/**
 * The squares' probabilities as a coarse grid over the photo. Squares overlap, so each cell holds
 * the maximum over the squares that cover it (not a sum), and cells no scored square covers stay NaN.
 */
export function heatGrid(
  tiles: readonly TileTrace[],
  imageSize: readonly [number, number],
  cols: number,
  code: string | null = null,
): HeatGrid {
  const [width, height] = imageSize;
  const rows = Math.max(1, Math.round((cols * height) / width));
  const values = new Float32Array(cols * rows).fill(Number.NaN);
  for (const tile of tiles) {
    const p = tileProbability(tile, code);
    if (p === null) continue;
    const [x1, y1, x2, y2] = tile.box;
    const c0 = Math.max(0, Math.floor((x1 / width) * cols));
    const c1 = Math.min(cols, Math.ceil((x2 / width) * cols));
    const r0 = Math.max(0, Math.floor((y1 / height) * rows));
    const r1 = Math.min(rows, Math.ceil((y2 / height) * rows));
    for (let r = r0; r < r1; r++) {
      for (let c = c0; c < c1; c++) {
        const i = r * cols + c;
        const old = values[i] ?? Number.NaN;
        if (Number.isNaN(old) || p > old) values[i] = p;
      }
    }
  }
  return { cols, rows, values };
}

/** The color of a probability: the gauge's blue-to-red, clearer as the probability rises. */
export function heatColor(probability: number): { color: string; alpha: number } {
  const p = Math.min(1, Math.max(0, probability));
  return { color: gaugeColor(100 * p), alpha: 0.12 + 0.5 * p };
}
