import type { HazardRecord, TileTrace, Trace } from "@/api/types";

/** A small trace and record for tests: 10 squares (4 off the road, 1 under an object, 5 scored, 2 flagged). */
export const PROBS = { D00: 0.01, D10: 0.02, D20: 0.01, D40: 0.02 };

export function tile(status: TileTrace["status"], flagged: string[] = [], p = PROBS): TileTrace {
  return {
    row: 0, col: 0, box: [0, 0, 100, 100], object_cover: status === "skipped_object" ? 0.9 : 0,
    road_cover: status === "skipped_off_road" ? 0 : 1, status,
    probabilities: status === "scored" ? p : null, flagged,
  };
}

export const TRACE: Trace = {
  image_size: [1000, 500], tile_size: 384, tile_overlap: 0.5,
  thresholds: { D00: 0.9, D10: 0.75, D20: 0.65, D40: 0.75 }, max_object_cover: 0.3, min_road_cover: 0.4,
  tiles: [
    tile("skipped_off_road"), tile("skipped_off_road"), tile("skipped_off_road"), tile("skipped_off_road"),
    tile("skipped_object"),
    tile("scored"), tile("scored"), tile("scored"),
    tile("scored", ["D40"], { ...PROBS, D40: 0.9996 }), tile("scored", ["D40"], { ...PROBS, D40: 0.8 }),
  ],
  road_mask: { width: 10, height: 5, png_base64: "AAAA" },
  fusion: [{
    name: "D40", merged_from: 2, in_road: true, road_multiplier: 1, severity: "high", severity_weight: 1,
    confidence: 0.9996, proximity: 0, score: 0.5,
  }],
};

export const RECORD: HazardRecord = {
  frame_id: "up-t", source: "upload", image_size: [1000, 500],
  objects: [{ label: "car", score: 0.9, box: [0, 0, 10, 10] }],
  defects: [{ name: "D40", confidence: 0.9996, box: [0, 0, 200, 100], severity: "high", in_road: true, near_objects: [] }],
  hazard_score: 0.5, meta: { time_of_day: "unknown", location: null },
};
