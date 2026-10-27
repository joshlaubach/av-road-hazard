import type { HazardRecord, Trace } from "@/api/types";
import { toDisplayScore } from "./gauge";
import { damageName } from "./labels";
import { HAZARD_TEX, SCORE_TEX, hazardTex, stepTex } from "./math";

/** Which layers of the picture are drawn at a step. */
export interface Layers {
  objects: boolean;
  road: boolean;
  tiles: boolean;
  skipped: boolean;
  heat: boolean;
  flagged: boolean;
  defects: boolean;
}

export interface Step {
  id: string;
  title: string;
  /** What happened, in plain words, with this photo's numbers. */
  text: string;
  layers: Layers;
  /** TeX for the mathematics of this step, shown behind a disclosure. */
  math: string[];
}

export interface TileCounts {
  total: number;
  scored: number;
  offRoad: number;
  underObjects: number;
  flagged: number;
}

export function tileCounts(trace: Trace): TileCounts {
  const count = (status: string) => trace.tiles.filter((t) => t.status === status).length;
  return {
    total: trace.tiles.length,
    scored: count("scored"),
    offRoad: count("skipped_off_road"),
    underObjects: count("skipped_object"),
    flagged: trace.tiles.filter((t) => t.flagged.length > 0).length,
  };
}

const NONE: Layers = {
  objects: false, road: false, tiles: false, skipped: false, heat: false, flagged: false, defects: false,
};

function pct(x: number): string {
  return `${Math.round(100 * x)}%`;
}

function chance(p: number): string {
  return p >= 0.995 ? "over 99%" : pct(p);
}

function strongest(trace: Trace): string {
  let best: { code: string; p: number } | null = null;
  for (const tile of trace.tiles) {
    for (const [code, p] of Object.entries(tile.probabilities ?? {})) {
      if (!best || p > best.p) best = { code, p };
    }
  }
  return best ? ` The strongest guess was ${damageName(best.code)}, at ${chance(best.p)}.` : "";
}

/** The steps of the walkthrough for one analysis: what the pipeline did, in order, with real numbers. */
export function buildSteps(record: HazardRecord, trace: Trace): Step[] {
  const [width, height] = record.image_size;
  const counts = tileCounts(trace);
  const objects = record.objects.length;
  const spots = record.defects.length;
  const score = toDisplayScore(record.hazard_score);
  const pothole = trace.thresholds["D40"];
  const fusionMath =
    spots > 0
      ? [SCORE_TEX, ...trace.fusion.map(stepTex), hazardTex(trace.fusion, record.hazard_score)]
      : [HAZARD_TEX];

  return [
    {
      id: "photo",
      title: "The photo",
      text:
        `This is the photo, ${width} by ${height} pixels. I look at it at full size, because small damage ` +
        "disappears when a photo is shrunk.",
      layers: NONE,
      math: [],
    },
    {
      id: "objects",
      title: "People and vehicles",
      text:
        objects > 0
          ? `First I look for people and vehicles: I found ${objects}. They matter twice. A person next to damage ` +
            "makes it more urgent, and I do not look for cracks underneath a car."
          : "First I look for people and vehicles. I found none. Without anyone nearby, damage counts for less.",
      layers: { ...NONE, objects: true },
      math: [],
    },
    {
      id: "road",
      title: "The road",
      text: trace.road_mask
        ? "Next I find the road, tinted here. I only look for cracks and potholes on the road, so a brick wall " +
          "or a shadow on a building is not mistaken for damage."
        : "I could not find the road in this photo, so I looked everywhere and counted every spot as possibly on the road.",
      layers: { ...NONE, objects: true, road: true },
      math: [],
    },
    {
      id: "squares",
      title: "Squares",
      text:
        `I cut the photo into overlapping squares, each ${trace.tile_size} pixels wide. There are ${counts.total}. ` +
        "The model looks at one square at a time, the way it learned from close-up photos of road surface.",
      layers: { ...NONE, road: true, tiles: true },
      math: [],
    },
    {
      id: "skip",
      title: "Squares I skip",
      text:
        `I skip a square when less than ${pct(trace.min_road_cover)} of it is road (${counts.offRoad} squares) or ` +
        `when more than ${pct(trace.max_object_cover)} of it is hidden by a person or vehicle ` +
        `(${counts.underObjects} squares). That leaves ${counts.scored} squares to look at.`,
      layers: { ...NONE, road: true, tiles: true, skipped: true },
      math: [
        String.raw`\text{road share} = \frac{\text{road pixels in the square}}{\text{all pixels in the square}} \ge ${trace.min_road_cover}`,
      ],
    },
    {
      id: "guess",
      title: "The model's guess",
      text:
        "For each square left, the model gives a chance from 0 to 100% for each of four kinds of damage. " +
        "Warmer colors mean a higher chance. A square is flagged when the chance for a kind passes that kind's " +
        `cutoff (for a pothole, ${pothole === undefined ? "its cutoff" : pct(pothole)}). ` +
        `${counts.flagged} square${counts.flagged === 1 ? " was" : "s were"} flagged.${strongest(trace)}`,
      layers: { ...NONE, skipped: true, heat: true, flagged: true },
      math: [String.raw`\text{flagged if } p_c \ge \tau_c, \qquad \tau_{\text{pothole}} = ${pothole ?? "?"}`],
    },
    {
      id: "merge",
      title: "Merging",
      text:
        counts.flagged === 0
          ? "No square was flagged, so there is nothing to merge."
          : "Flagged squares of the same kind that touch belong to one spot of damage, not several. " +
            `${counts.flagged} flagged square${counts.flagged === 1 ? "" : "s"} became ${spots} spot${spots === 1 ? "" : "s"}. ` +
            "The merged box covers all of its squares, and its confidence is the strongest square's.",
      layers: { ...NONE, objects: true, flagged: true, defects: true },
      math: [String.raw`\text{box} = \left(\min x_1,\ \min y_1,\ \max x_2,\ \max y_2\right), \qquad c = \max_i c_i`],
    },
    {
      id: "score",
      title: "The score",
      text:
        spots === 0
          ? "No spots, so the hazard score is 0."
          : "Last, each spot gets a score from how sure the model is, how serious that kind of damage is, whether it " +
            "is on the road and whether someone is close by. The worst spot sets the score for the photo: " +
            `${score} out of 100.`,
      layers: { ...NONE, objects: true, defects: true },
      math: fusionMath,
    },
  ];
}
