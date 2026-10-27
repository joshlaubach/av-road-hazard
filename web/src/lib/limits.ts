/**
 * How big a defect must look to be found, from the pasted-patch test in eval/results.json
 * (`effective_range`). A test keeps these numbers equal to that file.
 */
export interface RangeBand {
  /** The band's name in results.json. */
  band: string;
  /** Plain words for the band. */
  widthText: string;
  /** About how far ahead that is for a 0.5 m wide defect and the nuScenes front camera. */
  distanceText: string;
  found: number;
  trials: number;
}

export const RANGE_BANDS: readonly RangeBand[] = [
  { band: "large (>=128 px)", widthText: "128 pixels wide or more", distanceText: "about 5 meters ahead or closer", found: 34, trials: 57 },
  { band: "medium (64-128 px)", widthText: "64 to 128 pixels wide", distanceText: "about 5 to 10 meters ahead", found: 17, trials: 56 },
  { band: "small (<64 px)", widthText: "under 64 pixels wide", distanceText: "farther than about 10 meters", found: 3, trials: 57 },
];

export function percentFound(band: RangeBand): number {
  return Math.round((100 * band.found) / band.trials);
}
