/** Gauge geometry and colors. The stored hazard_score is 0 to 1; people see a whole number 0 to 100. */

export interface GaugeStop {
  at: number;
  color: string;
}

/** Blue, green, yellow, orange, red from the left end (0) to the right end (100). */
export const GAUGE_STOPS: readonly GaugeStop[] = [
  { at: 0, color: "#2f6fdb" },
  { at: 25, color: "#1fa971" },
  { at: 50, color: "#e4c51f" },
  { at: 75, color: "#f08a24" },
  { at: 100, color: "#d7372b" },
];

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** The number people see: the stored score times 100, rounded to a whole number. */
export function toDisplayScore(hazardScore: number): number {
  return Math.round(100 * clamp(hazardScore, 0, 1));
}

/**
 * The direction the needle points, in ordinary math coordinates (x right, y up).
 * A score of 0 points at (-1, 0), 100 at (1, 0) and 50 straight up at (0, 1):
 * the angle from the positive x axis is pi * (1 - score / 100).
 */
export function needleVector(score: number): [number, number] {
  const angle = Math.PI * (1 - clamp(score, 0, 100) / 100);
  return [Math.cos(angle), Math.sin(angle)];
}

function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex(rgb: readonly number[]): string {
  return `#${rgb.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}

/** The gauge color at a score, blending the two nearest stops. */
export function gaugeColor(score: number): string {
  const value = clamp(score, 0, 100);
  const upper = GAUGE_STOPS.findIndex((stop) => stop.at >= value);
  const high = GAUGE_STOPS[upper] ?? GAUGE_STOPS[GAUGE_STOPS.length - 1];
  const low = GAUGE_STOPS[Math.max(0, upper - 1)] ?? high;
  if (!high || !low || high.at === low.at) return high?.color ?? "#000000";
  const t = (value - low.at) / (high.at - low.at);
  const a = hexToRgb(low.color);
  const b = hexToRgb(high.color);
  return rgbToHex(a.map((v, i) => v + t * ((b[i] ?? v) - v)));
}

export interface ArcSegment {
  path: string;
  color: string;
}

/**
 * The colored band of the gauge as small arc pieces, one per step of the score, so each piece has
 * the color of its own score. The half circle is centered at (cx, cy); the left end is 0.
 */
export function arcSegments(
  steps: number,
  cx: number,
  cy: number,
  outer: number,
  inner: number,
): ArcSegment[] {
  const point = (score: number, radius: number): string => {
    const [x, y] = needleVector(score);
    return `${(cx + radius * x).toFixed(2)} ${(cy - radius * y).toFixed(2)}`;
  };
  return Array.from({ length: steps }, (_, i) => {
    const from = (100 * i) / steps;
    const to = (100 * (i + 1)) / steps;
    const path =
      `M ${point(from, outer)} A ${outer} ${outer} 0 0 1 ${point(to, outer)} ` +
      `L ${point(to, inner)} A ${inner} ${inner} 0 0 0 ${point(from, inner)} Z`;
    return { path, color: gaugeColor((from + to) / 2) };
  });
}

export interface ScoreBand {
  label: string;
  /** What the band means in one plain sentence. */
  meaning: string;
}

const BANDS: readonly (ScoreBand & { upTo: number })[] = [
  { upTo: 19, label: "Calm", meaning: "Nothing here looks urgent." },
  { upTo: 39, label: "Low", meaning: "Something is there, but it looks minor." },
  { upTo: 59, label: "Moderate", meaning: "Worth slowing down for." },
  { upTo: 79, label: "High", meaning: "A real hazard on the road." },
  { upTo: 100, label: "Urgent", meaning: "Damage right where someone is." },
];

/** The word for a score. These are zones on the gauge, not measured risk levels. */
export function scoreBand(score: number): ScoreBand {
  const value = clamp(Math.round(score), 0, 100);
  const band = BANDS.find((b) => value <= b.upTo) ?? BANDS[BANDS.length - 1];
  return { label: band?.label ?? "", meaning: band?.meaning ?? "" };
}
