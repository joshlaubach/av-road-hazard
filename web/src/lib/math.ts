import type { FusionStep } from "@/api/types";
import { toDisplayScore } from "./gauge";

/** The formulas behind the hazard score (fusion/fuse.py), as TeX for KaTeX. */
export const PROXIMITY_TEX = String.raw`r_{jk} = v_j \, c_j \, e^{-d_{jk}}, \qquad \pi_k = \max_j \, r_{jk}`;
export const SCORE_TEX = String.raw`s_k = m_k \cdot w_k \cdot c_k \cdot \left(\tfrac{1}{2} + \tfrac{1}{2}\,\pi_k\right)`;
export const HAZARD_TEX = String.raw`H = \min\!\left(1,\ \max_k \, s_k\right), \qquad \text{score} = \operatorname{round}(100\,H)`;

/** A number for TeX: fixed digits, no trailing surprises. */
export function fmt(value: number, digits = 2): string {
  return value.toFixed(digits);
}

/** One defect's score with its own numbers plugged into the formula. */
export function stepTex(step: FusionStep): string {
  const { road_multiplier: m, severity_weight: w, confidence: c, proximity: p, score: s } = step;
  return String.raw`s = ${fmt(m, 0)} \times ${fmt(w)} \times ${fmt(c)} \times \left(0.5 + 0.5 \times ${fmt(p)}\right) = ${fmt(s)}`;
}

/** The frame's score from the defect scores: the highest one, capped at 1, shown times 100. */
export function hazardTex(steps: readonly FusionStep[], hazardScore: number): string {
  if (steps.length === 0) return String.raw`H = 0 \quad (\text{no defects}), \qquad \text{score} = 0`;
  const scores = steps.map((s) => fmt(s.score)).join(",\\ ");
  return String.raw`H = \min\!\left(1,\ \max\{${scores}\}\right) = ${fmt(hazardScore)}, \qquad \text{score} = \operatorname{round}(100 \times ${fmt(hazardScore)}) = ${toDisplayScore(hazardScore)}`;
}
