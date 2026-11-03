import type { HazardRecord, QueryRequest } from "@/api/types";
import { damageName, objectName } from "./labels";
import { scoreBand, toDisplayScore } from "./gauge";

/** What the search form holds. Empty strings and 0 mean "no filter". */
export interface SearchForm {
  text: string;
  imageB64: string | null;
  damage: string;
  object: string;
  minScore: number; // 0 to 100, as people see it
}

export const EMPTY_FORM: SearchForm = { text: "", imageB64: null, damage: "", object: "", minScore: 0 };

/** How many photos a search brings back. The API accepts 1 to 50. */
export const TOP_K = 5;

export const DAMAGE_CHOICES = ["D00", "D10", "D20", "D40"] as const;
export const OBJECT_CHOICES = ["pedestrian", "cyclist", "motorcycle", "car", "truck", "bus", "barrier", "traffic_cone"] as const;

/** The request the API wants, or null when there is nothing to search for (no words and no photo). */
export function buildQuery(form: SearchForm, llm: QueryRequest["llm"] = "llama"): QueryRequest | null {
  const text = form.text.trim();
  if (!text && !form.imageB64) return null;
  const filters: NonNullable<QueryRequest["filters"]> = {};
  if (form.damage) filters.has_defect = form.damage as (typeof DAMAGE_CHOICES)[number];
  if (form.object) filters.has_object = form.object;
  if (form.minScore > 0) filters.min_hazard_score = form.minScore / 100;
  const hasFilter = Object.keys(filters).length > 0;
  return {
    text: text || null,
    image_b64: form.imageB64,
    filters: hasFilter ? filters : null,
    llm,
    top_k: TOP_K,
  };
}

/** The base64 part of a data URL ("data:image/png;base64,AAAA" becomes "AAAA"). */
export function stripDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma === -1 ? dataUrl : dataUrl.slice(comma + 1);
}

/**
 * Swap the long frame ids the model cites for the numbers on the result cards: "[frame-a]" becomes
 * "[1]" and "[frame-a, frame-b]" becomes "[1, 2]". An id that is not on the cards is left as it is.
 */
export function numberCitations(answer: string, records: readonly HazardRecord[]): string {
  const number = new Map(records.map((record, i) => [record.frame_id, i + 1]));
  return answer.replace(/\[([^\]]+)\]/g, (whole: string, inner: string) => {
    const parts = inner.split(",").map((part) => part.trim());
    const numbers = parts.map((part) => number.get(part));
    return numbers.every((n) => n !== undefined) ? `[${numbers.join(", ")}]` : whole;
  });
}

export interface CardFacts {
  score: number;
  zone: string;
  damages: string[];
  objects: string[];
}

/** The short facts a result card shows: score, zone, each kind of damage and each kind of object. */
export function cardFacts(record: HazardRecord): CardFacts {
  const score = toDisplayScore(record.hazard_score);
  const damages = [...new Set(record.defects.map((d) => damageName(d.name)))];
  const counts = new Map<string, number>();
  for (const o of record.objects) counts.set(o.label, (counts.get(o.label) ?? 0) + 1);
  const objects = [...counts].map(([label, n]) => `${n} ${objectName(label, n)}`);
  return { score, zone: scoreBand(score).label, damages, objects };
}
