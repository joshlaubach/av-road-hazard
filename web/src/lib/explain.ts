import type { DefectRecord, FusionStep, HazardRecord } from "@/api/types";
import { SEVERITY_PLAIN, damagePlain, objectName } from "./labels";
import { scoreBand, toDisplayScore } from "./gauge";

/** The plain-language account of one defect in the record. */
export interface Finding {
  title: string;
  sentences: string[];
}

export interface Summary {
  headline: string;
  findings: Finding[];
  objectsLine: string | null;
  scoreLine: string;
  caution: string;
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function percent(probability: number): string {
  // The model's output is a score, not a calibrated chance, so never claim 100%.
  return probability >= 0.995 ? "over 99%" : `${Math.round(100 * probability)}%`;
}

function roadSentence(inRoad: boolean | null): string {
  if (inRoad === true) return "It is on the road.";
  if (inRoad === false) return "It looks like it is off the road, so it does not add to the score.";
  return "I could not tell where the road is, so I counted it as on the road to be safe.";
}

function neighborSentence(defect: DefectRecord): string {
  const nearest = defect.near_objects[0];
  if (!nearest) return "Nobody is close by.";
  const what = objectName(nearest.label);
  const article = /^[aeiou]/.test(what) ? "An" : "A";
  return `${article} ${what} is close by, so this spot counts for more.`;
}

function finding(defect: DefectRecord, step: FusionStep | undefined): Finding {
  const sentences = [
    `The model is ${percent(defect.confidence)} confident.`,
    roadSentence(defect.in_road),
    neighborSentence(defect),
    SEVERITY_PLAIN[defect.severity],
  ];
  if (step && step.merged_from > 1) {
    sentences.push(
      `The model flagged ${step.merged_from} overlapping squares here, and I merged them into one spot.`,
    );
  }
  return { title: capitalize(damagePlain(defect.name)), sentences };
}

function objectsLine(record: HazardRecord): string | null {
  if (record.objects.length === 0) return null;
  const counts = new Map<string, number>();
  for (const object of record.objects) counts.set(object.label, (counts.get(object.label) ?? 0) + 1);
  const parts = [...counts.entries()].map(([label, n]) => `${n} ${objectName(label, n)}`);
  return `I also saw ${parts.join(", ")}.`;
}

/** Turn a record (and, when there is one, its fusion steps) into the text the page shows. */
export function summarize(record: HazardRecord, steps: readonly FusionStep[] = []): Summary {
  const count = record.defects.length;
  const score = toDisplayScore(record.hazard_score);
  const band = scoreBand(score);
  const headline =
    count === 0
      ? "I did not find any road damage in this photo."
      : `I found ${count} spot${count === 1 ? "" : "s"} of possible road damage.`;
  return {
    headline,
    findings: record.defects.map((defect, i) => finding(defect, steps[i])),
    objectsLine: objectsLine(record),
    scoreLine:
      `The hazard score is ${score} out of 100 (${band.label}). ` +
      "It ranks how urgent the worst spot looks. It is not the chance of a crash.",
    caution:
      count === 0
        ? "That does not prove the road is fine: small or faraway damage is often missed."
        : "I can be wrong in both directions: I miss small or faraway damage, and I sometimes flag plain road.",
  };
}
