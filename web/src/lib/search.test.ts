import { describe, expect, it } from "vitest";
import type { HazardRecord } from "@/api/types";
import { EMPTY_FORM, buildQuery, cardFacts, numberCitations, stripDataUrl } from "./search";

function record(frame_id: string, score = 0.5): HazardRecord {
  return {
    frame_id,
    source: "upload",
    image_size: [100, 100],
    objects: [
      { label: "pedestrian", score: 0.9, box: [0, 0, 10, 10] },
      { label: "pedestrian", score: 0.8, box: [20, 0, 30, 10] },
      { label: "car", score: 0.9, box: [40, 0, 60, 10] },
    ],
    defects: [
      { name: "D40", confidence: 0.9, box: [0, 0, 50, 50], severity: "high", in_road: true, near_objects: [] },
    ],
    hazard_score: score,
    meta: { time_of_day: "unknown", location: null },
  };
}

describe("buildQuery", () => {
  it("returns null when there are no words and no photo", () => {
    expect(buildQuery(EMPTY_FORM)).toBeNull();
    expect(buildQuery({ ...EMPTY_FORM, text: "   " })).toBeNull();
  });

  it("trims the words and sends no filters when none are chosen", () => {
    expect(buildQuery({ ...EMPTY_FORM, text: "  a pothole " })).toEqual({
      text: "a pothole",
      image_b64: null,
      filters: null,
      llm: "llama",
      top_k: 5,
    });
  });

  it("turns the 0 to 100 score filter into the API's 0 to 1 and keeps only the chosen filters", () => {
    const request = buildQuery({ ...EMPTY_FORM, text: "x", damage: "D40", minScore: 40 });
    expect(request?.filters).toEqual({ has_defect: "D40", min_hazard_score: 0.4 });
  });

  it("allows a photo with no words", () => {
    expect(buildQuery({ ...EMPTY_FORM, imageB64: "AAAA" })).toEqual({
      text: null,
      image_b64: "AAAA",
      filters: null,
      llm: "llama",
      top_k: 5,
    });
  });
});

describe("stripDataUrl", () => {
  it("keeps only the base64 part", () => {
    expect(stripDataUrl("data:image/png;base64,AAAA")).toBe("AAAA");
    expect(stripDataUrl("AAAA")).toBe("AAAA");
  });
});

describe("numberCitations", () => {
  const records = [record("frame-a"), record("frame-b")];

  it("swaps frame ids for the numbers on the cards", () => {
    expect(numberCitations("Seen in [frame-b] and [frame-a].", records)).toBe("Seen in [2] and [1].");
  });

  it("handles several ids in one bracket", () => {
    expect(numberCitations("Both [frame-a, frame-b].", records)).toBe("Both [1, 2].");
  });

  it("leaves a bracket alone when an id is not on the cards", () => {
    expect(numberCitations("Odd [unverified source] and [frame-z].", records)).toBe(
      "Odd [unverified source] and [frame-z].",
    );
  });
});

describe("cardFacts", () => {
  it("gives the whole-number score, its zone, the damage and the counted objects", () => {
    const facts = cardFacts(record("a", 0.51));
    expect(facts.score).toBe(51);
    expect(facts.zone).toBeTruthy();
    expect(facts.damages).toEqual(["pothole"]);
    expect(facts.objects).toEqual(["2 people", "1 car"]);
  });
});
