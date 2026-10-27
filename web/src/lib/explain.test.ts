import { describe, expect, it } from "vitest";
import type { FusionStep, HazardRecord } from "@/api/types";
import { summarize } from "./explain";
import { damagePlain, objectName } from "./labels";

function record(overrides: Partial<HazardRecord> = {}): HazardRecord {
  return {
    frame_id: "up-test",
    source: "upload",
    image_size: [640, 360],
    objects: [],
    defects: [],
    hazard_score: 0,
    meta: { time_of_day: "unknown", location: null },
    ...overrides,
  };
}

const POTHOLE = {
  name: "D40",
  confidence: 0.9996,
  box: [10, 10, 100, 100] as [number, number, number, number],
  severity: "high" as const,
  in_road: true,
  near_objects: [],
};

describe("summarize", () => {
  it("says so plainly when nothing was found, and warns that this proves nothing", () => {
    const s = summarize(record());
    expect(s.headline).toBe("I did not find any road damage in this photo.");
    expect(s.findings).toEqual([]);
    expect(s.caution).toContain("does not prove");
    expect(s.scoreLine).toContain("0 out of 100");
  });

  it("describes a pothole in the road with nobody near", () => {
    const s = summarize(record({ defects: [POTHOLE], hazard_score: 0.511 }));
    expect(s.headline).toBe("I found 1 spot of possible road damage.");
    expect(s.findings[0]?.title).toBe("A pothole");
    expect(s.findings[0]?.sentences).toContain("The model is over 99% confident.");
    expect(s.findings[0]?.sentences).toContain("It is on the road.");
    expect(s.findings[0]?.sentences).toContain("Nobody is close by.");
    expect(s.scoreLine).toContain("51 out of 100");
  });

  it("names a nearby person and says it raises the score", () => {
    const near = { label: "pedestrian", iou: 0, dist_px: 12 };
    const s = summarize(record({ defects: [{ ...POTHOLE, near_objects: [near] }] }));
    expect(s.findings[0]?.sentences).toContain("A person is close by, so this spot counts for more.");
  });

  it("uses the right article for a nearby object that starts with a vowel", () => {
    const near = { label: "ass", iou: 0, dist_px: 1 };
    expect(summarize(record({ defects: [{ ...POTHOLE, near_objects: [near] }] })).findings[0]?.sentences).toContain(
      "An ass is close by, so this spot counts for more.",
    );
  });

  it("explains an unknown road and an off-road spot honestly", () => {
    const unknown = summarize(record({ defects: [{ ...POTHOLE, in_road: null }] }));
    expect(unknown.findings[0]?.sentences.join(" ")).toContain("could not tell where the road is");
    const off = summarize(record({ defects: [{ ...POTHOLE, in_road: false }] }));
    expect(off.findings[0]?.sentences.join(" ")).toContain("off the road");
  });

  it("mentions merged squares only when more than one was merged", () => {
    const step = (n: number): FusionStep => ({
      name: "D40", merged_from: n, in_road: true, road_multiplier: 1, severity: "high",
      severity_weight: 1, confidence: 1, proximity: 0, score: 0.5,
    });
    const merged = summarize(record({ defects: [POTHOLE] }), [step(3)]);
    expect(merged.findings[0]?.sentences.join(" ")).toContain("3 overlapping squares");
    const single = summarize(record({ defects: [POTHOLE] }), [step(1)]);
    expect(single.findings[0]?.sentences.join(" ")).not.toContain("overlapping");
  });

  it("counts the objects it saw", () => {
    const o = (label: string) => ({ label, score: 0.9, box: [0, 0, 5, 5] as [number, number, number, number] });
    const s = summarize(record({ objects: [o("car"), o("car"), o("pedestrian")] }));
    expect(s.objectsLine).toBe("I also saw 2 cars, 1 person.");
    expect(summarize(record()).objectsLine).toBeNull();
  });

  it("always says the score is not the chance of a crash", () => {
    expect(summarize(record()).scoreLine).toContain("not the chance of a crash");
  });
});

describe("labels", () => {
  it("knows the four damage codes and falls back for others", () => {
    expect(damagePlain("D00")).toBe("a crack that runs along the road");
    expect(damagePlain("D10")).toBe("a crack that runs across the road");
    expect(damagePlain("D20")).toContain("alligator");
    expect(damagePlain("D40")).toBe("a pothole");
    expect(damagePlain("pothole")).toContain("pothole");
  });
  it("pluralizes object names", () => {
    expect(objectName("pedestrian", 3)).toBe("people");
    expect(objectName("traffic_cone", 1)).toBe("traffic cone");
    expect(objectName("scooter", 2)).toBe("scooters");
  });
});
