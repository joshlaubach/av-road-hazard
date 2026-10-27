import { describe, expect, it } from "vitest";
import { RECORD, TRACE, tile } from "./fixtures";
import { buildSteps, tileCounts } from "./walkthrough";

describe("tileCounts", () => {
  it("counts the squares by what happened to them", () => {
    expect(tileCounts(TRACE)).toEqual({ total: 10, scored: 5, offRoad: 4, underObjects: 1, flagged: 2 });
  });
});

describe("buildSteps", () => {
  const steps = buildSteps(RECORD, TRACE);
  const byId = Object.fromEntries(steps.map((s) => [s.id, s]));

  it("has the eight steps in order", () => {
    expect(steps.map((s) => s.id)).toEqual(["photo", "objects", "road", "squares", "skip", "guess", "merge", "score"]);
  });

  it("states this photo's own numbers", () => {
    expect(byId["photo"]?.text).toContain("1000 by 500");
    expect(byId["objects"]?.text).toContain("I found 1");
    expect(byId["squares"]?.text).toContain("384 pixels");
    expect(byId["squares"]?.text).toContain("There are 10");
    expect(byId["skip"]?.text).toContain("less than 40% of it is road (4 squares)");
    expect(byId["skip"]?.text).toContain("more than 30%");
    expect(byId["skip"]?.text).toContain("That leaves 5 squares");
    expect(byId["guess"]?.text).toContain("2 squares were flagged");
    expect(byId["guess"]?.text).toContain("pothole, at over 99%");
    expect(byId["merge"]?.text).toContain("2 flagged squares became 1 spot");
    expect(byId["score"]?.text).toContain("50 out of 100");
  });

  it("builds up the picture layer by layer", () => {
    expect(byId["photo"]?.layers.objects).toBe(false);
    expect(byId["objects"]?.layers.objects).toBe(true);
    expect(byId["road"]?.layers.road).toBe(true);
    expect(byId["squares"]?.layers.tiles).toBe(true);
    expect(byId["skip"]?.layers.skipped).toBe(true);
    expect(byId["guess"]?.layers.heat).toBe(true);
    expect(byId["merge"]?.layers.defects).toBe(true);
  });

  it("gives the mathematics with the photo's numbers at the last step", () => {
    const math = byId["score"]?.math.join(" ") ?? "";
    expect(math).toContain("0.50");
    expect(math).toContain("\\operatorname{round}");
  });

  it("says plainly when there is no road mask, nothing flagged and no spots", () => {
    const quiet = buildSteps(
      { ...RECORD, objects: [], defects: [], hazard_score: 0 },
      { ...TRACE, road_mask: null, tiles: [tile("scored"), tile("scored")], fusion: [] },
    );
    const q = Object.fromEntries(quiet.map((s) => [s.id, s]));
    expect(q["objects"]?.text).toContain("found none");
    expect(q["road"]?.text).toContain("could not find the road");
    expect(q["merge"]?.text).toBe("No square was flagged, so there is nothing to merge.");
    expect(q["score"]?.text).toBe("No spots, so the hazard score is 0.");
  });
});
