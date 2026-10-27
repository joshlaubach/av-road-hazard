import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HazardRecord } from "@/api/types";
import { summarize } from "@/lib/explain";
import { ExamplePicker } from "./ExamplePicker";
import { Findings } from "./Findings";
import { Gauge } from "./Gauge";
import { PhotoView } from "./PhotoView";
import { Uploader } from "./Uploader";

afterEach(cleanup);

const RECORD: HazardRecord = {
  frame_id: "up-test",
  source: "upload",
  image_size: [1000, 500],
  objects: [{ label: "pedestrian", score: 0.9, box: [100, 100, 150, 250] }],
  defects: [
    {
      name: "D40",
      confidence: 0.9,
      box: [400, 300, 600, 450],
      severity: "high",
      in_road: true,
      near_objects: [],
    },
  ],
  hazard_score: 0.51,
  meta: { time_of_day: "unknown", location: null },
};

describe("Gauge", () => {
  it("is a labeled image that reads the score and its zone", () => {
    render(<Gauge score={49} />);
    expect(screen.getByRole("img", { name: "Hazard score 49 out of 100: Moderate" })).toBeTruthy();
  });
});

describe("Findings", () => {
  it("shows the headline, each spot and the plain caution", () => {
    render(<Findings summary={summarize(RECORD)} />);
    expect(screen.getByRole("heading", { name: "I found 1 spot of possible road damage." })).toBeTruthy();
    expect(screen.getByText("A pothole")).toBeTruthy();
    expect(screen.getByText(/not the chance of a crash/)).toBeTruthy();
  });
});

describe("PhotoView", () => {
  it("draws one box and one label per object and defect, named in words", () => {
    const { container } = render(
      <PhotoView imageUrl="/x.jpg" alt="test photo" record={RECORD} showObjects showDefects />,
    );
    expect(container.querySelectorAll("rect")).toHaveLength(2);
    expect(screen.getByText("person")).toBeTruthy();
    expect(screen.getByText("pothole")).toBeTruthy();
    expect(screen.getByAltText("test photo")).toBeTruthy();
  });

  it("hides a layer when its switch is off", () => {
    const { container } = render(
      <PhotoView imageUrl="/x.jpg" alt="test photo" record={RECORD} showObjects={false} showDefects />,
    );
    expect(container.querySelectorAll("rect")).toHaveLength(1);
    expect(screen.queryByText("person")).toBeNull();
  });
});

describe("Uploader", () => {
  it("passes the chosen photo on", () => {
    const onFile = vi.fn();
    const { container } = render(<Uploader onFile={onFile} isBusy={false} />);
    const input = container.querySelector("input") as HTMLInputElement;
    const file = new File(["x"], "road.jpg", { type: "image/jpeg" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("is disabled while an analysis is running", () => {
    const { container } = render(<Uploader onFile={vi.fn()} isBusy />);
    expect((container.querySelector("input") as HTMLInputElement).disabled).toBe(true);
  });
});

describe("ExamplePicker", () => {
  it("offers both examples and reports which one was picked", () => {
    const onPick = vi.fn();
    render(<ExamplePicker onPick={onPick} isBusy={false} />);
    fireEvent.click(screen.getByRole("button", { name: /Where it gets it wrong/ }));
    expect(onPick).toHaveBeenCalledWith("stellenbosch-limits");
    expect(screen.getAllByRole("button")).toHaveLength(2);
  });
});
