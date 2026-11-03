import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HazardRecord } from "@/api/types";
import { ResultCard } from "./ResultCard";
import { SearchView } from "./SearchView";

const search = vi.fn();
const index = vi.fn();

vi.mock("@/api/client", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/api/client")>();
  return { ...real, searchPhotos: (...args: unknown[]) => search(...args), indexPhotos: (...args: unknown[]) => index(...args) };
});

const RECORD: HazardRecord = {
  frame_id: "frame-a",
  source: "upload",
  image_size: [100, 100],
  objects: [{ label: "pedestrian", score: 0.9, box: [0, 0, 10, 10] }],
  defects: [{ name: "D40", confidence: 0.9, box: [0, 0, 50, 50], severity: "high", in_road: true, near_objects: [] }],
  hazard_score: 0.51,
  meta: { time_of_day: "unknown", location: null },
};

beforeEach(() => {
  search.mockReset();
  index.mockReset();
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(cleanup);

describe("ResultCard", () => {
  it("shows the number, score, damage and objects, and says so when there is no preview", () => {
    render(
      <ol>
        <ResultCard number={2} record={RECORD} thumbnailUrl={null} />
      </ol>,
    );
    expect(screen.getByRole("heading", { name: /Result 2: 51 out of 100/ })).toBeTruthy();
    expect(screen.getByText(/Damage found: pothole/)).toBeTruthy();
    expect(screen.getByText(/1 person/)).toBeTruthy();
    expect(screen.getByText("No preview")).toBeTruthy();
  });

  it("shows the visitor's own preview when there is one", () => {
    render(
      <ol>
        <ResultCard number={1} record={RECORD} thumbnailUrl="blob:mine" />
      </ol>,
    );
    expect(screen.getByRole("img", { name: "Your photo, result 1" })).toBeTruthy();
  });
});

describe("SearchView", () => {
  it("asks for words or a photo before searching", async () => {
    render(<SearchView />);
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", expect.stringContaining("Type a question"));
    expect(search).not.toHaveBeenCalled();
  });

  it("sends the question and the chosen filter, then shows a numbered answer and cards", async () => {
    search.mockResolvedValue({
      answer: "A pothole is in [frame-a].",
      cited: ["frame-a"],
      llm: "llama",
      model: "llama3.2",
      notice: null,
      records: [RECORD],
    });
    render(<SearchView />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "which photos have a pothole" } });
    fireEvent.change(screen.getByLabelText("there is this kind of damage"), { target: { value: "D40" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("A pothole is in [1].")).toBeTruthy());
    expect(search).toHaveBeenCalledWith({
      text: "which photos have a pothole",
      image_b64: null,
      filters: { has_defect: "D40" },
      llm: "llama",
      top_k: 5,
    });
    expect(screen.getByRole("heading", { name: /Result 1: 51 out of 100/ })).toBeTruthy();
  });

  it("says plainly when nothing matched", async () => {
    search.mockResolvedValue({ answer: "", cited: [], llm: "llama", model: null, notice: null, records: [] });
    render(<SearchView />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "anything" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/No photos matched/)).toBeTruthy();
  });

  it("shows the server's plain error message", async () => {
    const { AnalyzeError } = await import("@/api/client");
    search.mockRejectedValue(new AnalyzeError(429, "Slow down a moment."));
    render(<SearchView />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText("Slow down a moment.")).toBeTruthy();
  });

  it("keeps a preview of each photo the visitor adds and counts them", async () => {
    index.mockResolvedValue(["frame-a"]);
    render(<SearchView />);
    const file = new File(["x"], "road.jpg", { type: "image/jpeg" });
    fireEvent.change(screen.getByLabelText("Choose photos"), { target: { files: [file] } });
    expect(await screen.findByText(/You have added 1 so far/)).toBeTruthy();
    expect(index).toHaveBeenCalledWith([file]);
  });
});
