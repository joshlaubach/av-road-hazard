import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RECORD, TRACE } from "@/lib/fixtures";
import { Walkthrough } from "./Walkthrough";

function mockBrowser(reducedMotion: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: reducedMotion,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
  // jsdom has no canvas; the layers draw only when a context exists.
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

function renderWalkthrough() {
  return render(<Walkthrough imageUrl="/x.jpg" alt="test photo" record={RECORD} trace={TRACE} />);
}

const heading = () => screen.getByRole("heading", { level: 3 }).textContent;

beforeEach(() => mockBrowser(false));
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("Walkthrough", () => {
  it("opens on the finished result, with the last step selected", () => {
    renderWalkthrough();
    expect(heading()).toBe("Step 8 of 8: The score");
    expect(screen.getByRole("button", { name: "8. The score" }).getAttribute("aria-current")).toBe("step");
    expect((screen.getByRole("button", { name: "Next" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("goes to any step from its button and says what happened with this photo's numbers", () => {
    renderWalkthrough();
    fireEvent.click(screen.getByRole("button", { name: "5. Squares I skip" }));
    expect(heading()).toBe("Step 5 of 8: Squares I skip");
    expect(screen.getByText(/less than 40% of it is road \(4 squares\)/)).toBeTruthy();
  });

  it("steps back and forward", () => {
    renderWalkthrough();
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(heading()).toBe("Step 7 of 8: Merging");
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(heading()).toBe("Step 8 of 8: The score");
  });

  it("plays from the start and moves on by itself", () => {
    vi.useFakeTimers();
    renderWalkthrough();
    fireEvent.click(screen.getByRole("button", { name: "Watch how it was found" }));
    expect(heading()).toBe("Step 1 of 8: The photo");
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    act(() => {
      vi.advanceTimersByTime(4600);
    });
    expect(heading()).toBe("Step 2 of 8: People and vehicles");
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    act(() => {
      vi.advanceTimersByTime(10000);
    });
    expect(heading()).toBe("Step 2 of 8: People and vehicles"); // paused, so it stays
  });

  it("offers no autoplay to someone who asked for less motion, but the steps still work", () => {
    mockBrowser(true);
    renderWalkthrough();
    expect(screen.queryByRole("button", { name: "Watch how it was found" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(heading()).toBe("Step 7 of 8: Merging");
  });

  it("lets the heatmap be chosen by kind of damage only while it is showing", () => {
    renderWalkthrough();
    const select = screen.getByLabelText("Heatmap shows") as HTMLSelectElement;
    expect(select.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText("Heatmap"));
    expect(select.disabled).toBe(false);
    fireEvent.change(select, { target: { value: "D40" } });
    expect(select.value).toBe("D40");
  });

  it("puts the math for a step behind a disclosure", () => {
    renderWalkthrough();
    fireEvent.click(screen.getByRole("button", { name: "5. Squares I skip" }));
    expect(screen.getByText("The math for this step")).toBeTruthy();
  });
});
