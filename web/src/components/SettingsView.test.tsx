import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LLM_KEY } from "@/lib/settings";
import { SearchView } from "./SearchView";
import { SettingsView } from "./SettingsView";

const search = vi.fn();

vi.mock("@/api/client", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/api/client")>();
  return { ...real, searchPhotos: (...args: unknown[]) => search(...args) };
});

beforeEach(() => {
  window.localStorage.clear();
  search.mockReset();
});
afterEach(cleanup);

describe("SettingsView", () => {
  it("starts on the local model and offers both choices in plain words", () => {
    render(<SettingsView />);
    expect((screen.getByRole("radio", { name: /Local model/ }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByRole("radio", { name: /Claude/ })).toBeTruthy();
    expect(screen.getByText(/needs an API key/)).toBeTruthy();
  });

  it("shows a choice that was saved earlier", async () => {
    window.localStorage.setItem(LLM_KEY, "claude");
    render(<SettingsView />);
    await waitFor(() => expect((screen.getByRole("radio", { name: /Claude/ }) as HTMLInputElement).checked).toBe(true));
  });

  it("saves a new choice in the browser", () => {
    render(<SettingsView />);
    fireEvent.click(screen.getByRole("radio", { name: /Claude/ }));
    expect(window.localStorage.getItem(LLM_KEY)).toBe("claude");
    expect(screen.getByText(/Saved/)).toBeTruthy();
  });
});

describe("the Search page uses the saved choice", () => {
  it("sends llm: claude when Claude is chosen", async () => {
    window.localStorage.setItem(LLM_KEY, "claude");
    search.mockResolvedValue({ answer: "", cited: [], llm: "claude", model: null, notice: null, records: [] });
    render(<SearchView />);
    await waitFor(() => expect(window.localStorage.getItem(LLM_KEY)).toBe("claude"));
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "any potholes?" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(search).toHaveBeenCalled());
    expect(search.mock.calls[0]?.[0]).toMatchObject({ llm: "claude" });
  });
});
