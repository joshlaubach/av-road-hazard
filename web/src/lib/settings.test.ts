import { describe, expect, it } from "vitest";
import { DEFAULT_LLM, LLM_KEY, isLlmChoice, readLlm, saveLlm, type KeyValueStore } from "./settings";

function fakeStore(initial: Record<string, string> = {}): KeyValueStore & { data: Record<string, string> } {
  const data = { ...initial };
  return {
    data,
    getItem: (key) => data[key] ?? null,
    setItem: (key, value) => {
      data[key] = value;
    },
  };
}

const blocked: KeyValueStore = {
  getItem: () => {
    throw new Error("blocked");
  },
  setItem: () => {
    throw new Error("blocked");
  },
};

describe("isLlmChoice", () => {
  it("accepts only the two models the API knows", () => {
    expect(isLlmChoice("llama")).toBe(true);
    expect(isLlmChoice("claude")).toBe(true);
    expect(isLlmChoice("gpt")).toBe(false);
    expect(isLlmChoice(null)).toBe(false);
  });
});

describe("readLlm", () => {
  it("returns the default when nothing is saved", () => {
    expect(readLlm(fakeStore())).toBe(DEFAULT_LLM);
    expect(DEFAULT_LLM).toBe("llama");
  });

  it("returns a saved choice", () => {
    expect(readLlm(fakeStore({ [LLM_KEY]: "claude" }))).toBe("claude");
  });

  it("ignores a saved value that is not one of the choices", () => {
    expect(readLlm(fakeStore({ [LLM_KEY]: "something-else" }))).toBe(DEFAULT_LLM);
  });

  it("falls back to the default when storage is missing or blocked", () => {
    expect(readLlm(null)).toBe(DEFAULT_LLM);
    expect(readLlm(blocked)).toBe(DEFAULT_LLM);
  });
});

describe("saveLlm", () => {
  it("stores the choice and says so", () => {
    const store = fakeStore();
    expect(saveLlm(store, "claude")).toBe(true);
    expect(store.data[LLM_KEY]).toBe("claude");
  });

  it("says false, and does not throw, when the browser would not store it", () => {
    expect(saveLlm(blocked, "claude")).toBe(false);
    expect(saveLlm(null, "claude")).toBe(false);
  });
});
