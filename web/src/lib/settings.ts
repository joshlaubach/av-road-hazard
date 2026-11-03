/** Which model writes the answers on the Search page. Kept in the browser only; no accounts. */
export type LlmChoice = "llama" | "claude";

export const LLM_KEY = "road-hazard-llm";
export const DEFAULT_LLM: LlmChoice = "llama";

/** The two choices, in plain words. */
export const LLM_CHOICES: readonly { value: LlmChoice; label: string; detail: string }[] = [
  {
    value: "llama",
    label: "Local model",
    detail:
      "A small model that runs on the same computer as the server. It costs nothing, but it is the weaker writer and sometimes cites a photo it was not given.",
  },
  {
    value: "claude",
    label: "Claude",
    detail:
      "A stronger model from Anthropic. The server needs an API key for it and has a monthly spend limit. If it cannot answer, you still get the matching photos with a note.",
  },
];

/** The part of the browser's storage this page uses, so tests can pass in a fake. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function isLlmChoice(value: unknown): value is LlmChoice {
  return value === "llama" || value === "claude";
}

/** The saved choice, or the default when nothing valid is saved (or storage is blocked). */
export function readLlm(store: KeyValueStore | null): LlmChoice {
  try {
    const saved = store?.getItem(LLM_KEY);
    return isLlmChoice(saved) ? saved : DEFAULT_LLM;
  } catch {
    return DEFAULT_LLM;
  }
}

/** Remember a choice. Returns false when the browser would not store it (private mode, blocked). */
export function saveLlm(store: KeyValueStore | null, choice: LlmChoice): boolean {
  try {
    store?.setItem(LLM_KEY, choice);
    return store !== null;
  } catch {
    return false;
  }
}
