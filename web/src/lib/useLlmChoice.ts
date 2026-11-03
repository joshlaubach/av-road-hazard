"use client";

import { useEffect, useState } from "react";
import { DEFAULT_LLM, readLlm, saveLlm, type LlmChoice } from "./settings";

function browserStore(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null; // storage can be blocked entirely
  }
}

/**
 * The saved answer-writer choice. It starts at the default so the server and the browser draw the
 * same first page, then loads the saved value. `saved` is false when the browser would not keep it.
 */
export function useLlmChoice(): [LlmChoice, (choice: LlmChoice) => void, boolean] {
  const [choice, setChoice] = useState<LlmChoice>(DEFAULT_LLM);
  const [saved, setSaved] = useState(true);

  useEffect(() => {
    setChoice(readLlm(browserStore()));
  }, []);

  function update(next: LlmChoice) {
    setChoice(next);
    setSaved(saveLlm(browserStore(), next));
  }

  return [choice, update, saved];
}
