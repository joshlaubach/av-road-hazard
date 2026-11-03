"use client";

import { LLM_CHOICES } from "@/lib/settings";
import { useLlmChoice } from "@/lib/useLlmChoice";

/** The Settings view: who writes the answers on the Search page. */
export function SettingsView() {
  const [choice, setChoice, saved] = useLlmChoice();
  return (
    <main className="page">
      <header className="masthead">
        <p className="tag">Early version</p>
        <h1>Settings</h1>
        <p className="lede">
          On the Search page, a language model writes the short answer from the photos it finds. Here you pick which
          one. Your choice is kept in this browser only. Nothing is sent anywhere except the name of the model, along
          with each search.
        </p>
      </header>

      <fieldset className="filters" aria-label="Answer writer">
        <legend>Who writes the answers?</legend>
        {LLM_CHOICES.map((option) => (
          <label key={option.value} className="choice">
            <input
              type="radio"
              name="llm"
              value={option.value}
              checked={choice === option.value}
              onChange={() => setChoice(option.value)}
            />
            <span>
              <strong>{option.label}</strong>
              <span className="hint">{option.detail}</span>
            </span>
          </label>
        ))}
      </fieldset>

      <div aria-live="polite" className="status">
        {saved ? (
          <p>Saved. The Search page will use this choice.</p>
        ) : (
          <p role="alert" className="error">
            Your browser would not keep this choice, so it will reset when you close the page.
          </p>
        )}
      </div>
    </main>
  );
}
