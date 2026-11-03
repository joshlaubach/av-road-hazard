import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import corpus from "@/lib/demo_corpus.json";
import { CreditsView } from "./CreditsView";

afterEach(cleanup);

describe("CreditsView", () => {
  it("credits every included source and states the demo is non-commercial", () => {
    render(<CreditsView />);
    for (const source of corpus.sources.filter((s) => s.include)) {
      expect(screen.getByText(source.title + ".")).toBeTruthy();
    }
    expect(screen.getByText(/non-commercial/)).toBeTruthy();
  });

  it("drops a source whose include flag is off, with no code change", () => {
    const first = corpus.sources[0];
    if (!first) throw new Error("the corpus file lists no sources");
    first.include = false;
    try {
      render(<CreditsView />);
      expect(screen.queryByText(first.title + ".")).toBeNull();
    } finally {
      first.include = true;
    }
  });
});
