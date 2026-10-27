"use client";

import katex from "katex";
import { useMemo } from "react";

interface KatexMathProps {
  /** TeX built by this app from fixed formulas and numbers (never from user text). */
  tex: string;
  display?: boolean;
}

/** A formula typeset by KaTeX. */
export function KatexMath({ tex, display = true }: KatexMathProps) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: display, throwOnError: false, output: "htmlAndMathml" }),
    [tex, display],
  );
  // KaTeX returns escaped markup for a TeX string we wrote ourselves, so this is safe to insert.
  return <span className="math-block" dangerouslySetInnerHTML={{ __html: html }} />;
}
