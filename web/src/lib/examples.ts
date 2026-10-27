import type { TracedAnalysis } from "@/api/types";

/** One bundled example: the real trace of a full-size photo, made by tools/make_example.py. */
export interface ExampleFile {
  id: string;
  title: string;
  credit: string;
  note: string;
  /** The size the pipeline analyzed (the full-size photo). */
  analyzed_size: [number, number];
  /** The size of the smaller copy the page shows. */
  display_size: [number, number];
  analysis: TracedAnalysis;
}

export interface ExampleCard {
  id: string;
  title: string;
  blurb: string;
}

/** The two examples, in the order they are offered. Both are photos we may show, with credit. */
export const EXAMPLES: readonly ExampleCard[] = [
  {
    id: "stellenbosch-potholes",
    title: "Potholes it finds",
    blurb: "A worn street where the pipeline finds the potholes.",
  },
  {
    id: "stellenbosch-limits",
    title: "Where it gets it wrong",
    blurb: "A pothole it misses, and a false alarm on plain road.",
  },
];

export function exampleImageUrl(id: string): string {
  return `/examples/${id}/display.jpg`;
}

export async function loadExample(id: string): Promise<ExampleFile> {
  const response = await fetch(`/examples/${id}/example.json`);
  if (!response.ok) throw new Error(`could not load example ${id}`);
  return (await response.json()) as ExampleFile;
}
