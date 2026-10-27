"use client";

import { useEffect, useRef, useState } from "react";
import { AnalyzeError, analyzeImage, checkFile } from "@/api/client";
import type { Analysis } from "@/api/types";
import { summarize } from "@/lib/explain";
import { exampleImageUrl, loadExample, type ExampleFile } from "@/lib/examples";
import { toDisplayScore, scoreBand } from "@/lib/gauge";
import { Details } from "./Details";
import { ExamplePicker } from "./ExamplePicker";
import { Findings } from "./Findings";
import { Gauge } from "./Gauge";
import { PhotoView } from "./PhotoView";
import { Uploader } from "./Uploader";
import { Walkthrough } from "./Walkthrough";

type View =
  | { kind: "idle" }
  | { kind: "loading"; label: string }
  | { kind: "error"; message: string }
  | { kind: "done"; analysis: Analysis; imageUrl: string; alt: string; example: ExampleFile | null };

/** The Analyze view: choose a photo, see what the pipeline found, and why, in plain language. */
export function AnalyzeView() {
  const [view, setView] = useState<View>({ kind: "idle" });
  const [showObjects, setShowObjects] = useState(true);
  const [showDefects, setShowDefects] = useState(true);
  const [showHow, setShowHow] = useState(true);
  const objectUrl = useRef<string | null>(null);

  // Release the preview of an uploaded photo when it is replaced and when the page goes away.
  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  function replaceObjectUrl(next: string | null) {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = next;
  }

  async function handleFile(file: File) {
    const problem = checkFile(file);
    if (problem) {
      setView({ kind: "error", message: problem });
      return;
    }
    setView({ kind: "loading", label: "Looking at your photo. This can take about ten seconds." });
    try {
      const analysis = await analyzeImage(file, true);
      const url = URL.createObjectURL(file);
      replaceObjectUrl(url);
      setView({ kind: "done", analysis, imageUrl: url, alt: "The photo you chose", example: null });
    } catch (error: unknown) {
      const message = error instanceof AnalyzeError ? error.message : "Something went wrong. Try again.";
      setView({ kind: "error", message });
    }
  }

  async function handleExample(id: string) {
    setView({ kind: "loading", label: "Loading the example." });
    try {
      const example = await loadExample(id);
      replaceObjectUrl(null);
      setView({
        kind: "done",
        analysis: { record: example.analysis.record, trace: example.analysis.trace },
        imageUrl: exampleImageUrl(id),
        alt: example.title,
        example,
      });
    } catch {
      setView({ kind: "error", message: "I could not load that example. Try again." });
    }
  }

  const isBusy = view.kind === "loading";
  const done = view.kind === "done" ? view : null;
  const summary = done
    ? summarize(done.analysis.record, showHow && done.analysis.trace ? done.analysis.trace.fusion : [])
    : null;
  const score = done ? toDisplayScore(done.analysis.record.hazard_score) : 0;
  const trace = done?.analysis.trace ?? null;
  const walkthrough = Boolean(showHow && trace);

  return (
    <main className="page">
      <header className="masthead">
        <p className="tag">Early version</p>
        <h1>Where is this road damaged?</h1>
        <p className="lede">
          Give me a photo from a car&rsquo;s front camera. I look for people and vehicles, then for cracks and
          potholes on the road, and I tell you how urgent the worst spot looks, on a scale of 0 to 100.
        </p>
      </header>

      <section className="intake" aria-label="Choose a photo">
        <Uploader onFile={handleFile} isBusy={isBusy} />
        <p className="or">or try an example</p>
        <ExamplePicker onPick={handleExample} isBusy={isBusy} />
        <label className="switch">
          <input type="checkbox" checked={showHow} onChange={(e) => setShowHow(e.target.checked)} />
          <span>Show how it was found</span>
        </label>
      </section>

      <div aria-live="polite" className="status">
        {view.kind === "loading" && <p>{view.label}</p>}
        {view.kind === "error" && (
          <p role="alert" className="error">
            {view.message}
          </p>
        )}
      </div>

      {done && summary && (
        <>
          <section className="result" aria-label="Result">
            <div className="stage">
              {walkthrough && trace ? (
                <Walkthrough
                  imageUrl={done.imageUrl}
                  alt={done.alt}
                  record={done.analysis.record}
                  trace={trace}
                />
              ) : (
                <>
                  <PhotoView
                    imageUrl={done.imageUrl}
                    alt={done.alt}
                    record={done.analysis.record}
                    showObjects={showObjects}
                    showDefects={showDefects}
                  />
                  <div className="layers">
                    <label className="switch">
                      <input type="checkbox" checked={showDefects} onChange={(e) => setShowDefects(e.target.checked)} />
                      <span>Damage</span>
                    </label>
                    <label className="switch">
                      <input type="checkbox" checked={showObjects} onChange={(e) => setShowObjects(e.target.checked)} />
                      <span>People and vehicles</span>
                    </label>
                  </div>
                </>
              )}
              {done.example && (
                <p className="credit">
                  <strong>{done.example.title}.</strong> {done.example.note} {done.example.credit}
                </p>
              )}
            </div>

            <div className="panel">
              <div className="meter">
                <Gauge score={score} />
                <p className="meter-number">
                  <strong>{score}</strong>
                  <span> out of 100</span>
                </p>
                <p className="meter-band">{scoreBand(score).label}</p>
              </div>
              <Findings summary={summary} />
            </div>
          </section>

          <Details analysis={done.analysis} showNumbers={showHow} />
        </>
      )}

      <footer className="foot">
        <p>
          Built from open data and models, and not a safety system. It finds large, near damage and mostly
          misses small, faraway damage. Example photos: Stellenbosch University, 2015, public domain (CC0) per
          the Kaggle and Dataset Ninja listings.
        </p>
      </footer>
    </main>
  );
}
