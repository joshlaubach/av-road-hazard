"use client";

import { useEffect, useMemo, useState } from "react";
import type { HazardRecord, Trace } from "@/api/types";
import { damageName } from "@/lib/labels";
import { heatGrid } from "@/lib/heat";
import { useReducedMotion } from "@/lib/useReducedMotion";
import { buildSteps } from "@/lib/walkthrough";
import { KatexMath } from "./KatexMath";
import { HeatLayer, RoadLayer, TileLayer } from "./Layers";
import { PhotoView } from "./PhotoView";

interface WalkthroughProps {
  imageUrl: string;
  alt: string;
  record: HazardRecord;
  trace: Trace;
}

const STEP_SECONDS = 4.5;
const HEAT_COLUMNS = 96;
const KINDS = ["D00", "D10", "D20", "D40"];

/**
 * The photo with the pipeline's own trace drawn over it, one step at a time. It opens on the finished
 * result; "Watch how it was found" plays the steps from the beginning. Everything shown comes from
 * the trace of this photo, so it is what the pipeline did.
 */
export function Walkthrough({ imageUrl, alt, record, trace }: WalkthroughProps) {
  const steps = useMemo(() => buildSteps(record, trace), [record, trace]);
  const last = steps.length - 1;
  const [index, setIndex] = useState(last);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showHeat, setShowHeat] = useState(false);
  const [kind, setKind] = useState<string>("");
  const reduced = useReducedMotion();
  const step = steps[index] ?? steps[last];
  const grid = useMemo(
    () => heatGrid(trace.tiles, trace.image_size, HEAT_COLUMNS, kind || null),
    [trace, kind],
  );

  useEffect(() => {
    if (!isPlaying) return;
    if (index >= last) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => setIndex((i) => i + 1), STEP_SECONDS * 1000);
    return () => clearTimeout(timer);
  }, [isPlaying, index, last]);

  if (!step) return null;
  const layers = step.layers;

  function play() {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    setIndex(index >= last ? 0 : index);
    setIsPlaying(true);
  }

  function go(next: number) {
    setIsPlaying(false);
    setIndex(Math.min(last, Math.max(0, next)));
  }

  return (
    <div className="walkthrough">
      <PhotoView
        imageUrl={imageUrl}
        alt={alt}
        record={record}
        showObjects={layers.objects}
        showDefects={layers.defects}
        overlay={
          <>
            {trace.road_mask && <RoadLayer preview={trace.road_mask} isVisible={layers.road} />}
            <HeatLayer grid={grid} isVisible={layers.heat || showHeat} />
            <TileLayer trace={trace} layers={layers} />
          </>
        }
      />

      <div className="transport">
        {!reduced && (
          <button type="button" className="primary" onClick={play}>
            {isPlaying ? "Pause" : index >= last ? "Watch how it was found" : "Play"}
          </button>
        )}
        <button type="button" onClick={() => go(index - 1)} disabled={index === 0}>
          Back
        </button>
        <button type="button" onClick={() => go(index + 1)} disabled={index === last}>
          Next
        </button>
        <label className="switch">
          <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} />
          <span>Heatmap</span>
        </label>
        <select
          aria-label="Heatmap shows"
          className="select"
          value={kind}
          onChange={(e) => setKind(e.target.value)}
          disabled={!(layers.heat || showHeat)}
        >
          <option value="">any damage</option>
          {KINDS.map((code) => (
            <option key={code} value={code}>
              {damageName(code)}
            </option>
          ))}
        </select>
      </div>

      {(layers.heat || showHeat) && (
        <p className="legend" aria-label="Heatmap colors">
          <span className="legend-bar" aria-hidden="true" /> unlikely to likely. Gray hatching is a square I skipped.
        </p>
      )}

      <ol className="steps" aria-label="Steps">
        {steps.map((s, i) => (
          <li key={s.id}>
            <button
              type="button"
              aria-current={i === index ? "step" : undefined}
              className={i < index ? "done" : undefined}
              onClick={() => go(i)}
            >
              {i + 1}. {s.title}
            </button>
          </li>
        ))}
      </ol>

      <div className="step-text" aria-live="polite">
        <h3>
          Step {index + 1} of {steps.length}: {step.title}
        </h3>
        <p>{step.text}</p>
        {step.math.length > 0 && (
          <details>
            <summary>The math for this step</summary>
            <div className="details-body">
              {step.math.map((tex) => (
                <KatexMath key={tex} tex={tex} />
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
