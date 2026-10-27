"use client";

import { useEffect, useState } from "react";
import { arcSegments, gaugeColor, scoreBand } from "@/lib/gauge";

interface GaugeProps {
  /** The score people see, a whole number from 0 to 100. */
  score: number;
}

const CX = 160;
const CY = 150;
const OUTER = 140;
const INNER = 96;
const SEGMENTS = arcSegments(60, CX, CY, OUTER, INNER);

/** A speedometer-style half circle: 0 at the left end, 100 at the right, blue through red. */
export function Gauge({ score }: GaugeProps) {
  // The needle starts at 0 and swings to the score once the page is showing, so the move is seen.
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setShown(score));
    return () => cancelAnimationFrame(frame);
  }, [score]);

  const band = scoreBand(score);
  return (
    <svg
      className="gauge"
      viewBox="0 0 320 180"
      role="img"
      aria-label={`Hazard score ${score} out of 100: ${band.label}`}
    >
      {SEGMENTS.map((segment, i) => (
        <path key={i} d={segment.path} fill={segment.color} stroke={segment.color} strokeWidth="0.6" />
      ))}
      <text x={CX - (OUTER + INNER) / 2} y={CY + 22} textAnchor="middle" className="gauge-end">
        0
      </text>
      <text x={CX + (OUTER + INNER) / 2} y={CY + 22} textAnchor="middle" className="gauge-end">
        100
      </text>
      <g
        className="needle"
        style={{ transform: `rotate(${(180 * shown) / 100}deg)`, transformOrigin: `${CX}px ${CY}px` }}
      >
        <path d={`M ${CX - OUTER + 8} ${CY} L ${CX} ${CY - 5} L ${CX} ${CY + 5} Z`} fill="var(--fg)" />
      </g>
      <circle cx={CX} cy={CY} r="9" fill="var(--fg)" />
      <circle cx={CX} cy={CY} r="3.5" fill={gaugeColor(score)} />
    </svg>
  );
}
