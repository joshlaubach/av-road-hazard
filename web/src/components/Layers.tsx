"use client";

import { useEffect, useRef } from "react";
import type { Trace } from "@/api/types";
import type { HeatGrid } from "@/lib/heat";
import { heatColor } from "@/lib/heat";
import type { Layers } from "@/lib/walkthrough";

type RoadMaskPreview = NonNullable<Trace["road_mask"]>;

/** The road, tinted. The preview PNG is white on road; the canvas turns that into a see-through tint. */
export function RoadLayer({ preview, isVisible }: { preview: RoadMaskPreview; isVisible: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const image = new Image();
    image.onload = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      canvas.width = image.width;
      canvas.height = image.height;
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const isRoad = (pixels.data[i] ?? 0) > 127;
        pixels.data[i] = 43;
        pixels.data[i + 1] = 179;
        pixels.data[i + 2] = 214;
        pixels.data[i + 3] = isRoad ? 100 : 0;
      }
      context.putImageData(pixels, 0, 0);
    };
    image.src = `data:image/png;base64,${preview.png_base64}`;
  }, [preview]);
  return <canvas ref={ref} className="layer" style={{ opacity: isVisible ? 1 : 0 }} aria-hidden="true" />;
}

/** The squares' probabilities as a smooth heatmap: each cell is one pixel, and the browser blends them. */
export function HeatLayer({ grid, isVisible }: { grid: HeatGrid; isVisible: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    canvas.width = grid.cols;
    canvas.height = grid.rows;
    context.clearRect(0, 0, grid.cols, grid.rows);
    grid.values.forEach((p, i) => {
      if (Number.isNaN(p)) return;
      const { color, alpha } = heatColor(p);
      context.globalAlpha = alpha;
      context.fillStyle = color;
      context.fillRect(i % grid.cols, Math.floor(i / grid.cols), 1, 1);
    });
  }, [grid]);
  return <canvas ref={ref} className="layer heat" style={{ opacity: isVisible ? 1 : 0 }} aria-hidden="true" />;
}

/** The grid of squares: outlined, hatched where skipped, bright where flagged. */
export function TileLayer({ trace, layers }: { trace: Trace; layers: Layers }) {
  const [width, height] = trace.image_size;
  const unit = width / 70;
  return (
    <svg
      className="layer tiles"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        {[
          ["hatch-road", "#c9d3dc"],
          ["hatch-object", "#f39a2d"],
        ].map(([id, color]) => (
          <pattern key={id} id={id} width={unit} height={unit} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2={unit} stroke={color} strokeWidth={unit / 5} opacity="0.7" />
          </pattern>
        ))}
      </defs>
      {trace.tiles.map((tile, i) => {
        const skipped = tile.status !== "scored";
        const isFlagged = tile.flagged.length > 0;
        let style = "";
        if (isFlagged && layers.flagged) style = "flagged";
        else if (skipped && layers.skipped) style = tile.status === "skipped_object" ? "skipped-object" : "skipped-road";
        else if (layers.tiles) style = "neutral";
        const [x1, y1, x2, y2] = tile.box;
        return (
          <rect
            key={`${tile.row}-${tile.col}-${i}`}
            x={x1}
            y={y1}
            width={x2 - x1}
            height={y2 - y1}
            className={`tile ${style}`}
            style={{ opacity: style ? 1 : 0, transitionDelay: `${Math.min(i * 10, 700)}ms` }}
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </svg>
  );
}
