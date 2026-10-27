import type { HazardRecord } from "@/api/types";
import { SEVERITY_COLOR, damageName, objectName } from "@/lib/labels";

interface PhotoViewProps {
  imageUrl: string;
  alt: string;
  record: HazardRecord;
  showObjects: boolean;
  showDefects: boolean;
}

function percent(x: number, whole: number): string {
  return `${(100 * x) / whole}%`;
}

/** The photo with the objects and damage boxes drawn over it, in the photo's own coordinates. */
export function PhotoView({ imageUrl, alt, record, showObjects, showDefects }: PhotoViewProps) {
  const [width, height] = record.image_size;
  return (
    <figure className="photo">
      <div className="photo-frame" style={{ aspectRatio: `${width} / ${height}` }}>
        {/* A plain img: the source is the visitor's own file or a bundled example. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={imageUrl} alt={alt} />
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          {showObjects &&
            record.objects.map((o, i) => (
              <rect
                key={`o${i}`}
                x={o.box[0]}
                y={o.box[1]}
                width={o.box[2] - o.box[0]}
                height={o.box[3] - o.box[1]}
                className="box-object"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          {showDefects &&
            record.defects.map((d, i) => (
              <rect
                key={`d${i}`}
                x={d.box[0]}
                y={d.box[1]}
                width={d.box[2] - d.box[0]}
                height={d.box[3] - d.box[1]}
                className="box-defect"
                style={{ stroke: SEVERITY_COLOR[d.severity] }}
                vectorEffect="non-scaling-stroke"
              />
            ))}
        </svg>
        {showObjects &&
          record.objects.map((o, i) => (
            <span
              key={`ol${i}`}
              className="chip chip-object"
              style={{ left: percent(o.box[0], width), top: percent(o.box[1], height) }}
            >
              {objectName(o.label)}
            </span>
          ))}
        {showDefects &&
          record.defects.map((d, i) => (
            <span
              key={`dl${i}`}
              className="chip chip-defect"
              style={{
                left: percent(d.box[0], width),
                top: percent(d.box[1], height),
                background: SEVERITY_COLOR[d.severity],
              }}
            >
              {damageName(d.name)}
            </span>
          ))}
      </div>
    </figure>
  );
}
