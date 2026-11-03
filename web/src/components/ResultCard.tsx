import type { HazardRecord } from "@/api/types";
import { cardFacts } from "@/lib/search";

interface ResultCardProps {
  number: number;
  record: HazardRecord;
  /** A preview, only for photos this visitor uploaded in this session. */
  thumbnailUrl: string | null;
}

/** One search result: the photo's number, its score and zone, and what the pipeline found in it. */
export function ResultCard({ number, record, thumbnailUrl }: ResultCardProps) {
  const facts = cardFacts(record);
  return (
    <li className="card" id={`result-${number}`}>
      {thumbnailUrl ? (
        // eslint-disable-next-line @next/next/no-img-element -- a local preview of the visitor's own file
        <img className="card-thumb" src={thumbnailUrl} alt={`Your photo, result ${number}`} />
      ) : (
        <div className="card-thumb card-nothumb" aria-hidden="true">
          No preview
        </div>
      )}
      <div className="card-body">
        <h3>
          Result {number}: {facts.score} out of 100, {facts.zone.toLowerCase()}
        </h3>
        <p>{facts.damages.length ? `Damage found: ${facts.damages.join(", ")}.` : "No road damage found."}</p>
        <p>{facts.objects.length ? `Seen nearby: ${facts.objects.join(", ")}.` : "No people or vehicles found."}</p>
      </div>
    </li>
  );
}
