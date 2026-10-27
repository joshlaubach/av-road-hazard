import type { Summary } from "@/lib/explain";

interface FindingsProps {
  summary: Summary;
}

/** The plain-language account of the result. The explanations are the default text on the page. */
export function Findings({ summary }: FindingsProps) {
  return (
    <div className="findings">
      <h2>{summary.headline}</h2>
      {summary.findings.map((finding, i) => (
        <section key={i} className="finding" aria-label={`Spot ${i + 1}`}>
          <h3>{finding.title}</h3>
          <ul>
            {finding.sentences.map((sentence) => (
              <li key={sentence}>{sentence}</li>
            ))}
          </ul>
        </section>
      ))}
      {summary.objectsLine && <p>{summary.objectsLine}</p>}
      <p className="score-line">{summary.scoreLine}</p>
      <p className="caution">{summary.caution}</p>
    </div>
  );
}
