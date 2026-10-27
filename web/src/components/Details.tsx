import type { Analysis } from "@/api/types";
import { RANGE_BANDS, percentFound } from "@/lib/limits";
import { damagePlain } from "@/lib/labels";
import { HAZARD_TEX, PROXIMITY_TEX, SCORE_TEX, hazardTex, stepTex } from "@/lib/math";
import { KatexMath } from "./KatexMath";

interface DetailsProps {
  analysis: Analysis;
  /** Whether to plug this photo's own numbers into the formulas. */
  showNumbers: boolean;
}

/** The mathematics, the limits and the raw record, each behind its own disclosure. */
export function Details({ analysis, showNumbers }: DetailsProps) {
  const { record, trace } = analysis;
  const steps = showNumbers && trace ? trace.fusion : [];
  return (
    <div className="details-stack">
      <details>
        <summary>How the score is made (the math)</summary>
        <div className="details-body">
          <p>
            Each spot of damage <em>k</em> gets a score <em>s</em>
            <sub>k</sub> between 0 and 1. Four numbers multiply into it: <em>m</em> is 1 if the spot is on the
            road (and 0 if it is confirmed off it), <em>w</em> is how serious that kind of damage is (0.3, 0.6
            or 1.0), <em>c</em> is how confident the model is, and <em>π</em> says how close a person or
            vehicle is to the spot.
          </p>
          <KatexMath tex={SCORE_TEX} />
          <p>
            The nearness <em>π</em> is the largest of the object scores <em>r</em>. An object <em>j</em> counts
            more if it is vulnerable (<em>v</em>: a person is 1.0, a car 0.3), if the detector is sure of it
            (<em>c</em>), and the closer it is: <em>d</em> is the distance in object heights.
          </p>
          <KatexMath tex={PROXIMITY_TEX} />
          <p>
            The frame&rsquo;s score <em>H</em> is the worst spot, capped at 1, and the page shows it times 100.
          </p>
          <KatexMath tex={HAZARD_TEX} />
          {steps.length > 0 && (
            <>
              <h4>Your photo&rsquo;s numbers</h4>
              {steps.map((step, i) => (
                <div key={i}>
                  <p>
                    Spot {i + 1}, {damagePlain(step.name)}
                    {step.merged_from > 1 ? ` (${step.merged_from} squares merged)` : ""}:
                  </p>
                  <KatexMath tex={stepTex(step)} />
                </div>
              ))}
              <KatexMath tex={hazardTex(steps, record.hazard_score)} />
            </>
          )}
          <p className="note">
            The weights are first guesses, chosen by judgment and not measured. The score ranks how urgent a
            spot looks. It is not a probability of a crash.
          </p>
        </div>
      </details>

      <details>
        <summary>What this tool cannot do</summary>
        <div className="details-body">
          <p>
            It finds damage that looks big in the photo, and mostly misses damage that looks small. In a test
            that pasted real damage patches onto driving photos, it found:
          </p>
          <ul>
            {RANGE_BANDS.map((band) => (
              <li key={band.band}>
                <strong>{percentFound(band)}%</strong> of patches {band.widthText} ({band.distanceText} for a
                half-meter pothole)
              </li>
            ))}
          </ul>
          <p className="note">
            Pasted patches are easier than real damage, so real roads are harder. A road photo with no boxes
            does not mean the road is fine.
          </p>
        </div>
      </details>

      <details>
        <summary>The raw record (JSON)</summary>
        <div className="details-body">
          <pre>{JSON.stringify(record, null, 2)}</pre>
        </div>
      </details>
    </div>
  );
}
