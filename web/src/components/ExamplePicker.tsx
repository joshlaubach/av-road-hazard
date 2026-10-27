"use client";

import { EXAMPLES, exampleImageUrl } from "@/lib/examples";

interface ExamplePickerProps {
  onPick: (id: string) => void;
  isBusy: boolean;
}

/** The bundled examples as picture buttons. */
export function ExamplePicker({ onPick, isBusy }: ExamplePickerProps) {
  return (
    <div className="examples" role="group" aria-label="Example photos">
      {EXAMPLES.map((example) => (
        <button
          key={example.id}
          type="button"
          className="example"
          disabled={isBusy}
          onClick={() => onPick(example.id)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={exampleImageUrl(example.id)} alt="" width={160} height={76} loading="lazy" />
          <span className="example-title">{example.title}</span>
          <span className="example-blurb">{example.blurb}</span>
        </button>
      ))}
    </div>
  );
}
