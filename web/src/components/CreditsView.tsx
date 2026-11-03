import corpus from "@/lib/demo_corpus.json";

/** Who made the demo photos, under what terms, and what is assumed. Driven by demo_corpus.json. */
export function CreditsView() {
  const sources = corpus.sources.filter((source) => source.include);
  return (
    <main>
      <h1>Credits and licenses</h1>
      <p>{corpus.notice}</p>
      <ul>
        {sources.map((source) => (
          <li key={source.id}>
            <p>
              <strong>{source.title}.</strong> {source.authors}. {source.images} photos
              ({source.countries.join(", ")}). {source.url && <a href={source.url}>Source</a>}
            </p>
            <p>{source.license_note}</p>
          </li>
        ))}
      </ul>
    </main>
  );
}
