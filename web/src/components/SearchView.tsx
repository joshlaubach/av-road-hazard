"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { AnalyzeError, checkFile, indexPhotos, searchPhotos } from "@/api/client";
import type { QueryResponse } from "@/api/types";
import { damageName, objectName } from "@/lib/labels";
import {
  DAMAGE_CHOICES,
  EMPTY_FORM,
  OBJECT_CHOICES,
  buildQuery,
  numberCitations,
  stripDataUrl,
  type SearchForm,
} from "@/lib/search";
import { useLlmChoice } from "@/lib/useLlmChoice";
import { ResultCard } from "./ResultCard";

type Status = { kind: "idle" } | { kind: "busy"; label: string } | { kind: "error"; message: string };

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** The Search view: add photos, ask a question or search by photo, and read an answer that cites them. */
export function SearchView() {
  const [llm] = useLlmChoice();
  const [form, setForm] = useState<SearchForm>(EMPTY_FORM);
  const [queryPhotoName, setQueryPhotoName] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [added, setAdded] = useState(0);
  const [result, setResult] = useState<QueryResponse | null>(null);
  // Previews exist only for photos this visitor added in this session; the API never returns photos.
  const [thumbnails, setThumbnails] = useState<Record<string, string>>({});
  const urls = useRef<string[]>([]);

  useEffect(() => {
    const kept = urls.current; // the same array for the whole life of the page
    return () => kept.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  function fail(error: unknown) {
    const message = error instanceof AnalyzeError ? error.message : "Something went wrong. Try again.";
    setStatus({ kind: "error", message });
  }

  async function handleAdd(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length === 0) return;
    const problem = files.map(checkFile).find((p) => p !== null);
    if (problem) {
      setStatus({ kind: "error", message: problem });
      return;
    }
    const plural = files.length === 1 ? "" : "s";
    setStatus({ kind: "busy", label: `Looking at ${files.length} photo${plural}. This takes a few seconds each.` });
    try {
      const ids = await indexPhotos(files);
      const fresh: Record<string, string> = {};
      ids.forEach((id, i) => {
        const file = files[i];
        if (!file) return;
        const url = URL.createObjectURL(file);
        urls.current.push(url);
        fresh[id] = url;
      });
      setThumbnails((old) => ({ ...old, ...fresh }));
      setAdded((n) => n + ids.length);
      setStatus({ kind: "idle" });
    } catch (error: unknown) {
      fail(error);
    }
  }

  async function handleQueryPhoto(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const problem = checkFile(file);
    if (problem) {
      setStatus({ kind: "error", message: problem });
      return;
    }
    const dataUrl = await readAsDataUrl(file);
    setForm((f) => ({ ...f, imageB64: stripDataUrl(dataUrl) }));
    setQueryPhotoName(file.name);
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    const request = buildQuery(form, llm);
    if (!request) {
      setStatus({ kind: "error", message: "Type a question or choose a photo to search with." });
      return;
    }
    setStatus({ kind: "busy", label: "Searching." });
    try {
      setResult(await searchPhotos(request));
      setStatus({ kind: "idle" });
    } catch (error: unknown) {
      fail(error);
    }
  }

  const busy = status.kind === "busy";
  return (
    <main className="page">
      <header className="masthead">
        <p className="tag">Early version</p>
        <h1>Search the road photos</h1>
        <p className="lede">
          Ask a question like &ldquo;which photos have a pothole near a person?&rdquo; or search with a photo. I look
          through the photos that were analyzed, and I write the answer only from what the analysis found in them,
          not from looking at the pictures again. Every photo I mention is numbered below the answer.
        </p>
      </header>

      <section className="intake" aria-label="Add your photos">
        <h2>1. Add your photos</h2>
        <p className="hint">
          Photos you add are kept for this visit only, and only you can search them. Each one takes a few seconds to
          analyze.{added > 0 ? ` You have added ${added} so far.` : ""}
        </p>
        <label className="button">
          <input type="file" accept="image/jpeg,image/png" multiple onChange={handleAdd} disabled={busy} />
          Choose photos
        </label>
      </section>

      <form className="intake" onSubmit={handleSearch} aria-label="Search">
        <h2>2. Ask or search</h2>
        <label className="field">
          <span>Your question</span>
          <input
            type="text"
            maxLength={500}
            value={form.text}
            onChange={(e) => setForm({ ...form, text: e.target.value })}
            placeholder="Which photos show a pothole?"
          />
        </label>
        <label className="field">
          <span>Or search with a photo{queryPhotoName ? ` (chosen: ${queryPhotoName})` : ""}</span>
          <input type="file" accept="image/jpeg,image/png" onChange={handleQueryPhoto} disabled={busy} />
        </label>
        <fieldset className="filters">
          <legend>Only show photos where&hellip; (optional)</legend>
          <label className="field">
            <span>there is this kind of damage</span>
            <select value={form.damage} onChange={(e) => setForm({ ...form, damage: e.target.value })}>
              <option value="">any</option>
              {DAMAGE_CHOICES.map((code) => (
                <option key={code} value={code}>
                  {damageName(code)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>I can see this</span>
            <select value={form.object} onChange={(e) => setForm({ ...form, object: e.target.value })}>
              <option value="">anything</option>
              {OBJECT_CHOICES.map((label) => (
                <option key={label} value={label}>
                  {objectName(label)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>the score is at least {form.minScore} out of 100</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={form.minScore}
              onChange={(e) => setForm({ ...form, minScore: Number(e.target.value) })}
            />
          </label>
        </fieldset>
        <button type="submit" className="button" disabled={busy}>
          Search
        </button>
      </form>

      <div aria-live="polite" className="status">
        {status.kind === "busy" && <p>{status.label}</p>}
        {status.kind === "error" && (
          <p role="alert" className="error">
            {status.message}
          </p>
        )}
      </div>

      {result && (
        <section className="result" aria-label="Search results">
          {result.answer && (
            <div className="answer">
              <h2>Answer</h2>
              {result.notice ? (
                // The server sends the same sentence as the answer and as the notice when no model wrote one.
                <p className="caution">{result.notice}</p>
              ) : (
                <>
                  <p>{numberCitations(result.answer, result.records)}</p>
                  <p className="hint">
                    Written by {result.model ?? result.llm}, only from what the analysis found. Check the numbered
                    photos before you rely on it. You can change the writer on the <Link href="/settings">Settings page</Link>.
                  </p>
                </>
              )}
            </div>
          )}
          {result.records.length === 0 ? (
            <p>No photos matched. Try different words, remove a filter, or add more photos.</p>
          ) : (
            <>
              <h2>{result.answer ? "Photos it used" : "Most similar photos"}</h2>
              <ol className="cards">
                {result.records.map((record, i) => (
                  <ResultCard
                    key={record.frame_id}
                    number={i + 1}
                    record={record}
                    thumbnailUrl={thumbnails[record.frame_id] ?? null}
                  />
                ))}
              </ol>
            </>
          )}
        </section>
      )}
    </main>
  );
}
