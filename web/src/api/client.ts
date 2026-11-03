import createClient from "openapi-fetch";
import type { paths } from "./schema";
import type { Analysis, QueryRequest, QueryResponse } from "./types";

/** The API's upload cap (api.max_upload_bytes in configs/default.yaml), checked here first. */
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

// The browser talks to /api/*, which the web server forwards to the Python API (next.config.mjs).
const client = createClient<paths>({ baseUrl: "/api" });

export class AnalyzeError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "AnalyzeError";
  }
}

/** A reason in plain words for each way a request can fail. */
export function friendlyError(status: number): string {
  switch (status) {
    case 0:
      return "I could not reach the server. Check that the API is running, then try again.";
    case 413:
      return "That file is too big. Try a photo under 10 megabytes.";
    case 415:
      return "I can only read JPEG and PNG photos.";
    case 422:
      return "I could not read that file as a photo. Try a different one.";
    case 429:
      return "That is a lot of requests, or this session already holds as many photos as it may. Wait a moment, then try again.";
    case 503:
      return "The models are not ready on the server yet. Try again in a minute.";
    default:
      return `Something went wrong on the server (code ${status}). Try again.`;
  }
}

/** A message when the file cannot be sent, or null when it is fine. */
export function checkFile(file: Pick<File, "type" | "size">): string | null {
  if (!ALLOWED_TYPES.includes(file.type)) return friendlyError(415);
  if (file.size > MAX_UPLOAD_BYTES) return friendlyError(413);
  return null;
}

/**
 * Send a photo to the API. With the trace, the answer also says what the pipeline did on the way
 * (tiles, probabilities, road mask, fusion steps); without it, only the record.
 */
export async function analyzeImage(file: File, withTrace: boolean): Promise<Analysis> {
  const problem = checkFile(file);
  if (problem) throw new AnalyzeError(problem === friendlyError(413) ? 413 : 415, problem);
  const form = new FormData();
  form.append("file", file);
  const options = { body: { file: "" }, bodySerializer: () => form };
  try {
    if (withTrace) {
      const { data, response } = await client.POST("/analyze/trace", options);
      if (!data) throw new AnalyzeError(response.status, friendlyError(response.status));
      return { record: data.record, trace: data.trace };
    }
    const { data, response } = await client.POST("/analyze", options);
    if (!data) throw new AnalyzeError(response.status, friendlyError(response.status));
    return { record: data, trace: null };
  } catch (error: unknown) {
    if (error instanceof AnalyzeError) throw error;
    throw new AnalyzeError(0, friendlyError(0));
  }
}

/** Send photos to the API so they can be searched. Returns the ids of the photos it stored. */
export async function indexPhotos(files: File[]): Promise<string[]> {
  for (const file of files) {
    const problem = checkFile(file);
    if (problem) throw new AnalyzeError(problem === friendlyError(413) ? 413 : 415, problem);
  }
  const form = new FormData();
  for (const file of files) form.append("files", file);
  try {
    const { data, response } = await client.POST("/index", { body: { files: [] }, bodySerializer: () => form });
    if (!data) throw new AnalyzeError(response.status, friendlyError(response.status));
    return data.indexed;
  } catch (error: unknown) {
    if (error instanceof AnalyzeError) throw error;
    throw new AnalyzeError(0, friendlyError(0));
  }
}

/** Ask a question or search by photo. The answer is written only from the photos it finds. */
export async function searchPhotos(request: QueryRequest): Promise<QueryResponse> {
  try {
    const { data, response } = await client.POST("/query", { body: request });
    if (!data) throw new AnalyzeError(response.status, friendlyError(response.status));
    return data;
  } catch (error: unknown) {
    if (error instanceof AnalyzeError) throw error;
    throw new AnalyzeError(0, friendlyError(0));
  }
}
