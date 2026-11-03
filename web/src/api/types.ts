import type { components } from "./schema";

// Types come from the generated client (openapi.json is the single contract); nothing here is
// written by hand except the short names.
export type HazardRecord = components["schemas"]["HazardRecord"];
export type DefectRecord = components["schemas"]["DefectRecord"];
export type ObjectRecord = components["schemas"]["Object"];
export type Trace = components["schemas"]["Trace"];
export type TileTrace = components["schemas"]["TileTrace"];
export type FusionStep = components["schemas"]["FusionStep"];
export type TracedAnalysis = components["schemas"]["TracedAnalysis"];
export type Severity = DefectRecord["severity"];

/** What the page holds after an analysis: the record, and the trace when it was asked for. */
export interface Analysis {
  record: HazardRecord;
  trace: Trace | null;
}

export type QueryRequest = components["schemas"]["QueryRequest"];
export type QueryFilters = components["schemas"]["QueryFilters"];
export type QueryResponse = components["schemas"]["QueryResponse"];
