// Types mirror satquery.api.schemas / satquery.agent.trace (only the fields the UI reads).

export interface ErrorInfo {
  code: string;
  message: string;
  details: Record<string, unknown>;
  hint: string | null;
}

export interface FinalAnswer {
  text: string;
  answer_type: string;
  value: boolean | number | string | null;
  unit: string | null;
  confidence: number;
  confidence_basis: string;
  facts: Record<string, unknown>;
  is_mock: boolean;
}

export interface Artifact {
  kind: "quicklook" | "overlay" | "mask_png" | "geotiff" | "geojson" | "trace";
  label: string;
  url: string;
  media_type: string;
  layer: string | null;
  step_id: string | null;
}

export interface LayerSummary {
  key: string;
  filename: string;
  modality: string;
  modality_source: string;
  acquired: string | null;
  crs: string | null;
  resolution: [number, number];
}

export interface StepRecord {
  step_id: string;
  tool: string;
  version: string;
  role: string;
  purpose: string;
  status: "ok" | "failed" | "skipped";
  runtime_ms: number;
  confidence: number | null;
  is_mock: boolean;
  error: ErrorInfo | null;
}

export interface Trace {
  request_id: string;
  status: string;
  query: string;
  inputs: { config: string | null; layers: LayerSummary[]; warnings: string[] };
  task: { task: string; targets: string[]; answer_type: string; rationale: string[] } | null;
  plan: { recipe: string; rationale: string[] } | null;
  steps: StepRecord[];
  warnings: string[];
  errors: ErrorInfo[];
  total_runtime_ms: number;
}

export interface AnalyzeResponse {
  request_id?: string;
  status?: string;
  answer?: FinalAnswer | null;
  confidence?: number | null;
  error: ErrorInfo | null;
  artifacts?: Artifact[];
  trace_url?: string;
  trace?: Trace | null;
}

export interface FileOverrides {
  modality: string;
  date: string;
  bandMap: string;
}

export async function analyze(
  files: File[],
  query: string,
  overrides: FileOverrides[],
  benchmark: boolean,
): Promise<AnalyzeResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("query", query);
  overrides.slice(0, files.length).forEach((o, i) => {
    const n = i + 1;
    if (o.modality) form.append(`modality_${n}`, o.modality);
    if (o.date) form.append(`date_${n}`, o.date);
    if (o.bandMap.trim()) form.append(`band_map_${n}`, o.bandMap.trim());
  });
  if (benchmark) form.append("benchmark", "true");
  const res = await fetch("/analyze", { method: "POST", body: form });
  const body = (await res.json()) as AnalyzeResponse;
  if (!body.error && !res.ok) {
    body.error = { code: `http_${res.status}`, message: res.statusText, details: {}, hint: null };
  }
  return body;
}
