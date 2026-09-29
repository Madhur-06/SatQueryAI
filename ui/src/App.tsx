import { useState } from "react";
import { analyze, type AnalyzeResponse, type FileOverrides } from "./api";
import AnswerCard from "./components/AnswerCard";
import OverlayViewer from "./components/OverlayViewer";
import TraceViewer from "./components/TraceViewer";
import UploadForm from "./components/UploadForm";

export default function App() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const submit = async (
    files: File[],
    query: string,
    overrides: FileOverrides[],
    benchmark: boolean,
  ) => {
    setBusy(true);
    try {
      setResult(await analyze(files, query, overrides, benchmark));
    } catch (err) {
      setResult({
        error: { code: "network_error", message: String(err), details: {}, hint: "Is the API running?" },
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <main>
      <h1>SatQuery AI</h1>
      <p className="muted">
        Upload a satellite image (or a pair), ask a question, and get a grounded answer with
        evidence and a full execution trace.
      </p>
      <UploadForm busy={busy} onSubmit={submit} />
      {result && <AnswerCard result={result} />}
      {result?.artifacts && result.trace && (
        <OverlayViewer artifacts={result.artifacts} layers={result.trace.inputs.layers} />
      )}
      {result?.trace && <TraceViewer trace={result.trace} url={result.trace_url} />}
    </main>
  );
}
