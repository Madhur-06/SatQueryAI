import { useState } from "react";
import type { FileOverrides } from "../api";

const EXAMPLES = [
  "How much area is covered by water?",
  "Is there any water in this image?",
  "Map the water using optical and SAR",
  "Has built-up area increased?",
  "Describe this image",
];

const EMPTY: FileOverrides = { modality: "", date: "", bandMap: "" };

interface Props {
  busy: boolean;
  onSubmit: (files: File[], query: string, overrides: FileOverrides[], benchmark: boolean) => void;
}

export default function UploadForm({ busy, onSubmit }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [overrides, setOverrides] = useState<FileOverrides[]>([{ ...EMPTY }, { ...EMPTY }]);
  const [benchmark, setBenchmark] = useState(false);
  const tooMany = files.length > 2;

  const setOverride = (i: number, patch: Partial<FileOverrides>) =>
    setOverrides((prev) => prev.map((o, j) => (j === i ? { ...o, ...patch } : o)));

  return (
    <form
      className="card"
      onSubmit={(e) => {
        e.preventDefault();
        if (files.length && !tooMany) onSubmit(files, query, overrides, benchmark);
      }}
    >
      <label>
        Images (1 or 2 GeoTIFFs: optical, SAR, an optical+SAR pair or two dates)
        <input
          type="file"
          multiple
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
      </label>
      {tooMany && <p className="error">Select at most two files.</p>}

      <label>
        Question
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about the image(s)"
        />
      </label>
      <div className="examples">
        {EXAMPLES.map((q) => (
          <button type="button" key={q} className="chip" onClick={() => setQuery(q)}>
            {q}
          </button>
        ))}
      </div>

      <details>
        <summary>Advanced: overrides</summary>
        {files.slice(0, 2).map((f, i) => (
          <fieldset key={f.name + i}>
            <legend>{f.name}</legend>
            <label>
              Modality
              <select
                value={overrides[i].modality}
                onChange={(e) => setOverride(i, { modality: e.target.value })}
              >
                <option value="">auto-detect</option>
                <option value="optical">optical</option>
                <option value="multispectral">multispectral</option>
                <option value="sar">sar</option>
              </select>
            </label>
            <label>
              Acquisition date
              <input
                type="date"
                value={overrides[i].date}
                onChange={(e) => setOverride(i, { date: e.target.value })}
              />
            </label>
            <label>
              Band map (JSON, 1-based)
              <input
                type="text"
                value={overrides[i].bandMap}
                placeholder='{"green": 2, "nir": 4}'
                onChange={(e) => setOverride(i, { bandMap: e.target.value })}
              />
            </label>
          </fieldset>
        ))}
        {files.length === 0 && <p className="muted">Select files to set per-file overrides.</p>}
        <label className="inline">
          <input
            type="checkbox"
            checked={benchmark}
            onChange={(e) => setBenchmark(e.target.checked)}
          />
          Benchmark image (allow PNG/JPEG)
        </label>
      </details>

      <button type="submit" disabled={busy || files.length === 0 || tooMany || !query.trim()}>
        {busy ? "Analysing…" : "Analyse"}
      </button>
    </form>
  );
}
