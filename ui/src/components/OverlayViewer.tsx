import { useMemo, useState } from "react";
import type { Artifact, LayerSummary } from "../api";

interface Props {
  artifacts: Artifact[];
  layers: LayerSummary[];
}

/** Quicklook per input layer with toggleable, opacity-controlled mask layers on top.
 *  Pairs (optical+SAR, two dates) are shown side by side. */
export default function OverlayViewer({ artifacts, layers }: Props) {
  const [opacity, setOpacity] = useState(0.6);
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const byLayer = useMemo(() => {
    const keys = layers.map((l) => l.key);
    return keys.map((key) => ({
      layer: layers.find((l) => l.key === key)!,
      quicklook: artifacts.find((a) => a.kind === "quicklook" && a.layer === key),
      masks: artifacts.filter((a) => a.kind === "mask_png" && a.layer === key),
    }));
  }, [artifacts, layers]);

  const downloads = artifacts.filter((a) => a.kind === "geotiff" || a.kind === "geojson");
  const toggle = (url: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });

  if (!byLayer.some((l) => l.quicklook)) return null;

  return (
    <section className="card">
      <header className="row">
        <h2>Evidence</h2>
        <label className="inline small">
          Mask opacity
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </label>
      </header>
      <div className={byLayer.length > 1 ? "grid-2" : ""}>
        {byLayer.map(({ layer, quicklook, masks }) => (
          <figure key={layer.key}>
            <div className="stack">
              {quicklook && <img src={quicklook.url} alt={layer.filename} />}
              {masks.map((m) =>
                hidden.has(m.url) ? null : (
                  <img key={m.url} src={m.url} alt={m.label} style={{ opacity }} className="mask" />
                ),
              )}
            </div>
            <figcaption>
              <strong>{layer.key}</strong> · {layer.filename} · {layer.modality}
              {layer.acquired && <> · {layer.acquired}</>}
              {masks.map((m) => (
                <label key={m.url} className="inline small">
                  <input type="checkbox" checked={!hidden.has(m.url)} onChange={() => toggle(m.url)} />
                  {m.label}
                </label>
              ))}
            </figcaption>
          </figure>
        ))}
      </div>
      {downloads.length > 0 && (
        <p className="small">
          Downloads:{" "}
          {downloads.map((d) => (
            <a key={d.url} href={d.url} download className="download">
              {d.label} ({d.kind === "geotiff" ? "GeoTIFF" : "GeoJSON"})
            </a>
          ))}
        </p>
      )}
    </section>
  );
}
