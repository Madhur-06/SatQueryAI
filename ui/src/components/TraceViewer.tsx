import type { Trace } from "../api";

export default function TraceViewer({ trace, url }: { trace: Trace; url?: string }) {
  return (
    <section className="card">
      <header className="row">
        <h2>Execution trace</h2>
        {url && (
          <a href={url} download={`trace_${trace.request_id}.json`} className="small">
            download JSON
          </a>
        )}
      </header>

      {trace.steps.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Tool</th>
              <th>Role</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>ms</th>
            </tr>
          </thead>
          <tbody>
            {trace.steps.map((s) => (
              <tr key={s.step_id} title={s.error?.message ?? s.purpose}>
                <td>{s.purpose}</td>
                <td>
                  <code>
                    {s.tool} v{s.version}
                  </code>
                  {s.is_mock && <span className="badge mock">MOCK</span>}
                </td>
                <td>{s.role}</td>
                <td className={`status-${s.status}`}>{s.status}</td>
                <td>{s.confidence == null ? "–" : s.confidence.toFixed(2)}</td>
                <td>{s.runtime_ms.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {trace.plan && trace.plan.rationale.length > 0 && (
        <details>
          <summary>Planner rationale</summary>
          <ul className="small">
            {trace.plan.rationale.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </details>
      )}

      <details>
        <summary>Full trace JSON</summary>
        <pre className="trace">{JSON.stringify(trace, null, 2)}</pre>
      </details>
    </section>
  );
}
