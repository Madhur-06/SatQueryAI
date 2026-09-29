import type { AnalyzeResponse } from "../api";

export default function AnswerCard({ result }: { result: AnalyzeResponse }) {
  const { answer, error, trace } = result;
  const warnings = trace?.warnings ?? [];

  return (
    <section className="card">
      <header className="row">
        <h2>Answer</h2>
        {result.status && <span className={`badge status-${result.status}`}>{result.status}</span>}
        {answer?.is_mock && <span className="badge mock">MOCK</span>}
      </header>

      {answer && (
        <>
          <p className="answer">{answer.text}</p>
          <div className="confidence">
            <span>Confidence {(answer.confidence * 100).toFixed(0)}%</span>
            <meter min={0} max={1} low={0.4} high={0.75} optimum={1} value={answer.confidence} />
          </div>
          <p className="muted small">{answer.confidence_basis}</p>
        </>
      )}

      {error && (
        <div className="error-box">
          <strong>{error.code}</strong>: {error.message}
          {error.hint && <p className="hint">Hint: {error.hint}</p>}
          {Object.keys(error.details ?? {}).length > 0 && (
            <pre className="small">{JSON.stringify(error.details, null, 2)}</pre>
          )}
        </div>
      )}

      {trace?.task && (
        <p className="muted small">
          Task <code>{trace.task.task}</code>
          {trace.task.targets.length > 0 && <> · targets {trace.task.targets.join(", ")}</>} ·
          input <code>{trace.inputs.config ?? "?"}</code>
          {trace.plan && (
            <>
              {" "}
              · recipe <code>{trace.plan.recipe}</code>
            </>
          )}{" "}
          · {trace.total_runtime_ms.toFixed(0)} ms
        </p>
      )}

      {warnings.length > 0 && (
        <ul className="warnings">
          {warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
