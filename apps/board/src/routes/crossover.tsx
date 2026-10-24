export function CrossoverPage() {
  return (
    <section>
      <header className="page-head">
        <h1>Self-host vs API</h1>
      </header>
      <div className="card stub">
        <p>
          <strong>Phase 5 continuation.</strong> This view is routed but not built yet.
        </p>
        <p className="muted">
          Planned (per docs/plan/02-architecture.md §Board): cost per 1k items against volume, and
          the break-even point where a local deployment on a machine in{" "}
          <code>catalog/machines/</code> becomes cheaper than the hosted providers — using measured
          run cost and the amortised machine rates.
        </p>
      </div>
    </section>
  );
}
