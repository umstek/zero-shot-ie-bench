export function ProvidersPage() {
  return (
    <section>
      <header className="page-head">
        <h1>Provider picker</h1>
      </header>
      <div className="card stub">
        <p>
          <strong>Phase 5 continuation.</strong> This view is routed but not built yet.
        </p>
        <p className="muted">
          Planned (per docs/plan/02-architecture.md §Board): the same model across its providers —
          price, ZDR status, latency and stability side by side. The data is already in{" "}
          <code>deployments.json</code> (one row per deployment); this page groups it by{" "}
          <code>model_id</code>.
        </p>
      </div>
    </section>
  );
}
