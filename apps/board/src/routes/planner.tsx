export function PlannerPage() {
  return (
    <section>
      <header className="page-head">
        <h1>GPU planner</h1>
      </header>
      <div className="card stub">
        <p>
          <strong>Phase 5 continuation.</strong> This view is routed but not built yet.
        </p>
        <p className="muted">
          Planned (per docs/plan/02-architecture.md §Board + R2 "feasibility is data"): pick a
          candidate machine spec and see which deferred runs it would unlock — a query over{" "}
          <code>status = deferred</code> runs × <code>deployment.requires</code> × the machine's
          VRAM/RAM/accel.
        </p>
      </div>
    </section>
  );
}
