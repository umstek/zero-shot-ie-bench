import { Link } from "@tanstack/react-router";
import { buildPicks } from "../picks";
import { useBoardData } from "../queries/data";

export function PicksPage() {
  const { deployments, runs } = useBoardData();
  const picks = buildPicks(deployments, runs);

  return (
    <section>
      <header className="page-head">
        <h1>Picks</h1>
        <p>
          Generated answers to concrete questions, computed in memory from the latest{" "}
          <code>status: ok</code> decision-track run per deployment. Every row links to the model's
          full history.
        </p>
      </header>

      {picks.map((pick) => (
        <article key={pick.id} className="card">
          <h2>{pick.question}</h2>
          <p className="muted">{pick.note}</p>
          {pick.rows.length === 0 ? (
            <p className="muted">No qualifying runs in the lake yet.</p>
          ) : (
            <ol className="pick-list">
              {pick.rows.map((row) => (
                <li key={row.deployment.deployment_id}>
                  <span className="pick-headline">{row.headline}</span>
                  <Link to="/model/$" params={{ _splat: row.deployment.model_id }}>
                    {row.deployment.display}
                  </Link>
                  <span className="muted pick-detail">
                    {row.deployment.org} · {row.deployment.kind} · {row.detail}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </article>
      ))}
    </section>
  );
}
