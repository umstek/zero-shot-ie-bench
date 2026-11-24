import { Link } from "@tanstack/react-router";
import { useBoardData } from "../queries/data";

export function OverviewPage() {
  const { deployments, runs } = useBoardData();

  const modelCount = new Set(deployments.map((d) => d.model_id)).size;
  const familyCount = new Set(deployments.map((d) => d.family)).size;
  const hostedCount = deployments.filter((d) => d.kind === "hosted").length;
  const localCount = deployments.length - hostedCount;

  return (
    <section>
      <header className="page-head">
        <h1>Zero-Shot IE &amp; Decision Benchmark</h1>
        <p>
          <code>zsib</code> (<code>zero-shot-ie-bench</code>) is an open, reproducible benchmark
          comparing hosted APIs and local on-device models on zero-shot text classification, typed
          decisions, reranking, and named-entity recognition (NER) — with no task-specific
          fine-tuning.
        </p>

        <div className="overview-stats">
          <div className="stat-box">
            <span className="stat-value">{modelCount}</span>
            <span className="stat-label muted">Models</span>
          </div>
          <div className="stat-box">
            <span className="stat-value">{deployments.length}</span>
            <span className="stat-label muted">
              Deployments ({hostedCount} hosted · {localCount} local)
            </span>
          </div>
          <div className="stat-box">
            <span className="stat-value">{familyCount}</span>
            <span className="stat-label muted">Model families</span>
          </div>
          <div className="stat-box">
            <span className="stat-value">{runs.length}</span>
            <span className="stat-label muted">Measured runs</span>
          </div>
        </div>

        <div className="overview-actions">
          <Link to="/picks" className="btn btn-primary">
            See Top Picks →
          </Link>
          <Link to="/explore" className="btn">
            Browse Explorer
          </Link>
          <a href="/try/" className="btn">
            Try in Browser
          </a>
          <a
            href="https://github.com/umstek/zero-shot-ie-bench"
            className="btn"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </div>
      </header>

      <article className="card">
        <h2>What is zero-shot IE &amp; decision benchmarking?</h2>
        <p>
          Zero-shot information extraction and decision systems classify text into arbitrary labels,
          pick structured answers, rerank candidates, or extract entity spans from a schema provided
          at inference time — without training or fine-tuning on task data.
        </p>
        <p className="muted">
          <code>zsib</code> evaluates four system archetypes side by side — typed-decision engines,
          zero-shot classifiers, cross-encoder rerankers, and span/NER extractors — across both
          local runner islands and hosted endpoints.
        </p>
      </article>

      <article className="card">
        <h2>How to use this site</h2>
        <div className="overview-grid">
          <Link to="/picks" className="guide-card">
            <strong>Picks</strong>
            <span className="muted">
              Concrete answers — cheapest at ≥90% accuracy, fastest on CPU, best on rare languages —
              computed from the latest runs.
            </span>
          </Link>
          <Link to="/explore" className="guide-card">
            <strong>Explorer</strong>
            <span className="muted">
              Filterable, sortable table of every deployment with accuracy, p50 latency, measured
              cost per 1k items, and ZDR status.
            </span>
          </Link>
          <Link to="/compare" className="guide-card">
            <strong>Charts</strong>
            <span className="muted">
              Speed vs. accuracy, price vs. accuracy, cost vs. speed Pareto frontiers, top-20
              rankings, item difficulty spectrum, and language coverage.
            </span>
          </Link>
          <Link to="/ner" className="guide-card">
            <strong>NER</strong>
            <span className="muted">
              Span-extraction leaderboard ranked by strict and partial micro-F1 on the graded NER
              pool.
            </span>
          </Link>
          <a href="/try/" className="guide-card">
            <strong>Try it</strong>
            <span className="muted">
              Browser-only BYOK playground for testing hosted models with your own API keys (keys
              stay in browser memory only).
            </span>
          </a>
        </div>
      </article>

      <article className="card">
        <h2>Methodology at a glance</h2>
        <ul className="overview-list">
          <li>
            <strong>Versioned suites:</strong> Mixed-pool decision suite (48 sentiment &amp; topic
            items), 9-language parallel multilingual suite (54 items across popular, medium, and
            rare tiers), and graded NER pool (18 items), each SHA-256 hashed so edits never rewrite
            history.
          </li>
          <li>
            <strong>Append-only run history:</strong> Every run in <code>runs/</code> is immutable
            (CI-enforced); rebenching records model drift over time instead of overwriting past
            measurements.
          </li>
          <li>
            <strong>Truthful latency &amp; cost:</strong> Local runs report runner-measured{" "}
            <code>compute_ms</code> and amortized hardware cost (never <code>$0</code>); hosted runs
            report client-side latency and provider-reported usage.
          </li>
          <li>
            <strong>Zero Data Retention (ZDR):</strong> Flags non-ZDR hosted endpoints so
            privacy-sensitive deployments are clearly marked.
          </li>
        </ul>
        <p className="muted overview-doc-links">
          Docs on GitHub:{" "}
          <a
            href="https://github.com/umstek/zero-shot-ie-bench/blob/main/docs/methodology.md"
            target="_blank"
            rel="noreferrer"
          >
            methodology
          </a>{" "}
          ·{" "}
          <a
            href="https://github.com/umstek/zero-shot-ie-bench/blob/main/docs/adding-a-model.md"
            target="_blank"
            rel="noreferrer"
          >
            adding a model
          </a>{" "}
          ·{" "}
          <a
            href="https://github.com/umstek/zero-shot-ie-bench/blob/main/CHANGELOG.md"
            target="_blank"
            rel="noreferrer"
          >
            changelog
          </a>
        </p>
      </article>
    </section>
  );
}
