import { Link } from "@tanstack/react-router";
import { useBoardData } from "../queries/data";

interface SystemType {
  key: string;
  title: string;
  /** catalog track ids that put a model in this group */
  tracks: string[];
  what: string;
  example: string;
  how: string;
  families: string;
}

const SYSTEM_TYPES: SystemType[] = [
  {
    key: "decision",
    title: "Decision models",
    tracks: ["decision", "score"],
    what: "Small and mid-size language models trained to answer one structured question. They pick one option, give a yes/no, or score text against a rubric. Instead of free text they return a probability for every allowed answer.",
    example:
      "“Is this review positive, negative, or mixed?” returns positive 0.91, mixed 0.07, negative 0.02.",
    how: "Choice questions from the mixed pool (sentiment and topic), the 9-language multilingual pool, and the gate suite. We score accuracy, and Brier and ECE for whether the probabilities are trustworthy.",
    families: "Decider, D1, Solar Decide, Laya, Lumma, Decision 2.0 / D3, TypeLLM, Clef…",
  },
  {
    key: "classify",
    title: "Zero-shot classifiers",
    tracks: ["classify"],
    what: "Encoder models, much smaller than LLMs, that check how well the text matches each candidate label. NLI models test whether “this text is about {label}” follows from the text. GLiClass-style models score every label in a single pass.",
    example:
      "Text: “My card was charged twice.” Labels: billing, shipping, login. Answer: billing.",
    how: "The same choice questions as the decision models. The highest-scoring label is the answer, so accuracy can be compared directly.",
    families: "NLI DeBERTa-v3 / ModernBERT, GLiClass, GLiNER2.5, GLiFormer",
  },
  {
    key: "rerank",
    title: "Rerankers",
    tracks: ["rerank"],
    what: "Cross-encoders built for search. Given a query and a list of documents, they score how relevant each document is so the best results come first. Retrieval and RAG pipelines use them as a second stage.",
    example:
      "Query: “refund policy”. They rank the FAQ entry on returns above the one on shipping times.",
    how: "Used as classifiers: the question is the query and each label is a document. The top-ranked label is the answer, and a softmax over the scores gives probabilities.",
    families: "Cohere Rerank, Voyage Rerank, BGE reranker, mxbai-rerank, GTE reranker",
  },
  {
    key: "ner",
    title: "NER & relation extractors",
    tracks: ["ner", "relation"],
    what: "Named-entity recognition (NER) finds and labels exact spans of text, such as people, places, drugs, or invoice numbers, for any entity types you name at run time. Relation extraction then links pairs of entities, for example person → works at → company.",
    example:
      "“Dr. Silva prescribed 20 mg atorvastatin in Kandy” gives person: Dr. Silva, dosage: 20 mg, drug: atorvastatin, location: Kandy.",
    how: "The graded NER pool. Strict micro-F1 needs exact span boundaries and labels, while partial F1 also credits overlapping spans. Results are on the NER tab.",
    families: "GLiNER2.5, GLiNER-relex, GLiFormer, GLiREL, ReLiK",
  },
];

const MODALITIES = [
  { key: "text", label: "Text" },
  { key: "image", label: "Image" },
  { key: "audio", label: "Audio" },
  { key: "video", label: "Video" },
] as const;

const GLOSSARY: [string, string][] = [
  [
    "Zero-shot",
    "The task is described only at inference time, through labels, descriptions, or an entity schema. The model sees no training examples for it.",
  ],
  [
    "Hosted vs. local",
    "Hosted means a paid API endpoint (OpenRouter, Workers AI, vendor APIs). Local means open weights running on our own CPU or GPU machine through a runner.",
  ],
  [
    "Deployment",
    "One way to run a model: a specific provider endpoint or a specific local runtime. The same model can have several deployments with different speed and cost.",
  ],
  [
    "Accuracy",
    "Share of choice questions answered correctly. This is the main metric for decision models, classifiers, and rerankers used as classifiers.",
  ],
  [
    "Brier / ECE",
    "Calibration metrics. They measure whether a model that says 90% is right about 90% of the time. Lower is better.",
  ],
  [
    "Strict / partial F1",
    "NER span scores. Strict needs exact boundaries and the right label; partial also credits overlapping spans with the right label.",
  ],
  [
    "p50 / p95 latency",
    "Median and 95th-percentile time per item. For local runs this is the runner's measured compute time, and for hosted runs it is client-side wall time.",
  ],
  [
    "ZDR",
    "Zero Data Retention: the provider promises not to store prompts or outputs. Hosted endpoints without ZDR are marked †.",
  ],
];

export function OverviewPage() {
  const { deployments, runs } = useBoardData();

  const modelCount = new Set(deployments.map((d) => d.model_id)).size;
  const familyCount = new Set(deployments.map((d) => d.family)).size;
  const hostedCount = deployments.filter((d) => d.kind === "hosted").length;
  const localCount = deployments.length - hostedCount;

  // unique models per claimed track / modality (deployments share a model's claims)
  const models = new Map(deployments.map((d) => [d.model_id, d.claims]));
  const countByTrack = (tracks: string[]) =>
    [...models.values()].filter((c) => c.tracks.some((t) => tracks.includes(t))).length;
  const countByModality = (modality: string) =>
    [...models.values()].filter((c) => c.modalities.includes(modality)).length;

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
          Labels and schemas are supplied at run time. Nothing is retrained between tasks, so the
          same model can sort support tickets today and triage medical notes tomorrow.
        </p>
      </article>

      <article className="card" id="what-we-cover">
        <h2>What kinds of models we cover</h2>
        <p className="muted">
          Every system in the catalog falls into one or more of these groups. Counts are unique
          models that claim the task; a model can belong to more than one group.
        </p>
        <div className="overview-grid">
          {SYSTEM_TYPES.map((t) => (
            <div key={t.key} className="guide-card type-card">
              <strong>
                {t.title} <span className="type-count">{countByTrack(t.tracks)}</span>
              </strong>
              <span>{t.what}</span>
              <span className="muted">
                <em>Example:</em> {t.example}
              </span>
              <span className="muted">
                <em>How we bench it:</em> {t.how}
              </span>
              <span className="muted">
                <em>Families:</em> {t.families}
              </span>
            </div>
          ))}
        </div>

        <h3 className="overview-subhead">Input modalities</h3>
        <p className="muted">
          A modality is the kind of input a model can read. Every model reads text, and some also
          accept images, audio, or video.{" "}
          <strong>The benchmark suites are text-only for now</strong>, so the other modalities are
          listed as the vendor describes them and have not been measured yet.
        </p>
        <div className="overview-stats">
          {MODALITIES.map((m) => (
            <div key={m.key} className="stat-box">
              <span className="stat-value">{countByModality(m.key)}</span>
              <span className="stat-label muted">
                {m.label}
                {m.key === "text" ? " (benchmarked)" : " (claimed, not yet benchmarked)"}
              </span>
            </div>
          ))}
        </div>

        <h3 className="overview-subhead">Glossary</h3>
        <dl className="glossary">
          {GLOSSARY.map(([term, def]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd className="muted">{def}</dd>
            </div>
          ))}
        </dl>
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
