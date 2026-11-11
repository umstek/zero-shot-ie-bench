// zsib local — the live-local-runner tryer (split out of the old playground).
// No API keys anywhere: the only backends this page talks to are runner
// islands on your own machine, always proxied by the local server (no CORS
// from the browser).
import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

type IslandModel = { id: string; display: string; versions: Array<string> };

type Island = {
  runner: string;
  envVar: string;
  baseUrl: string | null;
  models: Array<IslandModel>;
};

type Probe = {
  baseUrl: string;
  reachable: boolean;
  runner?: string;
  models?: Array<{ modelId: string; version: string | null }>;
  host?: unknown;
  error?: string;
} | null;

type AskResult = {
  source: string;
  model: string;
  prediction: unknown;
  prob: number | null;
  error: string | null;
  latencyMs: number;
  computeMs: number | null;
} | null;

async function postJson(url: string, body: unknown): Promise<Record<string, unknown>> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await res.json().catch(() => null)) as Record<string, unknown> | null;
  if (!res.ok) {
    throw new Error(typeof data?.error === "string" ? data.error : `HTTP ${res.status}`);
  }
  return data ?? {};
}

async function getJson(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url);
  const data = (await res.json().catch(() => null)) as Record<string, unknown> | null;
  if (!res.ok) {
    throw new Error(typeof data?.error === "string" ? data.error : `HTTP ${res.status}`);
  }
  return data ?? {};
}

function App() {
  const [stateText, setStateText] = useState("");
  const [optionsText, setOptionsText] = useState("");

  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8030");
  const [model, setModel] = useState("");
  const [islands, setIslands] = useState<Array<Island>>([]);
  const [probe, setProbe] = useState<Probe>(null);
  const [probeBusy, setProbeBusy] = useState(false);

  const [busy, setBusy] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResult>(null);

  const options = optionsText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const loadRunners = useCallback(async (probeUrl: string | null) => {
    const query = probeUrl ? `?baseUrl=${encodeURIComponent(probeUrl)}` : "";
    const data = await getJson(`/api/runners${query}`);
    setIslands(Array.isArray(data.islands) ? (data.islands as Array<Island>) : []);
    setProbe((data.probe as Probe) ?? null);
  }, []);

  useEffect(() => {
    loadRunners(null).catch((err: Error) => setAskError(err.message));
  }, [loadRunners]);

  async function askRunner() {
    if (stateText.trim().length === 0) {
      setAskError("state is required (free text)");
      return;
    }
    if (options.length < 2) {
      setAskError("at least two options (one per line) are required");
      return;
    }
    if (model.trim().length === 0) {
      setAskError("model is required (the runner's model_id — probe /info to fill it)");
      return;
    }
    setBusy(true);
    setAskError(null);
    setResult(null);
    try {
      const data = await postJson("/api/runner-ask", {
        baseUrl,
        model: model.trim(),
        state: stateText,
        question: { type: "choice", options },
      });
      setResult({
        source: `runner ${data.baseUrl ?? baseUrl}`,
        model: typeof data.model === "string" ? data.model : model,
        prediction: data.prediction ?? null,
        prob: typeof data.prob === "number" ? data.prob : null,
        error: typeof data.error === "string" ? data.error : null,
        latencyMs: typeof data.latencyMs === "number" ? data.latencyMs : 0,
        computeMs: typeof data.computeMs === "number" ? data.computeMs : null,
      });
    } catch (err) {
      setAskError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runProbe() {
    setProbeBusy(true);
    setProbe(null);
    try {
      await loadRunners(baseUrl.trim());
    } catch (err) {
      setAskError(err instanceof Error ? err.message : String(err));
    } finally {
      setProbeBusy(false);
    }
  }

  return (
    <main>
      <h1>zsib local</h1>
      <p className="muted">
        No API keys anywhere — this page talks only to local runner islands from the catalog,
        proxied by the server on your machine. Cloud deployments (bring your own key) live in the
        separate static app.
      </p>

      <section>
        <h2>Shared question</h2>
        <label htmlFor="state">State (free text)</label>
        <textarea
          id="state"
          rows={4}
          value={stateText}
          placeholder="The passage or context the question is about…"
          onChange={(e) => setStateText(e.target.value)}
        />
        <label htmlFor="options">Options (one per line)</label>
        <textarea
          id="options"
          rows={4}
          value={optionsText}
          placeholder={"yes\nno"}
          onChange={(e) => setOptionsText(e.target.value)}
        />
      </section>

      <section>
        <h2>Live local runner</h2>
        <label htmlFor="baseUrl">Runner base URL</label>
        <div className="row">
          <input
            id="baseUrl"
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://127.0.0.1:8030"
          />
          <button type="button" onClick={runProbe} disabled={probeBusy}>
            {probeBusy ? "probing…" : "Probe /info"}
          </button>
        </div>
        {probe && (
          <p className={probe.reachable ? "ok" : "error"}>
            {probe.reachable
              ? `reachable: runner "${probe.runner}" serves ${(probe.models ?? [])
                  .map((m) => m.modelId)
                  .join(", ")}`
              : `unreachable: ${probe.error}`}
          </p>
        )}
        {probe?.reachable && (probe.models?.length ?? 0) > 0 && (
          <button
            type="button"
            className="small"
            onClick={() => {
              const first = probe.models?.[0];
              if (first) setModel(first.modelId);
            }}
          >
            Use first served model
          </button>
        )}
        <label htmlFor="model">Model (runner model_id)</label>
        <input
          id="model"
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="wfzyx/von"
        />
        <button type="button" onClick={askRunner} disabled={busy}>
          {busy ? "asking…" : "Ask runner"}
        </button>

        <h3>Catalog runner islands</h3>
        {islands.length === 0 && <p className="muted">no local deployments in the catalog</p>}
        <ul className="islands">
          {islands.map((island) => (
            <li key={island.runner}>
              <code>{island.runner}</code>{" "}
              {island.baseUrl ? (
                <>
                  <code>{island.baseUrl}</code>{" "}
                  <button
                    type="button"
                    className="small"
                    onClick={() => setBaseUrl(island.baseUrl ?? baseUrl)}
                  >
                    Use URL
                  </button>
                </>
              ) : (
                <span className="muted">not configured — export {island.envVar}</span>
              )}
              <div className="muted">{island.models.map((m) => m.display).join(" · ")}</div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>Result</h2>
        {askError && <p className="error">{askError}</p>}
        {result === null && !askError && <p className="muted">no answer yet</p>}
        {result && (
          <dl className="result">
            <dt>source</dt>
            <dd>
              {result.source} <span className="muted">({result.model})</span>
            </dd>
            <dt>prediction</dt>
            <dd>
              <strong>{result.prediction === null ? "—" : String(result.prediction)}</strong>
            </dd>
            <dt>prob</dt>
            <dd>{result.prob === null ? "—" : result.prob.toFixed(4)}</dd>
            <dt>latency</dt>
            <dd>
              {result.latencyMs} ms
              {result.computeMs !== null && (
                <span className="muted"> (compute {result.computeMs} ms)</span>
              )}
            </dd>
            {result.error !== null && (
              <>
                <dt>item error</dt>
                <dd className="error">{result.error}</dd>
              </>
            )}
          </dl>
        )}
      </section>
    </main>
  );
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<App />);
