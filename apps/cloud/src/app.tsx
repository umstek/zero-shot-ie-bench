// zsib cloud — the BYOK tryer for hosted catalog deployments. One static
// page: pick a deployment, paste your own key(s) for its provider, ask a
// choice question, read the result. The adapter runs IN THIS BROWSER and the
// key is sent only to the provider's API — this page has no backend to send
// anything to. Keys live in React state only: no localStorage, no logging,
// never in the URL; a reload clears them by design.
import { Component, type ReactNode, Suspense, use, useState } from "react";
import { AskError, type AskOutcome, askHosted } from "./ask";
import { cloudData } from "./queries/data";

type AskResult = AskOutcome | null;

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return <p className="error">failed to load the catalog data: {this.state.error.message}</p>;
    }
    return this.props.children;
  }
}

function Tryer() {
  const { deployments, providers } = use(cloudData());

  const [selected, setSelected] = useState(deployments[0]?.deployment_id ?? "");
  const [stateText, setStateText] = useState("");
  const [optionsText, setOptionsText] = useState("");
  // user keys, keyed by the providers' env var NAMES — React state only
  const [keys, setKeys] = useState<Record<string, string>>({});

  const [busy, setBusy] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [corsHint, setCorsHint] = useState(false);
  const [result, setResult] = useState<AskResult>(null);

  const selectedDeployment = deployments.find((d) => d.deployment_id === selected) ?? null;
  const selectedProvider = providers.find((p) => p.id === selectedDeployment?.provider) ?? null;
  const keyNames = selectedProvider?.auth ?? [];
  const missingKeys = keyNames.filter((name) => !(keys[name] ?? "").trim());

  const options = optionsText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  async function ask() {
    if (!selectedDeployment) {
      setAskError("pick a deployment first");
      return;
    }
    if (stateText.trim().length === 0) {
      setAskError("state is required (free text)");
      return;
    }
    if (options.length < 2) {
      setAskError("at least two options (one per line) are required");
      return;
    }
    if (missingKeys.length > 0) {
      setAskError(`missing key(s): ${missingKeys.join(", ")}`);
      return;
    }
    setBusy(true);
    setAskError(null);
    setCorsHint(false);
    setResult(null);
    try {
      const outcome = await askHosted({
        deployment: selectedDeployment,
        providers,
        keys,
        state: stateText,
        question: { type: "choice", options },
      });
      setResult(outcome);
    } catch (err) {
      setAskError(err instanceof Error ? err.message : String(err));
      setCorsHint(err instanceof AskError && err.corsSuspected);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>
        zsib <span className="muted">cloud tryer</span>
      </h1>
      <p className="muted">
        Hosted catalog deployments through the same bench adapters, called from your browser with
        <strong> your own API keys</strong>.
      </p>
      <p className="note">
        your key stays in this browser tab — it is sent only to the provider, never to us (there is
        no us: this page is static)
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
        <h2>Hosted deployment</h2>
        <label htmlFor="deployment">Deployment (from the catalog)</label>
        <select
          id="deployment"
          value={selected}
          onChange={(e) => {
            setSelected(e.target.value);
            setAskError(null);
            setResult(null);
          }}
        >
          {deployments.map((d) => (
            <option key={d.deployment_id} value={d.deployment_id}>
              {d.display ?? d.deployment_id} — {d.provider}
            </option>
          ))}
        </select>
        {selectedDeployment && (
          <p className="muted">
            adapter <code>{selectedDeployment.adapter}</code> · endpoint{" "}
            <code>{selectedDeployment.endpoint}</code>
            {selectedDeployment.zdr_safe === false && (
              <span className="badge warn">not ZDR-safe</span>
            )}
            {selectedDeployment.zdr_safe === true && <span className="badge ok">ZDR-safe</span>}
          </p>
        )}

        {keyNames.length > 0 && (
          <>
            <label htmlFor={`key-${keyNames[0]}`}>
              {selectedProvider?.name ?? selectedDeployment?.provider} key
              {keyNames.length > 1 ? "s" : ""}{" "}
              <span className="muted">(env var names the adapter reads)</span>
            </label>
            {keyNames.map((name) => (
              <input
                key={name}
                id={`key-${name}`}
                type="password"
                autoComplete="off"
                placeholder={name}
                value={keys[name] ?? ""}
                onChange={(e) => setKeys({ ...keys, [name]: e.target.value })}
              />
            ))}
          </>
        )}
        <button type="button" onClick={ask} disabled={busy}>
          {busy ? "asking…" : "Ask hosted"}
        </button>
      </section>

      <section>
        <h2>Result</h2>
        {askError && (
          <p className="error">
            {askError}
            {corsHint && (
              <>
                <br />
                this provider may not allow direct browser calls (CORS) — it may work through the
                local app or CLI
              </>
            )}
          </p>
        )}
        {result === null && !askError && <p className="muted">no answer yet</p>}
        {result && (
          <dl className="result">
            <dt>source</dt>
            <dd>
              {result.deployment} <span className="muted">({result.model})</span>
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

export function App() {
  return (
    <ErrorBoundary>
      <Suspense
        fallback={
          <main>
            <p className="muted">loading catalog…</p>
          </main>
        }
      >
        <Tryer />
      </Suspense>
    </ErrorBoundary>
  );
}
