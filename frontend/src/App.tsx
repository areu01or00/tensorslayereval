import { useEffect, useMemo, useState } from "react";
import {
  applyHooks,
  clearHooks,
  fetchConfigs,
  generate,
  requestSuggestions,
} from "./api";
import type { InferenceConfig, Suggestion } from "./types";

interface HistoryEntry {
  id: number;
  timestamp: string;
  query: string;
  config: string;
  usedHooks: boolean;
}

const INITIAL_STATUS = "Ready. Model loaded.";

function App() {
  const [configs, setConfigs] = useState<Record<string, InferenceConfig>>({});
  const [selectedConfig, setSelectedConfig] = useState<string>("balanced");
  const [query, setQuery] = useState<string>("");
  const [originalResponse, setOriginalResponse] = useState<string>("");
  const [modifiedResponse, setModifiedResponse] = useState<string>("");
  const [hookCount, setHookCount] = useState<number>(0);
  const [hooksActive, setHooksActive] = useState<boolean>(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [selectedSuggestions, setSelectedSuggestions] = useState<Set<number>>(new Set());
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [status, setStatus] = useState<string>(INITIAL_STATUS);
  const [loading, setLoading] = useState<boolean>(false);
  const [suggestionsLoading, setSuggestionsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [capability, setCapability] = useState<string>("general");
  const [historyCounter, setHistoryCounter] = useState<number>(0);

  useEffect(() => {
    fetchConfigs()
      .then((data) => {
        setConfigs(data);
        if (!data[selectedConfig]) {
          const fallback = Object.keys(data)[0];
          if (fallback) {
            setSelectedConfig(fallback);
          }
        }
      })
      .catch((err) => {
        console.error(err);
        setError(err.message);
      });
  }, []);

  const configOptions = useMemo(
    () =>
      Object.entries(configs).map(([key, cfg]) => ({
        key,
        label: cfg.name,
      })),
    [configs]
  );

  const selectedConfigDetails = configs[selectedConfig];

  async function handleGenerate(useHooks: boolean) {
    if (!query.trim()) {
      setStatus("Please enter a query first!");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setStatus("Generating response...");
      const response = await generate({
        query,
        config: selectedConfig,
        use_hooks: useHooks,
      });

      setOriginalResponse(response.original);
      if (response.modified) {
        setModifiedResponse(response.modified);
      } else if (useHooks) {
        setModifiedResponse("Hooks are active but no modified response returned.");
      } else {
        setModifiedResponse("Hooks inactive. Apply suggestions to compare.");
      }
      setHookCount(response.hook_count);

      setHistory((entries) => [
        {
          id: historyCounter + 1,
          timestamp: new Date().toLocaleTimeString(),
          query,
          config: selectedConfig,
          usedHooks: useHooks && hooksActive,
        },
        ...entries,
      ]);
      setHistoryCounter((count) => count + 1);

      setStatus("Generation complete!");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setStatus("Generation failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleSuggestions() {
    try {
      setSuggestionsLoading(true);
      setError(null);
      setStatus("Fetching AI suggestions...");
      const response = await requestSuggestions(capability);
      setSuggestions(response.suggestions || []);
      setSelectedSuggestions(new Set());
      setStatus(`Received ${response.suggestions?.length ?? 0} suggestions`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setStatus("Suggestions failed");
    } finally {
      setSuggestionsLoading(false);
    }
  }

  async function handleApply() {
    const chosen = suggestions.filter((_, index) => selectedSuggestions.has(index));
    const payload = chosen.length ? chosen : suggestions;

    if (!payload.length) {
      setStatus("No suggestions selected to apply.");
      return;
    }

    try {
      setError(null);
      setStatus("Applying hooks...");
      const { hook_count } = await applyHooks(payload);
      setHookCount(hook_count);
      const active = hook_count > 0;
      setHooksActive(active);
      if (active) {
        setStatus(`Applied ${hook_count} modifications.`);
      } else {
        setStatus("No valid modifications were applied (modules not found).");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setStatus("Failed to apply hooks");
    }
  }

  async function handleClearHooks() {
    try {
      setError(null);
      setStatus("Clearing hooks...");
      await clearHooks();
      setHookCount(0);
      setHooksActive(false);
      setStatus("Hooks cleared.");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setStatus("Failed to clear hooks");
    }
  }

  function toggleSuggestion(index: number) {
    setSelectedSuggestions((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Inference Engine</h1>
          <p className="subtitle">Compare original vs modified outputs with AI-driven hooks.</p>
        </div>
        <div className="status">
          <span>{status}</span>
          {error && <span className="error">{error}</span>}
        </div>
      </header>

      <main className="layout">
        <section className="left-pane">
          <div className="panel">
            <h2>Query</h2>
            <textarea
              className="query-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Enter your query here..."
              rows={4}
            />
            <div className="control-row">
              <label htmlFor="config-select">Configuration:</label>
              <select
                id="config-select"
                value={selectedConfig}
                onChange={(event) => setSelectedConfig(event.target.value)}
              >
                {configOptions.map((option) => (
                  <option key={option.key} value={option.key}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            {selectedConfigDetails && (
              <p className="config-description">{selectedConfigDetails.description}</p>
            )}
            <div className="button-row">
              <button onClick={() => handleGenerate(false)} disabled={loading}>
                Generate
              </button>
              <button
                onClick={() => handleGenerate(true)}
                disabled={loading || !hooksActive}
              >
                Generate with Hooks
              </button>
              <button onClick={handleApply} disabled={!suggestions.length}>
                Apply Suggestions
              </button>
              <button onClick={handleClearHooks} disabled={!hooksActive}>
                Clear Hooks
              </button>
            </div>
          </div>

          <div className="panel suggestions">
            <div className="panel-header">
              <h2>AI Suggestions</h2>
              <div className="panel-actions">
                <select
                  value={capability}
                  onChange={(event) => setCapability(event.target.value)}
                >
                  <option value="general">General</option>
                  <option value="math">Math</option>
                  <option value="reasoning">Reasoning</option>
                </select>
                <button onClick={handleSuggestions} disabled={suggestionsLoading}>
                  {suggestionsLoading ? "Loading..." : "Get Suggestions"}
                </button>
              </div>
            </div>
            {suggestions.length === 0 && <p>No suggestions yet.</p>}
            <ul className="suggestion-list">
              {suggestions.map((suggestion, index) => (
                <li key={`${suggestion.tensor_name}-${index}`}>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedSuggestions.has(index)}
                      onChange={() => toggleSuggestion(index)}
                    />
                    <span className="tensor-name">{suggestion.tensor_name}</span>
                  </label>
                  <div className="suggestion-details">
                    <span>
                      <strong>{suggestion.operation}</strong> → {suggestion.value} ({" "}
                      {suggestion.target})
                    </span>
                    {suggestion.confidence !== undefined && (
                      <span className="confidence">
                        Confidence: {(suggestion.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  {suggestion.reason && <p className="reason">{suggestion.reason}</p>}
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="right-pane">
          <div className="panel comparison">
            <div className="comparison-columns">
              <article>
                <h2>Original Response</h2>
                <pre>{originalResponse}</pre>
              </article>
              <article>
                <h2>
                  Modified Response
                  {hooksActive ? ` (${hookCount} hooks active)` : ""}
                </h2>
                <pre>{modifiedResponse}</pre>
              </article>
            </div>
          </div>

          <div className="panel history">
            <h2>History</h2>
            {history.length === 0 ? (
              <p>No history yet.</p>
            ) : (
              <ul>
                {history.map((entry) => (
                  <li key={entry.id}>
                    <div className="history-header">
                      <span className="timestamp">{entry.timestamp}</span>
                      <span className="config">Config: {entry.config}</span>
                      {entry.usedHooks && <span className="hooks-label">Hooks</span>}
                    </div>
                    <p className="history-query">{entry.query}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
