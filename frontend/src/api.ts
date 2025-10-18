import {
  GenerateResponse,
  SuggestionsResponse,
  Suggestion,
  InferenceConfig,
} from "./types";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export async function fetchConfigs(): Promise<Record<string, InferenceConfig>> {
  const res = await fetch("/api/configs");
  return handleResponse(res);
}

export async function generate(payload: {
  query: string;
  config: string;
  use_hooks: boolean;
}): Promise<GenerateResponse> {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function requestSuggestions(
  capability: string
): Promise<SuggestionsResponse> {
  const res = await fetch("/api/suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ capability }),
  });
  return handleResponse(res);
}

export async function applyHooks(suggestions: Suggestion[]) {
  const res = await fetch("/api/hooks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ suggestions }),
  });
  return handleResponse<{ hook_count: number; modules: string[] }>(res);
}

export async function clearHooks() {
  const res = await fetch("/api/hooks", { method: "DELETE" });
  return handleResponse<{ hook_count: number }>(res);
}

export async function getHooksStatus() {
  const res = await fetch("/api/hooks");
  return handleResponse(res);
}
