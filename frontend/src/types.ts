export interface InferenceConfig {
  name: string;
  description: string;
  temperature: number;
  top_p: number;
  top_k: number;
  max_new_tokens: number;
  repetition_penalty: number;
  thinking_budget: number;
  do_sample?: boolean;
}

export interface Suggestion {
  tensor_name: string;
  operation: string;
  value: number;
  target: string;
  confidence?: number;
  reason?: string;
}

export interface GenerateResponse {
  original: string;
  modified?: string | null;
  hook_count: number;
}

export interface SuggestionsResponse {
  suggestions: Suggestion[];
}

export interface HooksStatus {
  active: boolean;
  stats: {
    total_hooks: number;
    total_modifications: number;
    modules: string[];
  };
}
