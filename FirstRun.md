# FirstRun – Qwen3-8B Hook Benchmark (GSM8K, 2025-10-19)

## Hardware & Environment
- **Host:** Prime-Intellect GPU node (A40 48 GB, 16 vCPU, 64 GB RAM)
- **Backend:** `uvicorn server.api:app` (MODEL_PATH → `models/Qwen__Qwen3-8B`)
- **Frontend:** Vite dev server (local, proxy to backend)
- **Python deps:** `torch 2.x`, `transformers 4.35`, `fastapi`, `accelerate`, `huggingface_hub`, `datasets`
- **LLM suggestions:** OpenRouter `moonshotai/kimi-k2-0905` via smolagents ➜ LiteLLM wrapper

## Benchmark Setup
- **Dataset:** GSM8K `main` split (first 100 prompts used for each capability)
- **Harness:** `bench/run_bench.py --model-id Qwen/Qwen3-8B --max-prompts 100`
- **Workflow per prompt:**
  1. Clear hooks → baseline generation (`use_hooks=false`).
  2. For capability in {general, math, reasoning}:
     - Fetch tensor stats from new `TensorInspector` (full safetensor sweep).
     - AI generates candidate patches; filter top 20 (confidence ≥ 0.65).
     - Apply hooks once → generate with `use_hooks=true`.
     - Clear hooks.
- **Config:** Text generation preset `balanced` (temperature 0.6, top_p 0.95, top_k 40, max_new_tokens 4096, thinking_budget 2048).
- **Results archive:** `runs-backup/results-20251019-181021.{json,csv}`

## Tensor Stats & Suggestions
- `TensorInspector` now indexes every tensor across all safetensor shards.
- Suggestion prompt receives full stats (min/max/mean/std/zeros%) for attention, MLP, embeddings across early/mid/late layers.
- AI output filtered to ≤20 high-confidence edits before registering hooks (mirrors original Tensor Slayer workflow).

## Accuracy Evaluation
- Evaluated with numeric extraction heuristic:
  - Strip `<think>...</think>` blocks.
  - Capture final numeric answer (regex `final answer:` fallback to last number).
  - Compare to GSM8K reference answer (case-insensitive exact match).

### Capability-wise Accuracy (100 prompts each)
- **Baseline (no hooks):** 21 % (21/100)
- **General hooks:** **27 %** (27/100)
- **Math hooks:** 19 % (19/100)
- **Reasoning hooks:** 23 % (23/100)

Δ vs baseline: +6 pts (general), –2 pts (math), +2 pts (reasoning).

## Latency Snapshot (A40, 8B model, CPU offload disabled)
- Baseline avg: ~18 s per prompt.
- Hooked runs: ~37 s per prompt (hooks + second generation pass).

## Observations
- Filters successfully diversified edits: suggestions targeted multiple layers (e.g., `layers.0`, `layers.5`, `layers.12`, `layers.18`, embeddings).
- General capability patches (mix of scaling/clamping on attention + MLP weights) delivered a consistent accuracy lift despite doubled latency.
- Math capability underperformed baseline; reasoning marginally better.
- Numeric extractor is simplistic—LLM grading or pattern-aware parsing recommended before claiming final numbers.

## Follow-up Actions
1. **Validation:** Re-run with alternative scoring (regex per GSM8K format or LLM judge) to confirm gains aren’t grading artifacts.
2. **Stability:** Test on disjoint GSM8K subsets / different seeds to ensure general patch holds up.
3. **Patch analysis:** Log filtered suggestion sets for audit; examine which edits correlate with improvements.
4. **Optimization:** Explore caching general patch across runs; profile latency overhead; investigate mixed precision hooks.
5. **Documentation:** Update README / Issues with inspector integration, filtering logic, and benchmark outcome.

## Files
- `runs-backup/results-20251019-181021.json` – raw log (baseline+hooks per prompt/capability)
- `runs-backup/results-20251019-181021.csv` – tabular version
- `bench/run_bench.py` – benchmark harness
- `tensor_inspector.py` – safetensor-based stats collector

**Summary:** With full-model tensor statistics and high-confidence patch filtering, the general capability hooks produced a +6 pt accuracy lift on 100 GSM8K prompts against the vanilla Qwen3-8B baseline, demonstrating end-to-end viability of the tensor-hook pipeline. Further validation required before publishing, but this is our first positive signal from the 8B run.
