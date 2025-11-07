# Next Steps – Extended Patch Search

## Objective
Increase the number and diversity of tensor modifications per capability to test whether richer hook sets deliver stronger gains on larger models.

## Planned Changes

1. **Expand Tensor Sampling**
   - Update `tensor_inspector.sample_stats()` to include a much larger set of tensors (e.g. `sample_count=None` or ≥100 stats per capability).

2. **Prompt for More Suggestions**
   - Modify the AI prompt in `ai_agent.generate_modifications()` to request ≥50 distinct edits per capability.

3. **Adjust Filtering Logic**
   - In `server/api.py`, raise `_filter_suggestions` settings to keep the top 15–20 high-confidence edits (`max_count=20`, `min_confidence≈0.70`).

4. **Apply All Filtered Hooks**
   - Remove the `[:10]` slice in `apply_hooks()` so every filtered suggestion registers (still capped by the new filter).

5. **Benchmark Combined Capabilities**
   - Optionally add a mode to `bench/run_bench.py` that merges suggestions from multiple capabilities before applying hooks.

6. **Rerun Benchmarks**
   - Test on GSM8K with the 14B (and larger) models using the expanded patch sets.
   - Compare baseline vs new general/combined hook runs; evaluate with the shared numeric extractor.

7. **Document Results**
   - Capture new metrics, hook counts, and qualitative observations for any positive runs.

## Goal
Determine whether richer, high-confidence patch sets improve accuracy beyond the current 4-hook general configuration, particularly on the 14B+ models.
