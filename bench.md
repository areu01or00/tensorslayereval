# Cloud Benchmark Plan

## Goal
Demonstrate repeatable improvements from tensor-hook modifications on large (≥70B) base models using the existing FastAPI endpoints without changing core code.

## Environment Checklist
- Prime-Intellect or comparable GPU node (≥80 GB VRAM or multi-GPU) with SSH access.
- Repo synced; Python venv with `pip install -r requirements.txt`.
- Node/npm installed if you need the React UI for monitoring (optional).
- Model weights mounted locally (e.g., `/data/models/qwen2-72b`).

## Services
1. **Backend**
   ```bash
   MODEL_PATH=/data/models/qwen2-72b \
   uvicorn server.api:app --host 0.0.0.0 --port 8000
   ```
2. **(Optional) Frontend** – only if you want live inspection.
   ```bash
   cd frontend
   npm install
   npm run dev -- --host 0.0.0.0 --port 5173
   ```
3. SSH tunnel ports 8000/5173 when working remotely.

## Benchmark Harness (No Core Code Changes)
- Create `bench/` with:
  - Optional `prompts/` JSONL overrides for custom tasks.
  - `run_bench.py` script that:
    1. Loads prompts from a Hugging Face dataset (defaults to `gsm8k/main:test`) or a JSONL file.
    2. Calls backend endpoints directly using `requests`.
    3. For each prompt:
       - `DELETE /api/hooks` to clear.
       - `POST /api/generate` (baseline) → log.
       - For each capability (general/math/reasoning by default):
         - `POST /api/suggestions` → capture tensor edits.
         - `POST /api/hooks` with those edits.
         - `POST /api/generate` with `use_hooks=true` → log.
       - `DELETE /api/hooks` before moving on.
    4. Scores outputs (exact match, normalized, or secondary model grader).
    5. Writes CSV/JSON results + summary stats.

## Patch Strategy
- Store hook payloads under `bench/patches/` for reproducibility.
- Apply same patch set across all prompts in a run.
- Record hook_count/module list from `/api/hooks` endpoint for provenance.

## Metrics & Reporting
- Per-task accuracy deltas (% success baseline vs hooked).
- Latency comparison (avg generation time, measured client-side).
- Failure counts (timeouts, API errors).
- Sample diff table (prompt, baseline answer, patched answer).

## Automation
- Run harness via CLI, e.g.: `python bench/run_bench.py --model-id <hf_repo> --max-prompts 200`.
- Optional: store raw outputs in `bench/runs/<timestamp>/` for auditing.
- Use cron or simple shell script to repeat across different patch sets/models.

## Validation Steps
1. Dry-run on small subset (e.g., 5 prompts) to ensure API flow.
2. Full run on selected task set.
3. Verify reproducibility by re-running same patch set.
4. Document run configuration (commit hash, MODEL_PATH, prompt set, patch file).

## Deliverables
- CSV of baseline vs patched scores per task.
- Markdown summary (e.g., `bench/reports/<date>.md`) with tables/plots.
- Optional: publish raw logs + scripts for community replication.
