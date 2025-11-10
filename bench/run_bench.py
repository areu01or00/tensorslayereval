#!/usr/bin/env python3
"""Benchmark harness for baseline vs hook-modified inference."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency for offline runs
    snapshot_download = None  # type: ignore

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

API_TIMEOUT = 900  # seconds per request (increased for hooked generations with thinking tokens)


@dataclass
class PromptItem:
    id: str
    prompt: str
    answer: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs hook benchmarks")
    parser.add_argument(
        "--benchmark-file",
        help="Path to JSONL prompt set (overrides dataset options)",
    )
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        help="Hugging Face dataset name (default: gsm8k)",
    )
    parser.add_argument(
        "--dataset-config",
        default="main",
        help="Dataset config name (default: main)",
    )
    parser.add_argument(
        "--dataset-split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument("--model-id", help="Hugging Face repo id to download")
    parser.add_argument("--model-path", help="Existing model path (skip download)")
    parser.add_argument("--model-revision", default=None, help="Optional HF revision")
    parser.add_argument(
        "--config",
        default="benchmark",
        help="Inference configuration preset (default: benchmark - optimized for short answers)",
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        default=["general", "math", "reasoning"],
        help="Suggestion capabilities to benchmark",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit number of prompts for quick smoke tests",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API host (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument(
        "--output-dir",
        default="bench/runs",
        help="Directory to store CSV/JSON outputs",
    )
    parser.add_argument(
        "--apply-mode",
        default="weights",
        choices=["weights", "hooks"],
        help="Apply patches as in-memory weight edits or legacy activation hooks",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "loss"],
        help="Evaluation metric: generation accuracy or teacher-forced loss on answer tokens",
    )
    parser.add_argument(
        "--calibrate-loss",
        action="store_true",
        help="In loss mode: decide whether to keep applying patches based on mean delta over first K prompts",
    )
    parser.add_argument(
        "--calibrate-loss-size",
        type=int,
        default=25,
        help="Number of prompts for loss calibration window (loss mode only)",
    )
    parser.add_argument(
        "--calibrate-loss-threshold",
        type=float,
        default=-0.005,
        help="Mean loss delta threshold to accept patches (negative = improvement)",
    )
    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help="Assume server is already running; skip auto-launch",
    )
    parser.add_argument(
        "--uvicorn-log-level",
        default="error",
        help="Log level forwarded to uvicorn when auto-launching",
    )
    # Suggestions are generated live; no fixed patch file or suggestion saving flags
    return parser.parse_args()


def load_prompts_from_file(path: Path, limit: Optional[int] = None) -> List[PromptItem]:
    items: List[PromptItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                PromptItem(
                    id=str(data.get("id") or len(items)),
                    prompt=str(data["prompt"]),
                    answer=(data.get("answer") or None),
                )
            )
            if limit and len(items) >= limit:
                break
    return items


def load_prompts_from_dataset(
    dataset: str,
    config: Optional[str],
    split: str,
    limit: Optional[int] = None,
) -> List[PromptItem]:
    if load_dataset is None:
        raise RuntimeError(
            "datasets library required for dataset loading. Install with pip install datasets"
        )

    ds = load_dataset(dataset, config, split=split)

    items: List[PromptItem] = []
    for idx, row in enumerate(ds):
        prompt_text: Optional[str] = None
        answer_text: Optional[str] = None

        if dataset == "gsm8k":
            prompt_text = str(row["question"]).strip()
            raw_answer = str(row["answer"]).strip()
            if "####" in raw_answer:
                answer_text = raw_answer.split("####")[-1].strip()
            else:
                answer_text = raw_answer
        else:
            # Generic fallback expecting 'prompt' and 'answer' keys
            prompt_text = str(row.get("prompt"))
            answer_text = row.get("answer")

        if prompt_text is None:
            continue

        items.append(
            PromptItem(
                id=f"{dataset}_{idx}",
                prompt=prompt_text,
                answer=str(answer_text).strip() if answer_text is not None else None,
            )
        )

        if limit and len(items) >= limit:
            break

    if not items:
        raise ValueError("Dataset loader produced no prompts")

    return items


def ensure_model(model_id: Optional[str], model_path: Optional[str], revision: Optional[str]) -> Path:
    if model_path:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model path '{resolved}' does not exist")
        return resolved

    if not model_id:
        raise ValueError("Either --model-path or --model-id must be provided")

    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for auto-download. Install with pip install huggingface_hub"
        )

    safe_name = model_id.replace("/", "__")
    target_dir = Path("models") / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.iterdir()):
        print(f"[bench] Using cached model at {target_dir}")
        return target_dir

    print(f"[bench] Downloading model '{model_id}' to {target_dir}")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_dir


def launch_server(host: str, port: int, model_path: Path, log_level: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["MODEL_PATH"] = str(model_path)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.api:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    print(f"[bench] Launching uvicorn with MODEL_PATH={model_path}")
    proc = subprocess.Popen(cmd, env=env)
    return proc


def wait_for_server(base_url: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    health_url = f"{base_url}/api/health"
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.ok:
                print("[bench] API is ready")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("API did not become ready within timeout")


def api_post(base_url: str, path: str, payload: Dict) -> Dict:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_delete(base_url: str, path: str) -> Dict:
    resp = requests.delete(f"{base_url}{path}", timeout=API_TIMEOUT)
    resp.raise_for_status()
    if resp.content:
        return resp.json()
    return {}


def normalize_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return " ".join(text.strip().lower().split())


def evaluate(answer: Optional[str], output: Optional[str]) -> Optional[bool]:
    if answer is None or output is None:
        return None
    return normalize_answer(answer) == normalize_answer(output)


def timestamped_path(directory: Path, stem: str, suffix: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return directory / f"{stem}-{ts}.{suffix}"


# No suggestion hashing in the base harness


def run_benchmark(args: argparse.Namespace) -> None:
    if args.benchmark_file:
        prompts = load_prompts_from_file(Path(args.benchmark_file), args.max_prompts)
    else:
        prompts = load_prompts_from_dataset(
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            args.max_prompts,
        )

    model_path = ensure_model(args.model_id, args.model_path, args.model_revision)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    server_proc: Optional[subprocess.Popen] = None
    try:
        if not args.reuse_server:
            server_proc = launch_server(args.host, args.port, model_path, args.uvicorn_log_level)
            wait_for_server(base_url)
        else:
            wait_for_server(base_url, timeout=10)

        # Generate AI suggestions once per capability (cached for all prompts)
        print("[bench] Generating AI suggestions for each capability...")
        cached_suggestions = {}
        # suggestions will be generated live per capability
        for capability in args.capabilities:
            if False:
                # Load fixed patches from file once
                p = Path(args.patches_file).expanduser().resolve()
                with p.open("r", encoding="utf-8") as fh:
                    import json as _json
                    suggestions = _json.load(fh)
                cached_suggestions[capability] = suggestions
                # fixed patches disabled
                print(f"[bench] → Using {len(suggestions)} patches from file for '{capability}'")
            else:
                print(f"[bench] → Generating AI suggestions for '{capability}'...")
                suggestions_resp = api_post(
                    base_url,
                    "/api/suggestions",
                    {"capability": capability},
                )
                suggestions = suggestions_resp.get("suggestions", [])
                cached_suggestions[capability] = suggestions
                # hashing disabled
                print(f"[bench] → Got {len(suggestions)} suggestions for '{capability}'")

                # Saving suggestions disabled in this mode

        print(f"[bench] Starting benchmark with {len(prompts)} prompts...")
        results: List[Dict[str, Optional[str]]] = []

        # Calibration state (loss mode only)
        calibrated = not (args.metric == "loss" and args.calibrate_loss)
        allow_patches = True
        calib_deltas: List[float] = []

        for item in prompts:
            print(f"[bench] Prompt {item.id}: {item.prompt[:60]}...")
            # Ensure clean state per prompt
            try:
                api_post(base_url, "/api/restore", {})
            except Exception:
                pass
            api_delete(base_url, "/api/hooks")

            baseline_output = None
            baseline_latency = 0.0
            baseline_loss: Optional[float] = None
            if args.metric == "accuracy":
                start = time.perf_counter()
                baseline = api_post(
                    base_url,
                    "/api/generate",
                    {"query": item.prompt, "config": args.config, "use_hooks": False},
                )
                baseline_latency = time.perf_counter() - start
                baseline_output = baseline.get("original")
            else:
                # loss metric: compute baseline loss once per prompt
                loss_resp = api_post(
                    base_url,
                    "/api/loss",
                    {
                        "prompt": item.prompt,
                        "answer": item.answer or "",
                        "answer_prefix": " #### ",
                        "use_chat_template": False,
                    },
                )
                baseline_loss = float(loss_resp.get("loss"))

            for capability in args.capabilities:
                api_delete(base_url, "/api/hooks")

                # Use cached suggestions instead of regenerating
                suggestions = cached_suggestions.get(capability, [])

                hook_resp = {"hook_count": 0}
                # If calibration disallowed patches already, skip applying patches
                hook_resp = {"hook_count": 0}
                if suggestions and allow_patches:
                    hook_resp = api_post(
                        base_url,
                        "/api/hooks",
                        {"suggestions": suggestions, "apply_mode": args.apply_mode},
                    )
                else:
                    if suggestions and not allow_patches:
                        print("[bench] Patches disabled by calibration; skipping application")
                    else:
                        print(f"[bench] No suggestions returned for capability '{capability}'")

                if args.metric == "accuracy":
                    start = time.perf_counter()
                    patched = api_post(
                        base_url,
                        "/api/generate",
                        {"query": item.prompt, "config": args.config, "use_hooks": True},
                    )
                    patched_latency = time.perf_counter() - start
                    match = evaluate(item.answer, patched.get("modified") or patched.get("original"))

                    results.append(
                        {
                            "prompt_id": item.id,
                            "capability": capability,
                            "prompt": item.prompt,
                            "answer": item.answer,
                            "baseline_output": baseline_output,
                            "patched_output": patched.get("modified"),
                            "baseline_latency_sec": f"{baseline_latency:.3f}",
                            "patched_latency_sec": f"{patched_latency:.3f}",
                            "hook_count": str(hook_resp.get("hook_count", 0)),
                            "suggestion_count": str(len(suggestions)),
                            "match": "" if match is None else str(match),
                        }
                    )
                else:
                    # loss metric path: compute patched loss
                    patched_loss_resp = api_post(
                        base_url,
                        "/api/loss",
                        {
                            "prompt": item.prompt,
                            "answer": item.answer or "",
                            "answer_prefix": " #### ",
                            "use_chat_template": False,
                        },
                    )
                    patched_loss = float(patched_loss_resp.get("loss"))
                    delta = patched_loss - (baseline_loss or 0.0)

                    # Update calibration window and decide once
                    if args.calibrate_loss and not calibrated:
                        calib_deltas.append(delta)
                        if len(calib_deltas) >= args.calibrate_loss_size:
                            mean_delta = sum(calib_deltas) / len(calib_deltas)
                            allow_patches = mean_delta < args.calibrate_loss_threshold
                            calibrated = True
                            decision = "ACCEPT" if allow_patches else "REJECT"
                            print(
                                f"[bench] Calibration complete over {len(calib_deltas)} prompts: mean_delta={mean_delta:.6f} → {decision} patches"
                            )

                    results.append(
                        {
                            "prompt_id": item.id,
                            "capability": capability,
                            "prompt": item.prompt,
                            "answer": item.answer,
                            "baseline_loss": f"{baseline_loss:.6f}" if baseline_loss is not None else "",
                            "patched_loss": f"{patched_loss:.6f}",
                            "loss_delta": f"{delta:.6f}",
                            "hook_count": str(hook_resp.get("hook_count", 0)),
                            "suggestion_count": str(len(suggestions)),
                            # suggestions hash removed
                            "calibrated": str(calibrated),
                            "allow_patches": str(allow_patches),
                        }
                    )

                # Restore weights and clear hooks after each capability run
                try:
                    api_post(base_url, "/api/restore", {})
                except Exception:
                    pass
                api_delete(base_url, "/api/hooks")

        csv_path = timestamped_path(output_dir, "results", "csv")
        json_path = timestamped_path(output_dir, "results", "json")

        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        with json_path.open("w", encoding="utf-8") as jsonfile:
            json.dump(results, jsonfile, indent=2)

        summary = summarize_results(results)
        print("\n=== Summary ===")
        for line in summary:
            print(line)
        summary_path = timestamped_path(output_dir, "summary", "txt")
        summary_path.write_text("\n".join(summary), encoding="utf-8")

    finally:
        if server_proc:
            print("[bench] Stopping uvicorn")
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def summarize_results(rows: List[Dict[str, Optional[str]]]) -> List[str]:
    lines = []
    total = len(rows)
    lines.append(f"Total evaluations: {total}")

    by_capability: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for row in rows:
        by_capability.setdefault(str(row["capability"]), []).append(row)

    for capability, entries in by_capability.items():
        if "baseline_loss" in entries[0]:
            # loss metric summary
            losses_base = [float(e["baseline_loss"]) for e in entries if e.get("baseline_loss")]
            losses_patch = [float(e["patched_loss"]) for e in entries if e.get("patched_loss")]
            deltas = [float(e["loss_delta"]) for e in entries if e.get("loss_delta")]
            n = min(len(losses_base), len(losses_patch))
            avg_base = sum(losses_base) / len(losses_base) if losses_base else 0.0
            avg_patch = sum(losses_patch) / len(losses_patch) if losses_patch else 0.0
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            avg_hooks = sum(int(e["hook_count"]) for e in entries) / len(entries)
            lines.append(
                f"Capability {capability}: avg_loss base={avg_base:.6f}, patched={avg_patch:.6f}, delta={avg_delta:.6f} (n={n}), avg hooks={avg_hooks:.2f}"
            )
        else:
            matches = [e["match"] == "True" for e in entries if e.get("match")]
            accuracy = (sum(matches) / len(matches) * 100) if matches else 0.0
            avg_baseline = sum(float(e["baseline_latency_sec"]) for e in entries) / len(entries)
            avg_patched = sum(float(e["patched_latency_sec"]) for e in entries) / len(entries)
            avg_hooks = sum(int(e["hook_count"]) for e in entries) / len(entries)
            lines.append(
                f"Capability {capability}: accuracy={accuracy:.1f}% (n={len(matches)}), "
                f"latency baseline={avg_baseline:.3f}s, patched={avg_patched:.3f}s, avg hooks={avg_hooks:.2f}"
            )

    return lines


if __name__ == "__main__":
    run_benchmark(parse_args())
