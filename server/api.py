#!/usr/bin/env python3
"""FastAPI service exposing inference workflow."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config_manager import ConfigManager
from hook_system import HookSystem
from model_loader import ModelLoader
from ai_agent import ai_agent
from weight_patcher import WeightPatcher
from tensor_inspector import TensorInspector

_config_manager = ConfigManager()
_model_loader = ModelLoader()
_hook_system = HookSystem()
_generation_lock = asyncio.Lock()
_hooks_active = False
_tensor_inspector: Optional[TensorInspector] = None
_weight_patcher: Optional[WeightPatcher] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model on startup, cleanup on shutdown."""
    model_path = os.environ.get("MODEL_PATH", "Qwen_0.6B")
    success = await asyncio.to_thread(_model_loader.load_model, model_path)
    if not success:
        raise RuntimeError(f"Failed to load model from '{model_path}'")
    _hook_system.set_model(_model_loader.model)
    global _tensor_inspector
    _tensor_inspector = TensorInspector(_model_loader.model)
    global _weight_patcher
    _weight_patcher = WeightPatcher(_model_loader.model)
    yield
    # Cleanup on shutdown (if needed)


app = FastAPI(title="Inference Engine API", version="0.1.0", lifespan=lifespan)


def _require_model_loaded() -> None:
    if _model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def _get_tensor_stats(capability: str) -> List[Dict[str, Any]]:
    if _tensor_inspector is not None:
        stats = _tensor_inspector.sample_stats(capability=capability)
        if stats:
            return stats
    return []


def _filter_suggestions(
    suggestions: List[Dict[str, Any]],
    max_count: int = 20,
    min_confidence: float = 0.85,
    min_count: int = 20,
) -> List[Dict[str, Any]]:
    """Keep highest-confidence suggestions, ensuring at least min_count are returned.

    We first take all entries >= min_confidence. If fewer than min_count survive,
    we top up from the remaining suggestions by descending confidence until we
    reach min_count (capped at max_count).
    """
    if not suggestions:
        return []

    def conf(entry: Dict[str, Any]) -> float:
        try:
            return float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    # Sort once by confidence desc
    ordered = sorted(suggestions, key=conf, reverse=True)

    # Take all that meet the threshold
    filtered = [s for s in ordered if conf(s) >= min_confidence]

    # Top-up from the remainder if we don't have enough
    if len(filtered) < min_count:
        remainder = [s for s in ordered if s not in filtered]
        needed = min(min_count, max_count) - len(filtered)
        if needed > 0:
            filtered.extend(remainder[:needed])

    # Enforce max_count cap
    return filtered[:max_count]


class GeneratePayload(BaseModel):
    query: str = Field(..., description="User prompt to send to the model")
    config: str = Field("balanced", description="Preset key from configuration manager")
    use_hooks: bool = Field(False, description="Whether to generate with currently active hooks")


class SuggestionPayload(BaseModel):
    capability: str = Field("general", description="Suggestion capability mode")


class Suggestion(BaseModel):
    tensor_name: str
    operation: str
    value: Optional[float] = None  # normalize operation doesn't need a value
    target: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


class ApplyHooksPayload(BaseModel):
    suggestions: List[Suggestion]
    apply_mode: Optional[str] = "weights"  # "weights" or "hooks"


@app.get("/api/health")
def healthcheck() -> Dict[str, Any]:
    """Simple liveness endpoint."""
    status = "online" if _model_loader.model is not None else "initialising"
    return {
        "status": status,
        "device": _model_loader.device,
        "hooks_active": _hooks_active,
    }


@app.get("/api/configs")
def list_configs() -> Dict[str, Any]:
    """Return all configuration presets."""
    return {name: cfg.to_dict() for name, cfg in _config_manager.get_all_presets().items()}


@app.post("/api/generate")
async def generate(payload: GeneratePayload) -> Dict[str, Any]:
    """Generate responses for the supplied query."""
    _require_model_loaded()

    config = _config_manager.get_preset(payload.config)
    params = dict(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_new_tokens=config.max_new_tokens,
        repetition_penalty=config.repetition_penalty,
        do_sample=config.do_sample,
    )

    async with _generation_lock:
        original_response = await asyncio.to_thread(
            _model_loader.generate_response,
            payload.query,
            **params,
        )

        modified_response: Optional[str] = None
        hook_count = 0

        if payload.use_hooks and _hooks_active:
            hook_count = _hook_system.get_active_modifications_count()
            modified_response = await asyncio.to_thread(
                _model_loader.generate_response,
                payload.query,
                **params,
            )

    return {
        "original": original_response,
        "modified": modified_response,
        "hook_count": hook_count,
    }


@app.post("/api/suggestions")
async def get_suggestions(payload: SuggestionPayload) -> Dict[str, Any]:
    """Return AI generated tensor modification suggestions."""
    tensor_stats = _get_tensor_stats(payload.capability)
    if not tensor_stats:
        return {"suggestions": []}

    suggestions = await asyncio.to_thread(
        ai_agent.generate_modifications,
        tensor_stats,
        payload.capability,
    )
    filtered = _filter_suggestions(suggestions)
    return {"suggestions": filtered}


@app.post("/api/hooks")
def apply_hooks(payload: ApplyHooksPayload) -> Dict[str, Any]:
    """Register hooks using provided suggestions."""
    global _hooks_active

    if not payload.suggestions:
        raise HTTPException(status_code=400, detail="No suggestions supplied")

    mode = (payload.apply_mode or "weights").lower()
    if mode not in ("weights", "hooks"):
        mode = "weights"

    if mode == "weights":
        if _weight_patcher is None:
            raise HTTPException(status_code=500, detail="Weight patcher not initialized")
        applied = 0
        applied_tensors: List[str] = []
        for s in payload.suggestions:
            name = s.tensor_name
            op = s.operation
            tgt = s.target
            try:
                val = float(s.value) if s.value is not None else 0.0
            except Exception:
                val = 0.0
            count = _weight_patcher.apply(name, op, val, tgt)
            if count > 0:
                applied += 1
                applied_tensors.append(name)
        _hooks_active = applied > 0
        return {
            "hook_count": applied,  # semantic: number of suggestions applied
            "applied_tensors": applied_tensors,
            "mode": mode,
        }
    else:
        # legacy hooks mode
        modifications_by_module: Dict[str, List[Dict[str, Any]]] = {}
        for suggestion in payload.suggestions:
            module_name = _hook_system.tensor_name_to_module_name(suggestion.tensor_name)
            modifications_by_module.setdefault(module_name, []).append(suggestion.model_dump())
        _hook_system.register_layer_hooks(modifications_by_module)
        hook_count = _hook_system.get_active_modifications_count()
        _hooks_active = hook_count > 0
        return {
            "hook_count": hook_count,
            "modules": list(modifications_by_module.keys()),
            "mode": mode,
        }


@app.delete("/api/hooks")
def clear_hooks() -> Dict[str, Any]:
    """Remove all registered hooks."""
    global _hooks_active
    _hook_system.clear_hooks()
    _hooks_active = False
    return {"hook_count": 0}


@app.post("/api/restore")
def restore_weights() -> Dict[str, Any]:
    """Restore all in-memory weight patches to original values."""
    if _weight_patcher is None:
        raise HTTPException(status_code=500, detail="Weight patcher not initialized")
    restored = _weight_patcher.restore_all()
    # also clear legacy hooks, just in case
    _hook_system.clear_hooks()
    return {"restored_params": restored}


@app.get("/api/hooks")
def hooks_status() -> Dict[str, Any]:
    """Return hook statistics."""
    return {
        "active": _hooks_active,
        "stats": _hook_system.get_hook_stats(),
    }


@app.get("/api/history")
def history_placeholder() -> Dict[str, Any]:
    """Placeholder endpoint for history integration (client maintains history)."""
    return {"entries": []}
