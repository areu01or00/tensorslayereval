#!/usr/bin/env python3
"""Tensor Inspector Module
Collects real tensor statistics from a loaded model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from safetensors import safe_open
from weight_patcher import ALLOWED_LINEAR_KEYS


@dataclass
class TensorStats:
    """Container for tensor statistics."""

    name: str
    shape: List[int]
    dtype: str
    device: str
    min: float
    max: float
    mean: float
    std: float
    zeros_percent: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "zeros_percent": self.zeros_percent,
        }


class TensorInspector:
    """Inspect model tensors using safetensors files to provide numpy-level stats."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model_dir = Path(getattr(model.config, "_name_or_path", ""))
        if not self.model_dir.exists():
            self.model_dir = Path(".")

        self.safetensor_files: List[Path] = sorted(self.model_dir.glob("*.safetensors"))
        if not self.safetensor_files:
            raise FileNotFoundError(f"No safetensors found in {self.model_dir}")

        self._tensor_to_shard: Dict[str, Path] = {}
        ordered_names: List[str] = []
        for shard in self.safetensor_files:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name not in self._tensor_to_shard:
                        self._tensor_to_shard[name] = shard
                        ordered_names.append(name)

        self._tensor_names = ordered_names
        self._stats_cache: Dict[str, TensorStats] = {}

    def sample_stats(
        self,
        capability: str = "general",
        sample_count: Optional[int] = None,
        only_linear_2d: bool = False,
    ) -> List[Dict[str, object]]:
        names = self._prioritized_names(capability)
        if only_linear_2d:
            # Keep .weight tensors that match allowed linear substrings
            names = [n for n in names if n.endswith(".weight") and any(k in n for k in ALLOWED_LINEAR_KEYS)]
        if sample_count is not None:
            names = names[:sample_count]
        stats: List[Dict[str, object]] = []

        for name in names:
            stat = self._stats_cache.get(name)
            if stat is None:
                tensor = self._load_tensor(name)
                if tensor is None:
                    continue
                # Enforce 2D if requested
                if only_linear_2d and tensor.ndim != 2:
                    continue
                stat = self._analyze_tensor(name, tensor)
                self._stats_cache[name] = stat
            stats.append(stat.to_dict())
        return stats

    def _prioritized_names(self, capability: str) -> List[str]:
        names = self._list_tensor_names()
        attn = [name for name in names if "attn" in name or "attention" in name]
        mlp = [name for name in names if "mlp" in name or "ffn" in name]
        embed = [name for name in names if "embed" in name and "position" not in name]

        capability = capability.lower()
        if capability == "math":
            primary = mlp + attn
        elif capability == "reasoning":
            primary = attn + mlp
        else:
            primary = attn + mlp + embed

        seen = set()
        ordered: List[str] = []
        for collection in (primary, names):
            for name in collection:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered

    def _list_tensor_names(self) -> List[str]:
        return self._tensor_names

    def _load_tensor(self, tensor_name: str) -> Optional[torch.Tensor]:
        shard = self._tensor_to_shard.get(tensor_name)
        if shard is None:
            return None
        with safe_open(shard, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
        return None

    def _analyze_tensor(self, name: str, tensor: torch.Tensor) -> TensorStats:
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        data = tensor.float()
        min_val = float(data.min())
        max_val = float(data.max())
        mean_val = float(data.mean())
        std_val = float(data.std(unbiased=False))
        zeros_percent = float((data == 0).sum().float() / data.numel() * 100)
        return TensorStats(
            name=name,
            shape=list(data.shape),
            dtype=str(data.dtype),
            device=str(data.device),
            min=min_val,
            max=max_val,
            mean=mean_val,
            std=std_val,
            zeros_percent=zeros_percent,
        )
