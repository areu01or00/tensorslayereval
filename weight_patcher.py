#!/usr/bin/env python3
"""In-memory weight patcher akin to boilerplate_v2's safetensors patcher.

Applies scale/add/clamp_max/clamp_min to slices of 2D weight tensors (Linear weights)
selected by targets: all, top {p}%, bottom {p}% (by absolute value).

Stores original copies per parameter to support full restore.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch


ALLOWED_LINEAR_KEYS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


class WeightPatcher:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        # Save original full tensors once per parameter name
        self._originals: Dict[str, torch.Tensor] = {}

    def _get_param(self, name: str) -> torch.nn.Parameter | None:
        for n, p in self.model.named_parameters():
            if n == name:
                return p
        return None

    def _is_allowed(self, name: str, param: torch.nn.Parameter) -> bool:
        if not name.endswith(".weight"):
            return False
        if param.ndim != 2:
            return False
        return any(k in name for k in ALLOWED_LINEAR_KEYS)

    def _mask_from_target(self, tensor: torch.Tensor, target: str) -> torch.Tensor:
        target = target.strip().lower().replace("_", " ")
        if target == "all":
            return torch.ones_like(tensor, dtype=torch.bool)
        if "%" in target:
            # parse like "top 10%" or "bottom 20%"
            is_top = "top" in target
            is_bottom = "bottom" in target
            pct_str = target.replace("top", "").replace("bottom", "").replace("%", "").strip()
            pct = float(pct_str) / 100.0
            flat = tensor.detach().abs().reshape(-1)
            k = int(max(1, round(flat.numel() * pct)))
            if k >= flat.numel():
                return torch.ones_like(tensor, dtype=torch.bool)
            if is_top:
                # threshold at top-k absolute
                vals, _ = torch.topk(flat, k)
                thr = vals.min()
                mask = tensor.abs() >= thr
                return mask
            if is_bottom:
                # bottom-k: smallest absolute values
                vals, _ = torch.topk(-flat, k)  # negative top-k = smallest
                thr = (-vals).max()
                mask = tensor.abs() <= thr
                return mask
        # default: no mask
        return torch.zeros_like(tensor, dtype=torch.bool)

    def apply(self, name: str, operation: str, value: float, target: str) -> int:
        param = self._get_param(name)
        if param is None:
            return 0
        if not self._is_allowed(name, param):
            return 0
        if name not in self._originals:
            self._originals[name] = param.detach().clone()
        with torch.no_grad():
            mask = self._mask_from_target(param.data, target)
            count = int(mask.sum().item())
            if count == 0:
                return 0
            if operation == "scale":
                param.data[mask] = param.data[mask] * float(value)
            elif operation == "add":
                param.data[mask] = param.data[mask] + float(value)
            elif operation == "clamp_max":
                param.data[mask] = torch.clamp(param.data[mask], max=float(value))
            elif operation == "clamp_min":
                param.data[mask] = torch.clamp(param.data[mask], min=float(value))
            else:
                # unsupported op
                return 0
            return count

    def restore_all(self) -> int:
        restored = 0
        with torch.no_grad():
            for name, orig in self._originals.items():
                p = self._get_param(name)
                if p is not None and p.shape == orig.shape:
                    p.data.copy_(orig)
                    restored += 1
        self._originals.clear()
        return restored

