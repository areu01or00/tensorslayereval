#!/usr/bin/env python3
"""
Hook System Module
Handles activation hooks for tensor modifications during inference
Extracted from dynamic_tensor_analyzer.py
"""

import torch
from typing import Dict, List, Any, Optional, Tuple

class HookSystem:
    """Manages activation hooks for real-time tensor modifications"""
    
    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.model = model
        self.hooks = []
        self.modifications_by_module = {}
        self.hook_stats = {}
        
    def set_model(self, model: torch.nn.Module):
        """Set the model for hook registration"""
        self.clear_hooks()
        self.model = model
        
    def register_layer_hooks(self, modifications_by_module: Dict[str, List[Dict]]):
        """Register hooks on specified layers with modifications"""
        if not self.model:
            raise ValueError("No model set for hook registration")
            
        self.clear_hooks()
        self.modifications_by_module = modifications_by_module
        
        for module_name, modifications in modifications_by_module.items():
            try:
                # Get the module by name
                module = self._get_module_by_name(module_name)
                if module is None:
                    print(f"Warning: Module {module_name} not found")
                    continue
                
                # Create hook function for this module
                def create_hook(mods, module_key):
                    def hook_fn(module, input, output):
                        modified_output, activated = self._apply_modifications(output, mods)
                        if activated and module_key in self.hook_stats:
                            self.hook_stats[module_key]["activated"] = True
                        return modified_output
                    return hook_fn

                # Register the hook
                hook = module.register_forward_hook(create_hook(modifications, module_name))
                self.hooks.append(hook)
                
                # Track statistics
                self.hook_stats[module_name] = {
                    "modifications": len(modifications),
                    "activated": False
                }
                
            except Exception as e:
                print(f"Error registering hook for {module_name}: {e}")
        
        print(f"Registered {len(self.hooks)} hooks on model layers")
        
    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Get a module from the model by its name"""
        if not self.model:
            return None
            
        parts = name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Try to handle numeric indices (like layers.0)
                if part.isdigit() and hasattr(module, '__getitem__'):
                    module = module[int(part)]
                else:
                    return None
        
        return module
    
    def _apply_modifications(self, output: Any, modifications: List[Dict]) -> Tuple[Any, bool]:
        """Apply modifications to supported output types and return activation flag."""
        if not modifications:
            return output, False

        if isinstance(output, torch.Tensor):
            modified_output = output.clone()

            for mod in modifications:
                operation = mod.get("operation", "none")
                value = mod.get("value", 1.0)
                target = mod.get("target", "all")

                try:
                    if operation == "scale":
                        modified_output = self._apply_scale(modified_output, value, target)
                    elif operation == "add":
                        modified_output = self._apply_add(modified_output, value, target)
                    elif operation == "normalize":
                        modified_output = self._apply_normalize(modified_output)
                    elif operation == "clamp_max":
                        modified_output = torch.clamp(modified_output, max=value)
                    elif operation == "clamp_min":
                        modified_output = torch.clamp(modified_output, min=value)

                except Exception as e:
                    print(f"Error applying {operation}: {e}")

            return modified_output, True

        if isinstance(output, tuple):
            activated = False
            modified_items = []
            for item in output:
                if not activated and isinstance(item, torch.Tensor):
                    modified_item, activated = self._apply_modifications(item, modifications)
                    modified_items.append(modified_item)
                else:
                    modified_items.append(item)
            return tuple(modified_items), activated

        if isinstance(output, dict):
            modified_dict = dict(output)
            activated = False
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    modified_value, activated = self._apply_modifications(value, modifications)
                    modified_dict[key] = modified_value
                    break
            return modified_dict, activated

        # Unsupported type; return as-is
        return output, False
    
    def _apply_scale(self, tensor: torch.Tensor, value: float, target: str) -> torch.Tensor:
        """Apply scaling to tensor based on target"""
        if target == "all":
            return tensor * value
        
        # Handle percentage targets
        if "%" in target:
            cleaned = (
                target
                .replace("top", "")
                .replace("bottom", "")
                .replace("%", "")
                .replace("_", "")
                .strip()
            )
            try:
                percentage = float(cleaned)
            except ValueError:
                raise ValueError(f"Invalid target percentage: {target}")
            is_top = "top" in target
            
            # Flatten tensor for percentile calculation
            flat = tensor.flatten()
            k = int(len(flat) * (percentage / 100))
            
            if k > 0:
                if is_top:
                    threshold = torch.topk(flat.abs(), k).values[-1]
                    mask = tensor.abs() >= threshold
                else:
                    threshold = torch.topk(flat.abs(), len(flat) - k).values[-1]
                    mask = tensor.abs() <= threshold
                
                result = tensor.clone()
                result[mask] = result[mask] * value
                return result
        
        return tensor
    
    def _apply_add(self, tensor: torch.Tensor, value: float, target: str) -> torch.Tensor:
        """Apply addition to tensor based on target"""
        if target == "all":
            return tensor + value
        
        # Similar logic to scale for targeted addition
        if "%" in target:
            cleaned = (
                target
                .replace("top", "")
                .replace("bottom", "")
                .replace("%", "")
                .replace("_", "")
                .strip()
            )
            try:
                percentage = float(cleaned)
            except ValueError:
                raise ValueError(f"Invalid target percentage: {target}")
            is_top = "top" in target
            
            flat = tensor.flatten()
            k = int(len(flat) * (percentage / 100))
            
            if k > 0:
                if is_top:
                    threshold = torch.topk(flat.abs(), k).values[-1]
                    mask = tensor.abs() >= threshold
                else:
                    threshold = torch.topk(flat.abs(), len(flat) - k).values[-1]
                    mask = tensor.abs() <= threshold
                
                result = tensor.clone()
                result[mask] = result[mask] + value
                return result
        
        return tensor
    
    def _apply_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to tensor"""
        mean = tensor.mean()
        std = tensor.std()
        if std > 0:
            return (tensor - mean) / std
        return tensor
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.modifications_by_module = {}
        self.hook_stats = {}
        
    def get_active_modifications_count(self) -> int:
        """Get count of active modifications"""
        return sum(len(mods) for mods in self.modifications_by_module.values())
    
    def get_hook_stats(self) -> Dict[str, Any]:
        """Get statistics about registered hooks"""
        return {
            "total_hooks": len(self.hooks),
            "total_modifications": self.get_active_modifications_count(),
            "modules": list(self.modifications_by_module.keys()),
            "stats": self.hook_stats
        }
    
    def tensor_name_to_module_name(self, tensor_name: str) -> str:
        """Convert tensor name to module name for hook registration"""
        # Remove weight/bias suffix
        if tensor_name.endswith('.weight') or tensor_name.endswith('.bias'):
            base_name = '.'.join(tensor_name.split('.')[:-1])
        else:
            base_name = tensor_name
        
        # Handle attention modules
        if 'self_attn' in base_name and any(proj in base_name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            parts = base_name.split('.')
            for i, part in enumerate(parts):
                if part == 'self_attn':
                    # Return up to self_attn level
                    return '.'.join(parts[:i+1])
        
        # Handle MLP modules
        if 'mlp' in base_name and any(proj in base_name for proj in ['gate_proj', 'up_proj', 'down_proj']):
            parts = base_name.split('.')
            for i, part in enumerate(parts):
                if part == 'mlp':
                    # Return up to mlp level
                    return '.'.join(parts[:i+1])
        
        return base_name
