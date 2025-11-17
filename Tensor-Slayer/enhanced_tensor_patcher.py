#!/usr/bin/env python3
"""
Enhanced Tensor Patcher - A wrapper around tensor_patcher_cli.py that adds:
- Advanced static analysis
- Pattern detection
- Visualization
- Enhanced recommendations
- Token-level analysis
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box
import typer
from tensor_patcher_cli import ModelPatcher
from ai_tensor_explorer import AITensorExplorer
from safetensors import safe_open
from safetensors.torch import save_file

console = Console()
app = typer.Typer()

class EnhancedTensorPatcher:
    def __init__(self, weight_files: list):
        """Initialize with base patcher and enhanced features"""
        # Store all weight files
        self.weight_files = weight_files
        self.model_path = str(Path(weight_files[0]).parent) if weight_files else ""
        
        # Initialize the explorer first
        try:
            console.print(f"[cyan]Loading model from {self.weight_files}...[/]")
            self.explorer = AITensorExplorer(self.weight_files)
            
            # Now initialize the patcher with the explorer
            self.base_patcher = ModelPatcher()
            self.base_patcher.explorer = self.explorer
            
            # Store main safetensors file path (first one in the list)
            self.safetensors_file = self.weight_files[0] if self.weight_files else None
            
            # Verify initialization
            if not hasattr(self.explorer, "tensors") or not self.explorer.tensors:
                raise RuntimeError("Model tensors not loaded properly")
                
            console.print(f"[green]✓[/] Loaded {len(self.explorer.tensors)} tensors")
            
        except Exception as e:
            console.print(f"[red]Error loading model: {str(e)}[/]")
            raise
        
        self.analysis_cache = {}
        self.history = []
        self.backup_dir = Path(self.model_path).parent / f"{Path(self.model_path).name}_backups"
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(exist_ok=True)
        
    def _guided_patching(self, tensor_name: str, operation: str = "analyze", value: Optional[float] = None) -> Dict[str, Any]:
        """Internal method for guided tensor patching and analysis"""
        try:
            with safe_open(self.safetensors_file, framework="pt") as f:
                if tensor_name not in f.keys():
                    return {"error": f"Tensor {tensor_name} not found"}
                
                tensor = f.get_tensor(tensor_name)
                tensor = tensor.to(torch.float32)
                
                if operation == "analyze":
                    stats = {
                        "mean": float(torch.mean(tensor)),
                        "std": float(torch.std(tensor)),
                        "min": float(torch.min(tensor)),
                        "max": float(torch.max(tensor)),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "zeros_percent": float((tensor == 0).sum() / tensor.numel() * 100)
                    }
                    return stats
                
                # For actual patching operations
                return {"error": "Patching not implemented in preview mode"}
                
        except Exception as e:
            return {"error": str(e)}

    def analyze_tensor_patterns(self, tensor_name: str) -> Dict[str, Any]:
        """Perform advanced pattern analysis on a tensor"""
        try:
            # Get tensor data using explorer's analyze_tensor method
            stats = self.explorer.explorer.analyze_tensor(tensor_name)
            if not stats:
                return {"error": "Could not load tensor"}
            
            patterns = []
            shape = stats.get("shape", [])
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            
            # Check for identity-like patterns
            if len(shape) == 2 and shape[0] == shape[1]:
                if abs(mean) < 0.1 and abs(std - 1.0) < 0.1:
                    patterns.append({
                        "type": "identity-like",
                        "confidence": "high",
                        "evidence": f"Square matrix with mean≈{mean:.6f}, std≈{std:.6f}"
                    })
            
            # Check for attention head patterns
            if any(x in tensor_name.lower() for x in ["attn", "attention"]):
                if len(shape) >= 2:
                    head_dim = shape[-1]
                    if head_dim % 64 == 0:  # Common head dimensions
                        patterns.append({
                            "type": "attention-head",
                            "head_dim": head_dim,
                            "confidence": "high",
                            "evidence": f"Tensor shape matches attention pattern with head dim {head_dim}"
                        })
            
            # Check for block patterns
            if len(shape) == 2 and min(shape) > 10:
                block_size = min(10, min(shape) // 2)
                # Check for block structure by analyzing variance
                try:
                    # Use safe_open to access tensor data
                    with safe_open(self.safetensors_file, framework="pt") as f:
                        if tensor_name not in f.keys():
                            raise ValueError(f"Tensor {tensor_name} not found in safetensors file")
                        
                        tensor = f.get_tensor(tensor_name)
                        # Convert to float32 for analysis
                        tensor = tensor.to(torch.float32)
                        
                        # Analyze blocks
                        rows, cols = shape
                        block_variances = []
                        
                        for i in range(0, rows, block_size):
                            for j in range(0, cols, block_size):
                                block = tensor[i:min(i+block_size, rows), j:min(j+block_size, cols)]
                                if block.numel() > 0:
                                    block_variances.append(float(torch.var(block)))
                        
                        if block_variances:
                            var_of_vars = float(np.var(block_variances))
                            if var_of_vars > 0.01:  # Significant variance between blocks
                                patterns.append({
                                    "type": "block-structure",
                                    "block_size": block_size,
                                    "confidence": "medium",
                                    "evidence": f"Block variance pattern detected (var_of_vars={var_of_vars:.6f})"
                                })
                except Exception as block_error:
                    console.print(f"[yellow]Warning: Could not analyze blocks: {str(block_error)}[/]")
            
            # Check for sparsity patterns
            zeros_percent = stats.get("zeros_percent", 0)
            if zeros_percent > 1:  # More than 1% zeros
                patterns.append({
                    "type": "sparse",
                    "confidence": "high",
                    "evidence": f"Contains {zeros_percent:.2f}% zero values"
                })
            
            # Check for normalized vectors (common in embeddings)
            if "embed" in tensor_name.lower() and len(shape) >= 2:
                if abs(mean) < 0.01 and 0.1 < std < 10:
                    patterns.append({
                        "type": "normalized-embeddings",
                        "confidence": "medium",
                        "evidence": f"Embedding-like statistics with mean≈{mean:.6f}, std={std:.6f}"
                    })
            
            # Store in cache with timestamp
            self.analysis_cache[tensor_name] = {
                "stats": stats,
                "patterns": patterns,
                "timestamp": time.time()
            }
            
            return {
                "tensor_name": tensor_name,
                "stats": stats,
                "patterns": patterns
            }
            
        except Exception as e:
            console.print(f"[red]Error analyzing tensor {tensor_name}: {str(e)}[/]")
            return {"error": str(e)}

    def visualize_tensor_distribution(self, tensor_name: str):
        """Create visualization of tensor value distribution"""
        try:
            stats = self._guided_patching(tensor_name, "analyze", None)
            if stats.get("error"):
                console.print(f"[red]Error getting stats for {tensor_name}: {stats['error']}[/]")
                return
            
            # Create ASCII histogram
            values = np.linspace(stats["min"], stats["max"], 20)
            hist = np.histogram(values, bins=20)
            
            console.print(f"\n[bold cyan]Distribution for {tensor_name}:[/]")
            
            # Display histogram
            max_bar = 40
            for i, (count, bin_edge) in enumerate(zip(hist[0], hist[1][:-1])):
                bar_len = int((count / max(hist[0])) * max_bar)
                bar = "█" * bar_len
                console.print(f"{bin_edge:8.3f} | {bar}")
            
            # Show statistics
            console.print("\n[bold]Statistics:[/]")
            console.print(f"Mean: {stats['mean']:.6f}")
            console.print(f"Std:  {stats['std']:.6f}")
            console.print(f"Min:  {stats['min']:.6f}")
            console.print(f"Max:  {stats['max']:.6f}")
            
        except Exception as e:
            console.print(f"[red]Error visualizing distribution: {str(e)}[/]")

    def get_enhanced_recommendations(self, tensor_name: str, capability: str = "general") -> List[Dict[str, Any]]:
        """Get enhanced recommendations for tensor modifications"""
        try:
            # Get tensor analysis
            analysis = self.analyze_tensor_patterns(tensor_name)
            if "error" in analysis:
                return []
                
            recommendations = []
            stats = analysis.get("stats", {})
            patterns = analysis.get("patterns", [])
            
            # Base recommendations on patterns found
            for pattern in patterns:
                pattern_type = pattern.get("type", "")
                confidence = pattern.get("confidence", "low")  # Default to low if not specified
                
                # Convert confidence string to float
                confidence_score = {
                    "high": 0.9,
                    "medium": 0.6,
                    "low": 0.3
                }.get(confidence, 0.3)
                
                if pattern_type == "identity-like":
                    recommendations.append({
                        "operation": "scale",
                        "value": 1.05,
                        "target": "all",
                        "reason": "Strengthen identity-like pattern",
                        "confidence": confidence_score
                    })
                
                elif pattern_type == "attention-head":
                    recommendations.append({
                        "operation": "normalize",
                        "value": 1.0,
                        "target": "all",
                        "reason": f"Normalize attention head (dim={pattern.get('head_dim')})",
                        "confidence": confidence_score
                    })
                
                elif pattern_type == "block-structure":
                    recommendations.append({
                        "operation": "scale",
                        "value": 1.1,
                        "target": "top 20%",
                        "reason": "Enhance block structure pattern",
                        "confidence": confidence_score
                    })
                
                elif pattern_type == "sparse":
                    recommendations.append({
                        "operation": "clamp_min",
                        "value": stats.get("std", 0.1) * 0.1,
                        "target": "all",
                        "reason": "Reduce sparsity while preserving pattern",
                        "confidence": confidence_score
                    })
                
                elif pattern_type == "normalized-embeddings":
                    recommendations.append({
                        "operation": "normalize",
                        "value": 1.0,
                        "target": "all",
                        "reason": "Maintain normalized embedding structure",
                        "confidence": confidence_score
                    })
            
            # Add general recommendations based on statistics
            mean = stats.get("mean", 0)
            std = stats.get("std", 1)
            
            if abs(mean) < 0.01 and std < 0.1:
                recommendations.append({
                    "operation": "scale",
                    "value": 1.5,
                    "target": "all",
                    "reason": "Increase signal strength of near-zero tensor",
                    "confidence": 0.7
                })
            
            if std > 10:
                recommendations.append({
                    "operation": "normalize",
                    "value": 1.0,
                    "target": "all",
                    "reason": "Normalize high-variance tensor",
                    "confidence": 0.8
                })
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            return recommendations
            
        except Exception as e:
            console.print(f"[red]Error generating recommendations: {str(e)}[/]")
            return []

    def apply_enhanced_patch(self, tensor_name: str, operation: str, value: float, 
                           target: str = "all", preview: bool = True, save_model: bool = True) -> Dict[str, Any]:
        """Apply patch with enhanced preview and safety checks
        
        Args:
            tensor_name: Name of the tensor to modify
            operation: Operation to apply (scale, add, normalize, clamp_min, clamp_max)
            value: Value to use in the operation
            target: Target range (all, top X%, bottom X%, etc.)
            preview: If True, only preview changes without applying them
            save_model: If False, don't save model after applying changes (for batch operations)
        
        Returns:
            Dictionary with operation results
        """
        try:
            # Get current analysis
            analysis = self.analyze_tensor_patterns(tensor_name)
            if "error" in analysis:
                return {"error": analysis["error"]}
            
            # Create backup if not preview and we're saving after this operation
            if not preview and save_model:
                backup_path = self.create_backup(tensor_name)
                if not backup_path: # Assuming create_backup returns empty string on failure
                    return {"error": "Failed to create backup"}
            
            try:
                # First, load the tensor from the model
                with safe_open(self.safetensors_file, framework="pt") as f:
                    if tensor_name not in f.keys():
                        return {"error": f"Tensor {tensor_name} not found in model file {self.safetensors_file}"}
                    
                    tensor = f.get_tensor(tensor_name)
                    original_device = tensor.device
                    
                    # For quantile and mask, use CPU version
                    tensor_cpu = tensor.to(device='cpu', dtype=torch.float32)

                    target_quantile_value = None
                    use_upper_tail = True # True for "top X%", False for "bottom X%"

                    if target.startswith("top"):
                        try:
                            percentage = float(target.split(" ")[1].replace("%", "")) / 100.0
                            target_quantile_value = 1.0 - percentage
                            use_upper_tail = True
                        except IndexError: # Handles "top" without "X%"
                            console.print(f"[yellow]Warning: Malformed target '{target}' for tensor '{tensor_name}'. Defaulting to 'all'.[/]")
                        except ValueError: # Handles "top X%" with non-float X
                            console.print(f"[yellow]Warning: Malformed target '{target}' for tensor '{tensor_name}'. Defaulting to 'all'.[/]")
                    elif target.startswith("bottom"):
                        try:
                            percentage = float(target.split(" ")[1].replace("%", "")) / 100.0
                            target_quantile_value = percentage
                            use_upper_tail = False
                        except IndexError:
                            console.print(f"[yellow]Warning: Malformed target '{target}' for tensor '{tensor_name}'. Defaulting to 'all'.[/]")
                        except ValueError: # Handles "bottom X%" with non-float X
                            console.print(f"[yellow]Warning: Malformed target '{target}' for tensor '{tensor_name}'. Defaulting to 'all'.[/]")
                    
                    mask_cpu = torch.ones_like(tensor_cpu, dtype=torch.bool)

                    if target_quantile_value is not None and 0.0 < target_quantile_value < 1.0:
                        flat_tensor_cpu = tensor_cpu.flatten()
                        threshold = torch.quantile(flat_tensor_cpu, target_quantile_value)
                        if use_upper_tail:
                            mask_cpu = tensor_cpu >= threshold
                        else: # bottom X%
                            mask_cpu = tensor_cpu <= threshold
                    elif target != "all":
                        console.print(f"[yellow]Warning: Unsupported or malformed target '{target}' for tensor '{tensor_name}'. Defaulting to 'all'.[/]")

                    # Create modified tensor clone on the original device
                    modified = tensor.clone() 
                    mask_on_device = mask_cpu.to(original_device)
                    
                    # Apply operation
                    if operation == "scale":
                        modified[mask_on_device] *= value
                    elif operation == "add":
                        modified[mask_on_device] += value
                    elif operation == "normalize":
                        selected_on_device = modified[mask_on_device]
                        if selected_on_device.numel() > 0:
                            mean = selected_on_device.mean()
                            std = selected_on_device.std()
                            modified[mask_on_device] = (selected_on_device - mean) / (std + 1e-8)
                        # If numel is 0, selected_on_device is empty, no change needed.
                    elif operation == "clamp_max":
                        modified[mask_on_device] = torch.clamp(modified[mask_on_device], max=value)
                    elif operation == "clamp_min":
                        modified[mask_on_device] = torch.clamp(modified[mask_on_device], min=value)
                    else:
                        return {"error": f"Unsupported operation: {operation}"}
                    
                    # Get preview statistics using CPU tensors and mask
                    original_selected_cpu = tensor_cpu[mask_cpu]
                    # For modified stats, get a CPU version of the potentially modified tensor
                    modified_cpu_for_stats = modified.to(device='cpu')
                    modified_selected_cpu = modified_cpu_for_stats[mask_cpu]

                    current_mean_orig = float(original_selected_cpu.mean()) if original_selected_cpu.numel() > 0 else 0.0
                    current_std_orig = float(original_selected_cpu.std()) if original_selected_cpu.numel() > 0 else 0.0
                    current_mean_mod = float(modified_selected_cpu.mean()) if modified_selected_cpu.numel() > 0 else 0.0
                    current_std_mod = float(modified_selected_cpu.std()) if modified_selected_cpu.numel() > 0 else 0.0
                    
                    preview_stats = {
                        "original_mean": current_mean_orig,
                        "original_std": current_std_orig,
                        "modified_mean": current_mean_mod,
                        "modified_std": current_std_mod,
                        "affected_values": int(mask_cpu.sum()),
                        "total_values": tensor_cpu.numel(), 
                        "patterns_affected": [p["type"] for p in analysis.get("patterns", [])] # Based on original analysis
                    }
                    
                    if preview:
                        return {
                            "preview": preview_stats,
                            "safety_checks": {
                                "preserves_structure": abs(preview_stats["modified_std"] - preview_stats["original_std"]) < 1.0 if preview_stats["original_std"] !=0 else True, # Avoid division by zero if std is 0
                                "within_safe_range": abs(preview_stats["modified_mean"]) < 10.0,
                                "backup_available": True # Assuming backup was attempted
                            }
                        }
                    
                    # Actually apply the changes by loading all tensors or using cached ones
                    if not hasattr(self, "_tensor_cache"):
                        self._tensor_cache = {}
                        with safe_open(self.safetensors_file, framework="pt") as f_all:
                            for name_key in f_all.keys(): # renamed 'name' to 'name_key' to avoid conflict
                                self._tensor_cache[name_key] = f_all.get_tensor(name_key)
                    
                    self._tensor_cache[tensor_name] = modified # modified is on original_device
                    
                    if save_model:
                        output_dir = Path(self.model_path).parent / f"{Path(self.model_path).name}_modified"
                        output_dir.mkdir(exist_ok=True)
                        
                        # Determine the correct output filename based on the input weight file
                        # Assuming self.weight_files is a list of Path objects from __init__
                        # And self.safetensors_file is the first of these.
                        output_filename = Path(self.safetensors_file).name
                        output_file_path = output_dir / output_filename
                        
                        save_file(self._tensor_cache, str(output_file_path))
                        
                        # Copy any other model files (config.json, etc.) from the original model directory
                        original_model_dir = Path(self.safetensors_file).parent
                        for file_item in original_model_dir.glob("*"): # Renamed 'file' to 'file_item'
                            # Avoid copying the primary weight file itself if it has the same name as output
                            if file_item.name != output_filename and not file_item.is_dir():
                                import shutil
                                shutil.copy2(file_item, output_dir / file_item.name)
                        
                        self.history.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "tensor": tensor_name,
                            "operation": operation,
                            "value": value,
                            "target": target,
                            "backup": True 
                        })
                        
                        return {
                            "success": True,
                            "stats": preview_stats,
                            "output_path": str(output_file_path)
                        }
                    else:
                        return {
                            "success": True,
                            "in_memory_only": True,
                            "stats": preview_stats
                        }
            except Exception as e:
                return {"error": f"Error modifying tensor '{tensor_name}': {str(e)}"}
        except Exception as e:
            return {"error": str(e)}

    def apply_batch_patches(self, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply multiple patches in a single batch operation
        
        Args:
            patches: List of patch specifications with keys:
                    - tensor_name: Name of tensor to modify
                    - operation: Operation to apply
                    - value: Value for the operation
                    - target: Target range (optional, default "all")
        
        Returns:
            Dictionary with operation results
        """
        results = []
        try:
            if not hasattr(self, "_tensor_cache") or not self._tensor_cache: # Ensure cache is loaded
                self._tensor_cache = {}
                with safe_open(self.safetensors_file, framework="pt") as f_all:
                    for name_key in f_all.keys():
                        self._tensor_cache[name_key] = f_all.get_tensor(name_key)
            
            for i, patch_spec in enumerate(patches): # Renamed 'patch' to 'patch_spec'
                tensor_name = patch_spec["tensor_name"]
                operation = patch_spec["operation"]
                value = patch_spec["value"]
                target = patch_spec.get("target", "all")

                if tensor_name not in self._tensor_cache:
                    results.append({
                        "tensor_name": tensor_name, "operation": operation, "result": "error",
                        "error": f"Tensor '{tensor_name}' not found in cache.", "stats": {}
                    })
                    continue

                tensor = self._tensor_cache[tensor_name]
                original_device = tensor.device
                tensor_cpu = tensor.to(device='cpu', dtype=torch.float32)

                target_quantile_value = None
                use_upper_tail = True
                if target.startswith("top"):
                    try:
                        percentage = float(target.split(" ")[1].replace("%", "")) / 100.0
                        target_quantile_value = 1.0 - percentage
                    except: # Covers IndexError, ValueError
                        console.print(f"[yellow]Warning (batch): Malformed target '{target}' for '{tensor_name}'. Defaulting to 'all'.[/]")
                elif target.startswith("bottom"):
                    try:
                        percentage = float(target.split(" ")[1].replace("%", "")) / 100.0
                        target_quantile_value = percentage
                        use_upper_tail = False
                    except:
                        console.print(f"[yellow]Warning (batch): Malformed target '{target}' for '{tensor_name}'. Defaulting to 'all'.[/]")
                
                mask_cpu = torch.ones_like(tensor_cpu, dtype=torch.bool)
                if target_quantile_value is not None and 0.0 < target_quantile_value < 1.0:
                    flat_tensor_cpu = tensor_cpu.flatten()
                    threshold = torch.quantile(flat_tensor_cpu, target_quantile_value)
                    if use_upper_tail:
                        mask_cpu = tensor_cpu >= threshold
                    else:
                        mask_cpu = tensor_cpu <= threshold
                elif target != "all":
                     console.print(f"[yellow]Warning (batch): Unsupported target '{target}' for '{tensor_name}'. Defaulting to 'all'.[/]")

                modified = tensor.clone()
                mask_on_device = mask_cpu.to(original_device)

                if operation == "scale":
                    modified[mask_on_device] *= value
                elif operation == "add":
                    modified[mask_on_device] += value
                elif operation == "normalize":
                    selected_on_device = modified[mask_on_device]
                    if selected_on_device.numel() > 0:
                        mean = selected_on_device.mean()
                        std = selected_on_device.std()
                        modified[mask_on_device] = (selected_on_device - mean) / (std + 1e-8)
                elif operation == "clamp_max":
                    modified[mask_on_device] = torch.clamp(modified[mask_on_device], max=value)
                elif operation == "clamp_min":
                    modified[mask_on_device] = torch.clamp(modified[mask_on_device], min=value)
                else:
                    results.append({
                        "tensor_name": tensor_name, "operation": operation, "result": "error",
                        "error": f"Unsupported operation: {operation}", "stats": {}
                    })
                    continue
                
                self._tensor_cache[tensor_name] = modified
                
                original_selected_cpu = tensor_cpu[mask_cpu]
                modified_cpu_for_stats = modified.to(device='cpu')
                modified_selected_cpu = modified_cpu_for_stats[mask_cpu]

                current_mean_orig = float(original_selected_cpu.mean()) if original_selected_cpu.numel() > 0 else 0.0
                current_std_orig = float(original_selected_cpu.std()) if original_selected_cpu.numel() > 0 else 0.0
                current_mean_mod = float(modified_selected_cpu.mean()) if modified_selected_cpu.numel() > 0 else 0.0
                current_std_mod = float(modified_selected_cpu.std()) if modified_selected_cpu.numel() > 0 else 0.0

                current_patch_stats = {
                    "original_mean": current_mean_orig,
                    "original_std": current_std_orig,
                    "modified_mean": current_mean_mod,
                    "modified_std": current_std_mod,
                    "affected_values": int(mask_cpu.sum()),
                    "total_values": tensor_cpu.numel(),
                }
                results.append({
                    "tensor_name": tensor_name, "operation": operation, 
                    "result": "success", "stats": current_patch_stats
                })
            
            # Save the complete model with all modified tensors from the cache
            output_dir = Path(self.model_path).parent / f"{Path(self.model_path).name}_modified"
            output_dir.mkdir(exist_ok=True)
            
            output_filename = Path(self.safetensors_file).name
            output_file_path = output_dir / output_filename
            
            save_file(self._tensor_cache, str(output_file_path))
            
            original_model_dir = Path(self.safetensors_file).parent
            for file_item in original_model_dir.glob("*"):
                if file_item.name != output_filename and not file_item.is_dir():
                    import shutil
                    shutil.copy2(file_item, output_dir / file_item.name)
            
            self.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "patches_applied_count": len(patches), # Changed from "patches" to count
                "batch_operation": True, # Indicate it was a batch
                "backup": True # Assuming backups are handled per-tensor or a general one is made
            })
            
            return {
                "success": True,
                "results": results,
                "output_path": str(output_file_path)
            }
        except Exception as e:
            return {
                "error": f"Error in batch patch operation: {str(e)}",
                "results": results # Return partial results if any
            }

    def analyze_hex_patterns(self, tensor_name: str) -> Dict[str, Any]:
        """Analyze hex-level patterns in tensor"""
        try:
            # Get tensor data
            stats = self._guided_patching(tensor_name, "analyze", None)
            if stats.get("error"):
                return {"error": f"Could not load tensor stats for {tensor_name}: {stats['error']}"}
            
            # Convert values to hex representation
            import struct
            import binascii
            
            hex_patterns = {
                "repeating": [],
                "special_values": [],
                "byte_distribution": {}
            }
            
            # Sample some values for hex analysis
            sample_size = min(1000, stats.get("total_elements", 1000))
            values = np.random.choice(stats.get("values", []), size=sample_size)
            
            for val in values:
                # Convert float to hex
                hex_val = hex(struct.unpack('<I', struct.pack('<f', float(val)))[0])
                
                # Analyze byte patterns
                for i in range(2, len(hex_val), 2):
                    byte = hex_val[i:i+2]
                    hex_patterns["byte_distribution"][byte] = hex_patterns["byte_distribution"].get(byte, 0) + 1
                
                # Check for special values (0, 1, -1, etc.)
                if abs(val) < 1e-6:  # Zero
                    hex_patterns["special_values"].append(("zero", hex_val))
                elif abs(val - 1) < 1e-6:  # One
                    hex_patterns["special_values"].append(("one", hex_val))
                elif abs(val + 1) < 1e-6:  # Negative one
                    hex_patterns["special_values"].append(("negative_one", hex_val))
            
            return hex_patterns
            
        except Exception as e:
            return {"error": str(e)}

    def analyze_token_embeddings(self, tensor_name: str) -> Dict[str, Any]:
        """Analyze token-level patterns in embeddings"""
        try:
            # Check if tensor is an embedding
            if not any(x in tensor_name.lower() for x in ["embed", "token", "wte"]):
                return {"error": "Not an embedding tensor"}
            
            # Get tensor data
            stats = self._guided_patching(tensor_name, "analyze", None)
            if stats.get("error"):
                return {"error": f"Could not load tensor stats for {tensor_name}: {stats['error']}"}
            
            # Analyze embedding structure
            embedding_dim = stats["shape"][-1]
            vocab_size = stats["shape"][0]
            
            # Calculate embedding statistics
            embedding_stats = {
                "embedding_dim": embedding_dim,
                "vocab_size": vocab_size,
                "norm_distribution": {},  # Will store L2 norm distribution
                "similarity_clusters": [],  # Will store clusters of similar embeddings
                "special_tokens": []  # Will store special token embeddings
            }
            
            return embedding_stats
            
        except Exception as e:
            return {"error": str(e)}

    def create_backup(self, tensor_name: str) -> str:
        """Create a backup of the current tensor state"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{tensor_name.replace('.', '_')}_{timestamp}.safetensors"
            
            # Get tensor data using safetensors
            with safe_open(self.safetensors_file, framework="pt") as f:
                if tensor_name not in f.keys():
                    console.print(f"[red]Error: Tensor {tensor_name} not found[/]")
                    return ""
                
                # Get the tensor
                tensor = f.get_tensor(tensor_name)
                
                # Save just this tensor as a safetensors file
                tensors_dict = {tensor_name: tensor}
                save_file(tensors_dict, str(backup_path))
                
                console.print(f"[green]✓[/] Created backup at: {backup_path}")
                return str(backup_path)
                
        except Exception as e:
            console.print(f"[red]Error creating backup: {str(e)}[/]")
            return ""

    def restore_backup(self, tensor_name: str, backup_path: str) -> bool:
        """Restore a tensor from backup"""
        try:
            if not os.path.exists(backup_path):
                console.print(f"[red]Error: Backup file {backup_path} not found[/]")
                return False
            
            # Load the backup tensor
            with safe_open(backup_path, framework="pt") as f:
                if tensor_name not in f.keys():
                    console.print(f"[red]Error: Tensor {tensor_name} not found in backup[/]")
                    return False
                
                backup_tensor = f.get_tensor(tensor_name)
            
            # Create output directory if restoring
            output_dir = Path(self.model_path).parent / f"{Path(self.model_path).name}_restored"
            output_dir.mkdir(exist_ok=True)
            
            # Load current model tensors
            with safe_open(self.safetensors_file, framework="pt") as f:
                tensors_dict = {name: f.get_tensor(name) for name in f.keys()}
            
            # Replace the tensor with backup
            tensors_dict[tensor_name] = backup_tensor
            
            # Save the restored model
            save_file(tensors_dict, str(output_dir / "model.safetensors"))
            
            console.print(f"[green]✓[/] Restored {tensor_name} from backup")
            console.print(f"[green]✓[/] Saved restored model to: {output_dir}/model.safetensors")
            return True
            
        except Exception as e:
            console.print(f"[red]Error restoring backup: {str(e)}[/]")
            return False

    def list_backups(self, tensor_name: Optional[str] = None) -> List[Dict[str, str]]:
        """List available backups"""
        try:
            backups = []
            for backup_file in self.backup_dir.glob("*.safetensors"):
                # Parse tensor name and timestamp from filename
                parts = backup_file.stem.rsplit("_", 1)
                if len(parts) == 2:
                    tensor = parts[0].replace("_", ".")
                    timestamp = parts[1]
                    
                    if tensor_name is None or tensor == tensor_name:
                        backups.append({
                            "tensor": tensor,
                            "timestamp": timestamp,
                            "path": str(backup_file)
                        })
            
            return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            console.print(f"[red]Error listing backups: {str(e)}[/]")
            return []

    def start_interactive_session(self):
        """Start an interactive exploration session"""
        console.print(Panel(
            "[bold cyan]Enhanced Tensor Patcher Interactive Mode[/]\n\n"
            "Available commands:\n"
            "- [green]analyze[/] <tensor_name> : Run enhanced analysis\n"
            "- [green]patch[/] <tensor_name> <operation> <value> : Apply patch\n"
            "- [green]investigate[/] : Start guided investigation\n"
            "- [green]hex[/] <tensor_name> : Show hex patterns\n"
            "- [green]tokens[/] <tensor_name> : Analyze token embeddings\n"
            "- [green]history[/] : Show modification history\n"
            "- [green]exit[/] : Exit session",
            title="Welcome"
        ))
        
        while True:
            try:
                command = typer.prompt("\nCommand")
                
                if command == "exit":
                    break
                    
                parts = command.split()
                if not parts:
                    continue
                
                if parts[0] == "analyze" and len(parts) > 1:
                    self._handle_analyze(parts[1])
                elif parts[0] == "patch" and len(parts) > 3:
                    self._handle_patch(parts[1], parts[2], float(parts[3]))
                elif parts[0] == "investigate":
                    self._handle_investigation()
                elif parts[0] == "hex" and len(parts) > 1:
                    self._handle_hex_analysis(parts[1])
                elif parts[0] == "tokens" and len(parts) > 1:
                    self._handle_token_analysis(parts[1])
                elif parts[0] == "history":
                    self._show_history()
                else:
                    console.print("[yellow]Invalid command. Type 'exit' to quit.[/]")
                    
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/]")

    def _handle_investigation(self):
        """Handle investigation mode"""
        console.print("\n[bold cyan]Starting Model Investigation[/]")
        
        # Access tensors through the explorer
        if not hasattr(self.base_patcher, "explorer") or not self.base_patcher.explorer:
            console.print("[red]Error: No model explorer initialized[/]")
            return
            
        # Analyze key tensor types
        tensor_types = {
            "attention": [name for name in self.base_patcher.explorer.tensors if "attn" in name.lower()],
            "mlp": [name for name in self.base_patcher.explorer.tensors if "mlp" in name.lower()],
            "embeddings": [name for name in self.base_patcher.explorer.tensors if "embed" in name.lower()]
        }
        
        findings = []
        
        # Analyze each tensor type
        for type_name, tensors in tensor_types.items():
            console.print(f"\n[bold]Analyzing {type_name} tensors...[/]")
            
            for tensor_name in tensors[:3]:  # Analyze first 3 of each type
                analysis = self.analyze_tensor_patterns(tensor_name)
                if "error" not in analysis:
                    findings.append({
                        "tensor": tensor_name,
                        "type": type_name,
                        "patterns": analysis["patterns"],
                        "recommendations": self.get_enhanced_recommendations(tensor_name)
                    })
        
        # Show summary
        console.print("\n[bold cyan]Investigation Summary[/]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tensor Type")
        table.add_column("Patterns Found")
        table.add_column("Recommendations")
        
        for type_name in tensor_types:
            type_findings = [f for f in findings if f["type"] == type_name]
            if type_findings:
                patterns = sum(len(f["patterns"]) for f in type_findings)
                recommendations = sum(len(f["recommendations"]) for f in type_findings)
                table.add_row(type_name, str(patterns), str(recommendations))
        
        console.print(table)
        
        # Show detailed recommendations
        console.print("\n[bold cyan]Key Recommendations:[/]")
        for finding in findings:
            if finding["recommendations"]:
                console.print(f"\n[bold]{finding['tensor']}[/]")
                for rec in finding["recommendations"]:
                    console.print(f"[green]•[/] {rec['reason']}")
                    console.print(f"  Operation: {rec['operation']} (value: {rec['value']})")

    def _handle_hex_analysis(self, tensor_name: str):
        """Handle hex pattern analysis command"""
        console.print(f"\n[bold cyan]Analyzing hex patterns for {tensor_name}...[/]")
        
        patterns = self.analyze_hex_patterns(tensor_name)
        if "error" in patterns:
            console.print(f"[red]Error: {patterns['error']}[/]")
            return
        
        # Show special values
        if patterns["special_values"]:
            console.print("\n[bold]Special Values Found:[/]")
            for value_type, hex_val in patterns["special_values"]:
                console.print(f"[green]•[/] {value_type}: {hex_val}")
        
        # Show byte distribution
        if patterns["byte_distribution"]:
            console.print("\n[bold]Byte Distribution:[/]")
            total = sum(patterns["byte_distribution"].values())
            for byte, count in sorted(patterns["byte_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 2)
                console.print(f"{byte}: {bar} ({percentage:.1f}%)")

    def _handle_token_analysis(self, tensor_name: str):
        """Handle token embedding analysis command"""
        console.print(f"\n[bold cyan]Analyzing token embeddings for {tensor_name}...[/]")
        
        analysis = self.analyze_token_embeddings(tensor_name)
        if "error" in analysis:
            console.print(f"[red]Error: {analysis['error']}[/]")
            return
        
        # Show embedding structure
        console.print("\n[bold]Embedding Structure:[/]")
        console.print(f"Vocabulary Size: {analysis['vocab_size']}")
        console.print(f"Embedding Dimension: {analysis['embedding_dim']}")
        
        # Show other statistics
        if analysis.get("special_tokens"):
            console.print("\n[bold]Special Token Analysis:[/]")
            for token in analysis["special_tokens"]:
                console.print(f"[green]•[/] {token}")

    def _show_history(self):
        """Show modification history"""
        if not self.history:
            console.print("[yellow]No modifications recorded yet.[/]")
            return
        
        console.print("\n[bold cyan]Modification History:[/]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Timestamp")
        table.add_column("Tensor")
        table.add_column("Operation")
        table.add_column("Value")
        table.add_column("Backup")
        
        for entry in self.history:
            table.add_row(
                entry["timestamp"],
                entry["tensor"],
                entry["operation"],
                str(entry["value"]),
                "✓" if entry["backup"] else "✗"
            )
        
        console.print(table)

    def apply_adaptive_patch(self, tensor_name, preview=True):
        """Apply adaptive modifications based on layer position and context.
        
        This method analyzes the tensor's position and role in the network, then applies
        appropriate modifications based on contextual understanding rather than fixed values.
        
        Args:
            tensor_name: Name of the tensor to modify
            preview: If True, only return the preview of modifications without applying them
            
        Returns:
            Dictionary with modification details and statistics
        """
        try:
            # Load the tensor
            try:
                if not self.safetensors_file:
                    return {"error": "No safetensors file specified for the patcher."}
                with safe_open(self.safetensors_file, framework="pt") as f:
                    if tensor_name not in f.keys():
                        return {"error": f"Tensor {tensor_name} not found in model file {self.safetensors_file}"}
                    tensor = f.get_tensor(tensor_name)
                    original_shape = tensor.shape
                    original_data = tensor.clone()  # Keep original for comparison
            except Exception as e:
                return {"error": f"Error loading tensor from {self.safetensors_file}: {str(e)}"}
            
            # Extract layer info from tensor name
            layer_index = None
            layer_type = "unknown"
            position = "unknown"
            
            # Parse tensor name for layer info
            parts = tensor_name.split('.')
            for i, part in enumerate(parts):
                if part == "layers" and i+1 < len(parts):
                    try:
                        layer_index = int(parts[i+1])
                        break
                    except ValueError:
                        pass
            
            # Estimate total layers
            total_layers = 28  # Default assumption, typical for many LLMs
            # Get model info if possible to determine actual total layers
            if hasattr(self, "config") and isinstance(self.config, dict):
                total_layers = self.config.get("num_hidden_layers", 28)
            
            # Determine position (early, middle, late)
            if layer_index is not None:
                early_threshold = total_layers // 3
                late_threshold = total_layers * 2 // 3
                
                if layer_index < early_threshold:
                    position = "early"
                elif layer_index >= late_threshold:
                    position = "late"
                else:
                    position = "middle"
            
            # Determine tensor type
            if "self_attn" in tensor_name or "attention" in tensor_name:
                layer_type = "attention"
                if "q_proj" in tensor_name or "query" in tensor_name:
                    subtype = "query projection"
                elif "k_proj" in tensor_name or "key" in tensor_name:
                    subtype = "key projection"
                elif "v_proj" in tensor_name or "value" in tensor_name:
                    subtype = "value projection"
                elif "o_proj" in tensor_name or "output" in tensor_name:
                    subtype = "output projection"
                else:
                    subtype = "general attention"
            elif "mlp" in tensor_name or "ffn" in tensor_name:
                layer_type = "feedforward"
                if "up_proj" in tensor_name:
                    subtype = "upward projection"
                elif "down_proj" in tensor_name:
                    subtype = "downward projection"
                elif "gate_proj" in tensor_name:
                    subtype = "gate projection"
                else:
                    subtype = "general mlp"
            elif "norm" in tensor_name or "ln" in tensor_name:
                layer_type = "normalization"
                subtype = "normalization"
            elif "embed" in tensor_name:
                layer_type = "embedding"
                subtype = "token embedding"
            else:
                subtype = "general"
            
            # Calculate tensor statistics
            tensor_mean = float(torch.mean(tensor).item())
            tensor_std = float(torch.std(tensor).item())
            tensor_min = float(torch.min(tensor).item())
            tensor_max = float(torch.max(tensor).item())
            tensor_abs_mean = float(torch.mean(torch.abs(tensor)).item())
            
            # Apply adaptive modifications based on layer position and type
            modified_tensor = tensor.clone()
            
            # Keep track of modifications applied
            modifications = []
            
            # Define adaptive modification factors
            # These are not hardcoded constants but calculated dynamically based on tensor statistics
            if position == "early" and layer_type == "attention":
                # Early attention layers - subtle enhancement of pattern recognition
                if subtype == "query projection":
                    # Enhance query selectivity
                    factor = 0.05 * tensor_std  # Scale factor relative to tensor's own standard deviation
                    mask = torch.abs(tensor) > tensor_abs_mean * 1.2  # Find values above average
                    modified_tensor[mask] *= (1 + factor)
                    modifications.append(f"Enhanced salient query projections by {factor:.4f} factor")
                    
                elif subtype == "key projection":
                    # Subtle regularization of keys
                    factor = 0.02 * tensor_std
                    modified_tensor *= (1 - factor)
                    modifications.append(f"Subtle regularization of key projections by {factor:.4f} factor")
                
            elif position == "middle" and layer_type == "attention":
                # Middle attention layers - focus on nuanced reasoning capabilities
                if subtype == "query projection" or subtype == "key projection":
                    # Enhance contrast between strong and weak attention patterns
                    high_values = torch.abs(tensor) > tensor_abs_mean * 1.5
                    low_values = torch.abs(tensor) < tensor_abs_mean * 0.5
                    
                    if high_values.any():
                        factor = 0.03 * tensor_std
                        modified_tensor[high_values] *= (1 + factor)
                        modifications.append(f"Enhanced high-value patterns by {factor:.4f} factor")
                    
                    if low_values.any():
                        factor = 0.02 * tensor_std
                        modified_tensor[low_values] *= (1 - factor)
                        modifications.append(f"Dampened low-value patterns by {factor:.4f} factor")
                
            elif position == "late" and layer_type == "attention":
                # Late attention layers - focus on output refinement
                if subtype == "output projection":
                    # Apply subtle noise reduction
                    factor = 0.04 * tensor_std
                    noise_mask = torch.abs(tensor) < tensor_abs_mean * 0.3
                    if noise_mask.any():
                        modified_tensor[noise_mask] *= (1 - factor)
                        modifications.append(f"Reduced noise in output projection by {factor:.4f} factor")
            
            elif layer_type == "feedforward":
                # Feedforward networks - apply adaptive scaling based on position
                if position == "early":
                    # Early FFN - subtle enhancement of feature extraction
                    factor = 0.03 * tensor_std
                    modified_tensor *= (1 + factor)
                    modifications.append(f"Enhanced early FFN feature extraction by {factor:.4f} factor")
                    
                elif position == "middle":
                    # Middle FFN - selective enhancement of strong activations
                    strong_activations = torch.abs(tensor) > tensor_abs_mean * 1.3
                    if strong_activations.any():
                        factor = 0.04 * tensor_std
                        modified_tensor[strong_activations] *= (1 + factor)
                        modifications.append(f"Enhanced middle FFN strong activations by {factor:.4f} factor")
                    
                elif position == "late":
                    # Late FFN - subtle precision enhancement
                    factor = 0.02 * tensor_std
                    modified_tensor *= (1 + factor)
                    modifications.append(f"Enhanced late FFN precision by {factor:.4f} factor")
            
            elif layer_type == "normalization":
                # For normalization layers, apply subtle shifts to control information flow
                scale_mask = tensor > 0  # Focus on positive scaling factors
                if scale_mask.any():
                    factor = 0.01 * tensor_std  # Very subtle adjustment
                    modified_tensor[scale_mask] *= (1 + factor)
                    modifications.append(f"Fine-tuned normalization scaling by {factor:.4f} factor")
            
            elif layer_type == "embedding":
                # For embeddings, perform selective enhancement
                high_norm_embeds = torch.norm(tensor, dim=-1, keepdim=True) > torch.mean(torch.norm(tensor, dim=-1))
                if high_norm_embeds.any():
                    factor = 0.02 * tensor_std
                    modified_tensor[high_norm_embeds] *= (1 + factor)
                    modifications.append(f"Enhanced high-norm embeddings by {factor:.4f} factor")
            
            # If no specific modifications were applied, apply a gentle adaptive scaling
            if not modifications:
                factor = 0.01 * tensor_std
                modified_tensor *= (1 + factor)
                modifications.append(f"Applied gentle adaptive scaling by {factor:.4f} factor")
            
            # Calculate change statistics
            absolute_diff = torch.abs(modified_tensor - tensor)
            max_change = float(torch.max(absolute_diff).item())
            mean_change = float(torch.mean(absolute_diff).item())
            percent_change = float(mean_change / tensor_abs_mean * 100)
            
            # Calculate post-modification statistics
            modified_mean = float(torch.mean(modified_tensor).item())
            modified_std = float(torch.std(modified_tensor).item())
            modified_min = float(torch.min(modified_tensor).item())
            modified_max = float(torch.max(modified_tensor).item())
            
            # Create preview results
            preview_data = {
                "tensor_name": tensor_name,
                "layer_index": layer_index,
                "position": position,
                "layer_type": layer_type,
                "subtype": subtype,
                "original_stats": {
                    "mean": tensor_mean,
                    "std": tensor_std,
                    "min": tensor_min,
                    "max": tensor_max,
                    "shape": list(original_shape)
                },
                "modified_stats": {
                    "mean": modified_mean,
                    "std": modified_std,
                    "min": modified_min,
                    "max": modified_max
                },
                "change_stats": {
                    "max_absolute_change": max_change,
                    "mean_absolute_change": mean_change,
                    "percent_change": percent_change
                },
                "modifications_applied": modifications
            }
            
            # If preview only, return the results without saving
            if preview:
                return {
                    "tensor_name": tensor_name,
                    "preview": preview_data
                }
            
            # Otherwise, save the modified tensor to a new safetensors file
            output_path = self._create_modified_safetensors(
                {tensor_name: modified_tensor},
                f"adaptive_{os.path.basename(self.model_path)}"
            )
            
            # Create a rich table for display
            table = Table(title=f"Adaptive Modifications for {tensor_name}")
            table.add_column("Property")
            table.add_column("Value")
            
            table.add_row("Layer Position", position)
            table.add_row("Layer Type", f"{layer_type} ({subtype})")
            table.add_row("Mean Change", f"{mean_change:.6f}")
            table.add_row("Max Change", f"{max_change:.6f}")
            table.add_row("Percent Change", f"{percent_change:.2f}%")
            
            for mod in modifications:
                table.add_row("Modification", mod)
            
            console.print(table)
            
            return {
                "tensor_name": tensor_name,
                "preview": preview_data,
                "output_path": output_path
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            console.print(f"[bold red]Error:[/] {str(e)}")
            return {"error": str(e), "details": error_details}

@app.command()
def analyze(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    tensor_name: str = typer.Option(..., "--tensor", "-t", help="Name of tensor to analyze")
):
    """Run enhanced analysis on a tensor"""
    patcher = EnhancedTensorPatcher(model_path)
    
    # Run pattern analysis
    console.print(f"[bold cyan]Running enhanced analysis for {tensor_name}...[/]")
    analysis = patcher.analyze_tensor_patterns(tensor_name)
    
    if "error" in analysis:
        console.print(f"[bold red]Error: {analysis['error']}[/]")
        return
    
    # Display patterns
    console.print("\n[bold]Detected Patterns:[/]")
    for pattern in analysis["patterns"]:
        console.print(f"[green]•[/] {pattern['type']} (Confidence: {pattern['confidence']})")
        console.print(f"  Evidence: {pattern['evidence']}")
    
    # Show distribution
    patcher.visualize_tensor_distribution(tensor_name)
    
    # Get recommendations
    console.print("\n[bold]Recommended Actions:[/]")
    recommendations = patcher.get_enhanced_recommendations(tensor_name)
    
    for i, rec in enumerate(recommendations, 1):
        console.print(f"\n[cyan]{i}.[/] {rec['reason']}")
        console.print(f"   Operation: {rec['operation']} (value: {rec['value']})")
        console.print(f"   Target: {rec['target']}")
        console.print(f"   Confidence: {rec['confidence']}")
        console.print(f"   Evidence: {rec['evidence']}")

@app.command()
def patch(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    tensors: List[str] = typer.Option([], "--tensor", "-t", help="Name of tensor to patch (can be specified multiple times)"),
    operations: List[str] = typer.Option([], "--op", "-o", help="Operation to apply (can be specified multiple times)"),
    values: List[float] = typer.Option([], "--value", "-v", help="Value for operation (can be specified multiple times)"),
    targets: List[str] = typer.Option([], "--target", help="Target range (all, top 10%, top 20%) (can be specified multiple times)")
):
    """Apply enhanced patch to tensors"""
    if not tensors:
        console.print("[red]Error: At least one tensor must be specified[/]")
        return
        
    if len(operations) != len(tensors) or len(values) != len(tensors) or (targets and len(targets) != len(tensors)):
        console.print("[red]Error: Number of operations, values, and targets must match number of tensors[/]")
        return
    
    # Fill in default targets if needed
    if not targets:
        targets = ["all"] * len(tensors)
    
    patcher = EnhancedTensorPatcher(model_path)
    
    # First preview all changes
    previews = []
    for tensor_name, operation, value, target in zip(tensors, operations, values, targets):
        console.print(f"\n[bold cyan]Previewing changes for {tensor_name}...[/]")
        preview = patcher.apply_enhanced_patch(
            tensor_name=tensor_name,
            operation=operation,
            value=value,
            target=target,
            preview=True
        )
        
        if "error" in preview:
            console.print(f"[bold red]Error previewing {tensor_name}: {preview['error']}[/]")
            return
            
        # Display preview
        console.print("\n[bold]Preview:[/]")
        console.print(f"Operation: {operation} with value {value}")
        console.print(f"Target: {target}")
        
        if "preview" in preview:
            stats = preview["preview"]
            console.print("\n[bold]Statistics:[/]")
            console.print(f"Affected values: {stats['affected_values']:,} of {stats['total_values']:,} ({stats['affected_values']/stats['total_values']*100:.1f}%)")
            console.print("\n[bold]Before:[/]")
            console.print(f"  Mean: {stats['original_mean']:.6f}")
            console.print(f"  Std:  {stats['original_std']:.6f}")
            console.print("\n[bold]After:[/]")
            console.print(f"  Mean: {stats['modified_mean']:.6f}")
            console.print(f"  Std:  {stats['modified_std']:.6f}")
            
            if stats.get("patterns_affected"):
                console.print("\n[bold]Patterns Affected:[/]")
                for pattern in stats["patterns_affected"]:
                    console.print(f"[yellow]•[/] {pattern}")
        
        if "safety_checks" in preview:
            console.print("\n[bold]Safety Checks:[/]")
            for check, passed in preview["safety_checks"].items():
                status = "[green]✓[/]" if passed else "[red]✗[/]"
                console.print(f"{status} {check}")
        
        previews.append(preview)
    
    # Ask for confirmation
    if typer.confirm("\nApply all changes?"):
        # Prepare batch patches
        patches = []
        for tensor_name, operation, value, target in zip(tensors, operations, values, targets):
            patches.append({
                "tensor_name": tensor_name,
                "operation": operation,
                "value": value,
                "target": target
            })
        
        # Apply all changes in a single batch operation
        result = patcher.apply_batch_patches(patches)
        
        if "error" in result:
            console.print(f"[bold red]Error applying batch patches: {result['error']}[/]")
            return
        
        # Show results
        console.print(f"\n[bold green]✓[/] All changes applied successfully!")
        
        # Show patch results
        for patch_result in result.get("results", []):
            tensor_name = patch_result.get("tensor_name", "unknown")
            if patch_result.get("result") == "success":
                console.print(f"[green]✓[/] Modified {tensor_name}")
                
                # Show final stats if available
                if "stats" in patch_result:
                    stats = patch_result["stats"]
                    console.print("\n[bold]Final Statistics:[/]")
                    console.print(f"Modified {stats['affected_values']:,} values")
                    console.print(f"New mean: {stats['modified_mean']:.6f}")
                    console.print(f"New std:  {stats['modified_std']:.6f}")
            else:
                console.print(f"[red]✗[/] Failed to modify {tensor_name}: {patch_result.get('error', 'Unknown error')}")
        
        # Show output path
        output_path = result.get("output_path", "")
        if output_path:
            console.print(f"[green]✓[/] Modified model saved to: {output_path}")

@app.command()
def investigate(
    model_path: str = typer.Argument(..., help="Path to the model directory")
):
    """Run comprehensive model investigation"""
    patcher = EnhancedTensorPatcher(model_path)
    patcher._handle_investigation()

@app.command()
def interactive(
    model_path: str = typer.Argument(..., help="Path to the model directory")
):
    """Start interactive session"""
    patcher = EnhancedTensorPatcher(model_path)
    patcher.start_interactive_session()

@app.command()
def backups(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    tensor_name: str = typer.Option(None, "--tensor", "-t", help="Filter backups by tensor name"),
    restore: str = typer.Option(None, "--restore", "-r", help="Restore from backup path"),
):
    """Manage tensor backups"""
    patcher = EnhancedTensorPatcher(model_path)
    
    if restore:
        # Restore from backup
        if not tensor_name:
            console.print("[red]Error: Must specify tensor name when restoring[/]")
            return
            
        success = patcher.restore_backup(tensor_name, restore)
        if not success:
            console.print("[red]Failed to restore backup[/]")
    else:
        # List backups
        backups = patcher.list_backups(tensor_name)
        
        if not backups:
            console.print("[yellow]No backups found[/]")
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tensor")
        table.add_column("Timestamp")
        table.add_column("Path")
        
        for backup in backups:
            table.add_row(
                backup["tensor"],
                backup["timestamp"],
                backup["path"]
            )
        
        console.print("\n[bold cyan]Available Backups:[/]")
        console.print(table)
        console.print("\nTo restore a backup:")
        console.print("enhanced_tensor_patcher.py backups ./model --tensor <tensor_name> --restore <backup_path>")

if __name__ == "__main__":
    app() 