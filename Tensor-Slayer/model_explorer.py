#!/usr/bin/env python3
"""
Model Explorer - Interactive tool for model investigation and modification
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import box
from enhanced_tensor_patcher import EnhancedTensorPatcher
from ai_tensor_explorer import AITensorExplorer
from rich.progress import Progress
from rich.table import Table
from safetensors import safe_open

console = Console()
app = typer.Typer()

class ModelExplorer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Find model weight files by extension rather than specific names
        self.weight_files = self._find_weight_files()
        
        # Initialize with the discovered files, not assumptions about structure
        self.tensor_patcher = EnhancedTensorPatcher(self.weight_files)
        self.tensor_explorer = AITensorExplorer(self.weight_files)
        
    def _find_weight_files(self):
        """Find all model weight files regardless of naming convention."""
        path = Path(self.model_path)
        
        # Look for common model weight file types
        weight_extensions = [
            "*.safetensors",  # Most common for newer models
            "*.bin",          # PyTorch binary files
            "*.pt",           # PyTorch files
            "*.ckpt",         # Checkpoint files
            "*.model"         # Some models use this extension
        ]
        
        # Collect all weight files
        weight_files = []
        for ext in weight_extensions:
            weight_files.extend(list(path.glob(ext)))
        
        # If no weight files found in root, check subdirectories (one level)
        if not weight_files:
            for subdir in path.iterdir():
                if subdir.is_dir():
                    for ext in weight_extensions:
                        weight_files.extend(list(subdir.glob(ext)))
        
        if not weight_files:
            console.print("[yellow]Warning: No model weight files found. This might not be a valid model directory.[/]")
        else:
            console.print(f"[green]Found {len(weight_files)} model weight files.[/]")
        
        return weight_files
        
    def start_interactive(self):
        """Start interactive exploration session"""
        console.print(Panel(
            "[bold cyan]Model Explorer[/]\n\n"
            "Available commands:\n"
            "- [green]explore[/] : Explore model structure and tensors\n"
            "- [green]investigate[/] <query> : Investigate specific model behaviors\n"
            "- [green]analyze[/] <tensor> : Analyze specific tensor\n"
            "- [green]patch[/] : Start guided patching session\n"
            "- [green]adaptive[/] <tensor> : Apply layer-aware adaptive modifications\n"
            "- [green]suggest[/] : Get modification suggestions\n"
            "- [green]help[/] : Show this help\n"
            "- [green]exit[/] : Exit session",
            title="Welcome",
            border_style="cyan"
        ))
        
        while True:
            try:
                command = Prompt.ask("\nCommand")
                
                if command == "exit":
                    break
                    
                parts = command.split(maxsplit=1)
                cmd = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd == "explore":
                    self._handle_explore()
                elif cmd == "investigate":
                    self._handle_investigate(args)
                elif cmd == "analyze":
                    self._handle_analyze(args)
                elif cmd == "patch":
                    self._handle_patch()
                elif cmd == "adaptive" and args:
                    self._handle_adaptive(args)
                elif cmd == "suggest":
                    self._handle_suggest()
                elif cmd == "help":
                    self._show_help()
                else:
                    console.print("[yellow]Unknown command. Type 'help' for available commands.[/]")
                    
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/]")
    
    def _handle_explore(self):
        """Handle explore command"""
        console.print("\n[bold cyan]Model Structure:[/]")
        
        # Show model overview
        self.tensor_explorer.explorer.display_tensors(self.tensor_explorer.tensors)
        
        # Show key tensor types
        tensor_types = {
            "attention": [name for name in self.tensor_explorer.tensors if "attn" in name.lower()],
            "mlp": [name for name in self.tensor_explorer.tensors if "mlp" in name.lower()],
            "embeddings": [name for name in self.tensor_explorer.tensors if "embed" in name.lower()]
        }
        
        for type_name, tensors in tensor_types.items():
            console.print(f"\n[bold]Found {len(tensors)} {type_name} tensors[/]")
            if tensors:
                for tensor in tensors[:3]:  # Show first 3 as examples
                    console.print(f"  • {tensor}")
                if len(tensors) > 3:
                    console.print("  ...")
    
    def _handle_investigate(self, query: str):
        """Handle investigate command with enhanced analysis capabilities"""
        if not query:
            query = Prompt.ask("What would you like to investigate")
        
        console.print(f"\n[bold cyan]Investigating:[/] {query}")
        
        # Check if this is a tensor-specific investigation
        if "tensor" in query.lower() or any(marker in query.lower() for marker in ["weight", "bias", "layer", "attention", "mlp", "embed"]):
            # Extract potential tensor types from the query
            tensor_types = []
            if "attention" in query.lower() or "attn" in query.lower():
                tensor_types.append("self_attn")
            if "mlp" in query.lower() or "feed forward" in query.lower():
                tensor_types.append("mlp")
            if "embedding" in query.lower() or "embed" in query.lower():
                tensor_types.append("embed")
            
            if tensor_types:
                console.print("[cyan]Detected tensor-specific investigation.[/]")
                
                # Find matching tensors
                matching_tensors = []
                for tensor_type in tensor_types:
                    tensors = [name for name in self.tensor_explorer.tensors if tensor_type in name.lower()]
                    matching_tensors.extend(tensors[:5])  # Limit to 5 per type
                
                if matching_tensors:
                    console.print(f"[green]Found {len(matching_tensors)} relevant tensors for analysis.[/]")
                    
                    # Collect statistics for each tensor
                    tensor_stats = []
                    for tensor_name in matching_tensors[:5]:  # Limit to 5 total tensors for display
                        try:
                            stats = self.tensor_patcher.explorer.explorer.calculate_tensor_statistics(tensor_name)
                            tensor_stats.append((tensor_name, stats))
                        except Exception as e:
                            console.print(f"[yellow]Error analyzing {tensor_name}: {str(e)}[/]")
                    
                    # Display comparative statistics
                    if tensor_stats:
                        # Create comparison table
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("Tensor")
                        table.add_column("Min", justify="right")
                        table.add_column("Max", justify="right")
                        table.add_column("Mean", justify="right")
                        table.add_column("Std", justify="right")
                        table.add_column("Sparsity", justify="right")
                        
                        for tensor_name, stats in tensor_stats:
                            if "error" not in stats:
                                # Shorten tensor name for display
                                display_name = tensor_name.split(".")[-3:] if len(tensor_name.split(".")) > 2 else tensor_name
                                display_name = ".".join(display_name)
                                
                                table.add_row(
                                    display_name,
                                    f"{stats.get('min', 'N/A'):.4f}",
                                    f"{stats.get('max', 'N/A'):.4f}",
                                    f"{stats.get('mean', 'N/A'):.4f}",
                                    f"{stats.get('std', 'N/A'):.4f}",
                                    f"{stats.get('sparsity', 0) * 100:.2f}%"
                                )
                        
                        console.print("\n[bold]Comparative Tensor Analysis:[/]")
                        console.print(table)
                        
                        # Show distribution of one representative tensor
                        if tensor_stats:
                            representative_tensor = tensor_stats[0][0]
                            console.print(f"\n[bold]Distribution of representative tensor ([cyan]{representative_tensor}[/]):[/]")
                            self.tensor_patcher.visualize_tensor_distribution(representative_tensor)
        
        # For embedding investigations related to specific tokens
        elif "token" in query.lower() or "embedding" in query.lower() or "word" in query.lower():
            # Try to identify any specific token in the query
            import re
            token_match = re.search(r'"([^"]+)"', query)
            if token_match:
                token = token_match.group(1)
                console.print(f"[cyan]Analyzing token embedding for:[/] {token}")
                
                try:
                    # Visualize token embedding if available
                    self.tensor_explorer.visualize_token_embeddings(token)
                    
                    # Find similar tokens
                    console.print("\n[bold]Finding similar tokens:[/]")
                    self.tensor_explorer.find_similar_embeddings(token, top_k=5)
                except Exception as e:
                    console.print(f"[yellow]Could not analyze token embedding: {str(e)}[/]")
        
        # Default to using the explorer's investigation capabilities
        self.tensor_explorer.run_investigation(query)
    
    def _handle_analyze(self, tensor_name: str):
        """Handle analyze command with enhanced statistics"""
        if not tensor_name:
            tensor_name = Prompt.ask("Enter tensor name to analyze")
        
        # Use enhanced tensor analysis
        analysis = self.tensor_patcher.analyze_tensor_patterns(tensor_name)
        if "error" in analysis:
            console.print(f"[red]Error: {analysis['error']}[/]")
            return
        
        # Show analysis results
        console.print(f"\n[bold cyan]Analysis of {tensor_name}:[/]")
        
        # Get detailed statistics including percentiles
        detailed_stats = self.tensor_patcher.explorer.explorer.calculate_tensor_statistics(tensor_name)
        if "error" not in detailed_stats:
            console.print("\n[bold]Detailed Statistics:[/]")
            console.print(f"Min: {detailed_stats['min']:.6f}")
            console.print(f"Max: {detailed_stats['max']:.6f}")
            console.print(f"Mean: {detailed_stats['mean']:.6f}")
            console.print(f"Std: {detailed_stats['std']:.6f}")
            console.print(f"Sparsity: {detailed_stats['sparsity']*100:.2f}%")
            
            # Show percentiles
            console.print("\n[bold]Percentiles:[/]")
            quantiles = detailed_stats.get("quantiles", {})
            for q, val in quantiles.items():
                console.print(f"{float(q)*100:3.0f}%: {val:.6f}")
            
            # Get tensor sample values
            try:
                tensor = self.tensor_patcher.explorer.explorer.load_tensor(tensor_name)
                sample_size = min(5, tensor.numel())
                if tensor.numel() > 0:
                    console.print("\n[bold]Value Samples:[/]")
                    flat_tensor = tensor.flatten()
                    for i in range(min(5, tensor.numel())):
                        console.print(f"Sample {i+1}: {flat_tensor[i].item():.6f}")
                    
                    # Simulate operations
                    console.print("\n[bold]Operation Simulation:[/]")
                    operations = [
                        ("scale", 1.1),
                        ("scale", 0.9),
                        ("add", 0.01),
                        ("clamp_max", detailed_stats['max'] * 0.9)
                    ]
                    
                    for op_name, op_value in operations:
                        console.print(f"\n[cyan]After {op_name} with {op_value}:[/]")
                        
                        if op_name == "scale":
                            console.print(f"Min: {detailed_stats['min'] * op_value:.6f}")
                            console.print(f"Max: {detailed_stats['max'] * op_value:.6f}")
                            console.print(f"Mean: {detailed_stats['mean'] * op_value:.6f}")
                        elif op_name == "add":
                            console.print(f"Min: {detailed_stats['min'] + op_value:.6f}")
                            console.print(f"Max: {detailed_stats['max'] + op_value:.6f}")
                            console.print(f"Mean: {detailed_stats['mean'] + op_value:.6f}")
                        elif op_name == "clamp_max":
                            new_max = min(detailed_stats['max'], op_value)
                            console.print(f"Min: {detailed_stats['min']:.6f}")
                            console.print(f"Max: {new_max:.6f}")
                            
                            # Estimate percentage of affected values
                            if "quantiles" in detailed_stats:
                                affected_pct = 0
                                for q, val in quantiles.items():
                                    if val > op_value:
                                        affected_pct = 100 - (float(q) * 100)
                                        break
                                console.print(f"Approximately {affected_pct:.1f}% of values would be affected")
            except Exception as e:
                console.print(f"[yellow]Could not load tensor samples: {str(e)}[/]")
        
        # Show patterns
        if analysis.get("patterns"):
            console.print("\n[bold]Detected Patterns:[/]")
            for pattern in analysis["patterns"]:
                console.print(f"[green]•[/] {pattern['type']} (Confidence: {pattern['confidence']})")
                console.print(f"  Evidence: {pattern['evidence']}")
        
        # Show distribution
        self.tensor_patcher.visualize_tensor_distribution(tensor_name)
        
        # Show recommendations
        recommendations = self.tensor_patcher.get_enhanced_recommendations(tensor_name)
        if recommendations:
            console.print("\n[bold]Recommended Actions:[/]")
            for i, rec in enumerate(recommendations, 1):
                console.print(f"\n[cyan]{i}.[/] {rec['reason']}")
                console.print(f"   Operation: {rec['operation']} (value: {rec['value']})")
                console.print(f"   Target: {rec['target']}")
                console.print(f"   Confidence: {rec['confidence']}")
    
    def _handle_patch(self):
        """Handle patch command"""
        console.print("\n[bold cyan]Guided Patching Session[/]")
        
        # Get tensor to patch
        tensor_name = Prompt.ask("Enter tensor name to patch")
        
        # Show analysis first
        self._handle_analyze(tensor_name)
        
        # Get operation details
        operation = Prompt.ask(
            "Choose operation",
            choices=["scale", "add", "normalize"],
            default="scale"
        )
        
        value = float(Prompt.ask("Enter value", default="1.05"))
        
        target = Prompt.ask(
            "Choose target range",
            choices=["all", "top 10%", "top 20%"],
            default="all"
        )
        
        # Preview changes
        preview = self.tensor_patcher.apply_enhanced_patch(
            tensor_name=tensor_name,
            operation=operation,
            value=value,
            target=target,
            preview=True
        )
        
        if "error" in preview:
            console.print(f"[red]Error: {preview['error']}[/]")
            return
        
        # Show preview
        if "preview" in preview:
            stats = preview["preview"]
            console.print("\n[bold]Preview:[/]")
            console.print(f"Operation: {operation} with value {value}")
            console.print(f"Target: {target}")
            
            console.print("\n[bold]Statistics:[/]")
            console.print(f"Affected values: {stats['affected_values']:,} of {stats['total_values']:,} ({stats['affected_values']/stats['total_values']*100:.1f}%)")
            
            console.print("\n[bold]Before:[/]")
            console.print(f"  Mean: {stats['original_mean']:.6f}")
            console.print(f"  Std:  {stats['original_std']:.6f}")
            
            console.print("\n[bold]After:[/]")
            console.print(f"  Mean: {stats['modified_mean']:.6f}")
            console.print(f"  Std:  {stats['modified_std']:.6f}")
        
        # Show safety checks
        if "safety_checks" in preview:
            console.print("\n[bold]Safety Checks:[/]")
            for check, passed in preview["safety_checks"].items():
                status = "[green]✓[/]" if passed else "[red]✗[/]"
                console.print(f"{status} {check}")
        
        # Ask for confirmation
        if Confirm.ask("\nApply these changes?"):
            result = self.tensor_patcher.apply_enhanced_patch(
                tensor_name=tensor_name,
                operation=operation,
                value=value,
                target=target,
                preview=False
            )
            
            if "error" in result:
                console.print(f"[red]Error: {result['error']}[/]")
            else:
                console.print("[bold green]✓[/] Changes applied successfully!")
                console.print(f"[green]✓[/] Modified model saved to: {result['output_path']}")
    
    def _handle_adaptive(self, tensor_name: str):
        """Apply layer-aware adaptive modifications to a tensor"""
        if not tensor_name:
            console.print("[yellow]Please specify a tensor name[/]")
            return
        
        # First show a preview
        console.print(f"[bold]Analyzing {tensor_name} for adaptive modification...[/]")
        
        # Check if tensor exists
        try:
            # Use the tensor list from the patcher's explorer, similar to _handle_suggest
            if tensor_name not in self.tensor_patcher.explorer.tensors:
                console.print(f"[yellow]Tensor '{tensor_name}' not found in model.[/]")
                return
        except AttributeError:
            # This would occur if self.tensor_patcher or self.tensor_patcher.explorer or self.tensor_patcher.explorer.tensors is not found
            console.print(f"[red]Error: Could not access tensor list (self.tensor_patcher.explorer.tensors). Ensure EnhancedTensorPatcher is correctly initialized and has an 'explorer' attribute with a 'tensors' collection.[/]")
            return
        except Exception as e: # Catch other potential errors during the check
            console.print(f"[red]An unexpected error occurred while checking tensor existence: {str(e)}[/]")
            return

        # self.tensor_patcher is initialized in __init__.
        # The following block was attempting to re-initialize it using a non-existent
        # 'self.safetensors_file' attribute and is therefore removed.
        # if not hasattr(self, "tensor_patcher"):
        #     from enhanced_tensor_patcher import EnhancedTensorPatcher
        #     # This line would also be problematic as self.safetensors_file doesn't exist
        #     self.tensor_patcher = EnhancedTensorPatcher(self.weight_files) # Corrected to self.weight_files if it were needed
        
        # First preview the modifications
        try:
            preview_result = self.tensor_patcher.apply_adaptive_patch(tensor_name, preview=True)
            
            if "error" in preview_result:
                console.print(f"[red]Error: {preview_result['error']}[/]")
                return
            
            # Display preview
            preview_data = preview_result.get("preview", {})
            
            # Create a rich table for display
            table = Table(title=f"Adaptive Modifications Preview for {tensor_name}")
            table.add_column("Property")
            table.add_column("Value")
            
            # Add general information
            table.add_row("Layer Position", preview_data.get("position", "unknown"))
            table.add_row("Layer Type", f"{preview_data.get('layer_type', 'unknown')} ({preview_data.get('subtype', 'unknown')})")
            
            # Add original statistics
            orig_stats = preview_data.get("original_stats", {})
            table.add_row("Original Mean", f"{orig_stats.get('mean', 0):.6f}")
            table.add_row("Original Std", f"{orig_stats.get('std', 0):.6f}")
            
            # Add modified statistics
            mod_stats = preview_data.get("modified_stats", {})
            table.add_row("Modified Mean", f"{mod_stats.get('mean', 0):.6f}")
            table.add_row("Modified Std", f"{mod_stats.get('std', 0):.6f}")
            
            # Add change statistics
            change_stats = preview_data.get("change_stats", {})
            table.add_row("Mean Change", f"{change_stats.get('mean_absolute_change', 0):.6f}")
            table.add_row("Percent Change", f"{change_stats.get('percent_change', 0):.2f}%")
            
            # Add modifications applied
            for mod in preview_data.get("modifications_applied", []):
                table.add_row("Modification", mod)
            
            console.print(table)
            
            # Ask user if they want to apply the changes
            apply_changes = Confirm.ask("Apply these adaptive modifications?")
            if apply_changes:
                # Apply the changes
                result = self.tensor_patcher.apply_adaptive_patch(tensor_name, preview=False)
                
                if "error" in result:
                    console.print(f"[red]Error: {result['error']}[/]")
                    return
                
                console.print(f"[bold green]✓[/] Successfully applied adaptive modifications")
                console.print(f"Output saved to: {result['output_path']}")
            else:
                console.print("[yellow]Changes not applied[/]")
            
        except Exception as e:
            console.print(f"[red]Error applying adaptive patch: {str(e)}[/]")
    
    def _handle_suggest(self):
        """Handle suggest command with actionable, per-tensor recommendations and patching option"""
        console.print("\n[bold cyan]Getting Model Recommendations[/]")
        
        capability = Prompt.ask(
            "Choose capability to improve",
            choices=["general", "math", "reasoning"],
            default="general"
        )
        recommendations = self.get_ai_recommendations(capability)

        if recommendations:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#")
            table.add_column("Tensor")
            table.add_column("Operation")
            table.add_column("Value")
            table.add_column("Target")
            table.add_column("Confidence")
            table.add_column("Reason")
            for i, rec in enumerate(recommendations, 1):
                table.add_row(
                    str(i),
                    rec.get("tensor_name", ""),
                    str(rec.get("operation", "")),
                    str(rec.get("value", "")),
                    str(rec.get("target", "")),
                    f"{float(rec.get('confidence', 0)):.2f}",
                    str(rec.get("reason", ""))
                )
            console.print("\n[bold]Actionable Recommendations:[/]")
            console.print(table)
        else:
            console.print("[yellow]No actionable recommendations generated.[/]")
            return

        # Ask user if they want to apply all, some, or none of the recommendations
        if not recommendations:
            return
        console.print("\n[bold cyan]Would you like to apply these changes?[/]")
        choice = Prompt.ask("Apply?", choices=["yes", "no", "choose"], default="no")
        to_apply = []
        if choice == "yes":
            to_apply = recommendations
        elif choice == "choose":
            console.print("Enter the numbers of the recommendations to apply, separated by commas (e.g. 1,3,5):")
            selected = Prompt.ask("Numbers").replace(" ", "")
            try:
                indices = [int(x)-1 for x in selected.split(",") if x.strip().isdigit()]
                to_apply = [recommendations[i] for i in indices if 0 <= i < len(recommendations)]
            except Exception:
                console.print("[red]Invalid selection. No changes will be applied.[/]")
                return
        else:
            console.print("[yellow]No changes applied.[/]")
            return

        # Prepare patches for batch application
        patches = []

        # Create a map for tensor sizes for easy lookup from the already gathered tensor_stats
        tensor_sizes_map = {ts['name']: ts.get('size_mb') for ts in tensor_stats if 'name' in ts}
        LARGE_TENSOR_THRESHOLD_MB = 10000  # Lowered threshold for large tensors in MB

        console.print(f"\n[bold cyan]Preparing patches based on user selection...[/]")
        for i, rec in enumerate(to_apply):
            original_target = rec.get("target", "all")
            current_target = original_target
            tensor_name = rec.get("tensor_name")
            operation = rec.get("operation")
            value_str = rec.get("value")

            console.print(f"Processing recommendation #{i+1}: Tensor='{tensor_name}', Op='{operation}', Value='{value_str}', Target='{original_target}'")

            if not all([tensor_name, operation, value_str is not None]):
                console.print(f"[yellow]  Skipping: Incomplete recommendation data.[/]")
                continue

            try:
                patch_value = float(value_str)
            except (ValueError, TypeError):
                console.print(f"[yellow]  Skipping: Invalid value '{value_str}' for tensor '{tensor_name}'. Must be a number.[/]")
                continue

            current_size_mb = tensor_sizes_map.get(tensor_name)
            is_quantile_target = isinstance(current_target, str) and \
                                 any(kw in current_target.lower() for kw in ["top", "bottom", "percent"])

            if is_quantile_target:
                if current_size_mb is not None:
                    console.print(f"  Tensor '{tensor_name}' size: {current_size_mb:.2f}MB. Quantile target: '{current_target}'.")
                    if current_size_mb > LARGE_TENSOR_THRESHOLD_MB:
                        console.print(f"[yellow]  Warning: Tensor '{tensor_name}' ({current_size_mb:.2f}MB) exceeds threshold ({LARGE_TENSOR_THRESHOLD_MB}MB). "
                                      f"Changing target from '{current_target}' to 'all'.[/]")
                        current_target = "all"
                    else:
                        console.print(f"  Tensor size is within threshold. Keeping target '{current_target}'.")
                else: # current_size_mb is None
                    console.print(f"[yellow]  Warning: Size for tensor '{tensor_name}' is unknown. "
                                  f"Proceeding with quantile target '{current_target}', which might be risky if the tensor is large.[/]")
            else:
                console.print(f"  Target '{current_target}' is not quantile-based. No change needed based on size.")
            
            patches.append({
                "tensor_name": tensor_name,
                "operation": operation,
                "value": patch_value,
                "target": current_target
            })
            if original_target != current_target:
                console.print(f"  Final patch for '{tensor_name}': Op='{operation}', Value={patch_value}, Target changed from '{original_target}' to '{current_target}'")
            else:
                console.print(f"  Final patch for '{tensor_name}': Op='{operation}', Value={patch_value}, Target='{current_target}'")
        
        if patches:
            console.print(f"Applying {len(patches)} patches in batch...")
            
            # Create output directory for reference (actual path will be determined by apply_batch_patches)
            import os
            from pathlib import Path
            output_dir = os.path.join(
                self.tensor_patcher.model_path + "_modified", 
                f"{Path(self.tensor_patcher.model_path).name}_patched"
            )
            
            # Apply batch patches
            try:
                result = self.tensor_patcher.apply_batch_patches(patches)
                
                if "error" in result:
                    console.print(f"[red]Error applying batch patches: {result['error']}[/]")
                else:
                    console.print(f"[green]✓[/] Successfully applied {len(patches)} patches")
                    console.print(f"   Modified model saved to: {result.get('output_path', output_dir)}")
                    
                    # Show detailed results
                    for patch_result in result.get("results", []):
                        tensor_name = patch_result.get("tensor_name", "unknown")
                        if patch_result.get("result") == "success":
                            console.print(f"[green]✓[/] {tensor_name}: successful")
                        else:
                            console.print(f"[red]✗[/] {tensor_name}: {patch_result.get('error', 'Failed')}")
            except Exception as e:
                import traceback
                console.print(f"[red]Exception applying batch patches: {str(e)}[/]")
                console.print(traceback.format_exc())

    def get_ai_recommendations(self, capability: str = "general") -> List[Dict[str, Any]]:
        """Expose Tensor-Slayer suggestion flow programmatically."""
        tensor_stats: List[Dict[str, Any]] = []
        for name, info in self.tensor_patcher.explorer.tensors.items():
            stats = self.tensor_patcher.explorer.explorer.analyze_tensor(name)
            tensor_stats.append({
                "name": name,
                "shape": str(info.get("shape")),
                "size_mb": info.get("size_mb"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            })

        prompt = self._build_suggestion_prompt(tensor_stats, capability)
        response = self.tensor_patcher.explorer.code_agent.run(prompt)
        return self._parse_recommendations(response)

    @staticmethod
    def _build_suggestion_prompt(tensor_stats: List[Dict[str, Any]], capability: str) -> str:
        prompt = f"""ONLY SUGGEST EXACT TENSOR MODIFICATIONS FOR {capability.upper()} CAPABILITY

Your output MUST be a valid JSON array ONLY containing tensor modification objects.

STRICT REQUIREMENTS:
1. NO FINE-TUNING DISCUSSION OR SUGGESTIONS WHATSOEVER
2. NO ARCHITECTURE CHANGES
3. NO DISCUSSIONS OF METHODOLOGY
4. NO TEXT BEFORE OR AFTER THE JSON ARRAY
5. FOCUS ON MLP, ATTENTION, INPUT, OUTPUT, AND OTHER HIGHLY IMPORTANT LAYERS - A MAXIMUM OF 2 MODIFICATIONS EACH LAYER BUT IF YOU THINK MORE IS REQUIRED, DO SO
6. ALL THE SUGGESTIONS SHOULD CORRELATE WITH EACH OTHER AND AIM TOWARDS COMMON GOAL OF IMPROVING THE MODEL PERFORMANCE
7. IT IS ESSENTIAL THAT YOU PROVIDE ABSOLUTELY HIGH QUALITY SUGGESTIONS
8. PROVIDE SUGGESTION FOR OF HIGH CONFIDENCE ONLY, THIS IS ESSENTIAL FOR THE SUCCESS OF THE FRAMEWORK


EACH recommendation object MUST HAVE:
- "tensor_name": EXACT name from the list below
- "operation": one of [scale, add, clamp_max, clamp_min]
- "value": specific number (e.g. 1.05, 0.9, 0.01)
- "target": one of [all, top 10%, top 20%, bottom 10%]
- "confidence": number between 0-1
- "reason": IN-DEPTH EXPLANATION OF THE SPECIFIC BENEFIT OF THE MODIFICATION

Available tensors and their statistics:
"""
        for t in tensor_stats:
            prompt += f"- {t['name']}: shape={t['shape']}, size={t['size_mb']:.2f}MB, mean={t['mean']:.4f}, std={t['std']:.4f}, min={t['min']:.4f}, max={t['max']:.4f}\n"

        prompt += """
WARNING: If you suggest fine-tuning, discuss approaches, or output anything other than a plain JSON array of tensor modifications, your response will be rejected.

Return ONLY a JSON array like: [ {"tensor_name": "...", "operation": "...", ...} ]
"""
        return prompt

    @staticmethod
    def _parse_recommendations(response: Any) -> List[Dict[str, Any]]:
        recommendations: List[Dict[str, Any]] = []
        if isinstance(response, str):
            try:
                recommendations = json.loads(response)
            except Exception:
                import re
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    recommendations = json.loads(match.group(0))
        elif isinstance(response, list):
            recommendations = response
        elif isinstance(response, dict):
            recommendations = response.get("recommendations", [])
        return recommendations
            
        console.print("[bold green]All selected changes applied![/]")
    
    def _show_help(self):
        """Show detailed help"""
        help_text = """
# Model Explorer Help

## Commands

### explore
Browse the model structure and see overview of different tensor types.

### investigate <query>
Investigate specific model behaviors or capabilities. Examples:
- investigate "why does the model avoid certain topics"
- investigate "how does the attention mechanism work"
- investigate "what causes mathematical errors"

### analyze <tensor>
Perform detailed analysis of a specific tensor, showing:
- Detected patterns
- Value distribution
- Recommendations for modification

### patch
Start a guided patching session to modify tensors with:
- Interactive prompts
- Safety checks
- Preview of changes
- Automatic backups

### adaptive <tensor>
Apply layer-aware adaptive modifications to a tensor

### suggest
Get AI-powered suggestions for model improvements based on:
- Current model state
- Target capabilities
- Known patterns

### help
Show this help message.

### exit
Exit the explorer.
"""
        console.print(Panel(Markdown(help_text), title="Help", border_style="cyan"))

@app.command()
def explore(
    model_path: str = typer.Argument(..., help="Path to the model directory")
):
    """Start interactive model exploration"""
    explorer = ModelExplorer(model_path)
    explorer.start_interactive()

if __name__ == "__main__":
    app() 
