#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from collections import defaultdict
import seaborn as sns
from matplotlib.figure import Figure
from io import BytesIO
import base64
from dotenv import load_dotenv
import requests
import tempfile
import subprocess
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import box
from rich.prompt import Prompt, Confirm
from typing import Optional, List, Dict, Any
import webbrowser
import textwrap

# Initialize typer app and rich console
app = typer.Typer()
console = Console()

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

class SafetensorsExplorer:
    def __init__(self, model_path: str, safetensors_file: str = "model.safetensors"):
        """Initialize the explorer with the model path and optionally a specific safetensors file name"""
        self.model_path = model_path
        self.safetensors_file = safetensors_file
        self.config = {}

        # Check if model_path is actually a file path rather than a directory
        if os.path.isfile(model_path):
            # If it's a file path, use it directly
            self.safetensors_file = model_path
            # Set model_path to the parent directory
            self.model_path = os.path.dirname(model_path)
        elif not os.path.exists(self.safetensors_file):
            # If safetensors_file doesn't exist, look for .safetensors files in the model_path directory
            try:
                files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
                if not files:
                    raise FileNotFoundError(f"No safetensors files found in {model_path}")
                self.safetensors_file = os.path.join(model_path, files[0])
            except NotADirectoryError:
                raise ValueError(f"Model path is not a directory and doesn't point to a valid file: {model_path}")

        # Load config.json if available
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            console.print(f"Loaded config with [bold green]{len(self.config)}[/] keys")
    
    def list_tensors(self) -> Dict[str, Dict]:
        """List all tensors in the safetensors file with their shapes and sizes"""
        tensors = {}
        
        with console.status("[bold green]Loading tensor information...[/]", spinner="dots"):
            with safe_open(self.safetensors_file, framework="pt") as f:
                for key in f.keys():
                    # Get tensor directly and extract properties
                    tensor = f.get_tensor(key)
                    shape = tensor.shape
                    dtype = str(tensor.dtype).split('.')[-1]  # Extract dtype name
                    
                    # Calculate size in MB
                    num_elements = np.prod(shape)
                    bytes_per_element = 4  # Assume float32 by default
                    if dtype in ["bfloat16", "float16", "BF16", "FP16"]:
                        bytes_per_element = 2
                    elif dtype in ["float8", "int8", "FP8", "INT8"]:
                        bytes_per_element = 1
                    
                    size_mb = (num_elements * bytes_per_element) / (1024 * 1024)
                    
                    tensors[key] = {
                        "shape": shape,
                        "dtype": dtype,
                        "size_mb": size_mb,
                        "num_elements": num_elements
                    }
                    
                    # Delete the tensor to free memory
                    del tensor
        
        return tensors
    
    def display_tensors(self, tensors: Dict[str, Dict], limit: int = 20) -> None:
        """Display tensor information in a rich table"""
        # Create a table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column("Tensor Name", style="cyan", no_wrap=True)
        table.add_column("Shape", style="green")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Data Type", style="yellow")
        
        # Sort tensors by size
        sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
        total_size_mb = sum(info["size_mb"] for info in tensors.values())
        
        # Add rows
        for i, (name, info) in enumerate(sorted_tensors[:limit]):
            table.add_row(
                str(i+1),
                name,
                str(info["shape"]),
                f"{info['size_mb']:.2f}",
                str(info["dtype"])
            )
        
        # Display the table with a header
        console.print(Panel(
            f"[bold]Found [green]{len(tensors)}[/] tensors, total size: [green]{total_size_mb:.2f}[/] MB[/]", 
            title="Tensor Overview", 
            title_align="left", 
            style="blue"
        ))
        console.print(table)
        
        if len(tensors) > limit:
            console.print(f"[dim]... and {len(tensors) - limit} more tensors[/]")
    
    def _normalize_tensor_name(self, tensor_name: str) -> str:
        """Map different model architecture tensor naming conventions.
        This allows users to use common tensor names across different model types."""
        
        # Common name mappings between different model architectures
        mappings = {
            # GPTX/LLaMA style to DeepSeek style
            "transformer.h": "model.layers",
            "attn.c_proj": "self_attn.o_proj",
            "mlp.c_proj": "mlp.down_proj",
            "attn.c_attn": "self_attn.q_proj",  # Approximate
            
            # Backward compatibility
            "attention": "self_attn",
            "feed_forward": "mlp",
        }
        
        # Apply mappings
        normalized_name = tensor_name
        for old, new in mappings.items():
            if old in normalized_name:
                normalized_name = normalized_name.replace(old, new)
        
        return normalized_name
    
    def analyze_tensor(self, tensor_name: str) -> Dict[str, Any]:
        """Analyze a tensor by name and return statistics"""
        try:
            # Normalize tensor name
            normalized_name = self._normalize_tensor_name(tensor_name)
            
            # If the normalized name is different and original not found, use normalized
            if normalized_name != tensor_name and normalized_name not in self.list_tensors():
                if normalized_name in self.list_tensors():
                    tensor_name = normalized_name
                    console.print(f"[yellow]Using mapped tensor name: {tensor_name}[/]")
            
            # Load the tensor and calculate statistics
            tensor = self.load_tensor(tensor_name)
            
            # Convert to float32 for analysis
            tensor = tensor.to(torch.float32)
            
            # Calculate statistics
            stats = {
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "zeros_percent": float((tensor == 0).sum() / tensor.numel() * 100),
            }
            return stats
        except Exception as e:
            print(f"Error analyzing tensor: {e}")
            return {"error": str(e)}
    
    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load a tensor by name and return it"""
        try:
            from safetensors import safe_open
            
            with safe_open(self.safetensors_file, framework="pt") as f:
                if tensor_name not in f.keys():
                    raise ValueError(f"Tensor {tensor_name} not found in safetensors file")
                return f.get_tensor(tensor_name)
        except Exception as e:
            raise RuntimeError(f"Error loading tensor: {e}")
    
    def _create_histogram_data(self, tensor: torch.Tensor) -> Dict:
        """Create histogram data for visualization"""
        # Sample the tensor if it's large
        if tensor.numel() > 100000:
            idx = torch.randperm(tensor.numel())[:100000]
            tensor_sample = tensor.flatten()[idx]
        else:
            tensor_sample = tensor.flatten()
        
        # Convert to numpy
        tensor_np = tensor_sample.cpu().numpy()
        
        # Calculate histogram
        hist, bin_edges = np.histogram(tensor_np, bins=30)
        
        # Normalize histogram for ASCII display
        max_count = max(hist)
        normalized_hist = [(count / max_count) if max_count > 0 else 0 for count in hist]
        
        return {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "normalized_hist": normalized_hist
        }
    
    def display_tensor_analysis(self, tensor_name: str, stats: Dict[str, Any]) -> None:
        """Display tensor analysis results"""
        if "error" in stats:
            console.print(f"[bold red]Error analyzing tensor:[/] {stats['error']}")
            return
        
        # Calculate size_mb if not provided
        if 'size_mb' not in stats:
            if 'shape' in stats and 'dtype' in stats:
                # Estimate size based on shape and dtype
                element_size = 2  # Default for float16/bfloat16
                if stats['dtype'] == 'torch.float32':
                    element_size = 4
                elif stats['dtype'] == 'torch.float64':
                    element_size = 8
                
                # Calculate total elements from shape
                total_elements = 1
                for dim in stats['shape']:
                    total_elements *= dim
                    
                # Calculate size in MB
                stats['size_mb'] = total_elements * element_size / (1024 * 1024)
            else:
                stats['size_mb'] = 0.0  # Fallback value
        
        # Display tensor information
        console.print(Panel(
            f"[bold]Tensor Analysis:[/] {tensor_name}\n\n"
            f"[bold cyan]Shape:[/] {stats.get('shape', 'unknown')}\n"
            f"[bold cyan]Data type:[/] {stats.get('dtype', 'unknown')}\n"
            f"[bold cyan]Size:[/] {stats.get('size_mb', 0.0):.2f} MB\n"
            f"[bold cyan]Value range:[/] [{stats.get('min', 'unknown')}, {stats.get('max', 'unknown')}]\n"
            f"[bold cyan]Mean:[/] {stats.get('mean', 'unknown')}\n"
            f"[bold cyan]Standard deviation:[/] {stats.get('std', 'unknown')}\n"
            f"[bold cyan]Zero values:[/] {stats.get('zeros_percent', 'unknown')}%",
            title=f"[bold]Tensor Analysis[/]",
            border_style="green"
        ))
        
        # Display ASCII histogram
        if "histogram_data" in stats:
            self._display_ascii_histogram(stats["histogram_data"])
    
    def _display_ascii_histogram(self, hist_data: Dict) -> None:
        """Display an ASCII histogram"""
        console.print(Panel(
            self._generate_ascii_histogram(hist_data),
            title="[bold]Value Distribution[/]",
            title_align="left",
            width=80
        ))
    
    def _generate_ascii_histogram(self, hist_data: Dict) -> str:
        """Generate an ASCII histogram string"""
        hist = hist_data["counts"]
        bin_edges = hist_data["bin_edges"]
        normalized_hist = hist_data["normalized_hist"]
        
        max_bar_width = 40
        result = []
        
        for i, (count, norm) in enumerate(zip(hist, normalized_hist)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            bar_width = int(norm * max_bar_width)
            bar = "■" * bar_width if bar_width > 0 else "▏"
            
            # Colorize the bar based on the bin position (blue to red)
            color_index = int((i / len(hist)) * 255)
            # Blue to cyan to green to yellow to red gradient
            if color_index < 64:
                color = f"[blue]{bar}[/]"
            elif color_index < 128:
                color = f"[cyan]{bar}[/]"
            elif color_index < 192:
                color = f"[green]{bar}[/]"
            else:
                color = f"[yellow]{bar}[/]"
            
            result.append(f"{bin_start:7.4f} to {bin_end:7.4f} | {color} {count}")
        
        return "\n".join(result)
    
    def get_ai_interpretation(self, tensor_name: str, stats: Dict[str, Any]) -> str:
        """Get AI interpretation of the tensor"""
        if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
            return "[yellow]AI interpretation unavailable: API key or model not configured in .env file[/]"
        
        prompt = f"""
        Analyze this tensor from a neural language model:
        
        Tensor name: {tensor_name}
        Statistics:
        - Shape: {stats.get('shape', 'unknown')}
        - Data type: {stats.get('dtype', 'unknown')}
        - Value range: [{stats.get('min', 'unknown')}, {stats.get('max', 'unknown')}]
        - Mean: {stats.get('mean', 'unknown')}
        - Standard deviation: {stats.get('std', 'unknown')}
        - Sparsity: {stats.get('zeros_percent', 'unknown')}%
        
        Based on these statistics and the tensor name:
        1. What part of the model architecture is this tensor likely from?
        2. What function does this tensor serve in the model?
        3. What would happen if we scaled this tensor by 1.2?
        4. What types of behaviors might be affected by modifying this tensor?
        5. Are there any unusual patterns in the statistics that stand out?
        
        Format your response as a concise analysis with bullet points.
        """
        
        with console.status("[bold green]Getting AI interpretation...[/]", spinner="dots"):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": OPENROUTER_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are an AI assistant specialized in analyzing neural network weights and parameters"},
                            {"role": "user", "content": prompt}
                        ]
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"[red]Error: {response.status_code}, {response.text}[/]"
                    
            except Exception as e:
                return f"[red]Error getting AI interpretation: {str(e)}[/]"
    
    def modify_tensor(self, tensor_name: str, operation: str, value: str, output_dir: Optional[str] = None) -> str:
        """Modify a specific tensor and save to a new model"""
        if output_dir is None:
            output_dir = f"{self.model_path}_modified_{tensor_name.split('.')[-1]}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        with console.status(f"[bold green]Modifying tensor {tensor_name}...[/]", spinner="dots"):
            try:
                # Load all tensors
                tensors = {}
                with safe_open(self.safetensors_file, framework="pt") as f:
                    for key in f.keys():
                        if key == tensor_name:
                            # Get the tensor to modify
                            tensor = f.get_tensor(key)
                            
                            # Apply operation
                            if operation == "scale":
                                scale_value = float(value)
                                modified = tensor * scale_value
                                operation_desc = f"scaled by {scale_value}"
                            elif operation == "add":
                                add_value = float(value)
                                modified = tensor + add_value
                                operation_desc = f"added {add_value}"
                            elif operation == "clamp":
                                min_val, max_val = [float(x) for x in value.split(",")]
                                modified = torch.clamp(tensor, min_val, max_val)
                                operation_desc = f"clamped to range [{min_val}, {max_val}]"
                            elif operation == "normalize":
                                std_factor = float(value)
                                mean = tensor.mean()
                                std = tensor.std()
                                modified = (tensor - mean) / (std * std_factor) + mean
                                operation_desc = f"normalized with std factor {std_factor}"
                            else:
                                raise ValueError(f"Unknown operation: {operation}")
                            
                            tensors[key] = modified
                        else:
                            # Copy other tensors as is
                            tensors[key] = f.get_tensor(key)
                
                # Save modified model
                output_file = os.path.join(output_dir, os.path.basename(self.safetensors_file))
                save_file(tensors, output_file)
                
                # Copy config and other files
                for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
                    src_path = os.path.join(self.model_path, filename)
                    if os.path.exists(src_path):
                        import shutil
                        shutil.copy(src_path, os.path.join(output_dir, filename))
                
                return f"[bold green]Success![/] Tensor [cyan]{tensor_name}[/] {operation_desc} and saved to [yellow]{output_dir}[/]"
                
            except Exception as e:
                return f"[bold red]Error modifying tensor:[/] {str(e)}"
    
    def explore_tensors(self) -> None:
        """Interactive CLI for exploring tensors"""
        try:
            tensors = self.list_tensors()
            self.display_tensors(tensors)
            
            while True:
                console.print("\n[bold cyan]Commands:[/]")
                console.print("  [yellow]list[/]       - List all tensors")
                console.print("  [yellow]analyze N[/]  - Analyze tensor by number")
                console.print("  [yellow]search X[/]   - Search for tensors containing X")
                console.print("  [yellow]modify N[/]   - Modify a tensor by number")
                console.print("  [yellow]exit[/]       - Exit the explorer")
                
                cmd = Prompt.ask("\n[bold cyan]Enter command[/]")
                
                if cmd.lower() == "exit":
                    break
                elif cmd.lower() == "list":
                    self.display_tensors(tensors)
                elif cmd.lower().startswith("analyze "):
                    try:
                        num = int(cmd.split(" ")[1]) - 1
                        sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
                        if 0 <= num < len(sorted_tensors):
                            tensor_name = sorted_tensors[num][0]
                            stats = self.analyze_tensor(tensor_name)
                            self.display_tensor_analysis(tensor_name, stats)
                            
                            if OPENROUTER_API_KEY and OPENROUTER_MODEL:
                                if Confirm.ask("[bold cyan]Get AI interpretation?[/]"):
                                    interpretation = self.get_ai_interpretation(tensor_name, stats)
                                    console.print(Panel(
                                        Markdown(interpretation),
                                        title="[bold]AI Interpretation[/]",
                                        title_align="left",
                                        width=80
                                    ))
                        else:
                            console.print("[bold red]Invalid tensor number[/]")
                    except ValueError:
                        console.print("[bold red]Please provide a valid number[/]")
                elif cmd.lower().startswith("search "):
                    search_term = cmd.split(" ", 1)[1].strip().lower()
                    results = {k: v for k, v in tensors.items() if search_term in k.lower()}
                    if results:
                        console.print(f"[bold green]Found {len(results)} tensors matching '{search_term}':[/]")
                        self.display_tensors(results)
                    else:
                        console.print(f"[yellow]No tensors found matching '{search_term}'[/]")
                elif cmd.lower().startswith("modify "):
                    try:
                        num = int(cmd.split(" ")[1]) - 1
                        sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
                        if 0 <= num < len(sorted_tensors):
                            tensor_name = sorted_tensors[num][0]
                            
                            # Get operation
                            console.print("\n[bold cyan]Available operations:[/]")
                            console.print("  [yellow]scale[/]     - Multiply tensor by a factor")
                            console.print("  [yellow]add[/]       - Add a value to tensor")
                            console.print("  [yellow]clamp[/]     - Clamp tensor values to range")
                            console.print("  [yellow]normalize[/] - Normalize tensor")
                            
                            operation = Prompt.ask(
                                "[bold cyan]Operation[/]", 
                                choices=["scale", "add", "clamp", "normalize"],
                                default="scale"
                            )
                            
                            # Get value
                            if operation == "clamp":
                                value = Prompt.ask("[bold cyan]Value (min,max)[/]", default="-1,1")
                            elif operation == "normalize":
                                value = Prompt.ask("[bold cyan]Standard deviation factor[/]", default="1.0")
                            else:
                                value = Prompt.ask("[bold cyan]Value[/]", default="1.1")
                            
                            # Get output dir
                            output_dir = Prompt.ask(
                                "[bold cyan]Output directory[/]", 
                                default=f"{self.model_path}_modified_{tensor_name.split('.')[-1]}"
                            )
                            
                            if Confirm.ask(f"[bold yellow]Modify tensor {tensor_name} with operation {operation}?[/]"):
                                result = self.modify_tensor(tensor_name, operation, value, output_dir)
                                console.print(result)
                        else:
                            console.print("[bold red]Invalid tensor number[/]")
                    except ValueError:
                        console.print("[bold red]Please provide a valid number[/]")
                else:
                    console.print("[bold red]Unknown command[/]")
        
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")

    def get_tensor_info(self, tensor_name: str) -> Dict[str, Any]:
        """
        Return detailed information about a tensor, including shape, dtype, device, min, max, mean, std, zeros_percent, num_elements, and histogram data.
        """
        try:
            tensor = self.load_tensor(tensor_name)
            tensor = tensor.to(torch.float32)
            info = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "num_elements": tensor.numel(),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "zeros_percent": float((tensor == 0).sum().item() / tensor.numel() * 100),
            }
            # Optionally add histogram data
            try:
                info["histogram_data"] = self._create_histogram_data(tensor)
            except Exception:
                pass
            return info
        except Exception as e:
            return {"error": f"Error in get_tensor_info: {str(e)}"}

    def calculate_tensor_statistics(self, tensor_name: str, quantiles=None) -> Dict[str, Any]:
        """
        Calculate quantiles, min, max, mean, std, sparsity, and optionally more for a tensor.
        Args:
            tensor_name (str): Name of the tensor
            quantiles (list, optional): List of quantiles to compute (values between 0 and 1)
        Returns:
            dict: Statistics including quantiles, min, max, mean, std, sparsity
        """
        try:
            tensor = self.load_tensor(tensor_name)
            tensor = tensor.to(torch.float32)
            tensor_np = tensor.cpu().numpy().flatten()
            if quantiles is None:
                quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            quantile_vals = {str(q): float(np.quantile(tensor_np, q)) for q in quantiles}
            stats = {
                "min": float(np.min(tensor_np)),
                "max": float(np.max(tensor_np)),
                "mean": float(np.mean(tensor_np)),
                "std": float(np.std(tensor_np)),
                "sparsity": float((tensor_np == 0).sum() / tensor_np.size),
                "quantiles": quantile_vals,
                "num_elements": int(tensor_np.size),
            }
            return stats
        except Exception as e:
            return {"error": f"Error in calculate_tensor_statistics: {str(e)}"}


@app.command()
def explore(model_path: str = typer.Argument("./Deepseek", help="Path to the model directory")):
    """Interactive exploration of model weights in the terminal"""
    console.rule("[bold green]SafeTensors Model Explorer[/]")
    console.print(f"[bold]Exploring model at:[/] [cyan]{model_path}[/]")
    
    try:
        explorer = SafetensorsExplorer(model_path)
        explorer.explore_tensors()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    model_path: str = typer.Argument("./Deepseek", help="Path to the model directory"),
    tensor_name: str = typer.Option(None, "--tensor", "-t", help="Name of tensor to analyze"),
    ai_interpret: bool = typer.Option(False, "--ai", help="Get AI interpretation of the tensor")
):
    """Analyze a specific tensor in the model"""
    console.rule("[bold green]Tensor Analysis[/]")
    
    try:
        explorer = SafetensorsExplorer(model_path)
        tensors = explorer.list_tensors()
        
        # If tensor not specified, show list and ask for selection
        if tensor_name is None:
            explorer.display_tensors(tensors)
            tensor_num = typer.prompt("Enter tensor number to analyze", type=int)
            sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
            if 1 <= tensor_num <= len(sorted_tensors):
                tensor_name = sorted_tensors[tensor_num-1][0]
            else:
                console.print("[bold red]Invalid tensor number[/]")
                raise typer.Exit(code=1)
        
        # Analyze the tensor
        stats = explorer.analyze_tensor(tensor_name)
        explorer.display_tensor_analysis(tensor_name, stats)
        
        # Get AI interpretation if requested
        if ai_interpret and OPENROUTER_API_KEY and OPENROUTER_MODEL:
            interpretation = explorer.get_ai_interpretation(tensor_name, stats)
            console.print(Panel(
                Markdown(interpretation),
                title="[bold]AI Interpretation[/]",
                title_align="left",
                width=80
            ))
    
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def modify(
    model_path: str = typer.Argument("./Deepseek", help="Path to the model directory"),
    tensor_name: str = typer.Option(None, "--tensor", "-t", help="Name of tensor to modify"),
    operation: str = typer.Option("scale", "--op", "-o", help="Operation: scale, add, clamp, normalize"),
    value: str = typer.Option("1.1", "--value", "-v", help="Value for the operation"),
    output_dir: Optional[str] = typer.Option(None, "--out", help="Output directory for modified model")
):
    """Modify a tensor and save a new model"""
    console.rule("[bold green]Tensor Modification[/]")
    
    try:
        explorer = SafetensorsExplorer(model_path)
        tensors = explorer.list_tensors()
        
        # If tensor not specified, show list and ask for selection
        if tensor_name is None:
            explorer.display_tensors(tensors)
            tensor_num = typer.prompt("Enter tensor number to modify", type=int)
            sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
            if 1 <= tensor_num <= len(sorted_tensors):
                tensor_name = sorted_tensors[tensor_num-1][0]
            else:
                console.print("[bold red]Invalid tensor number[/]")
                raise typer.Exit(code=1)
        
        # Check that the tensor exists
        if tensor_name not in tensors:
            similar_tensors = [name for name in tensors.keys() if tensor_name in name]
            console.print(f"[bold red]Tensor '{tensor_name}' not found.[/]")
            if similar_tensors:
                console.print("[yellow]Did you mean one of these?[/]")
                for i, name in enumerate(similar_tensors[:5]):
                    console.print(f"  {i+1}. {name}")
            raise typer.Exit(code=1)
        
        # Confirm operation
        if not typer.confirm(f"Modify tensor '{tensor_name}' with operation '{operation}' and value '{value}'?"):
            console.print("[yellow]Operation cancelled.[/]")
            raise typer.Exit()
        
        # Modify the tensor
        result = explorer.modify_tensor(tensor_name, operation, value, output_dir)
        console.print(result)
    
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app() 