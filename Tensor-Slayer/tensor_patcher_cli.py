#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import torch
import numpy as np
from typing import Optional, Dict, Any, List, TypeVar
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.status import Status
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import litellm
from smolagents import CodeAgent, tool
from dotenv import load_dotenv

# Initialize typer app and rich console
app = typer.Typer()
console = Console()

# Load environment variables
load_dotenv(override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

# Add timeout settings
TIMEOUT_SECONDS = 30

@tool
def analyze_tensor(model_path: str, tensor_name: str) -> Dict[str, Any]:
    """Analyze a tensor and return its statistics.
    
    Args:
        model_path: Path to the model directory
        tensor_name: Name of the tensor to analyze
    """
    try:
        with safe_open(model_path, framework="pt") as f:
            if tensor_name not in f.keys():
                return {"error": f"Tensor {tensor_name} not found"}
            
            tensor = f.get_tensor(tensor_name)
            tensor = tensor.to(torch.float32)
            
            stats = {
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "zeros_percent": float((tensor == 0).sum() / tensor.numel() * 100),
                "total_elements": tensor.numel()
            }
            return stats
    except Exception as e:
        return {"error": str(e)}

@tool
def apply_patch(model_path: str, tensor_name: str, operation: str, value: str, backup: bool = True) -> Dict[str, Any]:
    """Apply a modification to a tensor.
    
    Args:
        model_path: Path to the model file
        tensor_name: Name of the tensor to modify
        operation: Operation to perform (scale, add, clamp, normalize)
        value: Value for the operation
        backup: Whether to create a backup before modifying
    """
    try:
        # Load all tensors
        tensors = load_file(model_path)
        
        if tensor_name not in tensors:
            return {"error": f"Tensor {tensor_name} not found"}
        
        # Create backup if requested
        if backup:
            backup_path = f"{model_path}.backup"
            if not os.path.exists(backup_path):
                save_file(tensors, backup_path)
                console.print(f"Created backup at: [bold cyan]{backup_path}[/]")
        
        # Apply modification
        tensor = tensors[tensor_name]
        
        if operation == "scale":
            tensor *= float(value)
        elif operation == "add":
            tensor += float(value)
        elif operation == "clamp":
            min_val, max_val = map(float, value.split(","))
            tensor = torch.clamp(tensor, min_val, max_val)
        elif operation == "normalize":
            tensor = (tensor - tensor.mean()) / tensor.std()
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        # Save modified tensors
        tensors[tensor_name] = tensor
        save_file(tensors, model_path)
        
        return {
            "success": True,
            "message": f"Successfully modified tensor {tensor_name}",
            "new_stats": analyze_tensor(model_path, tensor_name)
        }
        
    except Exception as e:
        return {"error": str(e)}

class ModelPatcher:
    def __init__(self):
        self.llm = self._setup_llm()
        self.agent = self._setup_agent()
        self.model_path: Optional[Path] = None
        self.safetensors_file: Optional[str] = None
        self.config: Dict[str, Any] = {}
        
    def _setup_llm(self) -> Any:
        """Setup LiteLLM with OpenRouter configuration"""
        litellm.set_verbose = True  # Enable verbose mode for debugging
        
        # Configure LiteLLM
        litellm.api_key = OPENROUTER_API_KEY
        litellm.api_base = "https://openrouter.ai/api/v1"
        
        def llm_completion(*args, **kwargs):
            # Handle both positional and keyword arguments
            if args and isinstance(args[0], list):
                kwargs['messages'] = args[0]
            
            # Remove unsupported parameters
            kwargs.pop('stop_sequences', None)
            
            # Convert messages to the format expected by OpenRouter
            if 'messages' in kwargs:
                kwargs['messages'] = [
                    {
                        'role': msg.get('role', 'user'),
                        'content': msg.get('content', '')
                    }
                    for msg in kwargs['messages']
                ]
            
            # Add timeout and other parameters
            kwargs.update({
                'model': OPENROUTER_MODEL,
                'api_base': "https://openrouter.ai/api/v1",
                'timeout': TIMEOUT_SECONDS,
                'temperature': 0.7,
                'max_tokens': 500
            })
            
            try:
                with Status("[bold green]Waiting for AI response...", console=console):
                    response = litellm.completion(**kwargs)
                    
                    # Extract content from the response structure
                    if hasattr(response.choices[0].message, 'provider_specific_fields'):
                        content = response.choices[0].message.provider_specific_fields.get('reasoning_content', '')
                    else:
                        content = response.choices[0].message.content
                        
                    # Create a simple object with content attribute
                    class SimpleResponse:
                        def __init__(self, content):
                            self.content = content
                    
                    return SimpleResponse(content)
                    
            except Exception as e:
                console.print(f"[red]Error calling OpenRouter API: {str(e)}[/]")
                return SimpleResponse(f"Error: Unable to get AI suggestions. Please check your API key and try again. Error: {str(e)}")
        
        return llm_completion

    def _setup_agent(self) -> CodeAgent:
        """Setup smolagents CodeAgent with custom tools"""
        tools = [analyze_tensor, apply_patch]
        
        return CodeAgent(
            tools=tools,
            model=self.llm,
            additional_authorized_imports=[
                "torch", "numpy", "safetensors"
            ]
        )

    def load_model(self, model_path: str) -> None:
        """Load model from path"""
        self.model_path = Path(model_path)
        
        # Find safetensors file
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        self.safetensors_file = str(safetensors_files[0])
        console.print(f"Using safetensors file: [bold cyan]{self.safetensors_file}[/]")
        
        # Load config if available
        config_path = self.model_path / "config.json"
        if config_path.exists():
            self.config = json.loads(config_path.read_text())
            console.print(f"Loaded config with [bold green]{len(self.config)}[/] keys")

    def list_tensors(self) -> Dict[str, Dict]:
        """List all tensors in the model"""
        tensors = {}
        with safe_open(self.safetensors_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensors[key] = {
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype)
                }
        return tensors

@app.command()
def explore(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    limit: int = typer.Option(20, help="Number of tensors to display")
):
    """Explore tensors in a model"""
    try:
        patcher = ModelPatcher()
        patcher.load_model(model_path)
        
        tensors = patcher.list_tensors()
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column("Tensor Name", style="cyan")
        table.add_column("Shape", style="green")
        table.add_column("Data Type", style="yellow")
        
        # Add rows
        for i, (name, info) in enumerate(list(tensors.items())[:limit]):
            table.add_row(
                str(i+1),
                name,
                str(info["shape"]),
                str(info["dtype"])
            )
        
        console.print(Panel(
            f"[bold]Found [green]{len(tensors)}[/] tensors[/]",
            title="Tensor Overview",
            title_align="left",
            style="blue"
        ))
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")

@app.command()
def analyze_cmd(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    tensor_name: str = typer.Option(..., "--tensor", "-t", help="Name of tensor to analyze")
):
    """Analyze a specific tensor"""
    try:
        patcher = ModelPatcher()
        patcher.load_model(model_path)
        
        stats = analyze_tensor(patcher.safetensors_file, tensor_name)
        if "error" in stats:
            console.print(f"[red]Error: {stats['error']}[/]")
            return
        
        # Display stats
        console.print(Panel(
            "\n".join([
                f"[cyan]Shape:[/] {stats['shape']}",
                f"[cyan]Data Type:[/] {stats['dtype']}",
                f"[cyan]Min:[/] {stats['min']:.6f}",
                f"[cyan]Max:[/] {stats['max']:.6f}",
                f"[cyan]Mean:[/] {stats['mean']:.6f}",
                f"[cyan]Std:[/] {stats['std']:.6f}",
                f"[cyan]Zeros:[/] {stats['zeros_percent']:.2f}%"
            ]),
            title=f"Tensor Analysis: {tensor_name}",
            title_align="left",
            style="blue"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")

@app.command()
def suggest(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    tensor_name: str = typer.Option(..., "--tensor", "-t", help="Name of tensor to analyze")
):
    """Get AI suggestions for tensor modifications"""
    try:
        patcher = ModelPatcher()
        patcher.load_model(model_path)
        
        # First analyze the tensor
        stats = analyze_tensor(patcher.safetensors_file, tensor_name)
        if "error" in stats:
            console.print(f"[red]Error: {stats['error']}[/]")
            return
        
        # Get suggestions through the agent
        result = patcher.agent.run(
            f"Analyze these tensor statistics and suggest potential modifications to improve model performance:\n{json.dumps(stats, indent=2)}"
        )
        
        console.print(Panel(
            Markdown(result),
            title=f"Suggestions for: {tensor_name}",
            title_align="left",
            style="blue"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")

@app.command()
def patch_cmd(
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    tensor_name: str = typer.Option(..., "--tensor", "-t", help="Name of tensor to modify"),
    operation: str = typer.Option(..., "--op", "-o", help="Operation: scale, add, clamp, normalize"),
    value: str = typer.Option(..., "--value", "-v", help="Value for the operation")
):
    """Apply a modification to a tensor"""
    try:
        patcher = ModelPatcher()
        patcher.load_model(model_path)
        
        # Confirm action
        if not Confirm.ask(
            f"[yellow]Are you sure you want to modify tensor [bold]{tensor_name}[/] with operation [bold]{operation}[/]?[/]"
        ):
            return
        
        # Apply modification
        result = apply_patch(
            patcher.safetensors_file,
            tensor_name,
            operation,
            value
        )
        
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/]")
            return
        
        console.print(f"[green]Successfully modified tensor {tensor_name}[/]")
        
        # Show new stats
        if "new_stats" in result:
            console.print("\n[bold]New tensor statistics:[/]")
            stats = result["new_stats"]
            console.print(Panel(
                "\n".join([
                    f"[cyan]Min:[/] {stats['min']:.6f}",
                    f"[cyan]Max:[/] {stats['max']:.6f}",
                    f"[cyan]Mean:[/] {stats['mean']:.6f}",
                    f"[cyan]Std:[/] {stats['std']:.6f}"
                ]),
                title="Updated Statistics",
                title_align="left",
                style="blue"
            ))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")

if __name__ == "__main__":
    app() 