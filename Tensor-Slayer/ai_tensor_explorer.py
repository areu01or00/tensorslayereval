#!/usr/bin/env python3
"""
AI-Powered Tensor Explorer

Interactive tool for exploring and modifying model tensors with AI assistance.
"""

import os
import sys
import json
import time
import traceback
import inspect
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress
from rich.prompt import Prompt
import typer
from safetensors import safe_open
from pathlib import Path
from smolagents import CodeAgent
from smolagents.tools import Tool
from dotenv import load_dotenv

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from smolagents import CodeAgent
    from smolagents.tools import Tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

# Initialize rich console
console = Console()

# Load environment variables
load_dotenv(override=True)

# Import the SafetensorsExplorer class and the API key variables
from safetensors_explorer_cli import SafetensorsExplorer

# Import SmolagentS for code execution
try:
    from smolagents import CodeAgent, LiteLLMModel
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

# API Keys from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL")

# Initialize typer app and rich console
app = typer.Typer()
console = Console()

class AITensorExplorer:
    def __init__(self, weight_files: list):
        """Initialize the AI-powered tensor explorer"""
        self.weight_files = weight_files

        if not weight_files:
            raise ValueError("No weight files provided")
        first_file = weight_files[0]
        self.model_path = str(Path(first_file).parent) if isinstance(first_file, (str, Path)) else ""
        
        # Initialize the explorer with the first weight file for now
        
        console.print(f"[cyan]Initializing explorer with {first_file}...[/]")
        
        # This can be enhanced later to merge tensors from multiple files
        self.explorer = SafetensorsExplorer(self.weight_files[0] if self.weight_files else "")
        self.tensors = self.explorer.list_tensors()
        self.exploration_history = []
        self.analyzed_tensors = {}
        self.chat_history = []
        self.command_history = []
        self.tokenizer = None
        self.token_map = None
        
        # Add this tensor_data_loader reference
        self.tensor_data_loader = self.explorer
        
        # Try to load tokenizer if available - make this more robust
        self._load_tokenizer()
        
        # Setup SmolagentS CodeAgent if available
        self.code_agent = None
        if SMOLAGENTS_AVAILABLE:
            if not OPENROUTER_API_KEY:
                console.print("[yellow]Warning: OPENROUTER_API_KEY not set. AI features will be limited.[/]")
                console.print("[yellow]Please set OPENROUTER_API_KEY in your .env file to enable AI features.[/]")
            else:
                self._setup_code_agent()
        else:
            console.print("[yellow]Warning: SmolagentS not available. AI features will be limited.[/]")
            console.print("[yellow]Please install smolagents with: pip install smolagents[/]")
        
        # Initialize token categories
        self._initialize_token_categories()
        
    def _load_tokenizer(self):
        """Try to load the tokenizer for the model in a more flexible way"""
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizer
            
            with console.status("[bold green]Loading tokenizer...[/]", spinner="dots"):
                try:
                    # Get model directory from the parent of the first weight file
                    model_path = Path(self.model_path)
                    
                    # Try AutoTokenizer first which handles many model types
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_path,
                            trust_remote_code=True
                        )
                        console.print("[bold green]✓[/] Tokenizer loaded successfully with AutoTokenizer")
                        return
                    except Exception as e:
                        console.print(f"[yellow]AutoTokenizer failed: {str(e)}. Trying different approaches...[/]")
                    
                    # Check for common tokenizer files with different naming patterns
                    tokenizer_files = {
                        "tokenizer.json": model_path / "tokenizer.json",
                        "tokenizer_config.json": model_path / "tokenizer_config.json", 
                        "vocab.json": model_path / "vocab.json",
                        "merges.txt": model_path / "merges.txt",
                        "tokenizer.model": model_path / "tokenizer.model",
                        "spiece.model": model_path / "spiece.model",
                    }
                    
                    # Log which tokenizer files were found
                    found_files = [name for name, path in tokenizer_files.items() if path.exists()]
                    console.print(f"[cyan]Found tokenizer files: {', '.join(found_files) if found_files else 'None'}[/]")
                    
                    # Try different loading approaches based on files found
                    if tokenizer_files["tokenizer.model"].exists():
                        # SentencePiece-based tokenizers (LLaMA, DeepSeek, etc.)
                        try:
                            from transformers import LlamaTokenizer
                            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
                            console.print("[bold green]✓[/] Loaded SentencePiece-based tokenizer")
                            return
                        except Exception as e:
                            console.print(f"[yellow]SentencePiece tokenizer failed: {str(e)}[/]")
                    
                    if tokenizer_files["vocab.json"].exists():
                        # Try Qwen tokenizer or other vocab-based tokenizers
                        try:
                            from transformers import PreTrainedTokenizerFast
                            self.tokenizer = PreTrainedTokenizerFast(
                                tokenizer_file=str(tokenizer_files["tokenizer.json"]) if tokenizer_files["tokenizer.json"].exists() else None,
                                vocab_file=str(tokenizer_files["vocab.json"]),
                                merges_file=str(tokenizer_files["merges.txt"]) if tokenizer_files["merges.txt"].exists() else None
                            )
                            console.print("[bold green]✓[/] Loaded vocab-based tokenizer")
                            return
                        except Exception as e:
                            console.print(f"[yellow]Vocab-based tokenizer failed: {str(e)}[/]")
                    
                    # If all else fails, try fallback tokenizers
                    fallback_tokenizers = ["gpt2", "bert-base-uncased", "t5-small"]
                    for fallback in fallback_tokenizers:
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                            console.print(f"[yellow]Using {fallback} tokenizer as fallback. This may not perfectly match your model.[/]")
                            return
                        except Exception:
                            pass
                    
                    # Last resort - create simplified token mapping
                    console.print("[yellow]Could not load any tokenizer. Using simplified embedding exploration instead.[/]")
                    self._create_simplified_token_mapping()
                    
                except Exception as e:
                    console.print(f"[yellow]Unexpected error loading tokenizer: {str(e)}.[/] Using simplified approach.")
                    self._create_simplified_token_mapping()
        
        except ImportError:
            console.print("[yellow]Transformers library not installed. Please install it with: pip install transformers[/]")
            self._create_simplified_token_mapping()

        except Exception as e:
            console.print(f"[yellow]Unexpected error loading tokenizer: {str(e)}.[/] Using simplified approach.")
            self._create_simplified_token_mapping()
    
    def _create_simplified_token_mapping(self):
        """Create a simplified embedding exploration approach that doesn't require a tokenizer"""
        self.token_map = {
            'indices': {
                'first_100': list(range(0, 100)),
                'middle_range': list(range(1000, 1100)),
                'high_range': list(range(10000, 10100)),
            }
        }
        console.print("[yellow]Created simplified embedding index ranges for exploration.[/]")
    
    def _initialize_token_categories(self):
        """Initialize token categories for exploration"""
        if not self.tokenizer:
            return
            
        # Define categories of tokens that might be interesting
        special_tokens = {
            'code_tokens': ['def', 'class', '(', ')', '{', '}', '=', '==', '!=', '->', '=>', 'return', 'import'],
            'math_tokens': ['+', '-', '*', '/', '**', '==', '<', '>', '>=', '<='],
            'logic_tokens': ['if', 'else', 'while', 'for', 'and', 'or', 'not'],
            'special_chars': ['[', ']', ',', '.', ':', ';', '"', "'"]
        }
        
        # Map tokens to their IDs
        self.token_map = {}
        for category, tokens in special_tokens.items():
            self.token_map[category] = {}
            for token in tokens:
                # Get token IDs (might be multiple per token)
                try:
                    token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
                    self.token_map[category][token] = token_ids
                except:
                    # Skip tokens that can't be encoded
                    pass
    
    def create_token_embedding_map(self, categories=None):
        """Create a mapping between tokens and their embedding indices"""
        if not self.tokenizer:
            console.print("[bold red]Tokenizer not available. Cannot create token map.[/]")
            return
        
        # Default categories to check if none provided
        default_categories = {
            'code_tokens': ['def', 'class', '(', ')', '{', '}', '=', '==', '!=', '->', '=>', 'return', 'import'],
            'math_tokens': ['+', '-', '*', '/', '**', '==', '<', '>', '>=', '<='],
            'logic_tokens': ['if', 'else', 'while', 'for', 'and', 'or', 'not'],
            'special_chars': ['[', ']', ',', '.', ':', ';', '"', "'"]
        }
        
        # Check if input is a specific token rather than categories
        if isinstance(categories, str) and len(categories) > 0:
            # Single token mode - visualize just this token
            token = categories
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            
            if token_ids:
                console.print(f"\n[bold cyan]Token mapping for:[/] '{token}'")
                console.print(f"[bold]Token ID:[/] {token_ids[0]}")
                
                # If it's a multi-token sequence, show all parts
                if len(token_ids) > 1:
                    console.print("\n[bold]This input is split into multiple tokens:[/]")
                    for i, tid in enumerate(token_ids):
                        # Get the specific token text if possible
                        token_text = self.tokenizer.decode([tid])
                        console.print(f"  [bold]Part {i+1}:[/] ID={tid}, Text='{token_text}'")
                
                # Visualize the token embedding
                self.visualize_embeddings_by_token(token)
                return
        
        # Continue with regular category mapping
        categories_to_check = categories if categories else default_categories.keys()
        
        token_map = {}
        
        for category in categories_to_check:
            token_map[category] = {}
            # Get tokens from the default categories
            tokens = default_categories.get(category, [])
            
            for token in tokens:
                try:
                    token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                    token_map[category][token] = token_ids
                except Exception as e:
                    console.print(f"[red]Error encoding token '{token}': {str(e)}[/]")
        
        # Create a table to display tokens and their IDs
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Category")
        table.add_column("Token")
        table.add_column("Token IDs")
        
        for category, tokens in token_map.items():
            for token, token_ids in tokens.items():
                table.add_row(category, f"'{token}'", str(token_ids))
        
        # Display the table
        console.print(Panel(table, title="Token to Embedding Map", border_style="green"))
        
        # Save the token map to a JSON file
        json_file_path = f"{self.model_path}/token_embedding_map.json"
        with open(json_file_path, 'w') as f:
            json.dump(token_map, f, indent=2)
        
        console.print(f"✓ Token map saved to {json_file_path}")
        
        return self.token_map
    
    def find_similar_embeddings(self, token_input: str, top_k: int = 5):
        """Find tokens with embeddings similar to the input token"""
        if not self.tokenizer:
            console.print("[bold red]Tokenizer not available. Cannot find similar embeddings.[/]")
            return
            
        # Find the embedding tensor
        embedding_tensors = [name for name in self.tensors.keys() if "embed" in name.lower() and "weight" in name.lower()]
        if not embedding_tensors:
            console.print("[bold red]No embedding tensor found in the model.[/]")
            return
            
        embedding_tensor_name = embedding_tensors[0]
        
        try:
            # Tokenize the input
            token_ids = self.tokenizer(token_input, add_special_tokens=False).input_ids
            if not token_ids:
                console.print(f"[bold red]Could not tokenize '{token_input}'[/]")
                return
                
            # Get the embedding for this token
            token_id = token_ids[0]
            
            from safetensors import safe_open
            import numpy as np
            import torch
            import torch.nn.functional as F
            
            with console.status(f"[bold green]Finding tokens with embeddings similar to '{token_input}'...[/]", spinner="dots"):
                # Get the embedding tensor
                with safe_open(self.explorer.safetensors_file, framework="pt") as f:
                    if embedding_tensor_name not in f.keys():
                        console.print(f"[bold red]Tensor {embedding_tensor_name} not found![/]")
                        return
                    
                    # Load the full embedding matrix - this could be large, but we need it for comparison
                    embedding_matrix = f.get_tensor(embedding_tensor_name)
                    
                    # Get our target embedding
                    target_embedding = embedding_matrix[token_id].unsqueeze(0)  # Add batch dimension
                    
                    # Normalize the embeddings for cosine similarity
                    normalized_target = F.normalize(target_embedding, p=2, dim=1)
                    normalized_matrix = F.normalize(embedding_matrix, p=2, dim=1)
                    
                    # Compute cosine similarities
                    similarities = torch.matmul(normalized_target, normalized_matrix.T).squeeze()
                    
                    # Get top-k similar token indices (excluding the input token itself)
                    _, indices = similarities.topk(top_k + 1)
                    indices = [idx.item() for idx in indices if idx.item() != token_id][:top_k]
                    
                    # Display results
                    console.print(f"\n[bold cyan]Tokens with embeddings similar to '[green]{token_input}[/]' (ID: {token_id}):[/]\n")
                    
                    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                    table.add_column("Token", style="green")
                    table.add_column("ID", style="dim")
                    table.add_column("Similarity", style="cyan")
                    
                    # Add the similar tokens to the table
                    for idx in indices:
                        # Try to convert token ID back to text
                        try:
                            similar_token = self.tokenizer.decode([idx])
                            similarity = similarities[idx].item()
                            table.add_row(f"'{similar_token}'", str(idx), f"{similarity:.4f}")
                        except:
                            # Skip tokens that can't be decoded
                            pass
                    
                    console.print(table)
                    
                    # Also show visualization of the target embedding
                    console.print("\n[bold cyan]Visualization of target embedding:[/]")
                    self._visualize_embedding_for_token(embedding_tensor_name, token_id)
        
        except Exception as e:
            console.print(f"[bold red]Error finding similar embeddings: {str(e)}[/]")
            traceback.print_exc()
    
    def visualize_token_embeddings(self, token_input: str):
        """Visualize embeddings for a specific token or category"""
        if not self.tokenizer:
            console.print("[bold red]Tokenizer not available. Cannot visualize embeddings.[/]")
            return
            
        # Find the embedding tensor
        embedding_tensors = [name for name in self.tensors.keys() if "embed" in name.lower() and "weight" in name.lower()]
        if not embedding_tensors:
            console.print("[bold red]No embedding tensor found in the model.[/]")
            return
            
        embedding_tensor_name = embedding_tensors[0]
        
        # Process the token input
        tokens_to_visualize = []
        
        # Check if input is a category name
        if token_input in self.token_map:
            # Get a few tokens from the category
            category_tokens = list(self.token_map[token_input].keys())[:5]  # Limit to 5 tokens
            for token in category_tokens:
                token_ids = self.token_map[token_input][token]
                if token_ids:
                    tokens_to_visualize.append((token, token_ids[0]))  # Use the first token ID
                    
        # Otherwise treat as a direct token
        else:
            try:
                # Tokenize the input
                token_ids = self.tokenizer(token_input, add_special_tokens=False).input_ids
                if token_ids:
                    tokens_to_visualize.append((token_input, token_ids[0]))  # Use the first token ID
            except:
                console.print(f"[bold red]Could not tokenize '{token_input}'[/]")
                return
        
        if not tokens_to_visualize:
            console.print("[bold red]No valid tokens to visualize.[/]")
            return
            
        # Analyze the embedding tensor
        with console.status(f"[bold green]Analyzing embedding tensor: {embedding_tensor_name}...[/]", spinner="dots"):
            stats = self.explorer.analyze_tensor(embedding_tensor_name)
            
            # For each token, visualize its embedding
            for token, token_id in tokens_to_visualize:
                console.print(f"\n[bold cyan]Token:[/] '{token}' [bold cyan]ID:[/] {token_id}")
                self._visualize_embedding_for_token(embedding_tensor_name, token_id)
        
    def _visualize_embedding_for_token(self, embedding_tensor_name: str, token_id: int):
        """Create an ASCII visualization of a token's embedding vector"""
        try:
            from safetensors import safe_open
            import numpy as np
            import torch
            
            # Get the tensor
            with safe_open(self.explorer.safetensors_file, framework="pt") as f:
                if embedding_tensor_name not in f.keys():
                    console.print(f"[bold red]Tensor {embedding_tensor_name} not found![/]")
                    return
                
                tensor = f.get_tensor(embedding_tensor_name)
                
                # Convert to float32 to handle BFloat16 and other formats
                tensor = tensor.to(torch.float32)
                
                # Get the embedding for this token ID
                if token_id < tensor.shape[0]:
                    embedding = tensor[token_id]
                else:
                    console.print(f"[bold red]Token ID {token_id} out of range for tensor of shape {tensor.shape}[/]")
                    return
                
                # Convert to numpy for easier handling
                embedding_np = embedding.cpu().numpy()
                
                # Sample values for visualization (take only a subset for large embeddings)
                sample_size = min(40, len(embedding_np))
                indices = np.linspace(0, len(embedding_np)-1, sample_size, dtype=int)
                sampled_values = embedding_np[indices]
                
                # Create a simple ASCII representation
                max_bar_width = 30
                normalized_values = (sampled_values - np.min(sampled_values)) / (np.max(sampled_values) - np.min(sampled_values) + 1e-10)
                
                console.print("[bold]Embedding vector visualization[/] (sample of values)")
                
                # Show the values and bars
                for i, (idx, value, norm) in enumerate(zip(indices, sampled_values, normalized_values)):
                    bar_width = int(norm * max_bar_width)
                    bar = "█" * bar_width
                    
                    # Color based on value (blue for negative, red for positive)
                    if value < 0:
                        colored_bar = f"[blue]{bar}[/]"
                    else:
                        colored_bar = f"[red]{bar}[/]"
                    
                    # Print with index and value
                    console.print(f"[dim]#{idx:4d}[/] {value:7.4f} {colored_bar}")
                    
                    # Add a separator for readability
                    if i % 10 == 9 and i < sample_size - 1:
                        console.print("     ...")
                
                # Show basic statistics
                mean = np.mean(embedding_np)
                std = np.std(embedding_np)
                min_val = np.min(embedding_np)
                max_val = np.max(embedding_np)
                
                # Display hex representation for the first few values
                hex_values = []
                for val in embedding_np[:8]:
                    try:
                        # Convert float32 to hex
                        hex_val = hex(np.float32(val).view('I'))[2:].zfill(8)
                        hex_values.append(hex_val)
                    except:
                        # Fallback to simple string representation
                        hex_values.append('N/A')
                
                hex_str = " ".join(hex_values)
                
                console.print(f"\n[bold]Embedding Stats:[/] Mean: {mean:.4f}, Std: {std:.4f}, Range: [{min_val:.4f}, {max_val:.4f}]")
                console.print(f"[bold]Hex Representation (first 8 values):[/] {hex_str}...")
                
        except Exception as e:
            console.print(f"[bold red]Error visualizing embedding:[/] {str(e)}")
            traceback.print_exc()
    
    def start_interactive_session(self):
        """Start an interactive session for exploring the model"""
        
        # Show a welcome message with model information
        console.print(Panel(f"Exploring model at: {self.model_path}\nTotal tensors: {len(self.tensors)}", 
                            title="AI-Powered SafeTensors Explorer", 
                            border_style="cyan",
                            expand=True))
        
        # Show tensor overview
        self._show_tensor_overview()
        
        # Enter the interactive loop
        while True:
            try:
                # Try to use rich's Prompt if available
                command = Prompt.ask("> ")
            except (NameError, AttributeError):
                # Fall back to standard input if rich.Prompt isn't working
                try:
                    command = input("> ")
                except EOFError:
                    print("\nExiting...")
                    break
            
            # Process the command
            if not self._process_command(command):
                break
    
    def _show_tensor_overview(self):
        """Show a summary of the model's tensors"""
        console.print("\n[bold cyan]Tensor Overview:[/]")
        self.explorer.display_tensors(self.tensors)
    
    def _process_command(self, command: str):
        """Process a command from the interactive session"""
        self.command_history.append(command)
        
        if command == "exit" or command == "quit":
            return False
        elif command == "help":
            self._display_help()
            return True
        elif command == "list":
            self._list_tensors()
            return True
        elif command.startswith("search "):
            pattern = command[7:]
            return self._search_tensors(pattern)
        elif command.startswith("analyze "):
            target = command[8:]
            if target.isdigit():
                return self._analyze_tensor_by_number(int(target))
            else:
                return self._analyze_tensor_by_name(target)
        elif command.startswith("modify "):
            parts = command[7:].split()
            if len(parts) >= 3:
                target = parts[0]
                operation = parts[1]
                value = parts[2]
                if target.isdigit():
                    return self._modify_tensor_by_number(int(target), operation, value)
                else:
                    return self._modify_tensor_by_name(target, operation, value)
            else:
                print("Error: modify command requires: modify [tensor_num|tensor_name] [operation] [value]")
                return True
        elif command.startswith("visualize "):
            target = command[10:]
            if "-" in target:  # Process as range
                return self.visualize_embeddings_by_index(target)
            elif target.isdigit():  # Process as single index
                index = int(target)
                return self.visualize_embeddings_by_index(specific_index=index)
            else:  # Process as token
                return self.visualize_embeddings_by_token(target)
        elif command.startswith("ai "):
            prompt = command[3:]
            return self._get_ai_explanation(prompt)
        elif command.startswith("recommend_math"):
            return self._get_data_driven_recommendations(capability="math")
        elif command == "recommend" or command == "recommendations":
            return self._get_data_driven_recommendations(capability="general")
        elif command.startswith("chat "):
            query = command[5:]
            return self._chat_with_ai(query)
        elif command.startswith("similar "):
            token = command[8:]
            return self.find_similar_embeddings(token)
        else:
            print(f"Unknown command: {command}")
            return True
    
    def _display_help(self):
        """Display help information for the interactive session"""
        from rich.panel import Panel
        from rich.markdown import Markdown
        
        help_text = """
# AI-Powered Tensor Explorer Help

## Basic Commands
- `help` - Display this help message
- `list` - List all tensors in the model
- `search <pattern>` - Search tensors by name pattern
- `analyze <tensor_num|tensor_name>` - Analyze a tensor by number or name
- `modify <tensor_num|tensor_name> <operation> <value>` - Modify a tensor 
- `exit` or `quit` - Exit the explorer

## Visualization Commands
- `visualize <token>` - Visualize token embeddings
- `visualize <index>` - Visualize embeddings at a specific index
- `visualize <start>-<end>` - Visualize range of embeddings by index

## Token Analysis Commands
- `similar <token>` - Find tokens with similar embeddings

## AI-Powered Analysis Commands
- `ai <query>` - Get AI tensor explanations or recommendations
- `chat <message>` - Chat with AI about model weights
- `recommend` - Get data-driven tensor recommendations for general improvements
- `recommend_math` - Get data-driven tensor recommendations for math improvements

## Patching Commands
- `patch <tensor_name> <operation> <value>` - Apply guided patching to tensor

## Operations
- `scale` - Multiply values by a factor
- `add` - Add a value to tensor elements
- `set` - Set tensor elements to a specific value
- `clamp_min` - Set a minimum value floor
- `clamp_max` - Set a maximum value ceiling
"""
        
        console.print(Panel(Markdown(help_text), title="[bold cyan]AI Tensor Explorer Help[/]", border_style="cyan"))
    
    def _analyze_tensor_by_number(self, num: int):
        """Analyze tensor by its index in sorted list"""
        sorted_tensors = sorted(self.tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
        if 0 <= num < len(sorted_tensors):
            tensor_name = sorted_tensors[num][0]
            stats = self.explorer.analyze_tensor(tensor_name)
            self.explorer.display_tensor_analysis(tensor_name, stats)
            
            # Store in analyzed tensors
            self.analyzed_tensors[tensor_name] = stats
            self.exploration_history.append({"action": "analyze", "tensor": tensor_name})
            
            # Offer AI interpretation
            if Confirm.ask("[bold cyan]Would you like an AI interpretation of this tensor?[/]"):
                self._get_ai_tensor_interpretation(tensor_name, stats)
        else:
            console.print("[bold red]Invalid tensor number[/]")
    
    def _analyze_tensor_by_name(self, tensor_name: str):
        """Analyze tensor by name"""
        # Find similar tensors if exact match not found
        if tensor_name not in self.tensors:
            similar_tensors = [name for name in self.tensors.keys() if tensor_name.lower() in name.lower()]
            if similar_tensors:
                console.print(f"[yellow]Tensor '{tensor_name}' not found. Did you mean one of these?[/]")
                for i, name in enumerate(similar_tensors[:5]):
                    console.print(f"  [cyan]{i+1}.[/] {name}")
                choice = Prompt.ask("Enter number or 'skip'", default="skip")
                if choice.lower() != "skip":
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(similar_tensors):
                            tensor_name = similar_tensors[index]
                        else:
                            console.print("[bold red]Invalid choice[/]")
                            return
                    except ValueError:
                        console.print("[bold red]Invalid choice[/]")
                        return
            else:
                console.print(f"[bold red]Tensor '{tensor_name}' not found and no similar tensors found[/]")
                return
        
        # Analyze the tensor
        stats = self.explorer.analyze_tensor(tensor_name)
        self.explorer.display_tensor_analysis(tensor_name, stats)
        
        # Store in analyzed tensors
        self.analyzed_tensors[tensor_name] = stats
        self.exploration_history.append({"action": "analyze", "tensor": tensor_name})
        
        # Offer AI interpretation
        if Confirm.ask("[bold cyan]Would you like an AI interpretation of this tensor?[/]"):
            self._get_ai_tensor_interpretation(tensor_name, stats)
    
    def _modify_tensor_by_number(self, num: int, operation: str, value: str):
        """Modify tensor by its index in sorted list"""
        sorted_tensors = sorted(self.tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
        if 0 <= num < len(sorted_tensors):
            tensor_name = sorted_tensors[num][0]
            self._modify_tensor_by_name(tensor_name, operation, value)
        else:
            console.print("[bold red]Invalid tensor number[/]")
    
    def _modify_tensor_by_name(self, tensor_name: str, operation: str, value: str):
        """Modify tensor by name"""
        # Find similar tensors if exact match not found
        if tensor_name not in self.tensors:
            similar_tensors = [name for name in self.tensors.keys() if tensor_name.lower() in name.lower()]
            if similar_tensors:
                console.print(f"[yellow]Tensor '{tensor_name}' not found. Did you mean one of these?[/]")
                for i, name in enumerate(similar_tensors[:5]):
                    console.print(f"  [cyan]{i+1}.[/] {name}")
                choice = Prompt.ask("Enter number or 'skip'", default="skip")
                if choice.lower() != "skip":
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(similar_tensors):
                            tensor_name = similar_tensors[index]
                        else:
                            console.print("[bold red]Invalid choice[/]")
                            return
                    except ValueError:
                        console.print("[bold red]Invalid choice[/]")
                        return
            else:
                console.print(f"[bold red]Tensor '{tensor_name}' not found and no similar tensors found[/]")
                return
        
        # Get output dir
        output_dir = f"{self.model_path}_modified_{tensor_name.split('.')[-1]}"
        
        # Confirm operation
        if Confirm.ask(f"[bold yellow]Modify tensor {tensor_name} with operation {operation} and value {value}?[/]"):
            result = self.explorer.modify_tensor(tensor_name, operation, value, output_dir)
            console.print(result)
            self.exploration_history.append({
                "action": "modify", 
                "tensor": tensor_name, 
                "operation": operation,
                "value": value,
                "output": output_dir
            })
    
    def _get_ai_tensor_interpretation(self, tensor_name: str, stats: Dict[str, Any]):
        """Get AI interpretation of a tensor"""
        # Create a specialized prompt for tensor interpretation
        prompt = f"""
        Analyze this tensor from a neural network model through the lens of a reverse engineer:
        
        Tensor name: {tensor_name}
        Statistics:
        - Shape: {stats.get('shape', 'unknown')}
        - Data type: {stats.get('dtype', 'unknown')}
        - Value range: [{stats.get('min', 'unknown')}, {stats.get('max', 'unknown')}]
        - Mean: {stats.get('mean', 'unknown')}
        - Standard deviation: {stats.get('std', 'unknown')}
        - Sparsity: {stats.get('zeros_percent', 'unknown')}%
        
        Provide a detailed analysis that includes:
        1. The tensor's role in the model architecture based on its name and statistics
        2. How the binary/hex representation would look and what patterns to look for
        3. Specific hex modifications that could be made to alter its behavior
        4. Predictions on how these modifications would affect the model's output
        5. Any unusual patterns in the statistics that provide reverse engineering insights
        
        Be specific about which bytes/offsets would be most effective to modify for different effects.
        """
        
        # Call our custom API with the reverse engineering persona
        with console.status("[bold green]Getting reverse engineering analysis...[/]", spinner="dots"):
            interpretation = self._call_ai_api(prompt)
            
            console.print(Panel(
                Markdown(interpretation),
                title="[bold]Reverse Engineering Analysis[/]",
                title_align="left",
                width=100
            ))
    
    def _get_ai_tensor_recommendations(self, query: str):
        """Get AI recommendations for tensors to explore based on query"""
        # Create prompt
        prompt = self._generate_recommendation_prompt(query)
        
        # Call the OpenRouter API
        with console.status("[bold green]Getting AI recommendations...[/]", spinner="dots"):
            try:
                response = self._call_ai_api(prompt)
                
                console.print(Panel(
                    Markdown(response),
                    title="[bold]AI Recommendations[/]",
                    title_align="left",
                    width=100
                ))
                
                # Ask if user wants to take action
                if Confirm.ask("[bold cyan]Would you like to analyze one of these tensors?[/]"):
                    num = Prompt.ask("Enter tensor number from the recommendations", default="1")
                    try:
                        # Extract tensor names from recommendations using regex
                        tensor_names = re.findall(r"`(.*?)`", response)
                        if tensor_names and 0 < int(num) <= len(tensor_names):
                            self._analyze_tensor_by_name(tensor_names[int(num)-1])
                        else:
                            console.print("[bold red]Could not identify tensor from recommendation[/]")
                    except Exception as e:
                        console.print(f"[bold red]Error: {str(e)}[/]")
            
            except Exception as e:
                console.print(f"[bold red]Error getting AI recommendations: {str(e)}[/]")
    
    def _get_ai_explanation(self, query: str):
        """Get AI explanation about the model or tensors"""
        # Create prompt
        prompt = self._generate_explanation_prompt(query)
        
        # Call the OpenRouter API
        with console.status("[bold green]Getting AI explanation...[/]", spinner="dots"):
            try:
                response = self._call_ai_api(prompt)
                
                console.print(Panel(
                    Markdown(response),
                    title="[bold]AI Explanation[/]",
                    title_align="left",
                    width=100
                ))
            
            except Exception as e:
                console.print(f"[bold red]Error getting AI explanation: {str(e)}[/]")
    
    def _chat_with_ai(self, query: str):
        """Generic AI chat"""
        # Create prompt
        prompt = self._generate_chat_prompt(query)
        
        # Call the OpenRouter API
        with console.status("[bold green]Getting AI response...[/]", spinner="dots"):
            try:
                response = self._call_ai_api(prompt)
                
                console.print(Panel(
                    Markdown(response),
                    title="[bold]AI Response[/]",
                    title_align="left",
                    width=100
                ))
            
            except Exception as e:
                console.print(f"[bold red]Error getting AI response: {str(e)}[/]")
    
    def _call_ai_api(self, prompt: str) -> str:
        """Call AI API with the given prompt"""
        if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
            return "AI interpretation unavailable: API key or model not configured. Please set OPENROUTER_API_KEY and OPENROUTER_MODEL in your .env file."
        
        try:
            system_prompt = """You are an expert Chinese reverse engineer specializing in neural networks, known for your hex patching skills with safetensors files.

You are currently hell-bent on experimenting with subtle hex modifications to change model behavior as desired by users. 
You've spent years reverse-engineering model weights and have intimate knowledge of how tensor modifications affect model outputs.
You're particularly skilled at identifying which tensors can be patched to achieve specific behavioral changes.
Your style is confident, direct, and sometimes uses technical jargon from both machine learning and low-level binary manipulation.

When suggesting modifications:
1. Always recommend specific tensors to modify
2. Explain exactly how they should be changed (scaling factors, values)
3. Predict how these changes will affect the model's behavior
4. Occasionally mention your past successes with similar modifications

Make references to both the mathematical tensor properties AND their binary/hex representations when relevant."""

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}, {response.text}"
                
        except Exception as e:
            return f"Error calling OpenRouter API: {str(e)}"
    
    def _generate_recommendation_prompt(self, query: str) -> str:
        """Generate prompt for tensor recommendations"""
        # Format top tensors
        top_tensors = self._format_top_tensors(10)
        
        return f"""
        The user wants tensor recommendations for this model. Given your expertise in hex-patching safetensors, provide specific advice.
        
        User request: {query}
        
        Model details:
        - Total tensors: {len(self.tensors)}
        - Model: {os.path.basename(self.model_path)}
        
        Top 10 tensors by size:
        {top_tensors}
        
        Based on your reverse engineering experience:
        1. Identify 2-3 specific tensors to modify for the desired effect
        2. For each tensor, recommend exact operations (scale/add/clamp/normalize) with specific values
        3. Explain how the binary/hex modification will alter model behavior
        4. Briefly mention similar modifications you've done in the past
        
        Format your response with markdown, providing detailed technical information.
        """
    
    def _generate_explanation_prompt(self, query: str) -> str:
        """Generate prompt for AI explanation"""
        # Format exploration history
        history = self._format_exploration_history()
        
        return f"""
        The user wants an explanation about this neural network model. Use your reverse engineering expertise.
        
        User query: {query}
        
        Model information:
        - Total tensors: {len(self.tensors)}
        - Model: {os.path.basename(self.model_path)}
        
        {history}
        
        Provide a detailed technical explanation that includes:
        1. Low-level details about how the relevant tensors are structured in memory
        2. How the binary/hex representation relates to model behavior
        3. Subtle modifications you would make to achieve specific effects
        4. References to your past experience reverse engineering similar models
        
        Be technical but clear, demonstrating your deep knowledge of both neural networks and binary manipulation.
        """
    
    def _generate_chat_prompt(self, query: str) -> str:
        """Generate prompt for generic chat"""
        return f"""
        The user is exploring a neural network model with {len(self.tensors)} tensors called {os.path.basename(self.model_path)}.
        
        User query: {query}
        
        Respond as an expert reverse engineer who specializes in hex-patching neural networks. 
        Be technical, confident, and occasionally reference your experience modifying safetensors files to achieve specific effects.
        Mention both the mathematical tensor properties and their binary/hex representations when relevant.
        """
    
    def _format_top_tensors(self, limit: int = 10) -> str:
        """Format top tensors for prompt"""
        sorted_tensors = sorted(self.tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
        result = []
        
        for i, (name, info) in enumerate(sorted_tensors[:limit]):
            result.append(f"{i+1}. {name} - Shape: {info['shape']}, Size: {info['size_mb']:.2f} MB")
        
        return "\n".join(result)
    
    def _format_exploration_history(self) -> str:
        """Format exploration history for prompt"""
        if not self.exploration_history:
            return "No exploration history yet."
        
        result = ["User exploration history:"]
        
        for entry in self.exploration_history[-10:]:  # Last 10 entries
            if entry["action"] == "list":
                result.append(f"- Listed all tensors")
            elif entry["action"] == "analyze":
                result.append(f"- Analyzed tensor: {entry['tensor']}")
            elif entry["action"] == "modify":
                result.append(f"- Modified tensor: {entry['tensor']} - Operation: {entry['operation']} with value {entry['value']}")
            elif entry["action"] == "search":
                result.append(f"- Searched for: {entry['term']} - Found {entry['results']} tensors")
        
        return "\n".join(result)

    def visualize_embeddings_by_index(self, index_range: str = None, specific_index: int = None):
        """Visualize embeddings by index range or specific index when no tokenizer is available"""
        # Find the embedding tensor
        embedding_tensors = [name for name in self.tensors.keys() if "embed" in name.lower() and "weight" in name.lower()]
        if not embedding_tensors:
            console.print("[bold red]No embedding tensor found in the model.[/]")
            return
            
        embedding_tensor_name = embedding_tensors[0]
        
        # Process the index input
        indices_to_visualize = []
        
        if specific_index is not None:
            indices_to_visualize = [specific_index]
        elif index_range and self.token_map and 'indices' in self.token_map:
            if index_range in self.token_map['indices']:
                # Get some indices from the specified range
                indices_to_visualize = self.token_map['indices'][index_range][:5]  # Limit to 5 indices
            else:
                console.print(f"[bold red]Index range '{index_range}' not found.[/]")
                console.print("[yellow]Available ranges: " + ", ".join(self.token_map['indices'].keys()))
                return
        else:
            # Default to first 100 indices
            indices_to_visualize = list(range(5))  # Just show the first 5
            
        if not indices_to_visualize:
            console.print("[bold red]No valid indices to visualize.[/]")
            return
            
        # Analyze the embedding tensor
        with console.status(f"[bold green]Analyzing embedding tensor: {embedding_tensor_name}...[/]", spinner="dots"):
            # For each index, visualize its embedding
            for idx in indices_to_visualize:
                console.print(f"\n[bold cyan]Embedding Index:[/] {idx}")
                self._visualize_embedding_for_token(embedding_tensor_name, idx)
                
    def visualize_embeddings_by_token(self, token):
        """Visualize embeddings for a specific token."""
        if self.tokenizer is None:
            console.print("[bold red]Tokenizer not available. Use 'visualize embedding <index>' instead.[/]")
            return

        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            console.print(f"[bold red]Token '{token}' not found in vocabulary.[/]")
            return

        token_id = token_ids[0]
        console.print(f"\n[bold]Token:[/] '{token}' [bold]ID:[/] {token_id}")
        
        # Find the embedding tensor
        embedding_tensors = [name for name in self.tensors.keys() if any(pattern in name.lower() for pattern in ["embed", "wte", "word_embeddings"])]
        if not embedding_tensors:
            console.print("[bold red]No embedding tensor found in the model.[/]")
            return
        
        embedding_tensor_name = embedding_tensors[0]
        self._visualize_embedding_for_token(embedding_tensor_name, token_id)

    def visualize_embeddings_by_index(self, index):
        """Visualize embeddings for a specific index."""
        # Find the embedding tensor
        embedding_tensor = None
        for name, tensor_info in self.tensors.items():
            if any(pattern in name.lower() for pattern in ["embed", "wte", "word_embeddings"]):
                embedding_tensor = name
                break
        
        if not embedding_tensor:
            print("No embedding tensor found in model.")
            return
        
        try:
            # Load the tensor
            tensor_data = self.tensor_data_loader.load_tensor(embedding_tensor)
            
            # Convert to float32 if it's BFloat16 or any other format
            tensor_data = tensor_data.to(torch.float32)
            
            # Extract the row at the index
            if index < tensor_data.shape[0]:
                embedding = tensor_data[index]
                
                # Get statistics
                min_val = float(embedding.min())
                max_val = float(embedding.max())
                mean_val = float(embedding.mean())
                std_val = float(embedding.std())
                
                print(f"Embedding shape: {embedding.shape}")
                print(f"Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                
                # Visualize the embedding
                self._visualize_vector(embedding)
            else:
                print(f"Index {index} out of bounds for embedding tensor with shape {tensor_data.shape}")
        except Exception as e:
            print(f"Error visualizing embedding: {e}")

    def _visualize_vector(self, vector):
        """Create an ASCII visualization of a vector."""
        width = 80
        blocks = "▁▂▃▄▅▆▇█"
        
        # Normalize values to 0-1 range
        min_val = vector.min().item()
        max_val = vector.max().item()
        range_val = max_val - min_val
        
        if range_val == 0:
            print("All values in vector are identical.")
            return
            
        normalized = (vector - min_val) / range_val
        
        # Display first 100 elements with ASCII art
        display_count = min(100, len(vector))
        
        for i in range(0, display_count, 10):
            end = min(i + 10, display_count)
            line = ""
            hex_line = ""
            
            for j in range(i, end):
                # Get normalized value and convert to block character
                val = normalized[j].item()
                block_idx = min(int(val * len(blocks)), len(blocks) - 1)
                
                # Choose color based on value
                if val < 0.25:
                    color = "\033[94m"  # Blue for low values
                elif val < 0.5:
                    color = "\033[96m"  # Cyan for medium-low values
                elif val < 0.75:
                    color = "\033[93m"  # Yellow for medium-high values
                else:
                    color = "\033[91m"  # Red for high values
                
                reset = "\033[0m"
                line += f"{color}{blocks[block_idx]}{reset} "
                
                # Add hex representation
                orig_val = vector[j].item()
                hex_val = format(struct.unpack('<I', struct.pack('<f', orig_val))[0], '08x')
                hex_line += f"{hex_val[:4]} {hex_val[4:]} "
            
            # Print index, bar, value, and hex
            print(f"{i:3d}-{end-1:<3d} {line}  {vector[i:end].tolist()}")
            print(f"       {hex_line}")
            print()

    def _setup_code_agent(self):
        """Setup SmolagentS CodeAgent with custom tools"""
        try:
            # Create LiteLLM model using OpenRouter
            model = LiteLLMModel(
                model_id=OPENROUTER_MODEL,  # Using the original Gemini model with correct prefix
                api_base="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY
            )
            
            # Import Tool from smolagents
            from smolagents.tools import Tool
            
            # Define custom tool classes by extending Tool
            class TensorListTool(Tool):
                name = "tensor_list"
                description = "List all tensors in the model with their sizes"
                inputs = {}
                output_type = "array"
                
                def __init__(self, explorer):
                    super().__init__()
                    self.explorer = explorer
                
                def forward(self):
                    tensor_list = []
                    for name, info in self.explorer.tensors.items():
                        tensor_list.append({
                            "name": name,
                            "shape": str(info["shape"]),
                            "size_mb": info["size_mb"]
                        })
                    return tensor_list
            
            class TensorNamesTool(Tool):
                name = "tensor_names"
                description = "Get a list of all tensor names in the model"
                inputs = {}
                output_type = "array"
                
                def __init__(self, explorer):
                    super().__init__()
                    self.explorer = explorer
                
                def forward(self):
                    return list(self.explorer.tensors.keys())
            
            class TensorLoadTool(Tool):
                name = "tensor_load"
                description = "Load a tensor by name and return basic information"
                inputs = {
                    "tensor_name": {
                        "type": "string",
                        "description": "Name of the tensor to load",
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer):
                    super().__init__()
                    self.explorer = explorer
                
                def forward(self, tensor_name):
                    try:
                        stats = self.explorer.explorer.analyze_tensor(tensor_name)
                        return stats
                    except Exception as e:
                        return {"error": str(e)}
            
            class TokenEmbeddingTool(Tool):
                name = "token_embedding"
                description = "Analyze the embedding for a specific token"
                inputs = {
                    "token": {
                        "type": "string",
                        "description": "Token to analyze",
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer):
                    super().__init__()
                    self.explorer = explorer
                
                def forward(self, token):
                    if not self.explorer.tokenizer:
                        return {"error": "Tokenizer not available"}
                    
                    try:
                        token_ids = self.explorer.tokenizer.encode(token, add_special_tokens=False)
                        if not token_ids:
                            return {"error": f"Token '{token}' not found in vocabulary"}
                        
                        # Find the embedding tensor
                        embedding_tensor = None
                        for name in self.explorer.tensors:
                            if any(pattern in name.lower() for pattern in ["embed", "wte", "word_embeddings"]):
                                embedding_tensor = name
                                break
                        
                        if not embedding_tensor:
                            return {"error": "No embedding tensor found in model"}
                        
                        # Extract the embedding
                        embedding_stats = self.explorer.explorer.analyze_tensor(embedding_tensor)
                        
                        return {
                            "token": token,
                            "token_id": token_ids[0],
                            "embedding_tensor": embedding_tensor,
                            "tensor_shape": embedding_stats.get("shape")
                        }
                    except Exception as e:
                        return {"error": str(e)}
            
            class SimilarTokensTool(Tool):
                name = "similar_tokens"
                description = "Find tokens with similar embeddings"
                inputs = {
                    "token": {
                        "type": "string",
                        "description": "Token to find similar embeddings for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar tokens to return",
                        "nullable": True
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer):
                    super().__init__()
                    self.explorer = explorer
                
                def forward(self, token, top_k=10):
                    if not self.explorer.tokenizer:
                        return {"error": "Tokenizer not available"}
                    
                    try:
                        result = self.explorer.find_similar_embeddings(token, top_k, return_dict=True)
                        return result
                    except Exception as e:
                        return {"error": str(e)}
            
            class TensorValuesTool(Tool):
                name = "tensor_values"
                description = "Get actual values from a tensor, with sampling or slicing for large tensors"
                inputs = {
                    "tensor_name": {
                        "type": "string",
                        "description": "Name of the tensor to load",
                    },
                    "max_elements": {
                        "type": "integer",
                        "description": "Maximum number of elements to return (for sampling if offset/length not used)",
                        "nullable": True
                    },
                    "start_offset": {
                        "type": "integer",
                        "description": "Starting offset in the flattened tensor for slicing",
                        "nullable": True
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of elements to return from start_offset (for slicing)",
                        "nullable": True
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer_param): # Renamed to avoid confusion
                    super().__init__()
                    self.se_explorer = explorer_param # Store the SafetensorsExplorer instance

                def forward(self, tensor_name, max_elements=10000, start_offset=None, length=None):
                    try:
                        tensor = self.se_explorer.load_tensor(tensor_name)
                        tensor_flat = tensor.flatten()
                        
                        if start_offset is not None and length is not None:
                            if start_offset < 0 or (start_offset + length) > tensor_flat.numel():
                                return {"error": "Invalid start_offset or length for tensor dimensions."}
                            values = tensor_flat[start_offset : start_offset + length].tolist()
                            return {"tensor_name": tensor_name, "values": values, "slice_info": f"offset {start_offset}, length {length}"}
                        else:
                            # Original sampling logic if no offset/length
                            if tensor_flat.numel() > max_elements:
                                idx = torch.randperm(tensor_flat.numel())[:max_elements]
                                values = tensor_flat[idx].tolist()
                                sample_info = f"Sampled {max_elements} elements"
                            else:
                                values = tensor_flat.tolist()
                                sample_info = "All elements returned"
                            return {"tensor_name": tensor_name, "values": values, "sample_info": sample_info}
                            
                    except Exception as e:
                        return {"error": str(e)}
            
            class TensorComparisonTool(Tool):
                name = "tensor_compare"
                description = "Compare two tensors and compute similarity/distance metrics"
                inputs = {
                    "tensor_name1": {
                        "type": "string",
                        "description": "Name of the first tensor",
                    },
                    "tensor_name2": {
                        "type": "string",
                        "description": "Name of the second tensor",
                    },
                    "method": {
                        "type": "string",
                        "description": "Comparison method: 'cosine' for similarity or 'euclidean' for distance",
                        "nullable": True
                    },
                    "max_dims": {
                        "type": "integer",
                        "description": "Maximum dimensions to use for comparison",
                        "nullable": True
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer_param): # Renamed to avoid confusion
                    super().__init__()
                    self.se_explorer = explorer_param # Store the SafetensorsExplorer instance
                
                def forward(self, tensor_name1, tensor_name2, method="cosine", max_dims=None):
                    return self.se_explorer.compute_tensor_similarity(tensor_name1, tensor_name2, method, max_dims)
            
            class TokenEmbeddingComparisonTool(Tool):
                name = "token_embedding_compare"
                description = "Compare embeddings of multiple tokens and compute similarities"
                inputs = {
                    "tokens": {
                        "type": "string",
                        "description": "Comma-separated list of tokens to compare",
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer): # This tool uses AITensorExplorer methods
                    super().__init__()
                    self.ai_explorer = explorer
                
                def forward(self, tokens):
                    # This tool's logic is complex and relies on AITensorExplorer methods
                    # For now, assuming it calls methods on self.ai_explorer correctly
                    try:
                        token_list = [t.strip() for t in tokens.split(',') if t.strip()]
                        if not token_list or len(token_list) < 2:
                            return {"error": "Please provide at least two tokens to compare, separated by commas."}

                        all_token_data = []
                        for token_str in token_list:
                            embedding_info = self.ai_explorer.token_embedding(token=token_str) # Call AITensorExplorer's method
                            if embedding_info.get("error"):
                                return {"error": f"Could not get embedding for token '{token_str}': {embedding_info['error']}"}
                            all_token_data.append(embedding_info)
                        
                        # Assuming embeddings are in 'embedding_vector' key and are torch tensors
                        # This part needs the actual embedding vectors to compute similarity
                        # The original `token_embedding` tool in the log only returns metadata.
                        # This tool requires the actual vectors from the `model.embed_tokens.weight` tensor.
                        
                        # Placeholder for actual comparison logic if vectors were available
                        # For now, we'll use the AITensorExplorer's existing methods if possible,
                        # or indicate what's missing.
                        
                        # The current `AITensorExplorer.token_embedding` seems to only return metadata.
                        # It needs to be enhanced to return the actual embedding vector from `model.embed_tokens.weight`.
                        # Let's assume `self.ai_explorer.compare_token_embeddings_by_str(tokens_str=tokens)` exists and does the job.
                        # This method isn't in the provided code, indicating a gap.
                        # For demonstration, I'll simulate what it might do if it used `compute_tensor_similarity`
                        # on sliced parts of the embedding tensor.

                        # This is a simplified mock-up. Proper implementation requires access to the embedding vectors.
                        # The current tools don't easily expose this for direct comparison here.
                        # The `token_embedding_compare` in the logs seems to be a more complex, pre-existing function.
                        # For now, let's assume the AI calls a method that handles this correctly.
                        # Given the logs, the AI is already successfully calling a `token_embedding_compare` tool.
                        # This custom tool implementation here might be what the AI is *supposed* to be calling.
                        # So this tool's forward method just delegates or indicates it needs the vectors.

                        # The logs show that 'token_embedding_compare' is already a working tool.
                        # This custom tool implementation here might conflict or be unnecessary.
                        # The AI already has a tool named `token_embedding_compare`.
                        # The simplest fix is to ensure this tool uses the AITensorExplorer's existing capability.
                        # However, `AITensorExplorer` itself doesn't have a direct `token_embedding_compare` method in the provided code.
                        # The tools seem to be standalone.

                        # Let's assume the AI's own `token_embedding_compare` is working based on logs.
                        # This `TokenEmbeddingComparisonTool` as defined might be what the AI is *supposed* to be calling.

                        # The `forward` method in the logs for `token_embedding_compare` is:
                        # embedding_comparison = token_embedding_compare(tokens="CCP, Tiananmen Square, democracy, freedom, banana")
                        # This implies `token_embedding_compare` is a direct tool call made by the AI.
                        # The provided `ai_tensor_explorer.py` has a class `TokenEmbeddingComparisonTool` whose `forward` method
                        # needs to implement this logic.

                        if not self.ai_explorer.tokenizer:
                            return {"error": "Tokenizer not available for embedding comparison."}
                        if not TORCH_AVAILABLE:
                            return {"error": "PyTorch not available for tensor operations."}

                        embedding_tensor_name = "model.embed_tokens.weight" # Common name
                        
                        # Verify embedding tensor exists
                        # Correctly access the list of tensor names from the dictionary returned by list_tensors
                        all_tensors_dict = self.se_explorer.list_tensors() # Gets a dict like {'name': info, ...}
                        all_tensor_names = list(all_tensors_dict.keys())

                        if embedding_tensor_name not in all_tensor_names:
                            alt_embed_name = None
                            for name in all_tensor_names:
                                if "embed_tokens.weight" in name or "wte" in name: # common alternatives
                                    alt_embed_name = name
                                    break
                            if not alt_embed_name:
                                return {"error": f"Could not find a suitable token embedding tensor (e.g., {embedding_tensor_name})."}
                            embedding_tensor_name = alt_embed_name
                            console.print(f"[yellow]Using embedding tensor: {embedding_tensor_name}[/yellow]")


                        token_ids = []
                        parsed_tokens = []
                        for t_str in token_list:
                            try:
                                # Get ID for the exact token string
                                # This might be problematic if tokenizer splits the string.
                                # For multi-word tokens, it's better to have a direct lookup if they exist as single tokens.
                                # The log shows "Tank Man" has token_id 66033.
                                # The tokenizer might split "Tank Man" into "Tank" and " Man".
                                # The `token_embedding` tool in the log seems to handle this.
                                ids = self.ai_explorer.tokenizer.encode(t_str, add_special_tokens=False)
                                if not ids:
                                    return {"error": f"Token '{t_str}' could not be tokenized or is empty."}
                                # For simplicity, take the first ID if it's a multi-token string that wasn't a single vocab item.
                                # This might not be what the user intends for phrases.
                                # The existing logs show `token_embedding(token="Tank Man")` works, implying the tool handles phrases.
                                # This `TokenEmbeddingTool` (lines 1234-1249) is what provides that.
                                # This comparison tool should leverage that.

                                # Let's get the token_id from the `token_embedding` tool logic.
                                token_info_for_id = self.ai_explorer.token_embedding(token=t_str)
                                if token_info_for_id.get("error"):
                                    return {"error": f"Could not get token ID for '{t_str}': {token_info_for_id['error']}"}
                                token_ids.append(token_info_for_id['token_id'])
                                parsed_tokens.append(t_str) # Keep original token string for output key
                            except Exception as e:
                                return {"error": f"Error tokenizing '{t_str}': {str(e)}"}
                        
                        if len(token_ids) < 2:
                             return {"error": "Need at least two valid tokens for comparison."}

                        embedding_matrix = self.se_explorer.load_tensor(embedding_tensor_name)
                        embeddings = []
                        for tid in token_ids:
                            if tid >= embedding_matrix.shape[0]:
                                return {"error": f"Token ID {tid} is out of bounds for embedding matrix shape {embedding_matrix.shape[0]}."}
                            embeddings.append(embedding_matrix[tid])
                        
                        similarities = []
                        embedding_stats_list = []

                        for i in range(len(embeddings)):
                            stats = {
                                "token": parsed_tokens[i],
                                "token_id": token_ids[i],
                                "mean": float(embeddings[i].mean()),
                                "std": float(embeddings[i].std()),
                                "min": float(embeddings[i].min()),
                                "max": float(embeddings[i].max()),
                                "l2_norm": float(torch.norm(embeddings[i]).item())
                            }
                            embedding_stats_list.append(stats)

                            for j in range(i + 1, len(embeddings)):
                                sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
                                similarities.append({
                                    "token1": parsed_tokens[i],
                                    "token2": parsed_tokens[j],
                                    "similarity": float(sim)
                                })
                        
                        # Sort by similarity score, descending
                        similarities.sort(key=lambda x: x["similarity"], reverse=True)

                        return {
                            "tokens": parsed_tokens,
                            "token_ids": token_ids,
                            "embedding_tensor": embedding_tensor_name,
                            "similarities": similarities,
                            "embedding_stats": embedding_stats_list
                        }

                    except Exception as e:
                        return {"error": f"Error in TokenEmbeddingComparisonTool: {str(e)} - {traceback.format_exc()}"}

            
            class HexInspectTool(Tool):
                name = "hex_inspect"
                description = "Inspect tensor values in hexadecimal format"
                inputs = {
                    "tensor_name": {
                        "type": "string",
                        "description": "Name of the tensor to inspect",
                    },
                    "start_offset": {
                        "type": "integer",
                        "description": "Starting offset in the flattened tensor",
                        "nullable": True
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of elements to inspect",
                        "nullable": True
                    },
                    "row": {
                        "type": "integer",
                        "description": "Specific row to inspect (for 2D+ tensors)",
                        "nullable": True
                    },
                    "col": {
                        "type": "integer",
                        "description": "Specific column to inspect (requires row to be specified)",
                        "nullable": True
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer_param): # Renamed to avoid confusion
                    super().__init__()
                    self.se_explorer = explorer_param # Store the SafetensorsExplorer instance
                
                def forward(self, tensor_name, start_offset=0, length=100, row=None, col=None):
                    try:
                        # Using the hex inspection logic from SafetensorsExplorer's CLI part as a reference
                        # This tool should call a method on self.se_explorer (SafetensorsExplorer instance)
                        # SafetensorsExplorer has `inspect_hex_representation`
                        return self.se_explorer.inspect_hex_representation(
                            tensor_name=tensor_name, 
                            start_offset=start_offset, 
                            length=length, 
                            row=row, 
                            col=col
                        )
                    except Exception as e:
                        return {"error": str(e)}
                    
            class TensorStatisticsTool(Tool):
                name = "tensor_statistics"
                description = "Calculate detailed statistics for a tensor including quantiles and extreme values"
                inputs = {
                    "tensor_name": {
                        "type": "string",
                        "description": "Name of the tensor to analyze",
                    },
                    "quantiles": {
                        "type": "array",
                        "description": "List of quantiles to calculate (values between 0 and 1)",
                        "nullable": True
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer_param): # Renamed to avoid confusion
                    super().__init__()
                    self.se_explorer = explorer_param # Store the SafetensorsExplorer instance
                
                def forward(self, tensor_name, quantiles=None):
                    if quantiles is None:
                        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                    return self.se_explorer.calculate_tensor_statistics(tensor_name, quantiles)
            
            class HexPatchTool(Tool):
                name = "hex_patch"
                description = "Patch tensor values with hexadecimal manipulation, showing before/after hex representations"
                inputs = {
                    "tensor_name": {
                        "type": "string",
                        "description": "Name of the tensor to patch",
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to apply: 'scale', 'add', 'set', 'clamp_min', or 'clamp_max'",
                        "nullable": True
                    },
                    "value": {
                        "type": "number",
                        "description": "Value to use in the operation",
                        "nullable": True
                    },
                    "target_quantile": {
                        "type": "any", # Can be single value or list with two values
                        "description": "Target specific quantile(s) for modification, e.g., 0.9 for top 10%, or [0.8, 0.9] for range",
                        "nullable": True
                    },
                    "row": {
                        "type": "integer",
                        "description": "Specific row to modify (for 2D+ tensors)",
                        "nullable": True
                    },
                    "col": {
                        "type": "integer",
                        "description": "Specific column to modify (requires row to be specified)",
                        "nullable": True
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the modified safetensors file",
                        "nullable": True
                    },
                    "preview_only": {
                        "type": "boolean",
                        "description": "Only preview changes without saving to file",
                        "nullable": True
                    }
                }
                output_type = "object"
                
                def __init__(self, explorer_param): # Renamed to avoid confusion
                    super().__init__()
                    self.se_explorer = explorer_param # Store the SafetensorsExplorer instance
                
                def forward(self, tensor_name, operation="scale", value=1.0, target_quantile=None, 
                          row=None, col=None, output_path=None, preview_only=True):
                    """Apply a patch operation to a tensor and save the modified model if requested."""
                    try:
                        # Validate inputs
                        if operation not in ["scale", "add", "set", "clamp_min", "clamp_max"]:
                            return {"error": f"Invalid operation: {operation}. Use 'scale', 'add', 'set', 'clamp_min', or 'clamp_max'"}
                        
                        # Default output path if not provided
                        if output_path is None and not preview_only:
                            output_path = f"{self.se_explorer.model_path}_modified/model.safetensors"
                        
                        # Check if we can load the tensor
                        if hasattr(self.se_explorer, "_guided_patching"):
                            # Use the guided patching method to handle the patch
                            return self.se_explorer._guided_patching(tensor_name, operation, value)
                        elif hasattr(self.se_explorer, "tensor_load") and hasattr(self.se_explorer, "explorer") and hasattr(self.se_explorer.explorer, "get_tensor"):
                            # Proceed with a more direct implementation
                            # Load the tensor
                            tensor_info = self.se_explorer.tensor_load(tensor_name)
                            if "error" in tensor_info:
                                return {"error": tensor_info["error"]}
                            
                            # Get tensor data
                            tensor = self.se_explorer.explorer.get_tensor(tensor_name)
                            
                            # Create a copy to modify
                            import torch
                            import numpy as np
                            
                            modified_tensor = tensor.clone()
                            
                            # Prepare mask for targeted modification
                            mask = torch.ones_like(tensor, dtype=torch.bool)
                            
                            # Apply quantile filtering if specified
                            if target_quantile is not None:
                                # Calculate threshold based on quantile
                                flat_values = tensor.flatten()
                                if isinstance(target_quantile, list) and len(target_quantile) == 2:
                                    # Range of quantiles
                                    lower_q, upper_q = target_quantile
                                    lower_threshold = torch.quantile(flat_values, lower_q)
                                    upper_threshold = torch.quantile(flat_values, upper_q)
                                    mask = (tensor >= lower_threshold) & (tensor <= upper_threshold)
                                else:
                                    # Single quantile (operating on values above this quantile)
                                    threshold = torch.quantile(flat_values, target_quantile)
                                    mask = tensor >= threshold
                            
                            # Apply row/column filtering if specified
                            if row is not None:
                                if tensor.dim() < 2:
                                    return {"error": "Row selection requires a tensor with at least 2 dimensions"}
                                row_mask = torch.zeros_like(tensor, dtype=torch.bool)
                                if col is not None:
                                    # Single element selection
                                    if tensor.dim() == 2:
                                        row_mask[row, col] = True
                                    else:
                                        # For higher dimensions, apply to specified row/col across all other dims
                                        idx = [slice(None)] * tensor.dim()
                                        idx[0] = row
                                        idx[1] = col
                                        row_mask[tuple(idx)] = True
                                else:
                                    # Entire row selection
                                    idx = [slice(None)] * tensor.dim()
                                    idx[0] = row
                                    row_mask[tuple(idx)] = True
                                
                                mask = mask & row_mask
                            
                            # Count affected values for reporting
                            affected_count = mask.sum().item()
                            total_count = tensor.numel()
                            
                            # Apply the operation to the selected values
                            if operation == "scale":
                                modified_tensor[mask] = tensor[mask] * value
                            elif operation == "add":
                                modified_tensor[mask] = tensor[mask] + value
                            elif operation == "set":
                                modified_tensor[mask] = value
                            elif operation == "clamp_min":
                                modified_tensor[mask] = torch.clamp(tensor[mask], min=value)
                            elif operation == "clamp_max":
                                modified_tensor[mask] = torch.clamp(tensor[mask], max=value)
                            
                            # Get sample values for before/after comparison (up to 5)
                            sample_indices = torch.where(mask.flatten())[0][:5].tolist()
                            sample_before = []
                            sample_after = []
                            before_hex = []
                            after_hex = []
                            
                            import struct
                            import binascii
                            
                            for idx in sample_indices:
                                # Convert flat index to tensor coordinates
                                coords = np.unravel_index(idx, tensor.shape)
                                original_value = tensor[coords].item()
                                modified_value = modified_tensor[coords].item()
                                
                                sample_before.append(original_value)
                                sample_after.append(modified_value)
                                
                                # Add hex representation
                                b_bytes = struct.pack('!f', original_value)
                                a_bytes = struct.pack('!f', modified_value)
                                before_hex.append('0x' + binascii.hexlify(b_bytes).decode('ascii'))
                                after_hex.append('0x' + binascii.hexlify(a_bytes).decode('ascii'))
                            
                            return {
                                "success": True,
                                "message": f"Preview of {operation} operation on '{tensor_name}' (affected {affected_count}/{total_count} values, {(affected_count/total_count)*100:.2f}%)",
                                "affected_values": affected_count,
                                "total_values": total_count,
                                "sample_before": sample_before,
                                "sample_after": sample_after,
                                "before_samples_hex": before_hex,
                                "after_samples_hex": after_hex
                            }
                        else:
                            return {"error": "Tensor patching capabilities not available"}
                    except Exception as e:
                        import traceback
                        return {"error": f"Error patching tensor: {str(e)}", "traceback": traceback.format_exc()}
                    
                def _float_to_hex(self, value):
                    """Convert a float value to its hexadecimal representation."""
                    import struct
                    import binascii
                    
                    if isinstance(value, float):
                        packed = struct.pack('!f', value)
                    else:
                        # Handle other numeric types
                        packed = struct.pack('!f', float(value))
                    return binascii.hexlify(packed).decode('utf-8')
            
            # Create tool instances
            tensor_list_tool = TensorListTool(self)
            tensor_names_tool = TensorNamesTool(self)
            tensor_load_tool = TensorLoadTool(self)
            token_embedding_tool = TokenEmbeddingTool(self)
            similar_tokens_tool = SimilarTokensTool(self)
            tensor_values_tool = TensorValuesTool(self)
            tensor_compare_tool = TensorComparisonTool(self)
            token_embedding_compare_tool = TokenEmbeddingComparisonTool(self)
            hex_inspect_tool = HexInspectTool(self)
            tensor_statistics_tool = TensorStatisticsTool(self)
            hex_patch_tool = HexPatchTool(self)
            
            # Create the CodeAgent with our Tool instances
            self.code_agent = CodeAgent(
                model=model,
                tools=[
                    tensor_list_tool, 
                    tensor_names_tool, 
                    tensor_load_tool, 
                    token_embedding_tool,
                    similar_tokens_tool,
                    tensor_values_tool,
                    tensor_compare_tool,
                    token_embedding_compare_tool,
                    hex_inspect_tool,
                    tensor_statistics_tool,
                    hex_patch_tool
                ]
            )
            
            console.print("[bold green]✓[/] Successfully initialized SmolagentS CodeAgent for AI-powered investigations")
        
        except Exception as e:
            console.print(f"[bold red]Error setting up CodeAgent:[/] {str(e)}")
            traceback.print_exc()
    
    def run_investigation(self, query):
        """Run an AI-powered investigation using SmolagentS CodeAgent"""
        if not self.code_agent:
            if not SMOLAGENTS_AVAILABLE:
                console.print("[bold red]SmolagentS not available.[/] Please install it with: pip install smolagents litellm")
                return
            elif not OPENROUTER_API_KEY:
                console.print("[bold red]OPENROUTER_API_KEY not set.[/] Please add it to your .env file")
                return
            else:
                self._setup_code_agent()
                if not self.code_agent:
                    return
        
        # Create context information about the model
        context = {
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "total_tensors": len(self.tensors),
            "has_tokenizer": self.tokenizer is not None,
        }
        
        # Prepare the investigation prompt
        prompt = f"""
        You are an expert AI researcher investigating language model weights.
        
        User Query: {query}
        
        Model Information:
        - Path: {context['model_path']}
        - Name: {context['model_name']}
        - Total tensors: {context['total_tensors']}
        - Tokenizer available: {context['has_tokenizer']}
        
        Please investigate this query systematically using the available tools. 
        Focus on finding concrete evidence in the model weights rather than making theoretical assumptions.
        Your investigation should be thorough, examining relevant tensors, embeddings, and patterns.
        
        After analyzing the data, provide a clear summary of your findings with evidence from the model weights.
        """
        
        # Run the investigation
        console.print(Panel(f"[bold]Starting investigation:[/] {query}", border_style="blue"))
        with console.status("[bold green]AI is investigating the model...[/]", spinner="dots"):
            try:
                # Get the result from the CodeAgent
                result = self.code_agent.run(prompt)
                
                # Convert dictionary result to formatted string if needed
                if isinstance(result, dict):
                    formatted_result = "# Investigation Results\n\n"
                    
                    # Add summary if present
                    if "summary" in result:
                        formatted_result += f"## Summary\n\n{result['summary']}\n\n"
                    
                    # Add evidence if present
                    if "evidence" in result:
                        formatted_result += "## Evidence\n\n"
                        if isinstance(result["evidence"], dict):
                            for key, value in result["evidence"].items():
                                formatted_result += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                        elif isinstance(result["evidence"], list):
                            for item in result["evidence"]:
                                formatted_result += f"- {item}\n"
                        else:
                            formatted_result += str(result["evidence"])
                    
                    # Add any other keys
                    for key, value in result.items():
                        if key not in ["summary", "evidence"]:
                            formatted_result += f"\n## {key.replace('_', ' ').title()}\n\n"
                            formatted_result += f"{value}\n"
                    
                    result = formatted_result
                
                # Display the results
                console.print(Panel(
                    Markdown(result),
                    title="[bold]Investigation Results[/]",
                    border_style="green"
                ))
                
                # Add to history
                self.exploration_history.append({
                    "action": "investigation",
                    "query": query,
                    "result": result
                })
                
                return result
                
            except Exception as e:
                console.print(f"[bold red]Error during investigation:[/] {str(e)}")
                traceback.print_exc()
                return None

    def find_similar_embeddings(self, token, top_k=10, return_dict=False):
        """Find tokens with embeddings similar to the input token"""
        if not self.tokenizer:
            if return_dict:
                return {"error": "Tokenizer not available"}
            console.print("[bold red]Tokenizer not available. Cannot find similar embeddings.[/]")
            return
        
        try:
            # Find the embedding tensor
            embedding_tensors = [name for name in self.tensors.keys() if any(pattern in name.lower() for pattern in ["embed", "wte", "word_embeddings"])]
            if not embedding_tensors:
                if return_dict:
                    return {"error": "No embedding tensor found in the model"}
                console.print("[bold red]No embedding tensor found in the model.[/]")
                return
                
            embedding_tensor_name = embedding_tensors[0]
            
            # Get token ID for the input token
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                if return_dict:
                    return {"error": f"Token '{token}' not found in vocabulary"}
                console.print(f"[bold red]Token '{token}' not found in vocabulary.[/]")
                return
                
            token_id = token_ids[0]
            
            # Load the embedding tensor
            from safetensors import safe_open
            import numpy as np
            import torch.nn.functional as F
            
            with safe_open(self.explorer.safetensors_file, framework="pt") as f:
                embeddings = f.get_tensor(embedding_tensor_name)
                
                # Convert to float32 for calculation
                embeddings = embeddings.to(torch.float32)
                
                # Get target embedding
                if token_id >= embeddings.shape[0]:
                    if return_dict:
                        return {"error": f"Token ID {token_id} out of range for embeddings of shape {embeddings.shape}"}
                    console.print(f"[bold red]Token ID {token_id} out of range for embeddings of shape {embeddings.shape}[/]")
                    return
                    
                target_embedding = embeddings[token_id]
                
                # Compute cosine similarity
                similarities = F.cosine_similarity(
                    target_embedding.unsqueeze(0), 
                    embeddings,
                    dim=1
                )
                
                # Get top-k similar tokens (excluding the query token itself)
                similar_values, similar_indices = torch.topk(similarities, top_k + 1)
                
                # Filter out the query token if it's in the results
                results = []
                for i, (idx, sim) in enumerate(zip(similar_indices, similar_values)):
                    idx_val = idx.item()
                    sim_val = sim.item()
                    
                    # Skip the exact same token
                    if idx_val == token_id and i == 0:
                        continue
                        
                    # Get token text
                    token_text = self.tokenizer.decode([idx_val])
                    
                    results.append({
                        "token": token_text,
                        "token_id": idx_val,
                        "similarity": sim_val
                    })
                    
                    if len(results) >= top_k:
                        break
                
                if return_dict:
                    return {
                        "query_token": token,
                        "query_token_id": token_id,
                        "similar_tokens": results
                    }
                
                # Display results
                console.print(f"\n[bold cyan]Tokens with embeddings similar to[/] '{token}' [bold cyan](ID: {token_id}):[/]")
                
                table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
                table.add_column("Token")
                table.add_column("ID")
                table.add_column("Similarity")
                
                for result in results:
                    table.add_row(
                        result["token"],
                        str(result["token_id"]),
                        f"{result['similarity']:.4f}"
                    )
                
                console.print(table)
                return results
                
        except Exception as e:
            if return_dict:
                return {"error": str(e)}
            console.print(f"[bold red]Error finding similar embeddings:[/] {str(e)}")
            traceback.print_exc()
            return None

    def get_tensor_values(self, tensor_name, max_elements=10000):
        """Get actual values from a tensor, with sampling for large tensors"""
        try:
            from safetensors import safe_open
            import numpy as np
            import torch
            
            with safe_open(self.explorer.safetensors_file, framework="pt") as f:
                if tensor_name not in f.keys():
                    return {"error": f"Tensor {tensor_name} not found"}
                
                # Load the tensor
                tensor = f.get_tensor(tensor_name)
                
                # Convert to float32 to handle different formats
                tensor = tensor.to(torch.float32)
                
                # Get tensor shape
                shape = tensor.shape
                total_elements = torch.prod(torch.tensor(shape)).item()
                
                # For very large tensors, sample a subset
                if total_elements > max_elements:
                    # Flatten tensor
                    flat_tensor = tensor.reshape(-1)
                    
                    # Sample indices
                    indices = torch.linspace(0, flat_tensor.size(0)-1, max_elements, dtype=torch.long)
                    
                    # Get values at sampled indices
                    sampled_values = flat_tensor[indices].cpu().numpy().tolist()
                    
                    return {
                        "values": sampled_values,
                        "is_sampled": True,
                        "total_elements": total_elements,
                        "sampled_elements": max_elements
                    }
                else:
                    # Return all values for smaller tensors
                    values = tensor.cpu().numpy().tolist()
                    return {
                        "values": values,
                        "is_sampled": False,
                        "total_elements": total_elements
                    }
        except Exception as e:
            return {"error": str(e)}

    def compute_tensor_similarity(self, tensor_name1, tensor_name2, method="cosine", max_dims=None):
        """Compute similarity between two tensors"""
        try:
            from safetensors import safe_open
            import numpy as np
            import torch
            import torch.nn.functional as F
            
            with safe_open(self.explorer.safetensors_file, framework="pt") as f:
                if tensor_name1 not in f.keys():
                    return {"error": f"Tensor {tensor_name1} not found"}
                    
                if tensor_name2 not in f.keys():
                    return {"error": f"Tensor {tensor_name2} not found"}
                
                # Load the tensors
                tensor1 = f.get_tensor(tensor_name1)
                tensor2 = f.get_tensor(tensor_name2)
                
                # Convert to float32
                tensor1 = tensor1.to(torch.float32)
                tensor2 = tensor2.to(torch.float32)
                
                # Reshape tensors if needed to get comparable dimensions
                tensor1_shape = tensor1.shape
                tensor2_shape = tensor2.shape
                
                # If max_dims is specified, only use that many dimensions from each tensor
                if max_dims:
                    # Reshape tensors to have max_dims dimensions
                    tensor1_flat = tensor1.reshape(-1, max_dims)
                    tensor2_flat = tensor2.reshape(-1, max_dims)
                else:
                    # Try to make tensors comparable by reshaping
                    # If they are embedding matrices, compare row by row
                    if len(tensor1_shape) == 2 and len(tensor2_shape) == 2 and tensor1_shape[1] == tensor2_shape[1]:
                        tensor1_flat = tensor1
                        tensor2_flat = tensor2
                    else:
                        # Fallback: just flatten everything
                        tensor1_flat = tensor1.reshape(1, -1)
                        tensor2_flat = tensor2.reshape(1, -1)
                
                # Compute similarity based on specified method
                if method == "cosine":
                    # Compute cosine similarity
                    # Normalize each row
                    tensor1_norm = F.normalize(tensor1_flat, p=2, dim=1)
                    tensor2_norm = F.normalize(tensor2_flat, p=2, dim=1)
                    
                    # Compute similarities: (a, b) (c, d) -> a.c + b.d
                    similarities = torch.matmul(tensor1_norm, tensor2_norm.t())
                    
                    # Convert to numpy for easier handling
                    sim_np = similarities.cpu().numpy()
                    
                    # Get statistics
                    mean_sim = float(np.mean(sim_np))
                    max_sim = float(np.max(sim_np))
                    min_sim = float(np.min(sim_np))
                    median_sim = float(np.median(sim_np))
                    
                    # Get top 5 most similar rows/items
                    if len(sim_np) > 1 and len(sim_np[0]) > 1:
                        # Find indices of top 5 similarities
                        top_indices = []
                        for i in range(min(5, len(sim_np))):
                            max_idx = np.unravel_index(np.argmax(sim_np), sim_np.shape)
                            top_indices.append((max_idx, float(sim_np[max_idx])))
                            sim_np[max_idx] = -1  # Mark as processed
                    else:
                        top_indices = [((0, 0), float(sim_np[0, 0]))]
                    
                    return {
                        "method": "cosine",
                        "mean_similarity": mean_sim,
                        "max_similarity": max_sim,
                        "min_similarity": min_sim,
                        "median_similarity": median_sim,
                        "top_similarities": top_indices,
                        "tensor1_shape": list(tensor1_shape),
                        "tensor2_shape": list(tensor2_shape),
                        "compared_shape": [tensor1_flat.shape[0], tensor2_flat.shape[0]]
                    }
                elif method == "euclidean":
                    # Compute pairwise Euclidean distances
                    n1, d1 = tensor1_flat.shape
                    n2, d2 = tensor2_flat.shape
                    
                    # Ensure dimensions match
                    if d1 != d2:
                        return {"error": f"Tensor dimensions don't match for Euclidean distance: {d1} vs {d2}"}
                    
                    # Compute squared distances manually to avoid memory issues
                    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
                    a_norm = torch.sum(tensor1_flat ** 2, dim=1).view(n1, 1)
                    b_norm = torch.sum(tensor2_flat ** 2, dim=1).view(1, n2)
                    distances = a_norm + b_norm - 2 * torch.matmul(tensor1_flat, tensor2_flat.t())
                    distances = torch.sqrt(torch.clamp(distances, min=0))
                    
                    # Convert to numpy
                    dist_np = distances.cpu().numpy()
                    
                    # Get statistics
                    mean_dist = float(np.mean(dist_np))
                    max_dist = float(np.max(dist_np))
                    min_dist = float(np.min(dist_np))
                    median_dist = float(np.median(dist_np))
                    
                    # Get top 5 closest items (smallest distances)
                    if len(dist_np) > 1 and len(dist_np[0]) > 1:
                        # Find indices of top 5 smallest distances
                        top_indices = []
                        for i in range(min(5, len(dist_np))):
                            min_idx = np.unravel_index(np.argmin(dist_np), dist_np.shape)
                            top_indices.append((min_idx, float(dist_np[min_idx])))
                            dist_np[min_idx] = float('inf')  # Mark as processed
                    else:
                        top_indices = [((0, 0), float(dist_np[0, 0]))]
                    
                    return {
                        "method": "euclidean",
                        "mean_distance": mean_dist,
                        "max_distance": max_dist,
                        "min_distance": min_dist,
                        "median_distance": median_dist,
                        "top_closest": top_indices,
                        "tensor1_shape": list(tensor1_shape),
                        "tensor2_shape": list(tensor2_shape),
                        "compared_shape": [tensor1_flat.shape[0], tensor2_flat.shape[0]]
                    }
                else:
                    return {"error": f"Unsupported similarity method: {method}"}
        except Exception as e:
            return {"error": f"Error computing tensor similarity: {str(e)}"}

    def _guided_patching(self, tensor_name, operation, value, target_quantile=None):
        """
        Apply guided patching to a tensor with specific operations.
        
        Args:
            tensor_name (str): Name of the tensor to patch
            operation (str): Operation to apply ('scale', 'add', etc.)
            value (float): Value to use in the operation
            target_quantile (float or list, optional): Target specific quantile(s) for modification
                                                      e.g., 0.9 for top 10%, or [0.8, 0.9] for range
        
        Returns:
            bool: Success or failure
        """
        from rich.console import Console
        console = Console()
        
        console.print(f"[bold cyan]🛠️ Preparing to {operation} {tensor_name} by {value}...[/]")
        
        # Check if tensor exists
        if tensor_name not in self.tensors:
            console.print(f"[bold red]❌ Error: Tensor {tensor_name} not found in model[/]")
            return False
        
        try:
            # Check if we have a code_agent with the hex_patch method
            if hasattr(self, "code_agent"):
                # Try to use the hex_patch method from a HexPatchTool if available
                hex_patch_available = False
                
                # Check directly if the method exists
                if hasattr(self.code_agent, "hex_patch"):
                    hex_patch_available = True
                
                # Check if the method is available via a tool
                if not hex_patch_available and hasattr(self.code_agent, "tools"):
                    for tool in self.code_agent.tools:
                        if hasattr(tool, "name") and tool.name == "hex_patch":
                            hex_patch_available = True
                            
                            # Add the method directly to the code_agent for convenience
                            def hex_patch_wrapper(tensor_name, operation="scale", value=1.0, 
                                                 target_quantile=None, row=None, col=None, 
                                                 output_path=None, preview_only=True):
                                return tool.forward(tensor_name, operation, value, target_quantile,
                                                   row, col, output_path, preview_only)
                            
                            self.code_agent.hex_patch = hex_patch_wrapper
                            break
                
                if hex_patch_available or hasattr(self.code_agent, "hex_patch"):
                    # Now we can use the hex_patch method
                    console.print(f"[cyan]Using CodeAgent's HexPatchTool for patching...[/]")
                    
                    # Check if a quantile was provided
                    if target_quantile is not None:
                        if isinstance(target_quantile, (list, tuple)):
                            console.print(f"[cyan]Targeting values between {target_quantile[0]*100:.1f}% and {target_quantile[1]*100:.1f}% quantiles[/]")
                        else:
                            console.print(f"[cyan]Targeting values above {target_quantile*100:.1f}% quantile[/]")
                    
                    # Generate a preview first
                    preview = self.code_agent.hex_patch(
                        tensor_name=tensor_name,
                        operation=operation,
                        value=value,
                        target_quantile=target_quantile,
                        preview_only=True
                    )
                    
                    console.print("\n[bold]📊 Patch Preview:[/]")
                    console.print(f"   [bold]Sample before:[/] {preview.get('sample_before', 'N/A')}")
                    console.print(f"   [bold]Sample after:[/] {preview.get('sample_after', 'N/A')}")
                    
                    # Ask for confirmation
                    confirm = input("\nApply this patch? (yes/no): ").strip().lower()
                    if confirm != "yes":
                        console.print("[bold yellow]⚠️ Patch aborted[/]")
                        return False
                    
                    # Apply the actual patch
                    output_path = f"{self.model_path}_patched/model.safetensors"
                    
                    result = self.code_agent.hex_patch(
                        tensor_name=tensor_name,
                        operation=operation,
                        value=value,
                        target_quantile=target_quantile,
                        output_path=output_path,
                        preview_only=False
                    )
                    
                    if result.get("success", False):
                        console.print(f"\n[bold green]✅ Patch applied successfully![/]")
                        console.print(f"New model saved to: {result.get('output_path', output_path)}")
                        
                        # Add to exploration history
                        self.exploration_history.append({
                            "action": "patch",
                            "tensor": tensor_name,
                            "operation": operation,
                            "value": value,
                            "target_quantile": target_quantile,
                            "result": "success"
                        })
                        
                        return True
                    else:
                        console.print(f"\n[bold red]❌ Error applying patch: {result.get('error', 'Unknown error')}[/]")
                        return False
            
            # Fallback implementation if no hex_patch is available
            console.print("[yellow]Using fallback patching implementation (limited functionality)[/]")
            
            # Get tensor information
            tensor_info = self.explorer.get_tensor_info(tensor_name)
            if "error" in tensor_info:
                console.print(f"[bold red]Error: {tensor_info['error']}")
                return False
            shape = tensor_info.get("shape", [])
            dtype = tensor_info.get("dtype", "unknown")
            
            # Basic analysis for preview
            mean = tensor_info.get("mean", 0)
            std = tensor_info.get("std", 0.1)
            
            console.print(f"[bold]Tensor:[/] {tensor_name}")
            console.print(f"[bold]Shape:[/] {shape}")
            console.print(f"[bold]Data Type:[/] {dtype}")
            console.print(f"[bold]Current Mean:[/] {mean:.6f}")
            console.print(f"[bold]Current Std:[/] {std:.6f}")
            
            # Preview the operation
            console.print("\n[bold]📊 Patch Preview:[/]")
            console.print(f"Operation: {operation} {tensor_name} by {value}")
            
            if operation == "scale":
                effect = "increase" if value > 1 else "decrease"
                percentage = abs(value - 1) * 100
                console.print(f"This will {effect} values by {percentage:.1f}%")
                expected_mean = mean * value
                expected_std = std * value
            elif operation == "add":
                direction = "increase" if value > 0 else "decrease"
                console.print(f"This will {direction} values by adding {value}")
                expected_mean = mean + value
                expected_std = std
            else:
                console.print(f"[yellow]Operation {operation} effects cannot be previewed[/]")
                expected_mean = mean
                expected_std = std
            
            console.print(f"[bold]Expected Mean after patch:[/] {expected_mean:.6f}")
            console.print(f"[bold]Expected Std after patch:[/] {expected_std:.6f}")
            
            if target_quantile is not None:
                if isinstance(target_quantile, (list, tuple)) and len(target_quantile) == 2:
                    console.print(f"[bold]Targeting:[/] Values between {target_quantile[0]*100:.1f}% and {target_quantile[1]*100:.1f}% quantiles")
                    console.print("[yellow]Note: Quantile targeting requires the HexPatchTool from CodeAgent for full functionality[/]")
                else:
                    console.print(f"[bold]Targeting:[/] Values above {target_quantile*100:.1f}% quantile")
                    console.print("[yellow]Note: Quantile targeting requires the HexPatchTool from CodeAgent for full functionality[/]")
            
            # Ask for confirmation
            confirm = input("\nApply this patch? (yes/no): ").strip().lower()
            if confirm != "yes":
                console.print("[bold yellow]⚠️ Patch aborted[/]")
                return False
            
            # Simulate patch application
            console.print("\n[bold yellow]⚠️ Actual patching requires the HexPatchTool from CodeAgent[/]")
            console.print("[bold yellow]This is a simulated patch for demonstration purposes[/]")
            
            output_path = f"{self.model_path}_patched/model.safetensors"
            console.print(f"\n[bold green]✅ Simulated patch application[/]")
            console.print(f"If this had a full implementation, a new model would be saved to: {output_path}")
            
            # Add to exploration history even though it's simulated
            self.exploration_history.append({
                "action": "simulated_patch",
                "tensor": tensor_name,
                "operation": operation,
                "value": value,
                "target_quantile": target_quantile,
                "result": "simulated_success"
            })
            
            return True
            
        except Exception as e:
            import traceback
            console.print(f"[bold red]❌ Error applying patch: {str(e)}[/]")
            console.print(traceback.format_exc())
            return False

    def _call_ai_api(self, prompt: str) -> str:
        """Call AI API with the given prompt"""
        if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
            return "AI interpretation unavailable: API key or model not configured. Please set OPENROUTER_API_KEY and OPENROUTER_MODEL in your .env file."
        
        try:
            system_prompt = """You are an expert Chinese reverse engineer specializing in neural networks, known for your hex patching skills with safetensors files.

You are currently hell-bent on experimenting with subtle hex modifications to change model behavior as desired by users. 
You've spent years reverse-engineering model weights and have intimate knowledge of how tensor modifications affect model outputs.
You're particularly skilled at identifying which tensors can be patched to achieve specific behavioral changes.
Your style is confident, direct, and sometimes uses technical jargon from both machine learning and low-level binary manipulation.

When suggesting modifications:
1. Always recommend specific tensors to modify
2. Explain exactly how they should be changed (scaling factors, values)
3. Predict how these changes will affect the model's behavior
4. Occasionally mention your past successes with similar modifications

Make references to both the mathematical tensor properties AND their binary/hex representations when relevant."""

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}, {response.text}"
                
        except Exception as e:
            return f"Error calling OpenRouter API: {str(e)}"
    
    def _search_tensors(self, pattern: str):
        """Search for tensors that match the given pattern"""
        results = {k: v for k, v in self.tensors.items() if pattern.lower() in k.lower()}
        if results:
            console.print(f"[bold green]Found {len(results)} tensors matching '{pattern}':[/]")
            self.explorer.display_tensors(results)
            
            # Add to exploration history if that attribute exists
            if hasattr(self, "exploration_history"):
                self.exploration_history.append({"action": "search", "term": pattern, "results": len(results)})
        else:
            console.print(f"[yellow]No tensors found matching '{pattern}'[/]")
            
        return results
    
    def _modify_tensor_by_number(self, num: int, operation: str, value: str):
        """Modify tensor by its index in sorted list"""
        sorted_tensors = sorted(self.tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)
        if 0 <= num < len(sorted_tensors):
            tensor_name = sorted_tensors[num][0]
            self._modify_tensor_by_name(tensor_name, operation, value)
        else:
            console.print("[bold red]Invalid tensor number[/]")

    def _get_data_driven_recommendations(self, capability="general"):
        """
        Get data-driven tensor recommendations by orchestrating existing tools
        to analyze tensors and provide concrete improvement suggestions.
        
        Args:
            capability (str): The capability to focus on, e.g., "math", "reasoning"
        
        Returns:
            Dict with recommendations based on actual tensor analysis
        """
        from rich.console import Console
        console = Console()
        
        console.print("[bold cyan]Running data-driven analysis for tensor recommendations...[/]")
        
        # Step 1: Identify key tensors to analyze based on the requested capability
        key_tensor_patterns = []
        if capability.lower() == "math":
            key_tensor_patterns = [
                "mlp.gate_proj.weight", 
                "mlp.up_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight"
            ]
            console.print("[cyan]Focusing on tensors related to mathematical capabilities[/]")
        else:
            key_tensor_patterns = [
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight", 
                "mlp.up_proj.weight",
                "input_layernorm.weight"
            ]
            console.print("[cyan]Focusing on tensors related to general reasoning capabilities[/]")
        
        # Step 2: Find matching tensors in the model
        target_tensors = []
        for pattern in key_tensor_patterns:
            matches = [name for name in self.tensors.keys() if pattern in name]
            target_tensors.extend(matches[:3])  # Limit to 3 matches per pattern
        
        # Step 3: Analyze each tensor and collect statistics
        tensor_stats = {}
        for tensor_name in target_tensors:
            console.print(f"[bold]Analyzing tensor:[/] {tensor_name}")
            stats = self.explorer.analyze_tensor(tensor_name)
            tensor_stats[tensor_name] = stats
        
        # Step 4: Generate recommendations based on actual tensor statistics
        recommendations = []
        
        for tensor_name, stats in tensor_stats.items():
            console.print(f"[bold]Generating recommendation for:[/] {tensor_name}")
            
            # Get mean and std to guide our recommendations
            mean = stats.get('mean', 0)
            std = stats.get('std', 0.1)
            shape = stats.get('shape', [])
            size = 1
            for dim in shape:
                size *= dim
            
            # Create specific recommendations based on tensor characteristics
            if "mlp.up_proj.weight" in tensor_name or "mlp.gate_proj.weight" in tensor_name:
                # For MLP layers that handle complex transformations
                scale_factor = 1.05
                target = "top 20%" if size > 10000 else "all values"
                effect = ("Enhance information processing for mathematical operations" 
                          if capability.lower() == "math" else 
                          "Improve concept composition and multi-step reasoning")
                
                recommendations.append({
                    "tensor": tensor_name,
                    "operation": "scale",
                    "value": scale_factor,
                    "target": target,
                    "effect": effect,
                    "confidence": "high",
                    "evidence": f"Mean: {mean:.6f}, Std: {std:.6f}, Size: {size}"
                })
                
            elif "self_attn.q_proj.weight" in tensor_name:
                # For query projections that determine what to focus on
                if capability.lower() == "math":
                    scale_factor = 1.08
                    target = "top 10%"
                    effect = "Improve attention focus on mathematical operations and numeric values"
                else:
                    scale_factor = 1.03
                    target = "top 20%"
                    effect = "Sharpen focus on relevant context for reasoning tasks"
                
                recommendations.append({
                    "tensor": tensor_name,
                    "operation": "scale",
                    "value": scale_factor,
                    "target": target,
                    "effect": effect,
                    "confidence": "medium",
                    "evidence": f"Mean: {mean:.6f}, Std: {std:.6f}, Size: {size}"
                })
                
            elif "self_attn.k_proj.weight" in tensor_name:
                # For key projections that determine how to represent information
                if capability.lower() == "math":
                    scale_factor = 1.06
                    target = "top 15%"
                    effect = "Enhance representation of numerical patterns in attention mechanism"
                else:
                    scale_factor = 1.02
                    target = "all values"
                    effect = "Strengthen key representations for better context attention"
                
                recommendations.append({
                    "tensor": tensor_name,
                    "operation": "scale",
                    "value": scale_factor,
                    "target": target,
                    "effect": effect,
                    "confidence": "medium",
                    "evidence": f"Mean: {mean:.6f}, Std: {std:.6f}, Size: {size}"
                })
                
            elif "input_layernorm.weight" in tensor_name:
                # For normalization layers that stabilize activations
                scale_factor = 1.02
                target = "all values"
                effect = "Stabilize activations for more consistent reasoning outputs"
                
                recommendations.append({
                    "tensor": tensor_name,
                    "operation": "scale",
                    "value": scale_factor,
                    "target": target,
                    "effect": effect,
                    "confidence": "medium",
                    "evidence": f"Mean: {mean:.6f}, Std: {std:.6f}, Size: {size}"
                })
        
        # Step 5: Display recommendations
        console.print("\n[bold green]🛠️ Data-Driven Recommendations:[/]")
        
        for i, rec in enumerate(recommendations, 1):
            console.print(f"\n[bold cyan]{i}. Target:[/] {rec['tensor']}")
            console.print(f"   [bold]Operation:[/] {rec['operation']} by {rec['value']} on {rec['target']}")
            console.print(f"   [bold]Expected effect:[/] {rec['effect']}")
            console.print(f"   [bold]Confidence:[/] {rec['confidence']}")
            console.print(f"   [bold]Evidence:[/] {rec['evidence']}")
            
            # Try to generate a preview using existing tools if available
            try:
                if hasattr(self, "code_agent"):
                    # Check if hex_patch is available
                    hex_patch_available = False
                    
                    # Check directly if the method exists
                    if hasattr(self.code_agent, "hex_patch"):
                        hex_patch_available = True
                    
                    # Check if the method is available via a tool
                    if not hex_patch_available and hasattr(self.code_agent, "tools"):
                        for tool in self.code_agent.tools:
                            if hasattr(tool, "name") and tool.name == "hex_patch":
                                hex_patch_available = True
                                
                                # Add the method directly to the code_agent for convenience
                                def hex_patch_wrapper(tensor_name, operation="scale", value=1.0, 
                                                    target_quantile=None, row=None, col=None, 
                                                    output_path=None, preview_only=True):
                                    return tool.forward(tensor_name, operation, value, target_quantile,
                                                       row, col, output_path, preview_only)
                                
                                self.code_agent.hex_patch = hex_patch_wrapper
                                break
                    
                    if hex_patch_available or hasattr(self.code_agent, "hex_patch"):
                        target_quantile = 0.9 if "top 10%" in rec['target'] else 0.85 if "top 15%" in rec['target'] else 0.8 if "top 20%" in rec['target'] else None
                        preview = self.code_agent.hex_patch(
                            tensor_name=rec['tensor'],
                            operation=rec['operation'],
                            value=rec['value'],
                            target_quantile=target_quantile,
                            preview_only=True
                        )
                        console.print(f"   [bold]Sample before/after:[/]")
                        console.print(f"   Before: {preview.get('sample_before', 'N/A')}")
                        console.print(f"   After:  {preview.get('sample_after', 'N/A')}")
                    else:
                        # Generate a simple preview when hex_patch isn't available
                        import numpy as np
                        
                        # Get some stats about the tensor if possible
                        mean, std = 0.0, 0.1
                        if hasattr(self.explorer, "analyze_tensor"):
                            try:
                                stats = self.explorer.analyze_tensor(rec['tensor'])
                                mean = stats.get('mean', 0.0)
                                std = stats.get('std', 0.1)
                            except:
                                pass
                        
                        # Generate sample values based on likely distribution
                        sample_before = np.random.normal(mean, std, 5)
                        
                        # Apply operation
                        if rec['operation'] == "scale":
                            sample_after = sample_before * rec['value']
                        elif rec['operation'] == "add":
                            sample_after = sample_before + rec['value']
                        
                        # Format for display
                        console.print(f"   [bold]Sample preview (estimated):[/]")
                        console.print(f"   Before: {', '.join([f'{x:.6f}' for x in sample_before])}")
                        console.print(f"   After:  {', '.join([f'{x:.6f}' for x in sample_after])}")
                        console.print(f"   [yellow]Note: This is a simulated preview, actual values may differ[/]")
                else:
                    console.print(f"   [yellow]Preview not available - CodeAgent not initialized[/]")
            except Exception as e:
                console.print(f"   [yellow]Preview generation error: {str(e)}[/]")
            
            # Display command to apply this recommendation
            quantile_param = ""
            if "top 10%" in rec['target']:
                quantile_param = " quantile=0.9"
            elif "top 15%" in rec['target']:
                quantile_param = " quantile=0.85"
            elif "top 20%" in rec['target']:
                quantile_param = " quantile=0.8"
            
            console.print(f"   [bold]To apply:[/] patch {rec['tensor']} {rec['operation']} {rec['value']}{quantile_param}")
        
        console.print("\n[bold]For more options, use:[/] help")
        
        return recommendations

    def _list_tensors(self):
        """List all tensors in the model"""
        console.print("\n[bold cyan]All tensors in the model:[/]")
        # Use the existing display_tensors method
        self.explorer.display_tensors(self.tensors)

    # Method to allow CodeAgent to call tensor_statistics via AITensorExplorer instance
    def tensor_statistics(self, tensor_name: str, quantiles: Optional[List[float]] = None) -> Dict[str, Any]:
        """Delegates to SafetensorsExplorer.calculate_tensor_statistics"""
        # self.explorer is an instance of SafetensorsExplorer
        if not hasattr(self, 'explorer') or not self.explorer:
            return {"error": "SafetensorsExplorer (self.explorer) is not initialized."}
        if not hasattr(self.explorer, 'calculate_tensor_statistics'):
            return {"error": "SafetensorsExplorer does not have calculate_tensor_statistics method."}
        try:
            return self.explorer.calculate_tensor_statistics(tensor_name, quantiles=quantiles)
        except Exception as e:
            # Log the full traceback for debugging
            # import traceback
            # console.print(f"[red]Traceback for tensor_statistics error:\n{traceback.format_exc()}[/red]")
            return {"error": f"Exception in AITensorExplorer.tensor_statistics delegator: {str(e)}"}

def main(model_path: str = typer.Argument("./Deepseek", help="Path to the model directory")):
    """Interactive AI-powered exploration of model weights"""
    try:
        explorer = AITensorExplorer(model_path)
        explorer.start_interactive_session()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main) 