#!/usr/bin/env python3
"""
Inference Engine - Textual TUI Application
Main entry point for the standalone inference engine with AI capabilities
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Input, Button, Static, Label, 
    RichLog, DataTable, TabbedContent, TabPane, 
    ProgressBar, Select, TextArea
)
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

# Import our modules
from model_loader import ModelLoader
from hook_system import HookSystem
from config_manager import ConfigManager, InferenceConfig
from ai_agent import ai_agent
from ui.comparison_widget import ComparisonWidget

# Import tensor inspector (to be created)
try:
    from tensor_inspector import TensorInspector
except ImportError:
    TensorInspector = None
    print("Warning: TensorInspector not yet available")

# Import AI suggestion engine (to be created)
try:
    from ai_suggestion_engine import AISuggestionEngine
except ImportError:
    AISuggestionEngine = None
    print("Warning: AISuggestionEngine not yet available")

class InferenceEngineApp(App):
    """Main Textual application for the Inference Engine"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #query-input {
        dock: top;
        margin: 1 2;
        height: 3;
    }
    
    #control-panel {
        dock: top;
        height: 5;
        background: $panel;
        padding: 1;
        align: center middle;
    }
    
    #control-panel Button {
        margin: 0 1;
        min-width: 16;
    }
    
    #config-select {
        width: 25;
        margin: 0 1;
    }
    
    #comparison-widget {
        height: 100%;
        margin: 1;
    }
    
    #status-bar {
        dock: bottom;
        height: 3;
        background: $boost;
        padding: 1;
    }
    
    .button-row {
        height: 3;
        margin-top: 1;
    }
    
    DataTable {
        height: 100%;
    }
    
    .suggestions-table {
        height: 20;
        border: solid $primary;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        height: 100%;
        padding: 0;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate"),
        Binding("ctrl+c", "clear", "Clear"),
        Binding("ctrl+s", "suggestions", "Get Suggestions"),
        Binding("ctrl+h", "toggle_hooks", "Toggle Hooks"),
        Binding("ctrl+q", "quit", "Quit"),
    ]
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.hook_system = HookSystem()
        self.config_manager = ConfigManager()
        
        # Initialize tensor inspector if available
        if TensorInspector:
            self.tensor_inspector = TensorInspector(model_path)
        else:
            self.tensor_inspector = None
        
        # Initialize AI suggestion engine if available
        if AISuggestionEngine:
            self.suggestion_engine = AISuggestionEngine(model_path)
        else:
            self.suggestion_engine = None
        
        # State
        self.current_suggestions = []
        self.hooks_active = False
        self.generation_in_progress = False
        
    def compose(self) -> ComposeResult:
        """Compose the UI"""
        yield Header(show_clock=True)
        
        with TabbedContent(initial="comparison"):
            # Main comparison tab
            with TabPane("Comparison", id="comparison"):
                # Query input
                yield Input(
                    placeholder="Enter your query here...",
                    id="query-input"
                )
                
                # Control panel
                with Horizontal(id="control-panel"):
                    # Create simpler options for Select
                    config_options = [
                        ("Conservative", "conservative"),
                        ("Balanced", "balanced"),
                        ("Creative", "creative"),
                        ("Mathematical", "mathematical"),
                        ("Coding", "coding"),
                        ("Fast", "fast")
                    ]
                    yield Select(
                        options=config_options,
                        value="balanced",
                        id="config-select"
                    )
                    yield Button("Generate", variant="primary", id="generate-btn")
                    yield Button("Apply Suggestions", variant="success", id="apply-suggestions-btn")
                    yield Button("Clear Hooks", variant="warning", id="clear-hooks-btn")
                    yield Button("Get AI Suggestions", variant="default", id="get-suggestions-btn")
                
                # Comparison view
                yield ComparisonWidget(id="comparison-widget")
                
                # Status bar
                yield Static(
                    "Ready. Model not loaded.",
                    id="status-bar"
                )
            
            # Configuration tab
            with TabPane("Configurations", id="configs"):
                yield DataTable(id="config-table")
            
            # Suggestions tab
            with TabPane("AI Suggestions", id="suggestions"):
                with Vertical():
                    yield Static("AI-powered tensor modification suggestions", classes="title")
                    yield DataTable(id="suggestions-table", classes="suggestions-table")
                    with Horizontal(classes="button-row"):
                        yield Button("Generate Suggestions", id="gen-suggestions-btn")
                        yield Button("Apply Selected", id="apply-selected-btn")
                        yield Button("Clear All", id="clear-suggestions-btn")
            
            # History tab
            with TabPane("History", id="history"):
                yield RichLog(id="history-log")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Called when app starts"""
        # Load the model
        await self.load_model()
        
        # Setup config table
        self.setup_config_table()
        
        # Setup suggestions table
        self.setup_suggestions_table()
        
        # Update status
        self.update_status("Ready. Model loaded.")
    
    async def load_model(self) -> None:
        """Load the model asynchronously"""
        self.update_status("Loading model...")
        
        # Load model in background
        success = await asyncio.to_thread(self.model_loader.load_model, self.model_path)
        
        if success:
            # Set model for hook system
            self.hook_system.set_model(self.model_loader.model)
            self.update_status(f"Model loaded successfully. Device: {self.model_loader.device}")
        else:
            self.update_status("Failed to load model!")
    
    def setup_config_table(self) -> None:
        """Setup the configuration table"""
        table = self.query_one("#config-table", DataTable)
        table.add_columns("Parameter", "Conservative", "Balanced", "Creative", "Mathematical", "Coding")
        
        configs = self.config_manager.get_all_presets()
        params = ["temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"]
        
        for param in params:
            row = [param.replace("_", " ").title()]
            for config_key in ["conservative", "balanced", "creative", "mathematical", "coding"]:
                config = configs[config_key]
                value = getattr(config, param)
                row.append(str(value))
            table.add_row(*row)
    
    def setup_suggestions_table(self) -> None:
        """Setup the suggestions table"""
        table = self.query_one("#suggestions-table", DataTable)
        table.add_columns("Select", "Tensor", "Operation", "Value", "Target", "Confidence", "Reason")
    
    @on(Button.Pressed, "#generate-btn")
    async def on_generate(self, event: Button.Pressed) -> None:
        """Handle generate button press"""
        if self.generation_in_progress:
            self.update_status("Generation already in progress...")
            return
        
        query = self.query_one("#query-input", Input).value
        if not query:
            self.update_status("Please enter a query first!")
            return
        
        await self.run_comparison(query)
    
    async def run_comparison(self, query: str) -> None:
        """Run the comparison between original and modified"""
        self.generation_in_progress = True
        comparison_widget = self.query_one("#comparison-widget", ComparisonWidget)
        comparison_widget.clear()
        
        # Get selected config
        config_key = self.query_one("#config-select", Select).value
        config = self.config_manager.get_preset(config_key)
        
        # Log to history
        self.log_to_history(f"Query: {query}")
        self.log_to_history(f"Config: {config.name}")
        
        try:
            # Generate original response
            self.update_status("Generating original response...")
            original_response = await asyncio.to_thread(
                self.model_loader.generate_response,
                query,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_new_tokens,
                repetition_penalty=config.repetition_penalty
            )
            comparison_widget.update_original(original_response)
            
            # Generate modified response if hooks are active
            if self.hooks_active:
                hook_count = self.hook_system.get_active_modifications_count()
                self.update_status(f"Generating modified response with {hook_count} hooks...")
                
                modified_response = await asyncio.to_thread(
                    self.model_loader.generate_response,
                    query,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=config.repetition_penalty
                )
                comparison_widget.update_modified(modified_response, hook_count)
            else:
                comparison_widget.update_modified("No hooks active. Apply suggestions first.", 0)
            
            self.update_status("Generation complete!")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            self.generation_in_progress = False
    
    @on(Button.Pressed, "#get-suggestions-btn")
    async def on_get_suggestions(self, event: Button.Pressed) -> None:
        """Get AI suggestions"""
        self.update_status("Getting AI suggestions...")
        
        if not ai_agent.is_available():
            self.update_status("AI agent not available. Check API configuration.")
            return
        
        # Get tensor stats (simplified for now)
        tensor_stats = self.get_tensor_stats()
        
        # Generate suggestions
        suggestions = await asyncio.to_thread(
            ai_agent.generate_modifications,
            tensor_stats,
            "general"
        )
        
        self.current_suggestions = suggestions
        self.display_suggestions(suggestions)
        self.update_status(f"Generated {len(suggestions)} AI suggestions")
    
    @on(Button.Pressed, "#apply-suggestions-btn")
    async def on_apply_suggestions(self, event: Button.Pressed) -> None:
        """Apply current suggestions as hooks"""
        if not self.current_suggestions:
            self.update_status("No suggestions to apply. Generate suggestions first.")
            return
        
        # Convert suggestions to hook modifications
        modifications_by_module = {}
        
        for suggestion in self.current_suggestions[:10]:  # Limit to 10
            tensor_name = suggestion.get("tensor_name", "")
            module_name = self.hook_system.tensor_name_to_module_name(tensor_name)
            
            if module_name not in modifications_by_module:
                modifications_by_module[module_name] = []
            modifications_by_module[module_name].append(suggestion)
        
        # Apply hooks
        self.hook_system.register_layer_hooks(modifications_by_module)
        self.hooks_active = True
        
        hook_count = self.hook_system.get_active_modifications_count()
        self.update_status(f"Applied {hook_count} tensor modifications as hooks")
    
    @on(Button.Pressed, "#clear-hooks-btn")
    async def on_clear_hooks(self, event: Button.Pressed) -> None:
        """Clear all hooks"""
        self.hook_system.clear_hooks()
        self.hooks_active = False
        self.update_status("All hooks cleared")
    
    def get_tensor_stats(self) -> List[Dict]:
        """Get simplified tensor statistics"""
        # This would be implemented with actual tensor inspection
        # For now, return sample data
        return [
            {
                "name": "model.layers.10.self_attn.q_proj.weight",
                "shape": "[4096, 4096]",
                "mean": 0.002,
                "std": 0.045
            }
        ]
    
    def display_suggestions(self, suggestions: List[Dict]) -> None:
        """Display suggestions in the table"""
        table = self.query_one("#suggestions-table", DataTable)
        table.clear()
        
        for suggestion in suggestions:
            table.add_row(
                "[ ]",  # Checkbox placeholder
                suggestion.get("tensor_name", "")[:40] + "...",
                suggestion.get("operation", ""),
                str(suggestion.get("value", "")),
                suggestion.get("target", ""),
                f"{suggestion.get('confidence', 0):.2f}",
                suggestion.get("reason", "")[:50] + "..."
            )
    
    def update_status(self, message: str) -> None:
        """Update status bar"""
        status_bar = self.query_one("#status-bar", Static)
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_bar.update(f"[{timestamp}] {message}")
    
    def log_to_history(self, message: str) -> None:
        """Log to history tab"""
        history = self.query_one("#history-log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        history.write(f"[{timestamp}] {message}")
    
    def action_generate(self) -> None:
        """Action for generate binding"""
        asyncio.create_task(self.on_generate(None))
    
    def action_clear(self) -> None:
        """Action for clear binding"""
        comparison_widget = self.query_one("#comparison-widget", ComparisonWidget)
        comparison_widget.clear()
        self.update_status("Cleared comparison view")
    
    def action_suggestions(self) -> None:
        """Action for suggestions binding"""
        asyncio.create_task(self.on_get_suggestions(None))
    
    def action_toggle_hooks(self) -> None:
        """Toggle hooks on/off"""
        if self.hooks_active:
            asyncio.create_task(self.on_clear_hooks(None))
        else:
            asyncio.create_task(self.on_apply_suggestions(None))

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python inference_engine.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model path '{model_path}' does not exist")
        sys.exit(1)
    
    app = InferenceEngineApp(model_path)
    app.run()

if __name__ == "__main__":
    main()