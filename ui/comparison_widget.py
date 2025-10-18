#!/usr/bin/env python3
"""
Comparison Widget for Textual UI
Displays side-by-side comparison of original vs modified outputs
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Label, RichLog
from textual.widget import Widget
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

class ComparisonWidget(Widget):
    """Widget for displaying side-by-side inference comparison"""
    
    DEFAULT_CSS = """
    ComparisonWidget {
        height: 100%;
        width: 100%;
    }
    
    .comparison-container {
        height: 100%;
        width: 100%;
    }
    
    .output-panel {
        width: 50%;
        height: 100%;
        margin: 0 1;
    }
    
    .original-output {
        border: thick blue;
    }
    
    .modified-output {
        border: thick green;
    }
    
    .output-header {
        dock: top;
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: $panel;
        border-bottom: solid $primary;
    }
    
    .output-content {
        padding: 1;
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_text = ""
        self.modified_text = ""
        self.hook_count = 0
        
    def compose(self) -> ComposeResult:
        """Compose the comparison widget"""
        with Horizontal(classes="comparison-container"):
            # Original output panel
            with Vertical(classes="output-panel original-output"):
                yield Label("🔵 Original Response", classes="output-header")
                yield RichLog(id="original-content", classes="output-content", wrap=True)
            
            # Modified output panel
            with Vertical(classes="output-panel modified-output"):
                yield Label("🔧 Modified Response", id="modified-header", classes="output-header")
                yield RichLog(id="modified-content", classes="output-content", wrap=True)
    
    def update_original(self, text: str):
        """Update original response text"""
        self.original_text = text
        log = self.query_one("#original-content", RichLog)
        log.clear()
        log.write(text)
    
    def update_modified(self, text: str, hook_count: int = 0):
        """Update modified response text"""
        self.modified_text = text
        self.hook_count = hook_count
        
        # Update header with hook count
        header = self.query_one("#modified-header", Label)
        header.update(f"🔧 Modified Response ({hook_count} hooks active)")
        
        # Update content
        log = self.query_one("#modified-content", RichLog)
        log.clear()
        log.write(text)
    
    def clear(self):
        """Clear both panels"""
        self.original_text = ""
        self.modified_text = ""
        self.hook_count = 0
        
        self.query_one("#original-content", RichLog).clear()
        self.query_one("#modified-content", RichLog).clear()
        
        # Reset header
        header = self.query_one("#modified-header", Label)
        header.update("🔧 Modified Response")
    
    def append_to_original(self, text: str):
        """Append text to original panel (for streaming)"""
        self.original_text += text
        log = self.query_one("#original-content", RichLog)
        log.write(text)
    
    def append_to_modified(self, text: str):
        """Append text to modified panel (for streaming)"""
        self.modified_text += text
        log = self.query_one("#modified-content", RichLog)
        log.write(text)