#!/usr/bin/env python3
"""Test script to verify UI fixes"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Label, RichLog, Button, Select, Input
from ui.comparison_widget import ComparisonWidget

class TestUIApp(App):
    """Test the fixed UI components"""
    
    CSS = """
    #test-panel {
        height: 100%;
        margin: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical():
            yield Input(placeholder="Test input")
            
            with Horizontal():
                yield Select(
                    options=[("Option 1", "opt1"), ("Option 2", "opt2")],
                    value="opt1"
                )
                yield Button("Test Button")
            
            # Test the comparison widget
            yield ComparisonWidget(id="test-panel")
        
        yield Footer()
    
    def on_mount(self):
        # Test updating the comparison widget
        widget = self.query_one("#test-panel", ComparisonWidget)
        widget.update_original("This is the original text panel")
        widget.update_modified("This is the modified text panel", 5)

if __name__ == "__main__":
    app = TestUIApp()
    app.run()