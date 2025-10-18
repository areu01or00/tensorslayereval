#!/usr/bin/env python3
"""Test if Textual TUI works"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static

class TestApp(App):
    """Test Textual Application"""
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Textual is working! Press Ctrl+Q to quit.")
        yield Footer()
    
    BINDINGS = [("ctrl+q", "quit", "Quit")]

if __name__ == "__main__":
    app = TestApp()
    app.run()