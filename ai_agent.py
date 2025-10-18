#!/usr/bin/env python3
"""
Centralized AI Agent Module
Provides CodeAgent for AI-powered analysis and suggestions
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Try to import smolagents
try:
    from smolagents import CodeAgent, LiteLLMModel
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print("Warning: smolagents not available. AI features will be limited.")

class AIAgent:
    """Centralized AI agent for all AI-powered operations"""
    
    _instance = None
    _agent = None
    
    def __new__(cls):
        """Singleton pattern to reuse agent across modules"""
        if cls._instance is None:
            cls._instance = super(AIAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the AI agent once"""
        if self._agent is None:
            self._setup_agent()
    
    def _setup_agent(self):
        """Setup the CodeAgent with OpenRouter"""
        if not SMOLAGENTS_AVAILABLE:
            print("Smolagents not available. AI features disabled.")
            return
        
        # Get API credentials
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model_name = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not set. AI features will be limited.")
            return
        
        try:
            # Initialize the model
            model = LiteLLMModel(
                model_id=f"openrouter/{self.model_name}",
                api_key=self.api_key,
                api_base="https://openrouter.ai/api/v1"
            )
            
            # Create the agent
            self._agent = CodeAgent(
                tools=[],  # No tools needed for our use case
                model=model,
                max_steps=1  # Single response
                # verbose parameter removed - not supported in current version
            )
            
            print(f"AI Agent initialized with {self.model_name}")
            
        except Exception as e:
            print(f"Failed to initialize AI agent: {e}")
            self._agent = None
    
    def is_available(self) -> bool:
        """Check if AI agent is available"""
        return self._agent is not None
    
    def run(self, prompt: str, max_tokens: int = 4096) -> str:
        """Run the AI agent with a prompt"""
        if not self.is_available():
            return "AI agent not available. Please check API configuration."
        
        try:
            response = self._agent.run(prompt)
            return response
        except Exception as e:
            return f"AI agent error: {str(e)}"
    
    def analyze_tensor_patterns(self, tensor_stats: List[Dict]) -> str:
        """Analyze tensor patterns using AI"""
        if not self.is_available():
            return "AI analysis not available"
        
        prompt = f"""Analyze these tensor statistics and identify patterns:

{tensor_stats}

Provide insights about:
1. Unusual distributions or outliers
2. Potential issues or optimization opportunities
3. Layer-specific patterns
4. Recommendations for modifications

Be concise and technical."""
        
        return self.run(prompt)
    
    def generate_modifications(self, 
                              tensor_stats: List[Dict],
                              capability: str = "general") -> List[Dict]:
        """Generate tensor modification suggestions"""
        if not self.is_available():
            # Return hardcoded fallback suggestions
            return self._get_fallback_suggestions(capability)
        
        prompt = f"""ONLY SUGGEST EXACT TENSOR MODIFICATIONS FOR {capability.upper()} CAPABILITY

Your output MUST be a valid JSON array ONLY containing tensor modification objects.

STRICT REQUIREMENTS:
1. Output ONLY a JSON array, no other text
2. Focus on MLP, attention, and embedding layers
3. Each object must have: tensor_name, operation, value, target, confidence, reason

Available operations: scale, add, normalize, clamp_max, clamp_min
Available targets: all, top 5%, top 10%, top 20%, bottom 5%, bottom 10%

Tensor statistics:
{tensor_stats}

Return ONLY JSON array like: [{{"tensor_name": "...", "operation": "...", "value": ..., "target": "...", "confidence": ..., "reason": "..."}}]"""

        try:
            response = self.run(prompt)
            
            # Try to parse JSON from response
            import json
            import re
            
            # Extract JSON array from response
            if isinstance(response, str):
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            
            return self._get_fallback_suggestions(capability)
            
        except Exception as e:
            print(f"Failed to generate AI suggestions: {e}")
            return self._get_fallback_suggestions(capability)
    
    def _get_fallback_suggestions(self, capability: str) -> List[Dict]:
        """Fallback suggestions when AI is not available"""
        base_suggestions = {
            "general": [
                {
                    "tensor_name": "model.layers.10.self_attn.q_proj.weight",
                    "operation": "scale",
                    "value": 1.05,
                    "target": "top 10%",
                    "confidence": 0.7,
                    "reason": "Enhance attention focus in middle layers"
                },
                {
                    "tensor_name": "model.layers.15.mlp.gate_proj.weight",
                    "operation": "scale",
                    "value": 1.03,
                    "target": "all",
                    "confidence": 0.6,
                    "reason": "Slightly amplify MLP gating"
                }
            ],
            "math": [
                {
                    "tensor_name": "model.layers.12.mlp.up_proj.weight",
                    "operation": "scale",
                    "value": 1.08,
                    "target": "top 20%",
                    "confidence": 0.75,
                    "reason": "Enhance mathematical reasoning pathways"
                }
            ],
            "reasoning": [
                {
                    "tensor_name": "model.layers.8.self_attn.v_proj.weight",
                    "operation": "scale",
                    "value": 1.06,
                    "target": "top 15%",
                    "confidence": 0.8,
                    "reason": "Improve value processing for logical reasoning"
                }
            ]
        }
        
        return base_suggestions.get(capability, base_suggestions["general"])

# Global singleton instance
ai_agent = AIAgent()