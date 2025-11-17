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
                              capability: str = "general",
                              max_stats_lines: Optional[int] = None) -> List[Dict]:
        """Generate tensor modification suggestions"""
        if not self.is_available():
            # Return hardcoded fallback suggestions
            return self._get_fallback_suggestions(capability)
        
        # Build compact stats lines (boilerplate-style) to keep the prompt lean
        def _elem_size(dtype: str) -> int:
            try:
                if 'float16' in dtype or 'bfloat16' in dtype: return 2
                if 'float32' in dtype: return 4
                if 'float64' in dtype: return 8
                if 'int8' in dtype: return 1
            except Exception:
                pass
            return 4

        lines = []
        stats_iter = tensor_stats if max_stats_lines is None else tensor_stats[:max_stats_lines]
        for t in stats_iter:
            name = t.get('name') or t.get('tensor_name') or ''
            shape = t.get('shape')
            dtype = str(t.get('dtype', 'torch.float32'))
            mean = t.get('mean'); std = t.get('std'); vmin = t.get('min'); vmax = t.get('max')
            size_mb = t.get('size_mb')
            if size_mb is None:
                try:
                    if isinstance(shape, str):
                        # parse like "[4096, 4096]"
                        dims = [int(x) for x in shape.strip('[]').split(',') if x.strip()]
                    else:
                        dims = list(shape) if shape else []
                    numel = 1
                    for d in dims: numel *= int(d)
                    size_mb = numel * _elem_size(dtype) / (1024*1024)
                except Exception:
                    size_mb = 0.0
            lines.append(
                f"- {name}: shape={shape}, size={size_mb:.2f}MB, mean={float(mean):.4f}, std={float(std):.4f}, min={float(vmin):.4f}, max={float(vmax):.4f}"
            )
        stats_block = "\n".join(lines)

        prompt = f"""ONLY SUGGEST EXACT TENSOR MODIFICATIONS FOR {capability.upper()} CAPABILITY

Your output MUST be a valid JSON array ONLY containing tensor modification objects.

STRICT REQUIREMENTS:
1. NO FINE-TUNING DISCUSSION OR SUGGESTIONS WHATSOEVER
2. NO ARCHITECTURE CHANGES
3. NO DISCUSSIONS OF METHODOLOGY
4. NO TEXT BEFORE OR AFTER THE JSON ARRAY
5. FOCUS ON MLP, ATTENTION, INPUT, OUTPUT, AND OTHER CRITICAL LAYERS. AIM FOR HIGH-QUALITY, CONSISTENT MODIFICATIONS (MAX ~2 PER LAYER UNLESS MORE ARE NECESSARY)
6. ENSURE ALL SUGGESTIONS WORK TOGETHER TOWARD A COMMON PERFORMANCE GOAL
7. ONLY PROVIDE SUGGESTIONS YOU HAVE HIGH CONFIDENCE IN
8. DO NOT RETURN LOW-CONFIDENCE OR SPECULATIVE IDEAS

EACH recommendation object MUST HAVE:
- "tensor_name": EXACT name from the list below
- "operation": one of [scale, add, clamp_max, clamp_min]
- "value": specific number (e.g. 1.05, 0.9, 0.01)
- "target": one of [all, top 10%, top 20%, bottom 10%]
- "confidence": number between 0-1
- "reason": AN IN-DEPTH EXPLANATION OF THE SPECIFIC BENEFIT OF THE MODIFICATION

Available tensors and their statistics:
{stats_block}

WARNING: If you suggest fine-tuning, discuss approaches, or output anything other than a plain JSON array of tensor modifications, your response will be rejected.

Return ONLY a JSON array like: [ {{"tensor_name": "...", "operation": "...", "value": 1.05, "target": "top 10%", "confidence": 0.9, "reason": "..."}} ]"""

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
