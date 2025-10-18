#!/usr/bin/env python3
"""
Model Loader Module
Handles model loading and inference generation
Extracted from dynamic_tensor_analyzer.py
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        GenerationConfig,
        set_seed
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: transformers not available. Please install with: pip install transformers")

class ModelLoader:
    """Handles model loading and text generation"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = None
        self.is_qwen_model = False
        self.thinking_budget = 2048
        
    def load_model(self, model_path: str) -> bool:
        """Load model and tokenizer from path"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers library not available")
            return False
            
        try:
            print(f"Loading model from: {model_path}")
            self.model_path = model_path
            
            # Detect if this is a Qwen model
            model_name = Path(model_path).name.lower()
            self.is_qwen_model = "qwen" in model_name
            
            if self.is_qwen_model:
                print(f"🧠 Detected Qwen model - enabling thinking capabilities")
                print(f"💭 Thinking budget set to {self.thinking_budget} tokens")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimized parameters
            print(f"Loading model to {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def generate_response(self, 
                         prompt: str,
                         temperature: float = 0.6,
                         top_p: float = 0.95,
                         top_k: int = 40,
                         max_new_tokens: int = 4096,
                         repetition_penalty: float = 1.0,
                         do_sample: bool = True) -> str:
        """Generate response from the model"""
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        try:
            # Handle Qwen thinking tokens if applicable
            if self.is_qwen_model and hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Generation config
            gen_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Handle Qwen thinking tokens in output
            if self.is_qwen_model:
                response = self._process_qwen_response(response)
            
            return response
            
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def _process_qwen_response(self, response: str) -> str:
        """Process Qwen model response to handle thinking tokens"""
        # Remove thinking tokens if present
        if "<thinking>" in response and "</thinking>" in response:
            import re
            # Extract just the final response after thinking
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_qwen": self.is_qwen_model,
            "model_size": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(next(self.model.parameters()).dtype),
            "thinking_budget": self.thinking_budget if self.is_qwen_model else None
        }
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("Model resources cleaned up")