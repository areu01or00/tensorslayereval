#!/usr/bin/env python3
"""
Configuration Manager Module
Manages inference configurations and presets
"""

from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class InferenceConfig:
    """Data class for inference configuration"""
    name: str
    description: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    repetition_penalty: float
    thinking_budget: int
    do_sample: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ConfigManager:
    """Manage inference configurations and presets"""
    
    def __init__(self):
        self.presets = self._initialize_presets()
        self.custom_configs = {}
        self.current_config = self.presets["balanced"]
    
    def _initialize_presets(self) -> Dict[str, InferenceConfig]:
        """Initialize preset configurations"""
        return {
            "conservative": InferenceConfig(
                name="Conservative Reasoning",
                description="Low temperature, focused sampling for logical tasks",
                temperature=0.1,
                top_p=0.9,
                top_k=20,
                max_new_tokens=2048,
                repetition_penalty=1.0,
                thinking_budget=2048
            ),
            "balanced": InferenceConfig(
                name="Balanced Generation",
                description="Default settings for general use",
                temperature=0.6,
                top_p=0.95,
                top_k=40,
                max_new_tokens=4096,
                repetition_penalty=1.0,
                thinking_budget=2048
            ),
            "creative": InferenceConfig(
                name="Creative Writing",
                description="Higher temperature for creative tasks",
                temperature=0.9,
                top_p=0.98,
                top_k=60,
                max_new_tokens=4096,
                repetition_penalty=1.1,
                thinking_budget=3072
            ),
            "mathematical": InferenceConfig(
                name="Mathematical Reasoning",
                description="Optimized for mathematical problem solving",
                temperature=0.3,
                top_p=0.9,
                top_k=25,
                max_new_tokens=6144,
                repetition_penalty=1.0,
                thinking_budget=4096
            ),
            "coding": InferenceConfig(
                name="Code Generation",
                description="Precise settings for programming tasks",
                temperature=0.2,
                top_p=0.95,
                top_k=30,
                max_new_tokens=8192,
                repetition_penalty=1.05,
                thinking_budget=2048
            ),
            "fast": InferenceConfig(
                name="Fast Response",
                description="Quick generation with minimal tokens",
                temperature=0.5,
                top_p=0.9,
                top_k=30,
                max_new_tokens=1024,
                repetition_penalty=1.0,
                thinking_budget=1024
            )
        }
    
    def get_preset(self, name: str) -> InferenceConfig:
        """Get a preset configuration"""
        return self.presets.get(name, self.presets["balanced"])
    
    def get_all_presets(self) -> Dict[str, InferenceConfig]:
        """Get all preset configurations"""
        return self.presets
    
    def add_custom_config(self, config: InferenceConfig):
        """Add a custom configuration"""
        self.custom_configs[config.name] = config
    
    def create_custom_config(self,
                           name: str,
                           description: str = "Custom configuration",
                           temperature: float = 0.6,
                           top_p: float = 0.95,
                           top_k: int = 40,
                           max_new_tokens: int = 4096,
                           repetition_penalty: float = 1.0,
                           thinking_budget: int = 2048) -> InferenceConfig:
        """Create a new custom configuration"""
        config = InferenceConfig(
            name=name,
            description=description,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            thinking_budget=thinking_budget
        )
        self.add_custom_config(config)
        return config
    
    def get_all_configs(self) -> Dict[str, InferenceConfig]:
        """Get all configurations (presets + custom)"""
        return {**self.presets, **self.custom_configs}
    
    def set_current_config(self, config_name: str):
        """Set the current active configuration"""
        all_configs = self.get_all_configs()
        if config_name in all_configs:
            self.current_config = all_configs[config_name]
        else:
            raise ValueError(f"Configuration '{config_name}' not found")
    
    def get_current_config(self) -> InferenceConfig:
        """Get the current active configuration"""
        return self.current_config
    
    def compare_configs(self, config_names: List[str]) -> List[Dict[str, Any]]:
        """Get configurations for comparison"""
        all_configs = self.get_all_configs()
        configs = []
        
        for name in config_names:
            if name in all_configs:
                configs.append(all_configs[name].to_dict())
        
        return configs
    
    def get_config_summary(self, config: InferenceConfig) -> str:
        """Get a summary string for a configuration"""
        return (
            f"{config.name}\n"
            f"  Temp: {config.temperature:.2f} | "
            f"Top-p: {config.top_p:.2f} | "
            f"Top-k: {config.top_k} | "
            f"Max tokens: {config.max_new_tokens}"
        )