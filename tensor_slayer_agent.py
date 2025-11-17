#!/usr/bin/env python3
"""Bridge Tensor-Slayer's CodeAgent suggestion flow into the harness."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

TENSOR_SLAYER_PATH = Path(__file__).resolve().parent / "Tensor-Slayer"
if str(TENSOR_SLAYER_PATH) not in os.sys.path:
    os.sys.path.append(str(TENSOR_SLAYER_PATH))

from model_explorer import TensorModelExplorer  # type: ignore  # noqa


class TensorSlayerAgent:
    """Wrapper that exposes model_explorer's suggestion logic as a simple callable."""

    def __init__(self, model_path: str, capability: str = "general"):
        self.model_path = model_path
        self.capability = capability
        self._explorer: Optional[TensorModelExplorer] = None

    def _ensure_explorer(self) -> TensorModelExplorer:
        if self._explorer is None:
            self._explorer = TensorModelExplorer(self.model_path)
        return self._explorer

    def generate(self) -> List[Dict[str, Any]]:
        """Invoke Tensor-Slayer to get JSON suggestions."""
        explorer = self._ensure_explorer()
        recs = explorer.get_ai_recommendations(capability=self.capability)
        return recs or []


def get_tensor_slayer_suggestions(model_path: str, capability: str = "general") -> List[Dict[str, Any]]:
    agent = TensorSlayerAgent(model_path, capability)
    return agent.generate()

