"""
vLLM client for efficient LLM inference.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import httpx


@dataclass
class LLMResponse:
    """Response from the LLM."""
    text: str
    tokens_used: int
    finish_reason: str
    model: str
