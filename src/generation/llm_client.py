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


class VLLMClient:
    """Client for vLLM OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        model: str = "jais-13b-chat",
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> LLMResponse:
        """Generate text completion."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or []
        }
        
        response = self._client.post(
            f"{self.base_url}/v1/completions",
            headers=self._build_headers(),
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        choice = data["choices"][0]
        
        return LLMResponse(
            text=choice["text"],
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            model=data.get("model", self.model)
        )
