from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.providers.base import ProviderError


class ORMessage(BaseModel):
    role: str
    content: str


class ORChoice(BaseModel):
    index: Optional[int] = None
    message: ORMessage
    finish_reason: Optional[str] = None


class ORUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    native_tokens_prompt: Optional[int] = None
    native_tokens_completion: Optional[int] = None
    native_tokens_total: Optional[int] = None
    cost: Optional[float] = None


class OpenRouterChatResponse(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[ORChoice]
    usage: Optional[ORUsage] = None
    provider: Optional[Union[str, Dict[str, Any]]] = None


class OpenRouterChatRequest(BaseModel):
    model: str
    messages: List[ORMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Enable usage tracking for accurate cost calculation
    usage: Optional[Dict[str, bool]] = Field(default={"include": True})


class OpenRouterClient:
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self._client = client

    async def chat_completions(self, req: OpenRouterChatRequest) -> Tuple[OpenRouterChatResponse, Optional[float]]:
        if not self.api_key:
            raise ProviderError("OpenRouter API key not configured")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Ensure usage tracking is always enabled for cost accuracy
        if not req.usage:
            req.usage = {"include": True}
        payload = req.model_dump(exclude_none=True)

        created_client: Optional[httpx.AsyncClient] = None
        client = self._client
        if client is None:
            created_client = httpx.AsyncClient(timeout=60)
            client = created_client
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise ProviderError("OpenRouter error", status_code=resp.status_code, payload=resp.text)
            data = resp.json()
            response = OpenRouterChatResponse.model_validate(data)
            
            # Extract actual cost from usage data
            actual_cost = None
            if response.usage and hasattr(response.usage, 'cost'):
                actual_cost = response.usage.cost
            
            return response, actual_cost
        finally:
            if created_client:
                await created_client.aclose()


