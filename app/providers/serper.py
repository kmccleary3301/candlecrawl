from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.providers.base import ProviderError


class SerperOrganicResult(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    snippet: Optional[str] = None
    date: Optional[str] = None


class SerperSearchResponse(BaseModel):
    organic: List[SerperOrganicResult] = Field(default_factory=list)
    knowledgeGraph: Optional[Dict[str, Any]] = None
    answerBox: Optional[Dict[str, Any]] = None
    peopleAlsoAsk: Optional[List[Dict[str, Any]]] = None
    news: Optional[List[Dict[str, Any]]] = None
    topStories: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None
    videos: Optional[List[Dict[str, Any]]] = None
    searchParameters: Optional[Dict[str, Any]] = None
    # Cost tracking fields
    credits_used: int = Field(default=1)
    estimated_cost: float = Field(default=0.0003)  # $0.0003 per search


class SerperSearchRequest(BaseModel):
    q: str
    gl: Optional[str] = None  # country code
    hl: Optional[str] = None  # language code
    num: Optional[int] = Field(default=10, ge=1, le=100)
    tbs: Optional[str] = None  # time based search constraints


class SerperClient:
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key or settings.serper_dev_api_key
        self.base_url = (base_url or settings.serper_base_url).rstrip("/")
        self._client = client

    async def search(self, req: SerperSearchRequest) -> Tuple[SerperSearchResponse, float]:
        if not self.api_key:
            raise ProviderError("Serper.dev API key not configured")

        url = f"{self.base_url}/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = req.model_dump(exclude_none=True)

        created_client: Optional[httpx.AsyncClient] = None
        client = self._client
        if client is None:
            created_client = httpx.AsyncClient(timeout=30)
            client = created_client
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise ProviderError("Serper.dev error", status_code=resp.status_code, payload=resp.text)
            data = resp.json()
            response = SerperSearchResponse.model_validate(data)
            # Configurable fixed cost per search
            cost = float(getattr(settings, "cost_serper_per_request_usd", 0.001))
            return response, cost
        finally:
            if created_client:
                await created_client.aclose()

    # --- Additional endpoints ---
    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError("Serper.dev API key not configured")
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        created_client: Optional[httpx.AsyncClient] = None
        client = self._client
        if client is None:
            created_client = httpx.AsyncClient(timeout=30)
            client = created_client
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise ProviderError("Serper.dev error", status_code=resp.status_code, payload=resp.text)
            js = resp.json()
            if not isinstance(js, dict) or not js:
                raise ProviderError("Serper.dev empty or invalid response", status_code=resp.status_code, payload=resp.text)
            return js
        finally:
            if created_client:
                await created_client.aclose()

    # Convenience wrappers to other Serper endpoints
    async def news(self, req: SerperNewsRequest) -> Tuple[SerperGenericResponse, float]:
        js = await self._post("news", req.model_dump(exclude_none=True))
        return SerperGenericResponse.from_json(js), float(getattr(settings, "cost_serper_per_request_usd", 0.001))

    async def images(self, req: SerperImageRequest) -> Tuple[SerperGenericResponse, float]:
        js = await self._post("images", req.model_dump(exclude_none=True))
        return SerperGenericResponse.from_json(js), float(getattr(settings, "cost_serper_per_request_usd", 0.001))

    async def videos(self, req: SerperVideoRequest) -> Tuple[SerperGenericResponse, float]:
        js = await self._post("videos", req.model_dump(exclude_none=True))
        return SerperGenericResponse.from_json(js), float(getattr(settings, "cost_serper_per_request_usd", 0.001))

    async def places(self, req: SerperPlacesRequest) -> Tuple[SerperGenericResponse, float]:
        js = await self._post("places", req.model_dump(exclude_none=True))
        return SerperGenericResponse.from_json(js), float(getattr(settings, "cost_serper_per_request_usd", 0.001))

class SerperNewsRequest(BaseModel):
    q: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    num: Optional[int] = Field(default=10, ge=1, le=100)

class SerperImageRequest(BaseModel):
    q: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    num: Optional[int] = Field(default=10, ge=1, le=100)

class SerperVideoRequest(BaseModel):
    q: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    num: Optional[int] = Field(default=10, ge=1, le=100)

class SerperPlacesRequest(BaseModel):
    q: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    # Additional map/place parameters could be added later (e.g., ll=lat,long)

class SerperGenericResponse(BaseModel):
    success: bool = True
    raw: Dict[str, Any]

    @property
    def items(self) -> List[Dict[str, Any]]:
        # Try common fields
        for key in ("news", "images", "videos", "places", "organic"):
            val = self.raw.get(key)
            if isinstance(val, list):
                return val
        return []

    @property
    def estimated_cost(self) -> float:
        return float(getattr(settings, "cost_serper_per_request_usd", 0.001))

    @classmethod
    def from_json(cls, js: Dict[str, Any]) -> "SerperGenericResponse":
        return cls(raw=js)

    def first_title(self) -> Optional[str]:
        items = self.items
        if items:
            return items[0].get("title")
        return None

    def count(self) -> int:
        return len(self.items)

    def validate_required_field(self, field: str) -> bool:
        return field in self.raw

    def summarize(self) -> Dict[str, Any]:
        return {
            "count": self.count(),
            "first_title": self.first_title(),
            "keys": list(self.raw.keys()),
        }

    pass
