import json
import pytest
import httpx

from app.providers.serper import SerperClient, SerperSearchRequest
from app.providers.scrapedo import ScrapeDoClient, ScrapeDoRequest
from app.providers.openrouter import OpenRouterClient, OpenRouterChatRequest, ORMessage


@pytest.mark.asyncio
async def test_serper_search_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/search")
        payload = {
            "organic": [
                {"title": "Result A", "link": "https://a.com", "snippet": "A"},
                {"title": "Result B", "link": "https://b.com", "snippet": "B"},
            ]
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        serper = SerperClient(api_key="x", base_url="https://serper.dev/api", client=client)
        res, cost = await serper.search(SerperSearchRequest(q="test", num=2))
        assert len(res.organic) == 2
        assert res.organic[0].title == "Result A"
        assert cost >= 0


@pytest.mark.asyncio
async def test_scrapedo_fetch_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>ok</html>", headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sd = ScrapeDoClient(api_key="token", base_url="https://api.scrape.do", client=client)
        res, cost = await sd.fetch(ScrapeDoRequest(url="https://example.com", render_js=True, timeout_ms=10000))
        assert res.status_code == 200
        assert "ok" in res.content
        assert cost >= 0


@pytest.mark.asyncio
async def test_openrouter_chat_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-5-nano"
        payload = {
            "id": "cmpl_1",
            "object": "chat.completion",
            "created": 123,
            "model": "openai/gpt-5-nano",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        orc = OpenRouterClient(api_key="key", base_url="https://openrouter.ai/api/v1", client=client)
        req = OpenRouterChatRequest(model="openai/gpt-5-nano", messages=[ORMessage(role="user", content="hello")])
        res, cost = await orc.chat_completions(req)
        assert res.choices[0].message.content == "hi"
        assert cost is None or cost >= 0

