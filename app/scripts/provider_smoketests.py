import asyncio
import os
from loguru import logger

from app.providers.serper import SerperClient, SerperSearchRequest
from app.providers.scrapedo import ScrapeDoClient, ScrapeDoRequest
from app.providers.openrouter import OpenRouterClient, OpenRouterChatRequest, ORMessage


async def main():
    # SERPER
    if os.getenv("SERPER_DEV_API_KEY"):
        s = SerperClient()
        res, cost = await s.search(SerperSearchRequest(q="site:openai.com GPT"))
        logger.info(f"Serper organic count: {len(res.organic)} cost={cost}")
    else:
        logger.warning("SERPER_DEV_API_KEY not set; skipping Serper test")

    # SCRAPE.DO
    if os.getenv("SCRAPE_DO_API_KEY"):
        sd = ScrapeDoClient()
        resp, cost = await sd.fetch(ScrapeDoRequest(url="https://example.com", render_js=False, timeout_ms=10000))
        logger.info(f"Scrape.do status={resp.status_code} bytes={len(resp.content)} cost={cost}")
    else:
        logger.warning("SCRAPE_DO_API_KEY not set; skipping Scrape.do test")

    # OPENROUTER
    if os.getenv("OPENROUTER_API_KEY"):
        orc = OpenRouterClient()
        chat, cost = await orc.chat_completions(
            OpenRouterChatRequest(model="openai/gpt-5-nano", messages=[ORMessage(role="user", content="Say hi")])
        )
        logger.info(f"OpenRouter choice: {chat.choices[0].message.content[:60]} cost={cost}")
    else:
        logger.warning("OPENROUTER_API_KEY not set; skipping OpenRouter test")


if __name__ == "__main__":
    asyncio.run(main())

