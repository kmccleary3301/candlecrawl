from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 3002
    
    # Redis configuration
    # Supports URLs with optional password: redis://:password@host:port
    redis_url: str = "redis://:codex_web_agent_redis_password@localhost:6392"
    redis_user: Optional[str] = None
    redis_password: Optional[str] = None
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Scraping configuration
    default_timeout: int = 30
    max_concurrent_requests: int = 5
    # Default to a mainstream, JS-capable UA. Some modern sites (e.g., heavy SPA apps)
    # serve degraded "enable JavaScript / unsupported browser" shells to bot-like UAs.
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    # Retry/backoff
    retry_max_attempts: int = 3
    backoff_base_ms: int = 200
    backoff_max_ms: int = 3000
    
    # File handling
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    # Cache (Firecrawl v2 semantics)
    cache_enabled: bool = True
    # Default matches Firecrawl tooling docs: 172800000ms = 48h
    cache_default_max_age_ms: int = 172800000
    cache_ttl_seconds: int = 172800  # 48h
    
    # Optional LLM configuration for extraction
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Provider API keys and base URLs (Hermes)
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    serper_dev_api_key: Optional[str] = None
    serper_base_url: str = "https://google.serper.dev"

    scrape_do_api_key: Optional[str] = None
    scrapedo_base_url: str = "https://api.scrape.do/"

    # Pricing configuration (USD)
    cost_serper_per_request_usd: float = 0.001
    cost_scrapedo_per_request_usd: float = 0.0001
    cost_firecrawl_per_scrape_usd: float = 0.0002
    
settings = Settings() 
