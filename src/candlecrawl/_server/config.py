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
    # Optional endpoint-tiered v2 limits for external hardening.
    v2_rate_limit_window_seconds: int = 60
    v2_rate_limit_scrape_requests: int = 60
    v2_rate_limit_map_requests: int = 60
    v2_rate_limit_search_requests: int = 60
    v2_rate_limit_crawl_requests: int = 20
    v2_rate_limit_batch_scrape_requests: int = 20
    v2_rate_limit_extract_requests: int = 20

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
    # Document OCR configuration
    # Supported providers: auto, none, querylake_chandra, mistral
    document_ocr_provider_default: str = "none"
    document_ocr_fallback_threshold_chars: int = 120
    document_ocr_default_prompt: str = "Extract clean, readable markdown text."
    querylake_chandra_ocr_base_url: Optional[str] = None
    # OCR adapter modes:
    # - auto: try direct endpoint first, then QueryLake kernel/files fallback
    # - direct: call querylake_chandra_ocr_path JSON endpoint only
    # - querylake_files: use QueryLake kernel auth + files pipeline
    querylake_chandra_ocr_mode: str = "auto"
    # Legacy/direct endpoint (custom deployment contract).
    querylake_chandra_ocr_path: str = "/v1/chandra/ocr"
    querylake_chandra_ocr_auth_header: str = "X-Service-Token"
    querylake_chandra_ocr_auth_token: Optional[str] = None
    # QueryLake kernel/files mode auth + routing.
    querylake_chandra_ocr_username: Optional[str] = None
    querylake_chandra_ocr_password: Optional[str] = None
    querylake_chandra_ocr_auto_create_user: bool = True
    querylake_chandra_ocr_login_path: str = "/v2/kernel/api/login"
    querylake_chandra_ocr_add_user_path: str = "/v2/kernel/api/add_user"
    querylake_chandra_ocr_files_upload_path: str = "/v2/kernel/files"
    querylake_chandra_ocr_files_process_path_template: str = "/files/{file_id}/versions/{version_id}/process"
    querylake_chandra_ocr_search_chunks_path: str = "/v2/kernel/api/search_file_chunks"
    querylake_chandra_ocr_search_limit: int = 200
    querylake_chandra_ocr_profile: str = "balanced"
    querylake_chandra_ocr_enable_mistral_fallback: bool = False
    querylake_chandra_ocr_timeout_seconds: float = 60.0
    mistral_api_key: Optional[str] = None
    mistral_ocr_base_url: str = "https://api.mistral.ai/v1"
    mistral_ocr_model: str = "mistral-ocr-latest"
    mistral_ocr_timeout_seconds: float = 60.0

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

    # Firecrawl v2 compatibility shaping profile.
    # strict=true keeps parity-oriented response conventions and stable headers.
    strict_firecrawl_v2: bool = True
    v2_contract_version: str = "candlecrawl.firecrawl.v2.compat.2026-02-25"

    # v2 security baseline
    # Optional API-key auth gate for external-ready deployments.
    v2_auth_enabled: bool = False
    # Comma-separated valid API keys for bearer/x-api-key auth.
    v2_api_keys: str = ""
    # Tenant fallback used when request does not provide X-Tenant-ID.
    v2_default_tenant_id: str = "default"

    # SSRF protections for outbound URL targets.
    v2_ssrf_protection_enabled: bool = True
    v2_allow_private_network: bool = False
    v2_allow_localhost: bool = False

    # Budget guardrails and stop-loss controls
    budget_v2_crawl_limit_cap: int = 200
    budget_v2_extract_url_cap: int = 100
    budget_v2_search_limit_cap: int = 20
    kill_switch_v2_scrape: bool = False
    kill_switch_v2_crawl: bool = False

    # Async job webhook delivery settings
    webhook_timeout_seconds: float = 10.0
    webhook_max_retries: int = 2
    webhook_retry_base_delay_ms: int = 300
    webhook_signing_secret: Optional[str] = None

    # Artifact sink selector defaults (Track C foundation).
    artifact_mode_default: str = "inline"
    artifact_allow_fallback_to_inline: bool = True
    artifact_local_root: str = "/tmp/candlecrawl_artifacts"
    artifact_inline_max_bytes: int = 2 * 1024 * 1024
    artifact_querylake_enabled: bool = False
    querylake_files_base_url: Optional[str] = None
    querylake_files_ingestion_path: str = "/api/files/ingest_markdown"
    querylake_files_auth_header: str = "X-Service-Token"
    querylake_files_auth_token: Optional[str] = None
    querylake_files_timeout_seconds: float = 30.0
    querylake_files_max_retries: int = 2
    querylake_files_retry_base_delay_ms: int = 200
    querylake_files_collection_id: str = "candlecrawl_artifacts"

settings = Settings()
