from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)

# Location configuration
class LocationConfig(BaseModel):
    country: Optional[str] = None
    languages: Optional[List[str]] = None

# Webhook configuration  
class WebhookConfig(BaseModel):
    url: str
    headers: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, str]] = None

# Scrape options
class ScrapeOptions(BaseModel):
    formats: Optional[List[Literal["markdown", "html", "rawHtml", "content", "links", "screenshot"]]] = ["markdown"]
    headers: Optional[Dict[str, str]] = None
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    only_main_content: Optional[bool] = None
    wait_for: Optional[int] = None
    timeout: Optional[int] = None
    location: Optional[LocationConfig] = None
    mobile: Optional[bool] = None
    skip_tls_verification: Optional[bool] = None
    remove_base64_images: Optional[bool] = None
    use_relative_links: Optional[bool] = False
    include_file_body: Optional[bool] = False
    
# Document metadata
class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    source_url: Optional[str] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    file_metadata: Optional[Dict[str, Any]] = None
    blocked: Optional[bool] = None
    blocked_reason: Optional[Literal["captcha","ip_block","robots_disallow","rate_limited"]] = None
    
# Firecrawl document
class FirecrawlDocument(BaseModel):
    url: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    raw_html: Optional[str] = None
    links: Optional[List[str]] = None
    screenshot: Optional[str] = None
    content_base64: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None

# Scrape request
class ScrapeRequest(BaseModel):
    url: str
    formats: Optional[List[Literal["markdown", "html", "rawHtml", "content", "links", "screenshot"]]] = ["markdown"]
    headers: Optional[Dict[str, str]] = None
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    only_main_content: Optional[bool] = True
    wait_for: Optional[int] = None
    timeout: Optional[int] = None
    location: Optional[LocationConfig] = None
    mobile: Optional[bool] = False
    skip_tls_verification: Optional[bool] = False
    remove_base64_images: Optional[bool] = True
    use_relative_links: Optional[bool] = False
    include_file_body: Optional[bool] = False

# Scrape response
class ScrapeResponse(BaseModel):
    success: bool = True
    data: Optional[FirecrawlDocument] = None
    error: Optional[str] = None
    warning: Optional[str] = None

class CrawlRequest(BaseModel):
    url: str
    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    max_depth: Optional[int] = 2
    limit: Optional[int] = 50
    allow_backward_links: Optional[bool] = None
    allow_external_links: Optional[bool] = False
    include_subdomains: Optional[bool] = False
    ignore_sitemap: Optional[bool] = False
    scrape_options: Optional[ScrapeOptions] = None
    webhook: Optional[Union[str, WebhookConfig]] = None
    delay: Optional[int] = 0
    max_time: Optional[int] = None
    max_bytes: Optional[int] = None
    max_concurrency: Optional[int] = 10

# Crawl response
class CrawlResponse(BaseModel):
    success: bool = True
    id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None

# Crawl status response
class CrawlStatusResponse(BaseModel):
    success: bool = True
    status: Literal["scraping", "completed", "failed", "cancelled"] = "scraping"
    completed: int = 0
    total: int = 0
    credits_used: int = 0
    expires_at: Optional[datetime] = None
    next: Optional[str] = None
    data: List[FirecrawlDocument] = []
    error: Optional[str] = None

# Batch scrape request
class BatchScrapeRequest(BaseModel):
    urls: List[str]
    formats: Optional[List[Literal["markdown", "html", "rawHtml", "content", "links", "screenshot"]]] = ["markdown"]
    headers: Optional[Dict[str, str]] = None
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    only_main_content: bool = False
    wait_for: Optional[Union[int, float]] = None
    timeout: Optional[int] = None
    location: Optional[LocationConfig] = None
    mobile: bool = False
    skip_tls_verification: bool = False
    remove_base64_images: bool = True
    use_relative_links: bool = False
    include_file_body: bool = False
    max_concurrency: int = 5

# Batch scrape response
class BatchScrapeResponse(BaseModel):
    success: bool = True
    id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    invalid_urls: Optional[List[str]] = None

# Map request
class MapRequest(BaseModel):
    url: str
    search: Optional[str] = None
    ignore_sitemap: Optional[bool] = False
    include_subdomains: Optional[bool] = False
    limit: Optional[int] = 5000

# Map response
class MapResponse(BaseModel):
    success: bool = True
    links: Optional[List[str]] = None
    error: Optional[str] = None
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=utc_now)

# --- Hermes provider-facing minimal models ---
class HermesSearchRequest(BaseModel):
    query: str
    gl: Optional[str] = None
    hl: Optional[str] = None
    limit: int = 10

class HermesSearchResponse(BaseModel):
    success: bool = True
    leads: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    error: Optional[str] = None
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

class HermesComposeRequest(BaseModel):
    model: str = "openai/gpt-5-nano"
    system: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class HermesComposeResponse(BaseModel):
    success: bool = True
    content: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

class HermesExternalScrapeRequest(BaseModel):
    url: str
    render_js: Optional[bool] = False
    timeout_ms: Optional[int] = 30000
    geo_code: Optional[str] = None

class HermesExternalScrapeResponse(BaseModel):
    success: bool = True
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class HermesEnrichRequest(BaseModel):
    domains: List[str]
    scrapeOptions: Optional[Dict[str, Any]] = None

class HermesEnrichResponse(BaseModel):
    success: bool = True
    domains: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

# Contact Extraction System Models
from enum import Enum

class SocialMediaPlatform(str, Enum):
    """Supported social media platforms for extraction"""
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin" 
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    REDDIT = "reddit"
    GITHUB = "github"

class SocialMediaConfig(BaseModel):
    """Configuration for social media extraction"""
    # If True, extract any relevant social media found
    # If dict, extract only specified platforms
    platforms: Union[bool, Dict[SocialMediaPlatform, bool]] = True
    
    # Whether to extract follower counts, verification status, etc.
    include_metrics: bool = False
    
    # Whether to extract recent posts or activity indicators
    include_activity: bool = False

class ContactExtractionConfig(BaseModel):
    """Configuration for contact information extraction"""
    
    # Basic contact information
    phone_numbers: bool = Field(default=True, description="Extract phone numbers")
    emails: bool = Field(default=True, description="Extract email addresses")
    addresses: bool = Field(default=True, description="Extract physical addresses")
    websites: bool = Field(default=True, description="Extract website URLs")
    
    # Social media configuration
    social_media: Union[bool, SocialMediaConfig] = Field(
        default=True, 
        description="Extract social media profiles - boolean for all, or SocialMediaConfig for specific"
    )
    
    # Personnel information
    key_personnel: bool = Field(default=True, description="Extract key personnel contact info")
    departments: bool = Field(default=False, description="Extract department-specific contacts")
    
    # Business-specific information
    business_hours: bool = Field(default=False, description="Extract business hours")
    locations: bool = Field(default=False, description="Extract multiple location details")
    
    # Verification requirements
    require_verification: bool = Field(
        default=False, 
        description="Only include contact info that can be verified from sources"
    )
    
    # Output formatting
    include_context: bool = Field(
        default=True, 
        description="Include context about where/how contact info was found"
    )
