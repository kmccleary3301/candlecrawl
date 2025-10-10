from __future__ import annotations

import time
from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field

from app.providers.openrouter import ORUsage
from app.config import settings


class JobCostTracker(BaseModel):
    """Real-time cost tracking for research jobs across all providers"""
    
    job_id: str
    tier: Optional[str] = None
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    # Cost breakdown by provider
    openrouter_cost: float = Field(default=0.0)
    serper_cost: float = Field(default=0.0)
    scrapedo_cost: float = Field(default=0.0)
    firecrawl_cost: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    
    # Usage tracking
    openrouter_tokens: int = Field(default=0)
    openrouter_calls: int = Field(default=0)
    serper_calls: int = Field(default=0)
    scrapedo_calls: int = Field(default=0)
    firecrawl_calls: int = Field(default=0)
    
    # Stage-specific costs
    stage_costs: Dict[str, float] = Field(default_factory=dict)
    # Minimal trace of provider calls and key events
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_openrouter_cost(self, usage: Optional[ORUsage] = None, estimated_cost: Optional[float] = None):
        """Add OpenRouter completion cost from usage data or estimate"""
        if usage:
            if usage.cost is not None:
                cost = usage.cost
            elif usage.total_tokens and estimated_cost:
                # Fallback to estimation if exact cost unavailable
                cost = estimated_cost
            else:
                # Conservative estimate: $0.002 per 1K tokens for GPT-4o-mini
                cost = (usage.total_tokens or 0) * 0.000002
            
            self.openrouter_cost += cost
            self.total_cost += cost
            
            if usage.total_tokens:
                self.openrouter_tokens += usage.total_tokens
        elif estimated_cost:
            self.openrouter_cost += estimated_cost
            self.total_cost += estimated_cost
        
        self.openrouter_calls += 1
    
    def add_serper_cost(self):
        """Add fixed Serper.dev search cost (configurable)."""
        cost = float(getattr(settings, "cost_serper_per_request_usd", 0.001))
        self.serper_cost += cost
        self.total_cost += cost
        self.serper_calls += 1
    
    def add_scrapedo_cost(self):
        """Add fixed Scrape.do scraping cost (configurable)."""
        cost = float(getattr(settings, "cost_scrapedo_per_request_usd", 0.0001))
        self.scrapedo_cost += cost
        self.total_cost += cost
        self.scrapedo_calls += 1
    
    def add_firecrawl_scrape(self):
        """Add fixed Firecrawl processing cost per scrape (configurable)."""
        cost = float(getattr(settings, "cost_firecrawl_per_scrape_usd", 0.0002))
        self.firecrawl_cost += cost
        self.total_cost += cost
        self.firecrawl_calls += 1

    def add_firecrawl_cost(self, document_size_bytes: int):
        """Add Firecrawl processing cost based on document size"""
        # Estimate: $0.0001 per MB processed, max $0.0005 per document
        cost = min(0.0005, document_size_bytes / 1_000_000 * 0.0001)
        self.firecrawl_cost += cost
        self.total_cost += cost
        self.firecrawl_calls += 1
    
    def add_stage_cost(self, stage: str, cost: float):
        """Track cost by research pipeline stage"""
        self.stage_costs[stage] = self.stage_costs.get(stage, 0) + cost
    
    def complete_job(self):
        """Mark job as completed"""
        self.completed_at = time.time()

    def add_trace_event(
        self,
        *,
        provider: str,
        event_type: str,
        status: str,
        latency_ms: Optional[float] = None,
        cost_delta: Optional[float] = None,
        model: Optional[str] = None,
        request_summary: Optional[str] = None,
        response_excerpt: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        evt = {
            "ts": time.time(),
            "provider": provider,
            "type": event_type,
            "status": status,
            "latency_ms": latency_ms,
            "cost_delta": cost_delta,
            "model": model,
            "request": request_summary,
            "response": response_excerpt,
            "extra": extra or {},
        }
        self.trace.append(evt)

    def get_trace(self) -> List[Dict[str, Any]]:
        return self.trace
    
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def get_cost_breakdown(self) -> Dict[str, float | Dict[str, float]]:
        """Get detailed cost breakdown with compatibility aliases."""
        provider_costs = {
            "openrouter": self.openrouter_cost,
            "serper": self.serper_cost,
            "scrapedo": self.scrapedo_cost,
            "firecrawl": self.firecrawl_cost,
        }
        breakdown = {
            **provider_costs,
            "total": self.total_cost,
            # Backward/forward compatibility aliases expected by some scripts
            "provider_costs": provider_costs,
            "total_cost_usd": self.total_cost,
        }
        return breakdown
    
    def get_usage_summary(self) -> Dict[str, int]:
        """Get API call counts"""
        return {
            "openrouter_calls": self.openrouter_calls,
            "openrouter_tokens": self.openrouter_tokens,
            "serper_calls": self.serper_calls,
            "scrapedo_calls": self.scrapedo_calls,
            "firecrawl_calls": self.firecrawl_calls
        }
    
    def validate_tier_budget(self) -> bool:
        """Check if cost is within expected tier range"""
        tier_limits = {
            "BROADCAST": 0.010,    # $0.01 max
            "TARGETED": 0.030,     # $0.03 max  
            "PRECISION": 0.100     # $0.10 max
        }
        
        if not self.tier:
            return True  # No validation if tier not set
        
        limit = tier_limits.get(self.tier.upper(), tier_limits["PRECISION"])
        return self.total_cost <= limit
    
    def get_tier_efficiency_rating(self) -> str:
        """Rate cost efficiency for the tier"""
        if not self.tier:
            return "unknown"
        
        tier_ranges = {
            "BROADCAST": (0.002, 0.008),
            "TARGETED": (0.005, 0.025),
            "PRECISION": (0.020, 0.080)
        }
        
        min_cost, max_cost = tier_ranges.get(self.tier.upper(), (0, float('inf')))
        
        if self.total_cost <= min_cost:
            return "excellent"
        elif self.total_cost <= (min_cost + max_cost) / 2:
            return "good"
        elif self.total_cost <= max_cost:
            return "acceptable"
        else:
            return "over_budget"


class TierBudgetConfig(BaseModel):
    """Configuration for tier budget limits"""
    
    broadcast_max: float = Field(default=0.010)   # $0.01
    targeted_max: float = Field(default=0.030)    # $0.03
    precision_max: float = Field(default=0.100)   # $0.10
    
    # Alert thresholds (percentage of max)
    warning_threshold: float = Field(default=0.75)  # 75% of max
    critical_threshold: float = Field(default=0.90)  # 90% of max
    
    def get_limit(self, tier: str) -> float:
        """Get budget limit for tier"""
        limits = {
            "BROADCAST": self.broadcast_max,
            "TARGETED": self.targeted_max,
            "PRECISION": self.precision_max
        }
        return limits.get(tier.upper(), self.precision_max)
    
    def get_alert_level(self, tier: str, current_cost: float) -> str:
        """Get alert level based on current cost"""
        limit = self.get_limit(tier)
        percentage = current_cost / limit if limit > 0 else 0
        
        if percentage >= self.critical_threshold:
            return "critical"
        elif percentage >= self.warning_threshold:
            return "warning"
        else:
            return "normal"


# Global cost tracking registry
_cost_trackers: Dict[str, JobCostTracker] = {}


def create_cost_tracker(job_id: str, tier: Optional[str] = None) -> JobCostTracker:
    """Create and register a new cost tracker"""
    tracker = JobCostTracker(job_id=job_id, tier=tier)
    _cost_trackers[job_id] = tracker
    return tracker


def get_cost_tracker(job_id: str) -> Optional[JobCostTracker]:
    """Get existing cost tracker"""
    return _cost_trackers.get(job_id)


def complete_cost_tracker(job_id: str) -> Optional[JobCostTracker]:
    """Complete and remove cost tracker"""
    tracker = _cost_trackers.get(job_id)
    if tracker:
        tracker.complete_job()
        # Keep in registry for a short time for reporting
        # In production, this should be moved to persistent storage
    return tracker


def get_active_trackers() -> Dict[str, JobCostTracker]:
    """Get all active cost trackers"""
    return {k: v for k, v in _cost_trackers.items() if v.completed_at is None}


def cleanup_completed_trackers(max_age_seconds: int = 3600):
    """Remove completed trackers older than max_age_seconds"""
    current_time = time.time()
    to_remove = []
    
    for job_id, tracker in _cost_trackers.items():
        if (tracker.completed_at and 
            current_time - tracker.completed_at > max_age_seconds):
            to_remove.append(job_id)
    
    for job_id in to_remove:
        del _cost_trackers[job_id]
