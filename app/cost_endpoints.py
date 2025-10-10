from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.cost_tracking import (
    JobCostTracker, 
    get_cost_tracker, 
    get_active_trackers,
    cleanup_completed_trackers,
    TierBudgetConfig
)

router = APIRouter(prefix="/v1/hermes/costs", tags=["cost-tracking"])


class CostSummaryResponse(BaseModel):
    """Summary of costs across time periods"""
    total_jobs: int
    total_cost: float
    avg_cost_per_job: float
    
    # Breakdown by tier
    broadcast_jobs: int = 0
    broadcast_cost: float = 0.0
    targeted_jobs: int = 0
    targeted_cost: float = 0.0
    precision_jobs: int = 0
    precision_cost: float = 0.0
    
    # Provider breakdown
    openrouter_cost: float = 0.0
    serper_cost: float = 0.0
    scrapedo_cost: float = 0.0
    firecrawl_cost: float = 0.0
    
    # Performance metrics
    avg_duration_seconds: Optional[float] = None
    avg_efficiency_rating: str = "unknown"
    
    period_start: datetime
    period_end: datetime


class JobCostResponse(BaseModel):
    """Detailed cost breakdown for a specific job"""
    job_id: str
    tier: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    
    # Cost breakdown
    total_cost: float
    cost_breakdown: Dict[str, float]
    stage_costs: Dict[str, float]
    
    # Usage summary
    usage_summary: Dict[str, int]
    efficiency_rating: str
    within_budget: bool
    
    # Performance indicators
    cost_per_token: Optional[float] = None
    cost_per_search: Optional[float] = None
    # Optional minimal job trace
    trace: Optional[List[Dict[str, Any]]] = None


class ProviderStatsResponse(BaseModel):
    """Provider usage and cost statistics"""
    provider: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    total_cost: float
    avg_cost_per_call: float
    avg_response_time_ms: Optional[int] = None


class BudgetAlertResponse(BaseModel):
    """Budget alert information"""
    job_id: str
    tier: str
    alert_level: str
    current_cost: float
    budget_limit: float
    percentage_used: float
    message: str
    created_at: datetime
    acknowledged: bool = False


@router.get("/job/{job_id}", response_model=JobCostResponse)
async def get_job_costs(job_id: str, include_trace: bool = Query(default=False)):
    """Get detailed cost breakdown for a specific research job"""
    tracker = get_cost_tracker(job_id)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobCostResponse(
        job_id=tracker.job_id,
        tier=tracker.tier,
        started_at=datetime.fromtimestamp(tracker.started_at),
        completed_at=datetime.fromtimestamp(tracker.completed_at) if tracker.completed_at else None,
        duration_seconds=tracker.get_duration(),
        total_cost=tracker.total_cost,
        cost_breakdown=tracker.get_cost_breakdown(),
        stage_costs=tracker.stage_costs,
        usage_summary=tracker.get_usage_summary(),
        efficiency_rating=tracker.get_tier_efficiency_rating(),
        within_budget=tracker.validate_tier_budget(),
        cost_per_token=tracker.total_cost / max(tracker.openrouter_tokens, 1) if tracker.openrouter_tokens > 0 else None,
        cost_per_search=tracker.total_cost / max(tracker.serper_calls, 1) if tracker.serper_calls > 0 else None,
        trace=(tracker.get_trace() if include_trace else None)
    )


@router.get("/summary", response_model=CostSummaryResponse)
async def get_cost_summary(
    days: int = Query(default=7, description="Number of days to summarize", ge=1, le=365),
    tier: Optional[str] = Query(default=None, description="Filter by specific tier")
):
    """Get cost summary across all jobs for specified period"""
    
    # Get active trackers for recent data
    active_trackers = get_active_trackers()
    completed_trackers = {k: v for k, v in active_trackers.items() if v.completed_at is not None}
    
    # Calculate summary metrics
    total_jobs = len(completed_trackers)
    if total_jobs == 0:
        period_start = datetime.now() - timedelta(days=days)
        period_end = datetime.now()
        return CostSummaryResponse(
            total_jobs=0,
            total_cost=0.0,
            avg_cost_per_job=0.0,
            period_start=period_start,
            period_end=period_end
        )
    
    # Filter by tier if specified
    if tier:
        completed_trackers = {k: v for k, v in completed_trackers.items() 
                            if v.tier and v.tier.upper() == tier.upper()}
        total_jobs = len(completed_trackers)
    
    # Aggregate data
    total_cost = sum(t.total_cost for t in completed_trackers.values())
    avg_cost = total_cost / total_jobs if total_jobs > 0 else 0
    
    # Tier breakdown
    broadcast_trackers = [t for t in completed_trackers.values() if t.tier == "BROADCAST"]
    targeted_trackers = [t for t in completed_trackers.values() if t.tier == "TARGETED"]  
    precision_trackers = [t for t in completed_trackers.values() if t.tier == "PRECISION"]
    
    # Provider costs
    openrouter_cost = sum(t.openrouter_cost for t in completed_trackers.values())
    serper_cost = sum(t.serper_cost for t in completed_trackers.values())
    scrapedo_cost = sum(t.scrapedo_cost for t in completed_trackers.values())
    firecrawl_cost = sum(t.firecrawl_cost for t in completed_trackers.values())
    
    # Performance metrics
    durations = [t.get_duration() for t in completed_trackers.values() if t.get_duration()]
    avg_duration = sum(durations) / len(durations) if durations else None
    
    # Efficiency rating (most common)
    ratings = [t.get_tier_efficiency_rating() for t in completed_trackers.values()]
    rating_counts = {}
    for rating in ratings:
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    avg_efficiency_rating = max(rating_counts.items(), key=lambda x: x[1])[0] if rating_counts else "unknown"
    
    period_start = datetime.now() - timedelta(days=days)
    period_end = datetime.now()
    
    return CostSummaryResponse(
        total_jobs=total_jobs,
        total_cost=total_cost,
        avg_cost_per_job=avg_cost,
        broadcast_jobs=len(broadcast_trackers),
        broadcast_cost=sum(t.total_cost for t in broadcast_trackers),
        targeted_jobs=len(targeted_trackers),
        targeted_cost=sum(t.total_cost for t in targeted_trackers),
        precision_jobs=len(precision_trackers),
        precision_cost=sum(t.total_cost for t in precision_trackers),
        openrouter_cost=openrouter_cost,
        serper_cost=serper_cost,
        scrapedo_cost=scrapedo_cost,
        firecrawl_cost=firecrawl_cost,
        avg_duration_seconds=avg_duration,
        avg_efficiency_rating=avg_efficiency_rating,
        period_start=period_start,
        period_end=period_end
    )


@router.get("/providers", response_model=List[ProviderStatsResponse])
async def get_provider_costs():
    """Get current cost breakdown by provider"""
    
    # Get active and completed trackers
    all_trackers = get_active_trackers()
    
    providers = ["openrouter", "serper", "scrapedo", "firecrawl"]
    stats = []
    
    for provider in providers:
        if provider == "openrouter":
            total_calls = sum(t.openrouter_calls for t in all_trackers.values())
            total_cost = sum(t.openrouter_cost for t in all_trackers.values())
            successful_calls = total_calls  # Assume all successful if tracker exists
            failed_calls = 0
        elif provider == "serper":
            total_calls = sum(t.serper_calls for t in all_trackers.values()) 
            total_cost = sum(t.serper_cost for t in all_trackers.values())
            successful_calls = total_calls
            failed_calls = 0
        elif provider == "scrapedo":
            total_calls = sum(t.scrapedo_calls for t in all_trackers.values())
            total_cost = sum(t.scrapedo_cost for t in all_trackers.values())
            successful_calls = total_calls
            failed_calls = 0
        elif provider == "firecrawl":
            total_calls = sum(t.firecrawl_calls for t in all_trackers.values())
            total_cost = sum(t.firecrawl_cost for t in all_trackers.values())
            successful_calls = total_calls
            failed_calls = 0
        
        success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0
        
        stats.append(ProviderStatsResponse(
            provider=provider,
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            success_rate=success_rate,
            total_cost=total_cost,
            avg_cost_per_call=avg_cost_per_call
        ))
    
    return stats


@router.get("/active", response_model=List[JobCostResponse])
async def get_active_jobs():
    """Get all currently active (running) research jobs"""
    
    active_trackers = {k: v for k, v in get_active_trackers().items() if v.completed_at is None}
    
    jobs = []
    for tracker in active_trackers.values():
        jobs.append(JobCostResponse(
            job_id=tracker.job_id,
            tier=tracker.tier,
            started_at=datetime.fromtimestamp(tracker.started_at),
            completed_at=None,
            duration_seconds=None,
            total_cost=tracker.total_cost,
            cost_breakdown=tracker.get_cost_breakdown(),
            stage_costs=tracker.stage_costs,
            usage_summary=tracker.get_usage_summary(),
            efficiency_rating="in_progress",
            within_budget=tracker.validate_tier_budget(),
            cost_per_token=tracker.total_cost / max(tracker.openrouter_tokens, 1) if tracker.openrouter_tokens > 0 else None,
            cost_per_search=tracker.total_cost / max(tracker.serper_calls, 1) if tracker.serper_calls > 0 else None
        ))
    
    return jobs


@router.get("/budget", response_model=TierBudgetConfig)
async def get_budget_config():
    """Get current budget configuration"""
    return TierBudgetConfig()


@router.post("/budget", response_model=TierBudgetConfig)
async def update_budget_config(config: TierBudgetConfig):
    """Update budget configuration (in-memory only for now)"""
    # In production, this would update database configuration
    return config


@router.get("/alerts", response_model=List[BudgetAlertResponse])
async def get_budget_alerts():
    """Get recent budget alerts"""
    
    # Check current active jobs for budget alerts
    active_trackers = {k: v for k, v in get_active_trackers().items() if v.completed_at is None}
    alerts = []
    
    budget_config = TierBudgetConfig()
    
    for tracker in active_trackers.values():
        if tracker.tier:
            alert_level = budget_config.get_alert_level(tracker.tier, tracker.total_cost)
            if alert_level != "normal":
                budget_limit = budget_config.get_limit(tracker.tier)
                percentage = (tracker.total_cost / budget_limit * 100) if budget_limit > 0 else 0
                
                alerts.append(BudgetAlertResponse(
                    job_id=tracker.job_id,
                    tier=tracker.tier,
                    alert_level=alert_level,
                    current_cost=tracker.total_cost,
                    budget_limit=budget_limit,
                    percentage_used=percentage,
                    message=f"Job {tracker.job_id} ({tracker.tier}) at {percentage:.1f}% of budget: ${tracker.total_cost:.6f} / ${budget_limit:.6f}",
                    created_at=datetime.now()
                ))
    
    return alerts


@router.post("/cleanup")
async def cleanup_old_trackers(max_age_hours: int = Query(default=24, description="Max age in hours for completed trackers")):
    """Clean up old completed cost trackers"""
    max_age_seconds = max_age_hours * 3600
    cleanup_completed_trackers(max_age_seconds)
    return {"message": f"Cleaned up trackers older than {max_age_hours} hours"}


@router.get("/efficiency/tier/{tier}")
async def get_tier_efficiency_analysis(tier: str):
    """Get efficiency analysis for a specific tier"""
    
    tier = tier.upper()
    if tier not in ["BROADCAST", "TARGETED", "PRECISION"]:
        raise HTTPException(status_code=400, detail="Invalid tier. Must be BROADCAST, TARGETED, or PRECISION")
    
    # Get completed trackers for the tier
    all_trackers = get_active_trackers()
    tier_trackers = [t for t in all_trackers.values() 
                    if t.completed_at and t.tier and t.tier.upper() == tier]
    
    if not tier_trackers:
        return {
            "tier": tier,
            "sample_size": 0,
            "analysis": "No completed jobs found for this tier"
        }
    
    # Calculate efficiency metrics
    costs = [t.total_cost for t in tier_trackers]
    durations = [t.get_duration() for t in tier_trackers if t.get_duration()]
    
    efficiency_ratings = {}
    for tracker in tier_trackers:
        rating = tracker.get_tier_efficiency_rating()
        efficiency_ratings[rating] = efficiency_ratings.get(rating, 0) + 1
    
    # Expected cost ranges
    expected_ranges = {
        "BROADCAST": (0.002, 0.008),
        "TARGETED": (0.005, 0.025),
        "PRECISION": (0.020, 0.080)
    }
    
    min_expected, max_expected = expected_ranges[tier]
    
    analysis = {
        "tier": tier,
        "sample_size": len(tier_trackers),
        "cost_analysis": {
            "min_cost": min(costs),
            "max_cost": max(costs),
            "avg_cost": sum(costs) / len(costs),
            "median_cost": sorted(costs)[len(costs)//2],
            "expected_range": {"min": min_expected, "max": max_expected},
            "within_range": sum(1 for c in costs if min_expected <= c <= max_expected),
            "over_budget": sum(1 for c in costs if c > max_expected)
        },
        "performance_analysis": {
            "avg_duration": sum(durations) / len(durations) if durations else None,
            "efficiency_distribution": efficiency_ratings
        },
        "recommendations": []
    }
    
    # Generate recommendations
    over_budget_pct = (analysis["cost_analysis"]["over_budget"] / len(tier_trackers)) * 100
    avg_cost = analysis["cost_analysis"]["avg_cost"]
    
    if over_budget_pct > 20:
        analysis["recommendations"].append(f"High over-budget rate ({over_budget_pct:.1f}%). Consider optimizing research depth for {tier} tier.")
    
    if avg_cost < min_expected:
        analysis["recommendations"].append(f"Average cost (${avg_cost:.6f}) below expected range. Consider increasing research depth for better quality.")
    elif avg_cost > max_expected:
        analysis["recommendations"].append(f"Average cost (${avg_cost:.6f}) above expected range. Consider optimizing search parameters or model selection.")
    else:
        analysis["recommendations"].append(f"Cost performance is within expected range for {tier} tier.")
    
    return analysis
