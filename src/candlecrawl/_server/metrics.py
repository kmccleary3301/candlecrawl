from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest


def _registered(name: str) -> Any | None:
    return getattr(REGISTRY, "_names_to_collectors", {}).get(name)


def _counter(name: str, documentation: str, labelnames: list[str] | None = None):
    return _registered(name) or _registered(f"{name}_total") or Counter(name, documentation, labelnames or [])


def _gauge(name: str, documentation: str):
    return _registered(name) or Gauge(name, documentation)


def _histogram(name: str, documentation: str, *, buckets: tuple[float, ...]):
    return _registered(name) or Histogram(name, documentation, buckets=buckets)


# Crawl metrics. These helpers make the private packaged server importable in a
# dev process that has already imported the legacy top-level app module.
crawl_jobs_total = _counter("crawl_jobs_total", "Total crawl jobs started")
crawl_jobs_completed_total = _counter("crawl_jobs_completed_total", "Total crawl jobs completed")
crawl_jobs_cancelled_total = _counter("crawl_jobs_cancelled_total", "Total crawl jobs cancelled")
crawl_pages_fetched_total = _counter("crawl_pages_fetched_total", "Total pages fetched across jobs")
crawl_frontier_size = _gauge("crawl_frontier_size", "Current frontier queue size")
crawl_inflight_gauge = _gauge("crawl_inflight", "Current in-flight URL count")
http_request_duration_seconds = _histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

budget_guardrail_events_total = _counter(
    "candlecrawl_budget_guardrail_events_total",
    "Budget guardrail events observed by CandleCrawl.",
    ["endpoint", "guardrail", "action"],
)

kill_switch_events_total = _counter(
    "candlecrawl_kill_switch_events_total",
    "Kill-switch activation events observed by CandleCrawl.",
    ["endpoint", "kill_switch"],
)

security_denies_total = _counter(
    "candlecrawl_security_denies_total",
    "Denied requests for auth/tenant/ssrf/rate-limit controls.",
    ["endpoint", "reason"],
)


def record_budget_guardrail(endpoint: str, guardrail: str, action: str) -> None:
    budget_guardrail_events_total.labels(
        endpoint=endpoint,
        guardrail=guardrail,
        action=action,
    ).inc()


def record_kill_switch(endpoint: str, kill_switch: str) -> None:
    kill_switch_events_total.labels(
        endpoint=endpoint,
        kill_switch=kill_switch,
    ).inc()


def record_security_deny(endpoint: str, reason: str) -> None:
    security_denies_total.labels(endpoint=endpoint, reason=reason).inc()


def metrics_response():
    content = generate_latest()
    return content, CONTENT_TYPE_LATEST
