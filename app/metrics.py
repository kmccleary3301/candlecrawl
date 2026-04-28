from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Crawl metrics
crawl_jobs_total = Counter("crawl_jobs_total", "Total crawl jobs started")
crawl_jobs_completed_total = Counter("crawl_jobs_completed_total", "Total crawl jobs completed")
crawl_jobs_cancelled_total = Counter("crawl_jobs_cancelled_total", "Total crawl jobs cancelled")
crawl_pages_fetched_total = Counter("crawl_pages_fetched_total", "Total pages fetched across jobs")
crawl_frontier_size = Gauge("crawl_frontier_size", "Current frontier queue size")
crawl_inflight_gauge = Gauge("crawl_inflight", "Current in-flight URL count")
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)

budget_guardrail_events_total = Counter(
    "candlecrawl_budget_guardrail_events_total",
    "Budget guardrail events observed by CandleCrawl.",
    ["endpoint", "guardrail", "action"],
)

kill_switch_events_total = Counter(
    "candlecrawl_kill_switch_events_total",
    "Kill-switch activation events observed by CandleCrawl.",
    ["endpoint", "kill_switch"],
)

security_denies_total = Counter(
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


