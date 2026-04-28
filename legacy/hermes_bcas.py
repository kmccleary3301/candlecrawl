from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from app.providers.openrouter import OpenRouterClient, OpenRouterChatRequest, ORMessage
from app.providers.serper import SerperClient, SerperSearchRequest
from app.providers.scrapedo import ScrapeDoClient, ScrapeDoRequest
from app.scraper import scraper
from app.models import ScrapeOptions, ContactExtractionConfig
from app.cost_tracking import JobCostTracker, create_cost_tracker, complete_cost_tracker
from app.model_pricing import estimate_cost_usd

# Import the original BCAS tool-calling abstraction (ensure repo root on sys.path)
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
from bcas_original_tool_calling.base import (
    FunctionCallDefinition as BCAS_FunctionCallDefinition,
    FunctionCallArgumentDefinition,
)
from bcas_original_tool_calling.pythonic_02 import Pythonic02
from contact_extraction_specification import build_contact_extraction_prompt, build_contact_output_format_instructions


# ---------------- Prompts (preserved) ----------------
def get_search_limitation_hint(search_limit: int | Literal["unlimited"]):
    if search_limit == "unlimited":
        return "You get unlimited searches.\n"
    elif search_limit == 1:
        return "You only get to use one search to answer the question.\n"
    else:
        return f"You have a maximum of {search_limit} searches available for this task.\n"


def get_context_limitation_hint(context_limit: int | Literal["unlimited"]):
    if context_limit == "unlimited":
        return "You get to write as much as you want in your responses.\n"
    elif context_limit <= 750:
        word_count_approx = (context_limit * 3) // 4
        return f"You are limited to {word_count_approx} words in your responses combined. This is a small context window. Please keep this in mind, and be CONCISE.\n"
    else:
        word_count_approx = (context_limit * 3) // 4
        return f"You are limited to {word_count_approx} words in your responses combined.\n"


SGS_SYSTEM_PROMPT = (
    """
You are tasked with searching for information to answer a question.
You will be given a question, and you must perform searches to find the information necessary to answer it.
DO NOT answer with information not provided by searches.
DO NOT make up information.
ONLY answer with information you have found in the database.
"""
).strip()

PLANNER_USER_PROMPT = (
    """
You will be given a question on a topic requiring 1-2 paragraphs to answer
You must attempt to answer it by performing consecutive searches and retrieving sources until you are ready to answer.
You must perform these searches and make notes of the information as you parse through it until you feel confident that you can answer the question.

Create a plan on how you are going to use your searches to gather information to answer the question.
Keep in mind potential limitations in the sources and search corpus. 
After this, you will be given tools to search, and you should follow the plan you created.

{search_limitation}{context_limitation}

Your research goal is: {question}
"""
).strip()

SGS_SEARCHING_PROMPT_1 = (
    """
You will be given a question on a topic requiring 1-2 paragraphs to answer
You must attempt to answer it by performing consecutive searches and retrieving sources until you are ready to answer.
You must perform these searches and make notes of the information as you parse through it until you feel confident that you can answer the question.
When you perform a search or otherwise call a function, you will be met with the result as a response. You may then continue.
Complete the process in an ideal way by calling provided functions.
Your functions will be search_database() for searching, and ready_to_answer() to indicate that you are ready to answer.
{search_limitation}{context_limitation}

Here is your question: {question}
"""
).strip()

# Final answer guidance is constructed dynamically at answer time only
# to keep citation instructions out of earlier planning/search prompts.
def build_final_answer_prompt(sources_count: int, tier: Optional[str] = None, enforce_length: bool = False, contact_extraction: Optional[ContactExtractionConfig] = None) -> str:
    """Build the final answer instruction shown only when answering.

    - If sources are available, require inline citations like [S1], [S2]
      that refer to the `ref: [S#]` headers in <SEARCH_RESULTS>.
    - If no sources are available, instruct to return "insufficient evidence".
    - Tier is accepted for potential future formatting/length targeting, but
      we keep this minimal to match the current request.
    - If contact_extraction is provided, append structured XML output instructions.
    """
    if sources_count <= 0:
        return (
            "No reliable sources were found in <SEARCH_RESULTS>.\n"
            "Reply exactly with: insufficient evidence"
        )

    lines = [
        "Write your final answer only.",
        "Use only information supported by sources present in <SEARCH_RESULTS> (with headers `ref: [S#]`).",
        "If a claim cannot be supported by those sources, omit it or state 'insufficient evidence'.",
    ]
    if enforce_length and tier:
        tr = tier.upper()
        if tr == "BROADCAST":
            lines.append("Target length: ~600–1000 words (flexible).")
        elif tr == "TARGETED":
            lines.append("Target length: ~1200–2000 words (flexible).")
        elif tr == "PRECISION":
            lines.append("Target length: ~3000–5000 words (flexible).")
    
    # Add contact extraction instructions if configured
    if contact_extraction:
        contact_format_instructions = build_contact_output_format_instructions(contact_extraction)
        lines.append(contact_format_instructions)
    
    return "\n".join(lines)

ENCOURAGE_INTERMITENT_REASONING_PROMPT = (
    """
Analyze these search results.
Create a plan on how you are going to proceed with your search.
Keep in mind potential limitations in the sources and search corpus. 

After you are done, you MUST call one of the provided tools. Complete your response by calling one of the provided tools.

Here is your question: {question}
"""
).strip()


# ---------------- Tool caller (Pythonic02) ----------------


# ---------------- Retrieval structs ----------------
class RetrievedSource(BaseModel):
    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    meta: Dict[str, Any] = {}


class HermesSearchOrchestrator:
    def __init__(self, cost_tracker: Optional[JobCostTracker] = None, tier: Optional[str] = None, force_direct: bool = False):
        self.serper = SerperClient()
        self.scrapedo = ScrapeDoClient()
        self.cost_tracker = cost_tracker
        self.tier = (tier or "").upper() if tier else None
        self.force_direct = bool(force_direct)

    async def _scrape_one(self, url: str) -> RetrievedSource:
        # First try internal scraper
        options = ScrapeOptions(only_main_content=True, formats=["markdown", "links"])  # light
        doc = await scraper.scrape_url(url, options)
        if (not doc.markdown) or (doc.metadata and doc.metadata.blocked):
            # fallback to Scrape.do (without JS rendering due to plan limitations)
            try:
                resp, scrape_cost = await self.scrapedo.fetch(ScrapeDoRequest(url=url, render_js=False, timeout_ms=20000))
                if self.cost_tracker:
                    self.cost_tracker.add_scrapedo_cost()
                    self.cost_tracker.add_stage_cost("extraction", scrape_cost)
                html = resp.content or ""
                # Use scraper to convert HTML to markdown via _process_html_content
                try:
                    processed = await scraper._process_html_content(url, html, 200, options)  # type: ignore
                    markdown = processed.markdown or ""
                except Exception:
                    markdown = html
                if self.cost_tracker:
                    # Treat conversion as one Firecrawl processing unit
                    self.cost_tracker.add_firecrawl_scrape()
                return RetrievedSource(text=markdown[:20000], url=url, title=doc.metadata.title if doc.metadata else None, meta={"fallback": True})
            except Exception as e:
                # Trace failure and return empty content with error metadata
                if self.cost_tracker:
                    try:
                        self.cost_tracker.add_trace_event(
                            provider="scrapedo",
                            event_type="scrape",
                            status="error",
                            latency_ms=None,
                            cost_delta=None,
                            request_summary=f"url={url[:120]}",
                            response_excerpt=str(getattr(e, 'payload', '') or str(e))[:200],
                        )
                    except Exception:
                        pass
                return RetrievedSource(text="", url=url, title=None, meta={"fallback": True, "error": str(e)[:200]})
        
        # Track Firecrawl processing cost
        if self.cost_tracker:
            self.cost_tracker.add_firecrawl_scrape()
        
        # Process document with chunking if needed
        from .chunking import process_document_for_search
        markdown_content = doc.markdown or ""
        
        if not markdown_content.strip():
            return RetrievedSource(text="", url=url, title=doc.metadata.title if doc.metadata else None, meta={"fallback": False, "error": "No content"})
        
        # Use chunking logic - if document is small enough, return full doc; otherwise return best chunk
        processed_docs = process_document_for_search(
            text=markdown_content,
            url=url,
            title=doc.metadata.title if doc.metadata else None,
            max_tokens=500,  # Max tokens for full document
            chunk_size=1200,  # Chunk size in characters  
            overlap=200  # Overlap between chunks
        )
        
        if not processed_docs:
            return RetrievedSource(text="", url=url, title=doc.metadata.title if doc.metadata else None, meta={"fallback": False, "error": "No processable content"})
        
        # For direct scraping, return the first chunk/document with metadata
        best_doc = processed_docs[0]
        
        return RetrievedSource(
            text=best_doc['text'], 
            url=url, 
            title=doc.metadata.title if doc.metadata else None, 
            meta={
                "fallback": False, 
                "chunking_metadata": best_doc['metadata'],
                "total_processed_chunks": len(processed_docs)
            }
        )

    async def search_and_retrieve(self, query: str, limit: int = 10) -> List[RetrievedSource]:
        # Use Hermes RAG server for SERP -> scrape -> ingest -> retrieve pipeline
        import httpx
        
        hermes_rag_base = "http://127.0.0.1:8010"
        
        try:
            if self.force_direct:
                raise Exception("force_direct_scrape")
            import time as _time
            async with httpx.AsyncClient(timeout=120) as client:
                payload = {
                    "query": query,
                    "limit": limit,
                    "auth": None,  # Single-user mode
                    "rerank": True,  # Enable reranking for better results
                    "similarity_weight": 0.3,  # Increase similarity weight vs BM25
                    "bm25_weight": 0.7
                }
                
                _t0 = _time.time()
                # Prefer BM25-only for BROADCAST/TARGETED tiers to reduce fragility/latency
                path = "/v1/hermes/serp_ingest_bm25" if (self.tier in ("BROADCAST", "TARGETED")) else "/v1/hermes/serp_ingest_search"
                resp = await client.post(f"{hermes_rag_base}{path}", json=payload)
                resp.raise_for_status()
                _lat_ms = int((_time.time() - _t0) * 1000)
                
                data = resp.json()
                if not data.get("success"):
                    raise Exception(f"Hermes RAG search failed: {data}")
                
                search_id = data.get("search_id")
                sources = []
                if self.tier in ("BROADCAST", "TARGETED"):
                    # BM25-only path
                    bm25 = (data.get("bm25") or {})
                    rows = bm25.get("rows", []) if isinstance(bm25, dict) else []
                    for row in rows:
                        text = row.get("text", "")
                        meta = row.get("metadata", {}) if isinstance(row, dict) else {}
                        url = (meta or {}).get("url", "")
                        title = (meta or {}).get("title", "")
                        sources.append(RetrievedSource(text=text[:20000], url=url, title=title, meta={"search_id": search_id, "from_rag": True}))
                else:
                    # Hybrid path
                    hybrid_results = data.get("hybrid", {})
                    documents = hybrid_results.get("documents", [])
                    for doc in documents:
                        content = doc.get("content", "")
                        url = doc.get("metadata", {}).get("url", "")
                        title = doc.get("metadata", {}).get("title", "")
                        score = doc.get("score", 0.0)
                        sources.append(RetrievedSource(text=content[:20000], url=url, title=title, meta={"search_id": search_id, "score": score, "from_rag": True}))
                
                if self.cost_tracker:
                    try:
                        self.cost_tracker.add_trace_event(
                            provider="hermes_rag",
                            event_type="serp_ingest_search",
                            status="ok",
                            latency_ms=_lat_ms,
                            cost_delta=0.0,
                            request_summary=f"query='{query[:60]}', limit={limit}",
                            response_excerpt=f"docs={len(documents)}",
                        )
                    except Exception:
                        pass
                return sources
                
        except Exception as e:
            # Fallback to original implementation if RAG server fails
            import traceback
            print(f"Warning: RAG server failed ({e}), falling back to direct scraping")
            print(traceback.format_exc())
            if self.cost_tracker:
                try:
                    self.cost_tracker.add_trace_event(
                        provider="hermes_rag",
                        event_type="serp_ingest_search",
                        status="error",
                        latency_ms=None,
                        cost_delta=0.0,
                        request_summary=f"query='{query[:60]}'",
                        response_excerpt=str(e)[:200],
                    )
                except Exception:
                    pass
            
            search_id = str(uuid.uuid4())
            import time as _time
            _t_s = _time.time()
            res, serp_cost = await self.serper.search(SerperSearchRequest(q=query, num=limit))
            _lat_s = int((_time.time() - _t_s) * 1000)
            if self.cost_tracker:
                self.cost_tracker.add_serper_cost()
                self.cost_tracker.add_stage_cost("sourcing", serp_cost)
                try:
                    self.cost_tracker.add_trace_event(
                        provider="serper",
                        event_type="search",
                        status="ok",
                        latency_ms=_lat_s,
                        cost_delta=serp_cost,
                        request_summary=f"q='{query[:60]}', num={limit}",
                        response_excerpt=(res.organic[0].title if (res.organic) else None),
                    )
                except Exception:
                    pass
            urls: List[str] = []
            for o in (res.organic or [])[:limit]:
                if o.link and o.link not in urls:
                    urls.append(o.link)
            # Query diversification: if no URLs, broaden query across common sources
            if not urls:
                q2 = f"{query} site:linkedin.com OR site:facebook.com"
                try:
                    res2, serp_cost2 = await self.serper.search(SerperSearchRequest(q=q2, num=limit))
                    if self.cost_tracker:
                        self.cost_tracker.add_serper_cost()
                        self.cost_tracker.add_stage_cost("sourcing", serp_cost2)
                        try:
                            self.cost_tracker.add_trace_event(
                                provider="serper",
                                event_type="search",
                                status="ok",
                                latency_ms=None,
                                cost_delta=serp_cost2,
                                request_summary=f"q='{q2[:60]}', num={limit}",
                                response_excerpt=(res2.organic[0].title if (res2.organic) else None),
                            )
                        except Exception:
                            pass
                    for o in (res2.organic or [])[:limit]:
                        if o.link and o.link not in urls:
                            urls.append(o.link)
                except Exception:
                    pass
            # Scrape concurrently but politely keep a small concurrency
            sem = asyncio.Semaphore(4)

            async def guard(u: str) -> RetrievedSource:
                async with sem:
                    import time as _time
                    _t = _time.time()
                    result = await self._scrape_one(u)
                    _lat = int((_time.time() - _t) * 1000)
                    if self.cost_tracker:
                        try:
                            self.cost_tracker.add_trace_event(
                                provider="scrapedo" if result.meta.get("fallback") else "firecrawl",
                                event_type="scrape",
                                status="ok",
                                latency_ms=_lat,
                                cost_delta=None,
                                request_summary=f"url={u[:120]}",
                                response_excerpt=(result.title or "")[:120],
                                extra={"fallback": bool(result.meta.get("fallback"))},
                            )
                        except Exception:
                            pass
                    return result

            results = await asyncio.gather(*[guard(u) for u in urls])
            for r in results:
                r.meta["search_id"] = search_id
                r.meta["from_rag"] = False
            return results


class HermesBStar04:
    def __init__(self, *, model: str = "openai/gpt-5-nano", temperature: float = 0.0, max_completion_tokens: int = 2048, job_id: Optional[str] = None, tier: Optional[str] = None, contact_extraction: Optional[Any] = None, **kwargs: Any):
        self.openrouter = OpenRouterClient()
        self.model_name = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.method_kwargs = kwargs
        self.contact_extraction = contact_extraction
        self.tool_caller = Pythonic02()
        self.trace_level = str(kwargs.get("debug_trace_level", "summary")).lower()
        self.trace_excerpt_len = 800 if self.trace_level == "verbose" else 200
        
        # Initialize cost tracking
        self.job_id = job_id or str(uuid.uuid4())
        self.tier = tier
        self.cost_tracker = create_cost_tracker(self.job_id, tier)
        self.search = HermesSearchOrchestrator(self.cost_tracker, tier=tier, force_direct=bool(kwargs.get("force_direct_scrape", False)))

    async def _call_llm(self, chat_history: List[Dict[str, str]], functions_available: Optional[List[BCAS_FunctionCallDefinition]] = None, stage: Optional[str] = None):
        messages = [{"role": e["role"], "content": e["content"]} for e in chat_history]
        if functions_available:
            tools_prompt = self.tool_caller.generate_tools_prompt(functions_available)
            assert messages[-1]["role"] == "user"
            tool_prompt_formatted = f"SYSTEM MESSAGE - AVAILABLE FUNCTIONS\n<FUNCTIONS>{tools_prompt}</FUNCTIONS>\nEND SYSTEM MESSAGE\n\n"
            messages[-1]["content"] += f"\n\n{tool_prompt_formatted}"

        req = OpenRouterChatRequest(
            model=self.model_name,
            messages=[ORMessage(role=m["role"], content=m["content"]) for m in messages],
            temperature=self.temperature if not self.model_name.startswith("o4") else 1.0,
            max_tokens=self.max_completion_tokens,
        )
        import time as _time
        _t0 = _time.time()
        response, actual_cost = await self.openrouter.chat_completions(req)
        _lat_ms = int((_time.time() - _t0) * 1000)
        
        # Track OpenRouter cost
        if self.cost_tracker:
            est_cost = None
            try:
                if (actual_cost is None) and response.usage:
                    est_cost = estimate_cost_usd(
                        self.model_name,
                        prompt_tokens=int(response.usage.prompt_tokens or 0),
                        completion_tokens=int(response.usage.completion_tokens or 0),
                    )
            except Exception:
                est_cost = None

            self.cost_tracker.add_openrouter_cost(response.usage, actual_cost or est_cost)
            cost_to_add = (actual_cost or (response.usage.cost if response.usage else None) or est_cost or 0.0)
            if stage:
                self.cost_tracker.add_stage_cost(stage, cost_to_add)
            try:
                self.cost_tracker.add_trace_event(
                    provider="openrouter",
                    event_type="chat",
                    status="ok",
                    latency_ms=_lat_ms,
                    cost_delta=cost_to_add,
                    model=self.model_name,
                    request_summary=f"messages={len(messages)}, max_tokens={self.max_completion_tokens}",
                    response_excerpt=(response.choices[0].message.content or "")[:self.trace_excerpt_len] if response.choices else None,
                )
            except Exception:
                pass
        
        return response

    async def run_async(self, question: str, max_searches: int | Literal["unlimited"]) -> Dict[str, Any]:
        start_time = time.time()
        current_token_usage = 0
        
        # Enhance question with contact extraction requirements if configured
        enhanced_question = question
        if self.contact_extraction:
            enhanced_question = build_contact_extraction_prompt(question, self.contact_extraction)
            # Store output format instructions for later use
            self._contact_output_format = build_contact_output_format_instructions(self.contact_extraction)

        max_total_tokens = self.method_kwargs.get("max_total_tokens", 8000)
        max_responses_per_turn = self.method_kwargs.get("max_responses", 25)
        tell_search_limit = self.method_kwargs.get("tell_search_limit", True)
        tell_context_limit = self.method_kwargs.get("tell_context_limit", True)
        search_limit = self.method_kwargs.get("search_limit", 5)
        use_intermittent_reasoning = self.method_kwargs.get("use_intermittent_reasoning", False)
        use_preplanning = self.method_kwargs.get("use_preplanning", True)

        system_entry = {"role": "system", "content": SGS_SYSTEM_PROMPT}

        # Planner stage (optional)
        chat_history: List[Dict[str, str]] = [system_entry]
        if use_preplanning:
            plan_history = [system_entry, {"role": "user", "content": PLANNER_USER_PROMPT.format(
                search_limitation=get_search_limitation_hint(max_searches),
                context_limitation=get_context_limitation_hint(max_total_tokens) if tell_context_limit else "",
                question=enhanced_question,
            )}]
            planner_response = await self._call_llm(plan_history, stage="planning")
            planner_msg = planner_response.choices[0].message
            # Track token usage if available
            try:
                usage = planner_response.usage or {}
                current_token_usage += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            except Exception:
                pass
            plan_history.append({"role": "assistant", "content": planner_msg.content})
            chat_history = plan_history[:]

        all_retrieved_sources: List[RetrievedSource] = []
        global_search_count = 0
        previous_searches: set[str] = set()

        current_turn = 0
        ready_to_answer_flag = False
        final_answer_content = ""
        duplicate_search_flag = False
        consecutive_no_tool_turns = 0

        def append_to_chat_history(content: str) -> None:
            nonlocal chat_history
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history[-1]["content"] += "\n\n" + content
            else:
                chat_history.append({"role": "user", "content": content})

        usage_hint = f"You have used {current_token_usage}/{max_total_tokens} tokens."
        append_to_chat_history(SGS_SEARCHING_PROMPT_1.format(
            search_limitation=get_search_limitation_hint(max_searches) if tell_search_limit else "",
            context_limitation=usage_hint if tell_context_limit else "",
            question=question,
        ))

        ready_to_answer_def = BCAS_FunctionCallDefinition(
            name="ready_to_answer",
            description="Call this when you have sufficient information to answer the main question.",
            parameters=[],
        )
        search_def = BCAS_FunctionCallDefinition(
            name="search_database",
            description="Search the database for information relevant to the current research step.",
            parameters=[
                FunctionCallArgumentDefinition(
                    name="query", type="str", description="A targeted search query for the current sub-question."
                )
            ],
        )

        while True:
            current_turn += 1
            if current_turn > max_responses_per_turn:
                final_answer_content = "Model ran out of responses without calling ready_to_answer."
                break

            used_tokens = current_token_usage
            remaining_percentage = max(0, (1 - (used_tokens / max_total_tokens)) * 100 if max_total_tokens > 0 else 0)
            usage_hint = f"You have used {used_tokens}/{max_total_tokens} tokens. You have {remaining_percentage:.0f}% of your context remaining."
            searches_left = (max_searches - global_search_count) if max_searches != "unlimited" else "an unlimited number of"
            search_hint = f"You have {searches_left} searches remaining."

            user_prompt_content = (usage_hint + "\n" + search_hint) if (current_turn > 1) else ""
            if use_intermittent_reasoning and (global_search_count > 0):
                user_prompt_content += "\n\n" + ENCOURAGE_INTERMITENT_REASONING_PROMPT.format(question=question)
            else:
                user_prompt_content += "\n\nYou MUST call one of the provided tools. Only respond with your tool call."
            if ready_to_answer_flag:
                user_prompt_content = build_final_answer_prompt(
                    len(all_retrieved_sources), self.tier, bool(self.method_kwargs.get("enforce_tier_length")), self.contact_extraction
                )
            append_to_chat_history(user_prompt_content)

            all_available_tools = [] if ready_to_answer_flag else [ready_to_answer_def] + ([search_def] if ((not isinstance(max_searches, int)) or (global_search_count < max_searches)) else [])

            response = await self._call_llm(chat_history, all_available_tools, stage="synthesis")
            response_msg = response.choices[0].message
            # Track token usage if available
            try:
                usage = response.usage or {}
                current_token_usage += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            except Exception:
                pass
            chat_history.append({"role": "assistant", "content": response_msg.content})

            if ready_to_answer_flag:
                # If sources exist but the model omitted citations, enforce a rewrite.
                final_answer = response_msg.content or ""
                try:
                    have_sources = len(all_retrieved_sources) > 0
                    has_citation = bool(re.search(r"\[S\d+\]", final_answer))
                except Exception:
                    have_sources, has_citation = False, False

                if have_sources:
                    enforce_msg = (
                        "Include inline citations using [S1], [S2], etc., where each [S#] refers to the corresponding"
                        " source marked with `ref: [S#]` in <SEARCH_RESULTS>.\n"
                        "Rewrite the previous answer accordingly.\n"
                        "- Do not add any new claims.\n"
                        "- Remove any sentence that cannot be supported by the provided sources.\n"
                        "- Output only the rewritten answer."
                    )
                    chat_history.append({"role": "user", "content": enforce_msg})
                    rewritten = await self._call_llm(chat_history, None, stage="synthesis")
                    final_answer = rewritten.choices[0].message.content or final_answer
                    # If still missing citations, return insufficient evidence to avoid unsupported content
                    try:
                        has_citation = bool(re.search(r"\[S\d+\]", final_answer))
                    except Exception:
                        has_citation = False
                    # Precision: push for more citations
                    if (self.tier or "").upper() == "PRECISION":
                        try:
                            cites = re.findall(r"\[S\d+\]", final_answer or "")
                            if len(cites) < 3 and have_sources:
                                chat_history.append({"role": "user", "content": "Ensure at least three citations [S#] from <SEARCH_RESULTS>. Output only the revised answer."})
                                rewritten2 = await self._call_llm(chat_history, None, stage="synthesis")
                                final_answer = rewritten2.choices[0].message.content or final_answer
                        except Exception:
                            pass
                    if not has_citation:
                        final_answer = "insufficient evidence"
                    chat_history.append({"role": "assistant", "content": final_answer})

                # Complete cost tracking
                if self.cost_tracker:
                    self.cost_tracker.complete_job()
                    complete_cost_tracker(self.job_id)
                
                return {
                    "chat_history": chat_history,
                    "output": final_answer,
                    "responses": len(chat_history),
                    "time_taken": time.time() - start_time,
                    "sources": [s.model_dump() for s in all_retrieved_sources],
                    "cost_breakdown": self.cost_tracker.get_cost_breakdown() if self.cost_tracker else None,
                    "usage_summary": self.cost_tracker.get_usage_summary() if self.cost_tracker else None,
                    "total_cost": self.cost_tracker.total_cost if self.cost_tracker else None,
                    "efficiency_rating": self.cost_tracker.get_tier_efficiency_rating() if self.cost_tracker else None,
                }

            tool_calls = self.tool_caller.parse_tool_calls(response_msg.content, all_available_tools)
            if not tool_calls:
                consecutive_no_tool_turns += 1
                if consecutive_no_tool_turns >= 2:
                    # Force final answer to avoid spinning without tool calls
                    ready_to_answer_flag = True
                    append_to_chat_history(build_final_answer_prompt(len(all_retrieved_sources), self.tier, False, self.contact_extraction))
                    consecutive_no_tool_turns = 0
                else:
                    chat_history.append({"role": "user", "content": self.tool_caller.CORRECTION_PROMPT})
                continue
            else:
                consecutive_no_tool_turns = 0
                # Pythonic02 returns ToolCallParsed(BaseModel) with .model_dump()
                chat_history[-1]["tool_calls"] = [tc.model_dump() for tc in tool_calls]

            for tool_call in tool_calls:
                fn = tool_call.function
                args = tool_call.arguments
                if fn == "search_database":
                    query = args.get("query", question)
                    if query in previous_searches:
                        if duplicate_search_flag:
                            ready_to_answer_flag = True
                            append_to_chat_history(
                                "You attempted to make the same search three times. This is not allowed.\n\n"
                                + build_final_answer_prompt(len(all_retrieved_sources), self.tier, False, self.contact_extraction)
                            )
                            break
                        duplicate_search_flag = True
                        append_to_chat_history("You have already requested this search. Do NOT make the same search twice, or the attempt will end.")
                        continue

                    previous_searches.add(query)
                    global_search_count += 1
                    if max_searches != "unlimited" and global_search_count > max_searches:
                        final_answer_content = "Exceeded search budget. Cannot perform more searches."
                        break

                    # SERP -> scrape (fallback to Scrape.do) -> ingest/retrieve via RAG (if available)
                    results = await self.search.search_and_retrieve(query, limit=min(search_limit, 10))
                    # Attach reference ids for citation mapping
                    start_idx = len(all_retrieved_sources) + 1
                    for i, rs in enumerate(results, start=start_idx):
                        try:
                            rs.meta["ref"] = f"S{i}"
                        except Exception:
                            pass
                    all_retrieved_sources.extend(results)

                    # Present sources with explicit ref headers and metadata for citation
                    blocks = []
                    for idx, rs in enumerate(results, start=start_idx):
                        ref_id = f"S{idx}"
                        url = rs.url or ""
                        title = rs.title or ""
                        header = f"ref: [{ref_id}]\nurl: {url}\ntitle: {title}"
                        blocks.append(
                            f"<SOURCE>\n{header}\n<CONTENT>\n{rs.text}\n</CONTENT>\n</SOURCE>\n"
                        )
                    append_to_chat_history(f"<SEARCH_RESULTS>\n\n{''.join(blocks)}\n</SEARCH_RESULTS>")

                    if isinstance(max_searches, int) and global_search_count >= max_searches:
                        ready_to_answer_flag = True
                        append_to_chat_history(
                            "You have reached your search limit. You cannot perform any more searches.\n\n"
                            + build_final_answer_prompt(
                                len(all_retrieved_sources), self.tier, bool(self.method_kwargs.get("enforce_tier_length")), self.contact_extraction
                            )
                        )
                        break

                elif fn == "ready_to_answer":
                    ready_to_answer_flag = True
                    append_to_chat_history(
                        build_final_answer_prompt(
                            len(all_retrieved_sources), self.tier, bool(self.method_kwargs.get("enforce_tier_length")), self.contact_extraction
                        )
                    )
                    break
                else:
                    append_to_chat_history(self.tool_caller.CORRECTION_PROMPT)

            if (current_token_usage > max_total_tokens) and not ready_to_answer_flag:
                ready_to_answer_flag = True
                out_of_tokens_message = (
                    "You have run out of tokens. You cannot perform any more searches or reasoning steps.\n\n"
                    + build_final_answer_prompt(
                        len(all_retrieved_sources), self.tier, bool(self.method_kwargs.get("enforce_tier_length")), self.contact_extraction
                    )
                )
                if chat_history[-1]["role"] == "user":
                    chat_history[-1]["content"] = out_of_tokens_message
                else:
                    append_to_chat_history(out_of_tokens_message)

        return {
            "chat_history": chat_history,
            "output": final_answer_content,
            "responses": len(chat_history),
            "time_taken": time.time() - start_time,
            "sources": [s.model_dump() for s in all_retrieved_sources],
        }

    def run(self, question: str, max_searches: int | Literal["unlimited"]) -> Dict[str, Any]:
        return asyncio.run(self.run_async(question, max_searches))
