import re
import time
import groq
from langchain_core.messages import ToolMessage
from typing import Generator, Any
from multi_agent_agentic_application.agent.agent import *
from multi_agent_agentic_application.tool.scrap_url import scrape_url


def invoke_with_retry(runnable, input_data, max_retries: int = 4, initial_delay: float = 2.0):
    """Invoke a runnable/chain with exponential backoff on groq.RateLimitError."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except groq.RateLimitError as e:
            headers = getattr(e.response, "headers", {}) if getattr(e, "response", None) else {}
            retry_after = headers.get("retry-after") or headers.get("retry-after-ms")
            print(f"\n[Rate Limit] Groq RateLimitError encountered (attempt {attempt + 1}/{max_retries})")
            print(f"  Retry-After header: {retry_after}")
            print(f"  Response body: {getattr(e, 'body', str(e))}")
            if attempt == max_retries - 1:
                raise
            sleep_time = float(retry_after) if retry_after and str(retry_after).replace(".", "", 1).isdigit() else delay
            print(f"  Backing off for {sleep_time}s...")
            time.sleep(sleep_time)
            delay *= 2
    return None


def run_research_pipeline_stream(topic: str, max_urls: int = 3) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Execute the multi-agent research pipeline, yielding progress events at each stage."""
    state: dict[str, Any] = {
        "topic": topic,
        "search_result": "",
        "scraped_content": "",
        "sources": [],
        "report": "",
        "feedback": ""
    }

    # -------------------------
    # Stage 1: Search Agent
    # -------------------------
    yield {"stage": "search", "status": "running", "data": "Searching the web for authoritative sources..."}
    search_agent = build_search_agent()

    search_result = invoke_with_retry(
        search_agent,
        {
            "messages": [
                (
                    "user",
                    f"Find recent, reliable, and detailed information about: {topic}"
                )
            ]
        }
    )

    state["search_result"] = search_result["messages"][-1].content
    yield {"stage": "search", "status": "complete", "data": state["search_result"]}

    # -------------------------
    # Stage 2: URL Extraction & Direct Scraping
    # -------------------------
    yield {"stage": "scrape", "status": "running", "data": "Extracting URLs and scraping page contents..."}

    tool_messages_content = [
        msg.content for msg in search_result.get("messages", [])
        if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool"
    ]
    raw_search_text = "\n".join(tool_messages_content) if tool_messages_content else state["search_result"]

    extracted_urls = re.findall(r"https?://[^\s)\]\"'<>]+", raw_search_text)
    unique_urls = []
    for url in extracted_urls:
        if url not in unique_urls and "kaggle.com/datasets" not in url.lower():
            unique_urls.append(url)

    if not unique_urls:
        yield {"stage": "scrape", "status": "error", "data": "No valid URLs found in search output."}
        raise RuntimeError("Pipeline stopped: No valid URLs found in search output.")

    top_urls = unique_urls[:max_urls]
    scraped_entries = []
    sources_meta = []

    for url in top_urls:
        try:
            content = scrape_url.invoke({"url": url, "max_chars": 3000})
            if content and not content.startswith("Error:"):
                # Extract title if present
                title_match = re.search(r"^Title:\s*(.+)$", content, re.MULTILINE)
                title = title_match.group(1) if title_match else url
                scraped_text = content[:3000]
                scraped_entries.append(scraped_text)
                sources_meta.append({
                    "title": title,
                    "url": url,
                    "chars": len(scraped_text),
                    "content": scraped_text,
                    "success": True
                })
            else:
                sources_meta.append({
                    "title": "Extraction Failed",
                    "url": url,
                    "chars": 0,
                    "content": content or "Unknown scraping error.",
                    "success": False
                })
        except Exception as e:
            sources_meta.append({
                "title": "Request Failed",
                "url": url,
                "chars": 0,
                "content": str(e),
                "success": False
            })

    state["sources"] = sources_meta

    if not scraped_entries:
        yield {"stage": "scrape", "status": "error", "data": "All scraping attempts failed or returned errors."}
        raise RuntimeError("Pipeline stopped: All scraping attempts failed or returned errors.")

    state["scraped_content"] = "\n\n" + ("=" * 40) + "\n\n".join(scraped_entries)
    yield {"stage": "scrape", "status": "complete", "data": {"content": state["scraped_content"], "sources": sources_meta}}

    # -------------------------
    # Stage 3: Writer Chain
    # -------------------------
    yield {"stage": "write", "status": "running", "data": "Synthesizing research and drafting report..."}

    search_excerpt = state["search_result"][:3000]
    scraped_excerpt = state["scraped_content"][:5000]
    research_combined = (
        f"SEARCH RESULTS SUMMARY:\n{search_excerpt}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{scraped_excerpt}"
    )

    state["report"] = invoke_with_retry(
        writer_chain,
        {
            "topic": topic,
            "research": research_combined,
        }
    )
    yield {"stage": "write", "status": "complete", "data": state["report"]}

    # -------------------------
    # Stage 4: Critic Chain
    # -------------------------
    yield {"stage": "critic", "status": "running", "data": "Verifying statistics and reviewing report..."}

    state["feedback"] = invoke_with_retry(
        critic_chain,
        {
            "report": state['report'][:4000],
            "research": research_combined[:4000],
        }
    )
    yield {"stage": "critic", "status": "complete", "data": state["feedback"]}

    yield {"stage": "done", "status": "complete", "data": state}
    return state


def run_research_pipeline(topic: str) -> dict[str, Any]:
    """Backward-compatible wrapper that consumes run_research_pipeline_stream to completion."""
    final_state: dict[str, Any] = {}
    for event in run_research_pipeline_stream(topic):
        if event.get("stage") == "done" and "data" in event:
            final_state = event["data"]
    return final_state


