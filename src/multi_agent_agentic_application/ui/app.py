import os
import re
import time
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import groq
import streamlit as st
from dotenv import load_dotenv

from multi_agent_agentic_application.agent.agent import (
    build_search_agent,
    critic_chain,
    llm,
    writer_chain,
)
from multi_agent_agentic_application.pipeline.pipeline import run_research_pipeline_stream

load_dotenv()

# ==============================================================================
# 1. Page Configuration & Style Injection
# ==============================================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    """Load CSS from external styles.css file or embedded fallback."""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()


# ==============================================================================
# 2. Resource Caching & Startup Checks
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_cached_pipeline_resources():
    """Cache agent initializations across UI reruns."""
    return {
        "search_agent": build_search_agent(),
        "writer_chain": writer_chain,
        "critic_chain": critic_chain,
        "model_name": getattr(llm, "model_name", getattr(llm, "model", "Groq LLM")),
    }


def verify_api_keys() -> bool:
    """Verify required API keys exist before running pipeline."""
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    missing = []
    if not groq_key or groq_key.startswith("your_"):
        missing.append("GROQ_API_KEY")
    if not tavily_key or tavily_key.startswith("your_"):
        missing.append("TAVILY_API_KEY")

    if missing:
        st.error(
            f"Missing required API credentials: `{', '.join(missing)}`.\n\n"
            "Please configure your `.env` file before initiating research."
        )
        return False
    return True


# ==============================================================================
# 3. Session State Management
# ==============================================================================

if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "stepper_state" not in st.session_state:
    st.session_state.stepper_state = {
        "search": {"status": "pending", "time": 0.0, "desc": "Tavily web discovery"},
        "scrape": {"status": "pending", "time": 0.0, "desc": "Content extraction"},
        "write": {"status": "pending", "time": 0.0, "desc": "Report synthesis"},
        "critic": {"status": "pending", "time": 0.0, "desc": "Fact check & review"},
    }


# ==============================================================================
# 4. Critic Parsing & Formatting
# ==============================================================================

def parse_critic_feedback(feedback_text: str) -> dict[str, Any]:
    """Parse structured critic feedback with graceful fallback."""
    parsed: dict[str, Any] = {
        "score": None,
        "score_float": 0.0,
        "strengths": "",
        "improvements": "",
        "flagged": "",
        "verdict": "",
        "raw": feedback_text,
        "parsed_successfully": False,
    }

    if not feedback_text:
        return parsed

    try:
        # Score extraction
        score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback_text, re.IGNORECASE)
        if score_match:
            parsed["score"] = score_match.group(1)
            parsed["score_float"] = float(score_match.group(1))

        # Strengths
        strengths_match = re.search(
            r"Strengths:\s*\n(.*?)(?=\n(?:Areas to Improve|Flagged|One line verdict)|\Z)",
            feedback_text,
            re.DOTALL | re.IGNORECASE,
        )
        if strengths_match:
            parsed["strengths"] = strengths_match.group(1).strip()

        # Areas to Improve
        improvements_match = re.search(
            r"Areas to Improve:\s*\n(.*?)(?=\n(?:Flagged|One line verdict|Strengths)|\Z)",
            feedback_text,
            re.DOTALL | re.IGNORECASE,
        )
        if improvements_match:
            parsed["improvements"] = improvements_match.group(1).strip()

        # Flagged Stats
        flagged_match = re.search(
            r"Flagged\s*(?:/\s*Unsupported)?\s*Statistics:\s*\n(.*?)(?=\n(?:One line verdict)|\Z)",
            feedback_text,
            re.DOTALL | re.IGNORECASE,
        )
        if flagged_match:
            parsed["flagged"] = flagged_match.group(1).strip()

        # Verdict
        verdict_match = re.search(
            r"One line verdict:\s*\n?(.*)",
            feedback_text,
            re.DOTALL | re.IGNORECASE,
        )
        if verdict_match:
            parsed["verdict"] = verdict_match.group(1).strip()

        if parsed["score"] is not None and (parsed["strengths"] or parsed["verdict"]):
            parsed["parsed_successfully"] = True

    except Exception:
        parsed["parsed_successfully"] = False

    return parsed


# ==============================================================================
# 5. UI Render Components
# ==============================================================================

def render_stepper(stepper_data: dict[str, Any]):
    """Render horizontal 4-step pipeline status indicator."""
    stages = [
        ("search", "1. Search", stepper_data.get("search", {})),
        ("scrape", "2. Scrape", stepper_data.get("scrape", {})),
        ("write", "3. Write", stepper_data.get("write", {})),
        ("critic", "4. Critic", stepper_data.get("critic", {})),
    ]

    cards_html = []
    for key, title, info in stages:
        st_state = info.get("status", "pending")
        elapsed = info.get("time", 0.0)
        desc = info.get("desc", "")
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else ""

        card_html = f"""
        <div class="step-card {st_state}">
            <div class="step-header">
                <span class="step-number">{key}</span>
                <span class="step-time">{time_str}</span>
            </div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """
        cards_html.append(card_html)

    stepper_html = f"""
    <div class="stepper-container">
        {''.join(cards_html)}
    </div>
    """
    st.markdown(stepper_html, unsafe_allow_html=True)


def render_empty_state():
    """Render structured Linear-style workflow cards and interactive topic pills."""
    st.markdown(
        """
        <div class="empty-state-container">
            <div class="workflow-grid">
                <div class="workflow-card">
                    <span class="workflow-badge">STAGE 01</span>
                    <div class="workflow-title">Web Discovery</div>
                    <div class="workflow-text">Autonomous search agent queries Tavily for authoritative and recent sources.</div>
                </div>
                <div class="workflow-card">
                    <span class="workflow-badge">STAGE 02</span>
                    <div class="workflow-title">Deep Scraping</div>
                    <div class="workflow-text">Extracts full readable page text from top candidates, filtering irrelevant data.</div>
                </div>
                <div class="workflow-card">
                    <span class="workflow-badge">STAGE 03</span>
                    <div class="workflow-title">Synthesis</div>
                    <div class="workflow-text">Writer chain creates structured findings strictly backed by gathered research.</div>
                </div>
                <div class="workflow-card">
                    <span class="workflow-badge">STAGE 04</span>
                    <div class="workflow-title">Fact Verification</div>
                    <div class="workflow-text">Critic chain evaluates claims against raw sources and assigns an accuracy score.</div>
                </div>
            </div>
            <div class="presets-label">Suggested Research Topics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    sample_topics = [
        "The impact of AI on Job market in 2026",
        "Recent breakthroughs in Quantum Computing",
        "Commercialization status of Solid-State Batteries",
    ]

    with col1:
        if st.button(sample_topics[0], use_container_width=True):
            st.session_state.topic_input = sample_topics[0]
            st.rerun()
    with col2:
        if st.button(sample_topics[1], use_container_width=True):
            st.session_state.topic_input = sample_topics[1]
            st.rerun()
    with col3:
        if st.button(sample_topics[2], use_container_width=True):
            st.session_state.topic_input = sample_topics[2]
            st.rerun()


def render_report_tab(state: dict[str, Any]):
    """Render synthesized research report with centered 740px layout and download action."""
    report = state.get("report", "")
    topic = state.get("topic", "Research")

    if not report:
        st.info("No report generated yet.")
        return

    slug = re.sub(r"[^\w\s-]", "", topic).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)[:30]
    filename = f"research_report_{slug}.md"

    col_meta, col_dl = st.columns([3, 1])
    with col_meta:
        st.caption(f"Topic: **{topic}** &bull; Generated via LangChain & ChatGroq")
    with col_dl:
        st.download_button(
            label="Download .md",
            data=report,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div class="report-wrapper">
        {st.session_state.get('_report_html', '')}
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Render native markdown for standard styling
    st.markdown(report)


def render_sources_tab(state: dict[str, Any]):
    """Render clean source cards with domain badges and Google Favicons."""
    sources = state.get("sources", [])
    if not sources:
        st.info("No scraped sources available for this run.")
        return

    st.caption(f"Extracted {len(sources)} source references")

    for idx, src in enumerate(sources, 1):
        is_success = src.get("success", False)
        title = src.get("title", f"Source #{idx}")
        url = src.get("url", "#")
        char_count = src.get("chars", 0)
        content = src.get("content", "No content preview.")

        parsed_url = urlparse(url)
        domain = parsed_url.netloc or "web"
        favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"

        status_class = "success" if is_success else "failed"
        status_text = "Scraped" if is_success else "Failed"

        card_html = f"""
        <div class="source-card-v2">
            <div class="source-header-v2">
                <div class="source-site-info">
                    <img src="{favicon_url}" class="source-favicon" alt="Favicon" />
                    <span class="source-domain">{domain}</span>
                </div>
                <span class="source-status-pill {status_class}">{status_text}</span>
            </div>
            <div class="source-title-v2">{title}</div>
            <a href="{url}" target="_blank" rel="noopener noreferrer" class="source-url-v2">{url}</a>
            <div class="source-meta-v2">Extracted Length: {char_count:,} characters</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander(f"Preview extracted text ({domain})"):
            st.text(content[:2500] + ("..." if len(content) > 2500 else ""))


def render_critic_tab(state: dict[str, Any]):
    """Render critic review with score circle badge, verdict, and 3-column analysis."""
    feedback = state.get("feedback", "")
    if not feedback:
        st.info("No critic review available.")
        return

    parsed = parse_critic_feedback(feedback)

    if parsed["parsed_successfully"]:
        score_f = parsed["score_float"]
        if score_f >= 8.0:
            score_class = "high"
        elif score_f >= 5.0:
            score_class = "medium"
        else:
            score_class = "low"

        score_html = f"""
        <div class="critic-score-banner">
            <div class="score-circle {score_class}">
                <div class="score-number">{parsed['score']}</div>
                <div class="score-denom">OUT OF 10</div>
            </div>
            <div class="verdict-box">
                <div class="verdict-label">Executive Verdict</div>
                <div class="verdict-text">{parsed['verdict'] if parsed['verdict'] else 'Report meets research criteria.'}</div>
            </div>
        </div>
        """
        st.markdown(score_html, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="critic-card">
                    <div class="critic-card-title strengths">Strengths</div>
                    <div class="critic-card-content">{parsed['strengths'] if parsed['strengths'] else 'None reported.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="critic-card">
                    <div class="critic-card-title improvements">Areas to Improve</div>
                    <div class="critic-card-content">{parsed['improvements'] if parsed['improvements'] else 'None reported.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="critic-card">
                    <div class="critic-card-title flagged">Flagged Claims</div>
                    <div class="critic-card-content">{parsed['flagged'] if parsed['flagged'] else 'All facts supported.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(feedback)


def render_raw_tab(state: dict[str, Any]):
    """Render raw JSON/text output for debugging."""
    search_result = state.get("search_result", "No search output.")
    scraped_content = state.get("scraped_content", "No scraped content.")

    with st.expander("Search Agent Raw Context", expanded=False):
        st.text(search_result)

    with st.expander("Scraped Content Aggregation", expanded=False):
        st.text(scraped_content)


# ==============================================================================
# 6. Pipeline Execution Loop
# ==============================================================================

def execute_research_stream(topic: str, max_urls: int):
    """Execute streaming pipeline, updating live stepper and status."""
    st.session_state.is_running = True
    st.session_state.last_error = None

    stepper = {
        "search": {"status": "pending", "time": 0.0, "desc": "Tavily web discovery"},
        "scrape": {"status": "pending", "time": 0.0, "desc": "Content extraction"},
        "write": {"status": "pending", "time": 0.0, "desc": "Report synthesis"},
        "critic": {"status": "pending", "time": 0.0, "desc": "Fact check & review"},
    }
    st.session_state.stepper_state = stepper

    stepper_placeholder = st.empty()
    status_box = st.status("Executing Multi-Agent Research...", expanded=True)

    stage_starts: dict[str, float] = {}

    try:
        final_state: dict[str, Any] = {}

        for event in run_research_pipeline_stream(topic=topic, max_urls=max_urls):
            stage = event.get("stage", "")
            status = event.get("status", "")
            data = event.get("data", "")

            if status == "running":
                stage_starts[stage] = time.perf_counter()
                if stage in stepper:
                    stepper[stage]["status"] = "running"
                    stepper[stage]["desc"] = str(data)[:45] + "..." if len(str(data)) > 45 else str(data)

                status_box.update(label=f"Active Stage: {stage.capitalize()}...", state="running")
                status_box.write(f"&bull; **{stage.capitalize()}**: {data}")

            elif status == "complete":
                elapsed = time.perf_counter() - stage_starts.get(stage, time.perf_counter())
                if stage in stepper:
                    stepper[stage]["status"] = "complete"
                    stepper[stage]["time"] = elapsed

                if stage != "done":
                    status_box.write(f"&bull; Finished {stage.capitalize()} in {elapsed:.1f}s")
                else:
                    final_state = data

            elif status == "error":
                if stage in stepper:
                    stepper[stage]["status"] = "error"
                    stepper[stage]["desc"] = "Failed"

                status_box.update(label=f"Pipeline Error in {stage.capitalize()}", state="error")
                status_box.write(f"⚠️ {data}")
                st.session_state.last_error = str(data)
                break

            # Update stepper UI
            with stepper_placeholder.container():
                render_stepper(stepper)

        if final_state:
            status_box.update(label="Research Complete", state="complete", expanded=False)
            st.session_state.current_result = final_state
            st.session_state.history.append({
                "topic": topic,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "state": final_state,
            })
            st.rerun()

    except groq.RateLimitError as rle:
        status_box.update(label="Groq Rate Limit", state="error", expanded=True)
        headers = getattr(rle.response, "headers", {}) if getattr(rle, "response", None) else {}
        retry_after = headers.get("retry-after") or "60"
        st.warning(f"Groq API rate limit reached. Retry-After: {retry_after}s.")
        if st.button("Retry Research"):
            st.rerun()

    except Exception as exc:
        status_box.update(label="Error Occurred", state="error", expanded=True)
        st.error(f"Execution Error: {exc}")
        st.session_state.last_error = str(exc)

    finally:
        st.session_state.is_running = False


# ==============================================================================
# 7. Main Application UI Layout
# ==============================================================================

def main():
    if not verify_api_keys():
        st.stop()

    resources = get_cached_pipeline_resources()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Multi-Agent System</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="sidebar-model-badge">{resources["model_name"]}</span>', unsafe_allow_html=True)

        st.subheader("Options")
        max_urls = st.slider("Max Scraping Sources", min_value=1, max_value=5, value=3)

        if st.button("Clear Canvas", use_container_width=True):
            st.session_state.current_result = None
            st.session_state.last_error = None
            st.session_state.topic_input = ""
            st.rerun()

        # Run History
        if st.session_state.history:
            st.markdown("---")
            st.subheader("History")
            for idx, item in enumerate(reversed(st.session_state.history)):
                topic_txt = item["topic"]
                label_txt = (topic_txt[:22] + "...") if len(topic_txt) > 22 else topic_txt
                if st.button(f"{item['timestamp']} · {label_txt}", key=f"hist_btn_{idx}", use_container_width=True):
                    st.session_state.current_result = item["state"]
                    st.session_state.topic_input = topic_txt
                    st.rerun()

    # --- Hero Header ---
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">Autonomous Intelligence</div>
            <h1 class="hero-title">Deep Research Assistant</h1>
            <div class="hero-tagline">Multi-agent web search, content scraping, analytical synthesis, and automated critique.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Search Input & Execution Trigger ---
    col_input, col_action = st.columns([5, 1])
    with col_input:
        user_topic = st.text_input(
            "Research Query",
            value=st.session_state.topic_input,
            placeholder="Enter research topic (e.g. Next-generation lithium iron phosphate batteries)",
            label_visibility="collapsed",
            disabled=st.session_state.is_running,
        )

    with col_action:
        run_btn = st.button(
            "Research",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running,
        )

    # Execute on button trigger
    if run_btn:
        cleaned_topic = user_topic.strip()
        if len(cleaned_topic) < 4:
            st.error("Please provide a more descriptive topic query (at least 4 characters).")
        else:
            st.session_state.topic_input = cleaned_topic
            execute_research_stream(topic=cleaned_topic, max_urls=max_urls)

    # --- Content Display ---
    if st.session_state.current_result:
        st.markdown("---")
        tab_report, tab_sources, tab_critic, tab_raw = st.tabs([
            "Report",
            "Sources",
            "Critic Review",
            "Raw Context",
        ])

        with tab_report:
            render_report_tab(st.session_state.current_result)

        with tab_sources:
            render_sources_tab(st.session_state.current_result)

        with tab_critic:
            render_critic_tab(st.session_state.current_result)

        with tab_raw:
            render_raw_tab(st.session_state.current_result)

    elif not st.session_state.is_running:
        render_empty_state()


if __name__ == "__main__":
    main()
