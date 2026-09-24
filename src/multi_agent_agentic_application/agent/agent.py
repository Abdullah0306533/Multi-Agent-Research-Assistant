from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from multi_agent_agentic_application.tool.scrap_url import scrape_url
from multi_agent_agentic_application.tool.web_search import web_search
from dotenv import load_dotenv
import os

load_dotenv()

# Model Initialization
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# 1st Agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a web search assistant. Report only what the search tool returned. "
            "Never add statistics, facts, or numbers from memory. "
            "Keep the output short, concise, and faithful to the search results."
        ),
    )


# 2nd Agent:Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a reading and extraction agent. Only call scrape_url on URLs that were explicitly provided to you. "
            "If no URLs are provided, state that clearly and stop."
        ),
    )


writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured, and factual reports based strictly on provided research. Do not use placeholders and never invent facts or sources."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Rules:
- Use ONLY the provided research above. Do NOT introduce external knowledge or facts from memory.
- Do NOT use placeholders (e.g., [Client / Stakeholder], [Date], [Insert Name]).
- In the Sources section, list ONLY the URLs that were actually scraped and present in the research.
- Mark any claim not directly supported by the research explicitly as [Unsupported Claim].

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points supported by the research)
- Conclusion
- Sources (list only URLs that were actually scraped and present in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest, rigorous, and specific."),
    ("human", """Review the research report below against the raw research gathered and evaluate it strictly.

Research Gathered:
{research}

Report:
{report}

Evaluation Rules:
- Cross-reference all statistics, numbers, and factual claims in the report against the raw research.
- Explicitly flag any statistic, metric, or claim in the report that does NOT appear in the provided research.

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

Flagged / Unsupported Statistics:
- ... (or 'None' if all statistics are verified against the raw research)

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()