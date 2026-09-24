from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

import os

load_dotenv()


@tool()
def web_search(query: str):
    """" Search the web for recent, reliable, and relevant information about the given topic. Return the results in a structured format containing:
- Title
- URL
- Brief description

Prioritize authoritative and trustworthy sources, and prefer the most recent information available.
    """
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(query=query, max_results=5)
    out = []
    for r in results['results']:
        out.append(f"Title:{r["title"]}\nurl:{r['url']}\nsnippet:{r['content'][:300]}\n")
    return "\n-----------\n".join(out)


