import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from readability import Document
from langchain.tools import tool

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MIN_CONTENT_LENGTH = 200  # below this, we consider extraction a failure


def _clean_text(text: str) -> str:
    """Collapse excess whitespace/blank lines and strip junk."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_with_trafilatura(html: str, url: str) -> str | None:
    content = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    return _clean_text(content) if content and len(content) >= MIN_CONTENT_LENGTH else None


def _extract_with_readability(html: str) -> str | None:
    try:
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator="\n")
        return _clean_text(text) if len(text) >= MIN_CONTENT_LENGTH else None
    except Exception:
        return None


def _extract_with_bs4(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return _clean_text(text) if len(text) >= MIN_CONTENT_LENGTH else None


def _get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    return "Untitled"


@tool()
def scrape_url(url: str, max_chars: int = 8000) -> str:
    """Scrape and extract the main readable content from a given URL.

    Fetches the page, strips ads/navigation/boilerplate, and returns clean
    article text using a cascading extraction strategy (trafilatura ->
    readability -> BeautifulSoup) for maximum reliability across different
    site structures. Use this to read the full content of a page found via
    web search.

    Args:
        url: The webpage URL to scrape.
        max_chars: Maximum number of characters to return (default 8000).

    Returns:
        A structured string containing the page Title, URL, and extracted
        Content, or an error message if scraping failed.
    """
    if not url or not url.startswith(("http://", "https://")):
        return f"Error: '{url}' is not a valid URL."

    try:
        response = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return f"Error: Request to {url} timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {response.status_code} while fetching {url} ({e})"
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch {url} ({e})"

    html = response.text
    if not html or len(html) < 50:
        return f"Error: {url} returned empty or near-empty content."

    title = _get_title(html)

    content = (
        _extract_with_trafilatura(html, url)
        or _extract_with_readability(html)
        or _extract_with_bs4(html)
    )

    if not content:
        return f"Error: Could not extract meaningful content from {url}. The page may be JS-rendered, paywalled, or blocked scraping."

    if len(content) > max_chars:
        content = content[:max_chars].rsplit(" ", 1)[0] + "..."

    return f"Title: {title}\nURL: {url}\n\nContent:\n{content}"

