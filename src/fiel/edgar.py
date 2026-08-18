"""Minimal, cached EDGAR fetcher.

Deliberately small: the point of Fiel is judging grounding, not building a data
pipeline. This does exactly three things — fetch (with SEC's required User-Agent),
cache to disk so we never re-hit EDGAR, and strip a filing's HTML to clean text.
"""

from __future__ import annotations

import hashlib
import re
import time
import warnings
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC filings embed inline XBRL; parsing as HTML is fine for text extraction.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# SEC requires a descriptive User-Agent with a contact. Fair-access policy: <=10 req/s.
USER_AGENT = "Fiel eval harness - Humberto Mangino hmangino@gmail.com"
CACHE_DIR = Path("data/sec/.cache")
_MIN_INTERVAL_S = 0.2  # be polite; well under SEC's 10 req/s ceiling
_last_request_t = 0.0


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    suffix = ".json" if url.endswith(".json") else ".txt"
    return CACHE_DIR / f"{key}{suffix}"


def fetch(url: str, *, use_cache: bool = True) -> str:
    """GET a URL with the SEC User-Agent, caching the raw response body to disk."""
    cache_file = _cache_path(url)
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    global _last_request_t
    elapsed = time.monotonic() - _last_request_t
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)

    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)
    _last_request_t = time.monotonic()
    resp.raise_for_status()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def html_to_text(html: str) -> str:
    """Strip an SEC filing's HTML to readable text, collapsing whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\u00a0", " ", text)  # non-breaking spaces are rampant in filings
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()
