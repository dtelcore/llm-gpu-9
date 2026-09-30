"""Wikipedia OpenSearch + REST summary via stdlib urllib. No extra deps."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from version import __version__

OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = (
    f"Apple-MLX-llm-gpu-8/{__version__} "
    "(local cabinet router; https://github.com/dtelcore/Apple-MLX-llm-gpu-8)"
)
DEFAULT_TIMEOUT_S = 8.0


OPENSEARCH_LIMIT = 5
_ORDINAL_PREFIX = re.compile(
    r"^(?:the\s+)?(?:first|last|top|next)\s+\d+\s+",
    re.IGNORECASE,
)
_LIST_TITLE_PREFIXES = (
    "list of ",
    "lists of ",
    "communes of ",
    "outline of ",
    "index of ",
    "timeline of ",
    "glossary of ",
)
_LIST_QUERY_MARKERS = ("list", "communes", "outline", "timeline", "glossary")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[object]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _opensearch_titles(query: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> list:
    params = urllib.parse.urlencode(
        {
            "action": "opensearch",
            "search": query,
            "limit": str(OPENSEARCH_LIMIT),
            "namespace": "0",
            "format": "json",
        }
    )
    data = _get_json(f"{OPENSEARCH_URL}?{params}", timeout=timeout)
    if not isinstance(data, list) or len(data) < 2:
        return []
    titles = data[1]
    if not isinstance(titles, list):
        return []
    out = []
    for title in titles:
        if isinstance(title, str) and title.strip():
            out.append(title.strip())
    return out


def _is_stub_extract(text: str) -> bool:
    t = text.casefold().rstrip(" .")
    return t.endswith("may refer to")


def _summary_extract(title: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
    quoted = urllib.parse.quote(title.replace(" ", "_"), safe="")
    data = _get_json(SUMMARY_URL + quoted, timeout=timeout)
    if not isinstance(data, dict):
        return None
    if data.get("type") == "disambiguation":
        return None
    extract = data.get("extract") or data.get("description")
    if not isinstance(extract, str):
        return None
    text = " ".join(extract.split())
    if not text or _is_stub_extract(text):
        return None
    return text


def _query_variants(query: str) -> list:
    """Original phrase, then 'first 5 prime numbers' → 'prime numbers'."""
    q = " ".join((query or "").split())
    if not q:
        return []
    out = [q]
    stripped = _ORDINAL_PREFIX.sub("", q).strip()
    if stripped and stripped.casefold() != q.casefold():
        out.append(stripped)
    return out


def _srsearch_titles(query: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> list:
    """Full-text search when OpenSearch (autocomplete) returns nothing."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(OPENSEARCH_LIMIT),
            "srnamespace": "0",
            "format": "json",
        }
    )
    data = _get_json(f"{OPENSEARCH_URL}?{params}", timeout=timeout)
    if not isinstance(data, dict):
        return []
    hits = (data.get("query") or {}).get("search")
    if not isinstance(hits, list):
        return []
    out = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        title = hit.get("title")
        if isinstance(title, str) and title.strip():
            out.append(title.strip())
    return out


def _wants_list_page(query: str) -> bool:
    return any(m in query.casefold() for m in _LIST_QUERY_MARKERS)


def _is_list_title(title: str) -> bool:
    return title.casefold().startswith(_LIST_TITLE_PREFIXES)


def _clip_extract(text: str, max_sentences: int = 2) -> str:
    parts = [p for p in _SENTENCE_SPLIT.split((text or "").strip()) if p]
    return " ".join(parts[:max_sentences]) if parts else ""


def _format_hit(title: str, extract: str) -> str:
    short = _clip_extract(extract)
    if not short:
        return ""
    head = short[:96].casefold()
    if title and title.casefold() not in head:
        return f"{title} — {short}"
    return short


def _extract_from_titles(
    titles,
    *,
    seen: set,
    timeout: float,
    allow_lists: bool,
) -> Optional[str]:
    for title in titles:
        if title in seen:
            continue
        seen.add(title)
        if not allow_lists and _is_list_title(title):
            continue
        extract = _summary_extract(title, timeout=timeout)
        if extract:
            return _format_hit(title, extract)
    return None


def wiki_summary(query: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
    """Return a short Wikipedia extract, or None on any failure/empty result."""
    variants = _query_variants(query)
    if not variants:
        return None
    seen: set = set()
    allow_lists = _wants_list_page(variants[0])
    for q in variants:
        extract = _extract_from_titles(
            _opensearch_titles(q, timeout=timeout),
            seen=seen,
            timeout=timeout,
            allow_lists=allow_lists,
        )
        if extract:
            return extract
    return _extract_from_titles(
        _srsearch_titles(variants[0], timeout=timeout),
        seen=seen,
        timeout=timeout,
        allow_lists=allow_lists,
    )
