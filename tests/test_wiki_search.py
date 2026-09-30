"""Wikipedia helper: mocked urllib only."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import wiki_search


def _http_json(payload) -> mock.MagicMock:
    raw = json.dumps(payload).encode("utf-8")
    ctx = mock.MagicMock()
    ctx.read.return_value = raw
    ctx.__enter__.return_value = ctx
    ctx.__exit__.return_value = False
    return ctx


class WikiSearchTests(unittest.TestCase):
    def test_success_extract(self):
        open_search = ["unobtanium", ["Unobtainium"], ["desc"], ["http://example"]]
        summary = {"title": "Unobtainium", "extract": "A rare fictional metal."}

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(open_search)
            return _http_json(summary)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("What is unobtanium")
        self.assertEqual(text, "Unobtainium — A rare fictional metal.")

    def test_network_error_returns_none(self):
        with mock.patch(
            "tools.wiki_search.urllib.request.urlopen",
            side_effect=URLError("offline"),
        ):
            self.assertIsNone(wiki_search.wiki_summary("What is unobtanium"))

    def test_skips_disambiguation_for_next_title(self):
        open_search = ["python", ["Python", "Python (programming language)"], ["", ""], ["", ""]]
        dab = {"title": "Python", "type": "disambiguation", "extract": "Python may refer to:"}
        real = {
            "title": "Python (programming language)",
            "type": "standard",
            "extract": "Python is a high-level programming language.",
        }

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(open_search)
            if "Python_%28programming_language%29" in url:
                return _http_json(real)
            return _http_json(dab)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("python")
        self.assertEqual(
            text,
            "Python (programming language) — Python is a high-level programming language.",
        )

    def test_empty_extract_returns_none(self):
        open_search = ["x", ["X"], [""], ["http://example"]]
        summary = {"title": "X", "extract": "   "}

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(open_search)
            return _http_json(summary)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertIsNone(wiki_search.wiki_summary("X"))

    def test_ordinal_phrase_retries_stripped_topic(self):
        empty = ["first 5 prime numbers", [], [], []]
        primes = ["prime numbers", ["Prime number"], [""], [""]]
        summary = {
            "title": "Prime number",
            "type": "standard",
            "extract": "A prime number is a natural number greater than 1.",
        }

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                if "prime+numbers" in url and "first" not in url:
                    return _http_json(primes)
                return _http_json(empty)
            return _http_json(summary)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("first 5 prime numbers")
        self.assertEqual(text, "A prime number is a natural number greater than 1.")

    def test_fulltext_search_when_opensearch_empty(self):
        empty = ["widget flux", [], [], []]
        sr = {"query": {"search": [{"title": "Widget"}]}}
        summary = {"title": "Widget", "type": "standard", "extract": "A widget is a placeholder."}

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(empty)
            if "list=search" in url:
                return _http_json(sr)
            return _http_json(summary)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("widget flux")
        self.assertEqual(text, "A widget is a placeholder.")

    def test_skips_list_title_unless_query_asks_for_a_list(self):
        empty = ["largest city of france", [], [], []]
        sr = {
            "query": {
                "search": [
                    {"title": "List of communes in France with over 20,000 inhabitants"},
                    {"title": "Paris"},
                ]
            }
        }
        communes = {
            "title": "List of communes in France with over 20,000 inhabitants",
            "type": "standard",
            "extract": "As of January 2023, there were 482 communes in France.",
        }
        paris = {
            "title": "Paris",
            "type": "standard",
            "extract": "Paris is the capital and largest city of France. It sits on the Seine.",
        }

        def fake_urlopen(req, timeout=8.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "opensearch" in url:
                return _http_json(empty)
            if "list=search" in url:
                return _http_json(sr)
            if "List_of_communes" in url:
                return _http_json(communes)
            return _http_json(paris)

        with mock.patch("tools.wiki_search.urllib.request.urlopen", side_effect=fake_urlopen):
            text = wiki_search.wiki_summary("largest city of france")
        self.assertEqual(
            text,
            "Paris is the capital and largest city of France. It sits on the Seine.",
        )
