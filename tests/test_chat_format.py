"""Tests for chat formatting, story-chat wrapping, stop-string trim, and chat quality."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.make_story_chat import make_story_chat_lines, wrap_story
from training.chat_format import (
    ASSISTANT_PREFIX,
    USER_PREFIX,
    USER_ROLE,
    build_chat_prompt_ids,
    format_conversation,
    sanitize_assistant_reply,
    trim_generated_stop_strings,
)
from training.quality import (
    score_generation,
    score_instruction_follow,
    score_turn_format,
)


class _CharTok:
    """Identity tokenizer: ids are character ordinals."""

    def encode(self, text: str):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class ChatFormatTests(unittest.TestCase):
    def test_single_line_role_markers(self):
        text = format_conversation(
            [],
            pending_user="Tell me a story about a cat.",
            open_assistant=False,
        )
        self.assertTrue(text.startswith(USER_PREFIX))
        self.assertNotIn("\n", text)
        opened = format_conversation(
            [(USER_ROLE, "hi")],
            open_assistant=True,
        )
        self.assertTrue(opened.endswith(ASSISTANT_PREFIX.rstrip()))

    def test_front_truncate_drops_oldest_turns(self):
        tok = _CharTok()
        turns = [
            (USER_ROLE, "aaaaaaaaaa"),
            ("assistant", "bbbbbbbbbb"),
            (USER_ROLE, "cccccccccc"),
            ("assistant", "dddddddddd"),
        ]
        ids, text = build_chat_prompt_ids(
            tok, turns, "now", max_prompt_tokens=40,
        )
        decoded = tok.decode(ids)
        self.assertLessEqual(len(ids), 40)
        self.assertIn("now", decoded)
        self.assertNotIn("aaaaaaaaaa", text)

    def test_preserves_system_and_complete_turns(self):
        tok = _CharTok()
        system = "Be kind."
        turns = [
            (USER_ROLE, "aaaaaaaaaa"),
            ("assistant", "bbbbbbbbbb"),
            (USER_ROLE, "cccccccccc"),
            ("assistant", "dddddddddd"),
        ]
        ids, text = build_chat_prompt_ids(
            tok, turns, "now", system=system, max_prompt_tokens=50,
        )
        self.assertTrue(text.startswith(system))
        rest = text[len(system):].lstrip()
        self.assertTrue(rest.startswith(USER_PREFIX))
        self.assertTrue(text.endswith(ASSISTANT_PREFIX.rstrip()))
        self.assertNotIn("aaaaaaaaaa", text)
        self.assertEqual(ids, tok.encode(text))

    def test_overflow_shrinks_user_not_tail_slice(self):
        tok = _CharTok()
        system = "Be kind always."
        ids, text = build_chat_prompt_ids(
            tok, [], "alpha beta gamma delta epsilon zeta",
            system=system, max_prompt_tokens=40,
        )
        self.assertTrue(text.startswith(system) or text.startswith(USER_PREFIX))
        self.assertTrue(text.endswith(ASSISTANT_PREFIX.rstrip()))
        self.assertEqual(ids, tok.encode(text))
        self.assertLessEqual(len(ids), 40)
        if text.startswith(system):
            rest = text[len(system):].lstrip()
            self.assertTrue(rest.startswith(USER_PREFIX))


class StoryChatToolTests(unittest.TestCase):
    def test_wrap_story_is_one_line(self):
        line = wrap_story("Once upon a time there was a kind cat.")
        self.assertNotIn("\n", line)
        self.assertIn("User:", line)
        self.assertIn("Assistant:", line)
        self.assertIn("kind cat", line)

    def test_keep_raw_fraction(self):
        stories = ["Once upon a time a cat sat."] * 50
        mixed = make_story_chat_lines(stories, keep_raw=1.0, seed=0)
        self.assertTrue(all("User:" not in line for line in mixed))
        chats = make_story_chat_lines(stories, keep_raw=0.0, seed=0)
        self.assertTrue(all(line.startswith("User:") for line in chats))


class StopStringTests(unittest.TestCase):
    def test_trims_user_marker(self):
        tok = _CharTok()
        prompt = "Assistant:"
        gen = " Once upon a time User: extra"
        ids = tok.encode(prompt + gen)
        trimmed, hit = trim_generated_stop_strings(
            ids, prompt_len=len(tok.encode(prompt)), tokenizer=tok,
            stop_strings=[" User:", "User:"],
        )
        self.assertTrue(hit)
        out = tok.decode(trimmed)
        self.assertNotIn("User:", out)
        self.assertTrue(out.startswith(prompt))

    def test_noop_without_stop(self):
        tok = _CharTok()
        ids = tok.encode("hello")
        out, hit = trim_generated_stop_strings(ids, 0, tok, None)
        self.assertFalse(hit)
        self.assertEqual(out, ids)


class SanitizeReplyTests(unittest.TestCase):
    def test_strips_user_leak_and_partials(self):
        self.assertEqual(
            sanitize_assistant_reply("Once upon a time User: extra"),
            "Once upon a time",
        )
        self.assertEqual(sanitize_assistant_reply("Hello there User"), "Hello there")
        self.assertEqual(sanitize_assistant_reply("Hello there Use"), "Hello there")
        self.assertEqual(sanitize_assistant_reply("Hello there U"), "Hello there")
        self.assertEqual(sanitize_assistant_reply("  hi  \n  "), "hi")

    def test_does_not_eat_you(self):
        self.assertEqual(sanitize_assistant_reply("I like YOU"), "I like YOU")


class ChatQualityTests(unittest.TestCase):
    def test_story_mode_omits_chat_fields(self):
        scores = score_generation("Once upon a time there was a cat.", prompt="once upon a")
        d = scores.as_dict()
        self.assertIn("spelling", d)
        self.assertNotIn("turn_format", d)
        self.assertNotIn("instruction_follow", d)

    def test_turn_format_penalizes_extra_user(self):
        prompt = "User: Hi. Assistant:"
        good = prompt + " Hello there, friend!"
        bad = prompt + " Hello User: wait"
        self.assertGreater(score_turn_format(good, prompt), score_turn_format(bad, prompt))

    def test_yes_no_instruction(self):
        prompt = "User: Can cats fly? Answer yes or no. Assistant:"
        yes = prompt + " No."
        ramble = prompt + " Once upon a time the clouds were dancing forever."
        self.assertGreater(
            score_instruction_follow(yes, prompt),
            score_instruction_follow(ramble, prompt),
        )

    def test_chat_mode_includes_new_fields(self):
        prompt = "User: Tell me a short story about a cat. Assistant:"
        text = prompt + " Once upon a time a cat sat on a mat."
        scores = score_generation(text, prompt=prompt, mode="chat")
        d = scores.as_dict()
        self.assertIn("turn_format", d)
        self.assertIn("instruction_follow", d)
        self.assertIn("coherence", d)
        self.assertGreater(d["aggregate"], 0.0)


if __name__ == "__main__":
    unittest.main()
