"""Shared chat session for the REPL and the Flask UI.

One checkpoint, one process. Router: cabinet → calc → Wikipedia → miss.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import cli_common
from logging_config import logger, setup_generate_run_logging
from model.gpt import GPTModel
from paths import DATA_DIR, OUTPUT_ROOT, checkpoint_weights_relpath, ensure_output_dirs
from training.cabinet_diagnostics import (
    answers_match,
    append_jsonl,
    build_turn_record,
    classify_generation,
)
from training.router import (
    MISS_HINT,
    RouteDecision,
    alias_learned_topics,
    alias_trained_topics,
    remember_search_hit,
    route,
    router_enabled,
    search_topic,
    try_calc_query,
)
from tools.wiki_search import wiki_summary
from training.cabinet_index import CabinetIndex, load_cabinet, merge_cabinet
from training.checkpoint import load_checkpoint
from training.chat_format import (
    ASSISTANT_ROLE,
    CHAT_STOP_STRINGS,
    DEFAULT_CHAT_SYSTEM,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_K,
    DEFAULT_CHAT_TOP_P,
    USER_ROLE,
    build_chat_prompt_ids,
    is_chat_model_name,
    sanitize_assistant_reply,
)

CABINET_GENERATE_TEMP = 0.2
CABINET_GENERATE_TOP_K = 10
DEFAULT_FACTS = DATA_DIR / "chat_facts.jsonl"
DEFAULT_LEARNED = OUTPUT_ROOT / "cabinet_learned.jsonl"
DEFAULT_RETRAIN = OUTPUT_ROOT / "cabinet_retrain.jsonl"
DEFAULT_DIAGNOSTICS = OUTPUT_ROOT / "cabinet_diagnostics.jsonl"


@dataclass
class TurnResult:
    text: str
    kind: str = "generate"
    detail: str = ""
    quit: bool = False
    learned_added: int = 0
    related: List[str] = field(default_factory=list)
    classification: str = ""
    match_type: str = ""
    canonical: str = ""


@dataclass
class ChatSession:
    model: GPTModel
    tokenizer: object
    gpt_config: object
    args: argparse.Namespace
    chat_mode: bool
    router_on: bool
    search_enabled: bool
    learned_path: str
    cabinet: Optional[CabinetIndex]
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    system: Optional[str]
    stop_strings: Optional[List[str]]
    max_new_tokens: int
    tracer: object
    rng: object
    use_kv_cache: bool
    use_cuda_graph: bool
    history: List[Tuple[str, str]] = field(default_factory=list)
    last_route: Optional[RouteDecision] = None
    last_entities: List[str] = field(default_factory=list)
    last_related: List[str] = field(default_factory=list)
    trace_enabled: bool = False
    retrain_path: str = ""
    diagnostics_path: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, configure_logging: bool = True) -> "ChatSession":
        ensure_output_dirs()
        if configure_logging:
            log_path = setup_generate_run_logging(args.checkpoint)
            source = getattr(args, "_entry", None) or "interactive"
            logger.info(
                "interactive generation | source=%s | checkpoint=%s | log=%s",
                source, args.checkpoint, log_path,
            )
        else:
            log_path = None

        gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
        model = GPTModel(gpt_config, params)
        tracer = cli_common.build_tracer(args, default_trace_every=1)
        rng = np.random.default_rng(args.seed)
        model_name = getattr(gpt_config, "name", "") or ""
        chat_mode = _resolve_chat_mode(args, model_name)
        router_on = router_enabled(getattr(args, "router", None), model_name)
        search_enabled = router_on and not bool(getattr(args, "no_search", False))
        learned_path = str(getattr(args, "learned", DEFAULT_LEARNED))
        cabinet = None
        if router_on:
            cabinet = load_index(str(getattr(args, "facts", DEFAULT_FACTS)), learned_path)

        if chat_mode:
            temperature = args.temperature if args.temperature is not None else DEFAULT_CHAT_TEMPERATURE
            top_k = args.top_k if args.top_k is not None else DEFAULT_CHAT_TOP_K
            top_p = args.top_p if args.top_p is not None else DEFAULT_CHAT_TOP_P
            system = args.system if args.system is not None else DEFAULT_CHAT_SYSTEM
            stop_strings = list(args.stop) if args.stop else list(CHAT_STOP_STRINGS)
        else:
            temperature = args.temperature if args.temperature is not None else 0.8
            top_k = getattr(args, "top_k", None)
            top_p = getattr(args, "top_p", None)
            system = args.system
            stop_strings = list(args.stop) if args.stop else None

        session = cls(
            model=model,
            tokenizer=tokenizer,
            gpt_config=gpt_config,
            args=args,
            chat_mode=chat_mode,
            router_on=router_on,
            search_enabled=search_enabled,
            learned_path=learned_path,
            cabinet=cabinet,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            system=system,
            stop_strings=stop_strings,
            max_new_tokens=int(args.max_new_tokens),
            tracer=tracer,
            rng=rng,
            use_kv_cache=not getattr(args, "no_kv_cache", False),
            use_cuda_graph=bool(getattr(args, "cuda_graph", False)),
            trace_enabled=bool(tracer.any_enabled),
            retrain_path=str(getattr(args, "cabinet_retrain_log", DEFAULT_RETRAIN) or ""),
            diagnostics_path=str(getattr(args, "cabinet_diagnostics_log", DEFAULT_DIAGNOSTICS) or ""),
        )
        session.log_path = log_path
        return session

    def status(self) -> dict:
        n_cab = len(self.cabinet) if self.cabinet is not None else 0
        ckpt = str(self.args.checkpoint)
        return {
            "checkpoint": ckpt,
            "weights": checkpoint_weights_relpath(ckpt),
            "model": getattr(self.gpt_config, "name", "") or "",
            "vocab_size": int(getattr(self.gpt_config, "vocab_size", 0) or 0),
            "max_len": int(getattr(self.gpt_config, "max_len", 0) or 0),
            "chat": self.chat_mode,
            "router": self.router_on,
            "search": self.search_enabled,
            "cabinet": n_cab,
            "system": self.system,
            "learned": self.learned_path,
        }

    def clear(self) -> None:
        self.history = []
        self.last_entities = []
        self.last_related = []
        self.last_route = None

    def turn(self, prompt: str) -> TurnResult:
        with self._lock:
            return self._turn_unlocked(prompt)

    def _turn_unlocked(self, prompt: str) -> TurnResult:
        text = (prompt or "").strip()
        if not text:
            return TurnResult(text="", kind="empty")
        cmd = self._command(text)
        if cmd is not None:
            return cmd
        if self.router_on:
            routed = self._router_turn(text)
            if routed is not None:
                return routed
        return self._generate_turn(text)

    def _command(self, prompt: str) -> Optional[TurnResult]:
        if prompt in (":quit", ":exit"):
            return TurnResult(text="", kind="command", detail="quit", quit=True)
        if prompt.startswith(":temp "):
            self.temperature = float(prompt.split(maxsplit=1)[1])
            return TurnResult(text=f"[temperature -> {self.temperature}]", kind="command", detail="temp")
        if prompt.startswith(":tokens "):
            self.max_new_tokens = int(prompt.split(maxsplit=1)[1])
            return TurnResult(text=f"[max_new_tokens -> {self.max_new_tokens}]", kind="command", detail="tokens")
        if prompt.startswith(":topk "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            self.top_k = None if raw in ("none", "off", "0") else int(raw)
            return TurnResult(text=f"[top_k -> {self.top_k}]", kind="command", detail="topk")
        if prompt.startswith(":topp "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            self.top_p = None if raw in ("none", "off", "1", "1.0") else float(raw)
            return TurnResult(text=f"[top_p -> {self.top_p}]", kind="command", detail="topp")
        if prompt.startswith(":trace "):
            self.trace_enabled = prompt.split(maxsplit=1)[1].strip().lower() == "on"
            return TurnResult(
                text=f"[tracing -> {'on' if self.trace_enabled else 'off'}]",
                kind="command",
                detail="trace",
            )
        if self.chat_mode and prompt == ":clear":
            self.clear()
            return TurnResult(text="[history cleared]", kind="command", detail="clear")
        if self.chat_mode and prompt.startswith(":system"):
            rest = prompt[len(":system"):].strip()
            self.system = rest or None
            return TurnResult(text=f"[system -> {self.system!r}]", kind="command", detail="system")
        if self.router_on and prompt == ":route":
            if self.last_route is None:
                msg = "[route] none yet"
            else:
                msg = (
                    f"[route] kind={self.last_route.kind} detail={self.last_route.detail}"
                    f" match={self.last_route.match_type or 'none'}"
                )
            return TurnResult(text=msg, kind="command", detail="route")
        if self.router_on and prompt == ":related":
            if not self.last_related:
                return TurnResult(text="[related] none", kind="command", detail="related")
            lines = "\n".join(f"- {q}" for q in self.last_related)
            return TurnResult(
                text=f"[related]\n{lines}",
                kind="command",
                detail="related",
                related=list(self.last_related),
            )
        if self.router_on and prompt.startswith(":search"):
            raw_q = prompt[len(":search"):].strip()
            query = search_topic(raw_q) or raw_q
            extract = wiki_summary(query) if self.search_enabled and query else None
            if extract:
                self.last_route = RouteDecision(kind="search", text=extract, detail="wikipedia_forced")
                added = self._save_search(raw_q or query, extract)
                self._remember_history(query or prompt, extract)
                return TurnResult(text=extract, kind="search", detail="wikipedia_forced", learned_added=added)
            self.last_route = RouteDecision(kind="miss", text=MISS_HINT, detail="search_failed")
            self._remember_history(query or prompt, MISS_HINT)
            return TurnResult(text=MISS_HINT, kind="miss", detail="search_failed")
        if self.router_on and prompt.startswith(":calc"):
            expr = prompt[len(":calc"):].strip()
            value = try_calc_query(expr) if expr else None
            if value is not None:
                self.last_route = RouteDecision(kind="calc", text=value, detail="calc_forced")
                self._remember_history(expr, value)
                return TurnResult(text=value, kind="calc", detail="calc_forced")
            self.last_route = RouteDecision(kind="miss", text=MISS_HINT, detail="calc_failed")
            return TurnResult(text="[calc] not a safe arithmetic expression", kind="command", detail="calc_failed")
        return None

    def _router_turn(self, prompt: str) -> Optional[TurnResult]:
        decision = route(
            prompt,
            self.cabinet,
            search_enabled=self.search_enabled,
            search_fn=wiki_summary if self.search_enabled else None,
            last_entities=self.last_entities,
        )
        self.last_route = decision
        related = list(decision.related)
        self.last_related = related
        logger.info("route kind=%s detail=%s prompt=%r", decision.kind, decision.detail, prompt)
        if decision.kind == "cabinet" and decision.fact is not None:
            if decision.fact.source == "learned":
                self.last_entities = []
                self._remember_history(prompt, decision.fact.assistant)
                self._emit_diagnostics(prompt, decision, decision.fact.assistant, "replay", "LEARNED_REPLAY")
                return TurnResult(
                    text=decision.fact.assistant,
                    kind="cabinet",
                    detail=decision.detail,
                    related=related,
                    match_type=decision.match_type,
                    canonical=decision.canonical,
                    classification="LEARNED_REPLAY",
                )
            prompt_text = decision.fact.generate_prompt
            prompt_ids = self.tokenizer.encode(prompt_text)
            if not prompt_ids:
                return TurnResult(text="[No recognized characters in stored cabinet prompt]", kind="error")
            reply = self._generate(
                prompt_ids,
                prompt_text,
                temp=CABINET_GENERATE_TEMP,
                k=CABINET_GENERATE_TOP_K,
                p=None,
            )
            detail, classification = self._score_trained_generation(prompt, decision, reply)
            if self.cabinet is not None:
                self.last_entities = list(self.cabinet.entities_of(decision.fact))
            self._remember_history(prompt, reply)
            return TurnResult(
                text=reply,
                kind="cabinet",
                detail=detail,
                related=related,
                match_type=decision.match_type,
                canonical=decision.canonical,
                classification=classification,
            )
        if decision.kind == "search":
            self.last_entities = []
            added = self._save_search(prompt, decision.text)
            self._remember_history(prompt, decision.text)
            self._emit_diagnostics(prompt, decision, decision.text, decision.detail, "SEARCH")
            return TurnResult(
                text=decision.text,
                kind="search",
                detail=decision.detail,
                learned_added=added,
                related=related,
                match_type=decision.match_type,
                canonical=decision.canonical,
                classification="SEARCH",
            )
        if decision.kind in ("calc", "miss"):
            self._remember_history(prompt, decision.text)
            label = "CALC" if decision.kind == "calc" else "MISS"
            self._emit_diagnostics(prompt, decision, decision.text, decision.detail, label)
            return TurnResult(
                text=decision.text,
                kind=decision.kind,
                detail=decision.detail,
                related=related,
                match_type=decision.match_type,
                canonical=decision.canonical,
                classification=label,
            )
        return None

    def _generate_turn(self, prompt: str) -> TurnResult:
        if self.chat_mode:
            budget = max(1, int(self.gpt_config.max_len) - int(self.max_new_tokens))
            prompt_ids, prompt_text = build_chat_prompt_ids(
                self.tokenizer,
                self.history,
                prompt,
                system=self.system,
                max_prompt_tokens=budget,
            )
        else:
            prompt_text = prompt
            prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            return TurnResult(
                text="[No recognized characters in prompt for this vocabulary; try different text]",
                kind="error",
            )
        reply = self._generate(prompt_ids, prompt_text, temp=self.temperature, k=self.top_k, p=self.top_p)
        if self.chat_mode:
            self._remember_history(prompt, reply)
        return TurnResult(text=reply, kind="generate", detail="generate")

    def _generate(self, prompt_ids, prompt_text, *, temp, k, p) -> str:
        active_tracer = self.tracer if self.trace_enabled else None
        if active_tracer is not None:
            active_tracer.dump_tokens(prompt_ids, self.tokenizer, label="prompt")
        need_tokenizer = active_tracer is not None or bool(self.stop_strings)
        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=temp,
            top_k=k,
            top_p=p,
            tracer=active_tracer,
            tokenizer=self.tokenizer if need_tokenizer else None,
            rng=self.rng,
            use_kv_cache=self.use_kv_cache,
            use_cuda_graph=self.use_cuda_graph,
            stop_strings=self.stop_strings,
        )
        new_ids = generated_ids[len(prompt_ids):]
        reply = self.tokenizer.decode(new_ids)
        if self.chat_mode:
            reply = sanitize_assistant_reply(reply, self.stop_strings)
        else:
            reply = reply.strip()
        full_text = self.tokenizer.decode(generated_ids)
        logger.info("prompt=%r generated_text:\n%s", prompt_text, full_text)
        return reply if self.chat_mode else full_text

    def _score_trained_generation(self, typed: str, decision: RouteDecision, generated: str) -> tuple:
        expected = decision.fact.assistant if decision.fact is not None else ""
        if answers_match(expected, generated):
            detail = "generate"
            classification = "MATCH"
        else:
            detail = "generate_target_mismatch"
            classification = classify_generation(expected, generated, self.cabinet, decision.fact)
            logger.warning(
                "cabinet generate_target_mismatch typed=%r canonical=%r expected=%r generated=%r ckpt=%s",
                typed,
                decision.canonical or (decision.fact.user if decision.fact else ""),
                expected,
                generated,
                self.args.checkpoint,
            )
            if self.retrain_path:
                append_jsonl(
                    self.retrain_path,
                    {
                        "typed_question": typed,
                        "canonical_question": decision.canonical,
                        "expected": expected,
                        "generated": generated,
                        "checkpoint": str(self.args.checkpoint),
                        "detail": detail,
                        "classification": classification,
                        "match_type": decision.match_type,
                    },
                )
        self._emit_diagnostics(typed, decision, generated, detail, classification)
        return detail, classification

    def _emit_diagnostics(
        self,
        typed: str,
        decision: RouteDecision,
        generated: str,
        detail: str,
        classification: str,
    ) -> None:
        if not self.diagnostics_path:
            return
        expected = ""
        if decision.fact is not None:
            expected = decision.fact.assistant
        append_jsonl(
            self.diagnostics_path,
            build_turn_record(
                raw_query=typed,
                decision=decision,
                generated=generated,
                expected=expected,
                checkpoint=str(self.args.checkpoint),
                classification=classification,
                detail=detail,
            ),
        )

    def _save_search(self, typed: str, extract: str) -> int:
        if self.cabinet is None or not extract:
            return 0
        before = len(self.cabinet)
        fact = remember_search_hit(self.cabinet, self.learned_path, typed, extract)
        if fact is None:
            return 0
        added = len(self.cabinet) - before
        if added:
            logger.info("cabinet learned +%s user=%r path=%s", added, typed, self.learned_path)
        return added

    def _remember_history(self, user_text: str, reply: str) -> None:
        if self.chat_mode:
            self.history.append((USER_ROLE, user_text))
            self.history.append((ASSISTANT_ROLE, reply))


def _resolve_chat_mode(args: argparse.Namespace, model_name: str) -> bool:
    if getattr(args, "no_chat", False):
        return False
    if getattr(args, "chat", False):
        return True
    return is_chat_model_name(model_name)


def load_index(facts_path: str, learned_path: str) -> CabinetIndex:
    index = CabinetIndex()
    src = facts_path
    try:
        index = load_cabinet(facts_path)
    except FileNotFoundError:
        logger.warning("cabinet facts missing at %s; starting empty trained index", facts_path)
        print(f"[router] facts not found: {facts_path} (trained cabinet empty)")
        src = facts_path
    except (OSError, ValueError) as exc:
        logger.warning("cabinet facts failed to load from %s: %s", facts_path, exc)
        print(f"[router] could not load facts: {exc}")
    n_trained = len(index)
    alias_trained_topics(index)
    n_learned = merge_cabinet(index, learned_path, source="learned")
    alias_learned_topics(index)
    print(
        f"[router] cabinet index: {len(index)} unique "
        f"({n_trained} trained from {src}, {n_learned} learned from {learned_path})"
    )
    return index


def add_session_args(parser: argparse.ArgumentParser) -> None:
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    parser.add_argument("--chat", action="store_true", help="Multi-turn User/Assistant history")
    parser.add_argument(
        "--no-chat", action="store_true",
        help="Force single-prompt REPL even if the checkpoint name looks like chat_5m",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="Optional system prefix for --chat (default: a short simple-assistant line)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Initial sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Initial tokens generated per turn")
    parser.add_argument("--top-k", type=int, default=None, help="Only sample from top K tokens")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold (e.g. 0.9)")
    parser.add_argument(
        "--stop", action="append", default=None,
        help="Stop string (repeatable). Chat mode defaults to User: role markers.",
    )
    parser.add_argument(
        "--router", dest="router", action="store_true",
        help="Force cabinet/calc/search router (default on for chat checkpoints)",
    )
    parser.add_argument(
        "--no-router", dest="router", action="store_false",
        help="Generate every turn (story REPL; default for non-chat checkpoints)",
    )
    parser.set_defaults(router=None)
    parser.add_argument(
        "--facts", type=str, default=str(DEFAULT_FACTS),
        help="Cabinet JSONL/txt for --router (default: data/chat_facts.jsonl)",
    )
    parser.add_argument(
        "--learned", type=str, default=str(DEFAULT_LEARNED),
        help="JSONL overlay for Wikipedia hits (default: output/cabinet_learned.jsonl)",
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="With --router, skip Wikipedia (fact-shaped misses go to the polite hint)",
    )
    parser.add_argument(
        "--cabinet-retrain-log",
        type=str,
        default=str(DEFAULT_RETRAIN),
        help="JSONL path for generate_target_mismatch events",
    )
    parser.add_argument(
        "--cabinet-diagnostics-log",
        type=str,
        default=str(DEFAULT_DIAGNOSTICS),
        help="JSONL path for per-turn router/generation diagnostics",
    )
    cli_common.add_generate_decode_args(parser)
    cli_common.add_trace_args(parser)
