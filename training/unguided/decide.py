"""Pure decision engine for the unguided trainer. Fully unit-tested."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class Decision(str, Enum):
    CONTINUE = "continue"
    PROMOTE = "promote"
    EARLY_STOP = "early_stop"
    ABORT_SPIKE = "abort_spike"
    ABORT_REMIX = "abort_remix"
    STOP_LIMIT = "stop_limit"


@dataclass
class DecideContext:
    """Snapshot passed to decide()."""
    step: int
    max_steps: int
    wall_s: float
    max_wall_s: float
    val_loss: float | None
    best_val_loss: float | None
    recent_val_losses: Sequence[float]  # oldest → newest, length ≤ window
    nan_detected: bool
    cabinet_exact_match: float | None
    policy: dict[str, Any]
    no_improvement_count: int = 0


@dataclass
class DecideResult:
    action: Decision
    reason: str
    promote: bool = False
    next_mix_recipe: dict[str, Any] | None = None


@dataclass
class NextStepContext:
    """Post-stop snapshot: val + teacher-forced + generate probe + mix smells."""

    step: int
    max_steps: int
    val_loss: float | None
    best_val_loss: float | None
    cabinet_exact_match: float | None
    generate_exact_rate: float | None
    generate_swap_rate: float | None
    generate_n: int = 0
    ood_mix_copies: int = 0
    ood_n: int = 0
    dirty_gold: int = 0
    shared_assistants: int = 0
    collapsing_families: tuple[str, ...] = ()
    long_unique_fail: int = 0
    short_template_exact: float | None = None
    policy: dict[str, Any] | None = None


@dataclass
class NextStepItem:
    action: str
    needed: bool
    why: str


@dataclass
class NextStepResult:
    mode: str
    understands: bool
    headline: str
    primary: str
    items: list[NextStepItem]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "understands": self.understands,
            "headline": self.headline,
            "primary": self.primary,
            "items": [
                {"action": i.action, "needed": i.needed, "why": i.why} for i in self.items
            ],
            "reasons": list(self.reasons),
        }


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("inf")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def decide(ctx: DecideContext) -> DecideResult:
    """
    Deterministic policy. No I/O, no randomness, no input().
    Order of checks is intentional and stable.
    """
    p = ctx.policy
    spike_ratio = float(p.get("loss_spike_ratio", 2.0))
    patience = int(p.get("early_stop_patience", 4))
    remix_cfg = p.get("remix_if") or {}
    remix_threshold = remix_cfg.get("cabinet_exact_match_below")
    remix_after = int(remix_cfg.get("after_steps", 10**9))

    if ctx.nan_detected or (ctx.val_loss is not None and not _finite(ctx.val_loss)):
        return DecideResult(
            action=Decision.ABORT_SPIKE,
            reason="non_finite_loss",
        )

    if ctx.val_loss is not None and ctx.recent_val_losses:
        med = _median(ctx.recent_val_losses)
        if med > 0 and ctx.val_loss > spike_ratio * med:
            return DecideResult(
                action=Decision.ABORT_SPIKE,
                reason=f"loss_spike val={ctx.val_loss:.4f} median={med:.4f} ratio={spike_ratio}",
            )

    if ctx.step >= ctx.max_steps:
        return DecideResult(
            action=Decision.STOP_LIMIT,
            reason=f"max_steps={ctx.max_steps}",
        )
    if ctx.wall_s >= ctx.max_wall_s:
        return DecideResult(
            action=Decision.STOP_LIMIT,
            reason=f"max_wall_s={ctx.max_wall_s}",
        )

    probe_mode = str(p.get("probe_mode") or "cabinet").strip().lower()
    if (
        probe_mode not in {"english", "inject", "tinystories"}
        and remix_threshold is not None
        and ctx.cabinet_exact_match is not None
        and ctx.step >= remix_after
        and ctx.cabinet_exact_match < float(remix_threshold)
    ):
        next_mix = {
            "action": "abort_remix",
            "reason": "cabinet_exact_match_below_threshold",
            "cabinet_exact_match": ctx.cabinet_exact_match,
            "threshold": remix_threshold,
            "after_steps": remix_after,
            "suggested_cmd": [
                "python",
                "tools/make_fact_mix.py",
                "--config",
                p.get("recipe", "legacy/setup/chat_facts_v7_config.json"),
            ],
        }
        return DecideResult(
            action=Decision.ABORT_REMIX,
            reason=f"cabinet_exact_match={ctx.cabinet_exact_match:.3f} < {remix_threshold}",
            next_mix_recipe=next_mix,
        )

    promote = False
    if ctx.val_loss is not None:
        if ctx.best_val_loss is None or ctx.val_loss < ctx.best_val_loss:
            promote = True

    if ctx.no_improvement_count >= patience:
        return DecideResult(
            action=Decision.EARLY_STOP,
            reason=f"no_improvement_for_{patience}_evals",
            promote=promote,
        )

    return DecideResult(
        action=Decision.CONTINUE if not promote else Decision.PROMOTE,
        reason="improved" if promote else "ok",
        promote=promote,
    )


def decide_next_step(ctx: NextStepContext) -> NextStepResult:
    """After a successful stop: what to change next. No I/O, no Metal.

    Mid-train ``decide()`` does not call this. Unguided runs the generate
    prober once at max_steps / early_stop / wall, then this reads every
    signal together (val CE, teacher-forced cabinet exact, generate exact,
    neighbour swaps, mix smells, OOD).
    """
    exact = ctx.generate_exact_rate
    swap = ctx.generate_swap_rate or 0.0
    teacher = ctx.cabinet_exact_match
    collapsing = tuple(ctx.collapsing_families)
    reasons: list[str] = []
    probe_mode = str((ctx.policy or {}).get("probe_mode") or "cabinet").strip().lower()
    if probe_mode in {"english", "inject", "tinystories"}:
        dumps = int(ctx.ood_mix_copies or 0)
        n_ood = int(ctx.ood_n or 0)
        dump_rate = (dumps / n_ood) if n_ood else 0.0
        more_steps = ctx.step < ctx.max_steps
        if probe_mode == "tinystories":
            headline = (
                "TinyStories English. Score 160-token story continuations for "
                "coherent characters and grammar. Val loss will stay above the "
                "cabinet's 0.03. 10k steps is not a stopping point."
            )
            mix_why = "TinyStories only. Do not mix wiki prose or cabinet facts."
            config_why = "Keep C=256 L=6 until 100–200 token stories stay consistent."
            policy_why = "Cabinet remix is disabled for probe_mode=tinystories."
            mode_name = "tinystories_english"
        elif probe_mode == "inject":
            headline = (
                "v10 integration probe. Score paraphrased Paris/kidney/prime "
                "and valley/photosynthesis English. Neighbour Belgium on a "
                "France prompt is a fail. Not v9 96% exact."
            )
            mix_why = "v10 mix is unique frames plus a thin wiki slice; do not 300× remix."
            config_why = "Keep C=256 L=6 unless recitation-exact is the goal (that is v9)."
            policy_why = "Cabinet remix is disabled for probe_mode=inject."
            mode_name = "inject_integration"
        else:
            headline = (
                "Phase 1 English foundation. Score OOD completions for grammar, "
                "not cabinet exact-match."
            )
            mix_why = "Phase 1 uses data/train.txt only."
            config_why = "Keep C=256 L=6 until Phase 1 OOD looks like English."
            policy_why = "Cabinet remix is disabled for probe_mode=english."
            mode_name = "english_foundation"
        reasons.append(f"val_loss={ctx.val_loss:.4f}" if ctx.val_loss is not None else "val_loss=n/a")
        reasons.append(f"ood_dump={dumps}/{n_ood}")
        reasons.append(f"step={ctx.step}/{ctx.max_steps}")
        if dump_rate:
            reasons.append(f"ood_dump_rate={dump_rate:.2f}")
        items = [
            NextStepItem("change_data", False, "Not a recitation-cabinet success metric."),
            NextStepItem("change_mix", False, mix_why),
            NextStepItem(
                "more_steps",
                more_steps,
                "Hit max_steps before judging the next mix." if not more_steps
                else f"Stopped at {ctx.step}/{ctx.max_steps}; more steps are optional.",
            ),
            NextStepItem("new_config", False, config_why),
            NextStepItem("new_policy", False, policy_why),
        ]
        return NextStepResult(
            mode=mode_name,
            understands=False,
            headline=headline,
            primary="hold" if not more_steps else "more_steps",
            items=items,
            reasons=reasons,
        )

    understands = bool(
        exact is not None
        and exact >= 0.85
        and swap <= 0.05
        and ctx.ood_n >= 2
        and ctx.ood_mix_copies == 0
    )

    change_data = ctx.dirty_gold > 0 or ctx.shared_assistants > 0
    change_mix = swap >= 0.15 or bool(collapsing)
    more_steps = (exact is None or exact < 0.80) and not understands
    new_config = bool(
        (exact is not None and exact < 0.15 and ctx.step >= max(int(ctx.max_steps or 0), 1500))
        or (
            ctx.long_unique_fail >= 2
            and (ctx.short_template_exact or 0.0) >= 0.6
            and ctx.step >= 1200
        )
    )
    new_policy = bool(
        teacher is not None
        and exact is not None
        and teacher < 0.15
        and exact >= 0.35
    )

    if understands:
        mode = "generalizing"
        headline = "Binding holds and out-of-mix replies are not mix copies."
    elif exact is None:
        mode = "unknown"
        headline = "No generate probe; next step is from val / teacher-forced only."
    elif exact < 0.25:
        mode = "not_reciting"
        headline = "Stored keys are not being recited yet."
    elif collapsing and (exact or 0.0) >= 0.4:
        mode = "mixed_recitation"
        headline = "Some templates recite; neighbour frames still swap entities."
    else:
        mode = "memorizing"
        headline = "This snapshot is memorizing stored templates, not understanding."

    if understands:
        reasons.append("Generate exact is high and OOD did not copy another mix row.")
    else:
        reasons.append(
            "Cabinet generate is a reciter. OOD copies or neighbour swaps mean "
            "template memory, not understanding."
        )
    if exact is not None:
        reasons.append(f"generate_exact={exact:.3f} swap={swap:.3f} n={ctx.generate_n}")
    if teacher is not None:
        reasons.append(f"teacher_forced_cabinet_exact={teacher:.3f}")
    if ctx.val_loss is not None:
        reasons.append(f"val_loss={ctx.val_loss:.4f}")
    if collapsing:
        reasons.append("collapsing families: " + ", ".join(collapsing))

    data_why = "Gold looks unique and mentions the question slot."
    if ctx.dirty_gold and ctx.shared_assistants:
        data_why = (
            f"{ctx.dirty_gold} gold rows omit the question slot; "
            f"{ctx.shared_assistants} Assistant strings are shared by two or more User keys."
        )
    elif ctx.dirty_gold:
        data_why = f"{ctx.dirty_gold} gold rows omit the question slot (dirty labels)."
    elif ctx.shared_assistants:
        data_why = (
            f"{ctx.shared_assistants} Assistant strings are reused across different User keys "
            "(linked mix collapse)."
        )

    mix_why = "Neighbour-template swap rate is low."
    if change_mix:
        bits = []
        if swap >= 0.15:
            bits.append(f"swap_rate={swap:.3f}")
        if collapsing:
            bits.append("families " + ", ".join(collapsing))
        mix_why = (
            "Same-frame collisions: " + "; ".join(bits) + ". More steps on this mix "
            "can drop CE and still swap the one token that differs."
        )

    if exact is None:
        steps_why = "No generate exact; train further only if val is still improving."
    elif exact >= 0.80:
        steps_why = f"Generate exact {exact:.1%} is already high; extra steps are optional."
    elif mode == "not_reciting" and change_mix and swap > (exact or 0.0):
        steps_why = (
            f"Generate exact {exact:.1%} is low and swaps dominate. Fix the mix first; "
            f"a longer run on the same frames will keep swapping."
        )
        more_steps = False
    else:
        steps_why = (
            f"Generate exact {exact:.1%} at step {ctx.step}/{ctx.max_steps}. "
            "A longer from-scratch run can raise recitation on unique lines."
        )

    if new_config:
        config_why = (
            "Short templates recite but long unique lines stay garbled after many steps, "
            "or generate exact stayed near zero at the step budget. New C/L/T, new dir."
        )
    else:
        config_why = (
            "Too early or short recitation already works. Do not change C/L/T until "
            "mix + steps are exhausted."
        )

    if new_policy:
        policy_why = (
            f"Teacher-forced cabinet exact is {teacher:.3f} while generate exact is "
            f"{exact:.3f}. remix_if on teacher-forced 0.0 can abort a net that already "
            "recites. Keep generate probes at stop, not every eval."
        )
    else:
        policy_why = (
            "Stop gates and generate scores agree, or there is no generate probe. "
            "Do not add generate-every-eval (Metal, slow)."
        )

    items = [
        NextStepItem("change_data", change_data, data_why),
        NextStepItem("change_mix", change_mix, mix_why),
        NextStepItem("more_steps", more_steps, steps_why),
        NextStepItem("new_config", new_config, config_why),
        NextStepItem("new_policy", new_policy, policy_why),
    ]

    if understands:
        primary = "hold"
    elif mode == "not_reciting" and change_mix and swap > (exact or 0.0):
        primary = "change_mix"
    elif change_mix:
        primary = "change_mix"
    elif change_data:
        primary = "change_data"
    elif new_config:
        primary = "new_config"
    elif more_steps:
        primary = "more_steps"
    elif new_policy:
        primary = "new_policy"
    else:
        primary = "hold"

    return NextStepResult(
        mode=mode,
        understands=understands,
        headline=headline,
        primary=primary,
        items=items,
        reasons=reasons,
    )
