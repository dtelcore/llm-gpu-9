# Changelog

## 0.1.1 — Story-packed TinyStories

Active English is a fresh checkpoint, `english_tinystories_c1024_l6`.
C=1024, L=6, H=8, T=256, layer stream. Batch 4, accum 16 (16,384 tokens
per optimizer step), learning rate 0.0012, warmup 600, `min_lr_ratio` 0.05.
The first cosine length is 4,000 steps.

Training text is `data/tinystories_packed/`. Stories are separated by
`<|endofstory|>` (vocab 6103; the 6,000 merges from 0.1.0 stay, ids unchanged).
Windows stay inside one story. Short stories are padded and those positions
are ignored by the loss. Generation stops on the end token.

`tools/prepare_tinystories.py` still builds the space-joined BPE corpus in
`data/tinystories/`. `--pack-stories --vocab data/tinystories/vocab.json`
writes the packed corpus beside it and will not overwrite the 0.1.0 shards.

0.1.0 checkpoints, logs, and setup JSON now live under `legacy/`. Do not
`--resume` `english_tinystories_c256_l6` or the step-2000 smoke into this
recipe: the vocabulary grew by one token, and a resume does not stretch
the cosine. Product chat stays `chat_facts_v7`.

Recipe: `setup/english_tinystories_c1024_l6_config.json`.
Setup: `setup/english_tinystories_c1024_l6/`.
Policy: `setup/unguided_tinystories_policy.json`.

## 0.1.0 — Re-architecture: from closed cabinet to hybrid learner

Product chat remains `chat_facts_v7` / v9 (frozen fact brain).
Router, App, memory controller, streaming, and the unguided kernel stay.

New primary objective: open English on a TinyStories-scale corpus so the
generative path can produce coherent multi-sentence text. Cabinet + Python
tools stay the high-reliability core. English model is a separate checkpoint
family (`english_tinystories_*`).

Pre-0.1.0 English experiments (english_phase1 on wiki prose, early fact-inject,
v10 mixes) move to `legacy/`. They demonstrated that the wrong corpus does not
teach open English; they are not the active path.

Next gates (after story continuations of 100–200 tokens are consistent):
tool expansion (notes / files / calendar) and a short same-tokenizer SFT that
rewrites cabinet facts as conversation. Arithmetic remains `tools/calc.py`.

## 0.0.9

Unguided trainer kernel and autotrainer daemon (no stdin). Fresh checkpoint
dir and fresh BPE only; no `--resume` across a mix. App.py still requires a
restart to load a promoted net.

- `unguided_trainer.py` trains/evals/decides without `input()`. Remix aborts
  with `NEXT_MIX.json` + `ABORT_REASON`. Policy:
  `setup/unguided_v7_policy.json`. `--dry-run` prints the plan and exits
  without Metal.
- `autotrainer_daemon.py` harvests `output/cabinet_retrain.jsonl` by byte
  offset (BINDING_ENTITY_SWAP / TARGET_MISMATCH only), builds a bounded mix
  (harvested entities + Paris / Cubitt / streaming / how-to anchors + clean
  v7 rows), and spawns the kernel as an isolated subprocess. It refuses to
  launch while App holds Metal. Gate is fail-closed on anchors; promote
  uses existing `promote_best` under `output/checkpoints/`. Crash/NaN/timeout
  poisons that harvest block, backs off, and caps trains per hour so a growing
  mismatch log cannot schedule a from-scratch run every 30s. Gate-fail keeps
  rows for a bounded retry, then poisons. Re-appended questions are skipped.
- Status file `output/autotrainer_status.json` is shown in App (`/api/autotrainer`).
  Viewer mmap of promoted weights is safe; chat generate still needs a process
  restart. Do not `--resume` v7 into a harvest mix.
- Unguided `--log-every` / policy `log_every` (default 10) sets the `[train]`
  line cadence. Does not change `eval_every`. `--name` / `--run-name` /
  `--checkpoint-dir` pick a fresh run + weights dir so a occupied policy path
  is not reused.
- Host-only train monitor: App **Train** tab at `/train/` and standalone
  `trainmon.py` on 7862. Same `[train] step=N/M` series as
  `training_log_plotter.py` (loss, tok/s) plus a 2-D landscape (step ×
  rolling-std, color = loss). Reads `output/logs/*.log` and overlays
  `decisions.jsonl`. No Metal — safe beside `unguided_trainer.py`. Do not
  Load a chat checkpoint in App while a train is running.
- At a successful unguided stop (`max_steps` / wall / early_stop) the kernel
  runs a generate probe on the loaded net and writes
  `output/runs/<name>/generate_probe.md`. Cabinet policies use stored `User:`
  keys (temp 0.2). **`probe_mode=english`** (Phase 1) skips France/Paris
  teacher-forced eval and writes OOD prose completions instead.
  Recipe: `setup/english_phase1_config.json` on `data/train.txt` only
  (`combine` false). Policy: `setup/unguided_phase1_policy.json`. Do not
  `--resume` v7/v8/v9. Product chat stays `chat_facts_v7`. Standalone:
  `unguided_prober.py`. `--no-probe` skips it. Quicktest policy leaves
  `probe_on_stop` off. Do not generate-every-eval.
- Phase 2 inject (same BPE): `tools/build_english_phase2_mix.py` writes
  `data/english_phase2_mix.txt` (full `data/train.txt` plus four unique
  frames per v9 fact, no 300×). Continue with `auto_train.py --resume
  --checkpoint output/checkpoints/English-Phase1 --dataset-path
  data/english_phase2_mix.txt --reset-lr-schedule`. Unguided will not
  resume that dir. `--dataset-path` is refused without `--resume` and
  never rebuilds BPE. Probe: `unguided_prober.py --probe-mode inject`.
- Phase 2b dense inject: `tools/build_english_phase2_mix.py --dense` writes
  `data/english_phase2b_mix.txt` (geo/organ/prime only, ~15% unique frames,
  subsampled `train.txt`). Same BPE resume into English-Phase1. Leaves the
  0.15% mix on disk. Prefer `--learning-rate 6e-5` (loss moved more than at 5e-5). Product stays `chat_facts_v7`.
- v10 integration mix (fresh BPE, not a Phase 1 resume): `tools/build_chat_facts_v10.py`
  writes `data/chat_facts_v10.jsonl` from v9 uniques (split geo Assistants kept),
  pinned L’Hopital/Cramer named-after golds, prime-1 gold that keeps “not”.
  Native `User:`/`Assistant:` only (the v10_4 shared wrappers are gone), with a
  trailing ` User:` stop marker on chat lines so generate can halt. Probe
  core 40× / remaining keys 10×, not 300×. Wiki slice in the same file so
  native chat is ~40–50% of lines. Policy `setup/unguided_v10_policy.json`:
  `probe_mode=inject`, `max_steps` 10000, `early_stop_patience` 200. Inject
  prompts are held-out `User:` paraphrases, not wrapper stems. Do not
  `--resume` v7/v8/v9, English-Phase1, or `chat_facts_v10_4`/`_5`/`_6`. Product
  stays `chat_facts_v7`. Pass/fail: stored Paris/kidney/prime-1 without wiki
  tail; inject paraphrases on-target; valley/photosynthesis still English;
  Belgium on a France prompt is a fail. Not v9 96% exact.

## 0.0.8

One **App**, related cabinet follow-ups, and a **pick-a-neuron** weight view.
Router and 2 GB cap are unchanged from 0.0.7.

Cabinet routing now rewrites a leading `the`/`a`/`an` only inside inventor and
capital templates, and loads explicit aliases from `data/cabinet_aliases.json`
(trained keys win over learned Wikipedia overlays). Bad source rows in
`data/cabinet_quarantine.json` are excluded from the runtime cabinet and from
`make_fact_mix.py`. Trained cabinet replies stay model-generated; a stored-target
mismatch is flagged `generate_target_mismatch`, written to
`output/cabinet_retrain.jsonl`, and recorded in `output/cabinet_diagnostics.jsonl`.
Offline router/token-rank probe: `tools/eval_cabinet_bindings.py` (no cosine
thresholds; teacher-forced ranks only with `--checkpoint`).

- `App.py` + `app/` house chat (`/chat`) and npzviewer (`/weights`) with a
  checkpoint selector. First Load puts that net on Metal; the viewer mmaps the
  same `weights.npz`. Switching models **restarts** the process (no second net,
  no in-process hot-swap). Standalone `webui.py` (7860) and `npzviewer.py` (7861)
  still work; do not run them next to `App.py` or `interactive.py`.
- After a cabinet **generate**, fact-shaped misses list related **trained**
  questions (chips / `:related`) instead of auto-Wikipedia. Session
  `last_entities` is closed-world. Trained topic aliases (`what is neonics` →
  `Tell me about Neonics.`). `:search` and a cold-start `What is …` still hit
  Wikipedia. No fuzzy cabinet match.
- Finding: **one trained string is one key**. v6 recites
  `What is the capital of France?` → Paris, but `Where is Paris?` is a different
  question and misses. Wikipedia after generate poisoned follow-ups (commune
  dumps). Chips: `cabinet · generate` vs `cabinet · replay`.
- Linked mix: `tools/wikidata_to_facts.py` / `make_fact_mix.py --linked` adds
  `Where is {city}?`, `What country is {city} in?`, `What is {country}'s capital?`
  (no “largest city of”). `data/chat_facts_v7.jsonl` +
  `setup/chat_facts_v7_config.json` from scratch. Do not `--resume` v6 into v7.
  Smoke that line with `User: Where is Paris? Assistant:` — that is not the
  France→Paris generate probe.
- npzviewer pick-a-neuron: top-|weight| partners on a matrix row/column
  (`token_embedding` = token↔channel; `mlp_expand` = residual↔hidden). Inspect
  does not rebuild the page (no scroll jump). Heatmap is a block-mean preview;
  the partner list is the precise view. Tiny viewer recipe:
  `setup/user_facts_tiny_config.json` (C=16 H=2 L=2 T=16) on
  `data/user_facts_tiny.jsonl` (`make_fact_mix.py --user-only`). T=16 cannot
  hold a full fact line; that run is for the inspector, not recitation.
  C=16 H=16 is illegal here (RoPE needs even `head_dim`).
- Learned JSONL load splits glued records (missing newline) instead of crashing
  `remember()` pads a trailing newline. Wiki search: strip `first N`, then
  `list=search`; skip list/commune pages unless asked; clip to two sentences.

## 0.0.7

Python **router** around one chat checkpoint: cabinet lookup, then calc, then Wikipedia.

- `interactive.py` defaults `--router` on for chat checkpoints (`--no-router` for story).
  Hits feed the stored `User: … Assistant:` prompt into generate (temp 0.2). `2+2` is
  AST+Decimal (`tools/calc.py`, no `eval`). Unknown `What is …` questions use
  Wikipedia via stdlib urllib (`tools/wiki_search.py`). Search uses the stripped topic
  (`tell me about ford` → `ford`), not the wrapper phrase. A Wikipedia hit is appended
  to `output/cabinet_learned.jsonl` and replayed on later exact/topic hits (weights
  unchanged). Miss hints and calc are not stored. Network/empty extract falls
  through to a polite miss; snippets are not fed back into the GPT.
  Flask UI: `webui.py` (same session as `interactive.py`; one process).
  Weight inspector: `npzviewer.py` (host mmap of `.npz` / `.npy` / `.npx`; no
  Metal load). Cabinet chips: `generate` (Metal recitation) vs `replay`
  (learned JSONL).
  Short topic phrases (`first 5 prime numbers`) also try Wikipedia; story
  openings (`once upon a time`) still miss.
- Exact normalized cabinet index: [`training/cabinet_index.py`](training/cabinet_index.py).
  No fuzzy match (unobtanium must not hit layer-streaming). Cabinet wins over calc
  (`What is 0 factorial?`). One checkpoint only — do not load a story net in the
  same process (2 GB cap).
- Do not `combine: true`. Do not `--resume` v4 into a new mix. Keep
  `output/checkpoints/chat_facts_v4` as the 105×300 recitation baseline.
  Optional v6: unique learned extras folded into `data/chat_facts_v6.jsonl`
  (`tools/make_fact_mix.py --learned`), train `setup/chat_facts_v6_config.json`
  from scratch (new BPE). Do not `--resume` v5.

## 0.0.6

Chat **fact data pipelines** and the learning rules that actually stick on this 2 GB Air.

- Facts live as exact `User: … Assistant: …` lines. `tools/make_fact_mix.py` repeats
  `data/user_facts.txt` plus `data/facts/*.txt` and writes `data/chat_facts.jsonl`.
  `setup/chat_facts_config.json` trains **only** that file (`combine: false`, explicit
  `path`). `combine: true` or `chat_c256_l6_config.json` / `chat_train.txt` wash the
  cabinet out with ~1M unique wraps.
- What stuck in training: C=512 L=6 T=256 stream, ~20M params. **~300 repeats** of a
  few dozen short Q&As recites (v2). **20 repeats** of a 300-topic wiki slice in 300
  steps does not. New words need a **new BPE and a new checkpoint**, not `--resume`.
- `tools/wikidata_to_facts.py` pulls modest SPARQL groups (capitals, elements,
  inventors, birth years) into `data/facts/wikidata_facts.txt`. Drops Q-id labels.
  Uses stdlib urllib (optional SPARQLWrapper). Keep LIMITs small. Same-question
  conflicts are dropped entirely.
- `tools/wikidata_to_facts2.py` (`wikidatafetch2`) writes separate tech / health /
  maths packs (`tech_facts.txt`, `health_facts.txt`, `maths_facts.txt`). Distinct
  templates per kind; health stays encyclopedic (no dosing/treatment).
- Dataset `path` / `dataset_path` always wins over combine. JSONL query/response
  records load as native chat lines. `--stop` on generate/auto_train cuts at `User:`.

## 0.0.5

Sequential layer streaming so L=32–48 at C=256 can train under the hardcoded **2 GB** Metal cap.

- `layer_strategy=stream` keeps embeddings + final LN resident and pages one transformer
  block (weights + Adam m/v + VJP working set) onto Metal at a time. Idle layer Adam
  moments stay on host NumPy. Residual-stream checkpoints `h_in` stay on device (~48 MB
  at L=48, B=4, T=256, C=256); host offload of `h` is not in this release.
- Train: stream-forward → loss → recompute backward → global clip on the host grad dict
  → stream Adam (`t += 1` once, then `step_layer`). Generate prefill/decode load/unload
  the same way and pack KV into arenas before dropping a block. CUDA-graph decode is
  disabled while streaming.
- Autoscale enables stream + `eval_per_layer` **before** shrinking context `T`.
  `--layer-stream` forces it (L=6 bring-up). `--no-layer-stream` keeps the old refuse.
  Generate estimates no longer subtract Adam (that underflowed to negative MB on stream).
- Expect **2–4×** lower tok/s versus a fully resident stack, plus fanless thermal
  throttle on long GEMMs. GPT-2 residual scale `1/√(2L)` stays mandatory for deep stacks.

## 0.0.4

Process memory controller and a sequential-layer seam for later pipeline / layer parallel.

- Autoscale batch, context, gradient checkpointing, FP16 activation storage, and
  per-layer MLX `eval` to the hardcoded **2 GB** process cap (`model/mlx/env.py`).
  Architecture (C, L, H) is never changed. `--no-autoscale` keeps refuse-if-over.
  `--memory-headroom` (default 0.15) reserves compile/scratch inside that cap.
- `eval_per_layer` realizes the residual stream after each block so unused
  intermediates can free. That is the hook for future layer-pipeline / weight
  streaming; swapping idle layer weights is not implemented yet. If
  weights+Adam alone exceed the usable budget, training still refuses.
- Train and generate both go through `training/memory_controller.py`. The plan
  is logged as `[memory] ...` and stored on the run config as `memory_plan`.

## 0.0.3

Faster tokenizer path and safer M3-Air training since v0.0.2.

- Incremental BPE merge training on unique-word counts (4000 merges on 200k chars in ~1s).
- Parallel unique-word BPE encode (`LLM_BPE_WORKERS`) with tqdm bars for merges, scan, encode, and stitch.
- Memory preflight before weight allocation; first-run config/corpus seeding; training-log plotter fixes.

## 0.0.2

Apple MLX port: explicit VJPs, Metal train/generate, 2 GB process budget.

## 0.0.1

Planning.
