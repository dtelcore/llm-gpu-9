# GT 730 quick guide — story training (2026 recipe)

Fast path to train on Kepler GT 730. Full reference: [`README.md`](README.md) · CLI catalog: [`py_calls.md`](py_calls.md).

**Fast Stories (<1M params):** wizard preset **2** (`story_sub1m`) or `setup/story_sub1m_config.json`. Combines every `data/*.txt` file.

**TinyStories (~3M params):** wizard preset **3** (`tiny_stories`) — the 2026 BiggerTest-aligned recipe below.

**Chat 5M (~5M params, dialogue):** wizard preset **4** (`chat_5m`) or `setup/chat_5m_config.json`. Depth bump (L=4 → L=6) at the same C=256. Mix TinyStories with `data/story_chat.txt`. **Cannot resume** an L=4 TinyStories / BiggerTest checkpoint.

---

## Prerequisites

1. **Venv + CUDA smoke** (once per machine):
   ```powershell
   cd "c:\dev\llm gpu 8"
   .\venv\Scripts\Activate.ps1
   .\venv\Scripts\python.exe setup\2_test_workspace.py
   ```
2. **Corpus:** put TinyStories (or any) `.txt` files under `data/` (e.g. `data\tiny_stories.txt`).
3. **Use project Python** for training and parity — not a random system `python`.

---

## Quickest Fast Stories (<1M) run

Put one or more story `.txt` files under `data/` (one document per line). All matching files are concatenated in sorted filename order.

```powershell
cd "c:\dev\llm gpu 8"
.\venv\Scripts\Activate.ps1
python train.py --config setup\story_sub1m_config.json `
  --checkpoint output\checkpoints\story_sub1m `
  --steps 2000 --no-prompt
```

Or `python train.py --menu` and pick **2. Fast Stories (<1M)**.

| Knob | Value | Notes |
|------|------:|-------|
| Model | C=128, 8 heads, L=4, T=128 | rmsnorm, RoPE, tied embeddings |
| Params | ~0.83M | Stays under 1M at typical BPE vocab sizes |
| LR | 5e-4 | AdamW base |
| Warmup | 500 steps | Linear |
| Batch | 8 | Micro-batch |
| Grad accum | 2 | Effective batch **16** sequences per optimizer step |
| `window_stride` | 64 | Fewer windows per epoch vs dense stride-1 |
| Dataset | combine `data/*.txt` | Reserved name `data_dir` |

---

## Quickest good TinyStories run

```powershell
cd "c:\dev\llm gpu 8"
.\venv\Scripts\Activate.ps1
python train.py --menu
```

1. Resume or **new** run → pick a checkpoint name under `output/checkpoints/`.
2. Scaling preset → **3. Tiny Stories**.
3. **Flag groups** — Enter keeps recipe defaults; or enter e.g. `1,4` to customize Tokenizer / Observability / …. New runs default to **BPE** (`--tokenizer bpe`, 200 merges); use group Tokenizer or `--tokenizer char` for character-level.
4. When Length is customized (or you skip the menu groups and use non-menu prompts), start with **500–2000** steps for smoke, or **20k+** for a real run.

Same wizard from `auto_train.py --menu` (adds a **Smoke generate** flag group). One-shot sample: `python generate.py --menu`.

**Sampling that reads well** (same knobs as training generate-probes / quality trial). In `generate.py --menu` customize group **[1] Sampling** — Enter keeps `temp=0.8` and no top-k/p, which is noisier:

| Knob | Value |
|------|------:|
| `--temperature` | **0.6** |
| `--top-k` | **10** |
| `--top-p` | **0.9** |

```powershell
python generate.py --menu
# or
python generate.py --checkpoint output\checkpoints\YOUR_RUN `
  --prompt "once upon a" --max-new-tokens 256 `
  --temperature 0.6 --top-k 10 --top-p 0.9
```

Per-run log: `output/logs/generate_<checkpoint>_<YYYYMMDD_HHMMSS>.log`.

**Non-interactive length/LR only** (after you already ran the wizard once, or when resuming):

```powershell
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN `
  --steps 10000 --no-prompt
```

Preset hyperparams and architecture come from the checkpoint/config; override LR with `--learning-rate 3e-4` only if you mean to change the recipe.

---

## What preset **3** sets (defaults)

| Knob | Value | Notes |
|------|------:|-------|
| Model | C=256, 8 heads, L=4, T=128 | rmsnorm, RoPE, tied embeddings |
| Params | ~3M | BiggerTest-aligned width/depth; shorter T than BiggerTest256 |
| `dropout_prob` | 0 | Dropout is **not implemented** on GPU — nonzero logs a warning only |
| LR | 3e-4 | AdamW base |
| Warmup | 1000 steps | Linear |
| After warmup | Cosine → `min_lr_ratio` × base | Default floor **0.1** → 3e-5 at end of budget |
| Batch | 4 | Micro-batch |
| Grad accum | 4 | Effective batch **16** sequences per optimizer step |
| `window_stride` | 64 | Fewer windows per epoch vs dense stride-1 |
| LR schedule knobs | `--min-lr-ratio`, `--warmup-steps` | Cosine needs a step budget (`--steps` or epochs) |
| Tokenizer | **BPE** (200 merges) | Char via `--tokenizer char`; old checkpoints keep their vocab |

Training skips host logits sync on the GPU CE path (`need_host_logits=False`); parity and benches still use host logits by default.

---

## Chat 5M (~5M params)

Same Kepler stack; extra capacity comes from **depth** (L=6), not width. Register + train from scratch — architecture is frozen on resume.

```powershell
# Optional: wrap TinyStories as User/Assistant lines (keeps 20% raw stories)
python tools\make_story_chat.py --input data\tiny_stories.txt --output data\story_chat.txt

python train.py --config setup\chat_5m_config.json `
  --checkpoint output\checkpoints\chat_5m `
  --steps 2000 --val-every 500 --runtime-metrics --no-prompt
```

Or `python train.py --menu` and pick **4. Chat 5M**.

| Knob | Value | Notes |
|------|------:|-------|
| Model | C=256, 8 heads, L=6, T=128 | rmsnorm, RoPE, tied embeddings |
| Params | ~4.8–5.0M | At BPE vocab 256–1024; L=8 is ~6.4M (CLI override after VRAM gate) |
| LR / warmup | 3e-4 / 1000 | Cosine `min_lr_ratio=0.1` |
| Batch / accum | 4 / 4 | Effective batch **16** sequences (2048 tokens/step at T=128). `2×8` is the same update size, less VRAM |
| `window_stride` | 64 | Same as TinyStories |
| Tokenizer | BPE (200 merges) | Raise `--bpe-merges` only on a **fresh** run |
| Dataset | combine `data/*.txt` | Drop `tiny_stories.txt` + `story_chat.txt` together |

**Chat format** (one document per line):

```text
User: Tell me a story about a cat. Assistant: Once upon a time there was a kind cat who lived in a small house.
```

**VRAM gate:** if process VRAM is tight, drop `--batch-size 2` and raise `--grad-accum 8`, or `--grad-checkpoint`. Effective tokens per optimizer step stay **16 × T**. Do **not** raise C. Optional later: `--num-layers 8` or `--max-len 256` only after this gate passes. At C=256 a sizable slice of the ~5M params is the embedding table (grows with vocab); `--runtime-metrics` logs global `grad_norm` — watch it in the first few hundred steps.

**Cannot resume L=4 → L=6.** Stage A is from-scratch. Continue a `chat_5m` run only when tokenizer + shapes match. Optional continued pre-training: `--learning-rate 1e-4` on resume.

**Chat REPL** (auto-on when the checkpoint name is Chat 5M; or pass `--chat`):

```powershell
python interactive.py --checkpoint output\checkpoints\chat_5m --chat
```

Defaults: temp **0.7**, top-k **32**, top-p **0.9**. History is `User: … Assistant: …`. Front-truncation drops complete User+Assistant pairs and **keeps the system prefix**; it never tail-slices mid-marker. Generated replies are sanitized so leftover `User:` / ` Use` fragments are not stored in history. Commands: `:clear`, `:system TEXT`.

```powershell
python generate.py --checkpoint output\checkpoints\chat_5m `
  --prompt "User: Tell me a short story about a cat. Assistant:" `
  --temperature 0.7 --top-k 32 --top-p 0.9 --stop "User:" --stop " User:"
```

**Quality:** `--quality-mode chat` (auto for `chat_5m`). Probe prompt defaults to `User: Tell me a short story about a cat. Assistant:`. Scores add `turn_format` / `instruction_follow` / `coherence`. Story mode stays available for TinyStories runs.

```powershell
python train.py --compare-quarters --checkpoint output\checkpoints\chat_5m `
  --quality-mode chat --no-prompt
```

Held-out chat perplexity is the existing `--val-every` split if `story_chat.txt` is in the combined corpus. Training loss is still next-token on the sliding window (User: tokens are not masked); packing-and-masking assistant-only loss would need a different example layout. External GPT-4-style grading of short conversations is a manual step (same spirit as the TinyStories paper), not a runtime dependency.

---

## Hyperparameter scaling on GT 730 (GK208)

This runtime is **GEMM-bound on Kepler**, not a cloud-cluster scaling story. Tune `embedding_dim` (**C**), `num_layers` (**L**), and `max_len` (**T**) against the GT 730’s low DDR3 bandwidth and tight registers (`TILE=16` GEMM in [`model/cuda/kernels.py`](model/cuda/kernels.py)). Param counts use the formula in [`setup/model_config.py`](setup/model_config.py): each block is about **12C²** weights (QKV 3C² + attn-out C² + MLP 8C²) plus small norms/biases.

| Lever | Compute | Wall clock (this trainer) |
|-------|---------|---------------------------|
| **Width C** | Block GEMMs \(\Theta(BTC^{2})\) per layer | **Most expensive.** Doubling C (128 → 256) is ~**4×** those matmuls, plus worse occupancy / register pressure. Attention \(\Theta(BT^{2}C)\) only ~2×. |
| **Depth L** | Repeat the same block | **Linear \(\Theta(L)\).** Safest way to add capacity without a quadratic clock hit. |
| **Context T** | GEMMs \(\Theta(T)\); attention \(\Theta(T^{2}C)\) | At **T ≤ 256** attention is still the smaller term. 128 → 256 is ~**1.5–2.5×** per step, not 4×; tok/s often stays similar because tokens per step also double. |
| **Heads** | `head_dim = C / heads` | Almost free if **C** is fixed. Same FLOPs, a bit more kernel overhead. |
| **Batch / accum** | Linear in microbatch work | `tok/s` uses `batch × accum × T` per optimizer step. Accum trades wall-clock per step for VRAM, not for more tokens per second. |

VRAM is dominated by **activations** (~ **B × T × C × L**), not the few megabytes of weights. That is why wizard **preset 3** (`tiny_stories`, C=256, L=4, T=128) drops microbatch **8 → 4** and raises grad accum **2 → 4**: same effective batch **16**, more wall time per optimizer step, stays inside the ~4 GB DDR3 envelope ([`setup/cuda_activate.md`](setup/cuda_activate.md)).

**Measured on this card** (`output/checkpoints/sub1m_start`, C=128, L=4, T=128, batch=8, accum=2): ~**790 ms/step**, ~**2590 tok/s**, ~**100 MB** process VRAM. A C=256 / L=8 / T=256 combo (e.g. `256-256-8-8`) stacks the expensive axes and is several times slower per token — width and extra depth/context together, not “more params” in the abstract.

For quality vs time on this silicon: extra **depth** is the cheapest param increase; extra **width** is the real quality lever and the one that blows the clock; extra **T** only helps if stories need a longer window.

---

## Useful variations

**Context A/B (throughput vs BiggerTest-style length):**

```powershell
# Default preset T=128 (faster steps)
python train.py --menu

# Longer context (closer to BiggerTest256256), same width/depth
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN --max-len 256 --no-prompt --steps ...
```

**Mid-run val without quarterly I/O:**

```powershell
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN `
  --val-every 500 --no-prompt --steps ...
```

**Resume** (architecture fixed by checkpoint):

```powershell
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN `
  --steps 120000 --no-prompt
```

Quarters still fire at 25/50/75/100% of `--run-budget` (or session total on first run). `--checkpoint-every` updates **latest** only.

**Stride override** (dense windows = old behavior, slower epochs):

```powershell
python train.py ... --window-stride 1
```

---

## Verify before a long run

```powershell
.\venv\Scripts\python.exe -m tests.parity.run_parity
```

Expect **10/10**. Short train smoke:

```powershell
python train.py --menu
# preset 3, then e.g. --steps 200 if prompted, or:
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN --steps 200 --no-prompt
```

Optional throughput check (metrics **off** for bench honesty):

```powershell
python bench_step.py
```

Leave **`--grad-checkpoint` off** when measuring tok/s; use it only to save VRAM (~72 MB at BiggerTest shapes).

---

## Common pitfalls

| Mistake | Fix |
|---------|-----|
| Wrong Python / no PyCUDA | Always `.\venv\Scripts\python.exe` or activated venv |
| Old recipe (LR `1e-5`, batch 32, C=128/L=6, flat LR) | Use preset **2** (`story_sub1m`), **3** (`tiny_stories`), or **4** (`chat_5m`) |
| Resume BiggerTest / tiny_stories into Chat 5M | L=4 checkpoints cannot become L=6 — start a new `chat_5m` run |
| Doubling width and expecting 2× time | C is ~**4×** GEMMs; drop microbatch (preset 3: B=4, accum=4) |
| `dropout_prob: 0.1` expecting regularization | Set **0**; kernels not wired |
| `--grad-checkpoint` for speed | VRAM lever only — adds recompute |
| `--runtime-metrics` / `--memory-timeline` left on | Extra sync/I/O; off for max tok/s |
| Expecting bf16 / FlashAttention / AMP GEMM on GT 730 | Out of scope for this card |
| Expecting full-step CUDA Graph speedup on generate | Kernel-chain (KV append/decode/argmax) captures; GEMM/norm stay eager device |
| FP16 **storage** vs training | Device casts for storage; training math is still FP32 |
| `device_used_mb` ≈ full card | Process-only (excludes HDMI/display); see `vram_driver_used_mb` for total-free |
| Stride 64 “short epoch” | Normal — fewer unique windows per pass over the corpus |

---

## Related

- Historical long run (T=256, late LR 1e-5): [`README.md#notable-run-biggertest256256`](README.md)
- Setup wizard details: [`setup/README.md`](setup/README.md)
