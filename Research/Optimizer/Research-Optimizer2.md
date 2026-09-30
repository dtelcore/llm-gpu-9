# Optimization-step efficiency (research only)

Scope: this Apple MLX tree (MacBook Air M3, hardcoded 2 GB process cap). No training-loop or optimizer code was changed. Evidence is from this repo. The only published millisecond figure is a Kepler GT 730 measurement in `guide.md`; this tree does not check in an M3 `bench_step.py` log.

Two different clocks:

1. **Step time** — wall-clock for one optimizer step (all micro-batches, clip, AdamW).
2. **Steps to quality** — how many of those steps you need before val loss / generation is useful.

A change that cuts milliseconds by running a smaller update often increases the step count. Gradient accumulation is the clearest example already written down in `guide.md`. The 2000-step c512 run below is the case where that trade was taken on purpose, and the early quality result favors the harder step.

## Update: 2000-step c512 hardness result

The JSON/README disagreement is the hardness search for a **2000-step** budget, not a stale file to revert. The README still describes the long pass (accum 4, warmup 1000, 8192 tokens/step, ≈188,444 optimizer steps per epoch). The short run is testing fewer, heavier updates.

Reported result (not a log checked into this tree): at C=512, **batch 4, accum 16**, 2000 optimizer steps are reaching the same gains that previously took about **400,000 steps, one full epoch**, and misses are still there.

What that hardness is, in this trainer:

| Recipe | Micro-batch | Accum | Sequences / step | Tokens / step (T=256) | Estimator |
|---|---:|---:|---:|---:|---|
| Light epoch (c256 JSON: B=4, accum=4) | 4 | 4 | 16 | 4,096 | ~531 MB, resident |
| README long c512 pass | 8 | 4 | 32 | 8,192 | ~1711 MB, resident, little slack |
| **2000-step run as described** | **4** | **16** | **64** | **16,384** | **~1107 MB, resident, fits easily** |
| File on disk today (`english_tinystories_c512_l6_config.json`) | 8 | 16 | 128 | 32,768 | ~1711 MB, resident, ~30 MB under the usable cap |

`train.sh` does not pass `--batch-size` or `--grad-accum`. The next launch from that script uses **batch 8, accum 16**, which is a heavier step than the batch-4 run just described (twice the micro-batch work, twice the tokens per step, and close to the 2 GB planner edge).

Epoch arithmetic, using the README’s ≈188,444 steps at effective batch 32 over the same stride-128 windows:

- Windows in one pass ≈ 188,444 × 32 ≈ 6.03 million.
- Light effective batch 16 (c256): one pass ≈ 188,444 × 32 / 16 ≈ **377,000** optimizer steps. The “~400,000 steps, one epoch” figure sits on that lighter step.
- Batch 4 × accum 16 is effective batch 64: one pass ≈ 188,444 × 32 / 64 ≈ **94,000** optimizer steps.
- 2000 hard steps are about **2% of one pass** (2000 / 94,000) and about **32.8 million** token-presentations (2000 × 64 × 256). The full light epoch presents on the order of 1.5 billion (377,000 × 16 × 256).

So the quality match is not “we saw the corpus sooner.” It is “a wider net and a 4× larger token batch moved the probe as far in 2% of a pass as the light step did in a full pass.” Misses remaining means the 2000-step point is a checkpoint on that curve, not a finished model.

Wall-clock, using `guide.md` scaling and not a timed M3 log: doubling C is ~4× the block GEMMs, and accum 16 vs accum 4 is 4× the micro-batches, at the same micro-batch of 4. One hard step is on the order of **16×** one c256 step if the step is GEMM-bound. 2000 × 16 ≈ 32,000 light-step-equivalents, against ~377,000–400,000 light steps. That is about a **12×** shorter path to the same probe **if** the scaling holds and the samples really match. Fanless throttle (`README.md`) can eat part of that.

Schedule on a 2000-step budget with the JSON knobs (`warmup_steps` 600, `min_lr_ratio` 0.05, base 1.2e-3): the first 600 steps are still warmup (30% of the run below peak LR), then cosine over the remaining 1400 steps down to 6e-5. If `--steps 2000` is the cosine length, step 2000 is already at the floor. Extending that same process past 2000 without resetting `total_steps` keeps training at the floor. A longer hard-step run wants `--steps` (or `--run-budget`) set to the real length so the cosine is stretched, with warmup kept a small fraction of that length.

---

## A. Current optimization-step path

### What one train step does

Live entry points:

- `train.py` `train()` — guided loop, quarterly probes, checkpoint reload probe.
- `training/unguided/loop.py` `train_segment()` — same forward/loss/backward/Adam primitives, then a host weight sync and `save_checkpoint` at the end of every segment.
- `bench_step.py` — the GPU contract (`forward_batch` → GPU CE → `backward_batch_gpu` → `AdamWGPU`) on a toy shape. Metrics must be off.

There is one optimizer implementation that the loop actually calls: `training/gpu_optimizer.py` `AdamWGPU`. The string `"optimizer": "adamw"` in configs is printed by the wizard and is not a dispatch switch (`setup/training_setup.py` only prints it). `training/optimizer.py` `AdamW` is the host NumPy reference. `bench_profile.py` times that host optimizer, so its “optimizer %” is not the Metal step.

For each micro-batch inside an optimizer step (`train.py` around the `for batch, epoch` loop):

1. **Batch.** `WindowedDataset.iter_batches` slices a token memmap or array into `(x, y)` windows and `np.stack`s them. No packing mask, no document boundary.
2. **Forward.** `GPTModel.forward_batch(..., need_host_logits=False)` on the GPU path (`model/gpt.py`). Token ids are copied into MLX every call (`embedding_lookup_tokens`). Each block is RMSNorm (or LayerNorm), QKV as **three** GEMMs (`linear_qkv_split`), RoPE, causal attention built from `mx.matmul` + softmax, output proj, residual+norm, MLP as matmul + tanh-GELU + matmul. After every block the residual is forced with `cuda_ops.eval_for_host(h_d)` (`model/gpt.py` ~307), including when `eval_per_layer` is off.
3. **Loss.** `softmax_cross_entropy_batch_gpu` → `ops.cross_entropy`. Mean token CE. `eval_for_host` on the scalar loss **and** the full `dlogits` every micro-batch (`model/mlx/ops.py` `cross_entropy`). Training does not pull host logits unless a tracer asked (`need_host`).
4. **Backward.** `backward_batch_gpu`: explicit VJPs, no `mx.value_and_grad`. Attention and MLP caches are kept unless `gradient_checkpointing` is on, in which case those forwards are recomputed (`_attention_backward_batch_gpu`, `_mlp_backward_gpu`). `tests/parity/test_modern_step.py` checks that checkpoint grads match the full cache.
5. **Accumulate.** `_accumulate_grads_` adds device grads in place. One extra grad buffer, not one buffer per micro-batch (`training/memory_controller.py` adds a single extra `param_b` when accum > 1). Then `_scale_grads_` divides by the micro-batch count.
6. **Clip.** `AdamWGPU.clip_grads_` → `grad_global_norm_sq`, which `eval_for_host`s the sum of squares every optimizer step.
7. **AdamW.** Python loop over parameter tensors, `adamw_update` per tensor, then one `mx.eval` of the updated weights (`gpu_optimizer.py` `_update_keys`). Cosine LR with linear warmup (`current_lr`). `adamw_update_batched` raises `RuntimeError` (“CUDA pointer-table only”).
8. **Host sync of weights** is not on the resident hot path. `sync_host_weights` runs at checkpoint time. Resume restores `optimizer.t` and **zeros** `m`/`v` (`train.py` warning: weights-only checkpoints).

Streaming (`layer_strategy=stream`) replaces steps 2–7 with the sequence in `.cursor/plans/layer_streaming_phase_1_1be7128d.plan.md`:

stream-forward (save `h_in` only) → loss → **recompute** each block on the way back → copy grads to host → global clip on the host dict → per-layer Adam with host `m`/`v` uploaded, then stashed back, `scratch_pool.clear()` on unload.

`README.md` and `CHANGELOG.md` (0.0.5) both say to expect **2–4× lower tok/s** than a resident L=6 stack, plus fanless thermal throttle.

### Compile, fusion, allocator

| Mechanism | State in this tree |
|---|---|
| `mx.compile` / CUDA Graph | Stub. `model/mlx/graph.py`: “CUDA Graph dropped on MLX; use mx.compile on the closed step later.” `capture_gpu_callable` logs and returns fallback. Same note in `tools/stage3_milestones.py` `run_311` and `.cursor/plans/m3_mlx_gpt_port_19500603.plan.md`. |
| `mx.fast.*` (SDPA, RMSNorm) | Forbidden on the train path by that port plan, so the hand VJPs stay visible. `model/mlx/ops.py` header: “Forward = composed mx ops (no mx.fast.* on the train path). Backward = explicit VJPs. No autograd.” |
| MLP “fusion” | `fused_mlp_row` is `matmul_bias_gelu` then `matmul_bias` (`ops.py`). `bench_mlp_fusion.py` still says it decides `_USE_FUSED_MLP_ROW_KERNEL`. That flag is not in `ops.py`. The two bench paths are the same composed math. |
| QKV | Intentionally split into 3 GEMMs so a `[B·T, 3C]` buffer is never live (`linear_qkv_split` docstring; port plan requires parity on the split). |
| Attention | `fused_causal_attention_from_qkv` is a Python composition, not a fused Metal kernel. Causal mask is rebuilt with `mx.arange` every call (`_causal_mask`). |
| RoPE | `rope_apply_inplace` rebuilds NumPy `cos`/`sin` and uploads them on every forward and backward apply. |
| Scratch | `ScratchPool` (named reuse) and `LifetimeAllocator` (QKV split temps). Streaming clears the pool on every layer unload. |
| Per-layer `mx.eval` | Unconditional on the GPU forward residual. `eval_per_layer` additionally calls `_eval_stream` (backward too). The memory log line `eval_per_layer=0` does not mean the forward stayed lazy. |

`model/mlx/env.py` sets `mx.set_memory_limit` to 2 GB and keeps a 64 MB peak-transient slack “for Metal compile / freed VJP scratch” even though the train step is not compiled. `--memory-headroom` (default 0.15) reserves the same kind of slack inside the planner (`usable = 85% of 2 GB` ≈ 1741 MB).

### Logging, eval, checkpoint on or next to the hot path

| Cadence | What it costs | Default |
|---|---|---|
| Every micro-batch | CE `mx.eval` of loss + `dlogits` | Always |
| Every optimizer step | Grad-norm `mx.eval`; per-layer forward `mx.eval` | Always |
| `log_every` | Print, `get_memory_usage`, and (unless `--no-layer-grads`) `summarize_layer_grad_norms` | CLI default **100** (`cli_common.py`). c512 `train.sh` sets **25**. Unguided policy `log_every` **10**, but `train_segment` only logs a line (no memory query, no layer-grad walk). |
| `--runtime-metrics` / `--memory-timeline` | Extra `mx.eval` timing; `param_global_norm` syncs **per tensor**. `guide.md` pitfall table: leave them off for max tok/s. | Off |
| Traces | `trace_every` default is 10% of `total_steps`. Quarterly steps force full traces (`_force_quarter_traces`). | Off unless flags / quarter |
| `--val-every` | Up to **8** val forwards (`training/eval.py` `DEFAULT_VAL_MAX_BATCHES`) | **0** in `train.py` |
| `--checkpoint-every` | `sync_host_weights` + npz, then `training/probe.py` `run_probe`, which **reloads the checkpoint from disk** and runs another forward+CE | `min(1000, steps_per_epoch)` in `train.py`. c512 script: **10000**. |
| Quarters (25/50/75/100% of `--run-budget`) | Val, generate probe (default 256 new tokens), quarter save | On unless `--no-generate-probe` |
| Unguided segment end | `sync_host_weights` + `save_checkpoint` every `eval_every`, then `run_eval_suite` | TinyStories policy: **500** steps. `probe_mode=tinystories` is val CE only (no cabinet teacher-forced anchors). `probe_every` **2000** still runs a generate probe. |

Dropout in config is a no-op. `train.py` warns when `dropout_prob > 0`: “dropout is not implemented in GPU kernels.” Presets set it to 0. `guide.md` says the same.

---

## Measured or documented step-time signals

There is no checked-in M3 timing. Use these as the repo’s own signals:

- **`guide.md` (GT 730, not this Air):** `output/checkpoints/sub1m_start`, C=128, L=4, T=128, batch=8, accum=2: **~790 ms/step, ~2590 tok/s, ~100 MB** process VRAM. Same page: doubling **C** is ~4× the block GEMMs; **L** is linear; T=128→256 is ~1.5–2.5× per step at T≤256 because attention is still the smaller term; accum “trades wall-clock per step for VRAM, not for more tokens per second.”
- **`guide.md` scaling sentence:** VRAM is dominated by activations `~ B × T × C × L`, which is why preset 3 drops micro-batch 8→4 and raises accum 2→4.
- **`README.md` / `CHANGELOG.md` 0.0.5:** stream is **2–4×** fewer tok/s than resident L=6. Fanless GEMMs thermal-throttle; do not expect a short bench to hold for hours (port plan, same files).
- **`bench_step.py`:** live GPU step, model C=64, L=2, T=64, B=4, 2 warmup + 8 steps, prints `avg_step_ms` and tok/s. No result file in the tree. Too small to represent c256/c512, and it is the right shape for seeing Python/launch overhead.
- **`bench_profile.py`:** one step split into `forward_batch` / `backward_batch` / host `AdamW` / `sync_device`, model C=128, L=6, T=128, B=8. Host backward and host Adam, so the percentages are a different path from `train.py`.
- **`bench_mlp_fusion.py`:** no stored result. Both sides are composed MLX ops, so a “2b faster” print would not mean a fused kernel exists.
- **Planner, this tree, not a timed run.** `plan_train` with V=6102 and ~12 C² per block (the formula `guide.md` cites from `setup/model_config.py`):

| Recipe | Request | Estimator |
|---|---|---|
| `english_tinystories_c256_l6` | B=4 accum=4 T=256 | **531 MB**, resident, no checkpoint, no fp16, “no changes (already fits)”. Activations ~331 MB. |
| `english_tinystories_c512_l6` | B=8 accum=16 **or** accum=4, T=256 | **~1711 MB** vs usable **1741 MB**. Activations ~1094 MB. Still “fits” in this rough count, with little slack. Accum 4 vs 16 does **not** change the estimate (one extra grad buffer either way). |
| `story_sub1m` / `tiny_stories` / `chat_5m` | preset batches | 158–269 MB, resident, no autoscale actions. |

If a real c512 start prints `ckpt=1` or `layers=stream`, step time jumps for the reasons above. The `[memory]` line is the check (`README.md` example).

`guide.md` also says `--grad-checkpoint` is a VRAM lever and “adds recompute”; leave it off when measuring tok/s (~72 MB at BiggerTest shapes on the old card).

---

## B. Bottlenecks and waste

### What likely dominates step time

Ranked for the English recipes (C=256–512, L=6, T=256), which are the runs that care about wall-clock. Confidence is **medium** on the M3 split because the only millisecond number in-repo is the GT 730 line; the ordering follows this tree’s op list plus that scaling note.

1. **Block GEMMs (QKV×3, attn out, MLP expand, MLP contract) and the tied `lm_head`.** `guide.md`: width is the expensive axis (`Θ(B T C²)` per layer). `lm_head` is `Θ(B T C V)`. At C=256 and the TinyStories vocab (~6102 in `setup/english_tinystories_c512_l6/README.md`), `V/(12 C) ≈ 2`, so the head is on the order of two blocks of parameters. At C=512 it is about one block. Attention `Θ(B T² C)` is the smaller term at T≤256, per the same guide.
2. **Per-layer `mx.eval` plus a fresh Python graph every step.** No `mx.compile`. The forward realizes `h` after every layer (`gpt.py` ~307), so MLX cannot keep a cross-layer graph. RoPE tables and the causal mask are rebuilt every call. This is a large fraction on the toy bench shape and a real tax on L=6 even when GEMMs dominate.
3. **Micro-batch count.** c512 JSON asks for **16** forwards+backwards per optimizer step. c256 and the story presets use **4** (sub1m uses 2). `guide.md`: accum multiplies step wall-clock and does not raise tok/s.
4. **Streaming or grad-checkpoint, if the `[memory]` line turns them on.** Documented 2–4× for stream (recompute + load/unload + host grads and moments). Checkpoint recomputes attn and MLP (parity says the grads match; the time does not).
5. **Mandatory host syncs that are small next to (1) but never optional:** CE eval, grad-norm eval, final Adam `mx.eval`.
6. **Already cheap when left at the defaults:** val (`val_every=0`), runtime metrics (off), host logits (skipped). Checkpoint+`run_probe` and quarterly generate are large **when they fire**, and they are amortized at 1k–10k steps on `train.py`. They are **not** amortized on the unguided kernel (below).

Already in decent shape for step time, so not the first place to “optimize” again:

- GPU CE without host logits (`guide.md`, `train.py`).
- Device Adam for the resident path; host weight copy only at save.
- QKV split and ScratchPool / lifetime allocator (memory, not speed).
- Prebuilt TinyStories memmap (`training/tinystories_tokens.py`) so BPE is off the step.
- Preset micro-batches that the estimator already fits without checkpoint or stream (c256, sub1m, tiny_stories, chat_5m).

### What wastes steps (progress per update)

1. **The c512 files describe two different experiments.** The README and the JSON `metadata.description` are the long pass: accum 4, warmup 1000, 8192 tokens/step, ≈188,444 steps per epoch. The hyperparameters block is the short-budget hardness search: accum **16**, warmup **600**, and (on disk) batch **8**. The 2000-step result that matches a ~400k-step epoch was **batch 4, accum 16**. Those are not the same step. `train.sh` reads the file, so the next scripted launch is batch 8 unless `--batch-size 4` is passed. Keep the README as the epoch baseline; write the winning short-run tuple into the hyperparameters when that search is done, so a later 2000-step launch cannot silently become 32,768 tokens/step.
2. **Cosine is tied to `total_steps`.** `AdamWGPU.current_lr`: linear warmup, then cosine down to `min_lr_ratio × base` (presets **0.1**, c512 JSON **0.05**). On a 2000-step run, warmup 600 then a decay to 6e-5 is the hardness schedule. The waste appears only if that process is continued past 2000 while `total_steps` stays 2000: the extra steps sit on the floor. A 4k or 8k hard-step follow-up should set `--steps` to that new length. Flat `1e-5` is still the documented bad schedule (`guide.md`).
3. **Unguided TinyStories will not stop early.** `setup/unguided_tinystories_policy.json`: `max_steps` 50000, `early_stop_patience` **100000**. `decide()` stops after that many evals without a new best val (`training/unguided/decide.py`; default in code is 4, the policy overrides it). 50k steps / `eval_every` 500 = 100 evals, so patience never fires. You always pay the full budget. `loss_spike_ratio` 2.0 can **abort** the run instead.
4. **Unguided overhead between steps.** Every 500 steps: full weight download + checkpoint + up to 8 val forwards. Every 2000 steps: generate probe (`probe_every`). `make_train_args` also sets `--checkpoint-every` to `eval_every`, but `train_segment` does not call `run_probe` (the disk reload). The save itself is still on the segment boundary. Quicktest policy `eval_every` is **4** (`setup/unguided_quicktest_policy.json`) — fine for a smoke, bad if copied onto a real run.
5. **Resume throws away Adam moments.** `train.py` prints that `m`/`v` start at zero while `t` continues, and that early loss/grad spikes are expected. c512 README says leave `--reset-lr-schedule` off. Each resume pays a noisy stretch of steps. Checkpoints do not store moments (“intentional … to conserve memory”).
6. **Window overlap and cross-document cuts.** Stride 64 on T=128 and stride 128 on T=256 keep **50% overlap**. `guide.md`: stride 1 is the old “slow epoch” (more windows, more correlated updates). Stories are joined with spaces (`tokenizer/bpe.py` `encode_corpus`: “joined by single spaces”) and then cut into windows with no boundary token and no loss mask (`training/dataset.py`). Windows train on the seam between stories. That spends gradient on a transition the model should not learn. It is still the right call versus stride 1; it is not document packing.
7. **Chat objective.** `guide.md`: “Training loss is still next-token on the sliding window (User: tokens are not masked); packing-and-masking assistant-only loss would need a different example layout.” Cabinet quality is dominated by the **mix and the repeat count**, not by the optimizer: `CHANGELOG.md` 0.0.6 — ~300 repeats of a few dozen Q&As recite; 20 repeats of a 300-topic slice in 300 steps does not. `README.md`: dirty multi-answer mixes fail; fair one-answer mixes stick; do not `--resume` across vocab, arch, or mix.
8. **Clip and weight decay are fixed, not searched.** Every recipe uses global clip **1.0**, betas 0.9/0.999. Weight decay is 0.01 on story presets and c256, **0.02** on c512. No other optimizer exists. Dropout cannot regularize. Host and GPU Adam both decay the **post-update** weight (`params -= update; params -= lr*wd*params`). At these LRs that is a tiny departure from textbook decoupled AdamW and it matches between the two implementations. Not a step-count lever.
9. **The light recipes are noisy updates; the 2000-step c512 run is the large one.** Preset 3: 4×4×128 = **2048** tokens/step. c256: 4×4×256 = **4096**. Sub1m: 8×2×128 = **2048**. The reported c512 test: 4×16×256 = **16,384**. The file on disk at batch 8 is **32,768**. The early probe says 16,384-token steps at C=512 moved quality as far in 2000 steps as ~400k steps of the 4,096-token recipe. That is the steps-to-quality result. It does not make the hard step cheap: accum still multiplies wall-clock per step (`guide.md`).
10. **`num_epochs` counts micro-batches.** `train.py`: if `--steps` is omitted, `total_steps = epochs * dataset.num_batches()`, and `num_batches` is windows/`batch_size`, not windows/`(batch×accum)`. The loop increments `global_step` only on optimizer steps, and each of those consumes `grad_accum` micro-batches. An “epoch” therefore walks the data about `grad_accum` times. The English scripts pass `--steps`, so they are not hit. A wizard run that answers in epochs is.

---

## C. Options

Impact ratings are relative to a **resident** c256/c512 step with metrics off. “Already present” means the knob or the gap is in this tree.

### 1. Stay resident; treat stream and grad-checkpoint as memory fallbacks

- **What:** Read the `[memory]` line. `layers=resident`, `ckpt=0`, `fp16=0` is the fast path. Stream and checkpoint are how the controller fits L=32–48 or a shape that misses 2 GB (`training/memory_controller.py` order: eval_per_layer → checkpoint → fp16 storage → halve B and raise accum → stream → shrink T).
- **Connects:** `README.md` knobs list, `CHANGELOG.md` 0.0.4–0.0.5, `--no-layer-stream`, `--no-grad-checkpoint`, `--no-autoscale`.
- **Step time:** High if a run is silently streaming or checkpointing (documented 2–4× for stream; checkpoint adds a second attn/MLP forward per layer). None if the line already says “no changes”.
- **Steps to quality:** Unchanged when the math matches (checkpoint parity test). Shrink-T **does** change the task.
- **Confidence:** High on the cost of stream. The batch-4 accum-16 c512 estimate is ~1107 MB and fits. The on-disk batch-8 estimate is ~1711 MB of 1741 MB usable, so that variant is the one that can still flip to checkpoint or stream.
- **Effort:** Config / flag. No code.
- **Risk:** `--no-autoscale` or `--no-layer-stream` refuses or cuts T instead of streaming.
- **Present:** Yes.

### 2. Keep the hard 2000-step recipe, and name it separately from the epoch baseline

- **What:** The quality result supports fewer, heavier steps: C=512, batch 4, accum 16, 2000 steps, warmup 600, base LR 1.2e-3, `min_lr_ratio` 0.05. The README’s accum 4 / 8192-token / ~188k-step pass is the baseline that took ~400k light steps to a similar probe. When the search is finished, put batch 4 (not the on-disk batch 8) in the hyperparameters and leave the README numbers labeled as the long pass.
- **Step time:** A hard step is slower. Versus c256 (accum 4, C=256, same micro-batch 4) the guide’s scaling is about 4× from width and 4× from accum, ~16× per optimizer step, GEMM-bound, unmeasured on this Air. Versus the on-disk batch-8 config, batch 4 is about half the micro-batch work and ~1107 MB instead of ~1711 MB, so it stays resident with room.
- **Steps to quality:** This is the lever that just moved. 2000 hard steps ≈ 2% of a pass and matched a full light epoch, with misses left. The next quality experiment is more steps at this same hardness (stretch the cosine), or one more hardness bump to batch 8 / 32,768 tokens/step, which is already what `train.sh` will do if batch is left at 8.
- **Confidence:** High on the token counts and on the file still saying batch 8. The quality match is the run report; this tree has no probe log for it. Medium on the ~12× wall-clock estimate (guide scaling × 2000 vs ~400k).
- **Effort:** Flags or one config edit. No optimizer code.
- **Risk:** Launching `train.sh` and comparing it to the batch-4 result. Continuing past 2000 on a cosine that already hit 6e-5.
- **Present:** Accum 16, warmup 600, and LR 1.2e-3 are in the JSON. Batch 4 is the reported run, not the committed `batch_size`.

### 3. `mx.compile` on a closed resident step

- **What:** The port plan and `model/mlx/graph.py` defer compiling the train step. Today every step rebuilds a Python graph and `mx.eval`s the residual L times, plus CE, clip, and Adam.
- **Step time:** The largest **unmeasured** kernel-launch win in the roadmap, especially at L=6 where many small kernels (norms, RoPE, bias, split QKV) sit beside the GEMMs. Will not change FLOPs.
- **Steps to quality:** None, if numerics stay inside the parity tolerances (`rtol=1e-4`, `atol=1e-5` in the port plan).
- **Confidence:** Medium. The repo names it and does not time it. MLX compile plus in-graph `mx.eval`, streaming load/unload, and data-dependent shapes are the usual failure modes. A closed step wants fixed `B,T` and the resident path.
- **Effort:** Large. The unconditional `eval_for_host(h_d)` in `forward_batch` has to be inside the compiled region or removed on the resident path, or the compile breaks into L pieces. Parity on `tests/parity/test_step.py` and `test_modern_step.py` is the gate. Out of scope for v1 on purpose.
- **Risk:** Peak memory moves (lazy graph holds more), which is why `eval_per_layer` exists. The 2 GB cap is the constraint. Compile scratch is why headroom and `PEAK_TRANSIENT_BYTES` exist.
- **Present:** Stub only.

### 4. Stop realizing the residual after every resident layer

- **What:** `eval_for_host(h_d)` runs for every layer even when `config.eval_per_layer` is false (`gpt.py` ~307; streaming evals again at ~300). `eval_per_layer` was meant to be the memory knob (`README.md` item 1), not an always-on graph break.
- **Step time:** Medium on resident L=6 if MLX can fuse or overlap across blocks once the evals are gone. Prerequisite for option 3 more than a win by itself. Peak activation bytes go up; the controller’s first autoscale action turns `eval_per_layer` back on if the estimate does not fit.
- **Steps to quality:** None.
- **Confidence:** Medium-high that the eval is unconditional. Medium that removing it is faster rather than only a memory regression.
- **Effort:** Small experiment, must re-check the 2 GB abort (`model/mlx/env.py`).
- **Risk:** The lazy graph retains all L blocks. That is the failure `eval_per_layer` was added to prevent (`CHANGELOG.md` 0.0.4).
- **Present:** The extra eval is present. The flag does not gate it.

### 5. Cache RoPE and the causal mask; keep loss and grad-norm on device until a log step

- **What:** `rope_apply_inplace` rebuilds frequencies on the host every apply (forward and backward, Q and K). `_causal_mask` allocates every attention. `cross_entropy` and `grad_global_norm_sq` sync every micro-batch / step so a Python `float` can be logged or tested.
- **Step time:** Small-to-medium beside GEMMs at C=512; more visible on `bench_step.py`’s tiny model. Removing the CE sync until `log_every` also removes a full `dlogits` eval barrier (the tensor still has to be computed for backward).
- **Steps to quality:** None.
- **Confidence:** High that the work is repeated. Low-medium on the fraction of step time.
- **Effort:** Small, localized in `ops.py`.
- **Risk:** A stale mask if T changes mid-run (training T is fixed). Device loss must still be finite-checked often enough to catch NaNs (unguided aborts on non-finite loss).
- **Present:** Not cached. `bench_profile.py` is the place to see whether optimizer/sync is even visible; it currently profiles the host Adam path.

### 6. One QKV GEMM when the `[B·T, 3C]` buffer fits

- **What:** The live path launches three GEMMs so parity and the 2 GB plan never depend on a fused 3C buffer (`ops.py` docstring, port plan). At c256, B=4, T=256, 3C=768, that buffer is `4*256*768*4 ≈ 3 MB`. The memory win of the split is real at large B,T,C; at the English micro-batches it is small next to the ~300–1100 MB activation estimate.
- **Step time:** Low-to-medium (3 launches → 1, same FLOPs). Not a fusion of attention.
- **Steps to quality:** None if grads match.
- **Confidence:** Medium. No bench compares split vs packed QKV in this tree.
- **Effort:** Medium. A second path must not become the only path the parity tests hit (the port plan is explicit).
- **Risk:** Forgetting the buffer in the planner and tripping the 2 GB cap on a larger B.
- **Present:** Split path only. `fused_causal_attention_from_qkv` is the split.

### 7. Real MLP fusion, `mx.fast`, or AMP

- **What:** `bench_mlp_fusion.py` cannot justify a kernel: `fused_mlp_row` is the unfused pair. The port plan bans `mx.fast.scaled_dot_product_attention` and `mx.fast.rms_norm` on the train path because the VJPs are hand-written. `guide.md`: “Expecting bf16 / FlashAttention / AMP GEMM on GT 730 — out of scope.” FP16 in this tree is **storage**.
- **FP16 storage** (`model/mlx/fp16_storage.py`): compute stays FP32; caches are cast down and expanded again in `backward_batch_gpu`. Skipped when grad-checkpoint or streaming is on (`gpt.py`). Autoscale turns it on only after checkpointing. Cast traffic can make the step slower. It is a memory knob (estimator uses a 0.72 factor).
- **Step time:** Unknown, and the current fusion bench will not answer it. FP16 storage is more likely a loss than a win.
- **Steps to quality:** AMP would change noise and dynamic range; nothing here implements it. FP16 storage is supposed to match FP32 math after the round trip (casts can still move parity).
- **Confidence:** High that fusion/AMP are absent. Low that `mx.fast` is a safe speedup without abandoning explicit VJPs.
- **Effort:** Large if VJPs must be re-derived; a rewrite if the project switches to `mx.value_and_grad` (the port plan forbids that on the live path).
- **Present:** Storage casts and a misleading bench. No train-time quantization.

### 8. Optimizer, LR, clip, weight decay

- **What exists:** AdamW only. Warmup + cosine to `min_lr_ratio`. Clip 1.0. WD 0.01 or 0.02. Betas fixed.
- **Step time:** Swapping Adam for SGD would shrink `m`/`v` bytes (2× params in the estimator: c512 optimizer ~168 MB) and the Adam kernel loop. It would not move GEMM time. Not present, and it usually **increases** steps to quality for this model class.
- **Steps to quality, in-repo levers:**
  - Do not use flat `1e-5` (`guide.md` pitfall). Presets are 5e-4 (sub1m) and 3e-4 (tiny_stories, chat_5m). c256 is **9e-4**. c512 JSON is **1.2e-3** with WD 0.02 and floor 0.05.
  - Keep warmup a small fraction of the run you will actually finish. 1000 warmup on a 2000-step smoke is the waste. 1000 on 50k is 2%.
  - `min_lr_ratio` 0.1 means the last part of a full budget runs at 10% of base. If probes are already good, more steps at the floor buy little. If they are not, stopping at the floor wastes the tail.
  - `--reset-lr-schedule` on resume restarts warmup/cosine over the **new** step budget and still does not restore `m`/`v`.
  - Clip 1.0 is untested against other values here. `guide.md` says watch `grad_norm` in the first few hundred steps when metrics are on. Leaving metrics on for a short prefix, then off, matches the pitfall table.
- **Confidence:** High for “the schedule you think you are running is `current_lr` given `total_steps`”. Medium for any new LR number; the repo already picked the presets.
- **Effort:** Flags (`--learning-rate`, `--warmup-steps`, `--min-lr-ratio`, `--gradient-clip`, `--weight-decay`). No new optimizer code required to try these.
- **Risk:** c512’s 1.2e-3 with WD 0.02 is already the aggressive end of this tree. Raising it further is how the unguided spike abort (`loss_spike_ratio` 2) fires.
- **Present:** Fully present. No Lion/Adafactor/8-bit Adam.

### 9. Batch, accumulation, context

- **What the guide already says:** same effective batch via smaller B and larger accum **increases ms/step** and holds tok/s. Use that when VRAM is the problem (`2×8` vs `4×4` in the chat 5M section).
- **To make each step shorter:** lower accum (c512 JSON 16 is the outlier) or lower T. T=256 vs 128 is ~1.5–2.5× step time and doubles tokens per sequence, so tok/s often holds (`guide.md`). Quality of long stories wants the longer window; the guide says extra T only helps if the text needs it.
- **To make each step more informative:** raise tokens per optimizer step (B or accum) until the `[memory]` line is still resident. c256 at 4096 tokens/step has room in the estimator (531 MB vs 1741). A higher micro-batch (not just accum) also reduces Python/eval overhead **per token**, which accum does not.
- **Confidence:** High on the accum vs tok/s relationship (documented). Medium on how much val loss improves per step between 4096 and 8192 tokens; not measured in-repo.
- **Effort:** Flags `--batch-size`, `--grad-accum`, `--max-len`. Re-read `[memory]` after.
- **Risk:** Autoscale may undo a larger B by raising accum or enabling checkpoint. `--no-autoscale` then refuses.
- **Present:** Yes.

### 10. Skip or amortize eval, logs, probes, checkpoints

- **train.py, already mostly right:** `val_every` default 0; metrics default off; `need_host_logits` false; c512 script checkpoints every 10k. Still, every checkpoint calls `run_probe`, which builds a **second** model from disk (`training/probe.py`). Quarters still generate.
- **Cheap flags:** `--no-layer-grads` (skips the per-layer norm walk on log steps), `--no-generate-probe` on a throughput run, `--log-every 100` instead of 25 or 10, keep `--runtime-metrics` off except a short warmup watch.
- **Unguided:** `eval_every` 500 with a save is the steady tax. `early_stop_patience` 100000 disables the stopper that `decide()` already has (code default 4 evals). `probe_every` 2000 is a generate, which is the right quality signal and a bad thing to run on a tok/s bench.
- **Step time:** Log lines are small. `run_probe` and generate probes dominate the steps they touch. A 10k checkpoint interval makes that a fraction of a long run; an unguided save every 500 steps does not.
- **Steps to quality:** Less eval does not teach the model faster. It stops you from noticing a plateau. The waste is the opposite policy: patience so large that a flat val loss still runs to `max_steps`.
- **Confidence:** High.
- **Effort:** Policy JSON / flags.
- **Risk:** `--no-generate-probe` plus a huge `eval_every` means you only notice a bad run at the end. Unguided spike abort still needs val loss.
- **Present:** All of these flags and the patience override exist.

### 11. Data, stride, masks

- **What is already done:** stride 64–128, token memmap, BPE once in `tools/prepare_tinystories.py`. Stride 1 is the documented slow epoch.
- **What is not done:** a boundary or EOS between stories; loss only on tokens that belong to the story; assistant-only loss for chat (`guide.md` says the layout would have to change). `prepare_tinystories.py` collapses whitespace and encodes each shard with `encode_corpus` (space-joined docs).
- **Step time:** A mask is a cheap extra kernel beside the GEMMs. Packing that **shortens** sequences would cut T and step time; packing that **fills** T with real tokens (instead of cross-story seams) keeps step time and improves the update.
- **Steps to quality:** Medium for story coherence if many windows currently straddle two stories (T=256, no separator). High for cabinet chat, and the changelog’s repeat-count result matters more than the mask: repeats and a clean one-answer mix are what made v2/v4 recite. `.cursor/plans/two-phase_generalization_be934b9b.plan.md` explicitly deferred token masking.
- **Confidence:** High that seams and unmasked `User:` tokens are real. Medium on how large the seam fraction is (depends on story length vs T=256).
- **Effort:** Medium (dataset + loss). The plan said not to do it in that pass.
- **Risk:** A mask that drops too many tokens makes the effective batch smaller and noisier.
- **Present:** Stride and memmap yes. Mask and document packing no.

### 12. Unguided / autotrainer policies that inflate the budget

- **TinyStories policy** (`setup/unguided_tinystories_policy.json`): 50k steps, patience 100000, remix disabled (`cabinet_exact_match_below: -1`, `after_steps: 999999`). `decide_next_step` in `training/unguided/decide.py` will still **say** extra steps are optional when generate-exact is already high; nothing in the loop stops on that verdict except the existing `decide()` actions. Mid-train probe writes a report and continues (`unguided_trainer.py`).
- **Fact runs:** repeat count and mix cleanliness dominate step count (`CHANGELOG.md` 0.0.6, `README.md` lessons). Remix abort (`ABORT_REMIX`) throws away the run when cabinet exact-match stays low; the TinyStories policy turns that off on purpose (`decide.py` text: remix on a teacher-forced 0.0 can abort a net that already generates).
- **`auto_train.py`:** same `train()` plus a short generate at the end (`c512` README). Not a per-step cost.
- **Step time:** the per-segment checkpoint and val, not the policy logic.
- **Steps to quality:** patience and `max_steps` are the direct levers. Probe reports are advice unless you stop.
- **Confidence:** High.
- **Effort:** Edit the policy. The stopper already exists.
- **Risk:** Patience of 4 evals (2000 steps at `eval_every` 500) can stop a 9e-4 run during warmup noise. Warmup is 1000 steps, so the first two evals are still in warmup. Patience should start after warmup or the window should ignore those points. The code does not do that today.
- **Present:** Yes.

### 13. Anything else the benches and plans already name

- **Thermal.** Port plan and `README.md`: fanless Air, lid open, a one-step bench will not hold for hours. This caps how much “ms/step” you can buy back with kernels once the chip is throttling.
- **Headroom vs batch.** Default 15% is ~307 MB left unused so compile/scratch fit. Compile is unused. Lowering `--memory-headroom` only changes a run that is **failing** the fit check. c256 (~531 MB) and the reported c512 batch-4 step (~1107 MB) already fit. The on-disk batch-8 c512 estimate sits on the usable edge, so headroom can be the difference between resident and autoscale for that variant only.
- **`add_block` in `ops.py`** round-trips through the host. Nothing in `model/` calls it. Dead on the train path.
- **Epoch accounting** (section B.10) inflates work when the wizard uses epochs with accum > 1.
- **Second model on every `run_probe`.** Easy to miss in a profile of “the step” because it sits after the step, inside the checkpoint branch.

---

## D. Priority

### Quick wins on step time

The hard step is supposed to be slow. These keep it from getting slower than the batch-4 accum-16 run, and they make each of those 16 micro-batches cheaper once you touch code.

1. **Stay on the batch-4 step that just matched the epoch probe.** Estimator ~1107 MB, resident. The file’s batch 8 is a different, tighter step (~1711 MB). Confirm `[memory] … ckpt=0 fp16=0 layers=resident` on whichever one you launch.
2. **Leave metrics, traces, and val off while comparing hardness.** `--no-layer-grads` if you log often. A 2000-step run that checkpoints every 1000 and reloads via `run_probe` spends a real fraction of its wall-clock outside the optimizer step. For this comparison, checkpoint at the end (or once).
3. **Do not turn on `--grad-checkpoint`, `--layer-stream`, or fp16 storage to “go faster”.** Stream is the documented 2–4× tok/s hit. Checkpoint recomputes every micro-batch, and a hard step has 16 of them.
4. **Kernel work pays 16× on this recipe.** Gating the per-layer `eval_for_host`, caching RoPE and the causal mask, then `mx.compile` of one resident micro-batch, each cut time inside every accum slice. Do that after the hardness comparison is stable. `bench_profile.py` still times host Adam. `bench_mlp_fusion.py` still compares two unfused paths.

Code experiments, after the flags, in this order:

1. Gate the per-layer `eval_for_host(h_d)` on the resident path (option 4), re-check the 2 GB line.
2. Cache RoPE and the causal mask; consider deferring the CE host read to `log_every` (option 5).
3. `mx.compile` of one resident step at fixed `B,T` (option 3), with parity on `tests/parity/test_modern_step.py`.
4. Packed QKV only if a bench at the real `B,T,C` beats the split (option 6).

Leave `mx.fast` / AMP / a second optimizer until the project is willing to drop hand-written VJPs. The port plan treats that as a different program.

### Fewer steps to quality

The 2000-step result is this list’s first item. The rest is how to spend the misses that are still there.

1. **Stay with harder steps.** C=512, batch 4, accum 16, 16,384 tokens/step, got epoch-level gains by step 2000 (~2% of a pass). Going back to accum 4 / ~400k light steps is the path that already plateaued at “reasonable, with misses.”
2. **Next run: more steps at the same hardness, with the cosine stretched.** Try 4,000 or 8,000 at batch 4, accum 16, and set `--steps` to that number so warmup stays ~600 (or ~10% of the new budget) and `min_lr_ratio` 0.05 is only reached at the end. Do not resume the 2000-step process with the old `total_steps`: LR is already at 6e-5, and resume also zeros Adam moments.
3. **One hardness bump, as a separate A/B, not as a silent `train.sh` default.** Batch 8, accum 16 is 32,768 tokens/step and ~1711 MB. It is the file on disk. Run it only labeled as “harder than the 2000-step winner,” and check the `[memory]` line for `ckpt=0` and `layers=resident`. If autoscale turns on checkpointing, the extra recompute lands on all 16 micro-batches and the comparison is no longer the same update.
4. **Write the winner down** when you stop searching: hyperparameters `batch_size` 4, `gradient_accumulation_steps` 16, `warmup_steps` 600, and a metadata line that the README’s 8192-token / ~188k-step numbers are the old epoch baseline.
5. **Misses that survive a few thousand hard steps** are the data/objective gaps in section C.11 (story seams, no loss mask), not a reason to return to a 4,096-token step. Stride 128 is already the less-wasteful window setting.
6. **Unguided 50k with patience 100000** is a different policy. It will not notice that quality arrived at step 2000. For this recipe, stop on the probe, or set patience in post-warmup evals. The code default is 4 evals; the TinyStories policy overrides it.

### What not to do first

- Retuning betas, inventing Lion, or turning on dropout in JSON. Dropout does nothing on GPU. There is no second optimizer.
- Expecting `bench_mlp_fusion.py` or `graph.py` to speed up the current step. Both describe work that was not carried onto MLX.
- Reverting accum to 4 to “match the README.” That is the ~400k-step epoch the 2000-step run is already beating on probes.
- Shrinking T or enabling stream to improve the loss curve. Those cut (or slow) the step without a better update.
- Reading the GT 730 **790 ms** figure as an M3 baseline. It is the only millisecond anchor in the repo, and `guide.md` labels the card.
