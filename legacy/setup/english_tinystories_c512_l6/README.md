# english_tinystories_c512_l6

Width 512, 6 layers, 8 heads (head width 64), context 256. Residual scale on, RMSNorm, RoPE, tied embeddings. Recipe: `setup/english_tinystories_c512_l6_config.json`. Checkpoint: `output/checkpoints/english_tinystories_c512_l6`. Fresh weights. Corpus is `data/tinystories_packed`: the width-256 BPE merges plus an `<|endofstory|>` token, with each training window kept inside one story. Point `--resume` at this checkpoint only. Do not resume the c256 run or the 2000 smoke.

`train.sh` checks those shards, then calls `auto_train.py`. `generate.sh` calls `generate.py` on this checkpoint only.

| Item | Value |
|------|-------|
| Micro-batch | 8 |
| Accumulation | 4 |
| Effective batch | 32 |
| Tokens per optimizer step | 8,192 |
| One full pass | ≈188,444 optimizer steps |
| Peak learning rate | 0.0012 |
| Weight decay | 0.02 |
| Warmup | 1,000 |
| Min LR ratio | 0.05 |

`--steps` is how many optimizer steps to add. A fresh start begins at 0. A resume adds that many on top of the saved step. First chunk is 50,000 steps.

```text
setup/english_tinystories_c512_l6/train.sh              # 50000 steps from step 0
setup/english_tinystories_c512_l6/train.sh 20000        # another fresh length
setup/english_tinystories_c512_l6/train.sh --resume     # add 50000 to this checkpoint
setup/english_tinystories_c512_l6/train.sh --resume 100000
```

A second fresh start is refused once `weights.npz` exists. Resume is refused when it does not.

## `auto_train.py`

Trains, then writes a short smoke sample. `train.sh` expands to:

```text
./venv/bin/python auto_train.py \
  --config setup/english_tinystories_c512_l6_config.json \
  --checkpoint output/checkpoints/english_tinystories_c512_l6 \
  --steps 50000 \
  --seed 42 \
  --no-prompt \
  --log-every 25 \
  --checkpoint-every 10000
```

Resume adds `--resume` to that line. Leave `--reset-lr-schedule` off so the cosine keeps the saved step. Adam moments are not restored, so the first logs after a resume can bump.

Flags `train.sh` sets:

| Flag | This run | Meaning |
|------|----------|---------|
| `--config` | `setup/english_tinystories_c512_l6_config.json` | Recipe. Width 512, 6 layers, 8 heads, context 256, learning rate 0.0012, weight decay 0.02, batch 8, accum 4 (effective batch 32, 8192 tokens/step), stride 128, warmup 1000, `min_lr_ratio` 0.05 |
| `--checkpoint` | `output/checkpoints/english_tinystories_c512_l6` | Where weights, `state.json`, and `metrics.json` are written. The root file is overwritten every `--checkpoint-every` steps |
| `--steps` | `50000` | Optimizer steps to run. On `--resume`, added to the saved step |
| `--seed` | `42` | Init and window shuffle |
| `--no-prompt` | on | Skip the learning-rate / steps questions |
| `--log-every` | `25` | Progress line cadence (CLI default 100) |
| `--checkpoint-every` | `10000` | Save interval |
| `--resume` | only with `train.sh --resume` | Load this checkpoint and continue |

Smoke sample at the end of `auto_train.py` (not set by `train.sh`, so these defaults apply):

| Flag | Default | Meaning |
|------|---------|---------|
| `--prompt` | `the` | Seed text for the end-of-run sample |
| `--max-new-tokens` | `80` | Tokens in that sample |
| `--temperature` | `0.6` | Sampling temperature |
| `--top-k` | `10` | Keep the top K tokens |
| `--top-p` | `0.9` | Nucleus cutoff |
| `--stop` | off | Repeatable. Stop when this string appears in the new text |

Story checks use `generate.sh` below. The smoke sample is the short "the …" line at the end of the train log.

Optional flags. Omit them unless you mean to change the recipe:

| Flag | Default | Meaning |
|------|---------|---------|
| `--learning-rate` | config `0.0012` | Override the recipe |
| `--min-lr-ratio` | config `0.05` | Cosine floor as a fraction of the base rate |
| `--window-stride` | config `128` | Tokens between windows |
| `--val-every` | `0` | Val loss every N steps. `0` leaves it off |
| `--epochs` | config `1` | Used only when `--steps` is omitted |
| `--batch-size` | config `8` | Changes tokens per step and the ≈188,444 figure |
| `--grad-accum` | config `4` | Micro-batches per optimizer step. Alias `--gradient-accumulation-steps` |
| `--weight-decay` | config `0.02` | |
| `--warmup-steps` | config `1000` | |
| `--gradient-clip` | config `1.0` | |
| `--embedding-dim` `--num-heads` `--num-layers` `--max-len` | config | Architecture overrides. A different shape is a new checkpoint |
| `--run-budget` | this chunk's step total | Quarterly 25/50/75/100% milestones. Set it to `188444` if you want quarters across the full ≈188,444-step pass while chunking `--steps` |
| `--no-generate-probe` | off | Skip the mid-training probe text |
| `--generate-probe-prompt` | `once upon a` | Mid-training probe seed |
| `--generate-probe-tokens` | `256` | Mid-training probe length |
| `--runtime-metrics` | off | Extra grad / memory fields on `[train]` lines |
| `--memory-timeline` | off | ScratchPool JSONL under `output/logs/` |
| `--no-kv-cache` | off | Smoke sample recomputes every token |
| `--menu` | off | Interactive flag picker. `train.sh` does not use it |

## `generate.py`

One sample from the saved weights. `generate.sh` expands to:

```text
./venv/bin/python generate.py \
  --checkpoint output/checkpoints/english_tinystories_c512_l6 \
  --prompt "Once upon a time there was a brave little mouse named" \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --seed 42 \
  --no-prompt
```

```text
setup/english_tinystories_c512_l6/generate.sh
setup/english_tinystories_c512_l6/generate.sh "Once upon a time there was a little girl named Lily who found a"
```

The script refuses to run until `weights.npz` exists. Context is 256, so `--max-new-tokens` is shortened by the prompt length. Each run writes `output/logs/generate_english_tinystories_c512_l6_<timestamp>.log`.

| Flag | `generate.sh` | CLI default | Meaning |
|------|---------------|-------------|---------|
| `--checkpoint` | this run's dir | `output/checkpoints/run1` | Directory with `weights.npz` |
| `--prompt` | mouse opening, or the first argument | `the` | Seed text. Quote it in zsh |
| `--max-new-tokens` | `160` | `80` | New tokens. Clamped to 256 minus the prompt |
| `--temperature` | `0.8` | `0.8` | `0.6` is the tighter train-probe setting |
| `--seed` | `42` | `42` | Sampling seed |
| `--no-prompt` | on | off | Only matters with `--menu` |
| `--top-k` | unset | off | e.g. `10` |
| `--top-p` | unset | off | e.g. `0.9` |
| `--stop` | unset | off | Repeatable stop string |
| `--no-kv-cache` | unset | off | Full recompute each token |
| `--menu` | unset | off | Pick a checkpoint and sampling flags |
| `--models-dir` | unset | `output/checkpoints` | Scan root for `--menu` |
| `--verbose` `--trace-logits` `--trace-tokens` `--trace-neurons` | unset | off | Token and activation traces |

## Pastable

From the repo root. One line each.

```text
./venv/bin/python setup/english_tinystories_c512_l6/check_data.py
```

```text
./venv/bin/python auto_train.py --config setup/english_tinystories_c512_l6_config.json --checkpoint output/checkpoints/english_tinystories_c512_l6 --steps 50000 --seed 42 --no-prompt --log-every 25 --checkpoint-every 10000
```

```text
./venv/bin/python auto_train.py --config setup/english_tinystories_c512_l6_config.json --checkpoint output/checkpoints/english_tinystories_c512_l6 --resume --steps 50000 --seed 42 --no-prompt --log-every 25 --checkpoint-every 10000
```

```text
./venv/bin/python generate.py --checkpoint output/checkpoints/english_tinystories_c512_l6 --prompt "Once upon a time there was a brave little mouse named" --max-new-tokens 160 --temperature 0.8 --seed 42 --no-prompt
```

```text
./venv/bin/python generate.py --checkpoint output/checkpoints/english_tinystories_c512_l6 --prompt "Once upon a time there was a little girl named Lily who found a" --max-new-tokens 160 --temperature 0.8 --seed 42 --no-prompt
```

```text
./venv/bin/python generate.py --checkpoint output/checkpoints/english_tinystories_c512_l6 --prompt "Once upon a time there was a brave little mouse named" --max-new-tokens 160 --temperature 0.6 --top-k 10 --top-p 0.9 --seed 42 --no-prompt
```


**This is working cleanly.**

### Status at step 11

| Metric              | Value          | Assessment                  |
|---------------------|----------------|-----------------------------|
| Loss                | 5.18           | Healthy early drop          |
| Perplexity          | ~177           | Normal for this stage       |
| VRAM used           | 475 MB         | Very comfortable            |
| VRAM free           | 1573 MB        | Plenty of headroom          |
| Tokens/sec          | ~1550–1630     | Reasonable with streaming   |
| Layer strategy      | `stream`       | Active and effective        |
| Actual model width  | **1024**       | (head_dim=128)              |

### Important note

Your config name still says `c512`, but the actual model being trained is **width 1024** (8 heads × 128 head dim). That is a much larger model (~82 M parameters). Streaming is what is making it possible under the 2 GB budget.

### Verdict

- Batch 4 + accum 16 + layer streaming is stable.
- Memory has large headroom (you could probably try batch 8 later if you want).
- Loss is decreasing properly.
- You can let this 2000-step run finish.

This is the first configuration that has successfully trained a significantly wider model on your hardware. Good result.