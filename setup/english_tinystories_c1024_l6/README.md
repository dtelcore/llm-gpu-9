# english_tinystories_c1024_l6

v0.1.1 English recipe. Width 1024, 6 layers, 8 heads (head width 128), context 256. Residual scale on, RMSNorm, RoPE, tied embeddings, layer stream. Checkpoint: `output/checkpoints/english_tinystories_c1024_l6`. Fresh weights. Do not resume the 0.1.0 c256 run or the step-2000 smoke. Those weights are under `legacy/output/checkpoints/`.

| Item | Value |
|------|-------|
| Micro-batch | 4 |
| Accumulation | 16 |
| Tokens per optimizer step | 16,384 |
| Peak learning rate | 0.0012 |
| Weight decay | 0.02 |
| Warmup | 600 |
| Min LR ratio | 0.05 |
| First cosine length | 4,000 steps |

`--steps` is the cosine length on a fresh start, and an added step count on `--resume`. Resume restores the step counter and zeros Adam moments. Extending a finished 4,000-step cosine keeps later steps at the minimum learning rate. A longer schedule is a new checkpoint with a larger `--steps`.

## Data prep

Two corpora. `data/tinystories/` is the 6,000-merge BPE and the story text. `data/tinystories_packed/` is what this recipe trains: the same merges, one extra token `<|endofstory|>` (id 6102, vocab 6103), and a span file per shard. Training windows stay inside one story. Stories shorter than 256 tokens are padded, and the loss ignores the pad. Generation stops on the end token.

The text shards are already on disk after the first command. The second command does not download them again.

```text
./venv/bin/python tools/prepare_tinystories.py
./venv/bin/python tools/prepare_tinystories.py --pack-stories --skip-download \
  --text-dir data/tinystories/text --vocab data/tinystories/vocab.json
./venv/bin/python setup/english_tinystories_c1024_l6/check_data.py
```

`--pack-stories` refuses to overwrite `data/tinystories/` and writes `data/tinystories_packed/`. `--vocab` keeps the 0.1.0 token ids and appends the end token. Without `--vocab`, prepare trains a new BPE and the ids will not match this config.

`check_data.py` reads the packed manifest. It does not download or re-encode.

## Train

`train.sh` checks the packed corpus, then calls `auto_train.py`.

```text
setup/english_tinystories_c1024_l6/train.sh            # 4000 steps from step 0
setup/english_tinystories_c1024_l6/train.sh 8000       # another fresh length
setup/english_tinystories_c1024_l6/train.sh --resume   # add 4000 to this checkpoint
```

A second fresh start is refused once `weights.npz` exists.

Unguided policy: `setup/unguided_tinystories_policy.json` (`max_steps` 4000, `probe_mode` tinystories, `probe_every` 2000).

```text
./venv/bin/python unguided_trainer.py \
  --config setup/english_tinystories_c1024_l6_config.json \
  --policy setup/unguided_tinystories_policy.json \
  --dry-run
```

### `auto_train.py` flags this script sets

| Flag | Value | Role |
|------|-------|------|
| `--config` | `setup/english_tinystories_c1024_l6_config.json` | Architecture and optimizer |
| `--checkpoint` | `output/checkpoints/english_tinystories_c1024_l6` | Fresh run directory |
| `--steps` | `4000` | Cosine length. Added on top of the saved step when resuming |
| `--seed` | `42` | |
| `--no-prompt` | | No interactive menu |
| `--log-every` | `25` | |
| `--checkpoint-every` | `1000` | |
| `--resume` | off unless requested | Same checkpoint only |

Batch, accumulation, learning rate, warmup, and context come from the config. The script does not pass `--batch-size` or `--embedding-dim`.

## Generate

`generate.sh` calls `generate.py` on this checkpoint. Temperature 0.8, 160 new tokens, seed 42. Sampling stops when the model emits `<|endofstory|>`.

```text
setup/english_tinystories_c1024_l6/generate.sh
setup/english_tinystories_c1024_l6/generate.sh "Once upon a time there was a little girl named Lily who found a"
```

| Flag | Default here | Role |
|------|----------------|------|
| `--checkpoint` | this run | |
| `--prompt` | brave little mouse | Opening text |
| `--max-new-tokens` | `160` | Cap. The end token can stop sooner |
| `--temperature` | `0.8` | |
| `--seed` | `42` | |
| `--no-prompt` | | No menu |

## Pastable

```text
./venv/bin/python tools/prepare_tinystories.py
./venv/bin/python tools/prepare_tinystories.py --pack-stories --skip-download --text-dir data/tinystories/text --vocab data/tinystories/vocab.json
./venv/bin/python setup/english_tinystories_c1024_l6/check_data.py
./venv/bin/python auto_train.py --config setup/english_tinystories_c1024_l6_config.json --checkpoint output/checkpoints/english_tinystories_c1024_l6 --steps 4000 --seed 42 --no-prompt --log-every 25 --checkpoint-every 1000
./venv/bin/python generate.py --checkpoint output/checkpoints/english_tinystories_c1024_l6 --prompt "Once upon a time there was a brave little mouse named" --max-new-tokens 160 --temperature 0.8 --seed 42 --no-prompt
```
