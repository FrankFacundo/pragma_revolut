# PRAGMA Revolut Foundation Model

This repository is a clean PyTorch implementation of the PRAGMA paper design for
multi-source banking event histories. It includes:

- synthetic banking event generation inspired by `bank_lab`
- Parquet output and optional local MySQL export
- key-value-time tokenisation with categorical, numerical, and text value handling
- encoder-only PRAGMA backbone with profile, event, and history encoders
- masked-modelling pretraining
- downstream classification fine-tuning with optional LoRA
- embedding and prediction inference utilities

The implementation intentionally avoids Hugging Face training/model wrappers. It is
plain PyTorch and runs on CUDA, Apple MPS, or CPU.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,metrics]"

pragma-generate-data --out-dir data/synth --users 1000 --seed 7
pragma-build-tokenizer --data-dir data/synth --out data/tokenizer.json
pragma-pretrain --data-dir data/synth --tokenizer data/tokenizer.json --variant tiny --steps 50
pragma-finetune --data-dir data/synth --tokenizer data/tokenizer.json \
  --checkpoint runs/pretrain/checkpoints/model_last.safetensors \
  --task credit_default --variant tiny --steps 50 --lora
```

For MySQL export, start a local MySQL instance and pass a SQLAlchemy URI:

```bash
pragma-generate-data \
  --out-dir data/synth \
  --users 10000 \
  --mysql-uri "mysql+pymysql://root:password@127.0.0.1:3306/pragma"
```

The generator writes `users.parquet`, `events.parquet`, and `profiles.parquet`.
The MySQL export creates equivalent `pragma_users`, `pragma_events`, and
`pragma_profiles` tables.

## Model Variants

The paper variants are exposed as `s`, `m`, and `l`:

| Variant | Width | FFN | Profile | Event | History | Heads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s` | 192 | 768 | 1 | 5 | 2 | 3 |
| `m` | 512 | 2048 | 3 | 16 | 6 | 8 |
| `l` | 1024 | 4096 | 9 | 45 | 18 | 16 |

`tiny` is provided for local smoke tests.

## Repository Layout

```text
src/pragma/config.py          model and training configs
src/pragma/data/              synthetic data, tokenisation, datasets
src/pragma/model/             PRAGMA encoders, attention, LoRA, heads
src/pragma/train/             training loops, checkpoints, device utilities
src/pragma/cli/               console entry points
configs/                      runnable YAML examples
tests/                        smoke and contract tests
```

## Notes

This is not an official Revolut implementation and ships only synthetic data
generation. The architecture follows the public paper: typed key/value
embeddings, profile and event branches, history fusion, RoPE time encoding,
masked reconstruction, embedding probes, and LoRA fine-tuning.
