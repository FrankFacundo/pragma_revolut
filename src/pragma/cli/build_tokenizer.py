from __future__ import annotations

import argparse

from pragma.data.tokenizer import PragmaTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a PRAGMA tokenizer from generated Parquet data."
    )
    parser.add_argument("--data-dir", default="data/synth")
    parser.add_argument("--out", default="data/tokenizer.json")
    parser.add_argument("--numeric-bins", type=int, default=128)
    parser.add_argument("--categorical-threshold", type=int, default=20_000)
    parser.add_argument("--text-vocab-size", type=int, default=4096)
    parser.add_argument("--bpe-merges", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = PragmaTokenizer.fit(
        args.data_dir,
        numeric_bins=args.numeric_bins,
        categorical_threshold=args.categorical_threshold,
        text_vocab_size=args.text_vocab_size,
        bpe_merges=args.bpe_merges,
    )
    tokenizer.save(args.out)
    print(f"saved tokenizer vocab_size={tokenizer.vocab_size} path={args.out}")


if __name__ == "__main__":
    main()
