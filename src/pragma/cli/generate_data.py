from __future__ import annotations

import argparse

from pragma.data.synthetic import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic PRAGMA banking histories.")
    parser.add_argument("--out-dir", default="data/synth")
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mysql-uri", default=None)
    parser.add_argument(
        "--mysql-if-exists",
        choices=["fail", "replace", "append"],
        default="replace",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_synthetic_dataset(
        args.out_dir,
        users=args.users,
        min_events=args.min_events,
        max_events=args.max_events,
        seed=args.seed,
        mysql_uri=args.mysql_uri,
        mysql_if_exists=args.mysql_if_exists,
    )
    print(
        f"generated users={summary.users} events={summary.events} "
        f"profiles={summary.profiles} out={summary.out_dir}"
    )
    if summary.mysql_uri:
        print(f"mysql export complete: {summary.mysql_uri}")


if __name__ == "__main__":
    main()
