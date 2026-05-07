"""
Aggregate outputs/eval_minimax.csv across (depth, seed).

Default: one row per (strategy, params), averaged over all depths and seeds.
  --keep-depth      : break out per-depth (one row per depth × strategy × params)
  --collapse-params : one row per strategy (averages across all params too)

Usage:
    python -m src.scripts.aggregate_minimax
    python -m src.scripts.aggregate_minimax --keep-depth
    python -m src.scripts.aggregate_minimax --collapse-params
"""

from __future__ import annotations
import argparse
from pathlib import Path

from src.scripts._aggregate_helper import aggregate_eval, write_and_print


CSV_IN  = Path(__file__).parents[2] / 'outputs' / 'eval_minimax.csv'
CSV_OUT = Path(__file__).parents[2] / 'outputs' / 'eval_minimax_aggregated.csv'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--keep-depth',      action='store_true')
    p.add_argument('--collapse-params', action='store_true')
    args = p.parse_args()

    keep = ['depth'] if args.keep_depth else None
    df = aggregate_eval(CSV_IN, collapse_params=args.collapse_params, keep_cols=keep)
    write_and_print(df, CSV_OUT, 'eval_minimax aggregated')


if __name__ == '__main__':
    main()
