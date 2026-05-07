"""
Aggregate outputs/eval_ttt.csv across (shape, oracle_depth).

Default: one row per (strategy, params), averaged over shapes and oracle_depth.
  --keep-shape         : break out per shape
  --keep-oracle-depth  : break out per oracle_depth
  --collapse-params    : one row per strategy (averages across params too)

Usage:
    python -m src.scripts.aggregate_ttt
    python -m src.scripts.aggregate_ttt --keep-oracle-depth
"""

from __future__ import annotations
import argparse
from pathlib import Path

from src.scripts._aggregate_helper import aggregate_eval, write_and_print


CSV_IN  = Path(__file__).parents[2] / 'outputs' / 'eval_ttt.csv'
CSV_OUT = Path(__file__).parents[2] / 'outputs' / 'eval_ttt_aggregated.csv'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--keep-shape',        action='store_true')
    p.add_argument('--keep-oracle-depth', action='store_true')
    p.add_argument('--collapse-params',   action='store_true')
    args = p.parse_args()

    keep = []
    if args.keep_shape:        keep.append('shape')
    if args.keep_oracle_depth: keep.append('oracle_depth')

    df = aggregate_eval(CSV_IN, collapse_params=args.collapse_params,
                        keep_cols=keep or None)
    write_and_print(df, CSV_OUT, 'eval_ttt aggregated')


if __name__ == '__main__':
    main()
