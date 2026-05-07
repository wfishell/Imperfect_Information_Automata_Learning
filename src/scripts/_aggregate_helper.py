"""
Shared aggregation helpers for outputs/eval_*.csv files.

Each eval_*.csv has rows of the form:
  <game-config>, strategy, params, vs_random, vs_greedy, vs_optimal, states, time_s, ...

Where <game-config> is one or more parameterization columns:
  - eval_minimax.csv : depth, seed
  - eval_dab.csv     : shape, oracle_depth
  - eval_nim.csv     : shape, oracle_depth
  - eval_ttt.csv     : shape, oracle_depth

Aggregation collapses the game-config dimension(s) so each (strategy, params)
appears once with metrics averaged across all configs/seeds present.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


METRIC_PREFIXES = ('vs_',)
AUX_NUMERIC     = ('states', 'time_s')


def _is_metric(col: str) -> bool:
    return col.startswith(METRIC_PREFIXES) or col in AUX_NUMERIC


def aggregate_eval(csv_path: Path,
                   collapse_params: bool = False,
                   keep_cols: list | None = None) -> pd.DataFrame:
    """
    Load an eval CSV and average metric columns across all dimensions
    other than (strategy, [params], [keep_cols]).

    Parameters
    ----------
    csv_path        : path to eval_*.csv
    collapse_params : if True, also collapse params (one row per strategy);
                      if False, group by (strategy, params)
    keep_cols       : extra columns to retain as group keys (e.g., ['depth']
                      for minimax to see per-depth aggregation)
    """
    df = pd.read_csv(csv_path)

    group_cols = ['strategy']
    if not collapse_params and 'params' in df.columns:
        group_cols.append('params')
    if keep_cols:
        group_cols.extend(c for c in keep_cols if c in df.columns)

    # NaN params should group together — coerce to a sentinel string
    if 'params' in df.columns:
        df['params'] = df['params'].fillna('-')

    metric_cols = [c for c in df.columns if _is_metric(c)]

    agg_dict = {c: 'mean' for c in metric_cols}
    grouped = df.groupby(group_cols, dropna=False, sort=False).agg(agg_dict)
    grouped['n'] = df.groupby(group_cols, dropna=False, sort=False).size()
    grouped = grouped.reset_index()

    # Sort: prefer rows with higher vs_optimal (or vs_random fallback)
    sort_keys = [c for c in ('vs_optimal', 'vs_greedy', 'vs_random') if c in grouped.columns]
    if sort_keys:
        grouped = grouped.sort_values(sort_keys[0], ascending=False, kind='stable')

    return grouped


def write_and_print(df: pd.DataFrame, out_path: Path, label: str) -> None:
    """Write the aggregated DataFrame to disk and print a summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f'=== {label} ===')
    with pd.option_context('display.max_columns', None,
                           'display.width', 160,
                           'display.float_format', '{:.4f}'.format):
        print(df.to_string(index=False))
    print(f'\nWrote {len(df)} rows to {out_path}')
