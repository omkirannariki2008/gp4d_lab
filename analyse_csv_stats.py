"""
analyse_csv_stats.py
====================
Compute summary stats and sigma-band percentages for numeric CSV columns.

Usage
-----
  /path/to/.venv/bin/python analyse_csv_stats.py --csv /path/to/file.csv
  /path/to/.venv/bin/python analyse_csv_stats.py --csv /path/to/file.csv --cols logA,Q2_log_center
  /path/to/.venv/bin/python analyse_csv_stats.py --csv /path/to/file.csv --sigma-levels 4 --ddof 1
  /path/to/.venv/bin/python analyse_csv_stats.py --csv /path/to/file.csv --out report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _format_pct(count: int, total: int) -> str:
    if total == 0:
        return "0.000%"
    return f"{(100.0 * count / total):7.3f}%"


def _iter_sigma_labels(max_sigma: int) -> list[str]:
    labels = [f"<= mean-{max_sigma}sigma"]
    for k in range(max_sigma, 1, -1):
        labels.append(f"(mean-{k}sigma, mean-{k-1}sigma]")
    labels.append("(mean-1sigma, mean]")
    labels.append("(mean, mean+1sigma]")
    for k in range(1, max_sigma):
        labels.append(f"(mean+{k}sigma, mean+{k+1}sigma]")
    labels.append(f"> mean+{max_sigma}sigma")
    return labels


def _sigma_edges(mean: float, std: float, max_sigma: int) -> list[float]:
    edges = [float("-inf")]
    for k in range(max_sigma, 0, -1):
        edges.append(mean - k * std)
    edges.append(mean)
    for k in range(1, max_sigma + 1):
        edges.append(mean + k * std)
    edges.append(float("inf"))
    return edges


def _sigma_counts(vals: np.ndarray, mean: float, std: float, max_sigma: int) -> list[int]:
    edges = _sigma_edges(mean, std, max_sigma)
    counts = []
    for i in range(len(edges) - 1):
        if i == 0:
            mask = vals <= edges[i + 1]
        else:
            mask = (vals > edges[i]) & (vals <= edges[i + 1])
        counts.append(int(np.sum(mask)))
    return counts


def _print_column_report(
    col: str,
    vals: np.ndarray,
    ddof: int,
    max_sigma: int,
    floor_threshold: float | None,
    floor_col: str,
    out_lines: list[str],
) -> None:
    vals = vals[~np.isnan(vals)]
    n = len(vals)

    out_lines.append("=" * 80)
    out_lines.append(col)
    out_lines.append("-" * 80)

    if n == 0:
        out_lines.append("No numeric data.")
        return

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=ddof))
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    vmin, vmax = float(np.min(vals)), float(np.max(vals))

    out_lines.append(f"count : {n:,}")
    out_lines.append(f"mean  : {mean:.6f}")
    out_lines.append(f"std   : {std:.6f}")
    out_lines.append(f"min   : {vmin:.6f}")
    out_lines.append(f"q25   : {q25:.6f}")
    out_lines.append(f"median: {med:.6f}")
    out_lines.append(f"q75   : {q75:.6f}")
    out_lines.append(f"max   : {vmax:.6f}")

    if floor_threshold is not None and (not floor_col or col == floor_col):
        floor_count = int(np.sum(vals <= floor_threshold))
        out_lines.append(
            f"<= {floor_threshold:.6f}: {floor_count:10d}  {_format_pct(floor_count, n)}"
        )

    if std == 0.0:
        out_lines.append("All values are identical; sigma bands are degenerate.")
        return

    labels = _iter_sigma_labels(max_sigma)
    counts = _sigma_counts(vals, mean, std, max_sigma)

    out_lines.append("")
    out_lines.append("Sigma bands (counts and % of total):")
    for lab, c in zip(labels, counts):
        out_lines.append(f"  {lab:28s}  {c:10d}  {_format_pct(c, n)}")

    out_lines.append("")
    out_lines.append("Within symmetric bands:")
    for k in range(1, max_sigma + 1):
        within = int(np.sum((vals >= mean - k * std) & (vals <= mean + k * std)))
        out_lines.append(f"  +- {k}sigma : {within:10d}  {_format_pct(within, n)}")


def _get_numeric_columns(df: pd.DataFrame, cols: Iterable[str] | None) -> pd.DataFrame:
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        return df[list(cols)].select_dtypes(include=[np.number])
    return df.select_dtypes(include=[np.number])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute summary stats and sigma-band percentages for numeric CSV columns."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument(
        "--cols",
        default="",
        help="Comma-separated column list (default: all numeric columns)",
    )
    parser.add_argument(
        "--sigma-levels",
        type=int,
        default=3,
        help="Number of sigma levels to report (default: 3)",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Std dev ddof (0=population, 1=sample)",
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=-80.0,
        help="Report count at or below this value (default: -80.0)",
    )
    parser.add_argument(
        "--floor-col",
        default="logA",
        help="Column name for floor count (default: logA). Empty = all numeric columns",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to save the report (prints to stdout either way)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    num_df = _get_numeric_columns(df, cols if cols else None)

    out_lines: list[str] = []
    out_lines.append(f"Loaded: {csv_path}")
    out_lines.append(f"Rows: {len(df):,}  Cols: {len(df.columns)}")
    out_lines.append(
        f"Numeric columns ({len(num_df.columns)}): {', '.join(num_df.columns)}"
    )
    out_lines.append("Counts include duplicates (raw rows).")
    out_lines.append(f"Std uses ddof={args.ddof}.")
    out_lines.append(
        f"Floor threshold: {args.floor_threshold} (col: {args.floor_col or 'ALL'})"
    )
    out_lines.append("")

    for col in num_df.columns:
        _print_column_report(
            col,
            num_df[col].to_numpy(),
            ddof=args.ddof,
            max_sigma=args.sigma_levels,
            floor_threshold=args.floor_threshold,
            floor_col=args.floor_col,
            out_lines=out_lines,
        )
        out_lines.append("")

    report = "\n".join(out_lines).rstrip() + "\n"
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
