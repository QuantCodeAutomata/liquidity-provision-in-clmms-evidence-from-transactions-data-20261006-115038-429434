#!/usr/bin/env python3
"""
Experiment 5: Analyse profitable win-position types 3, 5, and 7 by range width
and path metrics.

Replicates Tables 8, 9, and 10 of the paper for consistently successful LPs,
grouped by position type (3, 5, 7) and capital-weighted average range-width bucket.

Key paper reference values (Section 4.3):
  Type 5 – Δ_start ≈ 47.8%, Δ ≈ 8%        (in-range win, dominant type)
  Type 7 – Δ_start ≈ 56.6%                  (cross-upper-boundary win)
  Type 3 – cross-lower-boundary win

Position types used:
  Type 3 : price crosses lower boundary upward (P_start < a < P_end ≤ b) → win
  Type 5 : price stays in range, moves up      (a ≤ P_start < P_end ≤ b) → win
  Type 7 : price crosses upper boundary upward (a ≤ P_start < b < P_end)  → win

Outputs:
  results/table8_type3_analysis.csv
  results/table9_type5_analysis.csv
  results/table10_type7_analysis.csv
  results/exp5_delta_comparison.png
  results/exp5_summary.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAMPLE_END, SAMPLE_START, WIDTH_BUCKETS
from src.metrics import win_type_analysis_table
from src.pipeline import run_pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"

WIN_POSITION_TYPES = [3, 5, 7]

# Paper target reference values (Section 4.3, capital-weighted averages)
PAPER_TARGETS = {
    5: {"delta_start_pct": 47.8, "delta_pct": 8.0},
    7: {"delta_start_pct": 56.6},
}

# Width bucket order for consistent axis labels
_BUCKET_ORDER = [b[2] for b in WIDTH_BUCKETS] + ["Overall"]

# Colours per position type
_TYPE_COLOURS = {3: "#e07b39", 5: "#2166ac", 7: "#228b22"}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fix_bool_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Restore bool dtype for columns that became strings after CSV round-trip."""
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower() == "true"
    return df


def _load_data(use_graph: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load positions_df and lp_class_df from cached CSVs, or run the pipeline.

    Returns (positions_df, lp_class_df).
    """
    positions_path = RESULTS_DIR / "positions.csv"
    lp_class_path  = RESULTS_DIR / "lp_classification.csv"

    if positions_path.exists() and lp_class_path.exists():
        logger.info("Loading pre-computed intermediates from %s", RESULTS_DIR)
        positions_df = pd.read_csv(positions_path)
        lp_class_df  = pd.read_csv(lp_class_path)
        lp_class_df  = _fix_bool_cols(lp_class_df, ["pnl_positive", "consistently_successful"])
    else:
        logger.info("Intermediates not found; running pipeline…")
        result       = run_pipeline(
            use_graph=use_graph,
            save_intermediates=True,
            results_dir=RESULTS_DIR,
        )
        positions_df = result["positions_df"]
        lp_class_df  = result["lp_class_df"]

    return positions_df, lp_class_df


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)
    print(df.to_string(index=False))


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN/inf."""
    try:
        f = float(val)
        return None if (f != f or abs(f) == float("inf")) else f  # NaN check: f != f
    except (TypeError, ValueError):
        return None


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_delta_comparison(analysis_df: pd.DataFrame) -> None:
    """
    Two-panel figure comparing Δ_start (%) and Δ (%) across position types
    (3, 5, 7) for each range-width bucket.

    Each panel shows grouped bars: one group per width bucket, three bars per
    group (one per position type). Only the 'Overall' and per-bucket rows are
    included; the 'Overall' bucket gets a distinct x-position.
    """
    if analysis_df.empty:
        logger.warning("Analysis table is empty – skipping delta comparison plot.")
        return

    # Collect buckets that appear in the data (excluding 'Overall')
    data_buckets = [
        b for b in _BUCKET_ORDER[:-1]
        if b in analysis_df["width_bucket"].values
    ]
    plot_buckets = data_buckets + ["Overall"]
    n_buckets = len(plot_buckets)
    n_types   = len(WIN_POSITION_TYPES)

    x       = np.arange(n_buckets)
    width   = 0.22
    offsets = np.linspace(-(n_types - 1) / 2, (n_types - 1) / 2, n_types) * width

    fig, (ax_ds, ax_d) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for ax, metric, ylabel, title_suffix in [
        (ax_ds, "delta_start_pct", "Δ_start (%)", "Δ_start (position at open relative to range)"),
        (ax_d,  "delta_pct",       "Δ (%)",        "Δ = Δ_end − Δ_start (price drift through range)"),
    ]:
        for i, ptype in enumerate(WIN_POSITION_TYPES):
            sub = analysis_df[analysis_df["position_type"] == ptype]
            vals = []
            for bucket in plot_buckets:
                row = sub[sub["width_bucket"] == bucket]
                vals.append(float(row[metric].iloc[0]) if not row.empty else float("nan"))

            bars = ax.bar(
                x + offsets[i],
                vals,
                width,
                label=f"Type {ptype}",
                color=_TYPE_COLOURS[ptype],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
            )

            # Annotate bars for the 'Overall' bucket
            overall_idx = plot_buckets.index("Overall")
            overall_val = vals[overall_idx]
            if not (overall_val != overall_val):  # not NaN
                ax.text(
                    x[overall_idx] + offsets[i],
                    overall_val + 0.5,
                    f"{overall_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_suffix, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(plot_buckets, fontsize=9, rotation=15, ha="right")
    ax_d.set_xlabel("Capital-Weighted Avg Range Width Bucket (USDC)", fontsize=11)

    fig.suptitle(
        "Win-Position Analysis: Δ_start and Δ by Type and Width Bucket\n"
        "(consistently successful LPs only, Tables 8–10)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "exp5_delta_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def _plot_metrics_by_type(analysis_df: pd.DataFrame) -> None:
    """
    Supplementary bar chart: r_cap_pct and avg_duration_sec for the 'Overall'
    bucket across types 3, 5, 7. Saved alongside the delta comparison.
    """
    if analysis_df.empty:
        return

    overall = analysis_df[analysis_df["width_bucket"] == "Overall"].copy()
    if overall.empty:
        return

    overall = overall.set_index("position_type").reindex(WIN_POSITION_TYPES)
    types   = [f"Type {t}" for t in WIN_POSITION_TYPES]
    x       = np.arange(len(types))

    fig, (ax_r, ax_dur) = plt.subplots(1, 2, figsize=(11, 5))

    # Capital return
    r_cap_vals = overall["r_cap_pct"].tolist()
    colours = [_TYPE_COLOURS[t] for t in WIN_POSITION_TYPES]
    ax_r.bar(x, r_cap_vals, color=colours, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(types, fontsize=11)
    ax_r.set_ylabel("Capital Return R_cap (%)", fontsize=11)
    ax_r.set_title("Capital Return (Overall bucket)", fontsize=11)
    ax_r.axhline(0, color="black", linewidth=0.8)
    ax_r.grid(axis="y", alpha=0.3)
    for xi, val in zip(x, r_cap_vals):
        if not (val != val):
            ax_r.text(xi, val + (0.01 * max(abs(v) for v in r_cap_vals if v == v) or 0.001),
                      f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    # Duration in hours
    dur_vals = [v / 3600 if not (v != v) else float("nan")
                for v in overall["avg_duration_sec"].tolist()]
    ax_dur.bar(x, dur_vals, color=colours, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_dur.set_xticks(x)
    ax_dur.set_xticklabels(types, fontsize=11)
    ax_dur.set_ylabel("Average Position Duration (hours)", fontsize=11)
    ax_dur.set_title("Average Duration (Overall bucket)", fontsize=11)
    ax_dur.grid(axis="y", alpha=0.3)
    for xi, val in zip(x, dur_vals):
        if not (val != val):
            ax_dur.text(xi, val * 1.02, f"{val:.1f}h", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Win-Position Metrics Summary (consistently successful LPs)",
        fontsize=12,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "exp5_metrics_by_type.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Win-Position Analysis – Tables 8, 9, 10"
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Use The Graph API for data loading instead of local files.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Experiment 5: Win-Position Type Analysis ===")
    logger.info("Position types analysed: %s", WIN_POSITION_TYPES)

    # ── Load data ──────────────────────────────────────────────────────────────
    positions_df, lp_class_df = _load_data(args.use_graph)

    if positions_df.empty or lp_class_df.empty:
        logger.error("Input data is empty – nothing to analyse.")
        sys.exit(1)

    logger.info("Loaded %d positions and %d LP records", len(positions_df), len(lp_class_df))

    n_consistent = int(lp_class_df["consistently_successful"].astype(bool).sum())
    logger.info("Consistently successful LPs: %d", n_consistent)

    # ── Compute win-type analysis table (Tables 8-10 combined) ────────────────
    logger.info("Computing win_type_analysis_table for types %s…", WIN_POSITION_TYPES)
    analysis_df = win_type_analysis_table(positions_df, lp_class_df, WIN_POSITION_TYPES)

    if analysis_df.empty:
        logger.warning("win_type_analysis_table returned an empty DataFrame.")
        logger.warning("This may happen if no consistently-successful LPs have positions "
                       "of types 3, 5, or 7 in the loaded data.")

    _print_section(
        "Combined Win-Type Analysis (Tables 8–10): Types 3, 5, 7 by Width Bucket",
        analysis_df,
    )

    # ── Validation against paper targets ─────────────────────────────────────
    print("\n--- Comparison to Paper Reference Values ---")
    for ptype, targets in PAPER_TARGETS.items():
        overall_row = analysis_df[
            (analysis_df["position_type"] == ptype) & (analysis_df["width_bucket"] == "Overall")
        ]
        if overall_row.empty:
            print(f"  Type {ptype}: no 'Overall' row found")
            continue
        for metric, paper_val in targets.items():
            obs_val = _safe_float(overall_row[metric].iloc[0])
            obs_str = f"{obs_val:.2f}%" if obs_val is not None else "N/A"
            print(f"  Type {ptype} {metric}: {obs_str}  (paper: ~{paper_val:.1f}%)")
            if obs_val is not None and abs(obs_val - paper_val) > 10.0:
                logger.warning(
                    "Type %d %s = %.2f%% deviates >10pp from paper ~%.1f%%",
                    ptype, metric, obs_val, paper_val,
                )

    # ── Save per-type tables ───────────────────────────────────────────────────
    table_paths = {
        3: ("table8_type3_analysis.csv",  8),
        5: ("table9_type5_analysis.csv",  9),
        7: ("table10_type7_analysis.csv", 10),
    }

    for ptype, (filename, table_num) in table_paths.items():
        sub = analysis_df[analysis_df["position_type"] == ptype].copy()
        _print_section(f"Table {table_num} – Type {ptype} Analysis by Width Bucket", sub)
        sub.to_csv(RESULTS_DIR / filename, index=False)
        logger.info("Saved %s", filename)

    # ── Plots ──────────────────────────────────────────────────────────────────
    logger.info("Generating delta comparison plot (Types 3, 5, 7)…")
    _plot_delta_comparison(analysis_df)

    logger.info("Generating per-type metrics summary plot…")
    _plot_metrics_by_type(analysis_df)

    # ── Save exp5_summary.json ─────────────────────────────────────────────────
    observed: dict = {}
    for ptype in WIN_POSITION_TYPES:
        overall_row = analysis_df[
            (analysis_df["position_type"] == ptype) & (analysis_df["width_bucket"] == "Overall")
        ]
        if overall_row.empty:
            observed[str(ptype)] = None
        else:
            row = overall_row.iloc[0]
            observed[str(ptype)] = {
                "n_lp":           int(row["n_lp"]) if not (row["n_lp"] != row["n_lp"]) else None,
                "n_tx":           int(row["n_tx"]) if not (row["n_tx"] != row["n_tx"]) else None,
                "delta_start_pct": _safe_float(row["delta_start_pct"]),
                "delta_pct":       _safe_float(row["delta_pct"]),
                "r_cap_pct":       _safe_float(row["r_cap_pct"]),
                "r_price_pct":     _safe_float(row["r_price_pct"]),
                "avg_duration_sec": _safe_float(row["avg_duration_sec"]),
            }

    summary = {
        "experiment": "exp5_win_position_analysis",
        "sample_start": SAMPLE_START,
        "sample_end":   SAMPLE_END,
        "position_types_analysed": WIN_POSITION_TYPES,
        "consistently_successful_lps": n_consistent,
        "observed": observed,
        "paper_targets": {
            str(ptype): targets for ptype, targets in PAPER_TARGETS.items()
        },
    }

    summary_path = RESULTS_DIR / "exp5_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved exp5_summary.json")

    logger.info("=== Experiment 5 complete ===")


if __name__ == "__main__":
    main()
