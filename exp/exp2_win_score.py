#!/usr/bin/env python3
"""
Experiment 2: Compute the win-score metric and classify LPs by success consistency.

Replicates Table 4 of the paper, which reports per-pool LP counts, the fraction
with terminal PnL > 0, and the fraction that are consistently successful
(PnL > 0 AND win-score ω > 0.5).

Paper targets (Table 4, aggregated across all four pools):
  PnL > 0                : ~15.7% of LPs
  Consistently successful: ~15.1% of LPs

Outputs:
  results/table4_lp_classification.csv
  results/exp2_win_score_distribution.png
  results/exp2_pnl_vs_win_score.png
  results/exp2_summary.json
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

from src.config import POOLS, SAMPLE_END, SAMPLE_START, WIN_SCORE_THRESHOLD
from src.pipeline import run_pipeline
from src.win_score import classify_lps, pool_classification_table

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"

PAPER_PNL_POS_PCT        = 15.7   # Table 4 target: % of LPs with PnL > 0
PAPER_CONSISTENT_PCT     = 15.1   # Table 4 target: % consistently successful

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


def _load_data(use_graph: bool) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Load positions_df and lp_class_df from cached CSVs, or run the pipeline.

    Returns (positions_df, lp_class_df, sample_start_ts, sample_end_ts).
    """
    positions_path = RESULTS_DIR / "positions.csv"
    lp_class_path  = RESULTS_DIR / "lp_classification.csv"

    sample_start_ts = int(pd.Timestamp(SAMPLE_START).timestamp())
    sample_end_ts   = int(pd.Timestamp(SAMPLE_END).timestamp())

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
        positions_df    = result["positions_df"]
        lp_class_df     = result["lp_class_df"]
        sample_start_ts = result["sample_start_ts"]
        sample_end_ts   = result["sample_end_ts"]

    return positions_df, lp_class_df, sample_start_ts, sample_end_ts


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(df.to_string(index=False))


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_win_score_histogram(lp_class_df: pd.DataFrame) -> None:
    """Histogram of win-score ω distribution across all LPs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    scores = lp_class_df["win_score"].dropna()
    ax.hist(scores, bins=50, edgecolor="white", linewidth=0.4, color="steelblue", alpha=0.85)
    ax.axvline(WIN_SCORE_THRESHOLD, color="crimson", linewidth=1.5, linestyle="--",
               label=f"ω threshold = {WIN_SCORE_THRESHOLD}")
    ax.axvline(scores.mean(), color="darkorange", linewidth=1.5, linestyle=":",
               label=f"Mean ω = {scores.mean():.3f}")

    ax.set_xlabel("Win-Score ω", fontsize=12)
    ax.set_ylabel("Number of LPs", fontsize=12)
    ax.set_title("Distribution of LP Win-Scores\n(all pools combined)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate consistent-success region
    ax.axvspan(WIN_SCORE_THRESHOLD, 1.0, alpha=0.08, color="green",
               label="Consistent (ω > 0.5)")

    fig.tight_layout()
    out = RESULTS_DIR / "exp2_win_score_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def _plot_pnl_vs_win_score(lp_class_df: pd.DataFrame) -> None:
    """Scatter plot of terminal PnL against win-score ω."""
    df = lp_class_df[["terminal_pnl", "win_score", "pnl_positive", "consistently_successful"]].dropna()

    # Separate into outcome groups for colour coding
    mask_consistent  = df["consistently_successful"].astype(bool)
    mask_pnl_only    = df["pnl_positive"].astype(bool) & ~mask_consistent
    mask_losing      = ~df["pnl_positive"].astype(bool)

    fig, ax = plt.subplots(figsize=(9, 6))

    common_kw = dict(s=8, alpha=0.45, linewidths=0)
    ax.scatter(df.loc[mask_losing,     "win_score"], df.loc[mask_losing,     "terminal_pnl"],
               color="steelblue",  label="PnL ≤ 0",             **common_kw)
    ax.scatter(df.loc[mask_pnl_only,   "win_score"], df.loc[mask_pnl_only,   "terminal_pnl"],
               color="goldenrod",  label="PnL > 0, ω ≤ 0.5",   **common_kw)
    ax.scatter(df.loc[mask_consistent, "win_score"], df.loc[mask_consistent, "terminal_pnl"],
               color="crimson",    label="Consistently successful", s=14, alpha=0.6, linewidths=0)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(WIN_SCORE_THRESHOLD, color="black", linewidth=0.8, linestyle="--",
               label=f"ω = {WIN_SCORE_THRESHOLD}")

    # Symmetric log scale for the PnL axis to handle extreme outliers
    ax.set_yscale("symlog", linthresh=1.0)

    ax.set_xlabel("Win-Score ω", fontsize=12)
    ax.set_ylabel("Terminal PnL (USDC, symlog scale)", fontsize=12)
    ax.set_title("Terminal PnL vs Win-Score ω\n(all LPs across all pools)", fontsize=13)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = RESULTS_DIR / "exp2_pnl_vs_win_score.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Win-Score LP Classification – Table 4"
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Use The Graph API for data loading instead of local files.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Experiment 2: Win-Score LP Classification ===")

    # ── Load data ──────────────────────────────────────────────────────────────
    positions_df, lp_class_df, sample_start_ts, sample_end_ts = _load_data(args.use_graph)

    if lp_class_df.empty:
        logger.error("lp_class_df is empty – nothing to analyse.")
        sys.exit(1)

    logger.info(
        "Loaded %d LPs across %d positions",
        len(lp_class_df),
        len(positions_df),
    )

    # ── Table 4: Pool-level LP classification ─────────────────────────────────
    table4 = pool_classification_table(lp_class_df)
    _print_section("Table 4 – LP Classification by Pool (Win-Score)", table4)

    # Locate total row for paper-target comparison
    total_row = table4[table4["pool"] == "Total"].iloc[0]
    obs_pnl_pct    = float(total_row["pnl_positive_pct"])
    obs_consist_pct = float(total_row["consistent_pct"])
    total_lp_count  = int(total_row["lp_count"])

    print("\n--- Comparison to Paper Table 4 Targets ---")
    print(f"  PnL > 0             : {obs_pnl_pct:.2f}%  (paper target ~{PAPER_PNL_POS_PCT:.1f}%)")
    print(f"  Consistently success: {obs_consist_pct:.2f}%  (paper target ~{PAPER_CONSISTENT_PCT:.1f}%)")

    deviation_pnl    = abs(obs_pnl_pct    - PAPER_PNL_POS_PCT)
    deviation_consist = abs(obs_consist_pct - PAPER_CONSISTENT_PCT)
    if deviation_pnl > 5.0:
        logger.warning("PnL>0 percentage %.2f%% deviates >5pp from paper target %.1f%%",
                       obs_pnl_pct, PAPER_PNL_POS_PCT)
    if deviation_consist > 5.0:
        logger.warning("Consistent pct %.2f%% deviates >5pp from paper target %.1f%%",
                       obs_consist_pct, PAPER_CONSISTENT_PCT)

    table4.to_csv(RESULTS_DIR / "table4_lp_classification.csv", index=False)
    logger.info("Saved table4_lp_classification.csv")

    # ── Win-score distribution statistics ─────────────────────────────────────
    scores = lp_class_df["win_score"].dropna()
    print("\n--- Win-Score Distribution Statistics ---")
    print(f"  Count  : {len(scores):,}")
    print(f"  Mean   : {scores.mean():.4f}")
    print(f"  Median : {scores.median():.4f}")
    print(f"  Std    : {scores.std():.4f}")
    print(f"  ω > {WIN_SCORE_THRESHOLD}: {(scores > WIN_SCORE_THRESHOLD).sum():,}  "
          f"({100.0 * (scores > WIN_SCORE_THRESHOLD).mean():.2f}%)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    logger.info("Generating win-score histogram…")
    _plot_win_score_histogram(lp_class_df)

    logger.info("Generating PnL vs win-score scatter plot…")
    _plot_pnl_vs_win_score(lp_class_df)

    # ── Save exp2_summary.json ─────────────────────────────────────────────────
    per_pool: dict = {}
    for row in table4[table4["pool"] != "Total"].itertuples(index=False):
        per_pool[row.pool] = {
            "lp_count":           int(row.lp_count),
            "pnl_positive_count": int(row.pnl_positive_count),
            "pnl_positive_pct":   round(float(row.pnl_positive_pct), 4),
            "consistent_count":   int(row.consistent_count),
            "consistent_pct":     round(float(row.consistent_pct), 4),
        }

    summary = {
        "experiment": "exp2_win_score",
        "sample_start": SAMPLE_START,
        "sample_end":   SAMPLE_END,
        "win_score_threshold": WIN_SCORE_THRESHOLD,
        "total_lps": total_lp_count,
        "observed": {
            "pnl_positive_pct":    round(obs_pnl_pct,    4),
            "consistent_pct":      round(obs_consist_pct, 4),
            "win_score_mean":      round(float(scores.mean()),   4),
            "win_score_median":    round(float(scores.median()), 4),
            "win_score_std":       round(float(scores.std()),    4),
        },
        "paper_targets": {
            "pnl_positive_pct": PAPER_PNL_POS_PCT,
            "consistent_pct":   PAPER_CONSISTENT_PCT,
        },
        "per_pool": per_pool,
    }

    summary_path = RESULTS_DIR / "exp2_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved exp2_summary.json")

    logger.info("=== Experiment 2 complete ===")


if __name__ == "__main__":
    main()
