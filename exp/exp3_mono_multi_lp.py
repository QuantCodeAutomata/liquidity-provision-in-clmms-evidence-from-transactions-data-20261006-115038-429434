#!/usr/bin/env python3
"""
Experiment 3: Identify mono-LPs and multi-LPs and evaluate cross-pool participation
and performance.

Replicates Tables 5 and 6 and Figure 4 of the paper:
  Table 5 – Pool-level decomposition into mono-LPs vs multi-LPs.
  Table 6 – Performance of multi-LPs by number of pools they participate in.
  Figure 4 – Upset-style bar chart of LP counts per pool-combination intersection.

Paper targets:
  Multi-LPs       : ~10% of unique LPs across the network.
  PnL > 0 in ≥1 cross-pool (multi-LPs): ~29.5% consistently positive.

Outputs:
  results/table5_mono_multi.csv
  results/table6_multi_lp_performance.csv
  results/exp3_cross_pool_intersections.png
  results/exp3_summary.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAMPLE_END, SAMPLE_START
from src.lp_classification import (
    cross_pool_intersection,
    mono_multi_decomposition_table,
    multi_lp_performance_table,
)
from src.pipeline import run_pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"

PAPER_MULTI_LP_PCT          = 10.0   # ~10% of unique LPs are multi-LPs
PAPER_MULTI_CONSISTENT_PCT  = 29.5   # ~29.5% of multi-LPs consistently positive in ≥1 pool

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


def _load_lp_class(use_graph: bool) -> pd.DataFrame:
    """
    Load lp_class_df from cached CSV or run the pipeline.

    Returns lp_class_df with correct bool column dtypes.
    """
    lp_class_path = RESULTS_DIR / "lp_classification.csv"

    if lp_class_path.exists():
        logger.info("Loading lp_classification.csv from %s", RESULTS_DIR)
        lp_class_df = pd.read_csv(lp_class_path)
        lp_class_df = _fix_bool_cols(lp_class_df, ["pnl_positive", "consistently_successful"])
    else:
        logger.info("lp_classification.csv not found; running pipeline…")
        result      = run_pipeline(
            use_graph=use_graph,
            save_intermediates=True,
            results_dir=RESULTS_DIR,
        )
        lp_class_df = result["lp_class_df"]

    return lp_class_df


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(df.to_string(index=False))


# ── Plot ───────────────────────────────────────────────────────────────────────

def _plot_cross_pool_intersections(intersection_df: pd.DataFrame) -> None:
    """
    Upset-style horizontal bar chart of LP counts per pool-combination.

    Bars are sorted descending by LP count. Each bar represents a unique
    combination of pools that a set of multi-LPs participated in simultaneously.
    """
    if intersection_df.empty:
        logger.warning("No cross-pool intersection data – skipping plot.")
        return

    df = intersection_df.sort_values("lp_count", ascending=True)

    # Shorten long combination labels for readability
    def _shorten(combo: str) -> str:
        """Replace full pool names with abbreviated protocol labels."""
        replacements = {
            "uniswap_v3_005":          "UNI-V3 0.05%",
            "aerodrome_slipstream_005": "AERO 0.05%",
            "pancakeswap_v3_001":       "CAKE-V3 0.01%",
            "sushiswap_v3_001":         "SUSHI-V3 0.01%",
        }
        for long, short in replacements.items():
            combo = combo.replace(long, short)
        return combo

    labels = [_shorten(c) for c in df["pool_combination"]]
    counts = df["lp_count"].tolist()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(df) + 1.5)))

    bars = ax.barh(range(len(counts)), counts, color="steelblue", edgecolor="white", linewidth=0.4)

    # Value annotations
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of LPs", fontsize=11)
    ax.set_title(
        "Cross-Pool LP Participation (Figure 4 data)\n"
        "LP counts per pool-combination intersection (multi-LPs only)",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(right=max(counts) * 1.15)

    fig.tight_layout()
    out = RESULTS_DIR / "exp3_cross_pool_intersections.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Mono-LP / Multi-LP Analysis – Tables 5, 6 & Figure 4"
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Use The Graph API for data loading instead of local files.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Experiment 3: Mono-LP / Multi-LP Classification ===")

    # ── Load data ──────────────────────────────────────────────────────────────
    lp_class_df = _load_lp_class(args.use_graph)

    if lp_class_df.empty:
        logger.error("lp_class_df is empty – nothing to analyse.")
        sys.exit(1)

    logger.info("Loaded %d LP–pool records", len(lp_class_df))

    # ── Table 5: Mono / Multi decomposition ───────────────────────────────────
    table5 = mono_multi_decomposition_table(lp_class_df)
    _print_section("Table 5 – Mono-LP / Multi-LP Decomposition Per Pool", table5)

    # Extract network-level stats for comparison
    net_row = table5[table5["pool"] == "Network unique"]
    if not net_row.empty:
        obs_multi_pct = float(net_row["multi_lp_pct"].iloc[0])
        n_unique      = int(net_row["total_lp"].iloc[0])
        n_multi       = int(net_row["multi_lp_count"].iloc[0])
        n_mono        = int(net_row["mono_lp_count"].iloc[0])
        print(f"\n  Multi-LP share: {obs_multi_pct:.2f}%  (paper target: ~{PAPER_MULTI_LP_PCT:.1f}%)")
        if abs(obs_multi_pct - PAPER_MULTI_LP_PCT) > 5.0:
            logger.warning(
                "Multi-LP share %.2f%% deviates >5pp from paper target %.1f%%",
                obs_multi_pct, PAPER_MULTI_LP_PCT,
            )
    else:
        obs_multi_pct = float("nan")
        n_unique = n_multi = n_mono = 0

    table5.to_csv(RESULTS_DIR / "table5_mono_multi.csv", index=False)
    logger.info("Saved table5_mono_multi.csv")

    # ── Figure 4: Cross-pool intersections ────────────────────────────────────
    intersection_df = cross_pool_intersection(lp_class_df)
    _print_section("Figure 4 Data – Cross-Pool LP Intersection Counts", intersection_df)

    logger.info("Generating cross-pool intersection bar chart…")
    _plot_cross_pool_intersections(intersection_df)

    # ── Table 6: Multi-LP performance ─────────────────────────────────────────
    table6 = multi_lp_performance_table(lp_class_df)
    _print_section("Table 6 – Multi-LP Performance by Number of Pools", table6)

    # Extract total-multi-LP row for paper-target comparison
    total_multi_row = table6[table6["n_pools_in"].astype(str) == "Total multi-LP"]
    if not total_multi_row.empty:
        obs_consistent_pct = float(total_multi_row["consistent_pct"].iloc[0])
        print(
            f"\n  Multi-LP consistent success rate: {obs_consistent_pct:.2f}%"
            f"  (paper target: ~{PAPER_MULTI_CONSISTENT_PCT:.1f}%)"
        )
        if abs(obs_consistent_pct - PAPER_MULTI_CONSISTENT_PCT) > 7.0:
            logger.warning(
                "Multi-LP consistent pct %.2f%% deviates >7pp from paper target %.1f%%",
                obs_consistent_pct, PAPER_MULTI_CONSISTENT_PCT,
            )
    else:
        obs_consistent_pct = float("nan")

    table6.to_csv(RESULTS_DIR / "table6_multi_lp_performance.csv", index=False)
    logger.info("Saved table6_multi_lp_performance.csv")

    # ── Save exp3_summary.json ─────────────────────────────────────────────────
    intersection_records = intersection_df.to_dict(orient="records") if not intersection_df.empty else []

    summary = {
        "experiment": "exp3_mono_multi_lp",
        "sample_start": SAMPLE_START,
        "sample_end":   SAMPLE_END,
        "network": {
            "unique_lps":    n_unique,
            "mono_lp_count": n_mono,
            "multi_lp_count": n_multi,
            "multi_lp_pct":  round(obs_multi_pct, 4) if not (obs_multi_pct != obs_multi_pct) else None,
        },
        "multi_lp_consistent_pct": (
            round(obs_consistent_pct, 4)
            if not (obs_consistent_pct != obs_consistent_pct)
            else None
        ),
        "paper_targets": {
            "multi_lp_pct":         PAPER_MULTI_LP_PCT,
            "multi_consistent_pct": PAPER_MULTI_CONSISTENT_PCT,
        },
        "cross_pool_intersections": [
            {"pool_combination": r["pool_combination"], "lp_count": int(r["lp_count"])}
            for r in intersection_records
        ],
    }

    summary_path = RESULTS_DIR / "exp3_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved exp3_summary.json")

    logger.info("=== Experiment 3 complete ===")


if __name__ == "__main__":
    main()
