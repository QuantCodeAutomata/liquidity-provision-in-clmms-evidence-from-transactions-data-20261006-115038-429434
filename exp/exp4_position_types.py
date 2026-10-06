#!/usr/bin/env python3
"""
Experiment 4: Classify reconstructed positions into 15 types and analyse dominant
position structures.

Replicates Table 7 of the paper, which reports the share of transactions and
minted capital per position type across all pools.

Position type taxonomy (Section 2.3, Figure 3):
  Types 1–12 : directional win/loss types determined by the ordering of
               P_start, P_end relative to the range boundaries [a, b].
  Types 13–15: PnL-neutral (P_start ≈ P_end).

Paper target (Table 7):
  Types 5 + 6 together account for > 90% of transactions and > 90% of capital.
  (Type 5 = in-range win, Type 6 = in-range loss; dominant in CLMM pools.)

Outputs:
  results/table7_position_type_distribution.csv
  results/exp4_type_distribution.png
  results/exp4_summary.json
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

from src.config import SAMPLE_END, SAMPLE_START
from src.pipeline import run_pipeline
from src.position_types import position_type_distribution

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Paper target: Types 5+6 > 90% of tx and capital
PAPER_TYPES_5_6_TX_PCT  = 90.0
PAPER_TYPES_5_6_CAP_PCT = 90.0

# Colour map: win=blue, loss=red, neutral=grey
_TYPE_COLOURS = {
    1: "#4c9be8",  3: "#4c9be8",  5: "#2166ac",  7: "#4c9be8",  9: "#4c9be8",  11: "#4c9be8",
    2: "#e87c4c",  4: "#e87c4c",  6: "#d6604d",  8: "#e87c4c", 10: "#e87c4c", 12: "#e87c4c",
    13: "#999999", 14: "#bbbbbb", 15: "#cccccc",
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_positions(use_graph: bool) -> pd.DataFrame:
    """
    Load positions_df from cached CSV or run the pipeline.

    The positions CSV saved by exp1 includes the position_type column;
    if it is absent the pipeline is run to regenerate it.
    """
    positions_path = RESULTS_DIR / "positions.csv"

    if positions_path.exists():
        logger.info("Loading positions.csv from %s", RESULTS_DIR)
        df = pd.read_csv(positions_path)
        # Ensure position_type column is present (pipeline adds it)
        if "position_type" not in df.columns:
            logger.warning("position_type column missing – re-running pipeline to compute it.")
            return _run_and_save(use_graph)
        return df

    logger.info("positions.csv not found; running pipeline…")
    return _run_and_save(use_graph)


def _run_and_save(use_graph: bool) -> pd.DataFrame:
    result = run_pipeline(
        use_graph=use_graph,
        save_intermediates=True,
        results_dir=RESULTS_DIR,
    )
    return result["positions_df"]


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(df.to_string(index=False))


# ── Plot ───────────────────────────────────────────────────────────────────────

def _plot_type_distribution(dist_df: pd.DataFrame) -> None:
    """
    Side-by-side bar chart of transaction share (%) and capital share (%)
    for each of the 15 position types.
    """
    df = dist_df.copy()
    types   = df["position_type"].tolist()
    x       = np.arange(len(types))
    colours = [_TYPE_COLOURS.get(t, "#888888") for t in types]

    fig, (ax_tx, ax_cap) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # ── Transaction share ──────────────────────────────────────────────────────
    bars_tx = ax_tx.bar(x, df["tx_share_pct"], color=colours, edgecolor="white", linewidth=0.5)
    ax_tx.set_ylabel("Share of Transactions (%)", fontsize=11)
    ax_tx.set_title(
        "Position Type Distribution – Transaction Share and Capital Share\n"
        "(Table 7 replication)",
        fontsize=12,
    )
    ax_tx.grid(axis="y", alpha=0.3)
    ax_tx.set_ylim(0, df["tx_share_pct"].max() * 1.15)

    # Annotate bars above 1%
    for bar, val in zip(bars_tx, df["tx_share_pct"]):
        if val >= 1.0:
            ax_tx.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    # ── Capital share ──────────────────────────────────────────────────────────
    bars_cap = ax_cap.bar(x, df["capital_share_pct"], color=colours, edgecolor="white", linewidth=0.5)
    ax_cap.set_ylabel("Share of Capital (%)", fontsize=11)
    ax_cap.set_xlabel("Position Type", fontsize=11)
    ax_cap.grid(axis="y", alpha=0.3)
    ax_cap.set_ylim(0, df["capital_share_pct"].max() * 1.15)

    for bar, val in zip(bars_cap, df["capital_share_pct"]):
        if val >= 1.0:
            ax_cap.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    ax_cap.set_xticks(x)
    ax_cap.set_xticklabels([str(t) for t in types], fontsize=10)

    # Legend for win/loss/neutral
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166ac", label="Win types (odd: 1,3,5,7,9,11)"),
        Patch(facecolor="#d6604d", label="Loss types (even: 2,4,6,8,10,12)"),
        Patch(facecolor="#999999", label="Neutral types (13,14,15)"),
    ]
    ax_tx.legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.tight_layout()
    out = RESULTS_DIR / "exp4_type_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: 15-Type Position Distribution – Table 7"
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Use The Graph API for data loading instead of local files.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Experiment 4: Position Type Distribution ===")

    # ── Load data ──────────────────────────────────────────────────────────────
    positions_df = _load_positions(args.use_graph)

    if positions_df.empty:
        logger.error("positions_df is empty – nothing to analyse.")
        sys.exit(1)

    logger.info("Loaded %d positions", len(positions_df))

    # ── Table 7: Position type distribution ───────────────────────────────────
    dist_df = position_type_distribution(
        positions_df,
        type_col="position_type",
        capital_col="open_capital",
    )
    _print_section("Table 7 – Position Type Distribution (tx share and capital share)", dist_df)

    # ── Validation: Types 5+6 > 90% target ────────────────────────────────────
    types_5_6 = dist_df[dist_df["position_type"].isin([5, 6])]
    tx_share_5_6  = float(types_5_6["tx_share_pct"].sum())
    cap_share_5_6 = float(types_5_6["capital_share_pct"].sum())

    print(f"\n--- Validation: Types 5 + 6 Combined ---")
    print(f"  Transaction share : {tx_share_5_6:.2f}%  (paper target: >{PAPER_TYPES_5_6_TX_PCT:.0f}%)")
    print(f"  Capital share     : {cap_share_5_6:.2f}%  (paper target: >{PAPER_TYPES_5_6_CAP_PCT:.0f}%)")

    if tx_share_5_6 < PAPER_TYPES_5_6_TX_PCT:
        logger.warning(
            "Types 5+6 tx share %.2f%% is below paper target >%.0f%%",
            tx_share_5_6, PAPER_TYPES_5_6_TX_PCT,
        )
    if cap_share_5_6 < PAPER_TYPES_5_6_CAP_PCT:
        logger.warning(
            "Types 5+6 capital share %.2f%% is below paper target >%.0f%%",
            cap_share_5_6, PAPER_TYPES_5_6_CAP_PCT,
        )

    # Top-5 types by transaction share
    top5_tx = dist_df.nlargest(5, "tx_share_pct")[["position_type", "tx_share_pct", "tx_count"]]
    print("\n--- Top 5 Position Types by Transaction Share ---")
    print(top5_tx.to_string(index=False))

    # ── Save Table 7 ──────────────────────────────────────────────────────────
    dist_df.to_csv(RESULTS_DIR / "table7_position_type_distribution.csv", index=False)
    logger.info("Saved table7_position_type_distribution.csv")

    # ── Plot ───────────────────────────────────────────────────────────────────
    logger.info("Generating position type distribution bar chart…")
    _plot_type_distribution(dist_df)

    # ── Save exp4_summary.json ─────────────────────────────────────────────────
    type_records = [
        {
            "position_type":    int(r.position_type),
            "tx_count":         int(r.tx_count),
            "tx_share_pct":     round(float(r.tx_share_pct),     4),
            "capital_sum_usdc": round(float(r.capital_sum_usdc), 2),
            "capital_share_pct": round(float(r.capital_share_pct), 4),
        }
        for r in dist_df.itertuples(index=False)
    ]

    summary = {
        "experiment": "exp4_position_types",
        "sample_start": SAMPLE_START,
        "sample_end":   SAMPLE_END,
        "total_positions": int(len(positions_df)),
        "type_5_6_combined": {
            "tx_share_pct":  round(tx_share_5_6,  4),
            "cap_share_pct": round(cap_share_5_6, 4),
        },
        "paper_targets": {
            "types_5_6_tx_pct":  PAPER_TYPES_5_6_TX_PCT,
            "types_5_6_cap_pct": PAPER_TYPES_5_6_CAP_PCT,
        },
        "by_type": type_records,
    }

    summary_path = RESULTS_DIR / "exp4_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved exp4_summary.json")

    logger.info("=== Experiment 4 complete ===")


if __name__ == "__main__":
    main()
