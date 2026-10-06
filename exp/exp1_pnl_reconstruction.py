#!/usr/bin/env python3
"""
Experiment 1: Reconstruct LP PnL from on-chain transaction data for Base WETH/USD CLMM pools.

Replicates Tables 1, 2, and 3 of the paper, reporting transaction counts at each
stage of the reconstruction pipeline:
  Table 1 – Initial raw event counts (mint, burn, collect) per pool.
  Table 2 – Counts after zero-value event removal (~50% reduction).
  Table 3 – Counts retained through the dynamic-balance stage (~65% of filtered).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import POOLS, SAMPLE_END, SAMPLE_START
from src.pipeline import run_pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_count_table(
    pool_stats: dict,
    col_map: dict[str, str],
    total_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a per-pool count DataFrame from pool_stats.

    Args:
        pool_stats: Mapping pool_name -> stats dict from run_pipeline.
        col_map:    {output_column_name: stats_key} pairs to extract.
        total_col:  If given, add a row-sum column with this name.

    Returns DataFrame with a pool column plus requested count columns
    and a 'Total' summary row.
    """
    rows = []
    for pool_cfg in POOLS:
        stats = pool_stats.get(pool_cfg.name, {})
        row: dict = {"pool": pool_cfg.name, "protocol": pool_cfg.protocol}
        for col_name, stat_key in col_map.items():
            row[col_name] = int(stats.get(stat_key, 0))
        rows.append(row)

    df = pd.DataFrame(rows)

    if total_col:
        count_cols = list(col_map.keys())
        df[total_col] = df[count_cols].sum(axis=1)

    # Totals row
    total_row: dict = {"pool": "Total", "protocol": "—"}
    for col in df.columns[2:]:
        total_row[col] = int(df[col].sum())
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(df.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1: PnL Reconstruction – Tables 1, 2, 3"
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Use The Graph API for data loading instead of local files.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Experiment 1: PnL Reconstruction ===")
    logger.info("Sample period: %s to %s", SAMPLE_START, SAMPLE_END)
    logger.info("Results directory: %s", RESULTS_DIR)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    logger.info("Running full pipeline (save_intermediates=True)…")
    result = run_pipeline(
        use_graph=args.use_graph,
        save_intermediates=True,
        results_dir=RESULTS_DIR,
    )

    pool_stats: dict = result["pool_stats"]
    positions_df: pd.DataFrame = result["positions_df"]
    lp_class_df: pd.DataFrame = result["lp_class_df"]

    # ── Table 1: Initial transaction counts ────────────────────────────────────
    table1 = _build_count_table(
        pool_stats,
        col_map={
            "mint_initial":    "initial_mint",
            "burn_initial":    "initial_burn",
            "collect_initial": "initial_collect",
        },
        total_col="total_initial",
    )
    _print_section("Table 1 – Initial Transaction Counts (raw)", table1)
    table1.to_csv(RESULTS_DIR / "table1_initial_counts.csv", index=False)
    logger.info("Saved table1_initial_counts.csv")

    total_initial = int(table1.loc[table1["pool"] == "Total", "total_initial"].iloc[0])

    # ── Table 2: After zero-value filter ───────────────────────────────────────
    table2 = _build_count_table(
        pool_stats,
        col_map={
            "mint_filtered":    "filtered_mint",
            "burn_filtered":    "filtered_burn",
            "collect_filtered": "filtered_collect",
        },
        total_col="total_filtered",
    )
    total_filtered = int(table2.loc[table2["pool"] == "Total", "total_filtered"].iloc[0])
    zero_removal_pct = 100.0 * (1.0 - total_filtered / total_initial) if total_initial else 0.0

    _print_section("Table 2 – Transaction Counts After Zero-Value Filter", table2)
    logger.info(
        "Zero-value removal: %.1f%% removed (paper target: ~50%%)",
        zero_removal_pct,
    )
    table2.to_csv(RESULTS_DIR / "table2_filtered_counts.csv", index=False)
    logger.info("Saved table2_filtered_counts.csv")

    # ── Table 3: After dynamic-balance stage ───────────────────────────────────
    table3 = _build_count_table(
        pool_stats,
        col_map={
            "mint_retained": "retained_mint",
            "burn_retained": "retained_burn",
        },
        total_col="total_retained",
    )
    total_retained = int(table3.loc[table3["pool"] == "Total", "total_retained"].iloc[0])
    dynamic_retention_pct = 100.0 * total_retained / total_filtered if total_filtered else 0.0

    _print_section("Table 3 – Transaction Counts Used (After Dynamic Balance)", table3)
    logger.info(
        "Dynamic-balance retention: %.1f%% retained (paper target: ~65%%)",
        dynamic_retention_pct,
    )
    table3.to_csv(RESULTS_DIR / "table3_used_counts.csv", index=False)
    logger.info("Saved table3_used_counts.csv")

    # ── LP counts per pool ─────────────────────────────────────────────────────
    lp_rows = []
    for pool_cfg in POOLS:
        stats = pool_stats.get(pool_cfg.name, {})
        lp_rows.append({
            "pool":                  pool_cfg.name,
            "protocol":              pool_cfg.protocol,
            "eligible_lp_count":     int(stats.get("eligible_lp_count", 0)),
            "reconstructed_lp_count": int(stats.get("reconstructed_lp_count", 0)),
        })
    lp_df = pd.DataFrame(lp_rows)
    total_lp_row = {
        "pool":                   "Total",
        "protocol":               "—",
        "eligible_lp_count":      int(lp_df["eligible_lp_count"].sum()),
        "reconstructed_lp_count": int(lp_df["reconstructed_lp_count"].sum()),
    }
    lp_df = pd.concat([lp_df, pd.DataFrame([total_lp_row])], ignore_index=True)

    _print_section("LP Counts Per Pool", lp_df)
    total_eligible_lps = int(lp_df.loc[lp_df["pool"] == "Total", "eligible_lp_count"].iloc[0])
    total_reconstructed_lps = int(lp_df.loc[lp_df["pool"] == "Total", "reconstructed_lp_count"].iloc[0])

    # ── Retained transaction count summary ────────────────────────────────────
    n_positions = int(len(positions_df)) if not positions_df.empty else 0
    n_lps_classified = int(len(lp_class_df)) if not lp_class_df.empty else 0

    print("\n=== Retained Transaction Count Summary ===")
    print(f"  Initial total transactions  : {total_initial:>10,}")
    print(f"  After zero-value filter     : {total_filtered:>10,}  ({100 - zero_removal_pct:.1f}% retained)")
    print(f"  After dynamic-balance stage : {total_retained:>10,}  ({dynamic_retention_pct:.1f}% of filtered)")
    print(f"  Reconstructed positions     : {n_positions:>10,}")
    print(f"  Eligible LPs (≥1 mint+burn) : {total_eligible_lps:>10,}")
    print(f"  LPs with ≥1 closed position : {total_reconstructed_lps:>10,}")
    print(f"  Classified LPs (win-score)  : {n_lps_classified:>10,}")

    # ── Validation log ─────────────────────────────────────────────────────────
    logger.info("--- Validation against paper targets ---")
    logger.info(
        "Zero-value removal rate  : %.1f%%  (paper ~50%%)",
        zero_removal_pct,
    )
    logger.info(
        "Dynamic-balance retention: %.1f%%  (paper ~65%%)",
        dynamic_retention_pct,
    )
    if zero_removal_pct < 30 or zero_removal_pct > 70:
        logger.warning("Zero-value removal rate %.1f%% is far from the ~50%% paper target", zero_removal_pct)
    if dynamic_retention_pct < 45 or dynamic_retention_pct > 85:
        logger.warning("Dynamic-balance retention %.1f%% is far from the ~65%% paper target", dynamic_retention_pct)

    # ── Save exp1_summary.json ─────────────────────────────────────────────────
    per_pool_stats: dict = {}
    for pool_cfg in POOLS:
        stats = pool_stats.get(pool_cfg.name, {})
        per_pool_stats[pool_cfg.name] = {k: int(v) for k, v in stats.items()}

    summary = {
        "experiment": "exp1_pnl_reconstruction",
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "transaction_counts": {
            "initial_total":              total_initial,
            "after_zero_value_filter":    total_filtered,
            "after_dynamic_balance":      total_retained,
            "zero_value_removal_pct":     round(zero_removal_pct, 2),
            "dynamic_balance_retention_pct": round(dynamic_retention_pct, 2),
        },
        "lp_counts": {
            "eligible_lps":       total_eligible_lps,
            "reconstructed_lps":  total_reconstructed_lps,
            "classified_lps":     n_lps_classified,
            "reconstructed_positions": n_positions,
        },
        "paper_targets": {
            "zero_value_removal_pct":       50.0,
            "dynamic_balance_retention_pct": 65.0,
        },
        "per_pool": per_pool_stats,
    }

    summary_path = RESULTS_DIR / "exp1_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved exp1_summary.json")

    logger.info("=== Experiment 1 complete ===")


if __name__ == "__main__":
    main()
