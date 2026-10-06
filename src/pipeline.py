"""
End-to-end pipeline: load events → reconstruct PnL → classify LPs → positions.

All experiments call this shared pipeline. No experiment defines its own
data loader or reconstruction logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import POOLS, SAMPLE_END, SAMPLE_START, PoolConfig
from src.data_loader import (
    build_price_series,
    load_events,
    tick_to_price,
)
from src.pnl_reconstruction import (
    lp_results_to_summary,
    positions_to_dataframe,
    reconstruct_pool,
)
from src.win_score import classify_lps
from src.position_types import classify_positions_df
from src.metrics import compute_position_metrics

logger = logging.getLogger(__name__)


def run_pipeline(
    pools: Optional[list[PoolConfig]] = None,
    use_graph: bool = False,
    sample_start: str = SAMPLE_START,
    sample_end: str = SAMPLE_END,
    save_intermediates: bool = False,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Run the full replication pipeline for all pools.

    Args:
        pools: List of PoolConfig objects to process; defaults to all four.
        use_graph: Force The Graph API for data loading.
        sample_start / sample_end: Date strings ('YYYY-MM-DD').
        save_intermediates: Write CSV intermediates to results_dir.
        results_dir: Directory for intermediate outputs.

    Returns:
        dict with keys:
          'positions_df': all reconstructed closed positions across pools
          'lp_class_df':  LP-level classification with win-score
          'pool_stats':   dict of per-pool transaction count stats
          'sample_start_ts', 'sample_end_ts': Unix timestamps
    """
    if pools is None:
        pools = POOLS

    sample_start_ts = int(pd.Timestamp(sample_start).timestamp())
    sample_end_ts = int(pd.Timestamp(sample_end).timestamp())

    all_positions: list[pd.DataFrame] = []
    pool_stats: dict[str, dict] = {}

    for pool_cfg in pools:
        pool_name = pool_cfg.name
        logger.info("=== Processing pool: %s ===", pool_name)

        # Load events
        events = load_events(pool_name, use_graph=use_graph)

        # Build price series from swaps
        price_series = build_price_series(events["swap"])
        logger.info(
            "[%s] price series: %d swap events, price range [%.2f, %.2f]",
            pool_name,
            len(price_series),
            price_series.min() if not price_series.empty else float("nan"),
            price_series.max() if not price_series.empty else float("nan"),
        )

        # Reconstruct LP PnL
        lp_results, stats = reconstruct_pool(
            pool=pool_name,
            events=events,
            price_series=price_series,
            tick_to_price_fn=tick_to_price,
        )
        pool_stats[pool_name] = stats
        logger.info("[%s] stats: %s", pool_name, stats)

        # Convert to DataFrame
        pos_df = positions_to_dataframe(lp_results)
        if not pos_df.empty:
            all_positions.append(pos_df)

    # Combine positions across pools
    if all_positions:
        positions_df = pd.concat(all_positions, ignore_index=True)
    else:
        positions_df = pd.DataFrame()

    # Classify positions into 15 types
    if not positions_df.empty:
        positions_df = classify_positions_df(positions_df)
        positions_df = compute_position_metrics(positions_df)

    # LP-level classification
    if not positions_df.empty:
        lp_class_df = classify_lps(positions_df, sample_start_ts, sample_end_ts)
    else:
        lp_class_df = pd.DataFrame()

    if save_intermediates and results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        if not positions_df.empty:
            positions_df.to_csv(results_dir / "positions.csv", index=False)
        if not lp_class_df.empty:
            lp_class_df.to_csv(results_dir / "lp_classification.csv", index=False)

    return {
        "positions_df": positions_df,
        "lp_class_df": lp_class_df,
        "pool_stats": pool_stats,
        "sample_start_ts": sample_start_ts,
        "sample_end_ts": sample_end_ts,
    }
