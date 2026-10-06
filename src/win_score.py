"""
Win-score metric ω for LP strategy evaluation.

Implements Appendix A of the paper exactly:

  For a sequence of closed positions ordered by closing time, build the
  cumulative realized PnL path PnL_cum(t), split into positive and negative
  parts, normalize both by S = max(P_max+, -P_min-), then compute the
  time-integrated areas A+ and A- over the LP observation window [t_start, t_end].
  The win-score is ω = A+ / (A+ + A-).  If S = 0, ω = 0.5.

  Since on-chain timestamps are discrete integers (Unix seconds), the integrals
  are computed exactly as sums of rectangle widths × heights (piecewise-constant
  step functions).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import WIN_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


def compute_win_score(
    close_timestamps: Sequence[int],
    position_pnls: Sequence[float],
    t_start: int,
    t_end: int,
) -> float:
    """
    Compute the win-score ω for one LP strategy.

    Args:
        close_timestamps: Closing times of each reconstructed position (Unix int).
        position_pnls: Realized PnL of each position (numeraire USDC), same order.
        t_start: Observation-period start (first mint timestamp of the LP).
        t_end: Observation-period end (last event timestamp or sample end).

    Returns:
        ω ∈ [0, 1]. Returns 0.5 if S = 0 (no movement in cumulative path).
    """
    if len(close_timestamps) == 0:
        return 0.5

    # Sort positions by closing time (Appendix A: ordered by position closing time)
    pairs = sorted(zip(close_timestamps, position_pnls), key=lambda x: x[0])
    times = [p[0] for p in pairs]
    pnls = [p[1] for p in pairs]

    # Build cumulative PnL path as an event-driven step function
    # Events: t_start, *close_times, t_end
    # PnL_cum(t) = sum of pnl[i] for all i where times[i] <= t
    # Clamp event times to [t_start, t_end]
    times_arr = np.array(times, dtype=np.int64)
    pnls_arr = np.array(pnls, dtype=np.float64)

    # Cumulative PnL just after each close event
    cum_pnl = np.cumsum(pnls_arr)

    # Construct time grid: boundaries of each constant segment
    # Segment k: [t_k, t_{k+1}), value = cum_pnl[k-1] (0 before first event)
    # Use t_start as the 0-value baseline before any positions close.
    segment_starts = np.concatenate([[t_start], times_arr])
    segment_ends = np.concatenate([times_arr, [t_end]])
    segment_values = np.concatenate([[0.0], cum_pnl])

    # Clip to observation window
    seg_s = np.maximum(segment_starts, t_start)
    seg_e = np.minimum(segment_ends, t_end)
    widths = np.maximum(seg_e - seg_s, 0).astype(np.float64)

    # Positive and negative parts of cumulative path
    pos_vals = np.maximum(segment_values, 0.0)
    neg_vals = np.minimum(segment_values, 0.0)

    # Normalization factor S = max(P_max+, -P_min-)
    P_max_plus = np.max(pos_vals)
    P_min_minus = np.min(neg_vals)
    S = max(P_max_plus, -P_min_minus)

    if S == 0:
        return 0.5  # Appendix A: set ω = 0.5 if S = 0

    # Normalized components (Appendix A)
    C_plus = pos_vals / S          # C+(t) = PnL_cum+(t) / S
    C_minus = (-neg_vals) / S      # C-(t) = -PnL_cum-(t) / S

    # Time-integrated areas (piecewise-constant → rectangle areas)
    A_plus = np.dot(C_plus, widths)
    A_minus = np.dot(C_minus, widths)

    denom = A_plus + A_minus
    if denom == 0:
        return 0.5
    return float(A_plus / denom)


def classify_lps(
    lp_positions_df: pd.DataFrame,
    sample_start_ts: int,
    sample_end_ts: int,
) -> pd.DataFrame:
    """
    Compute win-scores and classify all LPs in one DataFrame.

    Args:
        lp_positions_df: DataFrame with columns:
            pool, lp_address, close_timestamp, pnl
          (output of pnl_reconstruction.positions_to_dataframe).
        sample_start_ts: Unix timestamp for the start of the sample period.
        sample_end_ts:   Unix timestamp for the end of the sample period.

    Returns:
        DataFrame with one row per (pool, lp_address):
          pool, lp_address, terminal_pnl, win_score,
          pnl_positive (bool), consistently_successful (bool),
          n_positions, t_start, t_end
    """
    required = {"pool", "lp_address", "close_timestamp", "pnl"}
    missing = required - set(lp_positions_df.columns)
    if missing:
        raise ValueError(f"lp_positions_df missing columns: {missing}")

    records: list[dict] = []

    for (pool, lp), grp in lp_positions_df.groupby(["pool", "lp_address"]):
        grp = grp.sort_values("close_timestamp")
        close_ts = grp["close_timestamp"].tolist()
        pos_pnls = grp["pnl"].tolist()

        # Observation window: from first open_timestamp to sample end
        # Paper uses LP strategy observation period; we use sample bounds
        # with LP's first event as t_start.
        if "open_timestamp" in grp.columns:
            t_start = max(sample_start_ts, int(grp["open_timestamp"].min()))
        else:
            t_start = max(sample_start_ts, int(min(close_ts)) if close_ts else sample_start_ts)
        t_end = sample_end_ts

        terminal_pnl = sum(pos_pnls)
        omega = compute_win_score(close_ts, pos_pnls, t_start, t_end)

        records.append({
            "pool": pool,
            "lp_address": lp,
            "terminal_pnl": terminal_pnl,
            "win_score": omega,
            # Table 4 uses PnL > 0 (strict) as the profitability criterion
            "pnl_positive": terminal_pnl > 0,
            "consistently_successful": (terminal_pnl > 0) and (omega > WIN_SCORE_THRESHOLD),
            "n_positions": len(close_ts),
            "t_start": t_start,
            "t_end": t_end,
        })

    if not records:
        return pd.DataFrame(columns=[
            "pool", "lp_address", "terminal_pnl", "win_score",
            "pnl_positive", "consistently_successful",
            "n_positions", "t_start", "t_end",
        ])
    return pd.DataFrame(records)


def pool_classification_table(lp_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Table 4 aggregates (Section 4.2).

    Returns DataFrame with columns:
      pool, count, pnl_pos_count, pnl_pos_pct,
      consistent_count, consistent_pct
    """
    rows = []
    for pool, grp in lp_class_df.groupby("pool"):
        n = len(grp)
        n_pos = grp["pnl_positive"].sum()
        n_con = grp["consistently_successful"].sum()
        rows.append({
            "pool": pool,
            "lp_count": n,
            "pnl_positive_count": int(n_pos),
            "pnl_positive_pct": 100.0 * n_pos / n if n > 0 else 0.0,
            "consistent_count": int(n_con),
            "consistent_pct": 100.0 * n_con / n if n > 0 else 0.0,
        })
    # Total row
    n_total = len(lp_class_df)
    n_pos_total = lp_class_df["pnl_positive"].sum()
    n_con_total = lp_class_df["consistently_successful"].sum()
    rows.append({
        "pool": "Total",
        "lp_count": n_total,
        "pnl_positive_count": int(n_pos_total),
        "pnl_positive_pct": 100.0 * n_pos_total / n_total if n_total > 0 else 0.0,
        "consistent_count": int(n_con_total),
        "consistent_pct": 100.0 * n_con_total / n_total if n_total > 0 else 0.0,
    })
    return pd.DataFrame(rows)
