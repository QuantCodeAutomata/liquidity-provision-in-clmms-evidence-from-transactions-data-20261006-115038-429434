"""
Position-level metrics for win-position types 3, 5, and 7 (Section 4.3).

Metrics defined in the paper (Section 2.3 equations 2, 3.1, 3.2):

  Delta metric (Eq. 2): applicable to Types 3-8.
    Δ_start = clip((P_start - a) / (b - a), 0, 1)
    Δ_end   = clip((P_end   - a) / (b - a), 0, 1)
    Δ       = Δ_end - Δ_start

  Price-to-boundary metrics (Eqs. 3.1, 3.2): applicable to Types 1-10.
    m_a = P_start/a - 1     (opening: distance from lower boundary)
    m_b = P_start/b - 1     (opening: distance from upper boundary)
    d_a = P_end/a   - 1     (closing: distance from lower boundary)
    d_b = P_end/b   - 1     (closing: distance from upper boundary)
    rel_a = d_a - m_a       (change in position relative to lower boundary)
    rel_b = d_b - m_b       (change in position relative to upper boundary)

  Return metrics (Section 4.3, defined verbally):
    R_cap   = Q / W - 1     (position capital return)
    R_price = P_end / P_start - 1  (price return over position lifetime)

  Duration:
    duration_sec = close_timestamp - open_timestamp  (seconds)

  Range width in USDC:
    width_usdc = b - a  (absolute range width in USDC, the numeraire)
    The paper uses capital-weighted average range width per LP per type
    when bucketing by width.

  Capital-weighted average width for LP-type grouping (Section 4.3):
    For each LP-type pair, average position range width weighted by open_capital.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import WIDTH_BUCKETS

logger = logging.getLogger(__name__)


# ── Position-level metric computation ─────────────────────────────────────────

def compute_position_metrics(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all position-level metrics and add them as columns.

    Required columns:
      open_price, close_price_weighted, price_lower, price_upper,
      open_capital, close_capital, open_timestamp, close_timestamp,
      position_type (1-15).

    Added columns:
      delta_start, delta_end, delta,
      m_a, m_b, d_a, d_b, rel_a, rel_b,
      r_cap, r_price, duration_sec, width_usdc
    """
    df = positions_df.copy()

    ps = df["open_price"].to_numpy(dtype=float)
    pe = df["close_price_weighted"].to_numpy(dtype=float)
    a = df["price_lower"].to_numpy(dtype=float)
    b = df["price_upper"].to_numpy(dtype=float)
    W = df["open_capital"].to_numpy(dtype=float)
    Q = df["close_capital"].to_numpy(dtype=float)

    range_width = b - a
    # Guard against zero-width ranges
    valid_range = range_width > 0

    # Delta metric (Eq. 2) — applicable to Types 3-8
    delta_start = np.where(valid_range, np.clip((ps - a) / range_width, 0.0, 1.0), np.nan)
    delta_end = np.where(valid_range, np.clip((pe - a) / range_width, 0.0, 1.0), np.nan)
    delta = delta_end - delta_start

    # Price-to-boundary metrics (Eqs. 3.1, 3.2) — applicable to Types 1-10
    # Guard against zero boundaries
    safe_a = np.where(a > 0, a, np.nan)
    safe_b = np.where(b > 0, b, np.nan)
    m_a = ps / safe_a - 1.0
    m_b = ps / safe_b - 1.0
    d_a = pe / safe_a - 1.0
    d_b = pe / safe_b - 1.0
    rel_a = d_a - m_a
    rel_b = d_b - m_b

    # Capital returns
    r_cap = np.where(W > 0, Q / W - 1.0, np.nan)
    # Price return
    safe_ps = np.where(ps > 0, ps, np.nan)
    r_price = pe / safe_ps - 1.0

    # Duration
    dur = (df["close_timestamp"] - df["open_timestamp"]).to_numpy(dtype=float)
    dur = np.where(dur >= 0, dur, np.nan)

    # Range width in USDC (numeraire)
    width_usdc = range_width

    df["delta_start"] = delta_start
    df["delta_end"] = delta_end
    df["delta"] = delta
    df["m_a"] = m_a
    df["m_b"] = m_b
    df["d_a"] = d_a
    df["d_b"] = d_b
    df["rel_a"] = rel_a
    df["rel_b"] = rel_b
    df["r_cap"] = r_cap
    df["r_price"] = r_price
    df["duration_sec"] = dur
    df["width_usdc"] = width_usdc

    return df


def assign_width_bucket(width_usdc: float) -> str:
    """
    Assign a position-range-width USDC bucket label (Section 4.3).

    Buckets:
      (0, 10), [10, 100), [100, 1000), [1000, 10000), > 10000
    """
    for lo, hi, label in WIDTH_BUCKETS:
        if lo < width_usdc <= hi or (lo == 0 and 0 < width_usdc <= hi):
            return label
    # Edge: width exactly 0 or negative (degenerate)
    return "invalid"


def add_width_bucket(df: pd.DataFrame, width_col: str = "width_usdc") -> pd.DataFrame:
    """Add 'width_bucket' column to positions DataFrame."""
    df = df.copy()
    df["width_bucket"] = df[width_col].apply(assign_width_bucket)
    return df


# ── Capital-weighted average width per LP-type ─────────────────────────────────

def lp_type_weighted_width(
    positions_df: pd.DataFrame,
    lp_col: str = "lp_address",
    type_col: str = "position_type",
    capital_col: str = "open_capital",
    width_col: str = "width_usdc",
) -> pd.DataFrame:
    """
    Compute capital-weighted average range width per (LP, position_type).

    Used to bucket LPs into width categories for Tables 8-10 analysis.

    Returns DataFrame: lp_address, position_type, cw_avg_width, width_bucket
    """
    df = positions_df.copy()
    df = df[df[capital_col] > 0]

    def _cw_mean(grp: pd.DataFrame) -> float:
        w = grp[capital_col].to_numpy(dtype=float)
        v = grp[width_col].to_numpy(dtype=float)
        if w.sum() == 0:
            return float("nan")
        return float(np.dot(w, v) / w.sum())

    result = (
        df.groupby([lp_col, type_col])
        .apply(_cw_mean, include_groups=False)
        .reset_index()
        .rename(columns={0: "cw_avg_width"})
    )
    result["width_bucket"] = result["cw_avg_width"].apply(assign_width_bucket)
    return result


# ── Aggregate analysis for Tables 8-10 ────────────────────────────────────────

def win_type_analysis_table(
    positions_df: pd.DataFrame,
    lp_class_df: pd.DataFrame,
    position_types: list[int],
) -> pd.DataFrame:
    """
    Produce summary table for selected win-position types (Tables 8, 9, 10).

    Restricts to consistently successful LPs (Section 4.3).
    Groups by position type and capital-weighted average width bucket.

    Args:
        positions_df: Must include position_type, open_price, close_price_weighted,
                      price_lower, price_upper, open_capital, close_capital,
                      open_timestamp, close_timestamp, lp_address, pool.
        lp_class_df: Must include lp_address, consistently_successful.
        position_types: List of position types to include (e.g. [3, 5, 7]).

    Returns:
        DataFrame with metrics per (position_type, width_bucket).
    """
    # Filter to consistently successful LPs
    successful_lps = set(
        lp_class_df.loc[lp_class_df["consistently_successful"], "lp_address"]
    )
    df = positions_df[positions_df["lp_address"].isin(successful_lps)].copy()

    # Filter to selected win position types
    df = df[df["position_type"].isin(position_types)]

    if df.empty:
        return pd.DataFrame()

    # Compute metrics
    df = compute_position_metrics(df)

    # Compute capital-weighted avg width per LP-type
    cw_width = lp_type_weighted_width(df)
    df = df.merge(
        cw_width[["lp_address", "position_type", "cw_avg_width", "width_bucket"]],
        on=["lp_address", "position_type"],
        how="left",
    )

    # For overall row, use position-level width bucket
    df["pos_width_bucket"] = df["width_usdc"].apply(assign_width_bucket)

    rows = []
    for ptype in position_types:
        sub = df[df["position_type"] == ptype]
        # Group by LP-level capital-weighted width bucket
        all_buckets = [b[2] for b in WIDTH_BUCKETS] + ["Overall"]
        for bucket in all_buckets:
            if bucket == "Overall":
                grp = sub
            else:
                grp = sub[sub["width_bucket"] == bucket]

            if grp.empty:
                continue

            n_lp = grp["lp_address"].nunique()
            n_tx = len(grp)
            cap_sum = grp["open_capital"].sum()
            total_cap = sub["open_capital"].sum()

            def wmean(col: str) -> float:
                valid = grp[[col, "open_capital"]].dropna()
                if valid.empty or valid["open_capital"].sum() == 0:
                    return float("nan")
                return float(
                    np.dot(valid[col], valid["open_capital"]) / valid["open_capital"].sum()
                )

            rows.append({
                "position_type": ptype,
                "width_bucket": bucket,
                "n_lp": n_lp,
                "lp_pct": 100.0 * n_lp / sub["lp_address"].nunique()
                if sub["lp_address"].nunique() > 0
                else 0.0,
                "n_tx": n_tx,
                "tx_pct": 100.0 * n_tx / len(sub) if len(sub) > 0 else 0.0,
                "tvl_pct": 100.0 * cap_sum / total_cap if total_cap > 0 else 0.0,
                "avg_pos_size_usdc": grp["open_capital"].mean(),
                "avg_duration_sec": grp["duration_sec"].mean(),
                "delta_pct": 100.0 * wmean("delta"),
                "delta_start_pct": 100.0 * wmean("delta_start"),
                "m_a_pct": 100.0 * wmean("m_a"),
                "m_b_pct": 100.0 * wmean("m_b"),
                "d_a_pct": 100.0 * wmean("d_a"),
                "d_b_pct": 100.0 * wmean("d_b"),
                "rel_a_pct": 100.0 * wmean("rel_a"),
                "rel_b_pct": 100.0 * wmean("rel_b"),
                "r_cap_pct": 100.0 * wmean("r_cap"),
                "r_price_pct": 100.0 * wmean("r_price"),
            })
    return pd.DataFrame(rows)
