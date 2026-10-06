"""
15-type LP position taxonomy (Section 2.3, Figure 3).

Each position is classified by the relative ordering of:
  P_start (pool price at position opening)
  P_end   (capital-weighted closing price)
  a       (price_lower, lower range boundary in USDC)
  b       (price_upper, upper range boundary in USDC)

Classification table (Figure 3 / Table in paper pages 4-5):
  Type 1 : P_start < P_end <= a          → win
  Type 2 : P_end < P_start <= a          → loss
  Type 3 : P_start < a < P_end <= b      → win
  Type 4 : P_end < a < P_start <= b      → loss
  Type 5 : a <= P_start < P_end <= b     → win   (dominant in-range)
  Type 6 : a <= P_end < P_start <= b     → loss  (dominant in-range)
  Type 7 : a <= P_start < b < P_end      → win
  Type 8 : a <= P_end < b < P_start      → loss
  Type 9 : P_start < a < b < P_end       → win
  Type 10: P_end < a < b < P_start       → loss
  Type 11: b <= P_start < P_end          → win
  Type 12: b <= P_end < P_start          → loss
  Type 13: a < P_start = P_end < b       → PnL-neutral (in-range, no move)
  Type 14: P_start = P_end < a           → PnL-neutral (below range)
  Type 15: b < P_start = P_end           → PnL-neutral (above range)

Note: Boundary equality is treated as inclusive per the conditions above.
"PnL-neutral" corresponds numerically to P_start == P_end; in practice
a small tolerance is used (tolerance=1e-8 relative) to handle floating-point
representations of equal timestamps.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Win/loss/neutral sets for reference
WIN_TYPES = frozenset({1, 3, 5, 7, 9, 11})
LOSS_TYPES = frozenset({2, 4, 6, 8, 10, 12})
NEUTRAL_TYPES = frozenset({13, 14, 15})

# Approximate tolerance for P_start == P_end comparison
_EQ_TOL = 1e-8


def classify_position_type(
    p_start: float,
    p_end: float,
    a: float,
    b: float,
) -> int:
    """
    Classify a single position into one of 15 types.

    Args:
        p_start: Pool price at position opening (P_start).
        p_end:   Capital-weighted closing price (P_end).
        a:       Lower range boundary (price_lower).
        b:       Upper range boundary (price_upper).

    Returns:
        Integer position type in 1..15, or 0 if unclassifiable.
    """
    # Handle degenerate range
    if a >= b:
        logger.debug("Degenerate range a=%.6f >= b=%.6f", a, b)
        return 0

    # Determine if P_start == P_end (within tolerance)
    equal = abs(p_start - p_end) <= _EQ_TOL * max(abs(p_start), abs(p_end), 1.0)

    if equal:
        p_mid = (p_start + p_end) / 2.0
        if a < p_mid < b:
            return 13  # in-range neutral
        elif p_mid < a:
            return 14  # below-range neutral
        else:          # p_mid >= b
            return 15  # above-range neutral

    # Types 1-12: P_start ≠ P_end
    ps, pe = p_start, p_end

    # ── Below-range types (Types 1, 2) ────────────────────────────────────────
    if ps < pe and pe <= a:
        return 1
    if pe < ps and ps <= a:
        return 2

    # ── Cross-lower-boundary types (Types 3, 4) ───────────────────────────────
    if ps < a and a < pe <= b:
        return 3
    if pe < a and a < ps <= b:
        return 4

    # ── In-range types (Types 5, 6) ───────────────────────────────────────────
    if a <= ps and ps < pe and pe <= b:
        return 5
    if a <= pe and pe < ps and ps <= b:
        return 6

    # ── Cross-upper-boundary types (Types 7, 8) ───────────────────────────────
    if a <= ps and ps < b and b < pe:
        return 7
    if a <= pe and pe < b and b < ps:
        return 8

    # ── Full-range traversal types (Types 9, 10) ──────────────────────────────
    if ps < a and a < b and b < pe:
        return 9
    if pe < a and a < b and b < ps:
        return 10

    # ── Above-range types (Types 11, 12) ──────────────────────────────────────
    if b <= ps and ps < pe:
        return 11
    if b <= pe and pe < ps:
        return 12

    # Fallback: numerically ambiguous (boundary equalities)
    # Try relaxed assignment using midpoints
    return _classify_boundary_case(ps, pe, a, b)


def _classify_boundary_case(ps: float, pe: float, a: float, b: float) -> int:
    """Handle boundary-equality cases with a relaxed priority ordering."""
    # Win or loss by sign
    if ps < pe:  # upward → win types: 1, 3, 5, 7, 9, 11
        if pe <= a:
            return 1
        if ps < a <= pe <= b:
            return 3
        if a <= ps and pe <= b:
            return 5
        if a <= ps < b and pe > b:
            return 7
        if ps < a and pe > b:
            return 9
        if ps >= b:
            return 11
    else:  # downward → loss types: 2, 4, 6, 8, 10, 12
        if ps <= a:
            return 2
        if pe < a <= ps <= b:
            return 4
        if a <= pe and ps <= b:
            return 6
        if a <= pe < b and ps > b:
            return 8
        if pe < a and ps > b:
            return 10
        if pe >= b:
            return 12
    return 0


def classify_positions_df(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'position_type' column to a reconstructed-positions DataFrame.

    Required columns: open_price, close_price_weighted, price_lower, price_upper.
    """
    required = {"open_price", "close_price_weighted", "price_lower", "price_upper"}
    missing = required - set(positions_df.columns)
    if missing:
        raise ValueError(f"positions_df missing columns: {missing}")

    df = positions_df.copy()
    df["position_type"] = df.apply(
        lambda r: classify_position_type(
            r["open_price"],
            r["close_price_weighted"],
            r["price_lower"],
            r["price_upper"],
        ),
        axis=1,
    )
    return df


def position_type_distribution(
    positions_df: pd.DataFrame,
    type_col: str = "position_type",
    capital_col: str = "open_capital",
) -> pd.DataFrame:
    """
    Reproduce Table 7: share of transactions and mint capital per position type.

    Args:
        positions_df: DataFrame with position_type and open_capital columns.
        type_col: Column name for position type (int 1-15).
        capital_col: Column name for opening capital.

    Returns:
        DataFrame with columns: position_type, tx_count, tx_share,
          capital_sum, capital_share.
    """
    df = positions_df.copy()
    df = df[df[type_col].between(1, 15)]

    total_tx = len(df)
    total_cap = df[capital_col].sum()

    rows = []
    for ptype in range(1, 16):
        sub = df[df[type_col] == ptype]
        n = len(sub)
        cap = sub[capital_col].sum()
        rows.append({
            "position_type": ptype,
            "tx_count": n,
            "tx_share_pct": 100.0 * n / total_tx if total_tx > 0 else 0.0,
            "capital_sum_usdc": cap,
            "capital_share_pct": 100.0 * cap / total_cap if total_cap > 0 else 0.0,
        })
    return pd.DataFrame(rows)
