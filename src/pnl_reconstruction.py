"""
LP PnL reconstruction from on-chain mint/burn/collect/swap events.

Implements the dynamic-balance procedure described in Section 2.1 of the paper:

  1. Identify unique LPs with at least one mint and one burn (Section 2.1).
  2. Remove zero-value transactions (Section 4.1).
  3. For each LP-range pair, match burn and collect into composite burn+ events.
  4. Sort mint and burn+ events chronologically by (block_number, tx_index, log_index).
  5. Exclude burn+ events whose liquidity exceeds cumulative minted liquidity up
     to that point in time (unreconstructable events, Section 2.1 step 3).
  6. Reconstruct positions: each opening mint is closed by subsequent burn+ events.
     If a burn+ overshoots the remaining liquidity, split its capital proportionally
     between the current and the next position (Section 2.1 step 4-5).
  7. Compute position PnL = closing capital Q - opening capital W (Section 2.1).
  8. Aggregate LP-level PnL = sum of closed position PnLs.

Numeraire: USDC (token0 in all four pools).
Opening/closing capital formula:
  W  = amount0_mint   / 1e6  + (amount1_mint   / 1e18) * P_mint
  B  = amount0_burn+  / 1e6  + (amount1_burn+  / 1e18) * P_burn+
  Q  = sum of B contributions to this position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import get_price_at, token_amounts_to_usdc, build_price_series

logger = logging.getLogger(__name__)


# ── Output data structures ─────────────────────────────────────────────────────

@dataclass
class ReconstructedPosition:
    """A closed LP position and its realized PnL."""
    pool: str
    lp_address: str
    tick_lower: int
    tick_upper: int
    price_lower: float          # range lower boundary in USDC
    price_upper: float          # range upper boundary in USDC

    open_timestamp: int         # unix timestamp of opening mint
    close_timestamp: int        # unix timestamp of last contributing burn+
    open_price: float           # pool price at opening (USDC)
    close_price_weighted: float # capital-weighted closing price (USDC)

    open_capital: float         # W_jM (USDC)
    close_capital: float        # Q_jM (USDC)
    pnl: float                  # Q_jM - W_jM (USDC)

    n_burns: int                # number of burn+ events contributing to closure


@dataclass
class LPResult:
    """Aggregated result for one LP within one pool."""
    pool: str
    lp_address: str
    positions: list[ReconstructedPosition] = field(default_factory=list)

    @property
    def terminal_pnl(self) -> float:
        return sum(p.pnl for p in self.positions)

    @property
    def n_positions(self) -> int:
        return len(self.positions)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def remove_zero_value_events(
    events: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Remove zero-value transactions (Section 4.1).

    For mint/burn: zero liquidity is the zero-value criterion.
    For collect:  zero amount0 AND amount1 means no payout.
    Swap events are not filtered (used only for price lookup).
    """
    filtered: dict[str, pd.DataFrame] = {"swap": events["swap"].copy()}

    mint = events["mint"].copy()
    mint = mint[mint["liquidity"].astype(float) > 0]
    filtered["mint"] = mint.reset_index(drop=True)

    burn = events["burn"].copy()
    burn = burn[burn["liquidity"].astype(float) > 0]
    filtered["burn"] = burn.reset_index(drop=True)

    collect = events["collect"].copy()
    collect = collect[
        (collect["amount0"].astype(float) > 0) | (collect["amount1"].astype(float) > 0)
    ]
    filtered["collect"] = collect.reset_index(drop=True)

    for ev in ("mint", "burn", "collect"):
        before = len(events[ev])
        after = len(filtered[ev])
        logger.debug("zero-value filter %s: %d → %d", ev, before, after)
    return filtered


def filter_lps_with_mint_and_burn(
    mint_df: pd.DataFrame, burn_df: pd.DataFrame
) -> set[str]:
    """
    Return LP addresses that have at least one mint and one burn (Section 2.1).
    """
    minters = set(mint_df["lp_address"].str.lower())
    burners = set(burn_df["lp_address"].str.lower())
    eligible = minters & burners
    logger.info("LPs with ≥1 mint and ≥1 burn: %d", len(eligible))
    return eligible


# ── Burn+ matching ─────────────────────────────────────────────────────────────

def match_burn_collect(
    burn_df: pd.DataFrame,
    collect_df: pd.DataFrame,
    lp_address: str,
) -> pd.DataFrame:
    """
    Match burn and collect transactions for a single LP to form burn+ events.

    Matching key: (lp_address, tick_lower, tick_upper, tx_hash).
    For each burn transaction, its corresponding collect transaction is
    identified by the same LP address and the same block/transaction.
    If multiple collects occur in the same transaction for the same range,
    they are aggregated.

    The paper (Section 2.1) states that burn records the liquidity change
    while collect records the token payout; both are needed for capital
    computation. A burn without a matching collect results in zero payout
    amounts being used (some protocols emit collect separately from burn).

    Returns a DataFrame with columns:
      tx_hash, block_number, tx_index, log_index, timestamp,
      lp_address, tick_lower, tick_upper,
      liquidity (from burn), amount0 (from collect), amount1 (from collect)
    """
    lp = lp_address.lower()
    b = burn_df[burn_df["lp_address"].str.lower() == lp].copy()
    c = collect_df[collect_df["lp_address"].str.lower() == lp].copy()

    if b.empty:
        return pd.DataFrame(columns=[
            "tx_hash", "block_number", "tx_index", "log_index",
            "timestamp", "lp_address", "tick_lower", "tick_upper",
            "liquidity", "amount0", "amount1",
        ])

    # Aggregate collects by (tx_hash, tick_lower, tick_upper) to handle
    # protocols that emit multiple collect logs per burn.
    if not c.empty:
        c_agg = (
            c.groupby(["tx_hash", "tick_lower", "tick_upper"], as_index=False)
            .agg(
                amount0=("amount0", "sum"),
                amount1=("amount1", "sum"),
            )
        )
    else:
        c_agg = pd.DataFrame(columns=["tx_hash", "tick_lower", "tick_upper",
                                       "amount0", "amount1"])

    # Merge on tx_hash + range
    merged = b.merge(
        c_agg,
        on=["tx_hash", "tick_lower", "tick_upper"],
        how="left",
        suffixes=("_burn", "_collect"),
    )

    # Prefer collect amounts; fall back to burn amounts if collect not found
    for col in ("amount0", "amount1"):
        collect_col = f"{col}_collect"
        burn_col = f"{col}_burn"
        if collect_col in merged.columns:
            fallback = merged.get(burn_col, pd.Series(0.0, index=merged.index))
            merged[col] = pd.to_numeric(
                merged[collect_col].fillna(fallback), errors="coerce"
            ).fillna(0.0)
        elif burn_col in merged.columns:
            merged[col] = merged[burn_col]
        else:
            merged[col] = 0.0

    result = merged[
        ["tx_hash", "block_number", "tx_index", "log_index",
         "timestamp", "lp_address", "tick_lower", "tick_upper",
         "liquidity", "amount0", "amount1"]
    ].copy()
    result["lp_address"] = lp
    result = result.sort_values(["block_number", "tx_index", "log_index"])
    return result.reset_index(drop=True)


# ── Dynamic balance reconstruction ─────────────────────────────────────────────

def reconstruct_lp_range(
    lp_address: str,
    pool: str,
    tick_lower: int,
    tick_upper: int,
    price_lower: float,
    price_upper: float,
    mint_range: pd.DataFrame,
    burn_plus_range: pd.DataFrame,
    price_series: pd.Series,
) -> list[ReconstructedPosition]:
    """
    Reconstruct positions for one LP within one liquidity range (Section 2.1).

    Algorithm:
      1. Sort mints and burn+ events chronologically.
      2. Build a signed liquidity ledger: +liquidity for mint, -liquidity for burn+.
      3. Skip burn+ events whose liquidity exceeds cumulative minted liquidity
         (pre-window deposits, Section 2.1 step 3).
      4. Greedily close each mint with subsequent burn+ events.
         If a burn+ overshoots, split its capital proportionally.

    Args:
        mint_range, burn_plus_range: Already filtered to this lp+range.
        price_series: Indexed by timestamp, gives pool price in USDC.

    Returns:
        List of closed ReconstructedPosition objects.
    """
    positions: list[ReconstructedPosition] = []

    # Sort mints chronologically
    mints = mint_range.sort_values(["block_number", "tx_index", "log_index"]).copy()
    burns_plus = burn_plus_range.sort_values(
        ["block_number", "tx_index", "log_index"]
    ).copy()

    mints = mints.reset_index(drop=True)
    burns_plus = burns_plus.reset_index(drop=True)

    # Step 3: filter out burn+ events whose liquidity exceeds cumulative minted
    cumulative_minted = 0.0
    retained_burns: list[dict] = []
    mint_ptr = 0
    burn_ptr = 0
    all_events: list[dict] = []

    for _, row in mints.iterrows():
        ev = row.to_dict()
        ev["event_type"] = "mint"
        all_events.append(ev)
    for _, row in burns_plus.iterrows():
        ev = row.to_dict()
        ev["event_type"] = "burn_plus"
        all_events.append(ev)

    # Sort all events by ordering fields
    all_events.sort(
        key=lambda e: (
            int(e.get("block_number", 0)),
            int(e.get("tx_index", 0)),
            int(e.get("log_index", 0)),
        )
    )

    # Compute running cumulative minted and filter unreconstructable burns
    cum_minted = 0.0
    valid_events: list[dict] = []
    for ev in all_events:
        liq = float(ev.get("liquidity", 0))
        if ev["event_type"] == "mint":
            cum_minted += liq
            valid_events.append(ev)
        else:  # burn_plus
            if liq > cum_minted:
                # Pre-window liquidity: skip (Section 2.1 step 3)
                logger.debug(
                    "Skipping unreconstructable burn+ liq=%.0f > cum_minted=%.0f",
                    liq, cum_minted,
                )
                continue
            cum_minted -= liq
            valid_events.append(ev)

    # Now reconstruct positions from valid events
    # Walk mints in order; close each with subsequent burn+ events
    mint_queue: list[dict] = [e for e in valid_events if e["event_type"] == "mint"]
    burn_queue: list[dict] = [e for e in valid_events if e["event_type"] == "burn_plus"]

    burn_idx = 0
    for mint_ev in mint_queue:
        mint_liq = float(mint_ev.get("liquidity", 0))
        if mint_liq <= 0:
            continue

        open_ts = int(mint_ev.get("timestamp", 0))
        open_price = get_price_at(price_series, open_ts)
        if np.isnan(open_price) or open_price <= 0:
            # Use tick midpoint price as fallback
            open_price = (price_lower + price_upper) / 2.0

        open_capital = token_amounts_to_usdc(
            float(mint_ev.get("amount0", 0)),
            float(mint_ev.get("amount1", 0)),
            open_price,
        )

        remaining_liq = mint_liq
        close_capital = 0.0
        close_ts = open_ts
        n_burns = 0

        # weighted sum for capital-weighted closing price
        weighted_price_sum = 0.0
        total_burn_capital = 0.0

        while burn_idx < len(burn_queue) and remaining_liq > 0:
            burn_ev = burn_queue[burn_idx]
            burn_ts = int(burn_ev.get("timestamp", 0))
            burn_liq = float(burn_ev.get("liquidity", 0))
            burn_price = get_price_at(price_series, burn_ts)
            if np.isnan(burn_price) or burn_price <= 0:
                burn_price = (price_lower + price_upper) / 2.0

            burn_capital_full = token_amounts_to_usdc(
                float(burn_ev.get("amount0", 0)),
                float(burn_ev.get("amount1", 0)),
                burn_price,
            )

            if burn_liq <= remaining_liq:
                # Entire burn+ closes within this mint position
                fraction = 1.0
                burn_idx += 1
            else:
                # Burn+ overshoots: split proportionally (Section 2.1 step 4-5)
                fraction = remaining_liq / burn_liq
                # Do not advance burn_idx; residual will apply to next position
                # Adjust the burn event's remaining liquidity for next iteration
                burn_queue[burn_idx] = dict(burn_queue[burn_idx])
                burn_queue[burn_idx]["liquidity"] = burn_liq - remaining_liq
                burn_queue[burn_idx]["amount0"] = float(
                    burn_ev.get("amount0", 0)
                ) * (1 - fraction)
                burn_queue[burn_idx]["amount1"] = float(
                    burn_ev.get("amount1", 0)
                ) * (1 - fraction)
                # Also split burn_capital_full
                burn_capital_full = burn_capital_full * fraction

            contributed = burn_capital_full
            close_capital += contributed
            close_ts = max(close_ts, burn_ts)
            n_burns += 1
            remaining_liq -= fraction * burn_liq if fraction < 1.0 else burn_liq

            if contributed > 0:
                weighted_price_sum += contributed * burn_price
                total_burn_capital += contributed

            if fraction < 1.0:
                # Position is fully closed by a partial burn
                remaining_liq = 0
                break

        if remaining_liq < 1e-3 * mint_liq:
            # Position is closed (within numerical tolerance)
            if total_burn_capital > 0:
                close_price_w = weighted_price_sum / total_burn_capital
            else:
                # All contributing burns had zero token amounts.  Using open_price
                # here would pin P_end = P_start and wrongly force the position into
                # a PnL-neutral type (13/14/15) regardless of actual price movement.
                # Instead, use the pool price at the close timestamp.
                close_price_w = get_price_at(price_series, close_ts)
                if np.isnan(close_price_w) or close_price_w <= 0:
                    close_price_w = (price_lower + price_upper) / 2.0
            pos = ReconstructedPosition(
                pool=pool,
                lp_address=lp_address,
                tick_lower=tick_lower,
                tick_upper=tick_upper,
                price_lower=price_lower,
                price_upper=price_upper,
                open_timestamp=open_ts,
                close_timestamp=close_ts,
                open_price=open_price,
                close_price_weighted=close_price_w,
                open_capital=open_capital,
                close_capital=close_capital,
                pnl=close_capital - open_capital,
                n_burns=n_burns,
            )
            positions.append(pos)
        # else: unclosed mint — discarded per paper (only closed positions used)

    return positions


# ── Top-level reconstruction ───────────────────────────────────────────────────

def reconstruct_pool(
    pool: str,
    events: dict[str, pd.DataFrame],
    price_series: pd.Series,
    tick_to_price_fn,
) -> tuple[list[LPResult], dict]:
    """
    Run the full PnL reconstruction pipeline for one pool.

    Returns:
        lp_results: List of LPResult objects (one per eligible LP).
        stats: Dict of transaction count statistics for reporting.
    """
    from src.data_loader import token_amounts_to_usdc

    mint_df = events["mint"].copy()
    burn_df = events["burn"].copy()
    collect_df = events["collect"].copy()

    # Step 1: normalize addresses
    for df in (mint_df, burn_df, collect_df):
        df["lp_address"] = df["lp_address"].str.lower()

    stats: dict = {
        "initial_mint": len(events["mint"]),
        "initial_burn": len(events["burn"]),
        "initial_collect": len(events["collect"]),
    }

    # Step 2: remove zero-value events
    filtered = remove_zero_value_events(events)
    mint_df = filtered["mint"].copy()
    burn_df = filtered["burn"].copy()
    collect_df = filtered["collect"].copy()
    for df in (mint_df, burn_df, collect_df):
        df["lp_address"] = df["lp_address"].str.lower()

    stats.update({
        "filtered_mint": len(mint_df),
        "filtered_burn": len(burn_df),
        "filtered_collect": len(collect_df),
    })

    # Step 3: identify eligible LPs
    eligible_lps = filter_lps_with_mint_and_burn(mint_df, burn_df)

    mint_df = mint_df[mint_df["lp_address"].isin(eligible_lps)]
    burn_df = burn_df[burn_df["lp_address"].isin(eligible_lps)]
    collect_df = collect_df[collect_df["lp_address"].isin(eligible_lps)]

    stats["eligible_lp_count"] = len(eligible_lps)

    lp_results: list[LPResult] = []
    retained_mint = 0
    retained_burn = 0

    for lp_addr in sorted(eligible_lps):
        lp_mint = mint_df[mint_df["lp_address"] == lp_addr]
        lp_burn = burn_df[burn_df["lp_address"] == lp_addr]

        # Unique liquidity ranges for this LP
        ranges = set()
        ranges.update(zip(lp_mint["tick_lower"], lp_mint["tick_upper"]))
        ranges.update(zip(lp_burn["tick_lower"], lp_burn["tick_upper"]))

        lp_res = LPResult(pool=pool, lp_address=lp_addr)

        for (tl, tu) in ranges:
            # With TOKEN0=USDC, TOKEN1=WETH the lower tick gives the higher
            # USDC/WETH price; take min/max so that price_lower < price_upper
            # (i.e. a < b) as the position classifier requires.
            p_tl = tick_to_price_fn(tl)
            p_tu = tick_to_price_fn(tu)
            price_lower = min(p_tl, p_tu)
            price_upper = max(p_tl, p_tu)

            mint_range = lp_mint[
                (lp_mint["tick_lower"] == tl) & (lp_mint["tick_upper"] == tu)
            ]
            burn_range = lp_burn[
                (lp_burn["tick_lower"] == tl) & (lp_burn["tick_upper"] == tu)
            ]

            # Match burn + collect into burn+ events
            burn_plus = match_burn_collect(
                burn_range, collect_df, lp_addr
            )
            # Filter burn_plus to this range
            if not burn_plus.empty:
                burn_plus = burn_plus[
                    (burn_plus["tick_lower"] == tl) & (burn_plus["tick_upper"] == tu)
                ]

            if mint_range.empty:
                continue

            positions = reconstruct_lp_range(
                lp_address=lp_addr,
                pool=pool,
                tick_lower=tl,
                tick_upper=tu,
                price_lower=price_lower,
                price_upper=price_upper,
                mint_range=mint_range,
                burn_plus_range=burn_plus,
                price_series=price_series,
            )
            lp_res.positions.extend(positions)
            retained_mint += len(mint_range)
            if not burn_plus.empty:
                retained_burn += len(burn_plus)

        lp_results.append(lp_res)

    stats["retained_mint"] = retained_mint
    stats["retained_burn"] = retained_burn
    stats["retained_collect"] = retained_burn  # same count by construction (burn+)
    stats["reconstructed_lp_count"] = len(
        [r for r in lp_results if r.n_positions > 0]
    )
    return lp_results, stats


def positions_to_dataframe(lp_results: list[LPResult]) -> pd.DataFrame:
    """Convert list of LPResult objects to a flat DataFrame of positions."""
    rows = []
    for lp in lp_results:
        for pos in lp.positions:
            rows.append({
                "pool": pos.pool,
                "lp_address": pos.lp_address,
                "tick_lower": pos.tick_lower,
                "tick_upper": pos.tick_upper,
                "price_lower": pos.price_lower,
                "price_upper": pos.price_upper,
                "open_timestamp": pos.open_timestamp,
                "close_timestamp": pos.close_timestamp,
                "open_price": pos.open_price,
                "close_price_weighted": pos.close_price_weighted,
                "open_capital": pos.open_capital,
                "close_capital": pos.close_capital,
                "pnl": pos.pnl,
                "n_burns": pos.n_burns,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "pool", "lp_address", "tick_lower", "tick_upper",
            "price_lower", "price_upper",
            "open_timestamp", "close_timestamp",
            "open_price", "close_price_weighted",
            "open_capital", "close_capital", "pnl", "n_burns",
        ])
    return pd.DataFrame(rows)


def lp_results_to_summary(lp_results: list[LPResult]) -> pd.DataFrame:
    """Aggregate LP-level terminal PnL from reconstructed positions."""
    rows = []
    for lp in lp_results:
        rows.append({
            "pool": lp.pool,
            "lp_address": lp.lp_address,
            "terminal_pnl": lp.terminal_pnl,
            "n_positions": lp.n_positions,
        })
    if not rows:
        return pd.DataFrame(columns=["pool", "lp_address", "terminal_pnl", "n_positions"])
    return pd.DataFrame(rows)
