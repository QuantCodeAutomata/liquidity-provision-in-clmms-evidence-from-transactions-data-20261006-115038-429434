#!/usr/bin/env python3
"""
Synthetic on-chain event generator for pipeline validation.

Generates structurally valid mint/burn/collect/swap CSV files for all four
Base-network pools so that the reconstruction pipeline can be exercised
end-to-end without real on-chain data.

IMPORTANT: This generator produces SYNTHETIC data only. It is used solely
to validate that the pipeline runs without errors and produces the expected
output shapes. The paper's headline numbers (e.g. ~1 in 6 profitable LPs)
are based on the authors' real on-chain dataset, which is available from the
reference code repository [20] at GitHub (see README.md).

Token ordering (same as data_loader.py):
  token0 = USDC (decimals 6), token1 = WETH (decimals 18)
  Capital in USDC = amount0/1e6 + (amount1/1e18) * price
"""
from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import POOLS, SAMPLE_START, SAMPLE_END, TOKEN0_DECIMALS, TOKEN1_DECIMALS

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG_SEED = 42

# ── Price simulation parameters ────────────────────────────────────────────────
# WETH/USDC price range over Sep 2024 – Jul 2025 (~2200 – 4000 USDC)
PRICE_INITIAL = 2500.0       # USDC per WETH at sample start
PRICE_DAILY_VOL = 0.025      # ~2.5% daily volatility (realistic for WETH)
N_SWAPS_PER_DAY = 200        # swap events per day across the pool

# ── LP simulation parameters ───────────────────────────────────────────────────
# Per-pool LP counts (scaled to illustrate paper patterns)
# Paper describes hundreds to thousands of LPs per pool
LP_COUNTS = {
    "uniswap_v3_005":          150,
    "aerodrome_slipstream_005": 180,
    "pancakeswap_v3_001":       50,
    "sushiswap_v3_001":         30,
}
# Fraction of pool LPs that also appear in another pool (multi-LPs)
MULTI_LP_FRAC = 0.10

# Average positions per LP
AVG_POSITIONS_PER_LP = 3
POSITION_OPEN_FRAC = 0.6   # fraction of LP-days when position is open

# Width of liquidity ranges in ticks (log-normal distribution)
TICK_SPACING_MULTIPLE_MU = 3    # mean range = 2^3 * tick_spacing ticks wide
TICK_SPACING_MULTIPLE_SIGMA = 1.5

DATA_DIR = Path(__file__).parent


def price_to_sqrt_x96(price_usdc: float) -> int:
    """Convert USDC-per-WETH price to sqrtPriceX96."""
    # price = (sqrtPriceX96/2^96)^2 * 10^(6-18)
    # sqrtPriceX96 = 2^96 * sqrt(price / 10^(6-18))
    adj = price_usdc / (10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))
    return int(2**96 * adj**0.5)


def price_to_tick(price_usdc: float) -> int:
    """Convert USDC-per-WETH price to nearest Uniswap tick."""
    # raw = WETH/USDC in token units after decimal adjustment
    adj = price_usdc / (10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))
    # tick = floor(log(1/adj) / log(1.0001))
    tick = int(np.log(1.0 / adj) / np.log(1.0001))
    return tick


def tick_to_price_usdc(tick: int) -> float:
    """Convert tick to USDC-per-WETH price."""
    raw = 1.0001 ** tick
    return 1.0 / (raw * 10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))


def fake_address(seed: str) -> str:
    """Generate a deterministic fake Ethereum address from a seed string."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    return "0x" + h[:40]


def fake_tx(seed: str) -> str:
    """Generate a deterministic fake tx hash."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    return "0x" + h


def simulate_price_path(rng: np.random.Generator) -> pd.Series:
    """
    Generate a GBM price path for the entire sample period.
    Returns a Series with Unix-second timestamps and USDC prices.
    """
    start_ts = int(pd.Timestamp(SAMPLE_START).timestamp())
    end_ts = int(pd.Timestamp(SAMPLE_END).timestamp())
    n_days = (end_ts - start_ts) // 86400 + 1

    # Daily GBM increments
    dt = 1.0 / 252
    drift = -0.5 * PRICE_DAILY_VOL**2 * dt
    shocks = rng.normal(drift, PRICE_DAILY_VOL * dt**0.5, n_days)
    log_prices = np.cumsum(np.concatenate([[np.log(PRICE_INITIAL)], shocks]))
    prices = np.exp(log_prices)

    # Create intraday timestamps
    ts_list, price_list = [], []
    for day_idx in range(n_days):
        day_start = start_ts + day_idx * 86400
        day_swaps = rng.integers(N_SWAPS_PER_DAY // 2, N_SWAPS_PER_DAY * 2)
        offsets = np.sort(rng.integers(0, 86400, day_swaps))
        day_price = prices[day_idx]
        noise = rng.normal(0, day_price * 0.001, len(offsets))
        ts_list.extend(day_start + offsets)
        price_list.extend(day_price + noise)

    series = pd.Series(price_list, index=ts_list, name="price_usdc")
    series = series[series > 0]
    series = series.sort_index()
    return series


def build_lp_addresses(
    rng: np.random.Generator,
    pool_name: str,
    n_lps: int,
    shared_lps: list[str],
) -> list[str]:
    """
    Build LP address list: mix pool-unique LPs with some shared (multi-LP) addresses.
    """
    n_shared = min(len(shared_lps), max(1, int(n_lps * MULTI_LP_FRAC)))
    selected_shared = rng.choice(shared_lps, n_shared, replace=False).tolist()
    n_unique = n_lps - n_shared
    unique = [fake_address(f"{pool_name}_lp_{i}") for i in range(n_unique)]
    return unique + selected_shared


def generate_pool_events(
    pool_name: str,
    lp_addresses: list[str],
    price_series: pd.Series,
    tick_spacing: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic mint/burn/collect/swap events for one pool.

    The event generation follows the paper's Section 2.1:
    - Each LP has 1-5 unique ranges.
    - Each range has multiple mint events followed by burn/collect pairs.
    - Zero-value events are injected (to be filtered out) at ~50% rate.
    - Some positions start before sample window (to trigger dynamic-balance exclusion).

    Token amounts are proportional to position size:
      For in-range positions: both token0 (USDC) and token1 (WETH) are non-zero.
    """
    start_ts = int(pd.Timestamp(SAMPLE_START).timestamp())
    end_ts = int(pd.Timestamp(SAMPLE_END).timestamp())
    duration = end_ts - start_ts

    def get_price(ts: int) -> float:
        idx = price_series.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return float(price_series.iloc[0])
        return float(price_series.iloc[idx])

    mint_rows, burn_rows, collect_rows, swap_rows = [], [], [], []
    log_counter = [0]

    def next_log() -> int:
        log_counter[0] += 1
        return log_counter[0]

    # ── Swap events (background grid + injected at every mint/burn timestamp) ──
    # Background: every GBM hour point gives traders price continuity.
    sample_every = 1   # one swap per GBM hour → tightest lookup gap
    swap_sample = price_series.iloc[::sample_every]
    swap_ts_set: set = set()
    for ts_idx, px_val in swap_sample.items():
        ts = int(ts_idx)
        price = float(px_val)
        sqrt_x96 = price_to_sqrt_x96(price)
        tx_seed = f"{pool_name}_swap_{ts}_{next_log()}"
        swap_rows.append({
            "tx_hash": fake_tx(tx_seed),
            "block_number": 1000000 + ts // 12,
            "tx_index": 1,
            "log_index": next_log(),
            "timestamp": ts,
            "sqrt_price_x96": sqrt_x96,
            "price_usdc": price,
        })
        swap_ts_set.add(ts)
    # Mint/burn-aligned swaps are injected after the LP loop (see below).

    # ── Per-LP position generation ─────────────────────────────────────────────
    for lp_idx, lp_addr in enumerate(lp_addresses):
        n_ranges = rng.integers(1, 4)

        for range_idx in range(n_ranges):
            # ── Open time and price ───────────────────────────────────────────
            # Choose opening time first, then build the range around it so that
            # P_start is inside [a, b] for ~90% of positions (paper Table 7).
            open_ts = start_ts + rng.integers(0, int(duration * 0.85))
            open_price = get_price(open_ts)
            open_tick = price_to_tick(open_price)

            # Range width in multiples of tick_spacing (log-normal)
            half_width = max(
                tick_spacing,
                int(tick_spacing * 2 ** rng.normal(TICK_SPACING_MULTIPLE_MU, TICK_SPACING_MULTIPLE_SIGMA)),
            )
            # Place opening price near centre of range (realistic LP behaviour)
            centre_offset = rng.integers(-half_width // 4, half_width // 4 + 1)
            mid_tick = open_tick + centre_offset

            tick_lower = (mid_tick - half_width) - ((mid_tick - half_width) % tick_spacing)
            tick_upper = (mid_tick + half_width) - ((mid_tick + half_width) % tick_spacing) + tick_spacing
            if tick_upper <= tick_lower:
                tick_upper = tick_lower + tick_spacing

            # price_a < price_b in USDC space
            p_tl = tick_to_price_usdc(tick_lower)   # higher USDC/WETH (lower tick)
            p_tu = tick_to_price_usdc(tick_upper)   # lower  USDC/WETH (upper tick)
            price_a = min(p_tl, p_tu)
            price_b = max(p_tl, p_tu)

            # Generate positions in this range
            n_positions = rng.integers(1, AVG_POSITIONS_PER_LP + 1)

            for pos_idx in range(n_positions):
                if pos_idx > 0:
                    # Subsequent positions in same range start after the previous close.
                    open_ts = min(open_ts + rng.integers(3600, duration // (n_positions + 1) + 3601),
                                  end_ts - 7200)
                    # 90% of re-openings are also while price is still in-range.
                    if rng.random() < 0.90:
                        ps_idx_arr = np.asarray(price_series.index)
                        ps_val_arr = np.asarray(price_series.values, dtype=float)
                        after_open_m = ps_idx_arr > open_ts
                        in_range_m = (ps_val_arr >= price_a) & (ps_val_arr <= price_b)
                        cands_m = ps_idx_arr[after_open_m & in_range_m]
                        if len(cands_m) > 0:
                            open_ts = int(cands_m[int(rng.integers(0, max(1, len(cands_m) // 4)))])
                    open_price = get_price(open_ts)
                if open_ts >= end_ts:
                    break

                # liquidity units (arbitrary, proportional to capital)
                liquidity = int(rng.lognormal(20, 2))

                # Token amounts at opening (in raw token units)
                position_usdc = rng.lognormal(7, 2)  # USDC capital ~$1000 median
                usdc_fraction = rng.uniform(0.2, 0.8)
                amount0_open = int(position_usdc * usdc_fraction * 1e6)    # USDC raw
                amount1_open = int(position_usdc * (1 - usdc_fraction) / max(open_price, 1) * 1e18)  # WETH raw

                # Mint event
                tx_seed = f"{pool_name}_{lp_addr}_{range_idx}_{pos_idx}_mint"
                mint_rows.append({
                    "tx_hash": fake_tx(tx_seed),
                    "block_number": 1000000 + open_ts // 12,
                    "tx_index": 2,
                    "log_index": next_log(),
                    "timestamp": open_ts,
                    "lp_address": lp_addr,
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "liquidity": liquidity,
                    "amount0": amount0_open,
                    "amount1": amount1_open,
                })

                # ── Closing time: bias toward staying in-range ────────────────
                # 90% of positions close while the price is still in [price_a, price_b].
                # Reproduces the paper's >90% Types 5+6 concentration (Section 4.3).
                hold_secs = int(rng.lognormal(11, 1.5))  # median ~2 days
                close_ts_raw = min(open_ts + hold_secs, end_ts - 1)

                if rng.random() < 0.90:
                    # Find ALL timestamps in price_series that are after open_ts
                    # and where the price is still inside [price_a, price_b].
                    ps_idx = np.asarray(price_series.index)
                    ps_val = np.asarray(price_series.values, dtype=float)
                    after_open = ps_idx > open_ts
                    in_range = (ps_val >= price_a) & (ps_val <= price_b)
                    candidates = ps_idx[after_open & in_range]
                    if len(candidates) > 0:
                        # Prefer an early close (profit-target behaviour, Section 4.3)
                        n_early = max(1, len(candidates) // 4)
                        chosen_idx = int(rng.integers(0, n_early))
                        close_ts_raw = int(candidates[chosen_idx])

                # Require at least 3601 s (one GBM step + 1 s) so that open and
                # close always fall in different hourly price bins.  A 60 s floor
                # can put both events in the same bin, making get_price_at return
                # the same value for open and close and spuriously classifying the
                # position as PnL-neutral (Type 13/14/15).
                close_ts = max(open_ts + 3601, min(close_ts_raw, end_ts - 1))
                close_price = get_price(close_ts)

                # PnL: small random return (-20% to +20%)
                pnl_pct = rng.normal(0, 0.08)
                close_capital = position_usdc * (1 + pnl_pct)
                close_fraction = rng.uniform(0.2, 0.8)
                amount0_close = int(close_capital * close_fraction * 1e6)
                amount1_close = int(close_capital * (1 - close_fraction) / max(close_price, 1) * 1e18)

                # Burn and collect share the same tx_hash (one transaction on-chain)
                shared_tx_hash = fake_tx(f"{pool_name}_{lp_addr}_{range_idx}_{pos_idx}_close")
                burn_log = next_log()
                collect_log = next_log()

                burn_rows.append({
                    "tx_hash": shared_tx_hash,
                    "block_number": 1000000 + close_ts // 12,
                    "tx_index": 2,
                    "log_index": burn_log,
                    "timestamp": close_ts,
                    "lp_address": lp_addr,
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "liquidity": liquidity,
                    "amount0": amount0_close // 2,   # amounts in burn are pre-fee
                    "amount1": amount1_close // 2,
                })

                # Collect in same transaction as burn (paper Section 2.1)
                collect_rows.append({
                    "tx_hash": shared_tx_hash,
                    "block_number": 1000000 + close_ts // 12,
                    "tx_index": 2,
                    "log_index": collect_log,
                    "timestamp": close_ts,
                    "lp_address": lp_addr,
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "amount0": amount0_close,   # full realized payout in collect
                    "amount1": amount1_close,
                })

                remaining_ts = close_ts + rng.integers(3600, 86400)

    # ── Inject zero-value events to validate filtering (~50%) ──────────────────
    n_zero = len(mint_rows) // 2
    for i in range(n_zero):
        ts = start_ts + rng.integers(0, duration)
        lp = lp_addresses[rng.integers(0, len(lp_addresses))]
        tick_l = -887220
        tick_u = 887220
        tx_seed = f"{pool_name}_zero_mint_{i}"
        mint_rows.append({
            "tx_hash": fake_tx(tx_seed),
            "block_number": 1000000 + ts // 12,
            "tx_index": 4,
            "log_index": next_log(),
            "timestamp": ts,
            "lp_address": lp,
            "tick_lower": tick_l,
            "tick_upper": tick_u,
            "liquidity": 0,
            "amount0": 0,
            "amount1": 0,
        })

    # ── Inject swaps at every mint and burn timestamp (price-alignment guarantee)
    # This ensures the reconstruction's get_price_at() returns exactly the price
    # used when generating the range, preventing mis-classification due to
    # swap-lookup interpolation error.
    all_event_ts: set = set()
    for row in mint_rows:
        all_event_ts.add(int(row["timestamp"]))
    for row in burn_rows:
        all_event_ts.add(int(row["timestamp"]))
    for ts in sorted(all_event_ts):
        if ts in swap_ts_set:
            continue  # already covered by background grid
        price = get_price(ts)
        sqrt_x96 = price_to_sqrt_x96(price)
        tx_seed = f"{pool_name}_alignswap_{ts}_{next_log()}"
        swap_rows.append({
            "tx_hash": fake_tx(tx_seed),
            "block_number": 1000000 + ts // 12,
            "tx_index": 0,   # before LP tx_index=1 events at same block
            "log_index": next_log(),
            "timestamp": ts,
            "sqrt_price_x96": sqrt_x96,
            "price_usdc": price,
        })

    # Build DataFrames and sort by (block_number, tx_index, log_index)
    def _to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        sort_cols = [c for c in ["block_number", "tx_index", "log_index"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    return {
        "mint": _to_df(mint_rows),
        "burn": _to_df(burn_rows),
        "collect": _to_df(collect_rows),
        "swap": _to_df(swap_rows),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Generating synthetic on-chain event data for pipeline validation…")
    print("NOTE: This is SYNTHETIC data. Real results require actual on-chain data.")

    # Shared LP pool for multi-LP simulation
    n_shared = 40
    shared_lps = [fake_address(f"shared_multi_lp_{i}") for i in range(n_shared)]

    # Shared price path (one per pool for simplicity)
    for pool_cfg in POOLS:
        pool_name = pool_cfg.name
        out_dir = DATA_DIR / pool_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Pool: {pool_name}")

        # Price path
        price_series = simulate_price_path(rng)

        # LP addresses
        n_lps = LP_COUNTS[pool_name]
        lp_addresses = build_lp_addresses(rng, pool_name, n_lps, shared_lps)

        # Generate events
        events = generate_pool_events(
            pool_name=pool_name,
            lp_addresses=lp_addresses,
            price_series=price_series,
            tick_spacing=pool_cfg.tick_spacing,
            rng=rng,
        )

        # Save
        for ev_type, df in events.items():
            fpath = out_dir / f"{ev_type}.csv"
            df.to_csv(fpath, index=False)
            print(f"    {ev_type}: {len(df):>6,} rows → {fpath.name}")

    print("\nDone. Event files written to data/<pool_name>/")


if __name__ == "__main__":
    main()
