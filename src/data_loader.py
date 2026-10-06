"""
On-chain event data loader for Base-network WETH/USDC CLMM pools.

Supports two backends:
  1. CSV / Parquet files placed in data/<pool_name>/<event_type>.csv
     (schema documented below)
  2. The Graph Protocol GraphQL API (requires GRAPH_API_KEY env var)

The paper (Section 4.1) uses transaction-level data covering
September 2024 – July 2025 from four pools on the Base chain:
Uniswap V3 0.05%, Aerodrome SlipStream 0.05%,
PancakeSwap V3 0.01%, SushiSwap V3 0.01%.

Expected CSV column schemas
───────────────────────────
mint / burn:
  tx_hash, block_number, tx_index, log_index, timestamp (unix int),
  lp_address, tick_lower (int), tick_upper (int),
  liquidity (int/str), amount0 (int/str), amount1 (int/str)

collect:
  tx_hash, block_number, tx_index, log_index, timestamp (unix int),
  lp_address, tick_lower (int), tick_upper (int),
  amount0 (int/str), amount1 (int/str)

swap:
  tx_hash, block_number, tx_index, log_index, timestamp (unix int),
  sqrt_price_x96 (int/str)   ← pool sqrtPriceX96 after the swap

Token ordering for all four pools (USDC/WETH on Base):
  token0 = USDC (decimals=6), token1 = WETH (decimals=18)
  Pool price (WETH in USDC) = (sqrtPriceX96 / 2**96)**2 * 10**(6-18)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (
    CHAIN,
    GRAPH_ENDPOINTS,
    POOLS,
    POOL_BY_NAME,
    SAMPLE_END,
    SAMPLE_START,
    TOKEN0_DECIMALS,
    TOKEN1_DECIMALS,
    PoolConfig,
)

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Price helpers ──────────────────────────────────────────────────────────────

def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
    """
    Convert Uniswap V3 sqrtPriceX96 to pool price (WETH in USDC).

    Formula (Uniswap V3 whitepaper):
      price = (sqrtPriceX96 / 2**96)**2 * 10**(token0_decimals - token1_decimals)
    where token0=USDC (6 dec), token1=WETH (18 dec), so adjustment = 10**(6-18).

    Returns USDC per WETH.
    """
    if sqrt_price_x96 <= 0:
        return float("nan")
    ratio = float(sqrt_price_x96) / (2**96)
    price = ratio**2 * 10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS)
    return price


def tick_to_price(tick: int) -> float:
    """
    Convert Uniswap V3 tick to pool price (USDC per WETH).

    In the pool (token0=USDC, token1=WETH convention used here):
      raw = 1.0001^tick  [token1_raw / token0_raw ratio]
    Adjusting for decimals to get USDC per WETH:
      price_usdc = 1 / (raw * 10^(TOKEN1_DECIMALS - TOKEN0_DECIMALS))
    This is the exact inverse of price_to_tick in generate_synthetic.
    """
    raw = 1.0001**tick
    price = 1.0 / (raw * 10 ** (TOKEN1_DECIMALS - TOKEN0_DECIMALS))
    return price


# ── Raw token amount helpers ───────────────────────────────────────────────────

def token_amounts_to_usdc(
    amount0: float, amount1: float, price_usdc_per_weth: float
) -> float:
    """
    Convert raw on-chain token amounts to USDC-denominated capital.

    token0 = USDC (decimals 6): divide by 1e6
    token1 = WETH (decimals 18): divide by 1e18, multiply by price

    Capital = amount0/1e6 + (amount1/1e18) * price
    """
    usdc_value = amount0 / 1e6 + (amount1 / 1e18) * price_usdc_per_weth
    return usdc_value


# ── CSV / Parquet loader ───────────────────────────────────────────────────────

def _read_pool_events(
    pool_name: str,
    event_type: str,
) -> Optional[pd.DataFrame]:
    """
    Load events from data/<pool_name>/<event_type>.{csv,parquet} if available.
    Returns None if files are missing.
    """
    base = DATA_DIR / pool_name
    for ext in ("parquet", "csv"):
        fpath = base / f"{event_type}.{ext}"
        if fpath.exists():
            logger.info("Loading %s %s from %s", pool_name, event_type, fpath)
            if ext == "parquet":
                df = pd.read_parquet(fpath)
            else:
                df = pd.read_csv(fpath, low_memory=False)
            return df
    return None


def load_pool_events(pool_name: str) -> dict[str, pd.DataFrame]:
    """
    Load all event types for a pool from local files.

    Returns dict with keys: 'mint', 'burn', 'collect', 'swap'.
    Raises FileNotFoundError if any required event file is missing.
    """
    required = ["mint", "burn", "collect", "swap"]
    events: dict[str, pd.DataFrame] = {}
    for ev in required:
        df = _read_pool_events(pool_name, ev)
        if df is None:
            raise FileNotFoundError(
                f"Event file '{ev}' not found for pool '{pool_name}' "
                f"under {DATA_DIR / pool_name}. "
                "Place mint/burn/collect/swap CSV or Parquet files there, "
                "or configure The Graph API via GRAPH_API_KEY."
            )
        events[ev] = df
    return events


# ── The Graph loader ───────────────────────────────────────────────────────────

_GRAPHQL_MINT_QUERY = """
query getMints($pool: String!, $skip: Int!, $first: Int!) {
  mints(
    where: {pool: $pool}
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    id
    transaction { id blockNumber }
    timestamp
    owner
    tickLower
    tickUpper
    amount
    amount0
    amount1
    logIndex
  }
}
"""

_GRAPHQL_BURNS_QUERY = """
query getBurns($pool: String!, $skip: Int!, $first: Int!) {
  burns(
    where: {pool: $pool}
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    id
    transaction { id blockNumber }
    timestamp
    owner
    tickLower
    tickUpper
    amount
    amount0
    amount1
    logIndex
  }
}
"""

_GRAPHQL_COLLECTS_QUERY = """
query getCollects($pool: String!, $skip: Int!, $first: Int!) {
  collects(
    where: {pool: $pool}
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    id
    transaction { id blockNumber }
    timestamp
    owner
    tickLower
    tickUpper
    amount0
    amount1
    logIndex
  }
}
"""

_GRAPHQL_SWAPS_QUERY = """
query getSwaps($pool: String!, $skip: Int!, $first: Int!) {
  swaps(
    where: {pool: $pool}
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    id
    transaction { id blockNumber }
    timestamp
    sqrtPriceX96
    logIndex
  }
}
"""


def _graphql_paginate(
    endpoint: str,
    query: str,
    variables: dict,
    result_key: str,
    page_size: int = 1000,
    max_pages: int = 10000,
) -> list[dict]:
    """
    Paginate a GraphQL query using skip/first until no more results.
    """
    results: list[dict] = []
    skip = 0
    for page in range(max_pages):
        variables["skip"] = skip
        variables["first"] = page_size
        resp = requests.post(
            endpoint,
            json={"query": query, "variables": variables},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        batch = data.get("data", {}).get(result_key, [])
        if not batch:
            break
        results.extend(batch)
        skip += len(batch)
        if len(batch) < page_size:
            break
        time.sleep(0.1)  # polite rate limiting
    logger.info("Fetched %d %s records", len(results), result_key)
    return results


def _parse_mints(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "tx_hash": r["transaction"]["id"],
            "block_number": int(r["transaction"]["blockNumber"]),
            "log_index": int(r.get("logIndex", 0)),
            "timestamp": int(r["timestamp"]),
            "lp_address": r["owner"].lower(),
            "tick_lower": int(r["tickLower"]),
            "tick_upper": int(r["tickUpper"]),
            "liquidity": int(r["amount"]),
            "amount0": float(r["amount0"]),
            "amount1": float(r["amount1"]),
        })
    df = pd.DataFrame(rows)
    df["tx_index"] = 0  # not always available in subgraph
    return df


def _parse_burns(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "tx_hash": r["transaction"]["id"],
            "block_number": int(r["transaction"]["blockNumber"]),
            "log_index": int(r.get("logIndex", 0)),
            "timestamp": int(r["timestamp"]),
            "lp_address": r["owner"].lower(),
            "tick_lower": int(r["tickLower"]),
            "tick_upper": int(r["tickUpper"]),
            "liquidity": int(r["amount"]),
            "amount0": float(r["amount0"]),
            "amount1": float(r["amount1"]),
        })
    df = pd.DataFrame(rows)
    df["tx_index"] = 0
    return df


def _parse_collects(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "tx_hash": r["transaction"]["id"],
            "block_number": int(r["transaction"]["blockNumber"]),
            "log_index": int(r.get("logIndex", 0)),
            "timestamp": int(r["timestamp"]),
            "lp_address": r["owner"].lower(),
            "tick_lower": int(r["tickLower"]),
            "tick_upper": int(r["tickUpper"]),
            "amount0": float(r["amount0"]),
            "amount1": float(r["amount1"]),
        })
    df = pd.DataFrame(rows)
    df["tx_index"] = 0
    return df


def _parse_swaps(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        sqrt_px = int(r["sqrtPriceX96"])
        rows.append({
            "tx_hash": r["transaction"]["id"],
            "block_number": int(r["transaction"]["blockNumber"]),
            "log_index": int(r.get("logIndex", 0)),
            "timestamp": int(r["timestamp"]),
            "sqrt_price_x96": sqrt_px,
            "price_usdc": sqrt_price_x96_to_price(sqrt_px),
        })
    df = pd.DataFrame(rows)
    df["tx_index"] = 0
    return df


def fetch_pool_events_graph(
    pool_name: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch mint/burn/collect/swap events via The Graph Protocol.

    Requires GRAPH_API_KEY environment variable and a valid endpoint
    configured in src/config.py GRAPH_ENDPOINTS[pool_name].
    """
    import datetime
    api_key = os.environ.get("GRAPH_API_KEY", "")
    endpoint_tmpl = GRAPH_ENDPOINTS.get(pool_name, "")
    if not endpoint_tmpl:
        raise ValueError(
            f"No Graph endpoint configured for pool '{pool_name}'. "
            "Set GRAPH_ENDPOINTS[pool_name] in src/config.py."
        )
    endpoint = endpoint_tmpl.format(GRAPH_API_KEY=api_key)
    pool_cfg = POOL_BY_NAME[pool_name]
    pool_addr = pool_cfg.contract.lower()

    if start_ts is None:
        start_ts = int(pd.Timestamp(SAMPLE_START).timestamp())
    if end_ts is None:
        end_ts = int(pd.Timestamp(SAMPLE_END).timestamp())

    vars_base = {"pool": pool_addr}

    logger.info("Fetching mints for %s from The Graph …", pool_name)
    mint_recs = _graphql_paginate(
        endpoint, _GRAPHQL_MINT_QUERY, dict(**vars_base), "mints"
    )
    mint_df = _parse_mints(mint_recs)

    logger.info("Fetching burns for %s from The Graph …", pool_name)
    burn_recs = _graphql_paginate(
        endpoint, _GRAPHQL_BURNS_QUERY, dict(**vars_base), "burns"
    )
    burn_df = _parse_burns(burn_recs)

    logger.info("Fetching collects for %s from The Graph …", pool_name)
    collect_recs = _graphql_paginate(
        endpoint, _GRAPHQL_COLLECTS_QUERY, dict(**vars_base), "collects"
    )
    collect_df = _parse_collects(collect_recs)

    logger.info("Fetching swaps for %s from The Graph …", pool_name)
    swap_recs = _graphql_paginate(
        endpoint, _GRAPHQL_SWAPS_QUERY, dict(**vars_base), "swaps"
    )
    swap_df = _parse_swaps(swap_recs)

    # filter to sample window
    for df in (mint_df, burn_df, collect_df, swap_df):
        df.query("@start_ts <= timestamp <= @end_ts", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return {"mint": mint_df, "burn": burn_df, "collect": collect_df, "swap": swap_df}


def save_pool_events(pool_name: str, events: dict[str, pd.DataFrame]) -> None:
    """Persist event DataFrames to data/<pool_name>/<event>.parquet."""
    out_dir = DATA_DIR / pool_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for ev, df in events.items():
        path = out_dir / f"{ev}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)


# ── Price series helper ────────────────────────────────────────────────────────

def build_price_series(swap_df: pd.DataFrame) -> pd.Series:
    """
    Build a time-indexed pool price series from swap events.

    Returns a pd.Series with integer timestamp index and USDC price values.
    The series can be used to look up the pool price at or before any timestamp
    via forward-fill (last observation).
    """
    if swap_df.empty:
        return pd.Series(dtype=float)
    df = swap_df[["timestamp", "price_usdc"]].dropna()
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    series = df.set_index("timestamp")["price_usdc"]
    return series


def get_price_at(price_series: pd.Series, timestamp: int) -> float:
    """
    Return pool price at or before the given timestamp using the swap series.
    Returns NaN if no prior swap exists.
    """
    if price_series.empty:
        return float("nan")
    idx = price_series.index.searchsorted(timestamp, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(price_series.iloc[idx])


# ── Unified loader (files first, then Graph) ───────────────────────────────────

def load_events(
    pool_name: str,
    use_graph: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load events for a pool. Tries local files first; falls back to The Graph.

    Args:
        pool_name: Key in POOL_BY_NAME (e.g. 'uniswap_v3_005').
        use_graph: Force Graph API even if local files exist.

    Returns:
        dict with keys 'mint', 'burn', 'collect', 'swap'.
    """
    if not use_graph:
        try:
            events = load_pool_events(pool_name)
            # basic schema validation
            _validate_events(events, pool_name)
            return events
        except FileNotFoundError:
            logger.info("Local files not found for %s; trying The Graph.", pool_name)

    events = fetch_pool_events_graph(pool_name)
    _validate_events(events, pool_name)
    return events


def _validate_events(events: dict[str, pd.DataFrame], pool_name: str) -> None:
    """Fail loudly if required columns are missing or data is empty."""
    required = {
        "mint": ["tx_hash", "block_number", "timestamp", "lp_address",
                 "tick_lower", "tick_upper", "liquidity", "amount0", "amount1"],
        "burn": ["tx_hash", "block_number", "timestamp", "lp_address",
                 "tick_lower", "tick_upper", "liquidity", "amount0", "amount1"],
        "collect": ["tx_hash", "block_number", "timestamp", "lp_address",
                    "tick_lower", "tick_upper", "amount0", "amount1"],
        "swap": ["tx_hash", "block_number", "timestamp"],
    }
    for ev_type, cols in required.items():
        df = events.get(ev_type, pd.DataFrame())
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{pool_name}] {ev_type} DataFrame missing columns: {missing}"
            )
    counts = {k: len(v) for k, v in events.items()}
    logger.info("[%s] event counts: %s", pool_name, counts)
