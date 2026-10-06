"""
Configuration for CLMM LP PnL replication.

Pool addresses from paper Section 4.1 footnotes 1-4:
  Uniswap V3 0.05%   : 0xd0b53d9277642d899df5c87a3966a349a798f224
  Aerodrome 0.05%    : 0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59
  PancakeSwap 0.01%  : 0x72ab388e2e2f6facef59e3c3fa2c4e29011c2d38
  SushiSwap 0.01%    : 0x482fe995c4a52bc79271ab29a53591363ee30a89
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ── Sample period ──────────────────────────────────────────────────────────────
SAMPLE_START: str = "2024-09-01"
SAMPLE_END: str = "2025-07-31"

# ── Network ────────────────────────────────────────────────────────────────────
CHAIN: str = "base"

# ── Pool definitions ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PoolConfig:
    name: str
    protocol: str
    contract: str      # checksummed lower-case pool contract address
    fee_tier: float    # e.g. 0.0005 for 0.05%
    tick_spacing: int  # standard Uniswap V3 tick spacing for fee tier


POOLS: list[PoolConfig] = [
    PoolConfig(
        name="uniswap_v3_005",
        protocol="Uniswap V3",
        contract="0xd0b53d9277642d899df5c87a3966a349a798f224",
        fee_tier=0.0005,
        tick_spacing=10,
    ),
    PoolConfig(
        name="aerodrome_slipstream_005",
        protocol="Aerodrome SlipStream",
        contract="0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59",
        fee_tier=0.0005,
        tick_spacing=100,
    ),
    PoolConfig(
        name="pancakeswap_v3_001",
        protocol="PancakeSwap V3",
        contract="0x72ab388e2e2f6facef59e3c3fa2c4e29011c2d38",
        fee_tier=0.0001,
        tick_spacing=1,
    ),
    PoolConfig(
        name="sushiswap_v3_001",
        protocol="SushiSwap V3",
        contract="0x482fe995c4a52bc79271ab29a53591363ee30a89",
        fee_tier=0.0001,
        tick_spacing=1,
    ),
]

POOL_BY_NAME: dict[str, PoolConfig] = {p.name: p for p in POOLS}
POOL_BY_CONTRACT: dict[str, PoolConfig] = {p.contract.lower(): p for p in POOLS}

# ── Token ordering for WETH/USDC on Base ──────────────────────────────────────
# USDC on Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913  (decimals=6)
# WETH on Base: 0x4200000000000000000000000000000000000006  (decimals=18)
# token0 < token1 by address: USDC < WETH → token0=USDC, token1=WETH
# sqrtPriceX96 encodes sqrt(token1/token0) = sqrt(WETH/USDC)
# pool price in USDC per WETH = (sqrtPriceX96 / 2**96)**2 * 10**(decimals0 - decimals1)
#                              = (sqrtPriceX96 / 2**96)**2 * 10**(6-18)
USDC_ADDRESS: str = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
WETH_ADDRESS: str = "0x4200000000000000000000000000000000000006"
TOKEN0_DECIMALS: int = 6   # USDC
TOKEN1_DECIMALS: int = 18  # WETH

# ── Range width buckets (Section 4.3) ─────────────────────────────────────────
WIDTH_BUCKETS: list[tuple[float, float, str]] = [
    (0.0,      10.0,    "(0, 10)"),
    (10.0,     100.0,   "[10, 100)"),
    (100.0,    1000.0,  "[100, 1000)"),
    (1000.0,   10000.0, "[1000, 10000)"),
    (10000.0,  float("inf"), "> 10000"),
]

# ── Win-score threshold ────────────────────────────────────────────────────────
WIN_SCORE_THRESHOLD: float = 0.5

# ── The Graph subgraph endpoints for Base network ──────────────────────────────
# These are community/protocol subgraphs; set GRAPH_API_KEY in environment.
# Users can override these URLs via environment variables.
GRAPH_ENDPOINTS: dict[str, str] = {
    "uniswap_v3_005": (
        "https://gateway.thegraph.com/api/{GRAPH_API_KEY}"
        "/subgraphs/id/GqzP4Xaehti8KSfQmv3ZctFSjnSUYZ4En5NRsiTbvZpz"
    ),
    # Aerodrome, PancakeSwap, SushiSwap subgraphs on Base may require
    # separate subgraph IDs or custom indexers; configure via environment.
    "aerodrome_slipstream_005": "",
    "pancakeswap_v3_001": "",
    "sushiswap_v3_001": "",
}
