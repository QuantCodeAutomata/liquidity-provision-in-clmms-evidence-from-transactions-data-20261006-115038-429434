# Liquidity provision in CLMMs: evidence from transactions data

**Authors:** Andrey Urusov, Rostislav Berezovskiy, Anatoly Krestenko, Andrei Kornilov  
**Affiliation:** Vega Institute Foundation  
**Year:** 2024  

> Reference codebase: "Base LP Research: source code, data and scripts" (Reference [20] in the paper).

---

## Overview

This repository replicates the five core experiments from the paper using
transaction-level on-chain data from WETH/USDC concentrated-liquidity pools
on the Base network (Uniswap V3, Aerodrome SlipStream, PancakeSwap V3,
SushiSwap V3) over September 2024 – July 2025.

| Experiment | Title |
|---|---|
| exp_1 | Reconstruct LP PnL from on-chain transaction data |
| exp_2 | Compute the win-score metric and classify LPs by success consistency |
| exp_3 | Identify mono-LPs and multi-LPs across pools |
| exp_4 | Classify positions into 15 types and analyze dominant structures |
| exp_5 | Analyze profitable win-position types 3, 5, and 7 |

---

## Repository Structure

```
repo/
├── data/                  # Local event CSV/Parquet files (see Data section)
│   ├── uniswap_v3_005/
│   ├── aerodrome_slipstream_005/
│   ├── pancakeswap_v3_001/
│   └── sushiswap_v3_001/
├── src/                   # Core shared pipeline
│   ├── config.py          # Pool definitions, sample period, constants
│   ├── data_loader.py     # Event loading (CSV/Parquet + The Graph fallback)
│   ├── pnl_reconstruction.py  # Dynamic-balance PnL reconstruction (Section 2.1)
│   ├── win_score.py       # Win-score metric ω (Appendix A)
│   ├── lp_classification.py   # Mono/multi-LP classification (Section 4.2)
│   ├── position_types.py  # 15-type position taxonomy (Section 2.3)
│   ├── metrics.py         # Delta, boundary, return, duration metrics (Section 2.3)
│   └── pipeline.py        # End-to-end orchestration
├── exp/                   # Experiment entry points
│   ├── exp1_pnl_reconstruction.py
│   ├── exp2_win_score.py
│   ├── exp3_mono_multi_lp.py
│   ├── exp4_position_types.py
│   └── exp5_win_position_analysis.py
├── tests/                 # pytest suite (246 tests, all passing)
├── results/               # Output tables, plots, RESULTS.md
├── reproduce.sh           # One-command reproduction entry point
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the required key:

```bash
cp .env.example .env
# Edit .env: set GRAPH_API_KEY for The Graph fallback
```

---

## Data

The pipeline expects on-chain event data placed under `data/<pool_name>/`:

| Pool name | Protocol | Contract (Base) |
|---|---|---|
| `uniswap_v3_005` | Uniswap V3 0.05% | `0xd0b53d9277642d899df5c87a3966a349a798f224` |
| `aerodrome_slipstream_005` | Aerodrome SlipStream 0.05% | `0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59` |
| `pancakeswap_v3_001` | PancakeSwap V3 0.01% | `0x72ab388e2e2f6facef59e3c3fa2c4e29011c2d38` |
| `sushiswap_v3_001` | SushiSwap V3 0.01% | `0x482fe995c4a52bc79271ab29a53591363ee30a89` |

Each pool directory needs four files: `mint.csv`, `burn.csv`, `collect.csv`, `swap.csv`.  
See `src/data_loader.py` for the expected column schema.

If local files are absent and `GRAPH_API_KEY` is set, the pipeline will
fall back to fetching events via The Graph Protocol.

---

## Reproduction

```bash
./reproduce.sh
```

Or run experiments individually:

```bash
python exp/exp1_pnl_reconstruction.py   # generates intermediates for exp2-5
python exp/exp2_win_score.py
python exp/exp3_mono_multi_lp.py
python exp/exp4_position_types.py
python exp/exp5_win_position_analysis.py
```

---

## Tests

```bash
python -m pytest tests/ -v
```

246 tests covering all shared src/ modules with synthetic data.

---

## Key Results (see `results/RESULTS.md`)

| Metric | Paper | Replicated |
|---|---|---|
| LPs with positive terminal PnL | ~1 in 6 (~16.7%) | see RESULTS.md |
| Consistently successful LPs (PnL>0 and ω>0.5) | ~1 in 7 (~14.3%) | see RESULTS.md |
| Multi-LPs (cross-pool) | ~10% of unique LPs | see RESULTS.md |
| Types 5+6 share of transactions | >90% | see RESULTS.md |
| Type 5 Δ_start (midpoint placement) | ~47.8% | see RESULTS.md |
| Type 5 Δ (price traversal before close) | ~8% | see RESULTS.md |
