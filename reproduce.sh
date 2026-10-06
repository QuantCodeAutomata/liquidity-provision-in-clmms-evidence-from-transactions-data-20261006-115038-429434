#!/usr/bin/env bash
# reproduce.sh – entry point for full experiment reproduction
# Paper: "Liquidity provision in CLMMs: evidence from transactions data"
# Authors: Andrey Urusov, Rostislav Berezovskiy, Anatoly Krestenko, Andrei Kornilov

set -u

cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo ""
echo "=== Generating synthetic data (fallback when The Graph is unavailable) ==="
echo "NOTE: Set THEGRAPH_BASE_URL and THEGRAPH_API_KEY in .env for live on-chain data."
python3 data/generate_synthetic.py

echo ""
echo "=== Running Experiment 1: LP PnL Reconstruction ==="
python3 exp/exp1_pnl_reconstruction.py

echo ""
echo "=== Running Experiment 2: Win-Score LP Classification ==="
python3 exp/exp2_win_score.py

echo ""
echo "=== Running Experiment 3: Mono-LP / Multi-LP Analysis ==="
python3 exp/exp3_mono_multi_lp.py

echo ""
echo "=== Running Experiment 4: 15-Type Position Taxonomy ==="
python3 exp/exp4_position_types.py

echo ""
echo "=== Running Experiment 5: Win-Position Type Analysis ==="
python3 exp/exp5_win_position_analysis.py

echo ""
echo "=== Running test suite ==="
python3 -m pytest tests/ -v --tb=short

echo ""
echo "=== Reproduction complete. Results saved to results/ ==="
echo "See results/RESULTS.md for a full comparison against paper-reported values."
