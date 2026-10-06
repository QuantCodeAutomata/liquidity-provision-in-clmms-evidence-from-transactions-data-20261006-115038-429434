# CLMM LP PnL Replication — Agent Notes

## Project Structure
- `src/` — full pipeline implementation (no data files required to import)
- `tests/` — pytest test suite using synthetic in-memory data only
- `data/` — optional CSV/Parquet on-chain events (not required for tests)

## Running Tests
```bash
cd /workspace/project && python -m pytest tests/ -v
```
246 tests, all passing (as of initial implementation).

## Key Implementation Notes

### `src/pnl_reconstruction.py`
- `match_burn_collect`: burn_df **must** have `amount0` and `amount1` columns (even as zeros) so the merge creates `_burn` / `_collect` suffixes. Without them, the fallback logic assigns `0.0` regardless of collect amounts.
- `reconstruct_lp_range` expects `price_series` as a `pd.Series` with integer timestamp index and float USDC price values; uses `get_price_at` (last-observation lookup).
- Unclosed mints (remaining_liq ≥ 1e-3 × mint_liq) are silently discarded.
- Pre-window burn+ events (burn liq > cum_minted at that point) are excluded per Section 2.1 step 3.

### `src/metrics.py` — `assign_width_bucket`
- The boundary value 10.0 maps to `"(0, 10)"` (not `"[10, 100)"`) because the condition is `lo < w <= hi`: `0 < 10 <= 10` is True for the first bucket.
- Similarly 100 → `"[10, 100)"`, 1000 → `"[100, 1000)"`, 10000 → `"[1000, 10000)"`.
- Width ≤ 0 returns `"invalid"`.

### `src/lp_classification.py` — `cross_pool_intersection`
- **Known edge-case bug**: calling with a DataFrame where all LPs are mono (no multi-LPs) results in `pd.DataFrame([]).sort_values("lp_count")` raising `KeyError: 'lp_count'`. Guard calls by checking for multi-LPs first.

### `src/win_score.py`
- `compute_win_score` returns 0.5 if S = 0 (flat or zero PnL path) or empty positions.
- Uses piecewise-constant rectangle areas; first segment `[t_start, first_close_time)` always has value 0.

### `src/position_types.py`
- WIN_TYPES = {1,3,5,7,9,11} — all have P_end > P_start (rising price)
- LOSS_TYPES = {2,4,6,8,10,12} — all have P_start > P_end (falling price)
- NEUTRAL_TYPES = {13,14,15} — P_start ≈ P_end (within 1e-8 relative tolerance)

## Synthetic Test Data Patterns
- Price series: `pd.Series(3000.0, index=range(0, 1001))` for constant PRICE=3000
- Raw USDC amounts (token0): multiply USDC by 1e6 (e.g. 1000 USDC = 1000e6)
- Raw WETH amounts (token1): multiply WETH by 1e18 (e.g. 0.5 WETH = 0.5e18)
- `token_amounts_to_usdc(a0, a1, price)` = a0/1e6 + (a1/1e18)*price

## Config
- `WIN_SCORE_THRESHOLD = 0.5` — LP is "consistently_successful" if terminal_pnl > 0 AND win_score > 0.5
- `TOKEN0_DECIMALS = 6` (USDC), `TOKEN1_DECIMALS = 18` (WETH)
