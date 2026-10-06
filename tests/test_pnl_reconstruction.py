"""
Tests for src/pnl_reconstruction.py using purely synthetic data.

No file I/O or API calls – all DataFrames are constructed in-memory.
"""

import numpy as np
import pandas as pd
import pytest

from src.pnl_reconstruction import (
    LPResult,
    ReconstructedPosition,
    filter_lps_with_mint_and_burn,
    lp_results_to_summary,
    match_burn_collect,
    positions_to_dataframe,
    reconstruct_lp_range,
    remove_zero_value_events,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

PRICE = 3000.0  # constant synthetic USDC/WETH price

def _constant_price_series(lo: int = 0, hi: int = 1_000) -> pd.Series:
    """Return a pd.Series with timestamps lo..hi all priced at PRICE."""
    return pd.Series(PRICE, index=range(lo, hi + 1), dtype=float)


def _mint_row(
    block: int,
    ts: int,
    lp: str,
    liq: float,
    amount0_usdc_raw: float,
    amount1_weth_raw: float = 0.0,
    tx_hash: str = "0xmint",
    tick_lower: int = -887_272,
    tick_upper: int = 887_272,
) -> dict:
    return {
        "tx_hash": tx_hash,
        "block_number": block,
        "tx_index": 0,
        "log_index": 0,
        "timestamp": ts,
        "lp_address": lp,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity": liq,
        "amount0": amount0_usdc_raw,
        "amount1": amount1_weth_raw,
    }


def _burn_plus_row(
    block: int,
    ts: int,
    lp: str,
    liq: float,
    amount0_usdc_raw: float,
    amount1_weth_raw: float = 0.0,
    tx_hash: str = "0xburn",
    tick_lower: int = -887_272,
    tick_upper: int = 887_272,
) -> dict:
    return {
        "tx_hash": tx_hash,
        "block_number": block,
        "tx_index": 0,
        "log_index": 0,
        "timestamp": ts,
        "lp_address": lp,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity": liq,
        "amount0": amount0_usdc_raw,
        "amount1": amount1_weth_raw,
    }


def _swap_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"tx_hash": "0xswap", "block_number": 0, "tx_index": 0,
          "log_index": 0, "timestamp": 0, "price_usdc": PRICE}]
    )


# ── 1. remove_zero_value_events ────────────────────────────────────────────────

class TestRemoveZeroValueEvents:
    def _make_events(self) -> dict:
        return {
            "swap": _swap_df(),
            "mint": pd.DataFrame([
                {"liquidity": 100, "amount0": 0, "amount1": 0},
                {"liquidity": 0,   "amount0": 0, "amount1": 0},  # removed
            ]),
            "burn": pd.DataFrame([
                {"liquidity": 50,  "amount0": 0, "amount1": 0},
                {"liquidity": 0,   "amount0": 0, "amount1": 0},  # removed
                {"liquidity": 200, "amount0": 0, "amount1": 0},
            ]),
            "collect": pd.DataFrame([
                {"amount0": 100, "amount1": 0},    # kept: amount0 > 0
                {"amount0": 0,   "amount1": 50},   # kept: amount1 > 0
                {"amount0": 0,   "amount1": 0},    # removed: both zero
            ]),
        }

    def test_mint_zero_liquidity_removed(self):
        events = self._make_events()
        result = remove_zero_value_events(events)
        assert len(result["mint"]) == 1
        assert float(result["mint"].iloc[0]["liquidity"]) == 100

    def test_burn_zero_liquidity_removed(self):
        events = self._make_events()
        result = remove_zero_value_events(events)
        assert len(result["burn"]) == 2
        assert all(float(r) > 0 for r in result["burn"]["liquidity"])

    def test_collect_both_zero_removed(self):
        events = self._make_events()
        result = remove_zero_value_events(events)
        assert len(result["collect"]) == 2

    def test_collect_amount0_only_kept(self):
        events = self._make_events()
        result = remove_zero_value_events(events)
        amount0_vals = result["collect"]["amount0"].tolist()
        assert 100 in amount0_vals

    def test_swap_not_filtered(self):
        events = self._make_events()
        result = remove_zero_value_events(events)
        assert len(result["swap"]) == len(events["swap"])

    def test_returns_dict_with_all_keys(self):
        events = self._make_events()
        result = remove_zero_value_events(events)
        assert set(result.keys()) == {"mint", "burn", "collect", "swap"}


# ── 2. filter_lps_with_mint_and_burn ──────────────────────────────────────────

class TestFilterLpsWithMintAndBurn:
    def test_intersection_only(self):
        mint_df = pd.DataFrame({"lp_address": ["0xAAA", "0xBBB"]})
        burn_df = pd.DataFrame({"lp_address": ["0xAAA", "0xCCC"]})
        eligible = filter_lps_with_mint_and_burn(mint_df, burn_df)
        assert "0xaaa" in eligible
        assert "0xbbb" not in eligible
        assert "0xccc" not in eligible

    def test_case_insensitive(self):
        mint_df = pd.DataFrame({"lp_address": ["0xDeAdBeEf"]})
        burn_df = pd.DataFrame({"lp_address": ["0xDEADBEEF"]})
        eligible = filter_lps_with_mint_and_burn(mint_df, burn_df)
        assert "0xdeadbeef" in eligible

    def test_mint_only_lp_excluded(self):
        mint_df = pd.DataFrame({"lp_address": ["0xMINT"]})
        burn_df = pd.DataFrame({"lp_address": ["0xBURN"]})
        eligible = filter_lps_with_mint_and_burn(mint_df, burn_df)
        assert len(eligible) == 0

    def test_multiple_eligible(self):
        mint_df = pd.DataFrame({"lp_address": ["0xA", "0xB", "0xC"]})
        burn_df = pd.DataFrame({"lp_address": ["0xA", "0xB", "0xD"]})
        eligible = filter_lps_with_mint_and_burn(mint_df, burn_df)
        assert eligible == {"0xa", "0xb"}

    def test_returns_set(self):
        mint_df = pd.DataFrame({"lp_address": ["0xAAA"]})
        burn_df = pd.DataFrame({"lp_address": ["0xAAA"]})
        result = filter_lps_with_mint_and_burn(mint_df, burn_df)
        assert isinstance(result, set)


# ── 3. match_burn_collect ──────────────────────────────────────────────────────

class TestMatchBurnCollect:
    LP = "0x1111"
    TL, TU = -100, 100

    def _burn_df(self, tx_hash: str = "0xabc") -> pd.DataFrame:
        # amount0/amount1 must be present so the merge produces _burn/_collect
        # suffixes, enabling the prefer-collect fallback logic in match_burn_collect.
        return pd.DataFrame([{
            "tx_hash": tx_hash,
            "block_number": 100,
            "tx_index": 0,
            "log_index": 0,
            "timestamp": 200,
            "lp_address": self.LP,
            "tick_lower": self.TL,
            "tick_upper": self.TU,
            "liquidity": 1_000,
            "amount0": 0.0,
            "amount1": 0.0,
        }])

    def _collect_df(
        self,
        tx_hash: str = "0xabc",
        amount0: float = 1_200e6,
        amount1: float = 0.5e18,
    ) -> pd.DataFrame:
        return pd.DataFrame([{
            "tx_hash": tx_hash,
            "tick_lower": self.TL,
            "tick_upper": self.TU,
            "lp_address": self.LP,
            "amount0": amount0,
            "amount1": amount1,
        }])

    def test_basic_match_returns_one_row(self):
        result = match_burn_collect(self._burn_df(), self._collect_df(), self.LP)
        assert len(result) == 1

    def test_collect_amounts_used(self):
        result = match_burn_collect(
            self._burn_df(), self._collect_df(amount0=1_200e6, amount1=0.5e18), self.LP
        )
        assert result.iloc[0]["amount0"] == pytest.approx(1_200e6)
        assert result.iloc[0]["amount1"] == pytest.approx(0.5e18)

    def test_liquidity_comes_from_burn(self):
        result = match_burn_collect(self._burn_df(), self._collect_df(), self.LP)
        assert float(result.iloc[0]["liquidity"]) == pytest.approx(1_000.0)

    def test_no_collect_gives_zero_amounts(self):
        result = match_burn_collect(
            self._burn_df(), pd.DataFrame(columns=["tx_hash", "tick_lower", "tick_upper",
                                                    "lp_address", "amount0", "amount1"]),
            self.LP,
        )
        assert len(result) == 1
        assert float(result.iloc[0]["amount0"]) == pytest.approx(0.0)

    def test_multiple_collects_aggregated(self):
        """Two collect events in same tx for same range are summed."""
        collect_df = pd.DataFrame([
            {"tx_hash": "0xabc", "tick_lower": self.TL, "tick_upper": self.TU,
             "lp_address": self.LP, "amount0": 500e6, "amount1": 0.0},
            {"tx_hash": "0xabc", "tick_lower": self.TL, "tick_upper": self.TU,
             "lp_address": self.LP, "amount0": 700e6, "amount1": 0.0},
        ])
        result = match_burn_collect(self._burn_df(), collect_df, self.LP)
        assert len(result) == 1
        assert result.iloc[0]["amount0"] == pytest.approx(1_200e6)

    def test_wrong_lp_excluded(self):
        burn_other = self._burn_df()
        burn_other["lp_address"] = "0x9999"
        result = match_burn_collect(burn_other, self._collect_df(), self.LP)
        assert len(result) == 0

    def test_result_columns(self):
        result = match_burn_collect(self._burn_df(), self._collect_df(), self.LP)
        expected = {
            "tx_hash", "block_number", "tx_index", "log_index",
            "timestamp", "lp_address", "tick_lower", "tick_upper",
            "liquidity", "amount0", "amount1",
        }
        assert expected.issubset(set(result.columns))


# ── 4. reconstruct_lp_range – simple one-mint one-burn+ position ───────────────

class TestReconstructSimplePosition:
    LP = "0x2222"
    POOL = "test_pool"
    TL, TU = -887_272, 887_272
    PL, PU = 1.0, 9_999_999.0

    def test_pnl_equals_close_minus_open(self):
        """
        Mint 1000 USDC, close for 1200 USDC → PnL = +200 USDC.
        All amounts in raw on-chain units (amount0 in USDC-raw with 1e6 decimals).
        """
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=1_000,
                      amount0_usdc_raw=1_000 * 1e6, tx_hash="0xmint1")
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=2, ts=200, lp=self.LP, liq=1_000,
                           amount0_usdc_raw=1_200 * 1e6, tx_hash="0xburn1")
        ])

        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )

        assert len(positions) == 1
        pos = positions[0]
        assert pos.open_capital == pytest.approx(1_000.0, rel=1e-9)
        assert pos.close_capital == pytest.approx(1_200.0, rel=1e-9)
        assert pos.pnl == pytest.approx(200.0, rel=1e-9)
        assert pos.pnl == pytest.approx(pos.close_capital - pos.open_capital, abs=1e-9)

    def test_n_burns_is_one(self):
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=1_000,
                      amount0_usdc_raw=1_000 * 1e6)
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=2, ts=200, lp=self.LP, liq=1_000,
                           amount0_usdc_raw=1_000 * 1e6)
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        assert positions[0].n_burns == 1

    def test_metadata_fields(self):
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=1_000,
                      amount0_usdc_raw=1_000 * 1e6)
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=2, ts=200, lp=self.LP, liq=1_000,
                           amount0_usdc_raw=1_000 * 1e6)
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        pos = positions[0]
        assert pos.pool == self.POOL
        assert pos.lp_address == self.LP
        assert pos.open_timestamp == 100
        assert pos.close_timestamp == 200
        assert pos.tick_lower == self.TL
        assert pos.tick_upper == self.TU

    def test_open_price_from_price_series(self):
        """open_price is looked up at open_timestamp from the price series."""
        ps = pd.Series({50: 2800.0, 100: 3100.0, 200: 3200.0})
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=1_000,
                      amount0_usdc_raw=0, amount1_weth_raw=1e18)
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=2, ts=200, lp=self.LP, liq=1_000,
                           amount0_usdc_raw=0, amount1_weth_raw=1e18)
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        # open_price at ts=100 should be 3100.0 from the series
        assert positions[0].open_price == pytest.approx(3100.0)

    def test_no_burn_produces_no_positions(self):
        """An unclosed mint is discarded per paper definition."""
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=1_000,
                      amount0_usdc_raw=1_000 * 1e6)
        ])
        burn_plus_range = pd.DataFrame(columns=[
            "tx_hash", "block_number", "tx_index", "log_index", "timestamp",
            "lp_address", "tick_lower", "tick_upper", "liquidity", "amount0", "amount1",
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        assert positions == []


# ── 5. reconstruct_lp_range – overshooting burn+ splits proportionally ──────────

class TestReconstructOvershootingBurn:
    LP = "0x3333"
    POOL = "overshoot_pool"
    TL, TU = -887_272, 887_272
    PL, PU = 1.0, 9_999_999.0

    def test_two_positions_from_one_overshooting_burn(self):
        """
        Two sequential mints each with 500 liquidity.
        One burn+ of 1000 liquidity exactly covers both → fraction=0.5 for first position.

        mint1: open_capital = 500 USDC
        mint2: open_capital = 700 USDC
        burn:  total capital = 1500 USDC, split 50%/50% by liquidity

        pos1 close_capital = 750, pnl = +250
        pos2 close_capital = 750, pnl = +50
        """
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=500,
                      amount0_usdc_raw=500 * 1e6, tx_hash="0xmA"),
            _mint_row(block=2, ts=200, lp=self.LP, liq=500,
                      amount0_usdc_raw=700 * 1e6, tx_hash="0xmB"),
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=3, ts=300, lp=self.LP, liq=1_000,
                           amount0_usdc_raw=1_500 * 1e6, tx_hash="0xbig"),
        ])

        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )

        assert len(positions) == 2

        pos1, pos2 = positions
        assert pos1.open_capital == pytest.approx(500.0, rel=1e-9)
        assert pos1.close_capital == pytest.approx(750.0, rel=1e-9)
        assert pos1.pnl == pytest.approx(250.0, rel=1e-9)

        assert pos2.open_capital == pytest.approx(700.0, rel=1e-9)
        assert pos2.close_capital == pytest.approx(750.0, rel=1e-9)
        assert pos2.pnl == pytest.approx(50.0, rel=1e-9)

    def test_total_capital_conserved(self):
        """Sum of close_capitals equals total burn capital."""
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=500,
                      amount0_usdc_raw=500 * 1e6, tx_hash="0xmA"),
            _mint_row(block=2, ts=200, lp=self.LP, liq=500,
                      amount0_usdc_raw=700 * 1e6, tx_hash="0xmB"),
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=3, ts=300, lp=self.LP, liq=1_000,
                           amount0_usdc_raw=1_500 * 1e6, tx_hash="0xbig"),
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        total_close = sum(p.close_capital for p in positions)
        assert total_close == pytest.approx(1_500.0, rel=1e-9)


# ── 6. reconstruct_lp_range – pre-window burn+ excluded ───────────────────────

class TestReconstructExcludesPrewindowBurns:
    LP = "0x4444"
    POOL = "prewin_pool"
    TL, TU = -887_272, 887_272
    PL, PU = 1.0, 9_999_999.0

    def test_burn_before_mint_excluded(self):
        """
        A burn+ at block 1 (before any mint) has liquidity > cum_minted=0.
        It must be excluded per Section 2.1 step 3.
        The subsequent mint at block 2 has no burn to close it → no positions.
        """
        ps = _constant_price_series()
        # burn arrives first in chronological order
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=1, ts=50, lp=self.LP, liq=500,
                           amount0_usdc_raw=600 * 1e6, tx_hash="0xpre"),
        ])
        mint_range = pd.DataFrame([
            _mint_row(block=2, ts=100, lp=self.LP, liq=500,
                      amount0_usdc_raw=500 * 1e6, tx_hash="0xmint"),
        ])

        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        assert positions == []

    def test_burn_after_mint_not_excluded(self):
        """A burn+ after a mint is retained and closes the position normally."""
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=1, ts=100, lp=self.LP, liq=500,
                      amount0_usdc_raw=500 * 1e6, tx_hash="0xmint"),
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=2, ts=200, lp=self.LP, liq=500,
                           amount0_usdc_raw=600 * 1e6, tx_hash="0xpost"),
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        assert len(positions) == 1
        assert positions[0].pnl == pytest.approx(100.0, rel=1e-9)

    def test_valid_burn_after_excluded_prewindow_burn(self):
        """
        pre-window burn+ (block=1) is skipped; valid burn+ (block=3) closes mint.
        """
        ps = _constant_price_series()
        mint_range = pd.DataFrame([
            _mint_row(block=2, ts=100, lp=self.LP, liq=500,
                      amount0_usdc_raw=500 * 1e6, tx_hash="0xmint"),
        ])
        burn_plus_range = pd.DataFrame([
            _burn_plus_row(block=1, ts=50, lp=self.LP, liq=500,
                           amount0_usdc_raw=999 * 1e6, tx_hash="0xpre"),
            _burn_plus_row(block=3, ts=200, lp=self.LP, liq=500,
                           amount0_usdc_raw=600 * 1e6, tx_hash="0xpost"),
        ])
        positions = reconstruct_lp_range(
            lp_address=self.LP, pool=self.POOL,
            tick_lower=self.TL, tick_upper=self.TU,
            price_lower=self.PL, price_upper=self.PU,
            mint_range=mint_range, burn_plus_range=burn_plus_range,
            price_series=ps,
        )
        assert len(positions) == 1
        assert positions[0].close_capital == pytest.approx(600.0, rel=1e-9)


# ── 7. positions_to_dataframe ──────────────────────────────────────────────────

class TestPositionsToDataframe:
    def _make_position(
        self, pool: str, lp: str, pnl: float = 100.0
    ) -> ReconstructedPosition:
        return ReconstructedPosition(
            pool=pool, lp_address=lp,
            tick_lower=-100, tick_upper=100,
            price_lower=2_800.0, price_upper=3_200.0,
            open_timestamp=1_000, close_timestamp=2_000,
            open_price=3_000.0, close_price_weighted=3_100.0,
            open_capital=1_000.0, close_capital=1_000.0 + pnl,
            pnl=pnl, n_burns=1,
        )

    def test_returns_dataframe(self):
        res = LPResult(pool="p1", lp_address="0xA",
                       positions=[self._make_position("p1", "0xA")])
        df = positions_to_dataframe([res])
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        res1 = LPResult(pool="p1", lp_address="0xA",
                        positions=[self._make_position("p1", "0xA", pnl=50)])
        res2 = LPResult(pool="p1", lp_address="0xB",
                        positions=[self._make_position("p1", "0xB", pnl=200),
                                   self._make_position("p1", "0xB", pnl=-30)])
        res3 = LPResult(pool="p2", lp_address="0xC", positions=[])  # no positions
        df = positions_to_dataframe([res1, res2, res3])
        assert len(df) == 3  # 1 + 2 + 0

    def test_required_columns_present(self):
        res = LPResult(pool="p1", lp_address="0xA",
                       positions=[self._make_position("p1", "0xA")])
        df = positions_to_dataframe([res])
        required = {
            "pool", "lp_address", "tick_lower", "tick_upper",
            "price_lower", "price_upper",
            "open_timestamp", "close_timestamp",
            "open_price", "close_price_weighted",
            "open_capital", "close_capital", "pnl", "n_burns",
        }
        assert required.issubset(set(df.columns))

    def test_pnl_values_correct(self):
        pnls = [150.0, -75.0, 0.0]
        positions = [self._make_position("p1", "0xA", pnl=v) for v in pnls]
        res = LPResult(pool="p1", lp_address="0xA", positions=positions)
        df = positions_to_dataframe([res])
        assert df["pnl"].tolist() == pytest.approx(pnls, abs=1e-9)

    def test_empty_results_returns_empty_df(self):
        df = positions_to_dataframe([])
        assert len(df) == 0
        assert "pnl" in df.columns

    def test_pnl_numeric_type(self):
        res = LPResult(pool="p1", lp_address="0xA",
                       positions=[self._make_position("p1", "0xA")])
        df = positions_to_dataframe([res])
        assert pd.api.types.is_numeric_dtype(df["pnl"])
        assert pd.api.types.is_numeric_dtype(df["open_capital"])

    def test_lp_results_to_summary_aggregates_pnl(self):
        """lp_results_to_summary sums PnL across all positions for one LP."""
        positions = [
            self._make_position("p1", "0xA", pnl=100.0),
            self._make_position("p1", "0xA", pnl=-30.0),
        ]
        res = LPResult(pool="p1", lp_address="0xA", positions=positions)
        summary = lp_results_to_summary([res])
        assert summary.iloc[0]["terminal_pnl"] == pytest.approx(70.0, abs=1e-9)
        assert summary.iloc[0]["n_positions"] == 2
