"""
Tests for src/win_score.py using purely synthetic data.

All DataFrames are constructed in-memory; no file I/O or API calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.win_score import classify_lps, compute_win_score, pool_classification_table


# ── Helpers ────────────────────────────────────────────────────────────────────

def _positions_df(
    records: list[dict],
) -> pd.DataFrame:
    """Build a minimal lp_positions_df for classify_lps."""
    return pd.DataFrame(records)


def _lp_row(
    pool: str,
    lp: str,
    terminal_pnl: float,
    win_score: float,
    pnl_positive: bool | None = None,
    consistently_successful: bool = False,
) -> dict:
    if pnl_positive is None:
        pnl_positive = terminal_pnl > 0
    return {
        "pool": pool,
        "lp_address": lp,
        "terminal_pnl": terminal_pnl,
        "win_score": win_score,
        "pnl_positive": pnl_positive,
        "consistently_successful": consistently_successful,
    }


# ── 1. test_all_wins ──────────────────────────────────────────────────────────

class TestAllWins:
    def test_omega_equals_one(self):
        """Monotonically increasing cumulative PnL → A- = 0 → ω = 1.0."""
        ts = [10, 20, 30]
        pnls = [100.0, 200.0, 300.0]
        omega = compute_win_score(ts, pnls, t_start=0, t_end=40)
        assert omega == pytest.approx(1.0, abs=1e-12)

    def test_single_positive_position(self):
        omega = compute_win_score([100], [500.0], t_start=0, t_end=200)
        assert omega == pytest.approx(1.0, abs=1e-12)

    def test_large_positive_values(self):
        ts = [1_000, 2_000, 3_000]
        pnls = [1e6, 2e6, 3e6]
        omega = compute_win_score(ts, pnls, t_start=0, t_end=4_000)
        assert omega == pytest.approx(1.0, abs=1e-12)


# ── 2. test_all_losses ────────────────────────────────────────────────────────

class TestAllLosses:
    def test_omega_equals_zero(self):
        """Monotonically decreasing cumulative PnL → A+ = 0 → ω = 0.0."""
        ts = [10, 20]
        pnls = [-100.0, -200.0]
        omega = compute_win_score(ts, pnls, t_start=0, t_end=30)
        assert omega == pytest.approx(0.0, abs=1e-12)

    def test_single_negative_position(self):
        omega = compute_win_score([50], [-999.0], t_start=0, t_end=100)
        assert omega == pytest.approx(0.0, abs=1e-12)


# ── 3. test_zero_pnl ──────────────────────────────────────────────────────────

class TestZeroPnl:
    def test_all_zero_pnls_return_half(self):
        """S = 0 → ω = 0.5 by definition (Appendix A)."""
        omega = compute_win_score([10, 20], [0.0, 0.0], t_start=0, t_end=30)
        assert omega == pytest.approx(0.5, abs=1e-12)

    def test_empty_positions_return_half(self):
        omega = compute_win_score([], [], t_start=0, t_end=100)
        assert omega == pytest.approx(0.5, abs=1e-12)

    def test_zero_single_position(self):
        omega = compute_win_score([50], [0.0], t_start=0, t_end=100)
        assert omega == pytest.approx(0.5, abs=1e-12)


# ── 4. test_alternating ───────────────────────────────────────────────────────

class TestAlternating:
    def test_omega_between_zero_and_one(self):
        """Alternating win/loss produces ω in (0, 1)."""
        # pnls: +100, -300, +200 → cum_pnl: 100, -200, 0
        ts = [10, 20, 30]
        pnls = [100.0, -300.0, 200.0]
        omega = compute_win_score(ts, pnls, t_start=0, t_end=40)
        assert 0.0 < omega < 1.0

    def test_alternating_exact_value(self):
        """
        cum_pnl steps: 0 → 100 → -200 → 0
        S = max(100, 200) = 200
        C_plus = [0, 0.5, 0, 0], C_minus = [0, 0, 1, 0]
        widths = [10, 10, 10, 10]
        A_plus = 5, A_minus = 10  →  ω = 5/15 = 1/3
        """
        ts = [10, 20, 30]
        pnls = [100.0, -300.0, 200.0]
        omega = compute_win_score(ts, pnls, t_start=0, t_end=40)
        assert omega == pytest.approx(1.0 / 3.0, rel=1e-9)


# ── 5. test_observation_window ────────────────────────────────────────────────

class TestObservationWindow:
    def test_different_t_end_produces_different_omega(self):
        """Extending t_end into a negative-PnL region must reduce ω."""
        # cum_pnl: 0 → 100 (at t=100) → -100 (at t=150)
        ts = [100, 150]
        pnls = [100.0, -200.0]

        omega_200 = compute_win_score(ts, pnls, t_start=0, t_end=200)
        omega_250 = compute_win_score(ts, pnls, t_start=0, t_end=250)

        assert omega_200 != omega_250
        assert omega_200 > omega_250  # more negative-PnL time → lower ω

    def test_t_end_200_exact(self):
        """
        Segments [0,100)=0, [100,150)=100, [150,200)=-100; widths 100, 50, 50.
        S=100; A_plus=50, A_minus=50  →  ω=0.5
        """
        omega = compute_win_score([100, 150], [100.0, -200.0],
                                  t_start=0, t_end=200)
        assert omega == pytest.approx(0.5, abs=1e-9)

    def test_t_end_250_exact(self):
        """
        Segments [0,100)=0, [100,150)=100, [150,250)=-100; widths 100, 50, 100.
        S=100; A_plus=50, A_minus=100  →  ω=1/3
        """
        omega = compute_win_score([100, 150], [100.0, -200.0],
                                  t_start=0, t_end=250)
        assert omega == pytest.approx(1.0 / 3.0, rel=1e-9)

    def test_t_start_shifts_baseline(self):
        """
        Using t_start=90 instead of 0 shrinks the first (zero) segment by 90 units.
        The relative ω for an all-positive series stays 1.0 regardless.
        """
        omega = compute_win_score([100], [500.0], t_start=90, t_end=200)
        assert omega == pytest.approx(1.0, abs=1e-12)


# ── 6. test_classify_lps_structure ────────────────────────────────────────────

class TestClassifyLpsStructure:
    def _positions_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "pool": "poolA",
                "lp_address": "0xAAA",
                "open_timestamp": 100,
                "close_timestamp": 200,
                "pnl": 500.0,
            },
            {
                "pool": "poolA",
                "lp_address": "0xAAA",
                "open_timestamp": 210,
                "close_timestamp": 350,
                "pnl": -100.0,
            },
            {
                "pool": "poolB",
                "lp_address": "0xBBB",
                "open_timestamp": 50,
                "close_timestamp": 150,
                "pnl": -200.0,
            },
        ])

    def test_returns_dataframe(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_pool_lp(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        assert len(df) == 2  # (poolA, 0xAAA) and (poolB, 0xBBB)

    def test_required_columns_present(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        required = {
            "pool", "lp_address", "terminal_pnl", "win_score",
            "pnl_positive", "consistently_successful", "n_positions",
            "t_start", "t_end",
        }
        assert required.issubset(set(df.columns))

    def test_terminal_pnl_is_sum(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        row_a = df[df["lp_address"] == "0xAAA"].iloc[0]
        assert row_a["terminal_pnl"] == pytest.approx(400.0, abs=1e-9)

    def test_win_score_range_in_output(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        assert all(0.0 <= w <= 1.0 for w in df["win_score"])

    def test_pnl_positive_flag(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        row_a = df[df["lp_address"] == "0xAAA"].iloc[0]
        row_b = df[df["lp_address"] == "0xBBB"].iloc[0]
        assert row_a["pnl_positive"] is True or row_a["pnl_positive"] == True
        assert row_b["pnl_positive"] is False or row_b["pnl_positive"] == False

    def test_n_positions_count(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        row_a = df[df["lp_address"] == "0xAAA"].iloc[0]
        assert row_a["n_positions"] == 2

    def test_t_end_equals_sample_end(self):
        df = classify_lps(self._positions_df(), 0, 1_000)
        assert all(df["t_end"] == 1_000)

    def test_missing_required_columns_raises(self):
        bad_df = pd.DataFrame([{"pool": "p", "lp_address": "0xA", "pnl": 0.0}])
        with pytest.raises(ValueError, match="missing columns"):
            classify_lps(bad_df, 0, 100)


# ── 7. test_pool_classification_table_totals ──────────────────────────────────

class TestPoolClassificationTableTotals:
    def _lp_class_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            _lp_row("poolA", "0xA1", 200.0,  0.8, True,  True),
            _lp_row("poolA", "0xA2", -50.0,  0.3, False, False),
            _lp_row("poolA", "0xA3", 100.0,  0.6, True,  True),
            _lp_row("poolB", "0xB1", -30.0,  0.2, False, False),
            _lp_row("poolB", "0xB2", 75.0,   0.7, True,  False),
        ])

    def test_returns_dataframe(self):
        df = pool_classification_table(self._lp_class_df())
        assert isinstance(df, pd.DataFrame)

    def test_total_row_exists(self):
        df = pool_classification_table(self._lp_class_df())
        assert "Total" in df["pool"].values

    def test_total_lp_count_correct(self):
        df = pool_classification_table(self._lp_class_df())
        total_row = df[df["pool"] == "Total"].iloc[0]
        assert total_row["lp_count"] == 5

    def test_per_pool_lp_counts(self):
        df = pool_classification_table(self._lp_class_df())
        pool_a = df[df["pool"] == "poolA"].iloc[0]
        pool_b = df[df["pool"] == "poolB"].iloc[0]
        assert pool_a["lp_count"] == 3
        assert pool_b["lp_count"] == 2

    def test_pnl_positive_count_correct(self):
        df = pool_classification_table(self._lp_class_df())
        pool_a = df[df["pool"] == "poolA"].iloc[0]
        assert pool_a["pnl_positive_count"] == 2  # 0xA1, 0xA3

    def test_consistent_count_correct(self):
        df = pool_classification_table(self._lp_class_df())
        pool_a = df[df["pool"] == "poolA"].iloc[0]
        assert pool_a["consistent_count"] == 2

    def test_total_pnl_positive_count_sum(self):
        df = pool_classification_table(self._lp_class_df())
        total_row = df[df["pool"] == "Total"].iloc[0]
        # 0xA1, 0xA3, 0xB2 are pnl_positive
        assert total_row["pnl_positive_count"] == 3

    def test_pnl_positive_pct_range(self):
        df = pool_classification_table(self._lp_class_df())
        for _, row in df.iterrows():
            assert 0.0 <= row["pnl_positive_pct"] <= 100.0

    def test_pool_a_pnl_positive_pct(self):
        df = pool_classification_table(self._lp_class_df())
        pool_a = df[df["pool"] == "poolA"].iloc[0]
        assert pool_a["pnl_positive_pct"] == pytest.approx(200.0 / 3.0, rel=1e-9)


# ── 8. test_win_score_range ───────────────────────────────────────────────────

class TestWinScoreRange:
    @pytest.mark.parametrize("ts,pnls,t_start,t_end", [
        ([10], [100.0], 0, 20),
        ([10], [-100.0], 0, 20),
        ([10, 20, 30], [50.0, -200.0, 100.0], 0, 40),
        ([5, 15, 25, 35], [300.0, -100.0, -50.0, 200.0], 0, 50),
        ([100], [0.0], 0, 200),
        ([], [], 0, 100),
    ])
    def test_omega_in_unit_interval(self, ts, pnls, t_start, t_end):
        omega = compute_win_score(ts, pnls, t_start=t_start, t_end=t_end)
        assert 0.0 <= omega <= 1.0, f"ω={omega} out of [0,1] for ts={ts}, pnls={pnls}"

    def test_symmetric_opposite_pnls(self):
        """+X then -X in equal time intervals → ω depends on segment widths."""
        omega = compute_win_score([50], [100.0], t_start=0, t_end=100)
        # Pos segment [50,100) has value=100; zero segment [0,50) has value=0
        # S=100; A_plus = 1*(100-50)=50, A_minus=0 → ω=1.0
        assert omega == pytest.approx(1.0, abs=1e-12)
