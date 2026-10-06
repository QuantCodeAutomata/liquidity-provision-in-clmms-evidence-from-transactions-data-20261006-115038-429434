"""
Tests for src/position_types.py using purely synthetic data.

All 15 position types from Section 2.3 / Figure 3 of the paper are exercised
with carefully chosen numeric (p_start, p_end, a, b) tuples.
"""

import numpy as np
import pandas as pd
import pytest

from src.position_types import (
    LOSS_TYPES,
    NEUTRAL_TYPES,
    WIN_TYPES,
    classify_position_type,
    classify_positions_df,
    position_type_distribution,
)

# Shared range boundaries used for most single-type tests
A, B = 100.0, 200.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _classify(ps: float, pe: float) -> int:
    """Classify with the shared (A, B) range."""
    return classify_position_type(ps, pe, A, B)


def _make_pos_df(records: list[dict]) -> pd.DataFrame:
    """Build a minimal positions DataFrame for classify_positions_df."""
    defaults = {
        "open_price": 0.0,
        "close_price_weighted": 0.0,
        "price_lower": A,
        "price_upper": B,
        "open_capital": 1.0,
    }
    rows = [{**defaults, **r} for r in records]
    return pd.DataFrame(rows)


# ── Types 1-12: directional (rising and falling) ───────────────────────────────

class TestType1AllBelowRising:
    """P_start < P_end <= a  → Type 1 (win)"""

    def test_strictly_below(self):
        assert _classify(ps=50.0, pe=80.0) == 1

    def test_pe_equal_to_a(self):
        # pe == a satisfies pe <= a
        assert _classify(ps=50.0, pe=100.0) == 1

    def test_in_win_types(self):
        assert 1 in WIN_TYPES


class TestType2AllBelowFalling:
    """P_end < P_start <= a  → Type 2 (loss)"""

    def test_strictly_below(self):
        assert _classify(ps=90.0, pe=60.0) == 2

    def test_ps_equal_to_a(self):
        # ps == a satisfies ps <= a
        assert _classify(ps=100.0, pe=70.0) == 2

    def test_in_loss_types(self):
        assert 2 in LOSS_TYPES


class TestType3CrossLowerRising:
    """P_start < a < P_end <= b  → Type 3 (win)"""

    def test_basic(self):
        assert _classify(ps=50.0, pe=150.0) == 3

    def test_pe_equal_to_b(self):
        # pe == b satisfies pe <= b
        assert _classify(ps=50.0, pe=200.0) == 3

    def test_in_win_types(self):
        assert 3 in WIN_TYPES


class TestType4CrossLowerFalling:
    """P_end < a < P_start <= b  → Type 4 (loss)"""

    def test_basic(self):
        assert _classify(ps=150.0, pe=50.0) == 4

    def test_ps_equal_to_b(self):
        # ps == b satisfies ps <= b
        assert _classify(ps=200.0, pe=50.0) == 4

    def test_in_loss_types(self):
        assert 4 in LOSS_TYPES


class TestType5InRangeRising:
    """a <= P_start < P_end <= b  → Type 5 (win)"""

    def test_interior(self):
        assert _classify(ps=120.0, pe=180.0) == 5

    def test_ps_equal_to_a(self):
        assert _classify(ps=100.0, pe=180.0) == 5

    def test_pe_equal_to_b(self):
        assert _classify(ps=120.0, pe=200.0) == 5

    def test_both_boundaries(self):
        assert _classify(ps=100.0, pe=200.0) == 5

    def test_in_win_types(self):
        assert 5 in WIN_TYPES


class TestType6InRangeFalling:
    """a <= P_end < P_start <= b  → Type 6 (loss)"""

    def test_interior(self):
        assert _classify(ps=180.0, pe=120.0) == 6

    def test_pe_equal_to_a(self):
        assert _classify(ps=180.0, pe=100.0) == 6

    def test_ps_equal_to_b(self):
        assert _classify(ps=200.0, pe=120.0) == 6

    def test_in_loss_types(self):
        assert 6 in LOSS_TYPES


class TestType7CrossUpperRising:
    """a <= P_start < b < P_end  → Type 7 (win)"""

    def test_basic(self):
        assert _classify(ps=150.0, pe=250.0) == 7

    def test_ps_equal_to_a(self):
        assert _classify(ps=100.0, pe=250.0) == 7

    def test_in_win_types(self):
        assert 7 in WIN_TYPES


class TestType8CrossUpperFalling:
    """a <= P_end < b < P_start  → Type 8 (loss)"""

    def test_basic(self):
        assert _classify(ps=250.0, pe=150.0) == 8

    def test_pe_equal_to_a(self):
        assert _classify(ps=250.0, pe=100.0) == 8

    def test_in_loss_types(self):
        assert 8 in LOSS_TYPES


class TestType9FullTraversalRising:
    """P_start < a < b < P_end  → Type 9 (win)"""

    def test_basic(self):
        assert _classify(ps=50.0, pe=300.0) == 9

    def test_in_win_types(self):
        assert 9 in WIN_TYPES


class TestType10FullTraversalFalling:
    """P_end < a < b < P_start  → Type 10 (loss)"""

    def test_basic(self):
        assert _classify(ps=300.0, pe=50.0) == 10

    def test_in_loss_types(self):
        assert 10 in LOSS_TYPES


class TestType11AboveRangeRising:
    """b <= P_start < P_end  → Type 11 (win)"""

    def test_interior(self):
        assert _classify(ps=220.0, pe=280.0) == 11

    def test_ps_equal_to_b(self):
        assert _classify(ps=200.0, pe=280.0) == 11

    def test_in_win_types(self):
        assert 11 in WIN_TYPES


class TestType12AboveRangeFalling:
    """b <= P_end < P_start  → Type 12 (loss)"""

    def test_interior(self):
        assert _classify(ps=280.0, pe=220.0) == 12

    def test_pe_equal_to_b(self):
        assert _classify(ps=280.0, pe=200.0) == 12

    def test_in_loss_types(self):
        assert 12 in LOSS_TYPES


# ── Types 13-15: neutral (P_start ≈ P_end) ────────────────────────────────────

class TestType13NeutralInRange:
    """a < P_start = P_end < b  → Type 13"""

    def test_basic(self):
        assert _classify(ps=150.0, pe=150.0) == 13

    def test_near_lower_boundary(self):
        assert _classify(ps=101.0, pe=101.0) == 13

    def test_near_upper_boundary(self):
        assert _classify(ps=199.0, pe=199.0) == 13

    def test_in_neutral_types(self):
        assert 13 in NEUTRAL_TYPES


class TestType14NeutralBelow:
    """P_start = P_end < a  → Type 14"""

    def test_basic(self):
        assert _classify(ps=50.0, pe=50.0) == 14

    def test_near_lower_boundary(self):
        assert _classify(ps=99.0, pe=99.0) == 14

    def test_in_neutral_types(self):
        assert 14 in NEUTRAL_TYPES


class TestType15NeutralAbove:
    """b < P_start = P_end  → Type 15"""

    def test_basic(self):
        assert _classify(ps=250.0, pe=250.0) == 15

    def test_near_upper_boundary(self):
        assert _classify(ps=201.0, pe=201.0) == 15

    def test_in_neutral_types(self):
        assert 15 in NEUTRAL_TYPES


# ── classify_positions_df ──────────────────────────────────────────────────────

class TestClassifyPositionsDf:
    def test_adds_position_type_column(self):
        df = _make_pos_df([
            {"open_price": 50.0, "close_price_weighted": 80.0},   # type 1
            {"open_price": 120.0, "close_price_weighted": 180.0}, # type 5
        ])
        result = classify_positions_df(df)
        assert "position_type" in result.columns

    def test_correct_types_assigned(self):
        df = _make_pos_df([
            {"open_price": 50.0,  "close_price_weighted": 80.0},   # type 1
            {"open_price": 120.0, "close_price_weighted": 180.0},  # type 5
            {"open_price": 150.0, "close_price_weighted": 250.0},  # type 7
            {"open_price": 50.0,  "close_price_weighted": 300.0},  # type 9
        ])
        result = classify_positions_df(df)
        assert result["position_type"].tolist() == [1, 5, 7, 9]

    def test_neutral_type_in_df(self):
        df = _make_pos_df([
            {"open_price": 150.0, "close_price_weighted": 150.0},  # type 13
        ])
        result = classify_positions_df(df)
        assert result.iloc[0]["position_type"] == 13

    def test_missing_column_raises(self):
        df = pd.DataFrame([{"open_price": 100.0, "close_price_weighted": 150.0}])
        with pytest.raises(ValueError, match="missing columns"):
            classify_positions_df(df)

    def test_preserves_all_rows(self):
        rows = [{"open_price": float(x), "close_price_weighted": float(x + 10)}
                for x in range(10, 100, 10)]
        df = _make_pos_df(rows)
        result = classify_positions_df(df)
        assert len(result) == len(df)


# ── position_type_distribution ─────────────────────────────────────────────────

class TestTypeDistributionSumsTo100:
    def _all_types_df(self) -> pd.DataFrame:
        """One position for each of the 15 types, equal capital."""
        type_params = [
            # (ps, pe, a, b)
            (50.0, 80.0, A, B),   # 1
            (90.0, 60.0, A, B),   # 2
            (50.0, 150.0, A, B),  # 3
            (150.0, 50.0, A, B),  # 4
            (120.0, 180.0, A, B), # 5
            (180.0, 120.0, A, B), # 6
            (150.0, 250.0, A, B), # 7
            (250.0, 150.0, A, B), # 8
            (50.0, 300.0, A, B),  # 9
            (300.0, 50.0, A, B),  # 10
            (220.0, 280.0, A, B), # 11
            (280.0, 220.0, A, B), # 12
            (150.0, 150.0, A, B), # 13
            (50.0, 50.0, A, B),   # 14
            (250.0, 250.0, A, B), # 15
        ]
        rows = []
        for ps, pe, a, b in type_params:
            rows.append({
                "open_price": ps,
                "close_price_weighted": pe,
                "price_lower": a,
                "price_upper": b,
                "open_capital": 1.0,
            })
        df = pd.DataFrame(rows)
        return classify_positions_df(df)

    def test_all_15_types_present(self):
        df = self._all_types_df()
        dist = position_type_distribution(df)
        assert set(dist["position_type"].tolist()) == set(range(1, 16))

    def test_tx_share_pct_sums_to_100(self):
        df = self._all_types_df()
        dist = position_type_distribution(df)
        assert dist["tx_share_pct"].sum() == pytest.approx(100.0, abs=1e-9)

    def test_capital_share_pct_sums_to_100(self):
        df = self._all_types_df()
        dist = position_type_distribution(df)
        assert dist["capital_share_pct"].sum() == pytest.approx(100.0, abs=1e-9)

    def test_equal_capital_equal_share(self):
        """With equal open_capital for all 15 types, each gets 100/15 %."""
        df = self._all_types_df()
        dist = position_type_distribution(df)
        expected_pct = pytest.approx(100.0 / 15.0, rel=1e-9)
        for _, row in dist.iterrows():
            assert row["capital_share_pct"] == expected_pct

    def test_tx_count_column_present(self):
        df = self._all_types_df()
        dist = position_type_distribution(df)
        assert "tx_count" in dist.columns


# ── WIN / LOSS / NEUTRAL type membership ──────────────────────────────────────

class TestWinTypesAreProfitable:
    """All WIN_TYPES have P_end > P_start (rising price direction)."""

    @pytest.mark.parametrize("ptype,ps,pe", [
        (1,  50.0,  80.0),   # below-range rising
        (3,  50.0,  150.0),  # cross-lower rising
        (5,  120.0, 180.0),  # in-range rising
        (7,  150.0, 250.0),  # cross-upper rising
        (9,  50.0,  300.0),  # full-traversal rising
        (11, 220.0, 280.0),  # above-range rising
    ])
    def test_win_type_has_rising_price(self, ptype, ps, pe):
        assert classify_position_type(ps, pe, A, B) == ptype
        assert pe > ps, f"Type {ptype} expected pe > ps but ps={ps}, pe={pe}"

    @pytest.mark.parametrize("ptype,ps,pe", [
        (2,  90.0,  60.0),   # below-range falling
        (4,  150.0, 50.0),   # cross-lower falling
        (6,  180.0, 120.0),  # in-range falling
        (8,  250.0, 150.0),  # cross-upper falling
        (10, 300.0, 50.0),   # full-traversal falling
        (12, 280.0, 220.0),  # above-range falling
    ])
    def test_loss_type_has_falling_price(self, ptype, ps, pe):
        assert classify_position_type(ps, pe, A, B) == ptype
        assert ps > pe, f"Type {ptype} expected ps > pe but ps={ps}, pe={pe}"

    def test_win_types_are_exactly(self):
        assert WIN_TYPES == frozenset({1, 3, 5, 7, 9, 11})

    def test_loss_types_are_exactly(self):
        assert LOSS_TYPES == frozenset({2, 4, 6, 8, 10, 12})

    def test_neutral_types_are_exactly(self):
        assert NEUTRAL_TYPES == frozenset({13, 14, 15})

    def test_type_sets_disjoint(self):
        assert WIN_TYPES & LOSS_TYPES == frozenset()
        assert WIN_TYPES & NEUTRAL_TYPES == frozenset()
        assert LOSS_TYPES & NEUTRAL_TYPES == frozenset()

    def test_type_sets_cover_1_to_15(self):
        all_types = WIN_TYPES | LOSS_TYPES | NEUTRAL_TYPES
        assert all_types == frozenset(range(1, 16))


# ── Edge / degenerate cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_degenerate_range_returns_zero(self):
        assert classify_position_type(100.0, 150.0, 200.0, 200.0) == 0

    def test_inverted_range_returns_zero(self):
        assert classify_position_type(100.0, 150.0, 300.0, 100.0) == 0
