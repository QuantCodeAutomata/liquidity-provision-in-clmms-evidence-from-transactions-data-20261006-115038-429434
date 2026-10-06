"""
Tests for src/metrics.py using purely synthetic DataFrames.

Formulas under test (Section 2.3, paper equations 2, 3.1, 3.2):
  Δ_start = clip((P_start - a) / (b - a), 0, 1)
  Δ_end   = clip((P_end   - a) / (b - a), 0, 1)
  Δ       = Δ_end - Δ_start
  m_a = P_start/a - 1,  m_b = P_start/b - 1
  d_a = P_end/a   - 1,  d_b = P_end/b   - 1
  rel_a = d_a - m_a,    rel_b = d_b - m_b
  R_cap   = Q/W - 1
  R_price = P_end/P_start - 1
  duration_sec = close_ts - open_ts
  width_usdc   = b - a
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    assign_width_bucket,
    compute_position_metrics,
    lp_type_weighted_width,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _single_row(
    ps: float = 3_000.0,
    pe: float = 3_100.0,
    a: float = 2_900.0,
    b: float = 3_200.0,
    open_capital: float = 1_000.0,
    close_capital: float = 1_100.0,
    open_ts: int = 1_000,
    close_ts: int = 4_600,
    position_type: int = 5,
    lp_address: str = "0xAAA",
    pool: str = "poolA",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "open_price": ps,
        "close_price_weighted": pe,
        "price_lower": a,
        "price_upper": b,
        "open_capital": open_capital,
        "close_capital": close_capital,
        "open_timestamp": open_ts,
        "close_timestamp": close_ts,
        "position_type": position_type,
        "lp_address": lp_address,
        "pool": pool,
    }])


def _metrics(row_kw: dict | None = None) -> pd.Series:
    """Compute metrics for a single row and return the first row as a Series."""
    kwargs = row_kw or {}
    df = compute_position_metrics(_single_row(**kwargs))
    return df.iloc[0]


# ── 1. test_delta_type5 ────────────────────────────────────────────────────────

class TestDeltaType5:
    def test_delta_start(self):
        # a=100, b=200, ps=120 → (120-100)/(200-100) = 0.2
        row = _metrics({"ps": 120.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["delta_start"] == pytest.approx(0.2, rel=1e-9)

    def test_delta_end(self):
        row = _metrics({"ps": 120.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["delta_end"] == pytest.approx(0.6, rel=1e-9)

    def test_delta_equals_end_minus_start(self):
        row = _metrics({"ps": 120.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["delta"] == pytest.approx(
            row["delta_end"] - row["delta_start"], abs=1e-12
        )

    def test_delta_value(self):
        row = _metrics({"ps": 120.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["delta"] == pytest.approx(0.4, rel=1e-9)


# ── 2. test_delta_start_clips_to_0_1 ──────────────────────────────────────────

class TestDeltaStartClipsTo01:
    def test_below_range_clips_to_zero(self):
        # ps=50 < a=100 → raw=-0.5 → clipped to 0
        row = _metrics({"ps": 50.0, "pe": 150.0, "a": 100.0, "b": 200.0})
        assert row["delta_start"] == pytest.approx(0.0, abs=1e-12)

    def test_above_range_clips_to_one(self):
        # ps=250 > b=200 → raw=1.5 → clipped to 1
        row = _metrics({"ps": 250.0, "pe": 280.0, "a": 100.0, "b": 200.0})
        assert row["delta_start"] == pytest.approx(1.0, abs=1e-12)

    def test_at_lower_boundary(self):
        row = _metrics({"ps": 100.0, "pe": 150.0, "a": 100.0, "b": 200.0})
        assert row["delta_start"] == pytest.approx(0.0, abs=1e-12)

    def test_at_upper_boundary(self):
        row = _metrics({"ps": 200.0, "pe": 250.0, "a": 100.0, "b": 200.0})
        assert row["delta_start"] == pytest.approx(1.0, abs=1e-12)

    def test_delta_end_clips_below(self):
        # pe=50 < a=100 → delta_end=0
        row = _metrics({"ps": 150.0, "pe": 50.0, "a": 100.0, "b": 200.0})
        assert row["delta_end"] == pytest.approx(0.0, abs=1e-12)

    def test_delta_end_clips_above(self):
        # pe=300 > b=200 → delta_end=1
        row = _metrics({"ps": 150.0, "pe": 300.0, "a": 100.0, "b": 200.0})
        assert row["delta_end"] == pytest.approx(1.0, abs=1e-12)


# ── 3. test_boundary_metrics ───────────────────────────────────────────────────

class TestBoundaryMetrics:
    def test_m_a(self):
        # m_a = ps/a - 1 = 150/100 - 1 = 0.5
        row = _metrics({"ps": 150.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["m_a"] == pytest.approx(0.5, rel=1e-9)

    def test_m_b(self):
        # m_b = ps/b - 1 = 150/200 - 1 = -0.25
        row = _metrics({"ps": 150.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["m_b"] == pytest.approx(-0.25, rel=1e-9)

    def test_d_a(self):
        # d_a = pe/a - 1 = 160/100 - 1 = 0.6
        row = _metrics({"ps": 150.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["d_a"] == pytest.approx(0.6, rel=1e-9)

    def test_d_b(self):
        # d_b = pe/b - 1 = 160/200 - 1 = -0.2
        row = _metrics({"ps": 150.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["d_b"] == pytest.approx(-0.2, rel=1e-9)

    def test_m_a_negative_when_below_lower(self):
        # ps=80, a=100 → m_a = 80/100 - 1 = -0.2
        row = _metrics({"ps": 80.0, "pe": 150.0, "a": 100.0, "b": 200.0})
        assert row["m_a"] == pytest.approx(-0.2, rel=1e-9)

    def test_m_b_positive_when_above_upper(self):
        # ps=250, b=200 → m_b = 250/200 - 1 = 0.25
        row = _metrics({"ps": 250.0, "pe": 150.0, "a": 100.0, "b": 200.0})
        assert row["m_b"] == pytest.approx(0.25, rel=1e-9)


# ── 4. test_rel_metrics ────────────────────────────────────────────────────────

class TestRelMetrics:
    def test_rel_a(self):
        # rel_a = d_a - m_a = 0.6 - 0.5 = 0.1
        row = _metrics({"ps": 150.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["rel_a"] == pytest.approx(0.1, rel=1e-9)

    def test_rel_b(self):
        # rel_b = d_b - m_b = -0.2 - (-0.25) = 0.05
        row = _metrics({"ps": 150.0, "pe": 160.0, "a": 100.0, "b": 200.0})
        assert row["rel_b"] == pytest.approx(0.05, rel=1e-9)

    def test_rel_a_equals_d_a_minus_m_a(self):
        row = _metrics({"ps": 130.0, "pe": 170.0, "a": 100.0, "b": 200.0})
        assert row["rel_a"] == pytest.approx(row["d_a"] - row["m_a"], abs=1e-12)

    def test_rel_b_equals_d_b_minus_m_b(self):
        row = _metrics({"ps": 130.0, "pe": 170.0, "a": 100.0, "b": 200.0})
        assert row["rel_b"] == pytest.approx(row["d_b"] - row["m_b"], abs=1e-12)

    def test_rel_a_zero_when_price_unchanged(self):
        # ps == pe → d_a - m_a = 0
        row = _metrics({"ps": 150.0, "pe": 150.0, "a": 100.0, "b": 200.0})
        assert row["rel_a"] == pytest.approx(0.0, abs=1e-12)


# ── 5. test_r_cap ─────────────────────────────────────────────────────────────

class TestRCap:
    def test_positive_return(self):
        # r_cap = 1200/1000 - 1 = 0.2
        row = _metrics({"open_capital": 1_000.0, "close_capital": 1_200.0})
        assert row["r_cap"] == pytest.approx(0.2, rel=1e-9)

    def test_negative_return(self):
        row = _metrics({"open_capital": 1_000.0, "close_capital": 800.0})
        assert row["r_cap"] == pytest.approx(-0.2, rel=1e-9)

    def test_zero_return(self):
        row = _metrics({"open_capital": 1_000.0, "close_capital": 1_000.0})
        assert row["r_cap"] == pytest.approx(0.0, abs=1e-12)

    def test_zero_open_capital_gives_nan(self):
        row = _metrics({"open_capital": 0.0, "close_capital": 500.0})
        assert np.isnan(row["r_cap"])


# ── 6. test_r_price ───────────────────────────────────────────────────────────

class TestRPrice:
    def test_positive_price_return(self):
        # r_price = 3300/3000 - 1 = 0.1
        row = _metrics({"ps": 3_000.0, "pe": 3_300.0})
        assert row["r_price"] == pytest.approx(0.1, rel=1e-9)

    def test_negative_price_return(self):
        row = _metrics({"ps": 3_000.0, "pe": 2_700.0})
        assert row["r_price"] == pytest.approx(-0.1, rel=1e-9)

    def test_zero_price_return(self):
        row = _metrics({"ps": 3_000.0, "pe": 3_000.0})
        assert row["r_price"] == pytest.approx(0.0, abs=1e-12)

    def test_zero_open_price_gives_nan(self):
        row = _metrics({"ps": 0.0, "pe": 3_000.0})
        assert np.isnan(row["r_price"])


# ── 7. test_duration ──────────────────────────────────────────────────────────

class TestDuration:
    def test_duration_seconds(self):
        # close_ts=4600, open_ts=1000 → duration_sec = 3600
        row = _metrics({"open_ts": 1_000, "close_ts": 4_600})
        assert row["duration_sec"] == pytest.approx(3_600.0, abs=1e-9)

    def test_zero_duration(self):
        row = _metrics({"open_ts": 5_000, "close_ts": 5_000})
        assert row["duration_sec"] == pytest.approx(0.0, abs=1e-9)

    def test_one_day_in_seconds(self):
        row = _metrics({"open_ts": 0, "close_ts": 86_400})
        assert row["duration_sec"] == pytest.approx(86_400.0, abs=1e-9)


# ── 8. test_width_usdc ────────────────────────────────────────────────────────

class TestWidthUsdc:
    def test_basic_width(self):
        # b - a = 3100 - 2900 = 200
        row = _metrics({"a": 2_900.0, "b": 3_100.0})
        assert row["width_usdc"] == pytest.approx(200.0, abs=1e-9)

    def test_wide_range(self):
        row = _metrics({"a": 1_000.0, "b": 6_000.0})
        assert row["width_usdc"] == pytest.approx(5_000.0, abs=1e-9)

    def test_narrow_range(self):
        row = _metrics({"a": 2_999.0, "b": 3_001.0})
        assert row["width_usdc"] == pytest.approx(2.0, abs=1e-9)


# ── 9. test_assign_width_bucket ───────────────────────────────────────────────

class TestAssignWidthBucket:
    """
    Bucket boundaries per config.WIDTH_BUCKETS:
      (0, 10)       : 0 < w <= 10
      [10, 100)     : 10 < w <= 100
      [100, 1000)   : 100 < w <= 1000
      [1000, 10000) : 1000 < w <= 10000
      > 10000       : w > 10000
      invalid       : w <= 0
    """

    @pytest.mark.parametrize("width,expected", [
        (5.0,      "(0, 10)"),
        (10.0,     "(0, 10)"),      # 0 < 10 <= 10 → first bucket
        (10.001,   "[10, 100)"),    # 10 < 10.001 <= 100 → second bucket
        (50.0,     "[10, 100)"),
        (100.0,    "[10, 100)"),    # 10 < 100 <= 100
        (100.001,  "[100, 1000)"),
        (500.0,    "[100, 1000)"),
        (1000.0,   "[100, 1000)"),  # 100 < 1000 <= 1000
        (1000.001, "[1000, 10000)"),
        (5_000.0,  "[1000, 10000)"),
        (10_000.0, "[1000, 10000)"),  # 1000 < 10000 <= 10000
        (50_000.0, "> 10000"),
        (1e9,      "> 10000"),
    ])
    def test_bucket_assignment(self, width, expected):
        assert assign_width_bucket(width) == expected

    @pytest.mark.parametrize("invalid_width", [0.0, -1.0, -100.0])
    def test_invalid_width(self, invalid_width):
        assert assign_width_bucket(invalid_width) == "invalid"


# ── 10. test_compute_metrics_full ─────────────────────────────────────────────

class TestComputeMetricsFull:
    EXPECTED_COLS = {
        "delta_start", "delta_end", "delta",
        "m_a", "m_b", "d_a", "d_b", "rel_a", "rel_b",
        "r_cap", "r_price",
        "duration_sec",
        "width_usdc",
    }

    def test_all_metric_columns_present(self):
        df = _single_row()
        result = compute_position_metrics(df)
        assert self.EXPECTED_COLS.issubset(set(result.columns))

    def test_row_count_preserved(self):
        df = _single_row()
        result = compute_position_metrics(df)
        assert len(result) == len(df)

    def test_multiple_rows(self):
        rows = pd.concat([
            _single_row(ps=120.0, pe=180.0, a=100.0, b=200.0, position_type=5),
            _single_row(ps=50.0, pe=300.0, a=100.0, b=200.0, position_type=9),
        ], ignore_index=True)
        result = compute_position_metrics(rows)
        assert len(result) == 2
        assert self.EXPECTED_COLS.issubset(set(result.columns))

    def test_delta_start_in_range(self):
        df = _single_row(ps=3_000.0, pe=3_100.0, a=2_900.0, b=3_200.0)
        result = compute_position_metrics(df)
        ds = result.iloc[0]["delta_start"]
        assert 0.0 <= ds <= 1.0

    def test_r_cap_equals_expected(self):
        # open=1000, close=1500 → r_cap=0.5
        df = _single_row(open_capital=1_000.0, close_capital=1_500.0)
        result = compute_position_metrics(df)
        assert result.iloc[0]["r_cap"] == pytest.approx(0.5, rel=1e-9)

    def test_r_price_equals_expected(self):
        # ps=2000, pe=2500 → r_price=0.25
        df = _single_row(ps=2_000.0, pe=2_500.0)
        result = compute_position_metrics(df)
        assert result.iloc[0]["r_price"] == pytest.approx(0.25, rel=1e-9)

    def test_width_usdc_equals_expected(self):
        # a=2900, b=3200 → width=300
        df = _single_row(a=2_900.0, b=3_200.0)
        result = compute_position_metrics(df)
        assert result.iloc[0]["width_usdc"] == pytest.approx(300.0, abs=1e-9)

    def test_duration_equals_expected(self):
        df = _single_row(open_ts=0, close_ts=7_200)
        result = compute_position_metrics(df)
        assert result.iloc[0]["duration_sec"] == pytest.approx(7_200.0, abs=1e-9)


# ── 11. lp_type_weighted_width ────────────────────────────────────────────────

class TestLpTypeWeightedWidth:
    def test_single_lp_single_type(self):
        """Capital-weighted average width for one LP with one type."""
        df = pd.DataFrame([
            {"lp_address": "0xA", "position_type": 5,
             "open_capital": 1_000.0, "width_usdc": 200.0},
            {"lp_address": "0xA", "position_type": 5,
             "open_capital": 2_000.0, "width_usdc": 100.0},
        ])
        result = lp_type_weighted_width(df)
        # cw_avg = (1000*200 + 2000*100)/(1000+2000) = (200000+200000)/3000 = 400000/3000 ≈ 133.33
        row = result[result["lp_address"] == "0xA"].iloc[0]
        assert row["cw_avg_width"] == pytest.approx(400_000.0 / 3_000.0, rel=1e-9)

    def test_width_bucket_assigned(self):
        df = pd.DataFrame([
            {"lp_address": "0xA", "position_type": 5,
             "open_capital": 1.0, "width_usdc": 50.0},
        ])
        result = lp_type_weighted_width(df)
        assert result.iloc[0]["width_bucket"] == "[10, 100)"

    def test_zero_capital_excluded(self):
        """Rows with open_capital=0 are excluded from the weighted average."""
        df = pd.DataFrame([
            {"lp_address": "0xA", "position_type": 5,
             "open_capital": 0.0, "width_usdc": 9_999.0},
            {"lp_address": "0xA", "position_type": 5,
             "open_capital": 1.0, "width_usdc": 50.0},
        ])
        result = lp_type_weighted_width(df)
        row = result[result["lp_address"] == "0xA"].iloc[0]
        assert row["cw_avg_width"] == pytest.approx(50.0, abs=1e-9)

    def test_multiple_lps(self):
        df = pd.DataFrame([
            {"lp_address": "0xA", "position_type": 5,
             "open_capital": 1.0, "width_usdc": 200.0},
            {"lp_address": "0xB", "position_type": 5,
             "open_capital": 1.0, "width_usdc": 5_000.0},
        ])
        result = lp_type_weighted_width(df)
        assert len(result) == 2
        row_a = result[result["lp_address"] == "0xA"].iloc[0]
        row_b = result[result["lp_address"] == "0xB"].iloc[0]
        assert row_a["cw_avg_width"] == pytest.approx(200.0, abs=1e-9)
        assert row_b["cw_avg_width"] == pytest.approx(5_000.0, abs=1e-9)
