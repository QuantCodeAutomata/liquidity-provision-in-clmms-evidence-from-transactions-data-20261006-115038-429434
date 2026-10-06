"""
Tests for src/lp_classification.py using purely synthetic DataFrames.

No file I/O or API calls – all inputs are constructed in-memory.
"""

import pandas as pd
import pytest

from src.lp_classification import (
    classify_mono_multi,
    cross_pool_intersection,
    mono_multi_decomposition_table,
    multi_lp_performance_table,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lp_row(
    pool: str,
    lp: str,
    terminal_pnl: float = 100.0,
    win_score: float = 0.6,
    pnl_positive: bool = True,
    consistently_successful: bool = False,
) -> dict:
    return {
        "pool": pool,
        "lp_address": lp,
        "terminal_pnl": terminal_pnl,
        "win_score": win_score,
        "pnl_positive": pnl_positive,
        "consistently_successful": consistently_successful,
    }


def _base_df() -> pd.DataFrame:
    """
    5 rows covering 3 unique LP addresses across 2 pools.

    0xAAA: pool1 + pool2 → multi-LP
    0xBBB: pool1 + pool2 → multi-LP
    0xCCC: pool1 only    → mono-LP
    """
    return pd.DataFrame([
        _lp_row("pool1", "0xAAA", terminal_pnl=200.0,  win_score=0.8,
                pnl_positive=True,  consistently_successful=True),
        _lp_row("pool2", "0xAAA", terminal_pnl=-50.0,  win_score=0.3,
                pnl_positive=False, consistently_successful=False),
        _lp_row("pool1", "0xBBB", terminal_pnl=150.0,  win_score=0.7,
                pnl_positive=True,  consistently_successful=True),
        _lp_row("pool2", "0xBBB", terminal_pnl=75.0,   win_score=0.6,
                pnl_positive=True,  consistently_successful=False),
        _lp_row("pool1", "0xCCC", terminal_pnl=-100.0, win_score=0.2,
                pnl_positive=False, consistently_successful=False),
    ])


# ── 1. classify_mono_multi ─────────────────────────────────────────────────────

class TestClassifyMonoMulti:
    def test_mono_lp_in_one_pool(self):
        df = _base_df()
        result = classify_mono_multi(df)
        ccc_rows = result[result["lp_address"] == "0xCCC"]
        assert all(ccc_rows["n_pools_active"] == 1)
        assert all(~ccc_rows["is_multi_lp"])

    def test_multi_lp_in_two_pools(self):
        df = _base_df()
        result = classify_mono_multi(df)
        aaa_rows = result[result["lp_address"] == "0xAAA"]
        assert all(aaa_rows["n_pools_active"] == 2)
        assert all(aaa_rows["is_multi_lp"])

    def test_is_multi_lp_flag_correct(self):
        df = _base_df()
        result = classify_mono_multi(df)
        bbb_rows = result[result["lp_address"] == "0xBBB"]
        assert all(bbb_rows["is_multi_lp"])

    def test_row_count_preserved(self):
        df = _base_df()
        result = classify_mono_multi(df)
        assert len(result) == len(df)

    def test_columns_added(self):
        df = _base_df()
        result = classify_mono_multi(df)
        assert "n_pools_active" in result.columns
        assert "is_multi_lp" in result.columns

    def test_n_pools_active_is_integer(self):
        df = _base_df()
        result = classify_mono_multi(df)
        assert pd.api.types.is_integer_dtype(result["n_pools_active"])

    def test_single_lp_single_pool(self):
        df = pd.DataFrame([_lp_row("pool1", "0xX")])
        result = classify_mono_multi(df)
        assert result.iloc[0]["n_pools_active"] == 1
        assert result.iloc[0]["is_multi_lp"] is False or (
            result.iloc[0]["is_multi_lp"] == False
        )

    def test_lp_in_three_pools(self):
        df = pd.DataFrame([
            _lp_row("pool1", "0xZ"),
            _lp_row("pool2", "0xZ"),
            _lp_row("pool3", "0xZ"),
        ])
        result = classify_mono_multi(df)
        assert all(result["n_pools_active"] == 3)
        assert all(result["is_multi_lp"])


# ── 2. mono_multi_decomposition_table ─────────────────────────────────────────

class TestMonoMultiDecompositionTable:
    def test_returns_dataframe(self):
        table = mono_multi_decomposition_table(_base_df())
        assert isinstance(table, pd.DataFrame)

    def test_required_columns(self):
        table = mono_multi_decomposition_table(_base_df())
        required = {"pool", "total_lp", "mono_lp_count", "mono_lp_pct",
                    "multi_lp_count", "multi_lp_pct"}
        assert required.issubset(set(table.columns))

    def test_network_unique_row_present(self):
        table = mono_multi_decomposition_table(_base_df())
        assert "Network unique" in table["pool"].values

    def test_per_pool_total_is_correct(self):
        table = mono_multi_decomposition_table(_base_df())
        pool1 = table[table["pool"] == "pool1"].iloc[0]
        # pool1 has: 0xAAA, 0xBBB (multi), 0xCCC (mono) → total=3
        assert pool1["total_lp"] == 3

    def test_mono_plus_multi_equals_total(self):
        table = mono_multi_decomposition_table(_base_df())
        for _, row in table[table["pool"] != "Network unique"].iterrows():
            assert row["mono_lp_count"] + row["multi_lp_count"] == row["total_lp"]

    def test_percentages_sum_to_100(self):
        table = mono_multi_decomposition_table(_base_df())
        for _, row in table.iterrows():
            assert row["mono_lp_pct"] + row["multi_lp_pct"] == pytest.approx(100.0, abs=1e-9)

    def test_network_unique_counts(self):
        """3 unique LPs in the dataset; 1 mono (0xCCC), 2 multi (0xAAA, 0xBBB)."""
        table = mono_multi_decomposition_table(_base_df())
        net = table[table["pool"] == "Network unique"].iloc[0]
        assert net["total_lp"] == 3
        assert net["mono_lp_count"] == 1
        assert net["multi_lp_count"] == 2

    def test_pool2_lp_count(self):
        """pool2 contains 0xAAA and 0xBBB → 2 LPs, both multi."""
        table = mono_multi_decomposition_table(_base_df())
        pool2 = table[table["pool"] == "pool2"].iloc[0]
        assert pool2["total_lp"] == 2
        assert pool2["multi_lp_count"] == 2
        assert pool2["mono_lp_count"] == 0


# ── 3. multi_lp_performance_at_least_one ──────────────────────────────────────

class TestMultiLpPerformanceAtLeastOne:
    def _df_with_mixed_performance(self) -> pd.DataFrame:
        """
        0xAAA: pool1(pnl+, consistent), pool2(pnl-, not consistent) → pnl_pos_in_any=True
        0xBBB: pool1(pnl-, not), pool2(pnl-, not) → pnl_pos_in_any=False
        """
        return pd.DataFrame([
            _lp_row("pool1", "0xAAA", pnl_positive=True,  consistently_successful=True),
            _lp_row("pool2", "0xAAA", pnl_positive=False, consistently_successful=False),
            _lp_row("pool1", "0xBBB", pnl_positive=False, consistently_successful=False),
            _lp_row("pool2", "0xBBB", pnl_positive=False, consistently_successful=False),
        ])

    def test_at_least_one_pool_positive(self):
        table = multi_lp_performance_table(self._df_with_mixed_performance())
        # n_pools_in=2, count=2, 0xAAA pnl_pos_in_any=True, 0xBBB=False → n_pos=1
        n2 = table[table["n_pools_in"] == 2].iloc[0]
        assert n2["pnl_pos_in_any"] == 1

    def test_all_negative_gives_zero_pnl_pos(self):
        df = pd.DataFrame([
            _lp_row("pool1", "0xBBB", pnl_positive=False),
            _lp_row("pool2", "0xBBB", pnl_positive=False),
        ])
        table = multi_lp_performance_table(df)
        n2 = table[table["n_pools_in"] == 2].iloc[0]
        assert n2["pnl_pos_in_any"] == 0

    def test_consistent_in_any_pool(self):
        table = multi_lp_performance_table(self._df_with_mixed_performance())
        n2 = table[table["n_pools_in"] == 2].iloc[0]
        # 0xAAA is consistent in pool1 → consistent_in_any=True; 0xBBB is not
        assert n2["consistent_in_any"] == 1

    def test_total_multi_lp_row(self):
        table = multi_lp_performance_table(self._df_with_mixed_performance())
        assert "Total multi-LP" in table["n_pools_in"].values

    def test_required_columns(self):
        table = multi_lp_performance_table(self._df_with_mixed_performance())
        required = {"n_pools_in", "count", "pnl_pos_in_any", "pnl_pos_pct",
                    "consistent_in_any", "consistent_pct"}
        assert required.issubset(set(table.columns))

    def test_pnl_pos_pct_range(self):
        table = multi_lp_performance_table(self._df_with_mixed_performance())
        for _, row in table.iterrows():
            assert 0.0 <= row["pnl_pos_pct"] <= 100.0

    def test_three_pool_lp(self):
        """An LP in 3 pools should appear in n_pools_in=3 group."""
        df = pd.DataFrame([
            _lp_row("pool1", "0xZ", pnl_positive=True,  consistently_successful=True),
            _lp_row("pool2", "0xZ", pnl_positive=False, consistently_successful=False),
            _lp_row("pool3", "0xZ", pnl_positive=False, consistently_successful=False),
        ])
        table = multi_lp_performance_table(df)
        n3 = table[table["n_pools_in"] == 3]
        assert len(n3) == 1
        assert n3.iloc[0]["count"] == 1
        assert n3.iloc[0]["pnl_pos_in_any"] == 1  # True in pool1


# ── 4. cross_pool_intersection ─────────────────────────────────────────────────

class TestCrossPoolIntersection:
    def test_two_lps_same_pair_gives_count_two(self):
        """0xAAA and 0xBBB both in {pool1, pool2} → count=2 for 'pool1 ∩ pool2'."""
        df = pd.DataFrame([
            _lp_row("pool1", "0xAAA"),
            _lp_row("pool2", "0xAAA"),
            _lp_row("pool1", "0xBBB"),
            _lp_row("pool2", "0xBBB"),
        ])
        result = cross_pool_intersection(df)
        assert len(result) == 1
        assert result.iloc[0]["lp_count"] == 2
        assert "pool1" in result.iloc[0]["pool_combination"]
        assert "pool2" in result.iloc[0]["pool_combination"]

    def test_distinct_combinations(self):
        """0xAAA in {pool1, pool2}, 0xCCC in {pool1, pool3} → 2 distinct combinations."""
        df = pd.DataFrame([
            _lp_row("pool1", "0xAAA"),
            _lp_row("pool2", "0xAAA"),
            _lp_row("pool1", "0xCCC"),
            _lp_row("pool3", "0xCCC"),
        ])
        result = cross_pool_intersection(df)
        assert len(result) == 2

    def test_combination_counts_correct(self):
        df = pd.DataFrame([
            _lp_row("pool1", "0xAAA"),
            _lp_row("pool2", "0xAAA"),
            _lp_row("pool1", "0xBBB"),
            _lp_row("pool2", "0xBBB"),
            _lp_row("pool1", "0xCCC"),
            _lp_row("pool3", "0xCCC"),
        ])
        result = cross_pool_intersection(df)
        combo_p1p2 = result[result["pool_combination"].str.contains("pool1")
                            & result["pool_combination"].str.contains("pool2")]
        assert combo_p1p2.iloc[0]["lp_count"] == 2

    def test_returns_dataframe_with_correct_columns(self):
        df = pd.DataFrame([
            _lp_row("pool1", "0xAAA"),
            _lp_row("pool2", "0xAAA"),
        ])
        result = cross_pool_intersection(df)
        assert "pool_combination" in result.columns
        assert "lp_count" in result.columns

    def test_sorted_by_descending_count(self):
        """Result is sorted descending by lp_count."""
        df = pd.DataFrame([
            _lp_row("pool1", "0xAAA"),
            _lp_row("pool2", "0xAAA"),
            _lp_row("pool1", "0xBBB"),
            _lp_row("pool2", "0xBBB"),
            _lp_row("pool1", "0xCCC"),
            _lp_row("pool2", "0xCCC"),
            _lp_row("pool1", "0xDDD"),
            _lp_row("pool3", "0xDDD"),
        ])
        result = cross_pool_intersection(df)
        counts = result["lp_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_mono_lps_excluded(self):
        """Mono-LPs are not multi-LPs so they don't appear in the intersection."""
        df = pd.DataFrame([
            _lp_row("pool1", "0xMONO"),   # mono
            _lp_row("pool1", "0xMULTI"),  # multi
            _lp_row("pool2", "0xMULTI"),
        ])
        result = cross_pool_intersection(df)
        # Only 0xMULTI is multi; result should contain only one combination
        assert len(result) == 1
        assert result.iloc[0]["lp_count"] == 1


# ── 5. test_no_multi_lps ──────────────────────────────────────────────────────

class TestNoMultiLps:
    def _mono_only_df(self) -> pd.DataFrame:
        """Each LP appears in exactly one pool → no multi-LPs."""
        return pd.DataFrame([
            _lp_row("pool1", "0xA", pnl_positive=True),
            _lp_row("pool1", "0xB", pnl_positive=False),
            _lp_row("pool2", "0xC", pnl_positive=True),
        ])

    def test_multi_lp_performance_table_empty(self):
        """With no multi-LPs, table has no per-group rows (only maybe Total)."""
        table = multi_lp_performance_table(self._mono_only_df())
        # multi is empty → should return empty DataFrame or df with required columns
        assert isinstance(table, pd.DataFrame)
        # No LP is multi, so no n_pools_in groups exist
        numeric_rows = table[table["n_pools_in"] != "Total multi-LP"]
        assert len(numeric_rows) == 0

    def test_classify_mono_multi_all_mono(self):
        result = classify_mono_multi(self._mono_only_df())
        assert not any(result["is_multi_lp"])

    def test_network_unique_shows_zero_multi(self):
        """When all LPs are mono, the Network unique row shows 0 multi-LPs."""
        table = mono_multi_decomposition_table(self._mono_only_df())
        net = table[table["pool"] == "Network unique"].iloc[0]
        assert net["multi_lp_count"] == 0
        assert net["mono_lp_pct"] == pytest.approx(100.0, abs=1e-9)

    def test_decomposition_table_all_mono(self):
        table = mono_multi_decomposition_table(self._mono_only_df())
        for _, row in table[table["pool"] != "Network unique"].iterrows():
            assert row["multi_lp_count"] == 0
            assert row["mono_lp_pct"] == pytest.approx(100.0, abs=1e-9)
