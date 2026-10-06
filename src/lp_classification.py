"""
Mono-LP and Multi-LP classification across pools (Section 4.2, Tables 5-6).

An LP that provides liquidity in exactly one analyzed pool is a mono-LP.
An LP present in more than one analyzed pool is a multi-LP.

LP identity is defined by the lp_address field (Ethereum wallet address),
which is assumed to be directly comparable across pools.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def classify_mono_multi(lp_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mono_lp / multi_lp flag to LP-level classification DataFrame.

    Args:
        lp_class_df: Output of win_score.classify_lps, with columns
                     pool, lp_address, terminal_pnl, win_score, …

    Returns:
        Same DataFrame augmented with:
          n_pools_active (int): number of analyzed pools LP appears in
          is_multi_lp (bool): True if n_pools_active > 1
    """
    pool_counts = (
        lp_class_df.groupby("lp_address")["pool"]
        .nunique()
        .reset_index()
        .rename(columns={"pool": "n_pools_active"})
    )
    df = lp_class_df.merge(pool_counts, on="lp_address", how="left")
    df["is_multi_lp"] = df["n_pools_active"] > 1
    return df


def mono_multi_decomposition_table(lp_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Table 5: pool-level decomposition into mono and multi LPs.

    For each pool, reports:
      pool, total_lp, mono_lp_count, mono_lp_pct, multi_lp_count, multi_lp_pct

    Plus aggregate totals (unique LPs across network).
    """
    df = classify_mono_multi(lp_class_df)
    rows = []
    for pool, grp in df.groupby("pool"):
        n = len(grp)
        n_mono = int((~grp["is_multi_lp"]).sum())
        n_multi = int(grp["is_multi_lp"].sum())
        rows.append({
            "pool": pool,
            "total_lp": n,
            "mono_lp_count": n_mono,
            "mono_lp_pct": 100.0 * n_mono / n if n > 0 else 0.0,
            "multi_lp_count": n_multi,
            "multi_lp_pct": 100.0 * n_multi / n if n > 0 else 0.0,
        })

    # Network-level unique LPs
    n_unique = df["lp_address"].nunique()
    n_unique_mono = df.drop_duplicates("lp_address")["is_multi_lp"].eq(False).sum()
    n_unique_multi = df.drop_duplicates("lp_address")["is_multi_lp"].sum()
    rows.append({
        "pool": "Network unique",
        "total_lp": n_unique,
        "mono_lp_count": int(n_unique_mono),
        "mono_lp_pct": 100.0 * n_unique_mono / n_unique if n_unique > 0 else 0.0,
        "multi_lp_count": int(n_unique_multi),
        "multi_lp_pct": 100.0 * n_unique_multi / n_unique if n_unique > 0 else 0.0,
    })
    return pd.DataFrame(rows)


def multi_lp_performance_table(lp_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Table 6: cross-pool participation and performance of multi-LPs.

    For each group of multi-LPs by number of pools (2, 3, 4), reports:
      n_pools_in, count, pnl_pos_in_any_pool, pnl_pos_pct,
      consistent_in_any_pool, consistent_pct

    Performance metric: "PnL > 0 in at least one of the cross-pools" and
    "PnL > 0 and ω > 0.5 in at least one of the cross-pools" (Section 4.2).
    """
    df = classify_mono_multi(lp_class_df)
    multi = df[df["is_multi_lp"]].copy()

    if multi.empty:
        return pd.DataFrame(columns=[
            "n_pools_in", "count",
            "pnl_pos_in_any", "pnl_pos_pct",
            "consistent_in_any", "consistent_pct",
        ])

    # Per LP aggregation across pools
    lp_agg = (
        multi.groupby("lp_address")
        .agg(
            n_pools_in=("pool", "nunique"),
            pnl_pos_in_any=("pnl_positive", "any"),
            consistent_in_any=("consistently_successful", "any"),
        )
        .reset_index()
    )

    rows = []
    for n_pools, grp in lp_agg.groupby("n_pools_in"):
        n = len(grp)
        n_pos = int(grp["pnl_pos_in_any"].sum())
        n_con = int(grp["consistent_in_any"].sum())
        rows.append({
            "n_pools_in": int(n_pools),
            "count": n,
            "pnl_pos_in_any": n_pos,
            "pnl_pos_pct": 100.0 * n_pos / n if n > 0 else 0.0,
            "consistent_in_any": n_con,
            "consistent_pct": 100.0 * n_con / n if n > 0 else 0.0,
        })

    # Overall multi-LP total
    n_total = len(lp_agg)
    n_pos_total = int(lp_agg["pnl_pos_in_any"].sum())
    n_con_total = int(lp_agg["consistent_in_any"].sum())
    rows.append({
        "n_pools_in": "Total multi-LP",
        "count": n_total,
        "pnl_pos_in_any": n_pos_total,
        "pnl_pos_pct": 100.0 * n_pos_total / n_total if n_total > 0 else 0.0,
        "consistent_in_any": n_con_total,
        "consistent_pct": 100.0 * n_con_total / n_total if n_total > 0 else 0.0,
    })
    return pd.DataFrame(rows)


def cross_pool_intersection(lp_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build cross-pool LP intersection counts (Figure 4 data).

    Returns DataFrame with columns:
      pool_combination (frozenset → str), lp_count
    sorted by descending count.
    """
    df = classify_mono_multi(lp_class_df)
    multi = df[df["is_multi_lp"]]["lp_address"].unique()

    # For each multi-LP, get its set of pools
    lp_pools: dict[str, set[str]] = {}
    for lp, grp in df[df["lp_address"].isin(multi)].groupby("lp_address"):
        lp_pools[lp] = set(grp["pool"].tolist())

    combos: dict[str, int] = {}
    for pools_set in lp_pools.values():
        key = " ∩ ".join(sorted(pools_set))
        combos[key] = combos.get(key, 0) + 1

    result = pd.DataFrame(
        [{"pool_combination": k, "lp_count": v} for k, v in combos.items()]
    ).sort_values("lp_count", ascending=False).reset_index(drop=True)
    return result
