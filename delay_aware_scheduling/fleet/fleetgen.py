# -*- coding: utf-8 -*-
"""集群生成与读取。

make_fleet 造一个参数化的合成集群：机型构成、规模、机架粒度与机架纯度。机架纯度小于 1
时以「机架」为单位混入其他机型，这样"同机架不同机型"的比例才可控。

load_real_fleet 读供给侧清单里的参照日机器集合，作为合成集群之外的实测锚点。
"""
import numpy as np
import pandas as pd

from delay_aware_scheduling.config import CENSUS
from delay_aware_scheduling.fleet.constants import SCALE


def make_fleet(N, mix, rack_nodes, purity, seed):
    """按机型构成 mix 造 N 台机器。

    mix       单机卡数 -> 占比
    rack_nodes 每机架机器数
    purity     机架纯度；1.0 为完全同质机架
    """
    rng = np.random.default_rng(seed)
    cards = np.array(sorted(mix), dtype=np.int64)
    w = np.array([mix[c] for c in cards], dtype=np.float64)
    w = w / w.sum()
    size = rng.choice(cards, size=N, p=w).astype(np.int64)
    order = np.argsort(size, kind="stable")
    rack = np.empty(N, dtype=np.int64)
    rack[order] = np.arange(N) // rack_nodes
    n0 = int(rack.max()) + 1
    if purity < 1.0:
        # 以机架为单位混合：每个机架以 1-purity 的概率成为混装机架
        mixed_rack = rng.random(n0) < (1.0 - purity)
        is_mixed = mixed_rack[rack]
        if is_mixed.any():
            size = np.where(is_mixed, rng.choice(cards, size=N, p=w), size).astype(np.int64)
    _, rack = np.unique(rack, return_inverse=True)
    cap = size * SCALE
    return dict(cards=size, cap=cap, mrack=rack, n_machine=N,
                cap_total=int(cap.sum()), maxcap=int(cap.max()))


def load_real_fleet(census_path=None):
    """读参照日的实测机器集合（卡数、机架编号）。"""
    c = pd.read_parquet(census_path or CENSUS, columns=["gpu_count", "asw_id", "in_ref_day"])
    c = c[c["in_ref_day"]].reset_index(drop=True)
    cards = c["gpu_count"].values.astype(np.int64)
    rmap = {}
    rk = np.empty(len(c), dtype=np.int64)
    for i, r in enumerate(c["asw_id"].fillna("__NA__").astype(str).values):
        j = rmap.get(r)
        if j is None:
            j = len(rmap)
            rmap[r] = j
        rk[i] = j
    cap = cards * SCALE
    return dict(cards=cards, cap=cap, mrack=rk, n_machine=len(cap),
                cap_total=int(cap.sum()), maxcap=int(cap.max()))
