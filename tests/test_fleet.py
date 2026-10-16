# -*- coding: utf-8 -*-
"""合成集群生成的冒烟测试（不需要数据）。"""
import numpy as np

from delay_aware_scheduling.fleet.constants import SCALE
from delay_aware_scheduling.fleet.fleetgen import make_fleet

MIX = {1: 0.5, 8: 0.5}


def test_make_fleet_shape_and_totals():
    fl = make_fleet(256, MIX, 16, 0.905, 20261008)
    assert fl["n_machine"] == 256
    assert len(fl["cards"]) == 256
    assert len(fl["cap"]) == 256
    assert fl["cap_total"] == int(fl["cards"].sum()) * SCALE
    assert fl["maxcap"] == int(fl["cards"].max()) * SCALE
    assert set(np.unique(fl["cards"])) <= set(MIX)


def test_make_fleet_is_deterministic_per_seed():
    a = make_fleet(256, MIX, 16, 0.905, 20261008)
    b = make_fleet(256, MIX, 16, 0.905, 20261008)
    assert np.array_equal(a["cards"], b["cards"])
    assert np.array_equal(a["mrack"], b["mrack"])


def test_make_fleet_seed_changes_draw():
    a = make_fleet(512, MIX, 16, 1.0, 1)
    b = make_fleet(512, MIX, 16, 1.0, 2)
    assert not np.array_equal(a["cards"], b["cards"])


def _n_mixed_racks(fl):
    r, c = fl["mrack"], fl["cards"]
    return sum(1 for k in np.unique(r) if len(np.unique(c[r == k])) > 1)


def test_pure_fleet_segments_sizes_by_rack():
    """纯度 1.0 不混装：按卡数排序后逐机架分段，因此机架序号对卡数单调不减。"""
    fl = make_fleet(4096, MIX, 16, 1.0, 7)
    r, c = fl["mrack"], fl["cards"]
    means = np.array([c[r == k].mean() for k in np.unique(r)])
    assert np.all(np.diff(means) >= 0)
    # 分段点只可能落在卡数分界处，故至多一个机架跨两种卡数
    assert _n_mixed_racks(fl) <= 1


def test_purity_below_one_mixes_racks():
    """纯度小于 1 时刻意混装机架，跨档机架数明显增多。"""
    pure = make_fleet(4096, MIX, 16, 1.0, 7)
    mixed = make_fleet(4096, MIX, 16, 0.5, 7)
    assert _n_mixed_racks(mixed) > _n_mixed_racks(pure)
