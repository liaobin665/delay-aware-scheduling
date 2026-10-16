# -*- coding: utf-8 -*-
"""容量索引与放置规则的冒烟测试（不需要数据）。"""
import numpy as np

from delay_aware_scheduling.fleet.capacity import CapIdx, CapTree, Fleet, MaxSeg


def test_maxseg_first_ge():
    seg = MaxSeg(np.array([0, 5, 3, 9]))
    assert seg.first_ge(1) == 1          # 下标最小且空闲 >= 1 的是 1 号（5）
    assert seg.first_ge(5) == 1
    assert seg.first_ge(6) == 3
    assert seg.first_ge(10) == -1        # 谁都放不下
    seg.set(1, 0)
    assert seg.first_ge(5) == 3          # 1 号腾空后轮到 3 号


def test_maxseg_tracks_updates():
    seg = MaxSeg(np.array([7, 7]))
    seg.set(0, 2)
    assert seg.first_ge(3) == 1
    seg.set(1, 1)
    assert seg.first_ge(3) == -1


def test_captree_first_ge():
    t = CapTree(8)
    t.update(3, 0, 1)
    t.update(7, 0, 2)
    assert t.first_ge(0) == 3
    assert t.first_ge(3) == 3
    assert t.first_ge(4) == 7
    assert t.first_ge(8) == -1           # 8 号本身没有机器
    t.update(3, 0, 0)                    # 3 号档位清空
    assert t.first_ge(0) == 7


def test_capidx_smallest_ge_and_any_of():
    idx = CapIdx(np.array([100, 400, 400, 800]))
    assert idx.maxcap == 800
    assert idx.smallest_ge(400) == 400
    assert idx.any_of(400) in (1, 2)
    assert idx.smallest_ge(401) == 800
    assert idx.smallest_ge(801) == -1    # 超过最大单机
    idx.remove(400, 1)
    assert idx.smallest_ge(400) == 400   # 还剩 2 号
    idx.remove(400, 2)
    assert idx.smallest_ge(400) == 800   # 该档位空了


def test_fleet_take_keeps_indices_in_sync():
    f = Fleet(np.array([400, 800]))
    assert f.maxcap == 800
    assert f.capset == {400, 800}
    f.take(0, 100)
    assert f.free[0] == 300
    assert f.used == 100
    assert f.seg.first_ge(300) == 0      # 首次适配看下标
    assert f.idx.smallest_ge(301) == 800  # 最佳适配看容量
    f.take(0, 300)
    assert f.free[0] == 0
    assert f.idx.smallest_ge(1) == 800
