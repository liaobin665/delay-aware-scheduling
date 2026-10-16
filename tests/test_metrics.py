# -*- coding: utf-8 -*-
"""指标公式、折叠分配与排序规则的冒烟测试（不需要数据）。"""
import numpy as np
import pytest

from delay_aware_scheduling.prediction.features import fold_assign
from delay_aware_scheduling.prediction.metrics import metrics, pinball
from delay_aware_scheduling.scheduling.policies import order_of
from delay_aware_scheduling.scheduling.tables import _p, _sgn


def test_metrics_on_perfect_prediction():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = metrics(y, y.copy())
    assert m["n"] == 4
    assert m["mae_log"] == 0.0
    assert m["r2_log"] == pytest.approx(1.0)
    assert m["spearman_log"] == pytest.approx(1.0)
    assert m["within2_log"] == pytest.approx(1.0)


def test_metrics_are_rank_based():
    """整体平移不改变排序质量，只改绝对值误差。"""
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = metrics(y, y + 0.5)
    assert m["spearman_log"] == pytest.approx(1.0)
    assert m["mae_log"] == pytest.approx(0.5)
    assert m["medae_log"] == pytest.approx(0.5)


def test_metrics_ignore_non_finite_predictions():
    y = np.array([0.0, 1.0, 2.0])
    p = np.array([0.0, np.nan, 2.0])
    m = metrics(y, p)
    assert m["n"] == 2
    assert m["mae_log"] == pytest.approx(0.0)


def test_pinball_penalises_the_predictor_side_asymmetrically():
    y = np.array([0.0])
    assert pinball(y, np.array([-1.0]), 0.9) == pytest.approx(0.9)   # 偏低：按 α 罚
    assert pinball(y, np.array([1.0]), 0.9) == pytest.approx(0.1)    # 偏高：按 1-α 罚
    assert pinball(y, y.copy(), 0.5) == pytest.approx(0.0)


def test_fold_assign_keeps_a_machine_in_one_fold():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    f = fold_assign(groups, 20261008, 2)
    for g in ("a", "b", "c", "d"):
        assert len(np.unique(f[groups == g])) == 1
    assert set(np.unique(f).tolist()) == {0, 1}


def test_order_of_fixed_rules():
    req = np.array([2.0, 1.0, 4.0])
    dur = np.array([10.0, 5.0, 1.0])
    z = np.zeros(3)
    call = lambda pol: list(order_of(pol, req, dur, z, z, z, z, z, z, z, z))
    assert call("FCFS") == [0, 1, 2]
    assert call("SJF") == [2, 1, 0]         # 声明时长升序
    assert call("SSF") == [1, 0, 2]         # 请求卡数升序
    assert call("BigFirst") == [2, 0, 1]    # 请求卡数降序


def test_order_of_score_rules_follow_the_score():
    req = np.array([1.0, 1.0, 1.0])
    dur = np.ones(3)
    z = np.zeros(3)
    lgb = np.array([0.5, -1.0, 2.0])
    delay = np.array([9.0, 8.0, 7.0])
    # 打分越大越靠后
    assert list(order_of("DAS", req, dur, lgb, z, z, z, z, z, delay, z)) == [1, 0, 2]
    assert list(order_of("DelayOracle", req, dur, z, z, z, z, z, z, delay, z)) == [2, 1, 0]


def test_stats_formatting_helpers():
    assert _sgn(-0.25) == "-0.2500"
    assert _sgn(0.5) == "+0.5000"
    assert _p(None) == "—"
    assert _p(0.0123) == "0.0123"
    assert _p(1e-9).startswith("$<")
