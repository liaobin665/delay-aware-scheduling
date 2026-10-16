# -*- coding: utf-8 -*-

"""准入排序规则。

每条规则只产出一个排列；放置规则与它正交，两者互不改写。
"""

import numpy as np


MAIN_POL = ["FCFS", "SJF", "SSF", "BigFirst", "Random", "RuleDelay", "L0Delay",
            "DAS", "DelayOracle"]


EXT_POL = ["DASq50", "DASq90", "DASq99"]


# 复合打分（公平性-效率权衡）：s = d̂ − λ·(r/̄r)，λ 越大越偏向大请求
PARE_LAMBDAS = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]


PARE_POL = ["Comp%d" % i for i in range(len(PARE_LAMBDAS))]


POL_LAM = {p: PARE_LAMBDAS[i] for i, p in enumerate(PARE_POL)}


PLACEMENTS = ["BF", "FF", "WF"]


def order_of(policy, req, dur, p_lgb, p_q50, p_q90, p_q99, p_rule, p_l0, delay, rand):
    n = len(req)
    ar = np.arange(n, dtype=np.int64)
    if policy == "FCFS":
        return ar
    if policy == "SJF":
        return np.lexsort((ar, dur))
    if policy == "SSF":
        return np.lexsort((ar, req))
    if policy == "BigFirst":
        return np.lexsort((ar, -req))
    if policy == "Random":
        return np.lexsort((ar, rand))
    if policy == "DAS":
        return np.lexsort((ar, p_lgb))
    if policy == "DelayOracle":
        return np.lexsort((ar, delay))
    if policy == "DASq50":
        return np.lexsort((ar, p_q50))
    if policy == "DASq90":
        return np.lexsort((ar, p_q90))
    if policy == "DASq99":
        return np.lexsort((ar, p_q99))
    if policy == "RuleDelay":
        return np.lexsort((ar, p_rule))
    if policy == "L0Delay":
        return np.lexsort((ar, p_l0))
    raise ValueError(policy)
