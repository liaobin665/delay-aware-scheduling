# -*- coding: utf-8 -*-

"""回归与分位数的误差度量。

对数尺度与秒尺度同时报；排序相关性用 Spearman 与 Kendall，
以避开长尾对均值类指标的干扰。
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


def metrics(y_log, p_log):
    m = np.isfinite(p_log) & np.isfinite(y_log)
    y, p = y_log[m], p_log[m]
    e = p - y
    y_sec = np.expm1(y)
    p_sec = np.expm1(p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rho = float(spearmanr(y, p).statistic)
    tau = float(kendalltau(y, p).statistic)
    # 秒尺度上的相对误差（中位）
    rel = np.abs(p_sec - y_sec) / np.maximum(y_sec, 1.0)
    # 对数尺度落在 2 倍以内（|e| <= ln2）的占比；秒尺度上比值落在 [1/2, 2] 的占比
    within2_log = float(np.mean(np.abs(e) <= np.log(2.0)))
    ratio = p_sec / np.maximum(y_sec, 1.0)
    ratio_in2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    return dict(
        n=int(m.sum()),
        mae_log=float(np.mean(np.abs(e))),
        rmse_log=float(np.sqrt(np.mean(e ** 2))),
        r2_log=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        medae_log=float(np.median(np.abs(e))),
        spearman_log=rho,
        kendall_log=tau,
        within2_log=within2_log,
        ratio_in2=ratio_in2,
        mae_sec=float(np.mean(np.abs(p_sec - y_sec))),
        medae_sec=float(np.median(np.abs(p_sec - y_sec))),
        mape_rel_median=float(np.median(rel)),
    )


def pinball(y, p, a):
    d = y - p
    return float(np.mean(np.maximum(a * d, (a - 1.0) * d)))
