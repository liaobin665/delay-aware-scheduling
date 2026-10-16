# -*- coding: utf-8 -*-
r"""预测误差分析、分位数校准、公平性-效率权衡与预测质量-调度收益。

四块内容：预测误差的整体读数与按作业形态的分解，重点是长延迟子群被低估的程度；
分位数回归的实测覆盖率；复合打分在"大请求保障"与"各档均衡"之间的取舍曲线；以及把
各打分器与真实延迟的排序相关性同其调度收益并列，看两者是否一致。

输出  out/extras.json

用法  python -m delay_aware_scheduling.analysis.exp65_extras
"""
import os
import json

import numpy as np
import pandas as pd
from scipy import stats as st


from delay_aware_scheduling.config import OUT as PROD, POOL, ADM, QUANTILE as QUA
from delay_aware_scheduling.common.log import out


OUT = os.path.join(PROD, "extras.json")


LAMBDAS = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]


def spearman(a, b):
    return float(st.spearmanr(a, b).statistic)


def errors(pool):
    """预测误差分析：整体、按作业形态、以及长延迟子群的低估情况。"""
    y = pool["delay_log"].to_numpy(dtype=np.float64)          # 真实，对数尺度
    p = pool["pred_lgb"].to_numpy(dtype=np.float64)           # 折叠外预测，对数尺度
    ds = pool["delay_sec"].to_numpy(dtype=np.float64)
    ysec = np.expm1(y)
    psec = np.maximum(np.expm1(p), 1e-9)
    e = p - y
    res = dict(
        n=int(len(y)),
        med_abs_log=float(np.median(np.abs(e))),
        mean_abs_log=float(np.mean(np.abs(e))),
        med_sec=float(np.median(np.abs(psec - ysec))),
        mean_sec=float(np.mean(np.abs(psec - ysec))),
    )
    # 长延迟子群（真实延迟位于池内前 1%）被低估的比例与低估倍数
    thr = float(np.quantile(ds, 0.99))
    top = ds >= thr
    under = e[top] > 0                                  # e = pred − true < 0 才是低估
    res["top1_thr_sec"] = thr
    res["top1_n"] = int(top.sum())
    res["top1_underrate"] = float(np.mean(e[top] < 0))
    ratio = psec[top] / np.maximum(ysec[top], 1e-9)
    res["top1_ratio_med"] = float(np.median(ratio))
    # 操作上最危险的失败：被预测为"最快十分位"的作业里，真实延迟进入前 1% 的比例
    q10 = float(np.quantile(p, 0.10))
    fast = p <= q10
    res["pred_fast_n"] = int(fast.sum())
    res["pred_fast_is_top1"] = float(np.mean(ds[fast] >= thr))
    # 按作业形态分组（分母足够大才报）
    for col, key in (("spec", "spec"), ("jt", "jt")):
        g = {}
        for lvl in sorted(pool[col].astype(str).unique()):
            m = pool[col].astype(str).to_numpy() == lvl
            if m.sum() < 5000:
                continue
            ei = e[m]
            g[lvl] = dict(n=int(m.sum()),
                          med_abs_log=float(np.median(np.abs(ei))),
                          mean_abs_log=float(np.mean(np.abs(ei))),
                          underrate=float(np.mean(ei < 0)),
                          mean_delay_sec=float(np.mean(ds[m])))
        res[key] = g
    return res


def calib():
    q = json.load(open(QUA, encoding="utf-8"))
    nom = [q["q50"]["alpha"], q["q90"]["alpha"], q["q99"]["alpha"]]
    cov = [q["q50"]["coverage"], q["q90"]["coverage"], q["q99"]["coverage"]]
    return dict(nominal=nom, coverage=cov)


def pareto(adm):
    s = adm[(adm.place == "BF") & (adm.demand == "D_gpu")]
    cols = ["wait_mean_sec", "p99_wait_sec", "big_cov", "jain_place", "alloc_rate"]
    g = s.groupby("policy")[cols].mean()
    curve = []
    for i, lam in enumerate(LAMBDAS):
        pol = "Comp%d" % i
        if pol not in g.index:
            continue
        r = g.loc[pol]
        curve.append(dict(lam=lam, policy=pol, **{c: float(r[c]) for c in cols}))
    ref = {}
    for p in ["FCFS", "DAS", "BigFirst", "Random", "SJF", "DelayOracle"]:
        if p in g.index:
            ref[p] = {c: float(g.loc[p, c]) for c in cols}
    return dict(curve=curve, ref=ref)


def quality(pool, adm):
    """预测质量（与真实延迟的排序相关性）对 调度收益（平均等待）。

    除预测型打分器外，另把三种“只看单一作业属性”的固定序（最小请求、最短作业、
    大请求）作为对照点：它们与真实延迟的排序相关性都很低，用于在图上锚定“无预测”
    的一侧。
    """
    y = pool["delay_sec"].to_numpy(dtype=np.float64)
    req = pool["req"].to_numpy(dtype=np.float64)
    dur = pool["dur"].to_numpy(dtype=np.float64)
    score = {"SSF": req,
             "SJF": dur,
             "BigFirst": -req,
             "Random": pool["rand"].to_numpy(dtype=np.float64),
             "RuleDelay": pool["pred_rule"].to_numpy(dtype=np.float64),
             "L0Delay": pool["pred_l0"].to_numpy(dtype=np.float64),
             "DASq50": pool["pred_q50"].to_numpy(dtype=np.float64),
             "DASq90": pool["pred_q90"].to_numpy(dtype=np.float64),
             "DASq99": pool["pred_q99"].to_numpy(dtype=np.float64),
             "DAS": pool["pred_lgb"].to_numpy(dtype=np.float64)}
    s = adm[(adm.place == "BF") & (adm.demand == "D_gpu")]
    gw = s.groupby("policy")[["wait_mean_sec", "p99_wait_sec", "cap1_rate"]].mean()
    rows = []
    for p, sc in score.items():
        rows.append(dict(policy=p, rho=spearman(y, sc), wait=float(gw.loc[p, "wait_mean_sec"]),
                         p99=float(gw.loc[p, "p99_wait_sec"]), cap1=float(gw.loc[p, "cap1_rate"])))
    if "DelayOracle" in gw.index:
        rows.append(dict(policy="DelayOracle", rho=1.0, wait=float(gw.loc["DelayOracle", "wait_mean_sec"]),
                         p99=float(gw.loc["DelayOracle", "p99_wait_sec"]),
                         cap1=float(gw.loc["DelayOracle", "cap1_rate"])))
    return rows


def main():
    pool = pd.read_parquet(POOL)
    adm = pd.read_csv(ADM)
    res = dict(errors=errors(pool), calib=calib(), pareto=pareto(adm),
               quality=quality(pool, adm))
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    er = res["errors"]
    out("[extras] 误差：中位 |log| %.3f ｜ 中位秒误差 %.1f s（均值 %.0f s）"
        % (er["med_abs_log"], er["med_sec"], er["mean_sec"]))
    out("[extras] 长延迟子群（前 1%%，阈值 %.0f s）低估率 %.3f ｜ 中位 预测/真实 %.2f"
        % (er["top1_thr_sec"], er["top1_underrate"], er["top1_ratio_med"]))
    out("[extras] 被预测为最快十分位的作业中，真实延迟进前 1%% 的比例 %.4f"
        % er["pred_fast_is_top1"])
    out("[extras] 帕累托点 %d 个；质量点 %d 个 -> extras.json"
        % (len(res["pareto"]["curve"]), len(res["quality"])))
    for r in res["quality"]:
        out("   %-12s rho=%.4f  wait=%.1f  p99=%.0f  cap1=%.3f"
            % (r["policy"], r["rho"], r["wait"], r["p99"], r["cap1"]))
    for c in res["pareto"]["curve"]:
        out("   lam=%.2f  wait=%.1f  big=%.4f  jain=%.4f" % (c["lam"], c["wait_mean_sec"],
                                                            c["big_cov"], c["jain_place"]))


if __name__ == "__main__":
    main()
