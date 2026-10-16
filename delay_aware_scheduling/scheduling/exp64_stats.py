# -*- coding: utf-8 -*-
r"""汇总统计、配对检验与结果表。

配对设计：同一 (集群, 种子, 需求口径, 负载, 放置规则) 为一格，各策略共用同一批作业。
策略间因此用【配对差】（本文方法减对照）比较，给出配对 t 检验的 95% 置信区间与双侧
p 值，并以 Wilcoxon 符号秩检验作为分布无关的旁证。结果表由脚本生成，数字不手写。

输出  out/stats.json           扁平数值表
      out/tables.json          结果表的 LaTeX 表体
      out/tables_preview.md    同一批表的纯文本预览（人读用）

用法  python -m delay_aware_scheduling.scheduling.exp64_stats
"""
import os
import json
import math

import numpy as np
import pandas as pd
from scipy import stats as st


from delay_aware_scheduling.config import OUT as PROD, ADM, CENSUS
from delay_aware_scheduling.common.log import out


PREVIEW = os.path.join(PROD, "tables_preview.md")


MAIN = [("wait_mean_sec", "平均等待时间（秒）", "lower"),
        ("eff_time_frac", "有效计算时间占比", "higher"),
        ("big_cov", "大请求有效覆盖率", "higher"),
        ("alloc_rate", "已放置作业比例", "higher"),
        ("util", "GPU 占用率", "higher"),
        ("jain_place", "各请求规模档公平性", "higher")]
LAB = {"FCFS": "先到先服务", "SJF": "最短作业优先", "SSF": "最小请求优先",
       "BigFirst": "大请求优先", "Random": "随机排序", "RuleDelay": "查表延迟排序",
       "L0Delay": "仅提交时特征排序", "DAS": "延迟感知调度（本文）",
       "DelayOracle": "按真实延迟排序",
       "DASq50": "延迟感知调度（P50 打分）", "DASq90": "延迟感知调度（P90 打分）",
       "DASq99": "延迟感知调度（P99 打分）"}
ORDER = ["FCFS", "SJF", "SSF", "BigFirst", "Random", "RuleDelay", "L0Delay",
         "DAS", "DelayOracle"]
EXT = ["DASq50", "DASq90", "DASq99"]
PAIR_CTRL = ["FCFS", "SJF", "SSF", "BigFirst", "Random", "RuleDelay", "L0Delay", "DelayOracle"]
PARE_LAM = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
KEYMAP = {"wait_mean_sec": "wait", "eff_time_frac": "eff", "big_cov": "big",
          "alloc_rate": "alloc", "util": "util", "jain_place": "jain"}


def f2(x, nd=4):
    return ("%%.%df" % nd) % x


def f3(x):
    return "{:,.0f}".format(x)


def load():
    d = {k: json.load(open(os.path.join(PROD, f), encoding="utf-8"))
         for k, f in (("cal", "delay_caliber.json"), ("mod", "models.json"),
                      ("q", "quantile.json"))}
    adm = pd.read_csv(ADM)
    return d, adm


def paired(adm, metric, a="DAS", b="FCFS", place="BF", demand="D_gpu"):
    s = adm[(adm.place == place) & (adm.demand == demand)]
    piv = s.pivot_table(index=["fleet", "seed", "rho"], columns="policy", values=metric)
    if a not in piv.columns or b not in piv.columns:
        return None
    d = (piv[a] - piv[b]).dropna().to_numpy(dtype=np.float64)
    n = len(d)
    if n < 2:
        return None
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    tcrit = float(st.t.ppf(0.975, n - 1))
    try:
        w_p = float(st.wilcoxon(d).pvalue)
    except Exception:
        w_p = None
    try:
        t_p = float(st.ttest_rel(piv[a].dropna(), piv[b].dropna()).pvalue)
    except Exception:
        t_p = None
    return dict(n=n, mean=m, lo=m - tcrit * se, hi=m + tcrit * se,
                p=t_p, p_wilcoxon=w_p)


def tab1(d):
    c, dist = d["cal"], d["cal"]["dist"]
    rows = [
        ("作业数", f3(dist["n"])),
        ("中位数", f3(dist["median"])),
        ("P90", f3(dist["p90"])),
        ("P99", f3(dist["p99"])),
        ("最大值", f3(dist["max"])),
        ("均值", f3(dist["mean"])),
    ]
    s = [r"\begin{tabular}{lrrrr}", r"\hline",
         r"统计量 & 中位数 & P90 & P99 & 最大值 \\", r"\hline",
         r"就绪延迟（秒） & %s & %s & %s & %s \\" % (f3(dist["median"]), f3(dist["p90"]),
                                                     f3(dist["p99"]), f3(dist["max"])),
         r"\hline", r"\end{tabular}"]
    return "\n".join(s)


MORDER = ["const", "rule", "linear", "tree", "rf", "xgb", "lgb", "l0"]
MLAB = {"const": "常数中位数", "rule": "查表中位", "linear": "线性回归", "tree": "决策树",
        "rf": "随机森林", "xgb": "XGBoost", "lgb": "LightGBM", "l0": "LightGBM（仅 L0）"}


def tab2(d):
    ms = d["mod"]["models"]
    # 块 A：排序与典型误差（对长尾稳健的主口径）
    A = ["\\begin{tabular}{lrrrrr}", "\\hline",
         "模型 & Spearman $\\rho$ & Kendall $\\tau$ & 中位绝对误差 & 中位绝对误差 & 2 倍以内 \\\\",
         "     &                  &               & （对数）      & （秒）       & 命中率 \\\\", "\\hline"]
    for k in MORDER:
        m = ms[k]
        # 常数打分器（按折取训练中位数，逐折不同）的秩相关无定义，不报伪值
        rho_s = "---" if k == "const" else f2(m["spearman_log"], 4)
        tau_s = "---" if k == "const" else f2(m["kendall_log"], 4)
        A.append("%s & %s & %s & %s & %s & %s \\\\" % (
            MLAB[k], rho_s, tau_s,
            f2(m["medae_log"], 4), f3(m["medae_sec"]), f2(m["within2_log"], 4)))
    A += ["\\hline", "\\end{tabular}"]
    # 块 B：均值类对照（受长尾主导，仅作参照）
    B = ["\\begin{tabular}{lrrrr}", "\\hline",
         "模型 & MAE（对数） & RMSE（对数） & $R^2$（对数） & MAE（秒） \\\\", "\\hline"]
    for k in MORDER:
        m = ms[k]
        B.append("%s & %s & %s & %s & %s \\\\" % (
            MLAB[k], f2(m["mae_log"], 4), f2(m["rmse_log"], 4), f2(m["r2_log"], 4),
            f3(m["mae_sec"])))
    B += ["\\hline", "\\end{tabular}"]
    # 块 C：分位数回归
    C = ["\\begin{tabular}{lrrrr}", "\\hline",
         "分位数 & Pinball 损失 & 实测覆盖率 & 覆盖率偏差 & 中位绝对误差（对数） \\\\", "\\hline"]
    for k in ["q50", "q90", "q99"]:
        v = d["q"][k]
        bias = abs(float(v["coverage"]) - float(v["alpha"]))
        C.append("P%d & %s & %s & %s & %s \\\\" % (int(v["alpha"] * 100),
                 f2(v["pinball_log"], 4), f2(v["coverage"], 4), f2(bias, 4),
                 f2(v["mae_log"], 4)))
    C += ["\\hline", "\\end{tabular}"]
    return (("\n".join(A) + "\n\n\\vspace{3pt}\n"
             "\\noindent\\emph{\\footnotesize 均值类对照（受长尾主导，仅作参照）：}\n\n"
             + "\n".join(B) + "\n\n\\vspace{3pt}\n"
             "\\noindent\\emph{\\footnotesize 分位数回归（LightGBM）：}\n\n"
             + "\n".join(C)))


def tab3(d, adm):
    s = adm[(adm.place == "BF") & (adm.demand == "D_gpu")]
    g = s.groupby("policy")[[m for m, _, _ in MAIN]].mean()
    gp99 = s.groupby("policy")["p99_wait_sec"].mean()
    gcap = s.groupby("policy")["cap1_rate"].mean()
    rows = ["\\begin{tabular}{lrrrrrrr}", "\\hline",
            "策略 & 平均等待 & P99 等待 & 长延迟 & 有效计算 & 大请求 & 各档 & 已放置 \\\\",
            "     & 时间（秒） & 时间（秒） & 捕获率 & 时间占比 & 覆盖率 & 公平性 & 比例 \\\\", "\\hline"]
    for p in ORDER:
        r = g.loc[p]
        rows.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            LAB[p], f3(r["wait_mean_sec"]), f3(gp99.loc[p]), f2(gcap.loc[p], 4),
            f2(r["eff_time_frac"], 4), f2(r["big_cov"], 4),
            f2(r["jain_place"], 4), f2(r["alloc_rate"], 4)))
    rows += ["\\hline",
             "\\multicolumn{8}{l}{\\emph{配对差：延迟感知调度减各对照（同一格共用同一批作业）}} \\\\",
             "\\hline",
             "对照策略 & 平均等待差（95\\% 置信区间） & $p$ & 有效计算占比差 & $p$ & & & \\\\", "\\hline"]
    for b in PAIR_CTRL:
        pw = paired(adm, "wait_mean_sec", "DAS", b)
        pe = paired(adm, "eff_time_frac", "DAS", b)
        if pw is None:
            continue
        rows.append("%s & %s [%s, %s] & %s & %s & %s & & & \\\\" % (
            LAB[b], f3(pw["mean"]), f3(pw["lo"]), f3(pw["hi"]), _p(pw["p"]),
            f2(pe["mean"], 4), _p(pe["p"])))
    rows += ["\\hline", "\\end{tabular}"]
    return "\n".join(rows)


def _p(p):
    if p is None:
        return "—"
    if p < 1e-4:
        return "$<10^{-4}$"
    return f2(p, 4)


def tab4(adm):
    rows = ["\\begin{tabular}{llrrrr}", "\\hline",
            "放置规则 & 策略 & 平均等待（秒） & 有效计算占比 & 大请求覆盖率 & 已放置比例 \\\\",
            "\\hline"]
    NAMEP = {"BF": "最佳适配", "FF": "首次适配", "WF": "最差适配"}
    for pl in ["BF", "FF", "WF"]:
        s = adm[(adm.place == pl) & (adm.demand == "D_gpu")]
        g = s.groupby("policy")[["wait_mean_sec", "eff_time_frac", "big_cov", "alloc_rate"]].mean()
        for p in ["FCFS", "DAS", "DelayOracle"]:
            if p not in g.index:
                continue
            r = g.loc[p]
            rows.append("%s & %s & %s & %s & %s & %s \\\\" % (
                NAMEP[pl] if p == "FCFS" else "", LAB[p], f3(r["wait_mean_sec"]),
                f2(r["eff_time_frac"], 4), f2(r["big_cov"], 4), f2(r["alloc_rate"], 4)))
        rows.append("\\hline")
    rows += ["\\end{tabular}"]
    return "\n".join(rows)


SZ = ["le0p5", "lt1", "eq1", "eq2", "eq4", "eq8", "ge16"]
SZLAB = {"le0p5": "≤0.5", "lt1": "0.5–1", "eq1": "=1", "eq2": "=2", "eq4": "=4",
         "eq8": "=8", "ge16": "≥16"}


def tab5(adm):
    s = adm[(adm.place == "BF") & (adm.demand == "D_gpu")]
    rows = ["\\begin{tabular}{lrrrr}", "\\hline",
            "请求规模（GPU） & 提交数占比 & 先到先服务放置率 & 延迟感知放置率 & 变化 \\\\",
            "\\hline"]
    for c in SZ:
        ncol = "nsub_" + c
        rcol = "rate_" + c
        if ncol not in s.columns:
            continue
        sub = s[s.policy == "FCFS"]
        share = float(sub[ncol].sum()) / max(float(sub["submitted"].sum()), 1)
        if share < 0.001:
            continue          # 该档在数据中近乎为空（如"0.5 到 1 卡"），报出会误导
        a = s[s.policy == "FCFS"][rcol].mean()
        b = s[s.policy == "DAS"][rcol].mean()
        rows.append("%s & %s & %s & %s & %s \\\\" % (
            SZLAB[c], f2(share, 4), f2(a, 4), f2(b, 4), _sgn(b - a)))
    rows += ["\\hline", "\\end{tabular}"]
    return "\n".join(rows)


def _sgn(x):
    return ("+" if x >= 0 else "-") + f2(abs(x), 4)


def tab6(adm):
    """复合打分 s = d̂ − λ·(r/̄r) 的公平性-效率权衡。"""
    s = adm[(adm.place == "BF") & (adm.demand == "D_gpu")]
    g = s.groupby("policy")[["wait_mean_sec", "p99_wait_sec", "big_cov",
                             "jain_place", "alloc_rate"]].mean()
    rows = ["\\begin{tabular}{lrrrr}", "\\hline",
            "补偿权重 $\\lambda$ & 平均等待（秒） & 大请求覆盖率 & 各档公平性 & 已放置比例 \\\\",
            "\\hline",
            "0（延迟感知调度） & %s & %s & %s & %s \\\\" % (
                f3(g.loc["DAS", "wait_mean_sec"]), f2(g.loc["DAS", "big_cov"], 4),
                f2(g.loc["DAS", "jain_place"], 4), f2(g.loc["DAS", "alloc_rate"], 4))]
    for i, l in enumerate(PARE_LAM):
        p = "Comp%d" % i
        if p not in g.index:
            continue
        r = g.loc[p]
        rows.append("%s & %s & %s & %s & %s \\\\" % (
            f2(l, 2), f3(r["wait_mean_sec"]), f2(r["big_cov"], 4),
            f2(r["jain_place"], 4), f2(r["alloc_rate"], 4)))
    rows += ["\\hline", "\\end{tabular}"]
    return "\n".join(rows)


def main():
    d, adm = load()

    tabs = {"TAB1": tab1(d), "TAB2": tab2(d), "TAB3": tab3(d, adm),
            "TAB4": tab4(adm), "TAB5": tab5(adm), "TAB6": tab6(adm)}
    json.dump(tabs, open(os.path.join(PROD, "tables.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    s = adm[(adm.place == "BF") & (adm.demand == "D_gpu")]
    g = s.groupby("policy")[[m for m, _, _ in MAIN]].mean()
    nums = {}
    for p in ORDER:
        for met, _, _ in MAIN:
            nums["ADM_%s_%s" % (p.upper(), KEYMAP[met].upper())] = float(g.loc[p, met])
    for b in PAIR_CTRL + EXT:
        for met, _, _ in MAIN:
            pr = paired(adm, met, "DAS", b)
            if pr:
                t = "%s_%s" % (KEYMAP[met].upper(), b.upper())
                nums["PD_" + t] = pr["mean"]
                nums["PD_LO_" + t] = pr["lo"]
                nums["PD_HI_" + t] = pr["hi"]
                nums["PP_" + t] = pr["p"]
                nums["PW_" + t] = pr["p_wilcoxon"]
    # 尾部长度的分布读数与无效占用
    for p in ORDER:
        row = s[s.policy == p]
        nums["ADM_%s_P99WAIT" % p.upper()] = float(row["p99_wait_sec"].mean())
        nums["ADM_%s_CAP1" % p.upper()] = float(row["cap1_rate"].mean())
        nums["ADM_%s_WHOCC" % p.upper()] = float(row["w_occ_gpuh"].mean())
    # 复合打分的公平性-效率曲线
    gp = s.groupby("policy")[["wait_mean_sec", "big_cov", "jain_place", "alloc_rate"]].mean()
    for p in ["DAS", "FCFS", "BigFirst"]:
        nums["PC_%s_WAIT" % p.upper()] = float(gp.loc[p, "wait_mean_sec"])
        nums["PC_%s_BIG" % p.upper()] = float(gp.loc[p, "big_cov"])
        nums["PC_%s_JAIN" % p.upper()] = float(gp.loc[p, "jain_place"])
    for i, l in enumerate(PARE_LAM):
        p = "Comp%d" % i
        if p in gp.index:
            nums["PC%d_WAIT" % i] = float(gp.loc[p, "wait_mean_sec"])
            nums["PC%d_BIG" % i] = float(gp.loc[p, "big_cov"])
            nums["PC%d_JAIN" % i] = float(gp.loc[p, "jain_place"])
    nums["PC_LAM_LIST"] = "、".join("%.2f" % l for l in PARE_LAM)
    nums["PC_MAXLAM"] = "%.2f" % PARE_LAM[-1]
    nums["PC_MIDLAM"] = "%.2f" % PARE_LAM[3]
    # 落点敏感性：每档的 DAS 相对 FCFS 改善
    for pl in ["BF", "FF", "WF"]:
        for met, _, _ in MAIN[:3]:
            pr = paired(adm, met, "DAS", "FCFS", place=pl)
            if pr:
                nums["PLACE_%s_%s" % (pl, KEYMAP[met].upper())] = pr["mean"]
    # 预测器质量敏感性
    for p in ["Random", "RuleDelay", "L0Delay", "DASq50", "DASq90", "DASq99", "DAS", "DelayOracle"]:
        for met, _, _ in MAIN[:3]:
            v = s[s.policy == p][met].mean()
            if np.isfinite(v):
                nums["SENS_%s_%s" % (p.upper(), KEYMAP[met].upper())] = float(v)
    nums["ADM_RHO_LIST"] = "、".join("%.2f" % x for x in sorted(adm["rho"].unique()))
    nums["SENS_DL0_WAIT"] = float(nums["SENS_DAS_WAIT"] - nums["SENS_L0DELAY_WAIT"])
    nums["ADM_N_CELLS"] = int(s[["fleet", "seed", "rho"]].drop_duplicates().shape[0])
    nums["ADM_N_ROWS"] = int(len(s))
    nums["ADM_N_SEEDS"] = int(adm["seed"].nunique())
    nums["ADM_N_RHO"] = int(adm["rho"].nunique())
    try:
        cen = pd.read_parquet(CENSUS, columns=["gpu_count", "in_ref_day"])
        cen = cen[cen["in_ref_day"]]
        nums["REAL_MACH"] = int(len(cen))
        nums["REAL_CARDS"] = int(cen["gpu_count"].sum())
    except Exception as e:
        out("[stats] 参考日供给读数不可用：%s" % e)
    # 表 3 用于说明“条件均值 vs 条件中位数”口径差异的两个读数
    md = d["mod"]["models"]
    nums["M_LIN_MEDAELOG"] = float(md["linear"]["medae_log"])
    nums["M_CONST_MEDAELOG"] = float(md["const"]["medae_log"])
    json.dump(nums, open(os.path.join(PROD, "stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    with open(PREVIEW, "w", encoding="utf-8") as f:
        for k, v in tabs.items():
            f.write("## %s\n\n```\n%s\n```\n\n" % (k, v))
    out("[stats] 表体 %d 个；数值键 %d 个" % (len(tabs), len(nums)))
    out("[stats] 主口径（落点=BF，需求=GPU 卡数）\n" + g.round(4).to_string())
    out("\n[stats] 配对差（DAS − 对照，平均等待时间/秒）")
    for b in PAIR_CTRL:
        pr = paired(adm, "wait_mean_sec", "DAS", b)
        if pr:
            out("  %-22s 均值 %8.1f  95%%CI [%8.1f, %8.1f]  p=%s  n=%d"
                % (LAB[b], pr["mean"], pr["lo"], pr["hi"],
                   ("%.1e" % pr["p"]) if pr["p"] is not None else "—", pr["n"]))
    out("\n[stats] 产物：stats.json / tables.json / tables_preview.md")


if __name__ == "__main__":
    main()
