# -*- coding: utf-8 -*-

"""结果表的组装。表体由脚本生成，数字不手写。
"""

from .paired import paired


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


def f2(x, nd=4):
    return ("%%.%df" % nd) % x


def f3(x):
    return "{:,.0f}".format(x)


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
