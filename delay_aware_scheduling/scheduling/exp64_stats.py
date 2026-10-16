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

import json
import os
import numpy as np
import pandas as pd
from delay_aware_scheduling.common.log import out
from delay_aware_scheduling.config import OUT as PROD, ADM, CENSUS

from .paired import load, paired
from .tables import EXT, LAB, MAIN, ORDER, PAIR_CTRL, PARE_LAM, tab1, tab2, tab3, tab4, tab5, tab6


PREVIEW = os.path.join(PROD, "tables_preview.md")


KEYMAP = {"wait_mean_sec": "wait", "eff_time_frac": "eff", "big_cov": "big",
          "alloc_rate": "alloc", "util": "util", "jain_place": "jain"}


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
