# -*- coding: utf-8 -*-

"""重放结果的读取与配对差检验。

同一格上的各策略共用同一批作业，因此策略间用配对差比较。
"""

import json
import math
import os
import numpy as np
import pandas as pd
from scipy import stats as st
from delay_aware_scheduling.config import OUT as PROD, ADM, CENSUS


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
