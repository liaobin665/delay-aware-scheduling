# -*- coding: utf-8 -*-
r"""图件引用的读数。

只产出少量确定性的读数：按机型 / 作业类型的延迟中位数与平均 SHAP；归因空间主成分的
方差解释比例，以及它与真实对数延迟的排序相关性。这些数字由脚本生成，正文不手写。

输出  out/figure_readings.json

用法  python -m delay_aware_scheduling.analysis.exp69_fignums
"""
import json
import os

import numpy as np
import pandas as pd

from delay_aware_scheduling.prediction import exp62_models as E
from delay_aware_scheduling.common.seeds import SEED_EMBED
from delay_aware_scheduling.common.log import out


SHAPZ = os.path.join(E.PROD, "shap_values.npz")
EMBZ = os.path.join(E.PROD, "embed_sample.npz")
OUT = os.path.join(E.PROD, "figure_readings.json")

SPEC6 = ["XPU-E", "XPU-A", "H20", "A10", "L20", "XPU-C"]
JT3 = ["offline_inference", "training", "online_inference"]


def main():
    pool = pd.read_parquet(E.POOL, columns=["delay_sec", "spec", "jt"])
    dl = pool["delay_sec"].to_numpy(dtype=float)
    sp = pool["spec"].to_numpy()
    jt = pool["jt"].to_numpy()

    z = np.load(SHAPZ, allow_pickle=True)
    sv = z["shap"].astype(float)
    tot = sv.sum(1)                       # 该记录相对基准值在对数延迟上的偏移
    lab = np.load(EMBZ, allow_pickle=True)
    spec = np.asarray([str(x) for x in lab["spec"]])
    jt2 = np.asarray([str(x) for x in lab["jt"]])
    ylog = lab["y_log"].astype(float)

    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr
    P = PCA(n_components=2, random_state=SEED_EMBED)
    sc = P.fit_transform(sv)
    evr = float(P.explained_variance_ratio_[0])
    rho = float(spearmanr(sc[:, 0], ylog).statistic)

    K = {}
    for s in SPEC6:
        K["V4_MED_SPEC_%s" % s.replace("-", "")] = "{:,.0f}".format(float(np.median(dl[sp == s])))
        K["V4_SHAP_MACH_%s" % s.replace("-", "")] = "%+.2f" % float(tot[spec == s].mean())
    for j in JT3:
        K["V4_MED_JT_%s" % j.upper()] = "{:,.0f}".format(float(np.median(dl[jt == j])))
        K["V4_SHAP_JT_%s" % j.upper()] = "%+.2f" % float(tot[jt2 == j].mean())
    # 机型 × 作业类型交叉处的中位延迟（图 2(c) 的格子读数），n<50 的不报
    for s in SPEC6:
        for j in JT3:
            v = dl[(sp == s) & (jt == j)]
            if len(v) >= 50:
                K["V6_CROSS_%s_%s" % (s.replace("-", ""), j.upper())] = \
                    "{:,.0f}".format(float(np.median(v)))
    K["V4_EMB_PC1"] = "%.1f" % (100.0 * evr)
    K["V4_EMB_PC1_RHO"] = "%.3f" % rho

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(K, f, ensure_ascii=False, indent=1)
    out("[fignums] %d 键 -> %s" % (len(K), OUT))
    for k in sorted(K):
        out("   %-26s %s" % (k, K[k]))


if __name__ == "__main__":
    main()
