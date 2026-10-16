# -*- coding: utf-8 -*-
r"""低维嵌入与分群图用的样本。

逐行 SHAP 值文件里只有 SHAP 值与类别编码后的属性取值，既没有真实目标，也没有机型 /
作业类型等分组标签，无法给"特征空间降维"与"分群分布"图着色。本脚本用完全相同的
抽样口径补出标准化后的九项属性矩阵、真实对数就绪延迟，以及分组标签。

不改动其他脚本的任何产物，只新增 out/embed_sample.npz。

用法  python -m delay_aware_scheduling.analysis.exp68_embed
"""
import os

import numpy as np
import pandas as pd

from delay_aware_scheduling.prediction import exp62_models as E
from delay_aware_scheduling.common.log import out


OUT = os.path.join(E.PROD, "embed_sample.npz")


def main():
    df = pd.read_parquet(E.SAMPLE)
    rs = df["ready_status"].to_numpy()
    rd = pd.to_numeric(df["ready_delay_sec"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    df = df.loc[(rs == 1) & (rd >= 0)].reset_index(drop=True)
    y = np.log1p(pd.to_numeric(df["ready_delay_sec"], errors="coerce").to_numpy(dtype=np.float64))
    vc = df["server_id"].value_counts()
    src = df["server_id"].map(vc).astype(np.float32).to_numpy()
    X = E.build_X(df, src, E.ALL_FEATS)

    sub = X.sample(min(20_000, len(X)), random_state=7)
    idx = np.asarray(sub.index)
    Xv = sub.copy()
    for c in E.CATS:
        Xv[c] = Xv[c].cat.codes.astype(np.float32)
    Xv = Xv.astype(np.float32).to_numpy()
    sd = Xv.std(0)
    sd[sd == 0] = 1.0
    Xstd = ((Xv - Xv.mean(0)) / sd).astype(np.float32)

    np.savez_compressed(
        OUT,
        Xstd=Xstd,
        y_log=y[idx].astype(np.float32),
        spec=np.asarray(df["gpu_spec_public"].astype(str).to_numpy()[idx]),
        jt=np.asarray(df["job_type_public"].astype(str).to_numpy()[idx]),
        prio=np.asarray(df["priority_class"].astype(str).to_numpy()[idx]),
        feats=np.asarray(list(sub.columns)),
    )
    out("[embed] n=%d  y_log 中位=%.3f  ->  %s" % (len(idx), float(np.median(y[idx])), OUT))


if __name__ == "__main__":
    main()
