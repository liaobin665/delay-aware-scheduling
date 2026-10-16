# -*- coding: utf-8 -*-
r"""逐行 SHAP 值。

按与预测脚本完全相同的配置（同一份样本、同一组超参数、同一 random_state）重训一个
全量模型并算逐行 SHAP 值，落盘供蜂群图与依赖图使用。

不改动预测脚本的任何产物，只新增 out/shap_values.npz。

用法  python -m delay_aware_scheduling.analysis.exp67_shap
"""
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from delay_aware_scheduling.prediction import exp62_models as E  # 复用 build_X 与常量，保证同口径
from delay_aware_scheduling.common.log import out


OUT = os.path.join(E.PROD, "shap_values.npz")


def main():
    df = pd.read_parquet(E.SAMPLE)
    rs = df["ready_status"].to_numpy()
    rd = pd.to_numeric(df["ready_delay_sec"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    df = df.loc[(rs == 1) & (rd >= 0)].reset_index(drop=True)
    y = np.log1p(pd.to_numeric(df["ready_delay_sec"], errors="coerce").to_numpy(dtype=np.float64))
    vc = df["server_id"].value_counts()
    src = df["server_id"].map(vc).astype(np.float32).to_numpy()
    X = E.build_X(df, src, E.ALL_FEATS)
    cat_idx = [X.columns.get_loc(c) for c in E.CATS]

    out("[shap] 训练全量模型（与预测脚本同配置）")
    g = lgb.LGBMRegressor(**E.LGB_MSE)
    g.fit(X, y, categorical_feature=cat_idx)

    import shap
    sub = X.sample(min(20_000, len(X)), random_state=7)
    out("[shap] 对 %d 条子样本算 SHAP" % len(sub))
    ex = shap.TreeExplainer(g)
    sv = np.asarray(ex.shap_values(sub))
    if sv.ndim == 3:
        sv = sv[0]

    # 供蜂群图用的“数值化取值”：类别列用类别编码（图里按列各自归一化着色）
    Xv = sub.copy()
    for c in E.CATS:
        Xv[c] = Xv[c].cat.codes.astype(np.float32)
    Xv = Xv.astype(np.float32)

    base = float(np.asarray(ex.expected_value).ravel()[0])
    np.savez_compressed(
        OUT,
        shap=sv.astype(np.float32),
        val=Xv.to_numpy(dtype=np.float32),
        feats=np.array(list(sub.columns)),
        base_value=np.float32(base),
    )
    imp = np.mean(np.abs(sv), axis=0)
    order = np.argsort(-imp)
    out("[shap] base_value=%.4f（模型分数为对数延迟量纲）" % base)
    for i in order:
        out("   %-22s mean|SHAP|=%.4f" % (list(sub.columns)[i], imp[i]))
    out("[shap] ->", OUT)


if __name__ == "__main__":
    main()
