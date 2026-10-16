# -*- coding: utf-8 -*-

"""特征构造与折分配。

九项准入时刻属性按 L0 / L1 两层组装成特征矩阵；折按机器分组分配，
同一台机器只落在训练或验证的一侧。
"""

import numpy as np
import pandas as pd


CATS = ["gpu_spec_public", "priority_class", "job_type_public", "model_type_public"]


NUM_L0 = ["gpu_request", "duration_hours", "is_genai_request"]


NUM_L1 = ["schedule_delay_sec", "server_req_count"]


ALL_FEATS = CATS + NUM_L0 + NUM_L1


def fold_assign(groups, seed, k):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    of = {kk: i % k for i, kk in enumerate(uniq)}
    return np.array([of[kk] for kk in groups], dtype=np.int8)


def build_X(df, src, cols):
    X = pd.DataFrame(index=df.index)
    for c in CATS:
        if c in cols:
            X[c] = df[c].astype(str).astype("category")
    X["gpu_request"] = df["gpu_request"].astype(np.float32)
    X["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(0.0).astype(np.float32)
    X["schedule_delay_sec"] = pd.to_numeric(df["schedule_delay_sec"], errors="coerce").fillna(-1.0).astype(np.float32)
    X["is_genai_request"] = (pd.to_numeric(df["is_genai_request"], errors="coerce").fillna(0) > 0).astype(np.int8)
    if "server_req_count" in cols:
        X["server_req_count"] = src.astype(np.float32)
    return X[[c for c in cols]]
