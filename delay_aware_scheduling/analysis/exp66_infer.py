# -*- coding: utf-8 -*-
r"""预测器推理开销实测。

按与预测脚本完全相同的配置训练一个 LightGBM，然后实测三种调用口径的单次推理耗时：
sklearn 接口逐条、booster 逐条（numpy 行）、以及批量化摊薄。另外报出模型规模（树数、
序列化字节数）与每秒可处理的作业数。

输出  out/infer.json

用法  python -m delay_aware_scheduling.analysis.exp66_infer
"""
import os
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb


from delay_aware_scheduling.config import OUT as PROD, SAMPLE
from delay_aware_scheduling.common.log import out


CATS = ["gpu_spec_public", "priority_class", "job_type_public", "model_type_public"]
NUM = ["gpu_request", "duration_hours", "is_genai_request", "schedule_delay_sec", "server_req_count"]
ALL_FEATS = CATS + NUM
LGB_MSE = dict(objective="regression", learning_rate=0.05, num_leaves=63, n_estimators=400,
               min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
               num_threads=6, verbose=-1)


def build_X(df, src):
    X = pd.DataFrame(index=df.index)
    for c in CATS:
        X[c] = df[c].astype(str).astype("category")
    X["gpu_request"] = df["gpu_request"].astype(np.float32)
    X["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(0.0).astype(np.float32)
    X["schedule_delay_sec"] = pd.to_numeric(df["schedule_delay_sec"], errors="coerce").fillna(-1.0).astype(np.float32)
    X["is_genai_request"] = (pd.to_numeric(df["is_genai_request"], errors="coerce").fillna(0) > 0).astype(np.int8)
    X["server_req_count"] = src.astype(np.float32)
    return X[ALL_FEATS]


def main():
    t0 = time.time()
    df = pd.read_parquet(SAMPLE)
    rs = df["ready_status"].to_numpy()
    rd = pd.to_numeric(df["ready_delay_sec"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    df = df.loc[(rs == 1) & (rd >= 0)].reset_index(drop=True)
    y = np.log1p(pd.to_numeric(df["ready_delay_sec"], errors="coerce").to_numpy(dtype=np.float64))
    vc = df["server_id"].value_counts()
    src = df["server_id"].map(vc).astype(np.float32).to_numpy()
    X = build_X(df, src)
    cat_idx = [X.columns.get_loc(c) for c in CATS]
    out("[infer] 建模队列 %s 行，训练中…" % format(len(df), ","))

    tm = time.perf_counter()
    g = lgb.LGBMRegressor(**LGB_MSE)
    g.fit(X, y, categorical_feature=cat_idx)
    train_sec = time.perf_counter() - tm
    booster = g.booster_
    n_trees = int(booster.num_trees())
    model_bytes = len(booster.model_to_string().encode("utf-8"))

    # 取一段折叠外样本（此处只测延迟，不涉及精度）
    n_single = 2000
    head = X.iloc[:n_single].reset_index(drop=True)

    # (1) sklearn 接口逐条推理（含 pandas 行切片开销，最接近"逐作业调用"）
    t = time.perf_counter()
    for i in range(n_single):
        g.predict(head.iloc[[i]])
    us_sk = (time.perf_counter() - t) / n_single * 1e6

    # (2) booster + numpy 单行（框架开销已摊掉，只剩一次前向）
    Xhead = head.copy()
    for c in CATS:
        Xhead[c] = Xhead[c].cat.codes.astype(np.int16)
    arr = Xhead.to_numpy()
    t = time.perf_counter()
    for i in range(n_single):
        booster.predict(arr[i:i + 1])
    us_bo = (time.perf_counter() - t) / n_single * 1e6

    # (3) 批量化：一次预测 10 万行，摊薄到单条
    big = Xhead.iloc[:min(100_000, len(Xhead))].to_numpy()
    t = time.perf_counter()
    booster.predict(big)
    batch_sec = time.perf_counter() - t
    us_batch = batch_sec / len(big) * 1e6

    res = dict(
        n_train=int(len(df)), n_trees=n_trees, n_features=int(X.shape[1]),
        model_kb=round(model_bytes / 1024.0, 1), train_sec=round(train_sec, 1),
        single_sklearn_us=round(us_sk, 2), single_booster_us=round(us_bo, 2),
        batch_rows=int(len(big)), batch_us_per_row=round(us_batch, 3),
        per_sec_single=int(1e6 / us_bo), per_sec_batch=int(1e6 / us_batch),
        note="同一配置（LGB_MSE，400 棵树、63 叶）在建模队列上训练一次；仅测推理延迟，不涉及精度。",
    )
    json.dump(res, open(os.path.join(PROD, "infer.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    out("[infer] 树 %d ｜ 模型 %s KB ｜ sklearn 逐条 %.1f us ｜ booster 逐条 %.1f us ｜ "
        "批量 %.3f us/条（%.0f 万条/秒）"
        % (n_trees, res["model_kb"], us_sk, us_bo, us_batch, res["per_sec_batch"] / 1e4))
    out("[infer] -> infer.json（%.0fs）" % (time.time() - t0))


if __name__ == "__main__":
    main()
