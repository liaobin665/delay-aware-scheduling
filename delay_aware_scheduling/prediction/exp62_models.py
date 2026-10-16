# -*- coding: utf-8 -*-

r"""就绪延迟的点估计与分位数预测。

队列  ：已就绪且记录了延迟的记录；目标取对数，以压缩跨越五个数量级的延迟跨度。
特征  ：九项准入时刻可观测属性。L0 七项在作业提交时即可获得；L1 两项在队列与调度
        环节才产生。两者分开建一组对照，回答"只用提交时可得的信息还剩多少能力"。
切分  ：按 server_id 分组五折，同一台机器不同时出现在训练与测试两侧。

模型  ：常数中位数、线性回归、决策树、随机森林、XGBoost、LightGBM（MSE），以及
        LightGBM 分位数回归 P50/P90/P99。全部指标在【折叠外】预测上汇总。
另给三项对照：仅用 L0 特征、零参数查表基线（机型 × 作业类型 × 声明时长十分位的中位
延迟）、以及分位数实测覆盖率与名义值的偏差。

输出  out/delay_pool.parquet   带折叠外预测的重放需求池
      out/models.json         模型性能表
      out/importance.json     特征重要性（分裂增益 + 平均绝对 SHAP）
      out/quantile.json       分位数损失与覆盖率

用法  python -m delay_aware_scheduling.prediction.exp62_models [--quick]
"""

import argparse
import json
import os
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from delay_aware_scheduling.common.log import out
from delay_aware_scheduling.common.seeds import SEED_FOLD, SEED_SHUFFLE as SEED_RAND
from delay_aware_scheduling.config import OUT as PROD, SAMPLE, POOL

from .features import ALL_FEATS, CATS, NUM_L0, NUM_L1, build_X, fold_assign
from .metrics import metrics, pinball


N_FOLD = 5


LGB_MSE = dict(objective="regression", learning_rate=0.05, num_leaves=63, n_estimators=400,
               min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
               num_threads=6, verbose=-1)


LGB_Q = dict(objective="quantile", learning_rate=0.05, num_leaves=63, n_estimators=400,
             min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
             num_threads=6, verbose=-1)


QUANTILES = [0.50, 0.90, 0.99]


SIZE_BINS = [0.5, 1, 2, 4, 8, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_parquet(SAMPLE)
    rs = df["ready_status"].to_numpy()
    rd = pd.to_numeric(df["ready_delay_sec"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    df = df.loc[(rs == 1) & (rd >= 0)].reset_index(drop=True)
    if args.quick:
        df = df.sample(min(200_000, len(df)), random_state=1).reset_index(drop=True)
    y_raw = pd.to_numeric(df["ready_delay_sec"], errors="coerce").to_numpy(dtype=np.float64)
    y = np.log1p(y_raw)
    vc = df["server_id"].value_counts()
    src = df["server_id"].map(vc).astype(np.float32).to_numpy()
    out("[models] 建模队列 %s 行 ｜ 延迟中位 %.0fs ｜ log 目标均值 %.4f"
        % (format(len(df), ","), np.median(y_raw), y.mean()))

    X = build_X(df, src, ALL_FEATS)
    X_l0 = build_X(df, src, CATS + NUM_L0)
    # sklearn / xgboost 不吃 pandas category，另做整数编码版；LightGBM 继续用 category
    Xnum = X.copy()
    for c in CATS:
        Xnum[c] = Xnum[c].cat.codes.astype(np.int16)
    cat_idx = [X.columns.get_loc(c) for c in CATS]
    groups = df["server_id"].astype(str).to_numpy()
    fold = fold_assign(groups, SEED_FOLD, N_FOLD)
    out("[models] 机器 %s 台，五折（按机器分组）" % format(len(np.unique(groups)), ","))

    keys = ["const", "linear", "tree", "rf", "xgb", "lgb", "l0", "q50", "q90", "q99", "rule"]
    pred = {k: np.full(len(df), np.nan) for k in keys}
    fold_rows = []
    dur = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    jt = df["job_type_public"].astype(str).to_numpy()
    spec = df["gpu_spec_public"].astype(str).to_numpy()

    # sklearn 视角的输入（独热）
    Xsk_cols = pd.DataFrame(index=df.index)
    for c in CATS:
        Xsk_cols[c] = df[c].astype(str)
    for c in NUM_L0 + NUM_L1:
        Xsk_cols[c] = X[c].to_numpy()

    for f in range(N_FOLD):
        t1 = time.time()
        tr, va = fold != f, fold == f
        ytr, yva = y[tr], y[va]
        pred["const"][va] = float(np.median(ytr))

        # 线性回归
        ct = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
            ("num", StandardScaler(), NUM_L0 + NUM_L1)])
        lin = Pipeline([("ct", ct), ("lr", LinearRegression())])
        lin.fit(Xsk_cols[tr], ytr)
        pred["linear"][va] = lin.predict(Xsk_cols[va])

        dt = DecisionTreeRegressor(min_samples_leaf=50, random_state=SEED_FOLD)
        dt.fit(Xnum[tr], ytr)
        pred["tree"][va] = dt.predict(Xnum[va])

        rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=20, n_jobs=6,
                                   max_samples=0.5, random_state=SEED_FOLD)
        rf.fit(Xnum[tr], ytr)
        pred["rf"][va] = rf.predict(Xnum[va])

        xg = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                              tree_method="hist", n_jobs=6, random_state=SEED_FOLD,
                              subsample=0.8, colsample_bytree=0.8)
        xg.fit(Xnum[tr], ytr)
        pred["xgb"][va] = xg.predict(Xnum[va])

        g = lgb.LGBMRegressor(**LGB_MSE)
        g.fit(X[tr], ytr, categorical_feature=cat_idx)
        pred["lgb"][va] = g.predict(X[va])

        g0 = lgb.LGBMRegressor(**LGB_MSE)
        g0.fit(X_l0[tr], ytr, categorical_feature=cat_idx)
        pred["l0"][va] = g0.predict(X_l0[va])

        for q in QUANTILES:
            gq = lgb.LGBMRegressor(alpha=q, **LGB_Q)
            gq.fit(X[tr], ytr, categorical_feature=cat_idx)
            pred["q%d" % int(q * 100)][va] = gq.predict(X[va])

        # 零参数查表：机型 × 作业类型 × 时长十分位 的训练集中位
        edges = np.percentile(dur[tr], np.arange(10, 100, 10))
        key = spec + "#" + jt + "#" + np.digitize(dur, edges).astype(str)
        tab = pd.DataFrame({"k": key[tr], "v": ytr}).groupby("k")["v"].median()
        pred["rule"][va] = pd.Series(key[va]).map(tab).fillna(np.median(ytr)).to_numpy()

        fm = metrics(yva, pred["lgb"][va])
        fold_rows.append(dict(fold=f, n_train=int(tr.sum()), n_valid=int(va.sum()),
                              r2_lgb=round(fm["r2_log"], 6),
                              rho_lgb=round(fm["spearman_log"], 6),
                              sec=round(time.time() - t1, 1)))
        out("  [fold %d] 训练 %s / 验证 %s ｜ LGBM R²=%.4f ρ=%.4f  (%.0fs)"
            % (f, format(int(tr.sum()), ","), format(int(va.sum()), ","),
               fold_rows[-1]["r2_lgb"], fold_rows[-1]["rho_lgb"], fold_rows[-1]["sec"]))

    # ---------- 指标 ----------
    names = {"const": "常数中位数", "linear": "线性回归", "tree": "决策树", "rf": "随机森林",
             "xgb": "XGBoost", "lgb": "LightGBM（MSE）", "l0": "LightGBM（仅 L0，7 特征）",
             "rule": "查表中位（机型×作业类型×时长）"}
    mtab = {k: metrics(y, pred[k]) for k in ["const", "linear", "tree", "rf", "xgb", "lgb", "l0", "rule"]}
    for k in mtab:
        mtab[k]["label"] = names.get(k, k)

    # ---------- 分位数 ----------
    qtab = {}
    for q in QUANTILES:
        k = "q%d" % int(q * 100)
        yq, pq = y, pred[k]
        qtab[k] = dict(alpha=q, pinball_log=pinball(yq, pq, q),
                       coverage=float(np.mean(yq <= pq)),
                       mae_log=float(np.mean(np.abs(pq - yq))),
                       r2_log=metrics(yq, pq)["r2_log"])

    # ---------- 重要性 ----------
    imp = {}
    try:
        g_full = lgb.LGBMRegressor(**LGB_MSE)
        g_full.fit(X, y, categorical_feature=cat_idx)
        gain = g_full.booster_.feature_importance(importance_type="gain")
        tot = float(gain.sum()) or 1.0
        imp["gain"] = {c: float(v) / tot for c, v in zip(X.columns, gain)}
        imp["split"] = {c: int(v) for c, v in zip(X.columns,
                        g_full.booster_.feature_importance(importance_type="split"))}
    except Exception as e:
        imp["gain_error"] = str(e)

    try:
        import shap
        sub = X.sample(min(20_000, len(X)), random_state=7)
        ex = shap.TreeExplainer(g_full)
        sv = ex.shap_values(sub)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[0]
        imp["shap_mean_abs"] = {c: float(np.mean(np.abs(sv[:, i]))) for i, c in enumerate(sub.columns)}
    except Exception as e:
        imp["shap_error"] = str(e)

    # ---------- 写池 ----------
    req = df["gpu_request"].to_numpy(dtype=np.float64)
    keep = (req > 0) & np.isfinite(dur) & (dur >= 0) & np.isfinite(y_raw)
    rng = np.random.default_rng(SEED_RAND)
    rnd = rng.random(len(df))
    pool = pd.DataFrame(dict(
        req=req[keep].astype(np.float32), dur=dur[keep].astype(np.float32),
        delay_sec=y_raw[keep].astype(np.float64), delay_log=y[keep].astype(np.float32),
        pred_lgb=pred["lgb"][keep].astype(np.float32),
        pred_q50=pred["q50"][keep].astype(np.float32),
        pred_q90=pred["q90"][keep].astype(np.float32),
        pred_q99=pred["q99"][keep].astype(np.float32),
        pred_rule=pred["rule"][keep].astype(np.float32),
        pred_l0=pred["l0"][keep].astype(np.float32),
        rand=rnd[keep].astype(np.float32),
        spec=spec[keep], jt=jt[keep],
    )).reset_index(drop=True)
    pool.to_parquet(POOL, index=False)

    meta = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), n=int(len(df)),
                n_pool=int(len(pool)), n_fold=N_FOLD, seed_fold=SEED_FOLD, seed_rand=SEED_RAND,
                folds="按 server_id 分组五折", feats=ALL_FEATS, feats_l0=CATS + NUM_L0,
                folds_detail=fold_rows, models=mtab, quantile=qtab, importance=imp,
                note="全部预测均为折叠外；server_req_count 为同机作业条数（样本内统计口径，见 docs/schema.md）")
    json.dump(meta, open(os.path.join(PROD, "models.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(dict(importance=imp), open(os.path.join(PROD, "importance.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(qtab, open(os.path.join(PROD, "quantile.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    out("\n[models] ===== 表 2 候选 =====")
    out("  %-26s %9s %9s %8s %9s %8s" % ("模型", "MAE(log)", "RMSE(log)", "R²(log)", "MAE(s)", "ρ"))
    for k in ["const", "rule", "linear", "tree", "rf", "xgb", "lgb", "l0"]:
        m = mtab[k]
        out("  %-26s %9.4f %9.4f %8.4f %9.1f %8.4f"
            % (m["label"], m["mae_log"], m["rmse_log"], m["r2_log"], m["mae_sec"], m["spearman_log"]))
    out("\n[models] ===== 分位数 =====")
    for k, v in qtab.items():
        out("  %s α=%.2f  Pinball=%.4f  覆盖率=%.4f  MAE(log)=%.4f"
            % (k, v["alpha"], v["pinball_log"], v["coverage"], v["mae_log"]))
    out("\n[models] 池 %s 行 -> %s" % (format(len(pool), ","), POOL))
    out("[models] 总耗时 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
