# -*- coding: utf-8 -*-
r"""全量口径核验与就绪延迟分布统计。

对整张作业表做一次流式扫描，给出四类读数：就绪状态与就绪延迟的取值组合各多少行；
建模队列（已就绪且记录了延迟）的延迟分位数、极值与直方图原始计数；长延迟作业对
总等待时间的集中程度；以及按机型 / 作业类型 / 模型类型拆开后的集中形态。

输出  out/delay_caliber.json   全部读数
      out/delay_hist.npz       直方图与集中度曲线的原始计数（供出图复用，不重扫数据）

用法  python -m delay_aware_scheduling.audit.exp60_caliber
"""
import os
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


from delay_aware_scheduling.config import JOB as RAW, OUT as PROD
from delay_aware_scheduling.common.log import out

OUT_JSON = os.path.join(PROD, "delay_caliber.json")
OUT_HIST = os.path.join(PROD, "delay_hist.npz")

CATS = ["gpu_spec_public", "job_type_public", "model_type_public"]
COLS = ["ready_status", "ready_delay_sec", "duration_hours"] + CATS
BATCH = 400_000
TAIL_SEC = 3600.0
HIST_LO, HIST_HI, HIST_N = -1.0, 5.0, 120


def main():
    t0 = time.time()
    f = pq.ParquetFile(RAW)
    out("[caliber] 行组 %d，总行 %s" % (f.metadata.num_row_groups, format(f.metadata.num_rows, ",")))

    n_total = n_ready = n_notready = n_null_ready = 0
    n_pos = n_neg_ready = n_neg_notready = 0
    neg_dur_sum = 0.0
    neg_dur_n = 0

    delay_parts = []
    # 分类累计器： cat -> level -> [count, wait_sum, tail_count, dur_sum]
    acc = {c: {} for c in CATS}

    for bi, batch in enumerate(f.iter_batches(batch_size=BATCH, columns=COLS)):
        df = batch.to_pandas()
        n = len(df)
        n_total += n

        rs = df["ready_status"]
        null_ready = rs.isna()
        rv = rs.fillna(False).astype(bool).to_numpy()
        n_null_ready += int(null_ready.sum())
        n_ready += int(rv.sum())
        n_notready += int((~rv).sum())

        rd = pd.to_numeric(df["ready_delay_sec"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        m_pos = rv & (rd >= 0)
        m_neg_ready = rv & (rd < 0)
        n_pos += int(m_pos.sum())
        n_neg_ready += int(m_neg_ready.sum())
        n_neg_notready += int(((~rv) & (rd < 0)).sum())

        if m_neg_ready.any():
            d = pd.to_numeric(df["duration_hours"], errors="coerce").to_numpy(dtype=np.float64)[m_neg_ready]
            d = d[np.isfinite(d)]
            neg_dur_sum += float(d.sum())
            neg_dur_n += int(d.size)

        if m_pos.any():
            dp = rd[m_pos]
            delay_parts.append(dp)
            sub = df.loc[m_pos, CATS].copy()
            sub["_d"] = dp
            sub["_t"] = (dp > TAIL_SEC)
            sub["_dur"] = pd.to_numeric(df.loc[m_pos, "duration_hours"], errors="coerce").fillna(0.0).to_numpy()
            for c in CATS:
                g = sub.groupby(c, dropna=False).agg(cnt=("_d", "size"), wsum=("_d", "sum"),
                                                     tail=("_t", "sum"), dsum=("_dur", "sum"))
                d_acc = acc[c]
                for lvl, row in g.iterrows():
                    key = "__NULL__" if (lvl is None or (isinstance(lvl, float) and np.isnan(lvl))) else str(lvl)
                    cur = d_acc.get(key)
                    if cur is None:
                        d_acc[key] = [int(row["cnt"]), float(row["wsum"]), int(row["tail"]), float(row["dsum"])]
                    else:
                        cur[0] += int(row["cnt"]); cur[1] += float(row["wsum"])
                        cur[2] += int(row["tail"]); cur[3] += float(row["dsum"])
        if bi % 25 == 0:
            out("  [batch %3d] 累计 %s 行，建模队列 %s 行 (%.0fs)"
                % (bi, format(n_total, ","), format(n_pos, ","), time.time() - t0))

    delay = np.concatenate(delay_parts)
    del delay_parts
    out("[caliber] 扫描完成 %.0fs；建模队列 %s 行" % (time.time() - t0, format(delay.size, ",")))

    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    qv = np.percentile(delay.astype(np.float64), qs)
    dist = dict(
        n=int(delay.size), mean=float(delay.mean()),
        q={str(q): float(v) for q, v in zip(qs, qv)},
        median=float(np.median(delay)), p90=float(qv[qs.index(90)]),
        p99=float(qv[qs.index(99)]), max=int(delay.max()),
        n_zero=int((delay == 0).sum()), n_le60=int((delay <= 60).sum()),
        n_gt3600=int((delay > TAIL_SEC).sum()), n_at_cap=int((delay >= 86399).sum()),
        share_le60=float((delay <= 60).mean()), share_gt3600=float((delay > TAIL_SEC).mean()),
    )

    tot_wait = float(delay.sum())
    dsort = np.sort(delay.astype(np.float64))[::-1]
    cum = np.cumsum(dsort)
    def share_top(frac):
        k = max(int(round(frac * delay.size)), 1)
        return float(cum[k - 1] / tot_wait)
    conc = dict(
        total_wait_sec=tot_wait,
        share_top_0p1pct=share_top(0.001), share_top_1pct=share_top(0.01),
        share_top_5pct=share_top(0.05), share_top_10pct=share_top(0.10),
        share_top_20pct=share_top(0.20), n_jobs=int(delay.size),
        share_wait_le60=float(delay[delay <= 60].sum() / tot_wait),
        share_n_le60=float((delay <= 60).mean()),
        mean_wait_all=tot_wait / delay.size,
    )

    lg = np.log10(delay.astype(np.float64) + 1.0)
    hist, edges = np.histogram(lg, bins=HIST_N, range=(HIST_LO, HIST_HI))
    # 集中度曲线（精确，非分箱估计）：作业按延迟降序，累计等待占比 vs 作业占比
    grid = np.linspace(0.0, 1.0, 501)
    kk = np.maximum((grid * delay.size).astype(np.int64), 1)
    curve_y = cum[np.minimum(kk, delay.size) - 1] / tot_wait
    np.savez_compressed(OUT_HIST, hist=hist.astype(np.int64), edges=edges.astype(np.float64),
                        q_keys=np.array(list(dist["q"].keys()), dtype=object),
                        q_vals=np.array(list(dist["q"].values()), dtype=np.float64),
                        n=np.array([delay.size], dtype=np.int64),
                        curve_x=grid, curve_y=curve_y)

    by_cat = {}
    for c in CATS:
        lv = acc[c]
        keys = sorted(lv, key=lambda k: -lv[k][0])
        by_cat[c] = {
            k: dict(count=lv[k][0], count_share=lv[k][0] / n_pos,
                    wait_share=lv[k][1] / tot_wait, mean_delay=lv[k][1] / max(lv[k][0], 1),
                    tail_rate=lv[k][2] / max(lv[k][0], 1),
                    mean_dur_h=lv[k][3] / max(lv[k][0], 1))
            for k in keys
        }

    meta = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), raw=RAW,
        rows_total=int(n_total), ready=int(n_ready), not_ready=int(n_notready),
        null_ready=int(n_null_ready), ready_delay_pos=int(n_pos),
        ready_delay_neg=int(n_neg_ready), notready_delay_neg=int(n_neg_notready),
        ready_delay_neg_share_of_ready=float(n_neg_ready / max(n_ready, 1)),
        ready_delay_neg_mean_dur_h=float(neg_dur_sum / max(neg_dur_n, 1)),
        ready_delay_neg_n_dur=int(neg_dur_n),
        dist=dist, concentration=conc, by_category=by_cat,
        hist_spec=dict(lo=HIST_LO, hi=HIST_HI, n=HIST_N), tail_sec=TAIL_SEC,
    )
    json.dump(meta, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    out("\n[caliber] ===== 口径 =====")
    out("  总行 %s ｜ 就绪 %s (%.3f%%) ｜ 未就绪 %s ｜ ready 缺失 %s"
        % (format(n_total, ","), format(n_ready, ","), 100.0 * n_ready / n_total,
           format(n_notready, ","), format(n_null_ready, ",")))
    out("  建模队列 %s ｜ ready=1 且 delay=-1 %s（占就绪 %.4f%%）｜ 该子群平均声明时长 %.1f h"
        % (format(n_pos, ","), format(n_neg_ready, ","),
           100.0 * n_neg_ready / max(n_ready, 1), meta["ready_delay_neg_mean_dur_h"]))
    out("\n[caliber] ===== 分布（秒）=====")
    out("  中位 %s ｜ P90 %s ｜ P99 %s ｜ 均值 %.1f ｜ 最大 %s"
        % (format(dist["median"], ",.0f"), format(dist["p90"], ",.0f"),
           format(dist["p99"], ",.0f"), dist["mean"], format(dist["max"], ",")))
    out("  <=60s 占 %.4f ｜ >3600s 占 %.4f (n=%s) ｜ =86399 上限处 %s"
        % (dist["share_le60"], dist["share_gt3600"], format(dist["n_gt3600"], ","), format(dist["n_at_cap"], ",")))
    out("\n[caliber] ===== 集中度 =====")
    out("  前 0.1%% 作业占总等待 %.3f ｜ 前 1%% %.3f ｜ 前 5%% %.3f ｜ 前 10%% %.3f"
        % (conc["share_top_0p1pct"], conc["share_top_1pct"], conc["share_top_5pct"], conc["share_top_10pct"]))
    out("  <=60s 的 %.4f 作业只贡献总等待 %.4f ｜ 全体均值等待 %.1f s"
        % (conc["share_n_le60"], conc["share_wait_le60"], conc["mean_wait_all"]))
    out("\n[caliber] 产物：\n  %s\n  %s" % (OUT_JSON, OUT_HIST))
    out("[caliber] 总耗时 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
