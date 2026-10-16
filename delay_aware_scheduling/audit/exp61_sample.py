# -*- coding: utf-8 -*-
r"""建模样本抽取（按机器分组抽样，保留就绪延迟列）。

按 server_id 做 5% 的【机器分组】抽样（crc32(server_id) % 20 == 0）：同一台机器的全部行
一起进或一起出。后续按机器分组切分时，训练集与测试集就不会含同一台机器，避免模型
靠记住某台机器的固有快慢拿到表面性能。

用 crc32 而非内置 hash()：后者逐进程随机化，会让"同一份数据跑两遍结果一致"失效。

输出  mid/model_sample.parquet

用法  python -m delay_aware_scheduling.audit.exp61_sample
"""
import os
import zlib
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


from delay_aware_scheduling.config import JOB as RAW, MID
from delay_aware_scheduling.common.log import out

OUT = os.path.join(MID, "model_sample.parquet")


COLS = ["server_id", "gpu_spec_public", "priority_class", "job_type_public",
        "model_type_public", "is_genai_request", "gpu_request", "duration_hours",
        "schedule_delay_sec", "ready_status", "ready_delay_sec"]
MOD = 20          # 每 20 台机器取 1 台 ⇒ 约 5% 机器
BATCH = 400_000
_CACHE = {}


def keep_sid(s):
    v = _CACHE.get(s)
    if v is None:
        v = (zlib.crc32(s.encode("utf-8")) % MOD) == 0
        _CACHE[s] = v
    return v


def main():
    t0 = time.time()
    f = pq.ParquetFile(RAW)
    parts = []
    n_total = n_keep = 0
    for bi, batch in enumerate(f.iter_batches(batch_size=BATCH, columns=COLS)):
        df = batch.to_pandas()
        n_total += len(df)
        ids = df["server_id"].astype(str)
        uniq = pd.unique(ids)
        kmap = {u: keep_sid(u) for u in uniq}
        mask = ids.map(kmap).to_numpy()
        if mask.any():
            parts.append(df.loc[mask])
            n_keep += int(mask.sum())
        if bi % 25 == 0:
            out("  [batch %3d] %s 行 → 抽样 %s 行 (%.0fs)"
                % (bi, format(n_total, ","), format(n_keep, ","), time.time() - t0))

    s = pd.concat(parts, ignore_index=True)
    del parts
    # 布尔列含缺失（全表 392,427 行四列同缺）⇒ 统一编码为 int8：1/0/-1（-1=缺失）
    s["ready_status"] = pd.to_numeric(s["ready_status"], errors="coerce").fillna(-1).astype("int8")
    s["is_genai_request"] = pd.to_numeric(s["is_genai_request"], errors="coerce").fillna(-1).astype("int8")
    s = s.reset_index(drop=True)
    s.to_parquet(OUT, index=False)

    rs = s["ready_status"].to_numpy()
    rd = pd.to_numeric(s["ready_delay_sec"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    cohort = (rs == 1) & (rd >= 0)
    n_kept_m = int(sum(1 for v in _CACHE.values() if v))
    out("\n[sample] 抽样机器 %s 台 / 全表 %s 台 (%.3f%%)"
        % (format(n_kept_m, ","), format(len(_CACHE), ","), 100.0 * n_kept_m / max(len(_CACHE), 1)))
    out("[sample] 样本 %s 行（占全量 %.3f%%）｜ 就绪 %s (%.3f%%)"
        % (format(len(s), ","), 100.0 * len(s) / n_total, format(int((rs == 1).sum()), ","),
           100.0 * (rs == 1).mean()))
    out("[sample] 建模队列(ready=1 且 delay>=0) %s 行 ｜ ready=1 且 delay=-1 %s 行"
        % (format(int(cohort.sum()), ","), format(int(((rs == 1) & (rd < 0)).sum()), ",")))
    out("[sample] 产物：%s  (%.0fs)" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
