# -*- coding: utf-8 -*-
r"""供给侧机器清单还原。

从逐服务器小时表还原参照时刻的机器清单，并补上只在其他日出现过的机器，得到覆盖作业表
里全部 server_id 的并集清单：每台机器的机型、单机卡数、机架标识，以及它是否在参照日
出现。准入重放实验里的「实测集群」档直接读这份清单。

输入  data/asi_opensource_server_hourly/day=*/hour=*/part-*.parquet
输出  out/fleet_census.parquet   机器清单（含 in_ref_day 标记）
      out/fleet_census.json      读数汇总

用法  python -m delay_aware_scheduling.structure.exp70_supply [--ref-day 184] [--ref-hour 12]
"""
import argparse
import glob
import json
import os
import time

import pandas as pd

from delay_aware_scheduling.common.log import out
from delay_aware_scheduling.config import CENSUS, OUT as PROD, SERVER_HOURLY as SRV

COLS = ["server_id", "gpu_spec_public", "gpu_count", "asw_id"]
INFO = os.path.join(PROD, "fleet_census.json")


def part_path(day, hour):
    return os.path.join(SRV, "day=%d" % day, "hour=%02d" % hour, "part-000.parquet")


def day_of(p):
    return int(os.path.basename(p).split("=")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-day", type=int, default=184)
    ap.add_argument("--ref-hour", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()

    days = sorted(day_of(p) for p in glob.glob(os.path.join(SRV, "day=*")))
    if not days:
        raise SystemExit("[supply] 在 %s 下没找到 day=* 分区" % SRV)
    out("[supply] 覆盖 %d 天（day=%d..%d）" % (len(days), days[0], days[-1]))

    # 逐日读参照小时，记录每台机器出现过哪些天
    base, pres = None, {}
    for d in days:
        p = part_path(d, a.ref_hour)
        if not os.path.exists(p):
            out("[supply] 警告：缺少 %s" % p)
            continue
        t = pd.read_parquet(p, columns=COLS)
        for sid in t["server_id"].tolist():
            pres.setdefault(sid, []).append(d)
        if d == a.ref_day:
            base = t
    if base is None:
        raise SystemExit("[supply] 参照日 %d 的 hour=%d 分片不存在" % (a.ref_day, a.ref_hour))
    base = base.copy()

    # 只在其他日出现的机器，取它们最后一次出现的那天补齐
    have = set(base["server_id"])
    need = sorted(set(pres) - have)
    out("[supply] 参照时刻 %s 台；另有 %s 台只在其他日出现"
        % (format(len(base), ","), format(len(need), ",")))
    last_day = {s: max(v) for s, v in pres.items()}
    extra, taken = [], set(have)
    for d in sorted({last_day[s] for s in need}):
        todo = [s for s in need if s not in taken]
        if not todo:
            break
        p = part_path(d, a.ref_hour)
        if not os.path.exists(p):
            continue
        t = pd.read_parquet(p, columns=COLS)
        t = t[t["server_id"].isin(set(todo))]
        if len(t):
            extra.append(t)
            taken |= set(t["server_id"])

    uni = pd.concat([base] + extra, ignore_index=True) if extra else base.copy()
    uni = uni.drop_duplicates(subset=["server_id"], keep="last")
    uni["present_days"] = uni["server_id"].map(lambda s: len(pres[s])).astype("int64")
    uni["in_ref_day"] = uni["server_id"].isin(have)
    uni.to_parquet(CENSUS, index=False)

    info = dict(
        n_days=len(days), day_range=[int(days[0]), int(days[-1])],
        ref=dict(day=a.ref_day, hour=a.ref_hour),
        n_machine_ref=int(len(base)), n_machine_union=int(len(uni)),
        n_filled=int(len(uni) - len(base)),
        cards_ref=int(base["gpu_count"].sum()), cards_union=int(uni["gpu_count"].sum()),
        asw_nonnull_ref=int(base["asw_id"].notna().sum()),
        elapsed_sec=round(time.time() - t0, 1))
    with open(INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    out("[supply] 参照日 %s 台 / %s 卡；并集 %s 台 -> %s（%.0fs）"
        % (format(len(base), ","), format(info["cards_ref"], ","),
           format(len(uni), ","), os.path.basename(CENSUS), time.time() - t0))


if __name__ == "__main__":
    main()
