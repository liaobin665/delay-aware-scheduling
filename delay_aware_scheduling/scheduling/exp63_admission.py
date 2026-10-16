# -*- coding: utf-8 -*-

r"""延迟感知调度的准入重放实验。

只改【准入顺序】，放置规则固定。对照含先到先服务、最短作业优先、最小请求优先、
大请求优先、随机排序、零参数查表排序、仅提交时特征排序，以及按真实延迟排序的
理论上限；本文方法按【预测就绪延迟】升序。

指标：已放置作业按请求 GPU 数加权的平均就绪延迟、有效计算时间占比、大请求（8 卡
及以上）有效覆盖率、已放置比例、GPU 占用率、各请求规模档的放置公平性。

重放是静态准入重放而非含时间轴的离散事件仿真：作业表没有时间戳，到达时刻无法重建。
因此"有效计算时间占比"按"每个已放置作业占用其卡位（声明时长 + 就绪延迟）"折算，
该假设在 docs/pipeline.md 中写明，并做了放置规则与负载的敏感性。

输出  out/admission_replay.csv   逐格结果
      out/admission_meta.json   本次运行的档位与策略清单

用法  python -m delay_aware_scheduling.scheduling.exp63_admission
      [--smoke] [--seeds 3] [--placements BF,FF,WF] [--demands D_gpu]
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from delay_aware_scheduling.common.log import out
from delay_aware_scheduling.common.seeds import SEED_REPLAY as SEEDS
from delay_aware_scheduling.config import OUT as PROD, POOL
from delay_aware_scheduling.fleet import (
    Fleet, make_fleet, load_real_fleet, SCALE, LEVELS, SC_BINS, SC_NAMES,
)

from .policies import EXT_POL, MAIN_POL, PARE_LAMBDAS, PARE_POL, PLACEMENTS, POL_LAM
from .replay import run_replay, scenario_cum


RHO = [1.00, 1.25, 1.50, 2.00, 2.50]


FLEET_SET = [("H2", 4096), ("H4", 4096), ("REAL", 0)]


DEMANDS = ["D_gpu", "D_row"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--placements", default=",".join(PLACEMENTS))
    ap.add_argument("--demands", default="D_gpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    places = args.placements.split(",")
    demands = args.demands.split(",")
    rhos = [1.00, 2.50] if args.smoke else RHO
    t0 = time.time()

    pool = pd.read_parquet(POOL)
    out("[adm] 池 %s 行 ｜ 延迟中位 %.0fs ｜ 平均请求 %.3f 卡"
        % (format(len(pool), ","), np.median(pool["delay_sec"]), pool["req"].mean()))
    sc_arr = {sc: scenario_cum(pool, sc) for sc in demands}

    fleets = {}
    for lv, N in FLEET_SET:
        if lv == "REAL":
            fleets[lv] = load_real_fleet()
        else:
            cfg = LEVELS[lv]
            fleets[lv] = make_fleet(N, cfg["mix"], cfg["rack_nodes"], cfg["purity"], seeds[0])
    for lv in fleets:
        out("[adm] 集群 %-4s %s 台 / %s 卡 / 最大单机 %d 卡"
            % (lv, format(fleets[lv]["n_machine"], ","),
               format(fleets[lv]["cap_total"] // SCALE, ","), fleets[lv]["maxcap"] // SCALE))

    req_all = np.maximum(np.rint(pool["req"].to_numpy(dtype=np.float64) * SCALE).astype(np.int64), 1)
    dur_all = pool["dur"].to_numpy(dtype=np.float64)
    delay_all = pool["delay_sec"].to_numpy(dtype=np.float64)
    S = {k: pool["pred_" + k].to_numpy(dtype=np.float64)
         for k in ["lgb", "q50", "q90", "q99", "rule", "l0"]}
    S["rand"] = pool["rand"].to_numpy(dtype=np.float64)
    rows = []

    def run_cell(lv, sd, sc, rho, place, policy):
        fl = fleets[lv]
        n_jobs = int(round(rho * (fl["cap_total"] / SCALE) / sc_arr[sc]["Er"]))
        rng = np.random.default_rng((sd * 1000003 + int(rho * 1000) * 7919
                                     + DEMANDS.index(sc) * 104729) % (2 ** 63))
        idx = np.searchsorted(sc_arr[sc]["cum"], rng.random(n_jobs), side="left")
        sc_scores = {k: v[idx] for k, v in S.items()}
        res = run_replay(fl["cap"], req_all[idx], dur_all[idx], delay_all[idx], policy, place,
                         sc_scores, lam=POL_LAM.get(policy, 0.0))
        rows.append(dict(fleet=lv, seed=sd, demand=sc, rho=rho, place=place, policy=policy,
                         n_jobs=n_jobs, cap_cards=int(fl["cap_total"] // SCALE),
                         **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}))

    for lv, _N in FLEET_SET:
        t1 = time.time()
        for sc in demands:
            for rho in rhos:
                for sd in seeds:
                    for place in places:
                        pols = list(MAIN_POL) + (list(EXT_POL) + list(PARE_POL) if place == "BF" else [])
                        for pol in pols:
                            run_cell(lv, sd, sc, rho, place, pol)
        out("  [%s] 完成 %d 格 (%.0fs)" % (lv, len(rows), time.time() - t1))

    df = pd.DataFrame(rows)
    csv = os.path.join(PROD, "admission_replay%s.csv" % args.tag)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    METRICS = ["wait_mean_sec", "p99_wait_sec", "w_occ_gpuh", "cap1_rate", "eff_time_frac",
               "big_cov", "alloc_rate", "placeable_rate", "util", "jain_place", "frag_rate"]
    bf = df[(df.place == "BF") & (df.demand == demands[0])]
    out("\n[adm] 主口径（落点=BF，需求=%s）各策略均值" % demands[0])
    out(bf.groupby("policy")[METRICS].mean().round(4).to_string())
    json.dump(dict(seeds=[int(s) for s in seeds], rho=rhos, placements=places, demands=demands,
                   fleets=[lv for lv, _ in FLEET_SET], main_pol=MAIN_POL, ext_pol=EXT_POL,
                   pare_pol=PARE_POL, pare_lambdas=PARE_LAMBDAS, n_rows=int(len(df)),
                   smoke=bool(args.smoke),
                   note="静态准入重放；落点 BF/FF/WF 互不共用分支；全部作业为就绪作业；"
                        "Compi 为复合打分 s=d̂−λ·(r/̄r)，λ 见 pare_lambdas"),
              open(os.path.join(PROD, "admission_meta%s.json" % args.tag), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    out("[adm] 逐格 %d 行 -> %s (%.0fs)" % (len(df), os.path.basename(csv), time.time() - t0))


if __name__ == "__main__":
    main()
