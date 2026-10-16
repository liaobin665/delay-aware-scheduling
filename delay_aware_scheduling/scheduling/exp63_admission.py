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
import os
import json
import time
import heapq
import argparse

import numpy as np
import pandas as pd


from delay_aware_scheduling.config import OUT as PROD, POOL
from delay_aware_scheduling.fleet import (
    Fleet, make_fleet, load_real_fleet, SCALE, LEVELS, SC_BINS, SC_NAMES,
)

from delay_aware_scheduling.common.seeds import SEED_REPLAY as SEEDS
from delay_aware_scheduling.common.log import out

RHO = [1.00, 1.25, 1.50, 2.00, 2.50]
FLEET_SET = [("H2", 4096), ("H4", 4096), ("REAL", 0)]
DEMANDS = ["D_gpu", "D_row"]
MAIN_POL = ["FCFS", "SJF", "SSF", "BigFirst", "Random", "RuleDelay", "L0Delay",
            "DAS", "DelayOracle"]
EXT_POL = ["DASq50", "DASq90", "DASq99"]
# 复合打分（公平性-效率权衡）：s = d̂ − λ·(r/̄r)，λ 越大越偏向大请求
PARE_LAMBDAS = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
PARE_POL = ["Comp%d" % i for i in range(len(PARE_LAMBDAS))]
POL_LAM = {p: PARE_LAMBDAS[i] for i, p in enumerate(PARE_POL)}
PLACEMENTS = ["BF", "FF", "WF"]
BIG_IDX = (SC_NAMES.index("eq8"), SC_NAMES.index("ge16"))
N_BINS = len(SC_NAMES)
HOUR = 3600.0


def order_of(policy, req, dur, p_lgb, p_q50, p_q90, p_q99, p_rule, p_l0, delay, rand):
    n = len(req)
    ar = np.arange(n, dtype=np.int64)
    if policy == "FCFS":
        return ar
    if policy == "SJF":
        return np.lexsort((ar, dur))
    if policy == "SSF":
        return np.lexsort((ar, req))
    if policy == "BigFirst":
        return np.lexsort((ar, -req))
    if policy == "Random":
        return np.lexsort((ar, rand))
    if policy == "DAS":
        return np.lexsort((ar, p_lgb))
    if policy == "DelayOracle":
        return np.lexsort((ar, delay))
    if policy == "DASq50":
        return np.lexsort((ar, p_q50))
    if policy == "DASq90":
        return np.lexsort((ar, p_q90))
    if policy == "DASq99":
        return np.lexsort((ar, p_q99))
    if policy == "RuleDelay":
        return np.lexsort((ar, p_rule))
    if policy == "L0Delay":
        return np.lexsort((ar, p_l0))
    raise ValueError(policy)


def run_replay(cap, req, dur, delay, policy, place, scores, lam=0.0):
    """准入顺序由 policy 决定；落点由 place 决定。所有作业均为就绪作业（队列已过滤）。"""
    n = len(req)
    f = Fleet(cap)
    maxcap = f.maxcap
    feas = req <= maxcap
    n_feas = int(feas.sum())
    if policy in POL_LAM:
        # 复合打分：预测对数延迟 − λ ×（请求量 / 批内均值）
        rn = req / max(float(req.mean()), 1e-9)
        order = np.lexsort((np.arange(n, dtype=np.int64), scores["lgb"] - lam * rn))
    else:
        order = order_of(policy, req, dur, scores["lgb"], scores["q50"], scores["q90"],
                         scores["q99"], scores["rule"], scores["l0"], delay, scores["rand"])
    # 长延迟捕获率：真实延迟在批内前 1% 的作业，有多少被排在准入序列的后半段（只看顺序）
    thr = float(np.quantile(delay, 0.99))
    top1 = delay >= thr
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n, dtype=np.int64)
    cap1 = float(np.mean(pos[top1] >= n / 2.0)) if top1.any() else None

    use_wf = (place == "WF")
    maxh = None
    if use_wf:
        maxh = [(-int(v), i) for i, v in enumerate(f.free)]
        heapq.heapify(maxh)

    cls = np.digitize(req, SC_BINS)
    sub_s = [0] * N_BINS; plc_s = [0] * N_BINS
    placed = placed_cards = 0
    placed_delays = []
    big_feas_cards = big_placed_cards = 0
    wait_num = eff_num = occ_num = 0.0
    t0 = time.perf_counter()
    for k in order:
        r = int(req[k])
        c = cls[k]
        if r > maxcap:
            continue
        sub_s[c] += 1
        if c in BIG_IDX:
            big_feas_cards += r
        if place == "FF":
            i = f.seg.first_ge(r)
        elif place == "BF":
            cc = f.idx.smallest_ge(r)
            i = -1 if cc == -1 else f.idx.any_of(cc)
        elif place == "WF":
            i = -1
            while maxh:
                nf, j = maxh[0]
                nf = -nf
                if f.free[j] != nf:
                    heapq.heappop(maxh)
                    continue
                if nf >= r:
                    i = j
                break
        else:
            raise ValueError(place)
        if i < 0:
            continue
        f.take(i, r)
        if use_wf:
            heapq.heappush(maxh, (-int(f.free[i]), i))
        placed += 1
        placed_cards += r
        plc_s[c] += 1
        if c in BIG_IDX:
            big_placed_cards += r
        d = float(delay[k])
        du = float(dur[k])
        occ = du + d / HOUR
        wait_num += r * d
        eff_num += r * du
        occ_num += r * occ
        placed_delays.append(d)
    dt = time.perf_counter() - t0

    free = f.free
    mm = req[feas].min() if n_feas else None
    frag = int(free[(free > 0) & (free < mm)].sum()) if mm is not None else int(free.sum())
    rr = [plc_s[i] / sub_s[i] for i in range(N_BINS) if sub_s[i] > 0]
    a = np.array(rr, dtype=np.float64)
    jain = float((a.sum() ** 2) / (len(a) * (a ** 2).sum())) if len(a) else None

    res = dict(
        submitted=n, submitted_feas=n_feas, placed=placed, placed_cards=placed_cards,
        placeable_rate=n_feas / max(n, 1),
        alloc_rate=placed / max(n_feas, 1),
        util=placed_cards / float(cap.sum()),
        wait_mean_sec=wait_num / max(placed_cards, 1),
        w_occ_gpuh=wait_num / SCALE / HOUR,
        p99_wait_sec=float(np.quantile(placed_delays, 0.99)) if placed_delays else None,
        cap1_rate=cap1,
        wait_mean_row=float(np.mean(delay[order[:placed]])) if placed else None,
        eff_time_frac=eff_num / occ_num if occ_num > 0 else None,
        occ_hours_total=occ_num,
        big_cov=big_placed_cards / max(big_feas_cards, 1),
        big_feas_cards=big_feas_cards, big_placed_cards=big_placed_cards,
        frag_rate=frag / float(cap.sum()),
        jain_place=round(jain, 6) if jain is not None else None,
        us_per_decision=round(dt / max(n, 1) * 1e6, 3),
    )
    for nm, s, p in zip(SC_NAMES, sub_s, plc_s):
        res["rate_" + nm] = round(p / s, 6) if s else None
        res["nsub_" + nm] = s
    return res


def scenario_cum(pool, sc):
    r = pool["req"].to_numpy(dtype=np.float64)
    w = np.ones(len(r)) if sc == "D_row" else r.copy()
    tot = float(w.sum())
    return dict(cum=np.cumsum(w) / tot, Er=float((r * (w / tot)).sum()))


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
