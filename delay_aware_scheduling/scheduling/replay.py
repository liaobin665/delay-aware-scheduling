# -*- coding: utf-8 -*-

"""重放内核。

按给定顺序把作业依次喂给放置规则，放得下就占用、放不下就跳过，
并在过程中累计各项指标。
"""

import heapq
import time
import numpy as np
from delay_aware_scheduling.fleet import (
    Fleet, make_fleet, load_real_fleet, SCALE, LEVELS, SC_BINS, SC_NAMES,
)

from .policies import POL_LAM, order_of


BIG_IDX = (SC_NAMES.index("eq8"), SC_NAMES.index("ge16"))


N_BINS = len(SC_NAMES)


HOUR = 3600.0


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
