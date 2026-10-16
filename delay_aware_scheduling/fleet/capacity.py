# -*- coding: utf-8 -*-
"""容量结构与放置。

三种放置规则各有一个索引结构：

    MaxSeg   下标最小且空闲 >= r 的机器      —— 首次适配
    CapIdx   空闲卡数最小且 >= r 的机器      —— 最佳适配
    Fleet    上面两者的组合，另配 WF 用的最大堆（最差适配）

容量一律用 centi-GPU 整数表示（见 constants.SCALE），因此装箱过程完全确定、可复现。
"""
import numpy as np

from delay_aware_scheduling.fleet.constants import INF


class MaxSeg:
    """按机器下标的区间最大空闲；first_ge(r) 取下标最小且空闲 >= r 的机器。"""

    __slots__ = ("size", "mx", "n")

    def __init__(self, arr):
        n = len(arr)
        size = 1
        while size < n:
            size <<= 1
        self.size, self.n = size, n
        mx = [-1] * (2 * size)
        for i, v in enumerate(arr):
            mx[size + i] = int(v)
        for i in range(size - 1, 0, -1):
            a, b = mx[2 * i], mx[2 * i + 1]
            mx[i] = a if a > b else b
        self.mx = mx

    def set(self, i, v):
        i += self.size
        self.mx[i] = int(v)
        i >>= 1
        mx = self.mx
        while i:
            a, b = mx[2 * i], mx[2 * i + 1]
            mx[i] = a if a > b else b
            i >>= 1

    def first_ge(self, r):
        if self.mx[1] < r:
            return -1
        i = 1
        mx, size = self.mx, self.size
        while i < size:
            i <<= 1
            if mx[i] < r:
                i += 1
        j = i - size
        return j if j < self.n else -1


class CapTree:
    """按容量值索引的机器计数：first_ge(c) 取最小可用容量 >= c 的那一档。"""

    __slots__ = ("size", "n", "mn", "cnt")

    def __init__(self, maxcap):
        self.n = maxcap + 1
        size = 1
        while size < self.n:
            size <<= 1
        self.size = size
        self.mn = [INF] * (2 * size)
        self.cnt = [0] * (2 * size)

    def update(self, cap, mnval, cntval):
        i = cap + self.size
        mn, cnt = self.mn, self.cnt
        mn[i] = mnval
        cnt[i] = cntval
        i >>= 1
        while i:
            l = i << 1
            r = l | 1
            a, b = mn[l], mn[r]
            mn[i] = a if a < b else b
            cnt[i] = cnt[l] + cnt[r]
            i >>= 1

    def range_min(self, l, r):
        if l >= r:
            return INF
        l += self.size
        r += self.size
        res, mn = INF, self.mn
        while l < r:
            if l & 1:
                if mn[l] < res:
                    res = mn[l]
                l += 1
            if r & 1:
                r -= 1
                if mn[r] < res:
                    res = mn[r]
            l >>= 1
            r >>= 1
        return res

    def first_ge(self, l):
        if l >= self.n:
            return -1
        size, cnt = self.size, self.cnt
        i = l + size
        while True:
            while (i & 1) == 0:
                i >>= 1
            if cnt[i] != 0:
                while i < size:
                    i <<= 1
                    if cnt[i] == 0:
                        i += 1
                r = i - size
                return r if r < self.n else -1
            i += 1
            if (i & -i) == i:
                return -1


class CapIdx:
    """按空闲卡数分桶：smallest_ge(r) 给出最小可用容量 >= r 的档位。"""

    __slots__ = ("buckets", "tree", "maxcap")

    def __init__(self, cap):
        self.maxcap = int(cap.max())
        self.tree = CapTree(self.maxcap)
        self.buckets = {}
        for i, c in enumerate(cap):
            self.buckets.setdefault(int(c), set()).add(int(i))
        for c, s in self.buckets.items():
            self.tree.update(c, 0, len(s))

    def _upd(self, c):
        s = self.buckets.get(c)
        self.tree.update(c, 0, len(s) if s else 0)

    def add(self, c, i):
        self.buckets.setdefault(c, set()).add(i)
        self._upd(c)

    def remove(self, c, i):
        s = self.buckets[c]
        s.discard(i)
        if not s:
            del self.buckets[c]
        self._upd(c)

    def smallest_ge(self, r):
        if r > self.maxcap:
            return -1
        return self.tree.first_ge(r)

    def has(self, c):
        return c in self.buckets

    def any_of(self, c):
        return next(iter(self.buckets[c]))


class Fleet:
    """集群状态。取用一台机器上的卡后，三个索引结构同步更新。"""

    __slots__ = ("free", "seg", "idx", "cap", "capset", "maxcap", "used")

    def __init__(self, cap):
        self.cap = cap
        self.free = cap.copy()
        self.maxcap = int(cap.max())
        self.seg = MaxSeg(self.free)
        self.idx = CapIdx(self.free)
        self.capset = set(int(x) for x in np.unique(cap).tolist())
        self.used = 0

    def take(self, i, r):
        c0 = int(self.free[i])
        c1 = c0 - r
        self.free[i] = c1
        self.seg.set(i, c1)
        self.idx.remove(c0, i)
        if c1 > 0:
            self.idx.add(c1, i)
        self.used += r
