# -*- coding: utf-8 -*-
"""统一的控制台输出。

Windows 控制台默认是 GBK，中文日志会抛 UnicodeEncodeError。这里在导入时把标准输出切到
UTF-8，并给出一个 out()——各脚本都用它，不再各自包装。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def out(*a):
    print(*a, flush=True)
