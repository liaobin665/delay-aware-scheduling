# -*- coding: utf-8 -*-
"""路径约定。

默认按下述布局找数据与落产物（相对仓库根）：

    data/   原始数据，自行下载解包（见 data/README.md）
    mid/    中间表，例如建模样本
    out/    结果产物，json / csv / parquet

三个目录都能用环境变量顶掉，方便把几十 GB 的原始表放到别的盘：

    DELAY_AWARE_SCHEDULING_DATA=/data/asi
    DELAY_AWARE_SCHEDULING_MID=/scratch/mid
    DELAY_AWARE_SCHEDULING_OUT=/scratch/out
"""
import os

_PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_PKG)

RAW = os.environ.get("DELAY_AWARE_SCHEDULING_DATA", os.path.join(ROOT, "data"))
MID = os.environ.get("DELAY_AWARE_SCHEDULING_MID", os.path.join(ROOT, "mid"))
OUT = os.environ.get("DELAY_AWARE_SCHEDULING_OUT", os.path.join(ROOT, "out"))

# --- 原始数据入口 ---------------------------------------------------------
# 逐作业执行摘要表（需求侧）
JOB = os.path.join(RAW, "job_execution_summary.parquet")
# 逐服务器小时表（供给侧），按 day=/hour=/part-*.parquet 组织
SERVER_HOURLY = os.path.join(RAW, "asi_opensource_server_hourly")

# --- 跨步骤复用的产物 -----------------------------------------------------
SAMPLE = os.path.join(MID, "model_sample.parquet")       # 抽样脚本产出的建模样本
CENSUS = os.path.join(OUT, "fleet_census.parquet")       # 供给侧机器清单
POOL = os.path.join(OUT, "delay_pool.parquet")           # 带折叠外预测的重放需求池
ADM = os.path.join(OUT, "admission_replay.csv")          # 准入重放的逐格结果
QUANTILE = os.path.join(OUT, "quantile.json")            # 分位数回归的损失与覆盖率

for _d in (MID, OUT):
    os.makedirs(_d, exist_ok=True)
