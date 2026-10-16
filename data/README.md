# 原始数据

**本仓库不附带任何数据文件**，请按下面的说明自行下载。

## 来源

数据来自 Alibaba Cluster Trace Program 公开发布的 GPU 集群轨迹
`cluster-trace-gpu-v2026`（Alibaba Serverless Infrastructure，ASI）。
请到该项目的官方发布页面获取下载方式与最新的许可条款；本仓库只引用字段口径与统计结果，
不复制、不转发。

使用时请遵守数据发布方给出的许可与引用要求。

## 需要哪两张表

解包后，把下面两项放到本目录（或 `DELAY_AWARE_SCHEDULING_DATA` 指向的目录）：

| 文件 / 目录 | 内容 | 用途 |
|---|---|---|
| `job_execution_summary.parquet` | 逐作业执行摘要表（需求侧），单文件多 row group | 口径核验、建模样本、就绪延迟预测、准入重放 |
| `asi_opensource_server_hourly/` | 逐服务器小时表（供给侧），按 `day=/hour=/part-*.parquet` 组织 | 参照日的机器清单（卡数、机型、机架） |

本仓库不需要网络小时表。

## 目录要求

脚本只按固定文件名找数据，所以解包后请保持：

```
data/
├── job_execution_summary.parquet
└── asi_opensource_server_hourly/
    └── day=*/hour=*/part-*.parquet
```

若放在别处，设环境变量 `DELAY_AWARE_SCHEDULING_DATA=/path/to/unpacked` 即可（见
`delay_aware_scheduling/config.py`）。

## 字段口径

每个字段的含义、哪些字段不能进特征矩阵，见 [../docs/schema.md](../docs/schema.md)。
