# 运行顺序与产物依赖

## 依赖图

```
data/job_execution_summary.parquet ──┬─► exp60_caliber ──► out/delay_caliber.json
                                     │                    out/delay_hist.npz
                                     └─► exp61_sample ───► mid/model_sample.parquet
                                                                 │
data/asi_opensource_server_hourly ──► exp70_supply ──► out/fleet_census.parquet
                                                              │
mid/model_sample.parquet ──┬─► exp62_models ──► out/delay_pool.parquet
                           │                   out/models.json
                           │                   out/importance.json
                           │                   out/quantile.json
                           ├─► exp67_shap ───► out/shap_values.npz
                           ├─► exp68_embed ──► out/embed_sample.npz
                           └─► exp66_infer ──► out/infer.json

out/delay_pool.parquet + out/fleet_census.parquet
        └─► exp63_admission ──► out/admission_replay.csv
                                             │
out/admission_replay.csv ──┬─► exp64_stats ──► out/stats.json
                           │                  out/tables.json
                           │                  out/tables_preview.md
                           └─► exp65_extras ─► out/extras.json

out/delay_pool.parquet + out/shap_values.npz + out/embed_sample.npz
        └─► exp69_fignums ──► out/figure_readings.json
```

分步说明：

| 步骤 | 入口 | 读 | 写 |
|---|---|---|---|
| 1 | `structure.exp70_supply` | `data/asi_opensource_server_hourly` | `out/fleet_census.parquet`、`out/fleet_census.json` |
| 2 | `audit.exp60_caliber` | `data/job_execution_summary.parquet` | `out/delay_caliber.json`、`out/delay_hist.npz` |
| 3 | `audit.exp61_sample` | 同上 | `mid/model_sample.parquet` |
| 4 | `prediction.exp62_models` | `mid/model_sample.parquet` | `out/delay_pool.parquet`、`out/models.json`、`out/importance.json`、`out/quantile.json` |
| 5 | `analysis.exp67_shap` | `mid/model_sample.parquet` | `out/shap_values.npz` |
| 6 | `analysis.exp68_embed` | 同上 | `out/embed_sample.npz` |
| 7 | `analysis.exp66_infer` | 同上 | `out/infer.json` |
| 8 | `scheduling.exp63_admission` | `out/delay_pool.parquet`、`out/fleet_census.parquet` | `out/admission_replay.csv`、`out/admission_meta.json` |
| 9 | `scheduling.exp64_stats` | 步骤 8 的 csv、`out/models.json`、`out/quantile.json`、`out/delay_caliber.json`、`out/fleet_census.parquet` | `out/stats.json`、`out/tables.json`、`out/tables_preview.md` |
| 10 | `analysis.exp65_extras` | 步骤 8 的 csv、`out/delay_pool.parquet`、`out/quantile.json` | `out/extras.json` |
| 11 | `analysis.exp69_fignums` | `out/delay_pool.parquet`、`out/shap_values.npz`、`out/embed_sample.npz` | `out/figure_readings.json` |

步骤 5–7 与 11 只服务于出图与归因，不影响主结论；只想复现主结果的话，跑 1、2、3、4、8、9 即可。

---

## 准入重放：做了什么，假设了什么

### 重放的形式

作业表**没有时间戳**，到达时刻无法重建，因此不能做含时间轴的离散事件仿真。本仓库做的是
**静态准入重放**：给定一份机器快照和一个负载档，从重放需求池里抽一批作业，按某个准入顺序
把它们依次喂给放置规则；每一步只问"这台机器放得下吗"，放下就占用，放不下就跳过。

负载档 `rho` 定义为**提交作业的总 GPU 需求 ÷ 集群总 GPU 数**。需求抽样的权重按需求口径给：
`D_gpu` 按请求卡数加权，`D_row` 按条数等权。

### 主指标

| 指标 | 定义 | 方向 |
|---|---|---|
| `wait_mean_sec` | 已放置作业的**GPU 加权**平均就绪延迟：`Σ 卡 × 延迟 / Σ 卡` | 越低越好 |
| `eff_time_frac` | 有效计算时间占比 = `Σ 卡×时长 / Σ 卡×(时长 + 延迟/3600)` | 越高越好 |
| `big_cov` | 大请求（8 卡及以上）有效覆盖率 = 已放置大请求卡数 ÷ 可容纳大请求卡数 | 越高越好 |
| `alloc_rate` | 已放置作业 ÷ 可容纳作业 | 越高越好 |
| `util` | 已放置卡数 ÷ 集群总卡数 | — |
| `jain_place` | 各请求规模档放置率的 Jain 指数 | 越高越均衡 |

`eff_time_frac` 的分母按"每个已放置作业占用其卡位（**声明时长 + 就绪延迟**）"折算，
这正是"资源被占住但没产出计算"的那部分时间。这是静态重放引入的口径，替换成真实运行时长
会得到不同的绝对值。

### 准入顺序

对照含：先到先服务 `FCFS`、最短作业优先 `SJF`、最小请求优先 `SSF`、大请求优先 `BigFirst`、
随机排序 `Random`（预测器质量的下界）、零参数查表排序 `RuleDelay`、仅用提交时特征排序
`L0Delay`；本文方法 `DAS` 按**预测就绪延迟**升序；另有按**真实延迟**升序的 `DelayOracle`
作为该打分方向下的理论上限（不是全局最优）。

敏感性覆盖三个轴：**放置规则**（最佳适配 `BF` / 首次适配 `FF` / 最差适配 `WF`）、
**负载档**（5 档）、**打分来源**（上面几个排序器）。分位数打分 `DASq50/q90/q99` 与
复合打分（见下）只在最佳适配下跑。

### 复合打分

为了在"大请求保障"与"各档均衡"之间给出可调的取舍，定义

```
s_j = d̂_j − λ · r_j / r̄
```

`d̂_j` 为预测对数延迟，`r_j` 为请求卡数，`r̄` 为批内请求卡数均值。`λ` 越大越偏向大请求；
`λ = 0` 即本文方法本身。`λ` 取值见 `exp63_admission` 的 `PARE_LAMBDAS`。

### 配对检验

同一格（集群 × 种子 × 需求口径 × 负载 × 放置规则）上的各策略共用同一批作业，因此策略间
用**配对差**比较。`exp64_stats` 给出配对 t 检验的 95% 置信区间与双侧 p 值，并以 Wilcoxon
符号秩检验作为分布无关的旁证。

---

## 已知假设与边界

- 重放假设**就绪结局不随准入顺序改变**：同一作业换一个顺序，它的就绪延迟仍取原值。
  真实集群里镜像缓存冷热、同机并发启动竞争都会让这个假设不严格成立。
- 机架字段只在观测期后段可见；`fleet_census.parquet` 取参照日快照，不反映机器进出。
- 合成集群的形态档位（`fleet/constants.py` 的 `LEVELS`）只用来覆盖"异质度—规模"的不同组合，
  不声称与任何具体生产集群一一对应。
