# 异构 GPU 集群中作业就绪延迟的预测与延迟感知准入

这套代码做两件事：

1. **测量"就绪延迟"**——GPU 已经被分配给作业，但作业还没有进入就绪状态，于是这块卡被占住却不产出计算。
   代码从生产集群的公开轨迹里把这段延迟量出来，并分析它的分布、尾部与集中形态。
2. **做延迟感知调度（Delay-Aware Scheduling，DAS）**——只用准入时刻可观测的属性预测就绪延迟，
   在**放置规则不变**的前提下，把准入顺序改成按预测延迟升序；并在历史作业上重放验证这样做的效果。

代码按分析阶段分包，每个阶段有独立的入口脚本，产物写进 `out/`，下一步直接读上一步的产物。

> 仓库**不含任何数据**。原始轨迹需要自行下载，见 [data/README.md](data/README.md)。

---

## 目录结构

```
.
├── delay_aware_scheduling/      主包
│   ├── config.py                路径约定（data / mid / out，可用环境变量覆盖）
│   ├── common/                  跨步骤共用的小工具（日志、随机种子）
│   ├── audit/                   口径核验、分布统计与建模样本
│   ├── prediction/              就绪延迟的点估计与分位数预测
│   ├── analysis/                预测误差、归因与补充分析
│   ├── structure/               供给侧机器清单还原
│   ├── fleet/                   装箱引擎与合成 / 实测集群
│   └── scheduling/              准入重放实验与统计
├── data/                        原始数据放这里（不随仓库发布）
├── mid/                         中间表（建模样本等）
├── out/                         结果产物（json / csv / parquet）
├── docs/                        字段口径与运行顺序
├── tests/                       不需要数据的冒烟测试
├── requirements.txt
└── pyproject.toml
```

三个数据目录的分工：`data/` 只读，放下载解包后的原始表；`mid/` 放跨步骤复用的中间表；
`out/` 放每一步的结果。换机器时把大文件放别处即可：

```bash
export DELAY_AWARE_SCHEDULING_DATA=/data/asi      # 原始数据
export DELAY_AWARE_SCHEDULING_MID=/scratch/mid    # 中间表
export DELAY_AWARE_SCHEDULING_OUT=/scratch/out    # 产物
```

---

## 环境

Python ≥ 3.10。依赖见 `requirements.txt`；也可以直接装成包：

```bash
pip install -r requirements.txt
# 或
pip install -e ".[dev]"
```

---

## 运行顺序

按下面的顺序执行。括号里是入口脚本，全部用 `python -m` 调用（包内是相对导入）。

```bash
# 1. 供给侧机器清单（准入重放的"实测集群"档要用）
python -m delay_aware_scheduling.structure.exp70_supply

# 2. 口径核验与建模样本
python -m delay_aware_scheduling.audit.exp60_caliber
python -m delay_aware_scheduling.audit.exp61_sample

# 3. 就绪延迟预测
python -m delay_aware_scheduling.prediction.exp62_models

# 4. 逐行 SHAP 与降维样本（供归因分析，可选）
python -m delay_aware_scheduling.analysis.exp67_shap
python -m delay_aware_scheduling.analysis.exp68_embed

# 5. 准入重放与统计
python -m delay_aware_scheduling.scheduling.exp63_admission
python -m delay_aware_scheduling.scheduling.exp64_stats

# 6. 补充分析
python -m delay_aware_scheduling.analysis.exp65_extras
python -m delay_aware_scheduling.analysis.exp66_infer
python -m delay_aware_scheduling.analysis.exp69_fignums
```

`exp70_supply` 与 `exp63_admission` 是全部脚本里最重的两步（前者扫 185 天的服务器小时表，
后者在每个 (集群 × 负载 × 种子 × 放置规则) 格上重放一遍）。想要快速试跑，
`exp63_admission` 支持 `--smoke`（只跑两个负载档）；`exp62_models` 支持 `--quick`（只抽样 20 万条）。

---

## 每个文件做什么

### `delay_aware_scheduling/`（顶层）

| 文件 | 作用 |
|---|---|
| `config.py` | 路径约定。解析 `data/`、`mid/`、`out/` 与几个常用文件路径，支持环境变量覆盖。 |

### `common/` 共用工具

| 文件 | 作用 |
|---|---|
| `log.py` | 统一的 `out()` 输出（顺带把标准输出切到 UTF-8）。 |
| `seeds.py` | 抽样、切分、随机排序键用的固定种子。 |

### `audit/` 口径与样本

| 文件 | 作用 |
|---|---|
| `exp60_caliber.py` | 全量口径核验：就绪状态与就绪延迟的取值组合、延迟分位数与极值、集中度、按机型 / 作业类型 / 模型类型的拆分。 |
| `exp61_sample.py` | 按机器分组抽 5% 的建模样本（同一台机器的全部行一起进或一起出）。 |

### `prediction/` 就绪延迟预测

| 文件 | 作用 |
|---|---|
| `exp62_models.py` | 入口：九项准入时刻属性的回归与分位数回归（P50/P90/P99），按机器分组五折，全部指标在折叠外预测上汇总；另含仅用 L0 特征、零参数查表两组对照。 |
| `features.py` | 特征构造（L0 / L1 两层）与按机器分组的折分配。 |
| `metrics.py` | 误差度量：对数尺度与秒尺度，排序相关性用 Spearman 与 Kendall。 |

### `analysis/` 补充分析

| 文件 | 作用 |
|---|---|
| `exp65_extras.py` | 预测误差（尤其是长延迟子群的低估）、分位数校准、公平性-效率权衡曲线、预测质量与调度收益的对照。 |
| `exp66_infer.py` | 预测器推理开销实测：三种调用口径的单次耗时、模型规模、每秒可处理作业数。 |
| `exp67_shap.py` | 逐行 SHAP 值（供蜂群图与依赖图）。 |
| `exp68_embed.py` | 归因空间的降维与分群着色所需的样本（标准化属性矩阵、真实对数延迟、分组标签）。 |
| `exp69_fignums.py` | 图件引用的少量确定性读数（分群中位延迟、平均 SHAP、主成分方差解释比例）。 |

### `structure/` 供给侧

| 文件 | 作用 |
|---|---|
| `exp70_supply.py` | 从逐服务器小时表还原参照日的机器清单，并补上只在其他日出现的机器，得到覆盖全部 `server_id` 的并集清单。 |

### `fleet/` 装箱引擎

| 文件 | 作用 |
|---|---|
| `constants.py` | 容量刻度（1 张卡 = 100 个整数单位）、请求规模分档、集群形态档位。 |
| `capacity.py` | 三种放置规则要用的索引结构：`MaxSeg`（首次适配）、`CapIdx`（最佳适配）、`Fleet`（集群状态）。 |
| `fleetgen.py` | 参数化合成集群生成，以及引用实测机器清单。 |

### `scheduling/` 准入重放

| 文件 | 作用 |
|---|---|
| `exp63_admission.py` | 入口：重放实验，换准入顺序、换放置规则、换负载，看平均就绪延迟与有效计算时间占比怎么变。 |
| `policies.py` | 准入排序规则（先到先服务 / 最短作业优先 / 本文方法按预测延迟升序 / …）与复合打分的权重档。 |
| `replay.py` | 重放内核：按给定顺序把作业喂给放置规则，并累计各项指标。 |
| `exp64_stats.py` | 入口：重放结果的统计量与结果表。 |
| `paired.py` | 重放结果的读取与配对差检验（配对 t 区间 + Wilcoxon 旁证）。 |
| `tables.py` | 结果表的组装（表体由脚本生成，数字不手写）。 |

---

## 测试

只测不需要数据的部分（容量索引、集群生成、指标公式、折叠分配与排序规则）：

```bash
pytest
```

---

## 随机种子

抽样 `20261011`；按机器分组切分 `20261008`；随机排序基线 `20261010`；
准入重放用连号种子（`20261008` 起），方便逐个换着复跑。见 `delay_aware_scheduling/common/seeds.py`。

---

## 已知边界

- 就绪延迟取自发布数据的就绪延迟字段。未就绪的作业该字段一律记为 `-1`；另有少量**已经就绪**
  的作业也记为 `-1`（平均声明时长远长于总体），本文口径下将其单独报告、不并入建模队列。
  详见 [docs/schema.md](docs/schema.md)。
- 机架字段（`asw_id`）只在观测期后段可用，涉及机架的统计只覆盖那一段时间。
- 重放是**静态准入重放**而非含时间轴的离散事件仿真：作业表没有时间戳，到达时刻无法重建。
  "有效计算时间占比"按"每个已放置作业占用其卡位（声明时长 + 就绪延迟）"折算。
- 重放假设"就绪结局不随准入顺序改变"；镜像缓存冷热、同机并发启动竞争等都可能让同一作业在
  不同顺序下拿到不同的就绪结果，端到端效果需要在真实集群上再验一次。
- 结论来自单个生产集群的观测，外推前建议先比对供给结构（单机卡数分布、机架同质性）与
  需求结构（请求规模分布）。
