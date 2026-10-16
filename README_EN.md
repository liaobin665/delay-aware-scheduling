# Ready-Delay Prediction and Delay-Aware Admission for Heterogeneous GPU Clusters

This repository does two things:

1. **Measures ready delay** — the interval during which a job already holds GPU capacity but has not yet
   reached the ready state, so the cards are occupied without producing any computation. The code
   quantifies this delay from a public production trace and analyses its distribution, tail and
   concentration across accelerator types, job types and model types.
2. **Implements delay-aware scheduling (DAS)** — a predictor trained only on attributes observable at
   admission time estimates the ready delay, and the admission order is then sorted by the predicted
   delay while the **placement rule stays unchanged**. The effect is validated by replaying historical
   jobs.

The code is split into stage packages; each stage has its own entry script, writes its artefacts into
`out/`, and the next stage reads them.

> This repository ships **no data**. The raw trace must be downloaded separately — see
> [data/README.md](data/README.md).

---

## Layout

```
.
├── delay_aware_scheduling/      main package
│   ├── config.py                path conventions (data / mid / out, overridable by env vars)
│   ├── common/                  shared helpers (logging, random seeds)
│   ├── audit/                   caliber checks, distribution statistics, modelling sample
│   ├── prediction/              point and quantile prediction of the ready delay
│   ├── analysis/                prediction error, attribution and auxiliary analyses
│   ├── structure/               reconstruction of the supply-side machine list
│   ├── fleet/                   packing engine and synthetic / measured fleets
│   └── scheduling/              admission replay experiments and statistics
├── data/                        put the raw data here (not shipped)
├── mid/                         intermediate tables (e.g. the modelling sample)
├── out/                         results (json / csv / parquet)
├── docs/                        field definitions and run order
├── tests/                       smoke tests that need no data
├── requirements.txt
└── pyproject.toml
```

The three data directories have distinct roles: `data/` is read-only and holds the unpacked raw
tables; `mid/` holds intermediate tables reused across steps; `out/` holds each step's results. To keep
the large files elsewhere:

```bash
export DELAY_AWARE_SCHEDULING_DATA=/data/asi      # raw data
export DELAY_AWARE_SCHEDULING_MID=/scratch/mid    # intermediate tables
export DELAY_AWARE_SCHEDULING_OUT=/scratch/out    # results
```

---

## Environment

Python ≥ 3.10. Dependencies are listed in `requirements.txt`; alternatively install the package:

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

---

## Run order

Run the steps below in order. All entry points are invoked with `python -m` (the package uses
relative imports).

```bash
# 1. supply-side machine list (the "measured" fleet used by the replay)
python -m delay_aware_scheduling.structure.exp70_supply

# 2. caliber checks and modelling sample
python -m delay_aware_scheduling.audit.exp60_caliber
python -m delay_aware_scheduling.audit.exp61_sample

# 3. ready-delay prediction
python -m delay_aware_scheduling.prediction.exp62_models

# 4. row-wise SHAP and embedding sample (optional, for attribution figures)
python -m delay_aware_scheduling.analysis.exp67_shap
python -m delay_aware_scheduling.analysis.exp68_embed

# 5. admission replay and statistics
python -m delay_aware_scheduling.scheduling.exp63_admission
python -m delay_aware_scheduling.scheduling.exp64_stats

# 6. auxiliary analyses
python -m delay_aware_scheduling.analysis.exp65_extras
python -m delay_aware_scheduling.analysis.exp66_infer
python -m delay_aware_scheduling.analysis.exp69_fignums
```

`exp70_supply` and `exp63_admission` are the two heaviest steps (the former scans 185 days of hourly
server tables; the latter replays every (fleet × load × seed × placement) cell). For a quick trial,
`exp63_admission --smoke` runs only two load levels and `exp62_models --quick` subsamples 200k rows.

---

## What each file does

### `delay_aware_scheduling/` (top level)

| File | Purpose |
|---|---|
| `config.py` | Path conventions. Resolves `data/`, `mid/`, `out/` and a few shared file paths; overridable by environment variables. |

### `common/` shared helpers

| File | Purpose |
|---|---|
| `log.py` | The shared `out()` console helper (also switches stdout to UTF-8). |
| `seeds.py` | Fixed seeds for sampling, splitting and the random ranking key. |

### `audit/` caliber and sample

| File | Purpose |
|---|---|
| `exp60_caliber.py` | Full-table caliber check: combinations of the ready-status and ready-delay fields, delay quantiles and extremes, concentration, and the breakdown by accelerator / job / model type. |
| `exp61_sample.py` | Draws a 5% machine-grouped modelling sample (all rows of a machine enter or leave together). |

### `prediction/` ready-delay prediction

| File | Purpose |
|---|---|
| `exp62_models.py` | Entry point: regression and quantile regression (P50/P90/P99) over nine admission-time attributes, five machine-grouped folds, all metrics on out-of-fold predictions; plus an L0-only ablation and a zero-parameter lookup baseline. |
| `features.py` | Feature construction (L0 / L1 layers) and machine-grouped fold assignment. |
| `metrics.py` | Error metrics on both log and second scales; ranking agreement via Spearman and Kendall. |

### `analysis/` auxiliary analyses

| File | Purpose |
|---|---|
| `exp65_extras.py` | Prediction error (in particular the underestimation of the long-delay subgroup), quantile calibration, the fairness–efficiency trade-off curve, and predicting quality against scheduling gain. |
| `exp66_infer.py` | Measured inference overhead: per-call cost for three calling styles, model size, jobs per second. |
| `exp67_shap.py` | Row-wise SHAP values (for the beeswarm and dependence figures). |
| `exp68_embed.py` | Embedding/grouping sample: standardised attribute matrix, true log delay, group labels. |
| `exp69_fignums.py` | A handful of deterministic readings used by the figures (group median delay, mean SHAP, principal-component variance share). |

### `structure/` supply side

| File | Purpose |
|---|---|
| `exp70_supply.py` | Reconstructs the reference-hour machine list from the hourly server tables and fills in machines seen only on other days, yielding a union covering every `server_id`. |

### `fleet/` packing engine

| File | Purpose |
|---|---|
| `constants.py` | Capacity scale (1 GPU = 100 integer units), request-size bins, fleet-shape levels. |
| `capacity.py` | Index structures backing the three placement rules: `MaxSeg` (first fit), `CapIdx` (best fit), `Fleet` (cluster state). |
| `fleetgen.py` | Parameterised synthetic fleet generation, plus loading of the measured machine list. |

### `scheduling/` admission replay

| File | Purpose |
|---|---|
| `exp63_admission.py` | Entry point: the replay experiment — change the admission order, the placement rule and the offered load, and watch the mean ready delay and the effective compute-time fraction. |
| `policies.py` | Admission ordering rules (FCFS / SJF / the proposed method sorted by predicted delay / …) and the composite-score weight ladder. |
| `replay.py` | The replay kernel: feed jobs to the placement rule in the given order and accumulate the metrics. |
| `exp64_stats.py` | Entry point: statistics of the replay results and the result tables. |
| `paired.py` | Reading replay results and the paired-difference tests (paired-t interval plus a Wilcoxon cross-check). |
| `tables.py` | Assembly of the result tables (table bodies are generated by script, never hand-typed). |

---

## Tests

Only the data-free parts are tested (capacity indices, fleet generation, metric formulas, fold
assignment and ranking rules):

```bash
pytest
```

---

## Random seeds

Sampling `20261011`; machine-grouped split `20261008`; random ranking baseline `20261010`; the replay
uses consecutive seeds starting at `20261008` so runs can be swapped one at a time. See
`delay_aware_scheduling/common/seeds.py`.

---

## Known limitations

- The ready delay is taken from the released ready-delay field. For jobs that never became ready the
  field is `-1`; a small number of **already ready** jobs are also recorded as `-1` (their mean declared
  duration is far longer than the population's). Under our caliber they are reported separately and
  excluded from the modelling cohort. See [docs/schema.md](docs/schema.md).
- The rack field (`asw_id`) is only available in the later part of the observation window, so
  rack-related statistics cover only that period.
- The replay is a **static admission replay**, not a time-resolved discrete-event simulation: the job
  table carries no timestamps, so arrival times cannot be reconstructed. The effective compute-time
  fraction is therefore computed as "every placed job occupies its cards for the declared duration
  plus its ready delay".
- The replay assumes the ready outcome does not change with the admission order; image cache warmth and
  concurrent start-up contention on the same machine can both make the same job behave differently
  under a different order, so the end-to-end effect still needs validation on a live cluster.
- The findings come from a single production cluster; before extrapolating, compare the supply
  structure (per-machine GPU counts, rack homogeneity) and the demand structure (request-size
  distribution) with your own.
