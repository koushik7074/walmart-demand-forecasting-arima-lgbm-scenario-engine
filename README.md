# Walmart Store Sales Demand Forecasting
### A production-aware time series pipeline — ARIMA · SARIMA · ARIMAX · Prophet · LightGBM

---

## Overview

End-to-end retail demand forecasting pipeline built on Walmart's real store sales data (2010–2012). The project benchmarks five model families across five departments, makes data-driven regressor selection decisions, and implements a scenario stress-testing engine directly analogous to DFAST/CCAR macro conditioning in credit risk.

**The core question:** Can incorporating external drivers — temperature, promotions, holidays — improve weekly department-level sales forecasts beyond what seasonal autocorrelation alone captures?

---

## Results

| Dept | ARIMA RMSE | SARIMA RMSE | ARIMAX RMSE | Prophet RMSE | LightGBM RMSE | **Winner** | **vs ARIMA** |
|------|-----------|------------|------------|-------------|--------------|------------|-------------|
| 2    | 10,143    | 7,170      | **6,160**  | 10,116      | 6,545        | ARIMAX (refit) | **+39.3%** |
| 38   | 26,645    | 23,032     | 23,032     | **17,785**  | 20,684       | Prophet    | **+33.3%** |
| 72   | 71,602    | 27,769     | **24,881** | 31,500      | 29,019       | ARIMAX     | **+65.3%** |
| 92   | 24,268    | **14,002** | 14,002     | 23,010      | 15,844       | SARIMA     | **+42.3%** |
| 95   | 15,190    | 15,190     | **14,679** | 17,180      | 18,249       | ARIMAX     | **+3.4%**  |

> Lower RMSE is better. Holdout: last 30 weeks per department.

---

## Key Findings

**No single model dominates — different departments need different model families:**

- **SARIMA** — optimal for stable, externally-insensitive categories (Dept 92). Reduced RMSE 42% over ARIMA by capturing annual seasonality explicitly.
- **ARIMAX** — won 3 departments via per-department regressor selection. Dept 72 MAPE dropped from 40.3% to 10.1% with a single isholiday regressor.
- **Prophet** — the only model to overcome Dept 38's structural break. Piecewise linear trend adapted to a level-shift at Nov 2011 where all statistical models failed (SARIMA Ljung-Box p=0.000).
- **LightGBM** — `lag_52` was the top feature across all departments, confirming annual seasonality as the dominant signal. Improved Dept 38 over SARIMA despite the structural break but was outperformed by Prophet.

**Aggregation fallacy:** Day 1 aggregate CCF showed temperature at r=−0.439 across all departments. Per-department analysis revealed r=+0.121 for Dept 2 (wrong sign) — initial ARIMAX degraded all 5 departments until per-series diagnostics rebuilt the regressor config.

**Structural break:** Dept 38 showed a clear regime change at Nov 2011. Truncated training was mathematically infeasible (22 post-break training weeks vs 52 required for seasonal differencing). Prophet's trend flexibility resolved this without any manual intervention.

---

## Scenario Engine

ARIMAX enables conditioning forecasts on different macro paths — the same logic as DFAST stress testing in credit risk. Three scenarios were defined for departments with exogenous regressors:

| Scenario | Temperature | Fuel price | Markdown |
|---|---|---|---|
| Baseline | Unchanged | Unchanged | Same level |
| Demand Shock | −10°F cold snap | +20% | None |
| Growth Campaign | Unchanged | −5% | Doubled |

Dept 95 showed ~$40–50K weekly sales divergence between Demand Shock and Growth Campaign scenarios over a 30-week horizon.

---

## Methodology

### Data
- **Source:** [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) (Kaggle)
- **Files:** `train.csv`, `stores.csv`, `features.csv`
- **Scope:** 3 stores (one per type A/B/C, selected by volume) × 5 departments (top by sales)
- **Frequency:** Weekly (Friday-ending)
- **Period:** Feb 2010 — Oct 2012

### EDA (Day 1)
- Seasonal decomposition: strength 0.818–0.971 across all departments → justified SARIMA
- Stationarity: ADF + KPSS tests per department
- CCF analysis: temperature r=−0.439 strongest macro signal; CPI and unemployment excluded (near-zero per-dept correlation, Walmart counter-cyclical)
- Holiday lift: heterogeneous — Dept 72 +94.7%, Depts 2/38/95 negative lift
- Markdown lift: store-level not dept-level — all 5 depts had identical 153 markdown weeks

### Baseline Models (Day 2)
- `auto_arima` with AIC-based stepwise search
- SARIMA forced `D=1` (seasonal differencing) based on decomposition finding
- SARIMA beat ARIMA on 4/5 departments; Dept 72 improved 61% (seasonal misspecification corrected)
- Dept 95: SARIMA degraded — double differencing on 113 obs left ~60 usable; ARIMA retained

### ARIMAX (Day 3)
- Per-department correlation analysis before any fitting
- Single-regressor diagnostic testing before multi-regressor config
- Dept 38: structural break confirmed, truncation infeasible (22 post-break training weeks)
- Dept 72: convergence failure with multiple regressors — isholiday alone (9/112 holiday weeks)
- Scenario engine implemented for departments with exogenous regressors

### Prophet + LightGBM (Day 4)
- Prophet tested additive vs multiplicative seasonality per department; best selected by RMSE
- LightGBM features: lags at 1/4/8/12/26/52 weeks, rolling mean/std at 4/8/26 weeks, calendar features, exogenous variables
- `TimeSeriesSplit(n_splits=3)` used throughout — no random splitting on time series
- `lag_52` was top feature across all 5 departments — model implicitly learned annual seasonality

---

## Project Structure

```
walmart-demand-forecasting-arima-lgbm-scenario-engine/
│
├── raw_data/                    # Kaggle CSVs (gitignored)
│   ├── train.csv
│   ├── stores.csv
│   └── features.csv
│
├── outputs/                     # Generated plots and CSVs
│   ├── 01_sales_overview.png
│   ├── 02_decomposition.png
│   ├── 03_macro_trends.png
│   ├── 04_lagged_ccf.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_holiday_impact.png
│   ├── 07_markdown_impact.png
│   ├── 08_acf_pacf.png
│   ├── 09_forecast_comparison.png
│   ├── 10_residual_diagnostics.png
│   ├── 11_dept38_structural_break.png
│   ├── 12_arimax_forecast_comparison.png
│   ├── 13_arimax_residual_diagnostics.png
│   ├── 14_scenario_engine.png
│   ├── 15_lgbm_feature_importance.png
│   ├── 16_final_forecast_comparison.png
│   ├── 17_lgbm_residuals.png
│   ├── day2_sarima_baseline.csv
│   ├── day3_final_results.csv
│   └── day4_final_benchmark.csv
│
├── walmart_day1_eda.ipynb        # EDA — seasonality, CCF, holiday/markdown analysis
├── walmart_day2_baseline.ipynb  # ARIMA + SARIMA baseline
├── walmart_day3_arimax.ipynb    # ARIMAX + scenario engine
├── walmart_day4_models.ipynb    # Prophet + LightGBM + final benchmark
│
├── .gitignore
└── README.md
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Time series models | `statsmodels` (SARIMAX), `pmdarima` (auto_arima), `prophet` |
| ML model | `lightgbm` |
| Feature engineering | `pandas`, `numpy` |
| Evaluation | `sklearn.metrics` |
| Visualisation | `matplotlib`, `seaborn` |

---

## Limitations & Production Considerations

- **Sample size:** 113 training weeks (~2 years) limits SARIMA seasonal estimation. Dept 95 SARIMA degraded due to insufficient post-differencing observations.
- **Dept 38 structural break:** 22 post-break training weeks — insufficient for seasonal model. Production fix: intervention dummy variable or hierarchical forecasting.
- **Scenario engine confidence intervals:** Dept 72 Ljung-Box failed after AR order simplification — confidence bands should be treated with caution for interval forecasts.
- **Markdowns are store-level:** All 5 departments had identical markdown weeks — dept-level promotional response varies but the regressor captures store-wide intensity only.
- **Production deployment:** Would require retraining trigger on data drift (KS test), FastAPI inference endpoint, and automated holdout evaluation on new weeks.
