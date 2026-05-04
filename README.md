# Predicting-frequent-depression-in-U.S.-Adult
In this generation there are more and more people getting depression due to lot of reasons but most of them ignore what they really need, or think that it’s not a big deal. Therefore, if government can discover the high-risk group early and have the policy to handle it that will be good for the community.

This repository contains a reproducible analysis pipeline for a PSY 553 capstone project predicting frequent depression among U.S. adults using pooled NHIS 2022-2024 Sample Adult data.

## Project goal

The goal is to build an interpretable, survey-weighted risk-stratification model for frequent depression. This is a prediction and association project, not a causal analysis and not a diagnostic tool.

## Required data files

Download the NHIS Sample Adult public-use CSV files for 2022, 2023, and 2024 from the CDC/NCHS NHIS documentation pages. Place them in either the repository root or a `data/` folder:

```text
adult22.csv
adult23.csv
adult24.csv
```

Do not commit raw NHIS CSV files to a public GitHub repository. The `.gitignore` file excludes them.

## Main script

```bash
capstone pipeline.py
```

## Python environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## What the script produces

All outputs are saved in `outputs/`.

### Tables

- `outputs/tables/model_metrics_12var.csv`
- `outputs/tables/calibration_risk_groups_12var.csv`
- `outputs/tables/top_risk_thresholds_10_to_100.csv`
- `outputs/tables/top_risk_thresholds_10_to_30_summary.csv`
- `outputs/tables/rx_user_dual_barrier_prevalence.csv`

### Figures

- `outputs/figures/model_performance_12var_auc.png`
- `outputs/figures/chart1_12var_risk_distribution_percent.png`
- `outputs/figures/chart2_12var_predicted_vs_observed_percent.png`
- `outputs/figures/chart3_threshold_observed_prevalence_decline.png`
- `outputs/figures/chart4_threshold_case_capture_increase.png`
- `outputs/figures/rx_user_dual_barrier_prevalence.png`

## Final model

The final presentation model is the 12-variable weighted logistic model. LASSO logistic and random forest are included as comparators. Calibration and threshold analyses use the same 12-variable weighted logistic model to keep the performance, calibration, and policy story aligned.

## Final 12 predictors

- Self-rated health
- Food security
- Age group
- Income-to-poverty group
- Mental-health care cost barrier
- Race/ethnicity
- BMI category
- Could not get needed prescription due to cost
- Any prescription use
- Sex
- Marital status
- Smoking status

## Interpretation boundary

High predicted risk should trigger supportive outreach, screening invitations, care navigation, or resource planning. It should not be used as a causal explanation, diagnosis, eligibility rule, or service-denial rule.
