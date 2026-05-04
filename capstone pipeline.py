#!/usr/bin/env python3
"""
NHIS 2022-2024 Capstone Analysis Pipeline
Predicting frequent depression using an interpretable 12-variable model.

Author: Alan Hsu
Course: PSY 553 General Psychology Capstone

Purpose
-------
This script reproduces the final capstone workflow:
1. Load pooled NHIS 2022-2024 Sample Adult CSV files.
2. Define frequent_depression from DEPFREQ_A.
3. Apply pooled survey weight WTFA_A / 3.
4. Recode 12 human-readable predictors.
5. Compare three 12-variable models:
   - weighted logistic regression
   - LASSO logistic regression
   - random forest
6. Evaluate weighted AUC, weighted Brier score, calibration slope, and expected calibration error.
7. Produce simplified calibration figures and top-risk threshold figures.
8. Produce a secondary Rx-user dual-barrier descriptive extension.

Important interpretation
------------------------
This is a prediction / association workflow, not a causal analysis and not a diagnostic tool.
The model is intended for risk stratification, outreach prioritization, and public-health planning.

Expected input files
--------------------
Place the following files in the same folder as this script, or place them in a data/ folder:
- adult22.csv
- adult23.csv
- adult24.csv 

Outputs
-------
All outputs are saved under outputs/:
- tables/model_metrics_12var.csv
- tables/calibration_risk_groups_12var.csv
- tables/top_risk_thresholds_10_to_100.csv
- tables/rx_user_dual_barrier_prevalence.csv
- figures/chart1_12var_risk_distribution_percent.png
- figures/chart2_12var_predicted_vs_observed_percent.png
- figures/chart3_threshold_observed_prevalence_decline.png
- figures/chart4_threshold_case_capture_increase.png
- figures/rx_user_dual_barrier_prevalence.png

Dependencies
------------
Python 3.10+
pandas, numpy, scikit-learn, matplotlib, statsmodels
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 20260422
ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def find_file(candidates: Iterable[str]) -> Path:
    """Return the first existing file path among candidates."""
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            return p
        p2 = ROOT / cand
        if p2.exists():
            return p2
        p3 = ROOT / "data" / cand
        if p3.exists():
            return p3
    raise FileNotFoundError(f"Could not find any of: {list(candidates)}")


def yes_no(series: pd.Series) -> pd.Series:
    """NHIS convention: 1=yes, 2=no, other values missing."""
    return pd.Series(np.where(series == 1, 1, np.where(series == 2, 0, np.nan)), index=series.index)


def weighted_mean(x: pd.Series | np.ndarray, w: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def weighted_brier(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return weighted_mean((np.asarray(y) - np.asarray(p)) ** 2, w)


def weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUC using sklearn's sample_weight implementation."""
    return float(roc_auc_score(y, p, sample_weight=w))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Weighted logistic recalibration: y ~ intercept + slope * logit(prediction)."""
    eps = 1e-6
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p)
    weights = np.asarray(w, dtype=float) / np.mean(w)
    res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights).fit(maxiter=200, disp=False)
    return float(res.params[0]), float(res.params[1])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, w: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected calibration error using weighted quantile bins.
    ECE is the weighted average absolute gap between mean predicted risk and observed prevalence.
    """
    df = pd.DataFrame({"y": y, "p": p, "w": w}).sort_values("p")
    df["cum_w"] = df["w"].cumsum() / df["w"].sum()
    df["bin"] = np.minimum(np.floor(df["cum_w"] * n_bins).astype(int), n_bins - 1) + 1
    rows = []
    for _, g in df.groupby("bin"):
        share = g["w"].sum() / df["w"].sum()
        pred = weighted_mean(g["p"], g["w"])
        obs = weighted_mean(g["y"], g["w"])
        rows.append(share * abs(pred - obs))
    return float(np.sum(rows))


def load_and_recode() -> pd.DataFrame:
    """Load pooled NHIS CSV files and create final analytic variables."""
    p22 = find_file(["adult22.csv"])
    p23 = find_file(["adult23.csv"])
    p24 = find_file(["adult24.csv", "adult24(1).csv"])

    print(f"Loading files:\n  {p22}\n  {p23}\n  {p24}")
    # Read only the columns needed for this capstone workflow.
    needed_cols = [
        "DEPFREQ_A", "WTFA_A", "PPSU", "PSTRAT", "SRVY_YR", "INTV_QRT",
        "RX12M_A", "RXDG12M_A", "RXSK12M_A", "RXDL12M_A", "MHTHND_A",
        "AGEP_A", "PHSTAT_A", "FDSCAT4_A", "RATCAT_A", "HISPALLP_A",
        "BMICAT_A", "SEX_A", "MARSTAT_A", "SMKCIGST_A"
    ]
    frames = []
    for p in [p22, p23, p24]:
        header = pd.read_csv(p, nrows=0).columns.tolist()
        usecols = [c for c in needed_cols if c in header]
        frames.append(pd.read_csv(p, usecols=usecols))
    df = pd.concat(frames, ignore_index=True)

    # Outcome: daily or weekly depressed mood vs all lower-frequency valid responses.
    df["frequent_depression"] = np.where(
        df["DEPFREQ_A"].isin([1, 2]), 1,
        np.where(df["DEPFREQ_A"].isin([3, 4, 5]), 0, np.nan)
    )
    df = df.loc[df["frequent_depression"].notna()].copy()
    df["frequent_depression"] = df["frequent_depression"].astype(int)
    df["pooled_weight"] = df["WTFA_A"].astype(float) / 3.0

    # Core binary recodes.
    df["rx_user_num"] = yes_no(df["RX12M_A"])
    df["rx_cantget_num"] = yes_no(df["RXDG12M_A"])
    df["rx_skip_num"] = yes_no(df["RXSK12M_A"]) if "RXSK12M_A" in df.columns else np.nan
    df["rx_delay_num"] = yes_no(df["RXDL12M_A"]) if "RXDL12M_A" in df.columns else np.nan
    df["mh_cost_barrier_num"] = yes_no(df["MHTHND_A"])

    # Human-readable 12 predictor variables.
    age = df["AGEP_A"].where(df["AGEP_A"].between(18, 85), np.nan)
    df["age_group"] = pd.cut(
        age, bins=[17, 24, 34, 44, 54, 64, 74, 200],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    ).astype(str).replace("nan", "Missing/unknown")

    df["self_rated_health"] = df["PHSTAT_A"].map({
        1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"
    }).fillna("Missing/unknown")

    df["food_security"] = df["FDSCAT4_A"].map({
        1: "High food security", 2: "Marginal food security",
        3: "Low food security", 4: "Very low food security"
    }).fillna("Missing/unknown")

    rat = df["RATCAT_A"].where(df["RATCAT_A"].between(1, 14), np.nan)
    df["income_to_poverty_group"] = pd.cut(
        rat, bins=[0, 5, 8, 11, 14],
        labels=["Lowest income", "Lower-middle income", "Middle income", "Higher income"],
        include_lowest=True
    ).astype(str).replace("nan", "Missing/unknown")

    df["mental_health_care_cost_barrier"] = np.where(
        df["mh_cost_barrier_num"] == 1, "MH care cost barrier",
        np.where(df["mh_cost_barrier_num"] == 0, "No MH care cost barrier", "Missing/unknown")
    )

    df["race_ethnicity"] = df["HISPALLP_A"].map({
        1: "Hispanic", 2: "Non-Hispanic White", 3: "Non-Hispanic Black",
        4: "Non-Hispanic Asian", 5: "Non-Hispanic AIAN",
        6: "Non-Hispanic other/multiple", 7: "Unknown race/ethnicity"
    }).fillna("Missing/unknown")

    df["bmi_category"] = df["BMICAT_A"].map({
        1: "Underweight", 2: "Healthy weight", 3: "Overweight", 4: "Obesity"
    }).fillna("Missing/unknown")

    df["could_not_get_rx_due_to_cost"] = np.where(
        df["rx_cantget_num"] == 1, "Could not get needed Rx due to cost",
        np.where(df["rx_cantget_num"] == 0, "No Rx access cost barrier", "Missing/unknown")
    )

    df["any_prescription_use"] = np.where(
        df["rx_user_num"] == 1, "Used prescription medicine",
        np.where(df["rx_user_num"] == 0, "No prescription use", "Missing/unknown")
    )

    df["sex"] = df["SEX_A"].map({1: "Male", 2: "Female"}).fillna("Missing/unknown")

    df["marital_status"] = df["MARSTAT_A"].map({
        1: "Married/living with partner", 2: "Married/living with partner", 3: "Married/living with partner",
        4: "Widowed", 5: "Divorced/separated", 6: "Divorced/separated", 7: "Never married",
        8: "Not currently partnered", 9: "Unknown marital status"
    }).fillna("Missing/unknown")

    df["smoking_status"] = df["SMKCIGST_A"].map({
        1: "Current smoker", 2: "Current smoker", 3: "Former smoker", 4: "Never smoker",
        5: "Unknown smoking", 9: "Unknown smoking"
    }).fillna("Missing/unknown")

    df["survey_year"] = df["SRVY_YR"].astype(str)
    df["interview_quarter"] = df["INTV_QRT"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}).fillna("Missing/unknown")

    return df


FEATURES_12 = [
    "self_rated_health", "food_security", "age_group", "income_to_poverty_group",
    "mental_health_care_cost_barrier", "race_ethnicity", "bmi_category",
    "could_not_get_rx_due_to_cost", "any_prescription_use", "sex", "marital_status", "smoking_status"
]


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 train/validation/test split stratified by year and outcome."""
    strat = df["survey_year"].astype(str) + "_" + df["frequent_depression"].astype(str)
    train_df, temp_df = train_test_split(df, test_size=0.40, random_state=RANDOM_STATE, stratify=strat)
    strat_temp = temp_df["survey_year"].astype(str) + "_" + temp_df["frequent_depression"].astype(str)
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=strat_temp)
    for part in [train_df, valid_df, test_df]:
        part.loc[:, "fit_weight"] = part["pooled_weight"] / part["pooled_weight"].mean()
    return train_df.copy(), valid_df.copy(), test_df.copy()


def build_model(name: str) -> Pipeline:
    """Create a model pipeline for the 12 categorical predictors."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_12)
    ], remainder="drop")

    if name == "weighted_logistic":
        clf = LogisticRegression(max_iter=1500, solver="lbfgs", C=1e6)
    elif name == "lasso_logistic":
        clf = LogisticRegression(max_iter=1500, solver="liblinear", penalty="l1", C=0.12)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    return Pipeline([("prep", prep), ("clf", clf)])


def fit_predict_model(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Pipeline, np.ndarray]:
    model = build_model(name)
    model.fit(
        train_df[FEATURES_12],
        train_df["frequent_depression"],
        clf__sample_weight=train_df["fit_weight"]
    )
    pred = model.predict_proba(test_df[FEATURES_12])[:, 1]
    return model, pred


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    y_test = test_df["frequent_depression"].values
    w_test = test_df["pooled_weight"].values

    model_names = ["lasso_logistic", "weighted_logistic", "random_forest"]
    display_names = {
        "lasso_logistic": "LASSO logistic",
        "weighted_logistic": "Weighted logistic",
        "random_forest": "Random forest",
    }

    preds = {}
    rows = []
    fitted = {}
    for name in model_names:
        print(f"Fitting {name}...")
        model, pred = fit_predict_model(name, train_df, test_df)
        fitted[name] = model
        preds[name] = pred
        intercept, slope = calibration_slope_intercept(y_test, pred, w_test)
        rows.append({
            "model": display_names[name],
            "weighted_auc": weighted_auc(y_test, pred, w_test),
            "weighted_brier": weighted_brier(y_test, pred, w_test),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "weighted_ece_10_bins": expected_calibration_error(y_test, pred, w_test, n_bins=10),
        })

    metrics = pd.DataFrame(rows).sort_values("weighted_auc", ascending=False)
    metrics.to_csv(TAB_DIR / "model_metrics_12var.csv", index=False)
    return metrics, preds


def plot_model_metrics(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = metrics.sort_values("weighted_auc")
    ax.barh(plot_df["model"], plot_df["weighted_auc"])
    for i, v in enumerate(plot_df["weighted_auc"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Weighted AUC")
    ax.set_title("12-variable model performance")
    ax.set_xlim(0.72, 0.82)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_performance_12var_auc.png", dpi=220)
    plt.close(fig)


def make_calibration_and_threshold_figures(test_df: pd.DataFrame, pred: np.ndarray) -> None:
    """Produce final simplified calibration and threshold plots for the selected logistic model."""
    df = test_df.copy()
    df["predicted_risk"] = pred

    bins = np.array([0, 0.025, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00])
    labels = ["0-2.5%", "2.5-5%", "5-8%", "8-10%", "10-15%", "15-20%", "20-30%", "30-40%", "40-60%", "60-100%"]
    df["risk_group"] = pd.cut(df["predicted_risk"], bins=bins, labels=labels, include_lowest=True, right=False)

    rows = []
    total_weight = df["pooled_weight"].sum()
    for label in labels:
        g = df[df["risk_group"] == label]
        if len(g) == 0:
            rows.append({
                "risk_group": label,
                "weighted_population_pct": 0.0,
                "mean_predicted_risk_pct": np.nan,
                "observed_prevalence_pct": np.nan,
            })
        else:
            rows.append({
                "risk_group": label,
                "weighted_population_pct": 100 * g["pooled_weight"].sum() / total_weight,
                "mean_predicted_risk_pct": 100 * weighted_mean(g["predicted_risk"], g["pooled_weight"]),
                "observed_prevalence_pct": 100 * weighted_mean(g["frequent_depression"], g["pooled_weight"]),
            })
    cal = pd.DataFrame(rows)
    cal.to_csv(TAB_DIR / "calibration_risk_groups_12var.csv", index=False)

    # Figure 1: predicted-risk distribution.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(cal["risk_group"], cal["weighted_population_pct"])
    ax.set_title("12-variable model: weighted population percentage across predicted-risk groups")
    ax.set_xlabel("Predicted-risk group")
    ax.set_ylabel("Weighted population (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(cal["weighted_population_pct"]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart1_12var_risk_distribution_percent.png", dpi=220)
    plt.close(fig)

    # Figure 2: predicted vs observed within risk groups.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(cal))
    ax.plot(x, cal["mean_predicted_risk_pct"], marker="o", label="Mean predicted risk")
    ax.plot(x, cal["observed_prevalence_pct"], marker="o", label="Observed prevalence")
    ax.set_title("12-variable model: predicted risk vs observed prevalence by risk group")
    ax.set_xlabel("Predicted-risk group")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cal["risk_group"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart2_12var_predicted_vs_observed_percent.png", dpi=220)
    plt.close(fig)

    # Top-risk thresholds: ranked weighted population.
    ranked = df.sort_values("predicted_risk", ascending=False).copy()
    ranked["cum_weight"] = ranked["pooled_weight"].cumsum()
    ranked["cum_weight_share"] = ranked["cum_weight"] / ranked["pooled_weight"].sum()
    total_case_weight = (ranked["pooled_weight"] * ranked["frequent_depression"]).sum()

    thr_rows = []
    for t in np.arange(0.10, 1.01, 0.10):
        subset = ranked[ranked["cum_weight_share"] <= t].copy()
        if subset.empty:
            subset = ranked.iloc[[0]].copy()
        thr_rows.append({
            "top_pct": int(round(t * 100)),
            "top_group": f"Top {int(round(t * 100))}%",
            "risk_threshold_pct": 100 * subset["predicted_risk"].min(),
            "observed_prevalence_pct": 100 * weighted_mean(subset["frequent_depression"], subset["pooled_weight"]),
            "case_capture_pct": 100 * ((subset["pooled_weight"] * subset["frequent_depression"]).sum() / total_case_weight),
        })
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(TAB_DIR / "top_risk_thresholds_10_to_100.csv", index=False)
    thr.loc[thr["top_pct"].isin([10, 20, 30])].to_csv(TAB_DIR / "top_risk_thresholds_10_to_30_summary.csv", index=False)

    # Figure 3: observed prevalence declines as top group expands.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thr["top_pct"], thr["observed_prevalence_pct"], marker="o")
    ax.set_title("Observed prevalence declines as the selected top-risk group expands")
    ax.set_xlabel("Selected top-risk group (%)")
    ax.set_ylabel("Observed prevalence (%)")
    ax.set_xticks(thr["top_pct"])
    ax.grid(axis="y", alpha=0.25)
    for xval, yval in zip(thr["top_pct"], thr["observed_prevalence_pct"]):
        ax.text(xval, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart3_threshold_observed_prevalence_decline.png", dpi=220)
    plt.close(fig)

    # Figure 4: case capture rises as top group expands.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thr["top_pct"], thr["case_capture_pct"], marker="o")
    ax.set_title("Case capture increases as the selected top-risk group expands")
    ax.set_xlabel("Selected top-risk group (%)")
    ax.set_ylabel("Case capture (%)")
    ax.set_xticks(thr["top_pct"])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    for xval, yval in zip(thr["top_pct"], thr["case_capture_pct"]):
        ax.text(xval, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart4_threshold_case_capture_increase.png", dpi=220)
    plt.close(fig)


def rx_user_extension(df: pd.DataFrame) -> None:
    """Secondary Rx-user descriptive extension: barrier severity x MH care cost barrier."""
    rx = df.loc[
        (df["rx_user_num"] == 1)
        & df["rx_cantget_num"].notna()
        & df["rx_skip_num"].notna()
        & df["rx_delay_num"].notna()
        & df["mh_cost_barrier_num"].notna()
    ].copy()
    rx["barrier_count"] = (rx["rx_cantget_num"] + rx["rx_skip_num"] + rx["rx_delay_num"]).astype(int)
    rx["mh_cost_barrier_label"] = np.where(rx["mh_cost_barrier_num"] == 1, "MH care cost barrier", "No MH care cost barrier")

    rows = []
    for mh, g1 in rx.groupby("mh_cost_barrier_label"):
        for bc, g2 in g1.groupby("barrier_count"):
            rows.append({
                "mh_cost_barrier": mh,
                "barrier_count": int(bc),
                "n_unweighted": int(len(g2)),
                "weighted_population": float(g2["pooled_weight"].sum()),
                "observed_prevalence_pct": 100 * weighted_mean(g2["frequent_depression"], g2["pooled_weight"]),
            })
    rx_tab = pd.DataFrame(rows).sort_values(["mh_cost_barrier", "barrier_count"])
    rx_tab.to_csv(TAB_DIR / "rx_user_dual_barrier_prevalence.csv", index=False)

    # Plot Rx-user dual-barrier pattern.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, g in rx_tab.groupby("mh_cost_barrier"):
        ax.plot(g["barrier_count"], g["observed_prevalence_pct"], marker="o", label=label)
    ax.set_title("Rx users: medication barrier count by MH care cost barrier")
    ax.set_xlabel("Medication barrier count (0-3)")
    ax.set_ylabel("Observed frequent depression prevalence (%)")
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rx_user_dual_barrier_prevalence.png", dpi=220)
    plt.close(fig)

    # Optional weighted GLM interaction model. This is approximate survey weighting, not full design variance.
    try:
        rx["barrier_count_factor"] = rx["barrier_count"].astype(str)
        weights = rx["pooled_weight"] / rx["pooled_weight"].mean()
        model = smf.glm(
            "frequent_depression ~ C(barrier_count_factor) * C(mh_cost_barrier_label) + C(survey_year) + C(interview_quarter)",
            data=rx,
            family=sm.families.Binomial(),
            freq_weights=weights,
        ).fit(maxiter=200, disp=False)
        with open(TAB_DIR / "rx_user_interaction_glm_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
    except Exception as exc:
        with open(TAB_DIR / "rx_user_interaction_glm_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Interaction GLM could not be estimated: {exc}\n")


def main() -> None:
    df = load_and_recode()
    train_df, valid_df, test_df = split_data(df)

    summary = {
        "analytic_n_unweighted": int(len(df)),
        "weighted_prevalence_frequent_depression": weighted_mean(df["frequent_depression"], df["pooled_weight"]),
        "train_n": int(len(train_df)),
        "valid_n": int(len(valid_df)),
        "test_n": int(len(test_df)),
        "random_state": RANDOM_STATE,
    }
    with open(OUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    metrics, preds = evaluate_models(train_df, test_df)
    print(metrics)
    plot_model_metrics(metrics)

    # Final selected model: weighted logistic, because it is transparent and best aligned with calibration/policy use.
    make_calibration_and_threshold_figures(test_df, preds["weighted_logistic"])
    rx_user_extension(df)

    print(f"\nDone. Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
