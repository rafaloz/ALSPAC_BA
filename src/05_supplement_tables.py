"""
05_supplement_tables.py
=========================
Produces:
- Table S5: per-cohort model performance (MAE, R^2, Pearson r) for raw and
            bias-corrected predictions.
- Table S6: BrainPAD Cohen's d (controls vs pooled PE+) before and after
            the Cole+Zhang bias correction.

Reads:
    ../data/predictions_TestSet.csv
    ../data/predictions_AgeRisk.csv
    ../data/predictions_ALSPAC_20_MRI_I.csv
    ../data/predictions_ALSPAC_30_MRI_II.csv

Writes:
    ../figures/TableS5_per_wave_performance.csv
    ../figures/TableS6_bias_correction_effect_sizes.csv

Run from src/.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


DATA_DIR = "../data"
OUT_DIR  = "../figures"
os.makedirs(OUT_DIR, exist_ok=True)

COHORTS = {
    "TestSet":           "predictions_TestSet.csv",
    "AgeRisk":           "predictions_AgeRisk.csv",
    "ALSPAC-20 MRI-I":   "predictions_ALSPAC_20_MRI_I.csv",
    "ALSPAC-30 MRI-II":  "predictions_ALSPAC_30_MRI_II.csv",
}


# -----------------------------------------------------------------------------
# Table S5: per-cohort model performance
# -----------------------------------------------------------------------------
rows_s5 = []
for label, fname in COHORTS.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    for kind, pred_col in [("raw", "pred_Edad"),
                           ("bias-corrected", "pred_Edad_c")]:
        mae = mean_absolute_error(df["Edad"], df[pred_col])
        r2  = r2_score(df["Edad"], df[pred_col])
        r, _ = pearsonr(df["Edad"], df[pred_col])
        rows_s5.append({
            "cohort":     label,
            "n":          len(df),
            "prediction": kind,
            "MAE":        round(mae, 3),
            "R2":         round(r2, 3),
            "Pearson_r":  round(r, 3),
        })

table_s5 = pd.DataFrame(rows_s5)
table_s5.to_csv(os.path.join(OUT_DIR, "TableS5_per_wave_performance.csv"),
                index=False)
print("=== Table S5 — Per-wave model performance ===")
print(table_s5.to_string(index=False))


# -----------------------------------------------------------------------------
# Table S6: bias-correction effect on Cohen's d (controls vs PE+)
# -----------------------------------------------------------------------------
rows_s6 = []
for label, fname in COHORTS.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    if "pliks18TH" not in df.columns:
        continue
    for kind, pad_col in [("raw", "BrainPAD"),
                          ("bias-corrected", "BrainPAD_c")]:
        ctrl = df[df["pliks18TH"] == 0][pad_col].dropna()
        pe   = df[df["pliks18TH"] != 0][pad_col].dropna()
        n_ctrl, n_pe = len(ctrl), len(pe)
        if n_ctrl < 5 or n_pe < 5:
            continue
        s_pooled = np.sqrt(
            ((n_ctrl - 1) * ctrl.var() + (n_pe - 1) * pe.var())
            / (n_ctrl + n_pe - 2)
        )
        d = (pe.mean() - ctrl.mean()) / s_pooled
        rows_s6.append({
            "cohort":     label,
            "prediction": kind,
            "n_ctrl":     n_ctrl,
            "n_PE":       n_pe,
            "mean_ctrl":  round(ctrl.mean(), 3),
            "mean_PE":    round(pe.mean(), 3),
            "cohen_d":    round(d, 3),
        })

table_s6 = pd.DataFrame(rows_s6)
table_s6.to_csv(os.path.join(OUT_DIR, "TableS6_bias_correction_effect_sizes.csv"),
                index=False)
print("\n=== Table S6 — BrainPAD effect size pre/post bias correction ===")
print(table_s6.to_string(index=False))
