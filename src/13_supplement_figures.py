"""
13_supplement_figures.py
=========================
Produces:
- Figure S2: uncorrected predicted-vs-real age scatter (4 cohorts).
- Figure S3: BrainPAD raw vs Cole+Zhang corrected, showing the correction
             effect on every subject (4 cohorts).

Reads:
    ../data/predictions_TestSet.csv
    ../data/predictions_AgeRisk.csv
    ../data/predictions_ALSPAC_20_MRI_I.csv
    ../data/predictions_ALSPAC_30_MRI_II.csv

Writes:
    ../figures/FigureS2_uncorrected_predictions.svg
    ../figures/FigureS3_correction_effect.svg

Run from src/.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


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
# Figure S2 - uncorrected predicted-vs-real age
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (label, fname) in zip(axes, COHORTS.items()):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    ax.scatter(df["Edad"], df["pred_Edad"], s=14, alpha=0.55)
    lim = [
        min(df["Edad"].min(), df["pred_Edad"].min()),
        max(df["Edad"].max(), df["pred_Edad"].max()),
    ]
    ax.plot(lim, lim, "k--", lw=1, label="Ideal fit")
    ax.set_xlabel("Real age (years)")
    ax.set_ylabel("Predicted age, raw (years)")
    ax.set_title(f"{label} (n={len(df)})")
    ax.set_aspect("equal", adjustable="box")
fig.suptitle("Figure S2 - Uncorrected predicted vs real age",
             fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "FigureS2_uncorrected_predictions.svg"))
plt.close(fig)


# -----------------------------------------------------------------------------
# Figure S3 - BrainPAD raw vs corrected
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (label, fname) in zip(axes, COHORTS.items()):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    ax.scatter(df["BrainPAD"], df["BrainPAD_c"], s=14, alpha=0.55)
    lim = [
        min(df["BrainPAD"].min(), df["BrainPAD_c"].min()),
        max(df["BrainPAD"].max(), df["BrainPAD_c"].max()),
    ]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("BrainPAD (raw)")
    ax.set_ylabel("BrainPAD (Cole + Zhang corrected)")
    ax.set_title(f"{label} (n={len(df)})")
    ax.set_aspect("equal", adjustable="box")
fig.suptitle("Figure S3 - Effect of bias correction on BrainPAD",
             fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "FigureS3_correction_effect.svg"))
plt.close(fig)

print("Wrote Figure S2 and Figure S3 to", OUT_DIR)
