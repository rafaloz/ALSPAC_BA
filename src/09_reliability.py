"""
09_reliability.py
=================

Within-subject reliability of the bias-corrected BrainPAD on the longitudinal
control subset (n=61; supplement Fig S5).

Computes:
    - Per-wave precision (r, MAE, R^2)
    - ICC(3,1) with 95% CI (Pingouin)
    - SEM (in years)
    - Spearman rho (wave I vs wave II)
    - Lin's Concordance Correlation Coefficient with bootstrap CI
    - LMM: pred_Edad_c ~ Edad + (1 | ID) on the control subset
    - ΔBrainPAD vs ΔAge correlation
    - Standardised Response Mean (SRM)
    - Bland-Altman plot (saved as SVG)

Reads:
    ../data/lmm_input_long_pairs.csv    (long table, both visits per ID)

Writes:
    ../figures/Figure_S5_BlandAltman.svg

Run from src/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.formula.api as smf
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torchmetrics.regression import ConcordanceCorrCoef


DATA_DIR = "../data"
FIG_DIR  = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

LONG_PAIRS_CSV = f"{DATA_DIR}/lmm_input_long_pairs.csv"
OUT_BLAND      = f"{FIG_DIR}/Figure_S5_BlandAltman.svg"


def main() -> None:
    df = pd.read_csv(LONG_PAIRS_CSV)

    controls_long = df[df["pliks18TH"] == 0].copy()

    # 1) Per-wave precision -------------------------------------------------
    stats_by_wave = []
    for wave, label in [(0, "20y"), (1, "30y")]:
        sub = controls_long[controls_long["Time"] == wave]
        r = sub[["pred_Edad", "Edad"]].corr().iloc[0, 1]
        mae = mean_absolute_error(sub["Edad"], sub["pred_Edad"])
        R2  = r2_score(sub["Edad"], sub["pred_Edad"])
        stats_by_wave.append({"wave": label, "r": r, "MAE": mae, "R2": R2})
    print("\n=== Precision per wave ===")
    print(pd.DataFrame(stats_by_wave).to_string(index=False))

    # 2) Wide format on BrainPAD_c ------------------------------------------
    wide = (controls_long.pivot(index="ID", columns="Time", values="BrainPAD_c")
                          .rename(columns={0: "BrainPAD_20y", 1: "BrainPAD_30y"})
                          .dropna())

    # 3) ICC(3,1) via Pingouin ----------------------------------------------
    long_for_icc = (wide.reset_index()
                       .melt(id_vars="ID", var_name="Time", value_name="BrainPAD_c"))
    icc = pg.intraclass_corr(data=long_for_icc, targets="ID",
                             raters="Time", ratings="BrainPAD_c")
    icc_val = icc.loc[icc["Type"] == "ICC3", "ICC"].values[0]
    icc_ci  = icc.loc[icc["Type"] == "ICC3", "CI95%"].values[0]

    # 4) SEM -----------------------------------------------------------------
    sem = np.sqrt((1 - icc_val) * wide.var(axis=1, ddof=1).mean())

    # 5) Spearman rho between waves -----------------------------------------
    rho = wide["BrainPAD_20y"].corr(wide["BrainPAD_30y"], method="spearman")

    # 6) Lin's Concordance Correlation Coefficient + bootstrap CI -----------
    x = torch.tensor(wide["BrainPAD_20y"].values, dtype=torch.float32)
    y = torch.tensor(wide["BrainPAD_30y"].values, dtype=torch.float32)
    ccc_metric = ConcordanceCorrCoef()
    ccc = ccc_metric(x, y)

    n_boot = 1000
    bootstrapped_ccc = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        x_s = torch.tensor(x[idx], dtype=torch.float32)
        y_s = torch.tensor(y[idx], dtype=torch.float32)
        bootstrapped_ccc.append(ccc_metric(x_s, y_s).item())
    ci_lower = np.percentile(bootstrapped_ccc,  2.5)
    ci_upper = np.percentile(bootstrapped_ccc, 97.5)

    print("\n=== Within-subject reliability (longitudinal controls) ===")
    print(f"ICC(3,1)   = {icc_val:.2f} (95% CI {icc_ci})")
    print(f"SEM        = {sem:.2f} years")
    print(f"Spearman rho = {rho:.2f}")
    print(f"CCC = {ccc.item():.2f}, 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # 7) LMM pred_Edad_c ~ Edad + (1|ID) -----------------------------------
    df_wide = (wide.reset_index()
                   .melt(id_vars="ID", var_name="Time", value_name="BrainPAD"))
    df_wide["Age"]          = controls_long["Edad"].values
    df_wide["Age_centered"] = df_wide["Age"] - df_wide["Age"].mean()
    df_wide["pred_Edad_c"]  = controls_long.set_index(["ID", "Time"])["pred_Edad_c"].values

    model = smf.mixedlm("pred_Edad_c ~ Age", df_wide, groups="ID")
    result = model.fit()
    print(result.summary())

    # 8) Additional metrics -------------------------------------------------
    wide["delta_PAD"] = wide["BrainPAD_30y"] - wide["BrainPAD_20y"]
    wide["mean_PAD"] = wide[["BrainPAD_30y", "BrainPAD_20y"]].mean(axis=1)

    age_20 = (controls_long[controls_long["Time"] == 0][["ID", "Edad"]]
              .rename(columns={"Edad": "Edad_20"}))
    age_30 = (controls_long[controls_long["Time"] == 1][["ID", "Edad"]]
              .rename(columns={"Edad": "Edad_30"}))
    age_merged = pd.merge(age_20, age_30, on="ID")
    wide["delta_age"] = age_merged["Edad_30"].values - age_merged["Edad_20"].values

    delta_corr = wide["delta_PAD"].corr(wide["delta_age"])
    srm = wide["delta_PAD"].mean() / wide["delta_PAD"].std()
    print(f"\nDelta-BrainPAD vs Delta-Age correlation: r = {delta_corr:.2f}")
    print(f"Standardised Response Mean (SRM): {srm:.2f}")

    # 9) Bland-Altman plot --------------------------------------------------
    mean_diff = wide["delta_PAD"].mean()
    std_diff  = wide["delta_PAD"].std()
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(9, 6))
    plt.scatter(wide["mean_PAD"], wide["delta_PAD"], alpha=0.6)
    plt.axhline(mean_diff, color="gray", linestyle="--",
                label=f"Mean Δ = {mean_diff:.2f}")
    plt.axhline(loa_upper, color="red",  linestyle="--",
                label=f"+1.96 SD = {loa_upper:.2f}")
    plt.axhline(loa_lower, color="blue", linestyle="--",
                label=f"-1.96 SD = {loa_lower:.2f}")
    plt.title("Bland-Altman plot\n"
              f"ICC = {icc_val:.2f}, SEM = {sem:.2f} y, "
              f"SRM = {srm:.2f}, rho = {rho:.2f}, Δr = {delta_corr:.2f}")
    plt.xlabel("Mean Brain Age Prediction")
    plt.ylabel("ΔBrain Age Prediction (30y - 20y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BLAND)
    plt.close()

    print(f"\nSaved: {OUT_BLAND}")


if __name__ == "__main__":
    main()
