"""
07_export_LMM_input.py
======================

Pure-Python preprocessor for the LMM analysis.

What it does
------------
1. Loads the prediction CSVs from the MLP for ALSPAC waves I and II
   (`predictions_ALSPAC_20_MRI_I.csv` and `predictions_ALSPAC_30_MRI_II.csv`),
   plus the depression and Euler-number QC tables (when available).
2. Cleans IDs, attaches Time labels (0 for wave I, 1 for wave II),
   merges depression and Euler covariates.
3. Computes the centred Euler number, the MAD-standardised Euler,
   the Yeo-Johnson-transformed BrainPAD column, and the longitudinal
   trajectory variable used for LPEs-1 / LPEs-2.
4. Writes the tidy long-form table that the R LMM script reads.

Run from the project's `src/` directory:

    cd src
    python 07_export_LMM_input.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Paths (all relative to src/)
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
MODEL_DIR = "../model"

PRED_I_CSV   = f"{DATA_DIR}/predictions_ALSPAC_20_MRI_I.csv"
PRED_II_CSV  = f"{DATA_DIR}/predictions_ALSPAC_30_MRI_II.csv"

DEPR_I_CSV   = f"{DATA_DIR}/ALSPAC_I_depression.csv"
DEPR_II_CSV  = f"{DATA_DIR}/ALSPAC_II_depression.csv"

QC_I_CSV     = f"{DATA_DIR}/QC_fsqc_I_adq_fsqc-results.csv"
QC_II_CSV    = f"{DATA_DIR}/QC_fsqc_II_adq_fsqc-results.csv"

SELECTED_CSV = f"{DATA_DIR}/Datos_ALSPAC_II_Seleccionados.csv"

OUT_LONG_CSV     = f"{DATA_DIR}/lmm_input_long.csv"     # one row per (ID, Time)
OUT_DELTAS_CSV   = f"{DATA_DIR}/lmm_input_deltas.csv"   # one row per ID with ΔBrainPAD
OUT_PRED_I_CSV   = f"{DATA_DIR}/preds_ALSPAC_I_with_covariates.csv"
OUT_PRED_II_CSV  = f"{DATA_DIR}/preds_ALSPAC_II_with_covariates.csv"

PAD_COL = "BrainPAD_c"


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def _clean_id(series: pd.Series) -> pd.Series:
    """Strip the '_brain' / 'sub-' tags that appear in some ID columns."""
    return (series.astype(str)
                  .str.replace("_brain", "", regex=False)
                  .str.replace("sub-",   "", regex=False))


def add_euler_columns(qc: pd.DataFrame) -> pd.DataFrame:
    """Compute euler_n (centred) and euler_standard (MAD-z)."""
    qc = qc.copy()
    qc["euler_n"] = 2 - 2 * qc["holes_lh"] + 2 - 2 * qc["holes_rh"]
    mad = median_abs_deviation(qc["euler_n"])
    qc["euler_standard"] = (qc["euler_n"] - qc["euler_n"].median()) / mad
    qc["euler_n"] = qc["euler_n"] - qc["euler_n"].mean()
    return qc


def _safe_read(path: str, label: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[warn] {label} not found at {path} — skipping merge.")
    return None


# ---------------------------------------------------------------------------
# 3. Load & clean
# ---------------------------------------------------------------------------
def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_I  = pd.read_csv(PRED_I_CSV)
    pred_II = pd.read_csv(PRED_II_CSV)
    pred_I["ID"]  = _clean_id(pred_I["ID"])
    pred_II["ID"] = _clean_id(pred_II["ID"])
    pred_I["Time"]  = 0
    pred_II["Time"] = 1
    return pred_I, pred_II


def load_depression() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    dep_I  = _safe_read(DEPR_I_CSV,  "ALSPAC-I depression CSV")
    dep_II = _safe_read(DEPR_II_CSV, "ALSPAC-II depression CSV")
    if dep_I is not None:
        dep_I.rename(columns={"id": "ID"}, inplace=True)
    if dep_II is not None and "ID" in dep_II.columns:
        dep_II["ID"] = _clean_id(dep_II["ID"])
    return dep_I, dep_II


def load_qc() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    qc_I  = _safe_read(QC_I_CSV,  "ALSPAC-I QC CSV")
    qc_II = _safe_read(QC_II_CSV, "ALSPAC-II QC CSV")
    if qc_I is not None:
        qc_I.rename(columns={"subject": "ID"}, inplace=True)
        qc_I["ID"] = _clean_id(qc_I["ID"])
        qc_I = add_euler_columns(qc_I)
    if qc_II is not None:
        qc_II.rename(columns={"subject": "ID"}, inplace=True)
        qc_II["ID"] = _clean_id(qc_II["ID"])
        qc_II = add_euler_columns(qc_II)
    return qc_I, qc_II


# ---------------------------------------------------------------------------
# 4. Build the LMM input table
# ---------------------------------------------------------------------------
def build_long_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_I, pred_II = load_predictions()
    dep_I, dep_II   = load_depression()
    qc_I, qc_II     = load_qc()

    if dep_I is not None:
        cols = [c for c in ["ID", "FJCI1001", "FJCI1002", "FJCI501", "FKSH1070", "FJCI350"]
                if c in dep_I.columns]
        pred_I = pred_I.merge(dep_I[cols], on="ID", how="left")
    if dep_II is not None:
        cols = [c for c in ["ID", "pliks30TH", "FJCI1001", "FJCI1002",
                            "FJCI501", "FKSH1070", "FJCI350"]
                if c in dep_II.columns]
        pred_II = pred_II.merge(dep_II[cols], on="ID", how="left")

    bad_ids = ["SUBJ_00432", "SUBJ_00557", "SUBJ_00579",
               "SUBJ_00351", "SUBJ_00501"]
    pred_I = pred_I[~pred_I["ID"].isin(bad_ids)]

    selected = _safe_read(SELECTED_CSV, "ALSPAC-II selected-IDs CSV")
    if selected is not None:
        selected["ID"] = _clean_id(selected["ID"])
        pred_II_sel = pred_II[pred_II["ID"].isin(selected["ID"])].copy()
    else:
        pred_II_sel = pred_II.copy()

    if qc_I is not None:
        pred_I = pred_I.merge(qc_I[["ID", "euler_n", "euler_standard"]], on="ID", how="left")
    if qc_II is not None:
        pred_II = pred_II.merge(qc_II[["ID", "euler_n", "euler_standard"]], on="ID", how="left")
        pred_II_sel = pred_II_sel.merge(qc_II[["ID", "euler_n", "euler_standard"]],
                                        on="ID", how="left")

    for df in (pred_I, pred_II, pred_II_sel):
        df["pliks_bin"] = (df["pliks18TH"] > 0.5).astype(int)

    # Per-wave Yeo-Johnson on the raw BrainPAD column. The source assigns
    # both BrainPAD_YJ and BrainPAD_YJ_c from the same fit-transform on
    # `BrainPAD` — they are mathematically identical (kept for parity).
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    for df in (pred_I, pred_II, pred_II_sel):
        df["BrainPAD_YJ"]   = pt.fit_transform(df[["BrainPAD"]])
        df["BrainPAD_YJ_c"] = pt.fit_transform(df[["BrainPAD"]])

    pred_I.to_csv(OUT_PRED_I_CSV,  index=False)
    pred_II.to_csv(OUT_PRED_II_CSV, index=False)

    # Long table = subjects with both visits.
    preds_joint = pd.concat([pred_I, pred_II], axis=0).reset_index(drop=True)
    preds_joint["ID"] = _clean_id(preds_joint["ID"])
    id_counts = preds_joint["ID"].value_counts()
    double_ids = id_counts[id_counts == 2].index
    df_long = preds_joint[preds_joint["ID"].isin(double_ids)].copy()
    df_long = df_long.sort_values(["ID", "Time"]).reset_index(drop=True)

    # ΔBrainPAD via robust ID×Time pivot (replaces the half-array slicing
    # at source line 826 / 873).
    keep = ["ID", "Time", PAD_COL, "Edad", "eTIV", "sexo",
            "pliks18TH", "pliks30TH", "euler_n"]
    keep = [c for c in keep if c in df_long.columns]
    pivot_cols = [c for c in keep if c not in ("ID", "Time")]
    wide = df_long[keep].pivot(index="ID", columns="Time", values=pivot_cols)

    df_deltas = pd.DataFrame({
        "ID":         wide.index.values,
        PAD_COL:      (wide[(PAD_COL, 1)] - wide[(PAD_COL, 0)]).values,
        "delta_Age":  (wide[("Edad",   1)] - wide[("Edad",   0)]).values,
        "eTIV":        wide[("eTIV", 0)].values if ("eTIV", 0) in wide.columns else np.nan,
        "sexo":        wide[("sexo", 0)].values,
        "pliks18TH":   wide[("pliks18TH", 0)].values,
        "pliks30TH":   wide[("pliks30TH", 1)].values if ("pliks30TH", 1) in wide.columns else np.nan,
    })
    df_deltas["pliks_bin"] = (df_deltas["pliks18TH"] > 0.5).astype(int)

    conditions = [
        (df_deltas["pliks18TH"] == 0)            & (df_deltas["pliks30TH"] == 0),
        (df_deltas["pliks18TH"].isin([1, 2, 3])) & (df_deltas["pliks30TH"] == 0),
        (df_deltas["pliks18TH"].isin([1, 2, 3])) & (df_deltas["pliks30TH"].isin([1, 2, 3])),
        (df_deltas["pliks18TH"] == 0)            & (df_deltas["pliks30TH"].isin([1, 2, 3])),
    ]
    choices = [0, 1, 2, 3]
    df_deltas["trajectory"] = np.select(conditions, choices, default=np.nan)

    df_long = df_long.merge(
        df_deltas[["ID", "trajectory"]],
        on="ID", how="left",
    )
    df_long["Edad_c"] = df_long["Edad"] - df_long["Edad"].mean()

    # Cross-sectional + longitudinal in a single frame.
    all_data = pd.concat([pred_I, pred_II], ignore_index=True)
    all_data["Edad_c"] = all_data["Edad"] - all_data["Edad"].mean()
    if "FJCI1001" in all_data.columns and "FJCI1002" in all_data.columns:
        all_data["depression_Comp"] = all_data["FJCI1001"] + all_data["FJCI1002"]
    else:
        all_data["depression_Comp"] = np.nan
    all_data["ID"] = _clean_id(all_data["ID"])

    # Source line 919 only overwrites BrainPAD_YJ globally, NOT BrainPAD_YJ_c.
    # Keep BrainPAD_YJ_c from the per-wave concat.
    all_data["BrainPAD_YJ"] = pt.fit_transform(all_data[["BrainPAD"]])

    # Bring depression_Comp into df_long.
    if "depression_Comp" in all_data.columns:
        df_long = df_long.merge(
            all_data[["ID", "Time", "depression_Comp"]],
            on=["ID", "Time"], how="left",
        )

    # Trajectory for the cross-sectional all_data frame: -1 = no follow-up.
    if "pliks30TH" in all_data.columns:
        all_data["pliks30TH"] = all_data["pliks30TH"].astype("Int64").fillna(-1)
        cond = [
            (all_data["pliks30TH"] == -1),
            (all_data["pliks18TH"] == 0)            & (all_data["pliks30TH"] == 0),
            (all_data["pliks18TH"].isin([1, 2, 3])) & (all_data["pliks30TH"] == 0),
            (all_data["pliks18TH"].isin([1, 2, 3])) & (all_data["pliks30TH"].isin([1, 2, 3])),
            (all_data["pliks18TH"] == 0)            & (all_data["pliks30TH"].isin([1, 2, 3])),
        ]
        ch = [-1, 0, 1, 2, 3]
        all_data["trajectory"] = np.select(cond, ch, default=np.nan)

    # Cast to strings to keep R-side factor coding deterministic.
    all_data["pliks18TH"]    = all_data["pliks18TH"].astype(str)
    all_data["Time"]         = all_data["Time"].astype(str)
    all_data["pliksNumeric"] = all_data["pliks18TH"].astype(int)

    long_ids = all_data["ID"].value_counts().loc[lambda s: s > 1].index
    all_data["ID_long"] = np.where(all_data["ID"].isin(long_ids),
                                   all_data["ID"].astype(str),
                                   "single")

    return all_data, df_long, df_deltas


# ---------------------------------------------------------------------------
# 5. Quick descriptive printout
# ---------------------------------------------------------------------------
def descriptive_stats(all_data: pd.DataFrame) -> None:
    from scipy import stats

    print("\nValue counts pliks18TH (all_data):")
    print(all_data["pliks18TH"].value_counts())

    print("\nDescriptive statistics for Edad by pliks18TH groups:")
    print(all_data.groupby("pliks18TH")["Edad"].describe())

    edad_groups = [g["Edad"].dropna() for _, g in all_data.groupby("pliks18TH")]
    if all(len(g) > 1 for g in edad_groups):
        f_val, p_val = stats.f_oneway(*edad_groups)
        print(f"\nANOVA Edad ~ pliks18TH: F={f_val:.4f}, p={p_val:.4g}")

    if "sexo" in all_data.columns:
        ct = pd.crosstab(all_data["pliks18TH"], all_data["sexo"])
        print("\nContingency table sexo x pliks18TH:")
        print(ct)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        print(f"Chi2={chi2:.4f}  p={p:.4g}  dof={dof}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    all_data, df_long, df_deltas = build_long_table()

    descriptive_stats(all_data)

    all_data.to_csv(OUT_LONG_CSV, index=False)
    df_long.to_csv(f"{DATA_DIR}/lmm_input_long_pairs.csv", index=False)
    df_deltas.to_csv(OUT_DELTAS_CSV, index=False)

    print(f"\nWrote: {OUT_LONG_CSV}  ({len(all_data)} rows)")
    print(f"Wrote: {DATA_DIR}/lmm_input_long_pairs.csv  ({len(df_long)} rows)")
    print(f"Wrote: {OUT_DELTAS_CSV}  ({len(df_deltas)} rows)")


if __name__ == "__main__":
    main()
